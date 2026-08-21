"""Validation, coercion and grounding checks for LLM output.

Three layers, cheapest first:

1. **Structural** — is it JSON at all, and does it satisfy the JSON Schema?
2. **Semantic** — are the categorical values in our controlled vocabularies, and
   is the intent path real? Near-misses are coerced (and recorded as coerced)
   rather than discarded; an LLM that puts a valid L2 under the wrong L1 is
   giving us usable information.
3. **Grounding** — does every cited `turn_index` exist, and do quotes actually
   appear in the transcript? This is the anti-hallucination layer, and it is the
   one that catches the failure mode nobody else catches: fluent, schema-valid,
   entirely invented evidence.

A record that fails structurally after all repair attempts is **quarantined**,
not dropped. Silent drops make a pipeline look healthy while it loses data.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Any

from jsonschema import Draft202012Validator

from ..models.silver import Utterance
from .schema import enrichment_json_schema
from .taxonomy import coerce_path

_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.MULTILINE)
_VALIDATOR = Draft202012Validator(enrichment_json_schema())


@dataclass
class ValidationReport:
    ok: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    coerced: bool = False
    payload: dict[str, Any] | None = None


def extract_json(text: str) -> tuple[dict[str, Any] | None, str | None]:
    """Recover a JSON object from model output.

    Constrained decoding usually makes this trivial, but small models still
    occasionally wrap output in fences or prepend a sentence. Handling it here
    costs nothing and avoids burning a repair round-trip on formatting."""
    if not text or not text.strip():
        return None, "empty response"

    cleaned = _FENCE_RE.sub("", text).strip()
    try:
        parsed = json.loads(cleaned)
        return (parsed, None) if isinstance(parsed, dict) else (None, "top level is not an object")
    except json.JSONDecodeError:
        pass

    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start == -1 or end <= start:
        return None, "no JSON object found in response"
    try:
        parsed = json.loads(cleaned[start : end + 1])
        return (parsed, None) if isinstance(parsed, dict) else (None, "top level is not an object")
    except json.JSONDecodeError as exc:
        return None, f"invalid JSON: {exc.msg} at position {exc.pos}"


def validate_structure(payload: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    for err in sorted(_VALIDATOR.iter_errors(payload), key=lambda e: list(e.absolute_path)):
        location = "/".join(str(p) for p in err.absolute_path) or "<root>"
        errors.append(f"{location}: {err.message}")
    return errors[:25]


def coerce_semantics(payload: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Repair recoverable semantic problems in place. Returns (payload, notes)."""
    notes: list[str] = []

    for key in ("primary_intent", *(["secondary_intents"] if "secondary_intents" in payload else [])):
        value = payload.get(key)
        intents = value if isinstance(value, list) else [value] if isinstance(value, dict) else []
        for intent in intents:
            l1, l2, l3, was_coerced = coerce_path(
                intent.get("l1", ""), intent.get("l2", ""), intent.get("l3")
            )
            if was_coerced:
                notes.append(
                    f"{key}: coerced {intent.get('l1')}/{intent.get('l2')}/{intent.get('l3')} "
                    f"-> {l1}/{l2}/{l3}"
                )
                # A coerced path is a weaker claim; cap the confidence to match.
                intent["confidence"] = min(float(intent.get("confidence", 0.5)), 0.5)
            intent["l1"], intent["l2"], intent["l3"] = l1, l2, l3

    trajectory = payload.get("sentiment_trajectory")
    if isinstance(trajectory, list) and trajectory:
        ordered = sorted(trajectory, key=lambda p: p.get("turn_index", 0))
        if ordered != trajectory:
            notes.append("sentiment_trajectory: reordered by turn_index")
            payload["sentiment_trajectory"] = ordered

    # De-duplicate compliance rules, keeping the highest-confidence assessment.
    compliance = payload.get("compliance")
    if isinstance(compliance, list):
        best: dict[str, dict[str, Any]] = {}
        for check in compliance:
            rule = check.get("rule")
            if rule is None:
                continue
            if rule not in best or check.get("confidence", 0) > best[rule].get("confidence", 0):
                best[rule] = check
        if len(best) != len(compliance):
            notes.append("compliance: removed duplicate rule assessments")
        payload["compliance"] = list(best.values())

    return payload, notes


def check_grounding(
    payload: dict[str, Any], utterances: list[Utterance], quote_threshold: float = 0.55
) -> tuple[list[str], list[str]]:
    """Verify citations point at real turns and quotes reflect real text.

    Returns (errors, warnings). A citation to a non-existent turn is an error —
    it is unambiguous evidence of fabrication. A quote that only loosely matches
    is a warning, because ASR text and the model's rendering of it legitimately
    differ (punctuation, casing, spacing)."""
    errors: list[str] = []
    warnings: list[str] = []

    valid_indices = {u.turn_index for u in utterances}
    text_by_index = {u.turn_index: u.text_raw.lower() for u in utterances}
    customer_indices = {u.turn_index for u in utterances if u.speaker_role.value == "CUSTOMER"}

    def _check_indices(indices: Any, where: str) -> None:
        if not isinstance(indices, list):
            return
        for idx in indices:
            if idx not in valid_indices:
                errors.append(f"{where}: cites turn_index {idx}, which does not exist")

    _check_indices(payload.get("primary_intent", {}).get("evidence_turn_indices"), "primary_intent")
    _check_indices(payload.get("escalation", {}).get("evidence_turn_indices"), "escalation")

    for i, signal in enumerate(payload.get("dissatisfaction_signals") or []):
        idx = signal.get("turn_index")
        where = f"dissatisfaction_signals[{i}]"
        if idx not in valid_indices:
            errors.append(f"{where}: turn_index {idx} does not exist")
            continue
        if idx not in customer_indices:
            warnings.append(f"{where}: turn {idx} is not a customer turn")

        quote = (signal.get("quote") or "").strip().lower()
        if quote:
            ratio = SequenceMatcher(None, quote[:200], text_by_index[idx][:200]).ratio()
            if quote not in text_by_index[idx] and ratio < quote_threshold:
                warnings.append(
                    f"{where}: quote does not match turn {idx} text (similarity {ratio:.2f})"
                )

    for i, point in enumerate(payload.get("sentiment_trajectory") or []):
        idx = point.get("turn_index")
        if idx not in valid_indices:
            errors.append(f"sentiment_trajectory[{i}]: turn_index {idx} does not exist")

    for i, check in enumerate(payload.get("compliance") or []):
        _check_indices(check.get("evidence_turn_indices"), f"compliance[{i}]")

    return errors, warnings


def validate(payload: dict[str, Any], utterances: list[Utterance]) -> ValidationReport:
    payload, coercion_notes = coerce_semantics(payload)

    structural = validate_structure(payload)
    if structural:
        return ValidationReport(False, structural, coercion_notes, bool(coercion_notes), payload)

    grounding_errors, grounding_warnings = check_grounding(payload, utterances)
    if grounding_errors:
        return ValidationReport(
            False,
            grounding_errors,
            coercion_notes + grounding_warnings,
            bool(coercion_notes),
            payload,
        )

    return ValidationReport(
        True, [], coercion_notes + grounding_warnings, bool(coercion_notes), payload
    )
