"""Enrichment orchestration: prompt -> generate -> validate -> repair -> persist.

The control flow encodes the cost/quality policy:

    bulk model (7B)
        -> valid?  no  -> repair loop (max N, shows the model its own errors)
        -> valid?  yes -> confidence below threshold?
                             yes -> re-run on the heavy model (14B)
                             no  -> accept
        -> still invalid after repairs -> QUARANTINE (never silently dropped)

Compliance is merged in *after* the LLM from the deterministic phrase-bank
detector, with disagreements flagged for human review rather than resolved.

Everything is single-writer and idempotent per call_id: re-running enrichment for
a call overwrites exactly that call's row, which is what makes selective
reprocessing after a prompt change practical.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from ..config import Settings
from ..logging_conf import get_logger
from ..models.bronze import CallMetadata
from ..models.enrichment import PROMPT_VERSION, CallEnrichment
from ..models.enums import ValidationStatus
from ..models.silver import ConversationLanguageProfile, Utterance
from . import compliance as compliance_mod
from . import prompts
from .base import LlmBackend
from .schema import enrichment_json_schema
from .validation import ValidationReport, extract_json, validate

log = get_logger(__name__)


@dataclass
class EnrichmentOutcome:
    enrichment: CallEnrichment
    attempts: int
    used_heavy_model: bool
    warnings: list[str]


def _quarantine_record(call_id: str, errors: list[str], model: str, attempts: int) -> CallEnrichment:
    """A minimal but schema-valid row so the call still appears in Gold with an
    explicit failure status. Dashboards filter on validation_status; the DQ layer
    alerts on the quarantine rate."""
    from ..models.enrichment import EscalationAssessment, IntentAssignment
    from ..models.enums import ResolutionStatus, Sentiment

    return CallEnrichment(
        call_id=call_id,
        primary_intent=IntentAssignment(
            l1="OTHER", l2="UNCLASSIFIED", l3=None, confidence=0.0, evidence_turn_indices=[]
        ),
        resolution_status=ResolutionStatus.UNRESOLVED,
        resolution_confidence=0.0,
        fcr_candidate=False,
        sentiment_trajectory=[],
        customer_sentiment_start=Sentiment.NEUTRAL,
        customer_sentiment_end=Sentiment.NEUTRAL,
        dissatisfaction_signals=[],
        escalation=EscalationAssessment(
            escalated_in_call=False, risk_score=0.0, indicators=[], rationale="", evidence_turn_indices=[]
        ),
        compliance=[],
        summary_en="ENRICHMENT FAILED - quarantined for review",
        llm_model=model,
        validation_status=ValidationStatus.QUARANTINED,
        validation_errors=errors[:20],
        repair_attempts=attempts,
    )


def enrich_call(
    call: CallMetadata,
    utterances: list[Utterance],
    profile: ConversationLanguageProfile,
    backend: LlmBackend,
    settings: Settings,
) -> EnrichmentOutcome:
    if not utterances:
        return EnrichmentOutcome(
            _quarantine_record(call.call_id, ["no utterances"], backend.name, 0), 0, False, []
        )

    mean_conf = sum(u.asr_confidence for u in utterances) / len(utterances)
    messages = prompts.build_messages(
        call_id=call.call_id,
        queue=call.queue,
        direction=call.direction.value,
        duration_ms=call.duration_ms,
        hold_count=call.hold_count,
        transfer_count=call.transfer_count,
        dominant_language=profile.dominant_language.value,
        language_mix=profile.language_share,
        mean_asr_confidence=mean_conf,
        utterances=utterances,
    )
    schema = enrichment_json_schema()

    report, attempts, model_used, total_latency = _generate_with_repair(
        messages, schema, utterances, backend, settings, settings.ollama_model
    )

    used_heavy = False
    if report.ok and report.payload is not None:
        if _needs_heavy_model(report, settings):
            log.info(
                "low confidence; retrying on heavy model",
                extra={"call_id": call.call_id, "model": settings.ollama_model_heavy},
            )
            heavy_report, heavy_attempts, heavy_model, heavy_latency = _generate_with_repair(
                messages, schema, utterances, backend, settings, settings.ollama_model_heavy
            )
            attempts += heavy_attempts
            total_latency += heavy_latency
            if heavy_report.ok and heavy_report.payload is not None:
                report, model_used, used_heavy = heavy_report, heavy_model, True

    if not report.ok or report.payload is None:
        log.error(
            "enrichment quarantined",
            extra={"call_id": call.call_id, "errors": report.errors[:3], "attempts": attempts},
        )
        return EnrichmentOutcome(
            _quarantine_record(call.call_id, report.errors, model_used, attempts),
            attempts,
            used_heavy,
            report.warnings,
        )

    payload = report.payload
    payload["compliance"] = _merge_compliance(payload.get("compliance", []), utterances)

    enrichment = CallEnrichment.model_validate(
        {
            **payload,
            "call_id": call.call_id,
            "llm_backend": backend.name,
            "llm_model": model_used,
            "prompt_version": PROMPT_VERSION,
            "validation_status": (
                ValidationStatus.REPAIRED
                if (attempts > 1 or report.coerced)
                else ValidationStatus.VALID
            ),
            "validation_errors": report.warnings[:20],
            "repair_attempts": max(0, attempts - 1),
            "input_token_estimate": sum(len(m["content"]) for m in messages) // 4,
            "latency_ms": total_latency,
        }
    )
    return EnrichmentOutcome(enrichment, attempts, used_heavy, report.warnings)


def _generate_with_repair(
    messages: list[dict[str, str]],
    schema: dict,
    utterances: list[Utterance],
    backend: LlmBackend,
    settings: Settings,
    model: str,
) -> tuple[ValidationReport, int, str, int]:
    conversation = list(messages)
    last_report = ValidationReport(False, ["no attempt made"])
    total_latency = 0

    for attempt in range(1, settings.llm_max_retries + 1):
        started = time.perf_counter()
        # Nudge temperature up on later attempts: a deterministic model that
        # produced invalid output once will reproduce it exactly at temp 0.
        response = backend.chat(
            conversation, json_schema=schema, model=model, temperature=0.0 if attempt == 1 else 0.3
        )
        total_latency += response.latency_ms or int((time.perf_counter() - started) * 1000)

        payload, parse_error = extract_json(response.text)
        if payload is None:
            last_report = ValidationReport(False, [parse_error or "unparseable response"])
        else:
            last_report = validate(payload, utterances)
            if last_report.ok:
                return last_report, attempt, response.model, total_latency

        if attempt < settings.llm_max_retries:
            conversation = [
                *messages,
                {"role": "assistant", "content": response.text[:4000]},
                prompts.repair_message(last_report.errors, response.text),
            ]

    return last_report, settings.llm_max_retries, model, total_latency


def _needs_heavy_model(report: ValidationReport, settings: Settings) -> bool:
    payload = report.payload or {}
    intent_conf = float(payload.get("primary_intent", {}).get("confidence", 1.0))
    resolution_conf = float(payload.get("resolution_confidence", 1.0))
    return (
        min(intent_conf, resolution_conf) < settings.escalate_to_heavy_below
        or payload.get("primary_intent", {}).get("l1") == "OTHER"
    )


def _merge_compliance(llm_checks: list[dict], utterances: list[Utterance]) -> list[dict]:
    turns = [(u.turn_index, u.speaker_role.value, u.text_raw) for u in utterances]
    deterministic = compliance_mod.detect_all(turns)
    return compliance_mod.reconcile(deterministic, llm_checks)


def enrich_batch(
    items: list[tuple[CallMetadata, list[Utterance], ConversationLanguageProfile]],
    backend: LlmBackend,
    settings: Settings,
) -> list[EnrichmentOutcome]:
    """Thread-pool fan-out.

    Threads (not processes) because the work is IO-bound on the model server.
    Concurrency is capped by `CCMA_LLM_CONCURRENCY` — oversubscribing a single
    Ollama instance makes throughput *worse*, since requests queue behind one
    another on the GPU while each holds a KV cache slot."""
    if not items:
        return []

    workers = max(1, min(settings.llm_concurrency, len(items)))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [
            pool.submit(enrich_call, call, utts, profile, backend, settings)
            for call, utts, profile in items
        ]
        return [f.result() for f in futures]
