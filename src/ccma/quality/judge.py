"""LLM-as-judge evaluation, and its guardrails.

Used because human annotation of Hinglish support calls is slow and expensive,
and we need a quality signal on every deploy, not every quarter.

Used *carefully*, because LLM judges have well-documented pathologies:

* **Self-preference** — a model rates its own output higher. We therefore judge
  with a *different model family* than the generator (`gemma2` judging `qwen2.5`
  by default), and never with the same weights.
* **Position/verbosity bias** — longer, more confident answers score better
  regardless of correctness. We score against a rubric with per-dimension
  anchors, not a bare 1-10, and we require evidence citations for every score.
* **Miscalibration** — judges cluster on 4/5. We track the judge's own score
  distribution as a monitored metric; a collapsed distribution invalidates the
  eval, and the DQ layer flags it.
* **Correlation, not truth** — the judge is only trustworthy to the extent it
  agrees with humans. We measure that: a stratified sample is dual-labelled by
  judge and human, and we report Cohen's kappa. Below ~0.6 the judge is not
  usable as a release gate and we say so rather than shipping on it.

So the judge is a *screening* instrument: it ranks releases, sizes the human
review queue, and catches regressions. It is not the ground truth, and nothing
here treats it as such.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from ..logging_conf import get_logger
from ..models.enrichment import CallEnrichment
from ..models.silver import Utterance
from .base_prompts import JUDGE_SYSTEM, JUDGE_USER

log = get_logger(__name__)

JUDGE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "intent_correctness", "sentiment_faithfulness", "resolution_correctness",
        "evidence_grounding", "hallucination_detected", "overall", "reasoning",
    ],
    "properties": {
        "intent_correctness": {"type": "integer", "minimum": 1, "maximum": 5},
        "sentiment_faithfulness": {"type": "integer", "minimum": 1, "maximum": 5},
        "resolution_correctness": {"type": "integer", "minimum": 1, "maximum": 5},
        "evidence_grounding": {"type": "integer", "minimum": 1, "maximum": 5},
        "language_handling": {"type": "integer", "minimum": 1, "maximum": 5},
        "hallucination_detected": {"type": "boolean"},
        "hallucinated_claims": {"type": "array", "items": {"type": "string"}},
        "overall": {"type": "integer", "minimum": 1, "maximum": 5},
        "reasoning": {"type": "string", "maxLength": 800},
    },
}

DIMENSIONS = [
    "intent_correctness",
    "sentiment_faithfulness",
    "resolution_correctness",
    "evidence_grounding",
    "language_handling",
]


@dataclass
class JudgeVerdict:
    call_id: str
    scores: dict[str, int] = field(default_factory=dict)
    hallucination_detected: bool = False
    hallucinated_claims: list[str] = field(default_factory=list)
    overall: int = 0
    reasoning: str = ""
    judge_model: str = ""
    parse_failed: bool = False


def judge_enrichment(
    enrichment: CallEnrichment,
    utterances: list[Utterance],
    backend,
    judge_model: str,
) -> JudgeVerdict:
    from ..enrichment.prompts import render_transcript

    payload = enrichment.model_dump(mode="json")
    for noise_field in (
        "llm_backend", "llm_model", "prompt_version", "schema_version",
        "validation_status", "validation_errors", "repair_attempts",
        "input_token_estimate", "latency_ms",
    ):
        payload.pop(noise_field, None)

    messages = [
        {"role": "system", "content": JUDGE_SYSTEM},
        {
            "role": "user",
            "content": JUDGE_USER.format(
                transcript=render_transcript(utterances, max_turns=80),
                extraction=json.dumps(payload, ensure_ascii=False, indent=2)[:6000],
            ),
        },
    ]

    response = backend.chat(messages, json_schema=JUDGE_SCHEMA, model=judge_model, temperature=0.0)

    try:
        data = json.loads(response.text)
    except json.JSONDecodeError:
        from ..enrichment.validation import extract_json

        data, _ = extract_json(response.text)
        if data is None:
            log.warning("judge output unparseable", extra={"call_id": enrichment.call_id})
            return JudgeVerdict(enrichment.call_id, judge_model=judge_model, parse_failed=True)

    return JudgeVerdict(
        call_id=enrichment.call_id,
        scores={d: int(data.get(d, 0)) for d in DIMENSIONS if d in data},
        hallucination_detected=bool(data.get("hallucination_detected", False)),
        hallucinated_claims=list(data.get("hallucinated_claims", []))[:10],
        overall=int(data.get("overall", 0)),
        reasoning=str(data.get("reasoning", ""))[:800],
        judge_model=judge_model,
    )


def aggregate_verdicts(verdicts: list[JudgeVerdict]) -> dict[str, Any]:
    usable = [v for v in verdicts if not v.parse_failed]
    if not usable:
        return {"n": 0, "usable": 0}

    per_dimension: dict[str, float] = {}
    for dim in DIMENSIONS:
        scores = [v.scores[dim] for v in usable if dim in v.scores]
        if scores:
            per_dimension[dim] = round(sum(scores) / len(scores), 3)

    overalls = [v.overall for v in usable if v.overall]
    distribution = {str(s): overalls.count(s) for s in range(1, 6)}
    # A judge that has stopped discriminating is a broken instrument.
    spread = (max(overalls) - min(overalls)) if overalls else 0

    return {
        "n": len(verdicts),
        "usable": len(usable),
        "parse_failure_rate": round((len(verdicts) - len(usable)) / len(verdicts), 4),
        "per_dimension_mean": per_dimension,
        "overall_mean": round(sum(overalls) / len(overalls), 3) if overalls else 0.0,
        "overall_distribution": distribution,
        "score_spread": spread,
        "judge_degenerate": spread <= 1,
        "hallucination_rate": round(
            sum(1 for v in usable if v.hallucination_detected) / len(usable), 4
        ),
    }


def cohens_kappa(labels_a: list[str], labels_b: list[str]) -> float:
    """Chance-corrected agreement between judge and human on the dual-labelled
    sample. This is the number that decides whether the judge may gate a release."""
    if len(labels_a) != len(labels_b) or not labels_a:
        return 0.0

    categories = sorted(set(labels_a) | set(labels_b))
    n = len(labels_a)
    observed = sum(1 for a, b in zip(labels_a, labels_b) if a == b) / n

    expected = sum(
        (labels_a.count(c) / n) * (labels_b.count(c) / n) for c in categories
    )
    if expected >= 1.0:
        return 1.0
    return round((observed - expected) / (1 - expected), 4)
