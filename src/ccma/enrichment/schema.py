"""JSON Schema handed to the model's structured-output mode.

Deliberately hand-written rather than derived from the Pydantic model. Two
reasons:

1. Pydantic emits `$defs`/`$ref`, and constrained-decoding implementations vary
   in how well they follow references. A flat, inlined schema is understood by
   every backend (Ollama, vLLM guided decoding, hosted structured outputs).
2. The LLM should only produce the *inferred* fields. Lineage fields
   (`llm_model`, `validation_status`, `latency_ms`, ...) are set by the pipeline.
   Exposing them to the model is an invitation for it to fill them in with
   plausible lies.

`additionalProperties: false` at every level is what turns a hallucinated key
into a hard validation failure instead of silent data loss.
"""

from __future__ import annotations

from typing import Any

from ..models.enums import (
    ComplianceOutcome,
    ComplianceRule,
    DissatisfactionSignal,
    EscalationIndicator,
    ResolutionStatus,
    Sentiment,
)
from .taxonomy import l1_values, l2_values

_INTENT_OBJ: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["l1", "l2", "l3", "confidence", "evidence_turn_indices"],
    "properties": {
        "l1": {"type": "string", "enum": l1_values()},
        "l2": {"type": "string", "enum": l2_values()},
        "l3": {"type": ["string", "null"]},
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "evidence_turn_indices": {"type": "array", "items": {"type": "integer", "minimum": 0}},
    },
}


def enrichment_json_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "primary_intent",
            "resolution_status",
            "resolution_confidence",
            "fcr_candidate",
            "sentiment_trajectory",
            "customer_sentiment_start",
            "customer_sentiment_end",
            "dissatisfaction_signals",
            "escalation",
            "compliance",
            "summary_en",
        ],
        "properties": {
            "primary_intent": _INTENT_OBJ,
            "secondary_intents": {"type": "array", "maxItems": 4, "items": _INTENT_OBJ},
            "resolution_status": {
                "type": "string",
                "enum": [r.value for r in ResolutionStatus],
            },
            "resolution_confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            "fcr_candidate": {"type": "boolean"},
            "sentiment_trajectory": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": [
                        "turn_index", "at_ms", "speaker_role", "sentiment", "score", "drivers"
                    ],
                    "properties": {
                        "turn_index": {"type": "integer", "minimum": 0},
                        "at_ms": {"type": "integer", "minimum": 0},
                        "speaker_role": {"type": "string"},
                        "sentiment": {"type": "string", "enum": [s.value for s in Sentiment]},
                        "score": {"type": "number", "minimum": -1.0, "maximum": 1.0},
                        "drivers": {
                            "type": "array",
                            "maxItems": 5,
                            "items": {"type": "string"},
                        },
                    },
                },
            },
            "customer_sentiment_start": {"type": "string", "enum": [s.value for s in Sentiment]},
            "customer_sentiment_end": {"type": "string", "enum": [s.value for s in Sentiment]},
            "dissatisfaction_signals": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["signal", "turn_index", "quote", "severity"],
                    "properties": {
                        "signal": {
                            "type": "string",
                            "enum": [d.value for d in DissatisfactionSignal],
                        },
                        "turn_index": {"type": "integer", "minimum": 0},
                        "quote": {"type": "string", "maxLength": 300},
                        "language": {"type": ["string", "null"]},
                        "severity": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                    },
                },
            },
            "escalation": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "escalated_in_call", "risk_score", "indicators",
                    "rationale", "evidence_turn_indices",
                ],
                "properties": {
                    "escalated_in_call": {"type": "boolean"},
                    "risk_score": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                    "indicators": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "enum": [e.value for e in EscalationIndicator],
                        },
                    },
                    "rationale": {"type": "string", "maxLength": 600},
                    "evidence_turn_indices": {
                        "type": "array",
                        "items": {"type": "integer", "minimum": 0},
                    },
                },
            },
            "compliance": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["rule", "outcome", "confidence"],
                    "properties": {
                        "rule": {"type": "string", "enum": [c.value for c in ComplianceRule]},
                        "outcome": {
                            "type": "string",
                            "enum": [o.value for o in ComplianceOutcome],
                        },
                        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                        "matched_text": {"type": ["string", "null"], "maxLength": 400},
                        "matched_language": {"type": ["string", "null"]},
                        "evidence_turn_indices": {
                            "type": "array",
                            "items": {"type": "integer", "minimum": 0},
                        },
                        "detector_disagreement": {"type": "boolean"},
                    },
                },
            },
            "summary_en": {"type": "string", "maxLength": 800},
            "customer_requested_language": {"type": ["string", "null"]},
            "agent_language_accommodation": {"type": ["boolean", "null"]},
            "action_items": {"type": "array", "maxItems": 6, "items": {"type": "string"}},
        },
    }
