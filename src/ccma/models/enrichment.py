"""The LLM output contract.

This is the strictest schema in the codebase because it is the only place where
free-form model output becomes structured data. Design rules:

1. `extra="forbid"` everywhere — a hallucinated key fails loudly instead of
   being silently dropped.
2. Every categorical field is an Enum — the model cannot invent an intent.
3. Every subjective field carries a sibling `confidence` — downstream code can
   threshold rather than trusting blindly.
4. Every extracted claim carries `evidence_turn_indices` — an assertion with no
   pointer into the transcript is unverifiable and is rejected in validation.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, computed_field, field_validator

from .base import RoundTripModel
from .enums import (
    ComplianceOutcome,
    ComplianceRule,
    DissatisfactionSignal,
    EscalationIndicator,
    ResolutionStatus,
    Sentiment,
    ValidationStatus,
)

SCHEMA_VERSION = "silver.call_enrichment/1.2.0"
PROMPT_VERSION = "enrich-v3"


class IntentAssignment(RoundTripModel):
    model_config = ConfigDict(extra="forbid")

    l1: str = Field(..., description="Top-level intent, e.g. BILLING")
    l2: str = Field(..., description="Sub-intent, e.g. OVERCHARGE_DISPUTE")
    l3: str | None = Field(None, description="Leaf intent; null when not determinable")
    confidence: float = Field(..., ge=0.0, le=1.0)
    evidence_turn_indices: list[int] = Field(default_factory=list)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def path(self) -> str:
        return " / ".join(p for p in (self.l1, self.l2, self.l3) if p)


class SentimentPoint(BaseModel):
    """One reading on the sentiment trajectory. We sample per-turn rather than
    per-call so the dashboard can show *where* a call went wrong, which is the
    only version of sentiment a floor supervisor can act on."""

    model_config = ConfigDict(extra="forbid")

    turn_index: int = Field(..., ge=0)
    at_ms: int = Field(..., ge=0)
    speaker_role: str
    sentiment: Sentiment
    score: float = Field(..., ge=-1.0, le=1.0)
    drivers: list[str] = Field(default_factory=list, max_length=5)


class EscalationAssessment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    escalated_in_call: bool = Field(..., description="Escalation actually happened on this call")
    risk_score: float = Field(
        ..., ge=0.0, le=1.0, description="LLM's prior on post-call escalation/repeat contact"
    )
    indicators: list[EscalationIndicator] = Field(default_factory=list)
    rationale: str = Field("", max_length=600)
    evidence_turn_indices: list[int] = Field(default_factory=list)


class ComplianceCheck(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rule: ComplianceRule
    outcome: ComplianceOutcome
    confidence: float = Field(..., ge=0.0, le=1.0)
    matched_text: str | None = Field(None, max_length=400)
    matched_language: str | None = None
    evidence_turn_indices: list[int] = Field(default_factory=list)
    # Set when a deterministic phrase match and the LLM disagree — these rows are
    # the human-review queue, not an automatic fail.
    detector_disagreement: bool = False


class DissatisfactionEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    signal: DissatisfactionSignal
    turn_index: int = Field(..., ge=0)
    quote: str = Field(..., max_length=300)
    language: str | None = None
    severity: float = Field(..., ge=0.0, le=1.0)


class CallEnrichment(RoundTripModel):
    """One row per call. Written by ccma.enrichment.runner after validation."""

    model_config = ConfigDict(extra="forbid")

    call_id: str

    primary_intent: IntentAssignment
    secondary_intents: list[IntentAssignment] = Field(default_factory=list, max_length=4)

    resolution_status: ResolutionStatus
    resolution_confidence: float = Field(..., ge=0.0, le=1.0)
    fcr_candidate: bool = Field(
        ..., description="Call *looks* first-contact-resolved from content alone"
    )

    sentiment_trajectory: list[SentimentPoint] = Field(default_factory=list)
    customer_sentiment_start: Sentiment
    customer_sentiment_end: Sentiment

    dissatisfaction_signals: list[DissatisfactionEvidence] = Field(default_factory=list)
    escalation: EscalationAssessment
    compliance: list[ComplianceCheck] = Field(default_factory=list)

    summary_en: str = Field(..., max_length=800)
    customer_requested_language: str | None = None
    agent_language_accommodation: bool | None = Field(
        None, description="Did the agent switch to the customer's preferred language?"
    )

    action_items: list[str] = Field(default_factory=list, max_length=6)

    # --- lineage / observability -------------------------------------------
    llm_backend: str = "mock"
    llm_model: str = "mock"
    prompt_version: str = PROMPT_VERSION
    schema_version: str = SCHEMA_VERSION
    validation_status: ValidationStatus = ValidationStatus.VALID
    validation_errors: list[str] = Field(default_factory=list)
    repair_attempts: int = 0
    input_token_estimate: int = 0
    latency_ms: int = 0

    @field_validator("sentiment_trajectory")
    @classmethod
    def _trajectory_must_be_ordered(cls, v: list[SentimentPoint]) -> list[SentimentPoint]:
        if any(b.turn_index < a.turn_index for a, b in zip(v, v[1:])):
            raise ValueError("sentiment_trajectory must be ordered by turn_index")
        return v

    @computed_field  # type: ignore[prop-decorator]
    @property
    def sentiment_delta(self) -> float:
        """End minus start. Negative = the call made things worse, which is the
        single strongest leading indicator of a repeat contact."""
        from .enums import SENTIMENT_SCORE

        return round(
            SENTIMENT_SCORE[self.customer_sentiment_end]
            - SENTIMENT_SCORE[self.customer_sentiment_start],
            3,
        )

    @computed_field  # type: ignore[prop-decorator]
    @property
    def sentiment_min(self) -> float:
        return min((p.score for p in self.sentiment_trajectory), default=0.0)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def compliance_score(self) -> float:
        """Share of applicable rules that were satisfied. NOT_APPLICABLE rules are
        excluded from the denominator so a call that never touched payments is not
        penalised for skipping the card-readback rule."""
        applicable = [c for c in self.compliance if c.outcome != ComplianceOutcome.NOT_APPLICABLE]
        if not applicable:
            return 1.0
        earned = sum(
            1.0
            if c.outcome == ComplianceOutcome.PRESENT
            else 0.5
            if c.outcome == ComplianceOutcome.PARTIAL
            else 0.0
            for c in applicable
        )
        return round(earned / len(applicable), 4)
