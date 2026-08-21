"""Silver = conformed conversation. Turns are speaker-attributed, text is
normalised, language is resolved per-utterance *and* per-token, and every row
carries the quality metadata a downstream consumer needs to decide whether to
trust it."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, computed_field

from .base import RoundTripModel
from .enums import LanguageCode, SpeakerRole


class TokenLanguage(BaseModel):
    """Per-token language tag. This is what makes code-switch metrics possible;
    an utterance-level label alone cannot distinguish 'mixed' from 'ambiguous'."""

    model_config = ConfigDict(extra="forbid")

    token: str
    lang: LanguageCode
    confidence: float = Field(..., ge=0.0, le=1.0)
    # True when the token is Hindi written in Latin script ("nahi", "paisa").
    is_transliterated: bool = False


class Utterance(RoundTripModel):
    model_config = ConfigDict(extra="forbid")

    call_id: str
    turn_index: int = Field(..., ge=0)
    segment_ids: list[int] = Field(default_factory=list)

    speaker_role: SpeakerRole
    speaker_id: str
    start_ms: int
    end_ms: int

    text_raw: str
    text_normalized: str = Field(
        ..., description="Case/punct normalised, disfluencies tagged, numerals unified"
    )
    text_devanagari: str | None = Field(
        None, description="Romanised Hindi back-transliterated to Devanagari; None for pure English"
    )

    lang_primary: LanguageCode
    lang_confidence: float = Field(..., ge=0.0, le=1.0)
    lang_distribution: dict[str, float] = Field(default_factory=dict)
    token_langs: list[TokenLanguage] = Field(default_factory=list)

    is_code_switched: bool = False
    cmi: float = Field(0.0, ge=0.0, le=1.0, description="Code-Mixing Index for this utterance")
    switch_points: int = 0

    asr_confidence: float = Field(..., ge=0.0, le=1.0)
    quality_flags: list[str] = Field(default_factory=list)

    diarization_confidence: float = Field(1.0, ge=0.0, le=1.0)
    overlapped_with_next: bool = False

    schema_version: str = "silver.utterance/1.0.0"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def duration_ms(self) -> int:
        return max(0, self.end_ms - self.start_ms)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def word_count(self) -> int:
        return len(self.text_normalized.split())


class CallQuality(BaseModel):
    """Ground-truth-free ASR/pipeline quality estimate. One row per call.
    See ccma.quality.asr_quality for how each field is derived."""

    model_config = ConfigDict(extra="forbid")

    call_id: str

    mean_word_prob: float = Field(..., ge=0.0, le=1.0)
    mean_avg_logprob: float
    max_no_speech_prob: float
    mean_compression_ratio: float

    dual_decode_disagreement: float | None = Field(
        None, description="Normalised edit distance between primary and shadow ASR hypotheses"
    )
    lm_perplexity_z: float | None = Field(
        None, description="Z-scored LM perplexity of the hypothesis vs. in-domain corpus"
    )

    estimated_wer: float = Field(..., ge=0.0, description="Calibrated WER point estimate")
    estimated_wer_lo: float = Field(..., ge=0.0)
    estimated_wer_hi: float = Field(..., ge=0.0)
    quality_band: str = Field(..., description="GOOD | FAIR | POOR | UNUSABLE")

    per_language_wer: dict[str, float] = Field(default_factory=dict)
    low_confidence_utterance_ratio: float = Field(0.0, ge=0.0, le=1.0)
    unintelligible_ratio: float = Field(0.0, ge=0.0, le=1.0)

    schema_version: str = "silver.call_quality/1.0.0"


class ConversationLanguageProfile(BaseModel):
    """Call-level language summary — drives the language-switch visualisation and
    the per-language quality dashboards."""

    model_config = ConfigDict(extra="forbid")

    call_id: str
    dominant_language: LanguageCode
    language_share: dict[str, float] = Field(default_factory=dict)
    call_cmi: float = Field(0.0, ge=0.0, le=1.0)
    m_index: float = Field(0.0, ge=0.0, le=1.0)
    switch_point_fraction: float = Field(0.0, ge=0.0, le=1.0)
    agent_customer_language_match: float = Field(
        1.0, ge=0.0, le=1.0, description="Agreement between agent and customer language choice"
    )
    transliterated_token_ratio: float = Field(0.0, ge=0.0, le=1.0)
    schema_version: str = "silver.language_profile/1.0.0"
