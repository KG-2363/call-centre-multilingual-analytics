"""Bronze = immutable landing zone. Nothing here is interpreted; we only record
what the telephony platform gave us plus what the decoder emitted verbatim.

Rule: Bronze is append-only and replayable. If the ASR model or the enrichment
prompt changes, we re-derive Silver from Bronze rather than re-ingesting audio."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from .enums import CallDirection


class CallMetadata(BaseModel):
    """One row per call as landed from the CTI/recorder."""

    model_config = ConfigDict(extra="forbid")

    call_id: str
    tenant_id: str = "demo-tenant"
    audio_uri: str | None = None
    audio_sha256: str | None = None
    audio_format: str = "wav"
    sample_rate_hz: int = 16000
    channels: int = Field(2, description="2 = stereo, agent on L / customer on R")
    duration_ms: int = Field(..., ge=0)

    recorded_at: datetime
    ingested_at: datetime
    ingest_batch_id: str

    agent_id: str
    agent_name: str | None = None
    queue: str
    campaign: str | None = None
    direction: CallDirection = CallDirection.INBOUND
    customer_id_hash: str = Field(
        ..., description="SHA-256 of MSISDN + tenant salt. Raw phone numbers never enter the lake."
    )
    customer_city: str | None = None
    customer_state: str | None = None

    # Telephony-derived facts we get for free and should never re-derive from audio.
    hold_count: int = 0
    hold_total_ms: int = 0
    transfer_count: int = 0
    queue_wait_ms: int = 0
    disconnect_party: str | None = None

    source_system: str = "synthetic"
    schema_version: str = "bronze.call/1.0.0"


class WordToken(BaseModel):
    """Word-level output from faster-whisper (`word_timestamps=True`).
    `probability` is the single most useful confidence signal we get for free."""

    model_config = ConfigDict(extra="forbid")

    word: str
    start_ms: int
    end_ms: int
    probability: float = Field(..., ge=0.0, le=1.0)


class AsrSegment(BaseModel):
    """Verbatim decoder output. Deliberately *not* cleaned — normalisation happens
    in Silver so we can always trace a Silver utterance back to raw hypothesis."""

    model_config = ConfigDict(extra="forbid")

    call_id: str
    segment_id: int
    channel: int = Field(0, description="0 = agent, 1 = customer for stereo recordings")
    start_ms: int
    end_ms: int
    text_raw: str

    # Whisper decoder telemetry — the basis of ground-truth-free quality estimation.
    avg_logprob: float | None = None
    no_speech_prob: float | None = None
    compression_ratio: float | None = None
    temperature: float | None = None

    whisper_lang: str | None = Field(None, description="Whisper's own utterance-level LID guess")
    whisper_lang_prob: float | None = None

    words: list[WordToken] = Field(default_factory=list)

    asr_model: str = "mock"
    asr_revision: str = "0"
    schema_version: str = "bronze.asr_segment/1.0.0"
