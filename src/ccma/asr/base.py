"""ASR backend interface.

Two implementations exist (faster-whisper, mock) and the rest of the pipeline is
written against this protocol only. That indirection buys three things:

* CI and the offline demo run with zero model weights.
* Swapping in IndicWhisper / AI4Bharat models, or a managed API, is a one-class
  change with no downstream edits.
* The dual-decode quality signal (primary vs. shadow model) is expressible as
  "run the protocol twice with different configs".
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

from ..models.bronze import AsrSegment, CallMetadata


@dataclass
class TranscriptionRequest:
    call: CallMetadata
    audio_path: Path | None
    # Whisper's `language=None` triggers auto-detect. We pass a hint only when the
    # queue is known to be monolingual, because forcing `hi` on a Hinglish call
    # measurably degrades the English spans.
    language_hint: str | None = None
    beam_size: int = 5
    vad_filter: bool = True


class AsrBackend(ABC):
    name: str = "base"
    revision: str = "0"

    @abstractmethod
    def transcribe(self, request: TranscriptionRequest) -> list[AsrSegment]:
        """Return verbatim segments. Must populate decoder telemetry
        (avg_logprob / no_speech_prob / compression_ratio) where available —
        the quality layer depends on it."""

    def warmup(self) -> None:
        """Load weights ahead of the first request so latency percentiles are
        not polluted by a cold start."""
        return None


def get_backend(settings=None) -> AsrBackend:
    from ..config import get_settings

    settings = settings or get_settings()
    if settings.asr_backend == "faster_whisper":
        from .faster_whisper_backend import FasterWhisperBackend

        return FasterWhisperBackend(settings)
    if settings.asr_backend == "mock":
        from .mock_backend import MockAsrBackend

        return MockAsrBackend(settings)
    raise ValueError(f"unknown ASR backend: {settings.asr_backend}")
