"""faster-whisper backend.

Configuration choices and why:

* **large-v3** for the primary pass. On Indian-accented English and Hindi it is
  meaningfully better than medium; on Hinglish, nothing off-the-shelf is good, so
  we buy what accuracy we can and measure the rest (see docs/03).
* **int8_float16** compute type. ~2x throughput over float16 at a WER cost that
  sits inside the noise band on our eval set — the right trade when the bottleneck
  is cost per audio-hour, not single-call latency.
* **VAD filter on.** Contact-centre recordings are 30-60% silence and hold music.
  Silero VAD removes it before decoding, which both cuts GPU spend proportionally
  and suppresses Whisper's well-known tendency to hallucinate text over silence.
* **word_timestamps=True.** Word probabilities are the backbone of our
  ground-truth-free quality estimate, and word-level spans are what let us
  re-align a segment to a diarization turn.
* **condition_on_previous_text=False.** Whisper's default carries the previous
  segment into the prompt, which on code-switched audio causes language lock-in:
  once it decides the call is Hindi it keeps transcribing English spans into
  Devanagari. Disabling costs a little fluency and buys correct language handling.
* **No `language=` on mixed queues.** Forcing a language is right for a known
  monolingual queue and actively harmful on Hinglish.

Per-channel decoding: for stereo recordings we decode each channel separately.
Speaker attribution then comes from the channel index, which is exact — far
better than any diarizer. Mono audio falls back to ccma.asr.diarization.
"""

from __future__ import annotations

import time
from pathlib import Path

from ..config import Settings
from ..logging_conf import get_logger
from ..models.bronze import AsrSegment, WordToken
from .base import AsrBackend, TranscriptionRequest

log = get_logger(__name__)


class FasterWhisperBackend(AsrBackend):
    name = "faster-whisper"

    def __init__(self, settings: Settings, model_size: str | None = None) -> None:
        self.settings = settings
        self.model_size = model_size or settings.whisper_model
        self.revision = self.model_size
        self._model = None

    def _load(self):
        if self._model is None:
            from faster_whisper import WhisperModel

            device = self.settings.whisper_device
            if device == "auto":
                try:
                    import torch

                    device = "cuda" if torch.cuda.is_available() else "cpu"
                except ImportError:
                    device = "cpu"

            compute_type = self.settings.whisper_compute_type
            if device == "cpu" and compute_type in ("float16", "int8_float16"):
                compute_type = "int8"  # float16 is not supported on CPU

            log.info(
                "loading whisper",
                extra={"model": self.model_size, "device": device, "compute": compute_type},
            )
            self._model = WhisperModel(self.model_size, device=device, compute_type=compute_type)
        return self._model

    def warmup(self) -> None:
        self._load()

    def transcribe(self, request: TranscriptionRequest) -> list[AsrSegment]:
        if request.audio_path is None or not Path(request.audio_path).exists():
            raise FileNotFoundError(
                f"audio missing for {request.call.call_id}; "
                "run `ccma seed --tts edge_tts` or set CCMA_ASR_BACKEND=mock"
            )

        model = self._load()
        started = time.perf_counter()

        channels_to_decode = [0, 1] if request.call.channels == 2 else [0]
        segments: list[AsrSegment] = []
        seg_id = 0

        for channel in channels_to_decode:
            audio_input = self._channel_audio(request.audio_path, channel, request.call.channels)

            raw_segments, info = model.transcribe(
                audio_input,
                language=request.language_hint,
                beam_size=request.beam_size,
                vad_filter=request.vad_filter,
                vad_parameters={"min_silence_duration_ms": 500},
                word_timestamps=True,
                condition_on_previous_text=False,
                temperature=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
            )

            for seg in raw_segments:
                segments.append(
                    AsrSegment(
                        call_id=request.call.call_id,
                        segment_id=seg_id,
                        channel=channel,
                        start_ms=int(seg.start * 1000),
                        end_ms=int(seg.end * 1000),
                        text_raw=seg.text.strip(),
                        avg_logprob=seg.avg_logprob,
                        no_speech_prob=seg.no_speech_prob,
                        compression_ratio=seg.compression_ratio,
                        temperature=getattr(seg, "temperature", None),
                        whisper_lang=info.language,
                        whisper_lang_prob=info.language_probability,
                        words=[
                            WordToken(
                                word=w.word.strip(),
                                start_ms=int(w.start * 1000),
                                end_ms=int(w.end * 1000),
                                probability=float(w.probability),
                            )
                            for w in (seg.words or [])
                            if w.word.strip()
                        ],
                        asr_model=f"{self.name}:{self.model_size}",
                        asr_revision=self.revision,
                    )
                )
                seg_id += 1

        segments.sort(key=lambda s: (s.start_ms, s.channel))
        for new_id, seg in enumerate(segments):
            seg.segment_id = new_id

        log.info(
            "transcribed",
            extra={
                "call_id": request.call.call_id,
                "segments": len(segments),
                "elapsed_s": round(time.perf_counter() - started, 2),
            },
        )
        return segments

    @staticmethod
    def _channel_audio(path: Path, channel: int, total_channels: int):
        """Extract one channel as a float32 array for the decoder.

        Reading the WAV directly avoids an ffmpeg round-trip per channel, which
        at scale is a real share of pipeline CPU."""
        if total_channels == 1:
            return str(path)

        import numpy as np
        import wave

        with wave.open(str(path), "rb") as wf:
            n_channels = wf.getnchannels()
            frames = wf.readframes(wf.getnframes())

        audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
        if n_channels > 1:
            audio = audio.reshape(-1, n_channels)[:, min(channel, n_channels - 1)]
        return np.ascontiguousarray(audio)
