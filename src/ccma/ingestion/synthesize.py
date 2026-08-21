"""Optional TTS rendering of the template bank into stereo call audio.

Off by default (`CCMA_TTS_BACKEND=none`) because the whole pipeline is designed
to run without audio: Bronze ASR segments can be produced by the mock backend
directly from ground-truth text. Turn it on when you want to exercise real
faster-whisper decoding end to end.

Channel layout mirrors real contact-centre recorders: agent on the left channel,
customer on the right. That single choice removes most of the diarization
problem — see ccma.asr.diarization for what we do when only mono is available.
"""

from __future__ import annotations

import asyncio
import shutil
import struct
import wave
from pathlib import Path

from ..logging_conf import get_logger
from ..models.enums import SpeakerRole
from .corpus import GeneratedCall
from .templates import TTS_VOICES

log = get_logger(__name__)

SAMPLE_RATE = 16000


class TtsUnavailable(RuntimeError):
    pass


def _write_silent_stereo(path: Path, duration_ms: int) -> None:
    """Placeholder track so that URIs resolve and durations are correct even
    when no TTS engine is installed."""
    n_frames = int(SAMPLE_RATE * duration_ms / 1000)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(struct.pack("<h", 0) * 2 * n_frames)


async def _edge_tts_render(text: str, voice: str, out_path: Path) -> None:
    import edge_tts  # imported lazily; only needed when TTS is enabled

    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(str(out_path))


def synthesize_call(call: GeneratedCall, audio_dir: Path, backend: str = "none") -> Path | None:
    """Render one call to `audio_dir/<call_id>.wav`.

    With `edge_tts`, each turn is rendered to its own MP3 and then mixed onto the
    correct channel at its ground-truth offset using ffmpeg. ffmpeg is required
    for the mix step; without it we fall back to a silent track and log loudly
    rather than failing the batch."""
    out_path = audio_dir / f"{call.metadata.call_id}.wav"

    if backend == "none":
        return None

    if backend != "edge_tts":
        raise ValueError(f"unknown TTS backend: {backend}")

    if shutil.which("ffmpeg") is None:
        log.warning(
            "ffmpeg not found; writing silent placeholder audio",
            extra={"call_id": call.metadata.call_id},
        )
        _write_silent_stereo(out_path, call.metadata.duration_ms)
        return out_path

    voices = TTS_VOICES.get(call.template.language_style, TTS_VOICES["hinglish"])
    work_dir = audio_dir / "_parts" / call.metadata.call_id
    work_dir.mkdir(parents=True, exist_ok=True)

    async def _render_all() -> list[tuple[Path, int, int]]:
        tasks = []
        parts: list[tuple[Path, int, int]] = []
        for turn in call.turns:
            part_path = work_dir / f"{turn.turn_index:03d}.mp3"
            voice = voices["AGENT" if turn.role == SpeakerRole.AGENT else "CUSTOMER"]
            channel = 0 if turn.role == SpeakerRole.AGENT else 1
            tasks.append(_edge_tts_render(turn.text, voice, part_path))
            parts.append((part_path, turn.start_ms, channel))
        await asyncio.gather(*tasks)
        return parts

    try:
        parts = asyncio.run(_render_all())
    except Exception as exc:  # network failure, voice unavailable, rate limit
        log.error(
            "edge-tts render failed; falling back to silent audio",
            extra={"call_id": call.metadata.call_id, "error": str(exc)},
        )
        _write_silent_stereo(out_path, call.metadata.duration_ms)
        return out_path

    _mix_parts_to_stereo(parts, out_path, call.metadata.duration_ms)
    return out_path


def _mix_parts_to_stereo(
    parts: list[tuple[Path, int, int]], out_path: Path, duration_ms: int
) -> None:
    """ffmpeg filter graph: delay each part to its turn offset, pan it to the
    speaker's channel, then amix everything into one stereo file."""
    import subprocess

    inputs: list[str] = []
    filters: list[str] = []
    for idx, (part_path, offset_ms, channel) in enumerate(parts):
        inputs += ["-i", str(part_path)]
        pan = "c0=c0|c1=0*c0" if channel == 0 else "c0=0*c0|c1=c0"
        filters.append(
            f"[{idx}:a]adelay={offset_ms}|{offset_ms},"
            f"aformat=channel_layouts=stereo,pan=stereo|{pan}[a{idx}]"
        )
    mix_inputs = "".join(f"[a{i}]" for i in range(len(parts)))
    filters.append(
        f"{mix_inputs}amix=inputs={len(parts)}:duration=longest:normalize=0,"
        f"atrim=0:{duration_ms / 1000:.3f}[out]"
    )

    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        *inputs,
        "-filter_complex", ";".join(filters),
        "-map", "[out]",
        "-ar", str(SAMPLE_RATE), "-ac", "2",
        str(out_path),
    ]
    subprocess.run(cmd, check=True, capture_output=True)
