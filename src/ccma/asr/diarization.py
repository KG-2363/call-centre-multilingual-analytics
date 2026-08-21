"""Speaker attribution and diarization-error repair.

Ranked by reliability — always take the highest tier available:

**Tier 1 — channel separation (exact).** Contact-centre recorders almost always
capture agent and customer on separate channels. Attribution is then a lookup,
not an inference, with zero diarization error. Any project that runs a diarizer
over stereo call audio is choosing a worse answer for no reason. We insist on
stereo at the recorder-config level; it is the cheapest accuracy win available.

**Tier 2 — diarizer + role classification (mono).** Legacy mono archives need
pyannote to cluster speakers, then a role classifier to decide which cluster is
the agent. Errors here are systematic, not random, and are what the repair
functions below target.

Known failure modes and the mitigations implemented here:

* *Overlapped speech* — diarizers assign the overlap to one speaker, so an
  interruption gets attributed to whoever "won". We detect cross-speaker temporal
  overlap and flag both turns rather than trusting either. Flagged turns are
  excluded from talk-ratio KPIs, which would otherwise be badly skewed.
* *Speaker flip mid-call* — clustering swaps identities after a long hold or a
  transfer. Detected via a linguistic role score computed in sliding windows; a
  sustained sign reversal triggers a relabel of the affected span.
* *Over-segmentation* — one utterance split into many turns by short pauses.
  Repaired by merging same-speaker segments separated by less than `merge_gap_ms`.
* *Short-segment noise* — sub-400 ms fragments are usually backchannels
  ("hmm", "ji") mis-clustered. Merged into the neighbouring turn.
"""

from __future__ import annotations

from dataclasses import dataclass

import regex as re

from ..logging_conf import get_logger
from ..models.bronze import AsrSegment
from ..models.enums import SpeakerRole

log = get_logger(__name__)

# Phrases that identify the agent side. Multilingual on purpose — a
# monolingual role classifier fails on exactly the calls we care about.
_AGENT_CUES = [
    r"\bthis call is being recorded\b",
    r"\bfor quality and training\b",
    r"रिकॉर्ड की जा रही",
    r"\brecord ki ja rahi\b",
    r"\bhow may i (help|assist)\b",
    r"\bmy name is\b",
    r"\bmain .{0,18}bol (raha|rahi) (hoon|hun)\b",
    r"मैं .{0,18}बोल (रहा|रही) हूँ",
    r"\bthank you for calling\b",
    r"\bplease (hold|stay on the line)\b",
    r"\bhold kar(iye|o)\b",
    r"\bकृपया प्रतीक्षा\b",
    r"\bverification ke liye\b",
    r"\bmay i have your\b",
    r"\bcomplaint (number|reference)\b",
    r"\bमैं जाँच (कर|करती|करता)\b",
    r"\bmain check kar(ta|ti) hoon\b",
    r"\bhave a (great|good|nice) day\b",
]

# Phrases that identify the customer side.
_CUSTOMER_CUES = [
    r"\bi (want|need) to\b",
    r"\bmy (bill|account|plan|connection|order|refund)\b",
    r"\bmera (bill|paisa|account|plan|number|order)\b",
    r"\bमेरा (बिल|पैसा|खाता|कनेक्शन)\b",
    r"\bmujhe .{0,20}(chahiye|karna|problem)\b",
    r"\bमुझे .{0,20}(चाहिए|दिक्कत)\b",
    r"\bmain .{0,12}call kar raha hoon\b",
    r"\byeh galat hai\b",
    r"\bकट गए\b",
    r"\bkat ga(ye|ya)\b",
    r"\baap log\b",
    r"\bmujhe aapke senior se baat karni hai\b",
    r"\bconsumer forum\b",
]

_AGENT_RE = [re.compile(p, re.IGNORECASE) for p in _AGENT_CUES]
_CUSTOMER_RE = [re.compile(p, re.IGNORECASE) for p in _CUSTOMER_CUES]


@dataclass
class SpeakerTurn:
    turn_index: int
    speaker_role: SpeakerRole
    speaker_id: str
    start_ms: int
    end_ms: int
    segment_ids: list[int]
    text: str
    confidence: float
    overlapped_with_next: bool = False
    flags: list[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.flags is None:
            self.flags = []


def role_score(text: str) -> float:
    """+1 => certainly agent, -1 => certainly customer, 0 => no evidence."""
    agent_hits = sum(1 for r in _AGENT_RE if r.search(text))
    customer_hits = sum(1 for r in _CUSTOMER_RE if r.search(text))
    total = agent_hits + customer_hits
    if total == 0:
        return 0.0
    return (agent_hits - customer_hits) / total


def assign_roles_from_channel(segments: list[AsrSegment]) -> dict[int, SpeakerRole]:
    """Tier 1. Channel 0 = agent by recorder convention; we verify that against
    linguistic cues and swap if the recorder was wired the other way round,
    because a mis-wired site would otherwise invert every agent metric."""
    by_channel: dict[int, list[str]] = {}
    for seg in segments:
        by_channel.setdefault(seg.channel, []).append(seg.text_raw)

    if len(by_channel) < 2:
        return {ch: SpeakerRole.UNKNOWN for ch in by_channel}

    scores = {ch: sum(role_score(t) for t in texts) for ch, texts in by_channel.items()}
    agent_channel = max(scores, key=lambda c: scores[c])

    # No linguistic evidence either way -> fall back to the convention.
    if abs(scores[agent_channel] - min(scores.values())) < 0.5:
        agent_channel = 0

    return {
        ch: SpeakerRole.AGENT if ch == agent_channel else SpeakerRole.CUSTOMER
        for ch in by_channel
    }


def diarize_mono(audio_path: str, num_speakers: int | None = 2) -> list[tuple[float, float, str]]:
    """Tier 2. Returns (start_s, end_s, speaker_label) using pyannote.

    Kept behind a lazy import and a clear error: mono diarization is a heavy,
    gated dependency and most deployments should never reach this path."""
    try:
        from pyannote.audio import Pipeline
    except ImportError as exc:
        raise RuntimeError(
            "mono diarization requires `pip install pyannote.audio` and a HuggingFace "
            "token for pyannote/speaker-diarization-3.1. Prefer stereo recording instead."
        ) from exc

    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")
    annotation = pipeline(audio_path, num_speakers=num_speakers)
    return [
        (turn.start, turn.end, label)
        for turn, _, label in annotation.itertracks(yield_label=True)
    ]


def build_turns(
    segments: list[AsrSegment],
    role_by_channel: dict[int, SpeakerRole],
    call_id: str,
    merge_gap_ms: int = 900,
    min_turn_ms: int = 400,
) -> list[SpeakerTurn]:
    """Collapse ASR segments into speaker turns, repairing the known failures."""
    if not segments:
        return []

    ordered = sorted(segments, key=lambda s: (s.start_ms, s.channel))
    turns: list[SpeakerTurn] = []

    for seg in ordered:
        role = role_by_channel.get(seg.channel, SpeakerRole.UNKNOWN)
        if turns and turns[-1].speaker_role == role and seg.start_ms - turns[-1].end_ms <= merge_gap_ms:
            prev = turns[-1]
            prev.end_ms = max(prev.end_ms, seg.end_ms)
            prev.segment_ids.append(seg.segment_id)
            prev.text = f"{prev.text} {seg.text_raw}".strip()
        else:
            turns.append(
                SpeakerTurn(
                    turn_index=len(turns),
                    speaker_role=role,
                    speaker_id=f"{call_id}:{role.value}",
                    start_ms=seg.start_ms,
                    end_ms=seg.end_ms,
                    segment_ids=[seg.segment_id],
                    text=seg.text_raw,
                    confidence=1.0 if role != SpeakerRole.UNKNOWN else 0.4,
                )
            )

    turns = _absorb_short_turns(turns, min_turn_ms)
    turns = _flag_overlaps(turns)
    turns = _repair_speaker_flip(turns)

    for idx, turn in enumerate(turns):
        turn.turn_index = idx
    return turns


def _absorb_short_turns(turns: list[SpeakerTurn], min_turn_ms: int) -> list[SpeakerTurn]:
    """A sub-400 ms turn between two turns by the same other speaker is almost
    always a backchannel mis-assigned by the diarizer. Absorb it and record the
    repair so the flag is visible downstream rather than silently applied."""
    if len(turns) < 3:
        return turns

    out: list[SpeakerTurn] = [turns[0]]
    for i in range(1, len(turns) - 1):
        cur, prev, nxt = turns[i], out[-1], turns[i + 1]
        is_short = (cur.end_ms - cur.start_ms) < min_turn_ms
        sandwiched = prev.speaker_role == nxt.speaker_role != cur.speaker_role
        if is_short and sandwiched and len(cur.text.split()) <= 3:
            prev.end_ms = max(prev.end_ms, cur.end_ms)
            prev.segment_ids.extend(cur.segment_ids)
            prev.text = f"{prev.text} {cur.text}".strip()
            prev.flags.append("absorbed_backchannel")
            prev.confidence = min(prev.confidence, 0.85)
        else:
            out.append(cur)
    out.append(turns[-1])
    return out


def _flag_overlaps(turns: list[SpeakerTurn]) -> list[SpeakerTurn]:
    """Mark cross-speaker temporal overlap. We do not attempt to un-mix the
    audio; we mark it so KPIs can exclude it and reviewers can see why."""
    for a, b in zip(turns, turns[1:]):
        if b.start_ms < a.end_ms and a.speaker_role != b.speaker_role:
            overlap_ms = a.end_ms - b.start_ms
            a.overlapped_with_next = True
            a.flags.append("overlapped_speech")
            b.flags.append("overlapped_speech")
            penalty = min(0.35, overlap_ms / 4000)
            a.confidence = round(max(0.3, a.confidence - penalty), 3)
            b.confidence = round(max(0.3, b.confidence - penalty), 3)
    return turns


def _repair_speaker_flip(turns: list[SpeakerTurn], window: int = 4) -> list[SpeakerTurn]:
    """Detect a sustained inversion between assigned role and linguistic role.

    A single contrary turn means nothing (agents quote customers, customers read
    back reference numbers). We require the whole window to disagree with decent
    magnitude before relabelling."""
    if len(turns) < window * 2:
        return turns

    for start in range(0, len(turns) - window + 1):
        chunk = turns[start : start + window]
        scores = [role_score(t.text) for t in chunk]
        if any(abs(s) < 0.5 for s in scores):
            continue

        inverted = all(
            (s > 0 and t.speaker_role == SpeakerRole.CUSTOMER)
            or (s < 0 and t.speaker_role == SpeakerRole.AGENT)
            for s, t in zip(scores, chunk)
        )
        if inverted:
            log.warning(
                "speaker flip detected; relabelling window",
                extra={"start_turn": chunk[0].turn_index, "window": window},
            )
            for turn in chunk:
                turn.speaker_role = (
                    SpeakerRole.AGENT
                    if turn.speaker_role == SpeakerRole.CUSTOMER
                    else SpeakerRole.CUSTOMER
                )
                turn.speaker_id = turn.speaker_id.rsplit(":", 1)[0] + f":{turn.speaker_role.value}"
                turn.flags.append("speaker_flip_repaired")
                turn.confidence = round(min(turn.confidence, 0.7), 3)
    return turns
