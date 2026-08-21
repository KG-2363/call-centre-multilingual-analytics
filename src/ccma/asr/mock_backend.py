"""Deterministic offline ASR backend.

It reconstructs the ground-truth turns for a call (same seed => same corpus) and
degrades them into a realistic hypothesis. This is not a toy: the degradation
model reproduces the error *types* that matter for this domain, so every
downstream component — LID fusion, confidence thresholds, quality estimation,
the human-review queue — is exercised against plausible input.

Simulated error classes, in rough order of real-world frequency on Indian call
audio:

1. **Romanisation drift** — the same Hindi word decoded with a different Latin
   spelling (`nahi`/`nahin`/`nhi`). Harmless to a human, fatal to exact matching.
2. **Script flip** — a Hindi word emitted in Devanagari inside an otherwise Latin
   utterance (or vice versa), which is what Whisper actually does at code-switch
   boundaries.
3. **Near-homophone substitution** — `paisa`/`pesa`, `bill`/`build`.
4. **Deletion** of short function words under noise.
5. **Hallucinated filler** over silence — the classic Whisper failure mode.
6. **Segment merge/split** relative to true turn boundaries.

Error probability scales with the call's acoustic `difficulty`, so the corpus
spans clean and hostile conditions.
"""

from __future__ import annotations

import math
import random

from ..config import Settings
from ..logging_conf import get_logger
from ..models.bronze import AsrSegment, WordToken
from ..models.enums import SpeakerRole
from .base import AsrBackend, TranscriptionRequest

log = get_logger(__name__)

# Romanisation variants ASR realistically swaps between.
_ROMAN_DRIFT: dict[str, list[str]] = {
    "nahi": ["nahin", "nhi", "nahee"],
    "nahin": ["nahi", "nhi"],
    "hai": ["hain", "he"],
    "hain": ["hai", "hein"],
    "bahut": ["bohot", "bahot"],
    "paisa": ["pesa", "paise"],
    "paise": ["pesa", "paisa"],
    "kar": ["ker"],
    "theek": ["thik", "tik"],
    "achha": ["acha", "accha"],
    "dhanyavaad": ["dhanyavad", "dhanyawad"],
    "shikayat": ["shikaayat"],
    "samajh": ["samjh"],
    "jaldi": ["jaldee"],
    "zyada": ["jyada"],
    "rupaye": ["rupay", "rupaiya"],
    "kyun": ["kyu", "kyon"],
    "abhi": ["abi"],
    "thoda": ["thora"],
}

# Cross-lingual near-homophones — the substitutions that silently change meaning.
_HOMOPHONE_SWAP: dict[str, list[str]] = {
    "bill": ["build", "bil"],
    "plan": ["plane", "plaan"],
    "data": ["date", "deta"],
    "network": ["net work"],
    "signal": ["single"],
    "charge": ["church", "charj"],
    "refund": ["re fund"],
    "account": ["a count"],
    "sir": ["sar", "sur"],
    "card": ["cart", "kard"],
    "band": ["band", "bandh"],
    "kat": ["cat", "kaat"],
}

_DEVANAGARI_FLIP: dict[str, str] = {
    "nahi": "नहीं", "hai": "है", "paisa": "पैसा", "aap": "आप", "main": "मैं",
    "kya": "क्या", "bahut": "बहुत", "theek": "ठीक", "haan": "हाँ",
    "shikayat": "शिकायत", "dhanyavaad": "धन्यवाद", "samasya": "समस्या",
}

_HALLUCINATIONS = [
    "thank you for watching",
    "subscribe",
    "please",
    "hmm",
    "okay okay",
]


class MockAsrBackend(AsrBackend):
    name = "mock"
    revision = "degradation-v1"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._corpus_index: dict[str, object] | None = None

    def _index(self) -> dict:
        if self._corpus_index is None:
            from ..ingestion.corpus import generate_corpus

            calls = generate_corpus(
                n_calls=self.settings.corpus_size, seed=self.settings.random_seed
            )
            self._corpus_index = {c.metadata.call_id: c for c in calls}
        return self._corpus_index  # type: ignore[return-value]

    def transcribe(self, request: TranscriptionRequest) -> list[AsrSegment]:
        call_id = request.call.call_id
        generated = self._index().get(call_id)
        if generated is None:
            raise KeyError(
                f"{call_id} not in the generated corpus; "
                "the mock backend requires the same CCMA_RANDOM_SEED used by `ccma seed`"
            )

        # Per-call RNG keyed on call_id: regenerating one call does not perturb
        # any other, which keeps diffs reviewable.
        rng = random.Random(f"{self.settings.random_seed}:{call_id}")
        difficulty = generated.difficulty

        segments: list[AsrSegment] = []
        seg_id = 0

        for turn in generated.turns:
            channel = 0 if turn.role == SpeakerRole.AGENT else 1
            hyp_words, word_probs = self._degrade(turn.text, difficulty, rng)

            if not hyp_words:
                continue

            # Whisper emits ~10-15s chunks, not conversational turns. Split long
            # turns so downstream code must handle the realistic mismatch between
            # ASR segments and speaker turns.
            for chunk_words, chunk_probs, c_start, c_end in self._chunk(
                hyp_words, word_probs, turn.start_ms, turn.end_ms, rng
            ):
                text = " ".join(chunk_words)
                avg_logprob = self._avg_logprob(difficulty, rng)
                segments.append(
                    AsrSegment(
                        call_id=call_id,
                        segment_id=seg_id,
                        channel=channel,
                        start_ms=c_start,
                        end_ms=c_end,
                        text_raw=text,
                        avg_logprob=round(avg_logprob, 4),
                        no_speech_prob=round(min(0.95, abs(rng.gauss(0.04, 0.05)) + difficulty * 0.15), 4),
                        compression_ratio=round(rng.uniform(1.1, 2.4), 3),
                        temperature=0.0 if difficulty < 0.2 else round(rng.choice([0.0, 0.2, 0.4]), 1),
                        whisper_lang=self._whisper_lang_guess(turn.lang, rng),
                        whisper_lang_prob=round(rng.uniform(0.55, 0.98), 4),
                        words=[
                            WordToken(
                                word=w,
                                start_ms=c_start + int((c_end - c_start) * i / max(1, len(chunk_words))),
                                end_ms=c_start + int((c_end - c_start) * (i + 1) / max(1, len(chunk_words))),
                                probability=round(p, 4),
                            )
                            for i, (w, p) in enumerate(zip(chunk_words, chunk_probs))
                        ],
                        asr_model=f"mock:{self.revision}",
                        asr_revision=self.revision,
                    )
                )
                seg_id += 1

        # Hallucinated segment over a long silence — rare, but the pipeline must
        # not treat it as a real customer statement.
        if difficulty > 0.25 and rng.random() < 0.30 and segments:
            anchor = rng.choice(segments)
            segments.append(
                AsrSegment(
                    call_id=call_id,
                    segment_id=len(segments),
                    channel=anchor.channel,
                    start_ms=anchor.end_ms + 400,
                    end_ms=anchor.end_ms + 1400,
                    text_raw=rng.choice(_HALLUCINATIONS),
                    avg_logprob=round(rng.uniform(-1.9, -1.1), 4),
                    no_speech_prob=round(rng.uniform(0.55, 0.9), 4),
                    compression_ratio=round(rng.uniform(0.4, 0.9), 3),
                    temperature=0.8,
                    whisper_lang="en",
                    whisper_lang_prob=0.5,
                    words=[],
                    asr_model=f"mock:{self.revision}",
                    asr_revision=self.revision,
                )
            )

        segments.sort(key=lambda s: (s.start_ms, s.channel))
        for new_id, seg in enumerate(segments):
            seg.segment_id = new_id
        return segments

    # -- degradation helpers -------------------------------------------------

    def _degrade(
        self, text: str, difficulty: float, rng: random.Random
    ) -> tuple[list[str], list[float]]:
        words = text.split()
        out_words: list[str] = []
        out_probs: list[float] = []

        for word in words:
            bare = word.strip(".,?!।").lower()
            roll = rng.random()

            if roll < difficulty * 0.30 and bare in _ROMAN_DRIFT:
                emitted = rng.choice(_ROMAN_DRIFT[bare])
                confidence = rng.uniform(0.45, 0.72)
            elif roll < difficulty * 0.45 and bare in _HOMOPHONE_SWAP:
                emitted = rng.choice(_HOMOPHONE_SWAP[bare])
                confidence = rng.uniform(0.35, 0.62)
            elif roll < difficulty * 0.55 and bare in _DEVANAGARI_FLIP:
                emitted = _DEVANAGARI_FLIP[bare]
                confidence = rng.uniform(0.50, 0.78)
            elif roll < difficulty * 0.62 and len(bare) <= 3:
                continue  # deletion of a short function word
            else:
                emitted = word
                confidence = rng.uniform(0.78, 0.99) - difficulty * 0.18

            out_words.append(emitted)
            out_probs.append(max(0.05, min(0.999, confidence)))

            if rng.random() < difficulty * 0.05:  # spurious insertion
                out_words.append(rng.choice(["uh", "hmm", "the"]))
                out_probs.append(rng.uniform(0.2, 0.45))

        return out_words, out_probs

    @staticmethod
    def _chunk(
        words: list[str], probs: list[float], start_ms: int, end_ms: int, rng: random.Random
    ):
        """Split a turn into 1-2 pseudo-Whisper segments."""
        if len(words) <= 12 or rng.random() < 0.5:
            yield words, probs, start_ms, end_ms
            return
        cut = rng.randint(5, len(words) - 4)
        mid = start_ms + int((end_ms - start_ms) * cut / len(words))
        yield words[:cut], probs[:cut], start_ms, mid
        yield words[cut:], probs[cut:], mid, end_ms

    @staticmethod
    def _avg_logprob(difficulty: float, rng: random.Random) -> float:
        """Whisper's avg_logprob sits near -0.15 on clean speech and drops toward
        -1.2 as conditions degrade. Modelled as a noisy monotone function."""
        base = -0.12 - 1.05 * math.pow(difficulty, 0.8)
        return max(-2.0, min(-0.02, base + rng.gauss(0, 0.09)))

    @staticmethod
    def _whisper_lang_guess(true_lang: str, rng: random.Random) -> str:
        """Reproduces the failure this whole project exists to handle: Whisper
        labels romanised Hindi and Hinglish as English most of the time."""
        if true_lang == "hi":
            return "hi" if rng.random() < 0.93 else "en"
        if true_lang == "en":
            return "en" if rng.random() < 0.96 else "hi"
        if true_lang == "hi_latn":
            return "en" if rng.random() < 0.72 else "hi"
        return "en" if rng.random() < 0.80 else "hi"  # hinglish
