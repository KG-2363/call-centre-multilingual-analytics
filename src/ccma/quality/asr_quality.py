"""Estimating ASR quality when you have no reference transcripts.

This is the question that actually decides whether a multilingual speech
analytics product is trustworthy, because you will never have ground truth for
production traffic — only for whatever tiny slice you paid someone to transcribe.

The approach here is **reference-free quality estimation**: predict WER from
signals that are available at inference time, then calibrate that prediction
against a small human-labelled seed set.

Signals, in descending order of usefulness:

1. **Dual-decode disagreement.** Decode the same audio with two different models
   (large-v3 and small) and measure the normalised edit distance between the
   hypotheses. Where two independently-erring systems agree, both are usually
   right; where they diverge, at least one is wrong. This correlates with true
   WER far better than any single-model confidence, and it needs no labels at
   all. Cost is the reason it is sampled (~10% of traffic) rather than universal.
2. **Mean word probability.** Free from `word_timestamps=True`. Well-ordered but
   badly *scaled* — Whisper is systematically overconfident, which is exactly
   what calibration fixes.
3. **avg_logprob / no_speech_prob / compression_ratio.** Segment-level decoder
   telemetry. Their main value is catching hallucination, which mean word
   probability does not (hallucinated text is emitted confidently).
4. **LM perplexity of the hypothesis** under an in-domain multilingual LM
   (MuRIL/IndicBERT). Garbled ASR is improbable text. Optional — it needs a model
   load, and it mis-fires on legitimate code-switching, so it is weighted low.

**Calibration.** Raw signals are not WER. We take a stratified seed set
(~200 utterances, balanced across language mode and acoustic profile), have it
transcribed by humans once, compute true WER, and fit a monotone (isotonic)
mapping from the fused signal to WER. The result is a *calibrated* point estimate
with an empirical prediction interval — good enough to gate on, which raw
confidence never is.

**Per-language quality without labels.** Because the language layer already tags
every utterance, we group the same calibrated estimate by `lang_primary`. If
estimated WER for `hi_latn` drifts up while `en` stays flat, that is a real,
actionable, label-free regression signal. Absolute values need the seed set;
*relative* movement across languages and over time does not, and relative
movement is what alerting should fire on.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

from ..logging_conf import get_logger
from ..models.bronze import AsrSegment
from ..models.silver import CallQuality, Utterance

log = get_logger(__name__)

QUALITY_BANDS: list[tuple[float, str]] = [
    (0.15, "GOOD"),
    (0.30, "FAIR"),
    (0.50, "POOR"),
    (float("inf"), "UNUSABLE"),
]

# Default weights, used until a calibration file exists. Chosen so the fused
# score is dominated by dual-decode disagreement when it is available and falls
# back gracefully to decoder telemetry when it is not.
DEFAULT_WEIGHTS: dict[str, float] = {
    "intercept": 0.06,
    "inv_word_prob": 0.34,
    "neg_avg_logprob": 0.16,
    "no_speech": 0.10,
    "compression_penalty": 0.06,
    "disagreement": 0.45,
    "lm_perplexity_z": 0.04,
}


@dataclass
class CalibrationModel:
    """Signal -> WER mapping fitted on the human-labelled seed set.

    Isotonic (monotone, piecewise-constant) rather than linear: the relationship
    between confidence and WER is reliably monotone but definitely not linear,
    and isotonic regression will not extrapolate nonsense outside the observed
    range."""

    weights: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_WEIGHTS))
    breakpoints: list[float] = field(default_factory=list)
    values: list[float] = field(default_factory=list)
    residual_std: float = 0.08
    n_samples: int = 0
    per_language_bias: dict[str, float] = field(default_factory=dict)
    fitted: bool = False

    def raw_score(self, signals: dict[str, float]) -> float:
        w = self.weights
        score = w["intercept"]
        score += w["inv_word_prob"] * (1.0 - signals.get("mean_word_prob", 0.7))
        score += w["neg_avg_logprob"] * min(1.0, -signals.get("mean_avg_logprob", -0.3) / 1.5)
        score += w["no_speech"] * signals.get("max_no_speech_prob", 0.0)

        cr = signals.get("mean_compression_ratio", 1.6)
        score += w["compression_penalty"] * min(1.0, abs(cr - 1.6) / 1.2)

        if signals.get("dual_decode_disagreement") is not None:
            score += w["disagreement"] * signals["dual_decode_disagreement"]

        if signals.get("lm_perplexity_z") is not None:
            score += w["lm_perplexity_z"] * max(0.0, signals["lm_perplexity_z"])

        return max(0.0, score)

    def predict(self, signals: dict[str, float], language: str | None = None) -> float:
        raw = self.raw_score(signals)
        wer = self._apply_isotonic(raw) if self.fitted and self.breakpoints else raw
        wer += self.per_language_bias.get(language or "", 0.0)
        return max(0.0, min(1.5, wer))

    def _apply_isotonic(self, x: float) -> float:
        if x <= self.breakpoints[0]:
            return self.values[0]
        if x >= self.breakpoints[-1]:
            return self.values[-1]
        for i in range(1, len(self.breakpoints)):
            if x <= self.breakpoints[i]:
                x0, x1 = self.breakpoints[i - 1], self.breakpoints[i]
                y0, y1 = self.values[i - 1], self.values[i]
                t = 0.0 if x1 == x0 else (x - x0) / (x1 - x0)
                return y0 + t * (y1 - y0)
        return self.values[-1]

    def interval(self, point: float) -> tuple[float, float]:
        """~80% prediction interval from the calibration residuals."""
        margin = 1.28 * self.residual_std
        return max(0.0, point - margin), min(1.5, point + margin)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "weights": self.weights,
                    "breakpoints": self.breakpoints,
                    "values": self.values,
                    "residual_std": self.residual_std,
                    "n_samples": self.n_samples,
                    "per_language_bias": self.per_language_bias,
                    "fitted": self.fitted,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path) -> CalibrationModel:
        if not path.exists():
            return cls()
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(**data)


def _pool_adjacent_violators(xs: list[float], ys: list[float]) -> tuple[list[float], list[float]]:
    """Isotonic regression via PAVA. Implemented directly to keep the calibration
    path dependency-free — sklearn is optional in this project."""
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    sorted_x = [xs[i] for i in order]
    blocks: list[list[float]] = [[ys[i]] for i in order]
    block_x: list[float] = list(sorted_x)

    i = 0
    while i < len(blocks) - 1:
        mean_i = sum(blocks[i]) / len(blocks[i])
        mean_next = sum(blocks[i + 1]) / len(blocks[i + 1])
        if mean_i > mean_next:
            blocks[i].extend(blocks.pop(i + 1))
            block_x[i] = max(block_x[i], block_x.pop(i + 1))
            if i > 0:
                i -= 1
        else:
            i += 1

    return block_x, [sum(b) / len(b) for b in blocks]


def fit_calibration(
    samples: list[tuple[dict[str, float], float, str | None]],
) -> CalibrationModel:
    """Fit on (signals, true_wer, language) triples from the labelled seed set."""
    model = CalibrationModel()
    if len(samples) < 10:
        log.warning("too few calibration samples; keeping defaults", extra={"n": len(samples)})
        return model

    raw_scores = [model.raw_score(s) for s, _, _ in samples]
    true_wers = [w for _, w, _ in samples]

    breakpoints, values = _pool_adjacent_violators(raw_scores, true_wers)
    model.breakpoints, model.values = breakpoints, values
    model.fitted = True
    model.n_samples = len(samples)

    residuals = [
        model._apply_isotonic(raw) - true for raw, true in zip(raw_scores, true_wers)
    ]
    mean_res = sum(residuals) / len(residuals)
    model.residual_std = round(
        math.sqrt(sum((r - mean_res) ** 2 for r in residuals) / max(1, len(residuals) - 1)), 4
    )

    # Residual bias per language: catches the systematic component our shared
    # signal fusion cannot express (e.g. we are consistently optimistic on
    # hi_latn because word probabilities there are misleadingly high).
    by_lang: dict[str, list[float]] = {}
    for (signals, true_wer, lang) in samples:
        if lang:
            by_lang.setdefault(lang, []).append(model._apply_isotonic(model.raw_score(signals)) - true_wer)
    model.per_language_bias = {
        lang: round(-sum(rs) / len(rs), 4) for lang, rs in by_lang.items() if len(rs) >= 5
    }

    log.info(
        "fitted ASR calibration",
        extra={
            "n": model.n_samples,
            "residual_std": model.residual_std,
            "languages": list(model.per_language_bias),
        },
    )
    return model


def collect_signals(
    segments: list[AsrSegment], disagreement: float | None = None, lm_perplexity_z: float | None = None
) -> dict[str, float]:
    if not segments:
        return {
            "mean_word_prob": 0.0,
            "mean_avg_logprob": -2.0,
            "max_no_speech_prob": 1.0,
            "mean_compression_ratio": 0.0,
            "dual_decode_disagreement": disagreement,
            "lm_perplexity_z": lm_perplexity_z,
        }

    all_words = [w for s in segments for w in s.words]
    logprobs = [s.avg_logprob for s in segments if s.avg_logprob is not None]
    no_speech = [s.no_speech_prob for s in segments if s.no_speech_prob is not None]
    compression = [s.compression_ratio for s in segments if s.compression_ratio is not None]

    return {
        "mean_word_prob": (
            sum(w.probability for w in all_words) / len(all_words) if all_words else 0.6
        ),
        "mean_avg_logprob": sum(logprobs) / len(logprobs) if logprobs else -0.5,
        "max_no_speech_prob": max(no_speech) if no_speech else 0.0,
        "mean_compression_ratio": sum(compression) / len(compression) if compression else 1.6,
        "dual_decode_disagreement": disagreement,
        "lm_perplexity_z": lm_perplexity_z,
    }


def band_for(wer: float) -> str:
    for threshold, label in QUALITY_BANDS:
        if wer < threshold:
            return label
    return "UNUSABLE"


def assess_call(
    call_id: str,
    segments: list[AsrSegment],
    utterances: list[Utterance],
    model: CalibrationModel,
    disagreement: float | None = None,
    lm_perplexity_z: float | None = None,
) -> CallQuality:
    signals = collect_signals(segments, disagreement, lm_perplexity_z)
    point = model.predict(signals)
    lo, hi = model.interval(point)

    per_language: dict[str, float] = {}
    by_lang: dict[str, list[Utterance]] = {}
    for utt in utterances:
        by_lang.setdefault(utt.lang_primary.value, []).append(utt)

    segments_by_id = {s.segment_id: s for s in segments}
    for lang, utts in by_lang.items():
        lang_segments = [
            segments_by_id[sid] for u in utts for sid in u.segment_ids if sid in segments_by_id
        ]
        if not lang_segments:
            continue
        lang_signals = collect_signals(lang_segments, disagreement, lm_perplexity_z)
        per_language[lang] = round(model.predict(lang_signals, language=lang), 4)

    low_conf = sum(1 for u in utterances if u.asr_confidence < 0.5)
    unintelligible = sum(1 for u in utterances if "hallucination_suspect" in u.quality_flags)

    return CallQuality(
        call_id=call_id,
        mean_word_prob=round(signals["mean_word_prob"], 4),
        mean_avg_logprob=round(signals["mean_avg_logprob"], 4),
        max_no_speech_prob=round(signals["max_no_speech_prob"], 4),
        mean_compression_ratio=round(signals["mean_compression_ratio"], 4),
        dual_decode_disagreement=disagreement,
        lm_perplexity_z=lm_perplexity_z,
        estimated_wer=round(point, 4),
        estimated_wer_lo=round(lo, 4),
        estimated_wer_hi=round(hi, 4),
        quality_band=band_for(point),
        per_language_wer=per_language,
        low_confidence_utterance_ratio=round(low_conf / len(utterances), 4) if utterances else 0.0,
        unintelligible_ratio=round(unintelligible / len(utterances), 4) if utterances else 0.0,
    )
