"""Edit-distance metrics used for both evaluation and reference-free QA.

Normalisation before scoring is not cosmetic here. On Hinglish, a naive WER
punishes `nahi` vs `nahin` as a full substitution even though both are correct
renderings of नहीं. We therefore report three numbers:

* **WER** — standard, on normalised text. Comparable to published benchmarks.
* **CER** — character-level. More stable across romanisation variance and the
  right primary metric for Devanagari, where word segmentation is itself noisy.
* **Transliteration-invariant WER** — both sides projected to Devanagari before
  scoring, so `paisa kat gaya` and `पैसा कट गया` score as identical. This is the
  number that reflects whether we understood the customer; the plain WER reflects
  whether we spelled it the same way. Reporting only the former hides real ASR
  degradation, reporting only the latter overstates it on code-switched audio.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..lang.lid import normalize_text
from ..lang.translit import back_transliterate


def levenshtein(a: list[str], b: list[str]) -> int:
    """Standard DP edit distance, O(min(|a|,|b|)) memory."""
    if len(a) < len(b):
        a, b = b, a
    if not b:
        return len(a)

    previous = list(range(len(b) + 1))
    for i, token_a in enumerate(a, 1):
        current = [i]
        for j, token_b in enumerate(b, 1):
            current.append(
                min(
                    previous[j] + 1,           # deletion
                    current[j - 1] + 1,        # insertion
                    previous[j - 1] + (token_a != token_b),  # substitution
                )
            )
        previous = current
    return previous[-1]


@dataclass
class ErrorRates:
    wer: float
    cer: float
    translit_invariant_wer: float
    ref_words: int
    hyp_words: int


def word_error_rate(reference: str, hypothesis: str) -> float:
    ref = normalize_text(reference).split()
    hyp = normalize_text(hypothesis).split()
    if not ref:
        return 0.0 if not hyp else 1.0
    return levenshtein(ref, hyp) / len(ref)


def character_error_rate(reference: str, hypothesis: str) -> float:
    ref = list(normalize_text(reference).replace(" ", ""))
    hyp = list(normalize_text(hypothesis).replace(" ", ""))
    if not ref:
        return 0.0 if not hyp else 1.0
    return levenshtein(ref, hyp) / len(ref)


def transliteration_invariant_wer(reference: str, hypothesis: str) -> float:
    ref = normalize_text(back_transliterate(reference, only_hindi_tokens=True)).split()
    hyp = normalize_text(back_transliterate(hypothesis, only_hindi_tokens=True)).split()
    if not ref:
        return 0.0 if not hyp else 1.0
    return levenshtein(ref, hyp) / len(ref)


def error_rates(reference: str, hypothesis: str) -> ErrorRates:
    ref_words = normalize_text(reference).split()
    hyp_words = normalize_text(hypothesis).split()
    return ErrorRates(
        wer=round(word_error_rate(reference, hypothesis), 4),
        cer=round(character_error_rate(reference, hypothesis), 4),
        translit_invariant_wer=round(transliteration_invariant_wer(reference, hypothesis), 4),
        ref_words=len(ref_words),
        hyp_words=len(hyp_words),
    )


def normalized_disagreement(hyp_a: str, hyp_b: str) -> float:
    """Symmetric normalised distance between two hypotheses of the same audio.

    Symmetric because neither is a reference — normalising by the longer side
    keeps the value in [0,1] and stops a truncated hypothesis from scoring
    artificially well."""
    a = normalize_text(hyp_a).split()
    b = normalize_text(hyp_b).split()
    denom = max(len(a), len(b))
    if denom == 0:
        return 0.0
    return round(levenshtein(a, b) / denom, 4)
