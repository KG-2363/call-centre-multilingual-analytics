"""Code-switching measurement.

We report three complementary numbers because each answers a different question
and any one alone is misleading:

* **CMI** (Code-Mixing Index, Das & Gambäck 2014) — *how much* mixing.
  A call that is 50/50 Hindi/English scores high.
* **M-index** (Barnett et al. 2000) — how *evenly* the languages are distributed,
  normalised for the number of languages present.
* **Switch-point fraction** (Guzmán et al. 2017) — how *often* the speaker
  alternates. This is the one that predicts ASR difficulty, because switch points
  are where acoustic and language models disagree most.

An utterance can have high CMI and low SPF (one long English clause inside a
Hindi turn — easy) or moderate CMI and high SPF (word-level alternation — hard).
Only the pair distinguishes them, which is why per-language quality monitoring
(docs/03) buckets on SPF, not CMI.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..models.enums import LanguageCode

_HINDI_LANGS = {LanguageCode.HI, LanguageCode.HI_LATN}


def _collapse(lang: LanguageCode) -> str | None:
    """Hindi-in-Devanagari and Hindi-in-Latin are one language for mixing
    purposes. Non-lexical tokens are excluded entirely."""
    if lang in _HINDI_LANGS:
        return "hi"
    if lang == LanguageCode.EN:
        return "en"
    return None


@dataclass
class CodeSwitchMetrics:
    cmi: float
    m_index: float
    switch_points: int
    switch_point_fraction: float
    language_counts: dict[str, int]
    n_tokens: int
    is_code_switched: bool


def compute(token_langs: list[LanguageCode]) -> CodeSwitchMetrics:
    seq = [c for c in (_collapse(t) for t in token_langs) if c is not None]
    n = len(seq)

    if n == 0:
        return CodeSwitchMetrics(0.0, 0.0, 0, 0.0, {}, 0, False)

    counts: dict[str, int] = {}
    for lang in seq:
        counts[lang] = counts.get(lang, 0) + 1

    max_count = max(counts.values())
    n_langs = len(counts)

    # CMI: fraction of tokens not belonging to the dominant language.
    # Single-language utterances score 0 by definition.
    cmi = 0.0 if n_langs <= 1 else (n - max_count) / n

    # M-index: 1 - sum(p_i^2) normalised by (1 - 1/k). 0 = monolingual,
    # 1 = perfectly even split across the k languages present.
    if n_langs <= 1:
        m_index = 0.0
    else:
        sum_sq = sum((c / n) ** 2 for c in counts.values())
        m_index = (1.0 - sum_sq) / (1.0 - 1.0 / n_langs)

    switch_points = sum(1 for a, b in zip(seq, seq[1:]) if a != b)
    spf = switch_points / (n - 1) if n > 1 else 0.0

    return CodeSwitchMetrics(
        cmi=round(cmi, 4),
        m_index=round(m_index, 4),
        switch_points=switch_points,
        switch_point_fraction=round(spf, 4),
        language_counts=counts,
        n_tokens=n,
        is_code_switched=n_langs > 1 and switch_points > 0,
    )


def segment_by_language(
    tokens: list[str], token_langs: list[LanguageCode]
) -> list[tuple[str, str]]:
    """Split a token stream into maximal same-language spans.

    Used by the dashboard's language-switch ribbon, and by language-aware intent
    classification: each span can be routed to the model that handles it best
    instead of forcing one model to cope with the whole mixed utterance."""
    spans: list[tuple[str, str]] = []
    current_lang: str | None = None
    buffer: list[str] = []

    for tok, lang in zip(tokens, token_langs):
        collapsed = _collapse(lang)
        if collapsed is None:
            if buffer:
                buffer.append(tok)
            continue
        if collapsed != current_lang:
            if buffer and current_lang:
                spans.append((current_lang, " ".join(buffer)))
            current_lang, buffer = collapsed, [tok]
        else:
            buffer.append(tok)

    if buffer and current_lang:
        spans.append((current_lang, " ".join(buffer)))
    return spans


def call_level_profile(
    utterance_metrics: list[CodeSwitchMetrics],
) -> tuple[float, float, float]:
    """Token-weighted roll-up of per-utterance metrics to the call.

    Token-weighted, not mean-of-means: a 2-word "ok theek hai" must not carry the
    same weight as a 40-word explanation when describing how mixed a call was."""
    total_tokens = sum(m.n_tokens for m in utterance_metrics)
    if total_tokens == 0:
        return 0.0, 0.0, 0.0

    cmi = sum(m.cmi * m.n_tokens for m in utterance_metrics) / total_tokens
    m_idx = sum(m.m_index * m.n_tokens for m in utterance_metrics) / total_tokens
    spf = sum(m.switch_point_fraction * m.n_tokens for m in utterance_metrics) / total_tokens
    return round(cmi, 4), round(m_idx, 4), round(spf, 4)
