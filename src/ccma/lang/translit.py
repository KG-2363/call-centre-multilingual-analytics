"""Handling of transliterated (romanised) Hindi.

Two jobs:

1. `romanization_score` — decide whether an *out-of-lexicon* Latin token is
   romanised Hindi. The lexicon covers the head of the distribution; this covers
   the tail (names, inflections, ASR misspellings) using orthographic evidence.

2. `back_transliterate` — map romanised Hindi to Devanagari so that a single
   downstream representation exists. This matters more than it looks: without it,
   "paisa kat gaya" and "पैसा कट गया" are two unrelated strings to every
   embedding model, phrase bank and GROUP BY in the warehouse.

Production note: for the real system I would replace the rule-based
transliterator with AI4Bharat IndicXlit (a seq2seq transliteration model trained
on Dakshina), keeping this implementation as the zero-dependency fallback and as
the deterministic path for unit tests. The rules below get the common cases right
and degrade gracefully; IndicXlit gets the long tail right at the cost of a model
load. See docs/03-multilingual-strategy.md.
"""

from __future__ import annotations

import regex as re

from .lexicon import normalize_key, to_devanagari

# Orthographic signatures of Devanagari-to-Latin romanisation.
_HINDI_SUFFIXES = (
    "iye", "iyega", "ega", "egi", "enge", "ungi", "unga", "aoge", "ayenge",
    "kar", "ker", "ne", "na", "ni", "nu", "ke", "ka", "ki", "ko", "se", "me",
    "ta", "ti", "te", "ya", "ye", "yi", "wa", "wi", "on", "ain", "aan", "hai",
    "hun", "ho", "ga", "gi", "ge", "raha", "rahi", "rahe", "gaya", "gayi",
)
_HINDI_DIGRAPHS = ("kh", "gh", "chh", "jh", "th", "dh", "ph", "bh", "sh", "ny")
_HINDI_VOWEL_RUNS = ("aa", "ee", "oo", "ai", "au", "ii", "uu")
# Letter sequences that are common in English and essentially absent in
# romanised Hindi — strong negative evidence.
_ENGLISH_MARKERS = ("tion", "sion", "ing", "ness", "ment", "able", "ible",
                    "ous", "ph", "qu", "wh", "ck", "ex", "sc")

_VOWELS = set("aeiou")


def romanization_score(token: str) -> float:
    """0.0 = looks English, 1.0 = looks like romanised Hindi.

    Deliberately a transparent additive scorer rather than a classifier: it is
    inspectable, needs no training data, and its failure modes are obvious in
    review. It is fused with (not trusted over) neural LID in lid.py."""
    tok = normalize_key(token)
    if len(tok) < 2 or not tok.isalpha():
        return 0.0

    score = 0.0

    matched_suffix = ""
    for suf in _HINDI_SUFFIXES:
        if tok.endswith(suf) and len(tok) > len(suf):
            score += 0.28
            matched_suffix = suf
            break

    if any(d in tok for d in _HINDI_DIGRAPHS):
        score += 0.18
    if any(v in tok for v in _HINDI_VOWEL_RUNS):
        score += 0.18

    # Hindi romanisation is highly syllabic: CV.CV.CV. English tolerates far
    # heavier clusters. A high vowel ratio with no 3-consonant cluster is a tell.
    vowel_ratio = sum(1 for c in tok if c in _VOWELS) / len(tok)
    if 0.33 <= vowel_ratio <= 0.60:
        score += 0.16
    if not re.search(r"[^aeiou]{3}", tok):
        score += 0.10

    for marker in _ENGLISH_MARKERS:
        if marker in tok:
            score -= 0.45
            break

    # Silent-e is an English pattern ("service", "charge"), but a long Hindi verb
    # ending such as -enge or -iye also ends consonant + e; don't penalise those.
    if (tok.endswith("e") and len(tok) > 4 and tok[-2] not in _VOWELS
            and len(matched_suffix) < 3):
        score -= 0.15

    return max(0.0, min(1.0, score))


# --- rule-based back-transliteration ---------------------------------------
# Ordered longest-first so digraphs win over their component letters.
_CONSONANTS: list[tuple[str, str]] = [
    ("chh", "छ"), ("shh", "श"),
    ("kh", "ख"), ("gh", "घ"), ("ch", "च"), ("jh", "झ"), ("th", "थ"),
    ("dh", "ध"), ("ph", "फ"), ("bh", "भ"), ("sh", "श"), ("ng", "ं"),
    ("k", "क"), ("g", "ग"), ("j", "ज"), ("t", "त"), ("d", "द"),
    ("n", "न"), ("p", "प"), ("b", "ब"), ("m", "म"), ("y", "य"),
    ("r", "र"), ("l", "ल"), ("v", "व"), ("w", "व"), ("s", "स"),
    ("h", "ह"), ("f", "फ़"), ("z", "ज़"), ("c", "क"), ("x", "क्स"),
    ("q", "क़"),
]
# (roman, independent vowel, matra)
_VOWEL_TABLE: list[tuple[str, str, str]] = [
    ("aa", "आ", "ा"), ("ai", "ऐ", "ै"), ("au", "औ", "ौ"),
    ("ee", "ई", "ी"), ("ii", "ई", "ी"), ("oo", "ऊ", "ू"),
    ("uu", "ऊ", "ू"), ("ou", "औ", "ौ"), ("ey", "ए", "े"),
    ("a", "अ", ""), ("i", "इ", "ि"), ("u", "उ", "ु"),
    ("e", "ए", "े"), ("o", "ओ", "ो"),
]
_VIRAMA = "्"


def _transliterate_oov(token: str) -> str:
    """Best-effort syllabic mapping for tokens the lexicon has never seen."""
    tok = normalize_key(token)
    out: list[str] = []
    i = 0
    at_word_start = True

    while i < len(tok):
        # Try a vowel first when we're at a position expecting one.
        matched_vowel = next(
            ((r, indep, matra) for r, indep, matra in _VOWEL_TABLE if tok.startswith(r, i)),
            None,
        )
        matched_cons = next(((r, d) for r, d in _CONSONANTS if tok.startswith(r, i)), None)

        if matched_cons and (not matched_vowel or not at_word_start):
            roman, deva = matched_cons
            out.append(deva)
            i += len(roman)
            at_word_start = False

            nxt = next(
                ((r, indep, matra) for r, indep, matra in _VOWEL_TABLE if tok.startswith(r, i)),
                None,
            )
            if nxt:
                _, _, matra = nxt
                out.append(matra)
                i += len(nxt[0])
            elif i < len(tok):
                out.append(_VIRAMA)  # consonant cluster
        elif matched_vowel:
            roman, indep, _ = matched_vowel
            out.append(indep)
            i += len(roman)
            at_word_start = False
        else:
            i += 1

    return "".join(out)


def back_transliterate_token(token: str) -> str:
    """Lexicon first (correct), orthographic rules second (approximate)."""
    hit = to_devanagari(token)
    return hit if hit else _transliterate_oov(token)


def back_transliterate(text: str, only_hindi_tokens: bool = True) -> str:
    """Convert romanised Hindi in `text` to Devanagari, leaving English alone.

    `only_hindi_tokens=False` forces every Latin token through the mapper, which
    is what you want when you already know the utterance is monolingual Hindi."""
    from .script import Script, token_script

    def _convert(match: re.Match) -> str:
        tok = match.group(0)
        if token_script(tok) != Script.LATIN:
            return tok
        if only_hindi_tokens:
            from .lexicon import is_english_word, is_romanized_hindi

            if is_english_word(tok):
                return tok
            if not is_romanized_hindi(tok) and romanization_score(tok) < 0.5:
                return tok
        return back_transliterate_token(tok)

    return re.sub(r"[\p{L}\p{Nd}][\p{L}\p{M}\p{Nd}'’\-]*", _convert, text or "")
