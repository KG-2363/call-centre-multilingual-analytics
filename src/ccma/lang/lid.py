"""Language identification: token-level first, utterance-level by aggregation.

Design: a **cascade of decreasing certainty**, so each token is resolved by the
cheapest sufficient signal and only genuinely ambiguous tokens reach the
expensive step.

  1. Devanagari script          -> hi          (certain, free)
  2. Curated romanised lexicon  -> hi_latn     (high confidence, free)
  3. English stopword list      -> en          (high confidence, free)
  4. Homograph                  -> neighbour context voting
  5. OOV Latin                  -> orthographic scorer (graded confidence)

Utterance-level labels are then *fused* with the decoder's own LID posterior
rather than replacing it, because Whisper's prior is genuinely informative on
long, clean, monolingual segments — exactly where the lexicon is least needed.
The fusion weight is a function of the segment's token count, so short segments
lean on the lexicon and long segments lean on the neural prior.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import regex as re

from ..models.enums import LanguageCode
from .lexicon import is_english_word, is_homograph, is_romanized_hindi, normalize_key
from .script import Script, normalize_devanagari, token_script, tokenize
from .translit import back_transliterate, romanization_score

# Confidence assigned by each cascade stage.
_CONF_SCRIPT = 0.99
_CONF_LEXICON = 0.92
_CONF_STOPWORD = 0.90
_CONF_CONTEXT = 0.70
_CONF_ORTHOGRAPHIC_MAX = 0.80

_DISFLUENCY_RE = re.compile(
    r"\b(uh+|um+|hmm+|haan+|arre+|matlab|you know|i mean|actually)\b", re.IGNORECASE
)
_REPEAT_RE = re.compile(r"\b(\p{L}+)(\s+\1\b)+", re.IGNORECASE)


@dataclass
class TokenTag:
    token: str
    lang: LanguageCode
    confidence: float
    is_transliterated: bool = False
    stage: str = ""


@dataclass
class LidResult:
    lang_primary: LanguageCode
    confidence: float
    distribution: dict[str, float]
    tokens: list[TokenTag] = field(default_factory=list)


def _tag_token_context_free(token: str) -> TokenTag | None:
    """Stages 1-3 and 5. Returns None only for homographs (stage 4)."""
    script = token_script(token)

    if script in (Script.DEVANAGARI, Script.MIXED):
        return TokenTag(token, LanguageCode.HI, _CONF_SCRIPT, False, "script")
    if script in (Script.NUMERIC, Script.OTHER):
        return TokenTag(token, LanguageCode.UNKNOWN, 0.5, False, "non_lexical")

    if is_homograph(token):
        return None

    if is_romanized_hindi(token):
        return TokenTag(token, LanguageCode.HI_LATN, _CONF_LEXICON, True, "lexicon")
    if is_english_word(token):
        return TokenTag(token, LanguageCode.EN, _CONF_STOPWORD, False, "stopword")

    score = romanization_score(token)
    if score >= 0.5:
        return TokenTag(
            token,
            LanguageCode.HI_LATN,
            min(_CONF_ORTHOGRAPHIC_MAX, 0.5 + score * 0.35),
            True,
            "orthographic",
        )
    return TokenTag(
        token,
        LanguageCode.EN,
        min(_CONF_ORTHOGRAPHIC_MAX, 0.5 + (1.0 - score) * 0.30),
        False,
        "orthographic",
    )


def _resolve_homographs(tokens: list[str], tags: list[TokenTag | None]) -> list[TokenTag]:
    """Stage 4: a homograph takes the language of its resolved neighbours.

    'to' in "to main aapko bata deta hun" is Hindi (तो); 'to' in "I want to
    check my bill" is English. A +/-2 token window settles almost all of them."""
    resolved: list[TokenTag] = []
    for idx, tag in enumerate(tags):
        if tag is not None:
            resolved.append(tag)
            continue

        window = [
            t
            for j, t in enumerate(tags)
            if t is not None and j != idx and abs(j - idx) <= 2 and t.lang != LanguageCode.UNKNOWN
        ]
        votes: dict[LanguageCode, float] = {}
        for neighbour in window:
            weight = neighbour.confidence
            lang = (
                LanguageCode.HI_LATN
                if neighbour.lang in (LanguageCode.HI, LanguageCode.HI_LATN)
                else LanguageCode.EN
            )
            votes[lang] = votes.get(lang, 0.0) + weight

        if votes:
            best = max(votes, key=lambda k: votes[k])
            total = sum(votes.values())
            conf = _CONF_CONTEXT * (votes[best] / total) + 0.25
        else:
            best, conf = LanguageCode.EN, 0.45

        resolved.append(
            TokenTag(
                tokens[idx],
                best,
                round(min(0.88, conf), 4),
                best == LanguageCode.HI_LATN,
                "context",
            )
        )
    return resolved


def tag_tokens(text: str) -> list[TokenTag]:
    tokens = tokenize(text)
    if not tokens:
        return []
    provisional = [_tag_token_context_free(t) for t in tokens]
    return _resolve_homographs(tokens, provisional)


def _aggregate(tags: list[TokenTag]) -> dict[str, float]:
    """Confidence-weighted language shares over lexical tokens only."""
    lexical = [t for t in tags if t.lang != LanguageCode.UNKNOWN]
    if not lexical:
        return {}
    weights: dict[str, float] = {}
    for t in lexical:
        weights[t.lang.value] = weights.get(t.lang.value, 0.0) + t.confidence
    total = sum(weights.values())
    return {k: round(v / total, 4) for k, v in weights.items()}


def _fuse_with_prior(
    dist: dict[str, float], prior_lang: str | None, prior_prob: float | None, n_tokens: int
) -> dict[str, float]:
    """Blend lexicon evidence with the decoder's LID posterior.

    Weight on the neural prior grows with token count: on a 3-word segment the
    decoder is guessing, on a 40-word segment it is usually right."""
    if not prior_lang or prior_prob is None or not dist:
        return dist

    mapped = {"hi": LanguageCode.HI.value, "en": LanguageCode.EN.value}.get(prior_lang)
    if mapped is None:
        return dist

    # Whisper reports Devanagari-transcribed Hindi as "hi"; if our tokens are
    # Latin, the equivalent bucket is hi_latn.
    if mapped == LanguageCode.HI.value and dist.get(LanguageCode.HI_LATN.value, 0) > dist.get(
        LanguageCode.HI.value, 0
    ):
        mapped = LanguageCode.HI_LATN.value

    prior_weight = min(0.45, 0.10 + 0.012 * n_tokens) * prior_prob
    fused = {k: v * (1.0 - prior_weight) for k, v in dist.items()}
    fused[mapped] = fused.get(mapped, 0.0) + prior_weight
    total = sum(fused.values()) or 1.0
    return {k: round(v / total, 4) for k, v in fused.items()}


def identify(
    text: str,
    prior_lang: str | None = None,
    prior_prob: float | None = None,
    mixed_threshold: float = 0.25,
) -> LidResult:
    """Utterance-level LID.

    `mixed_threshold`: when the minority language holds at least this share of
    confidence mass, we label the utterance `mixed` rather than forcing a winner.
    Forcing a winner is how Hinglish gets silently analysed as monolingual."""
    tags = tag_tokens(text)
    if not tags:
        return LidResult(LanguageCode.UNKNOWN, 0.0, {}, [])

    dist = _aggregate(tags)
    if not dist:
        return LidResult(LanguageCode.UNKNOWN, 0.0, {}, tags)

    n_lexical = sum(1 for t in tags if t.lang != LanguageCode.UNKNOWN)
    dist = _fuse_with_prior(dist, prior_lang, prior_prob, n_lexical)

    # Collapse hi + hi_latn when deciding "is this mixed?" — they are the same
    # language in two scripts, not two languages.
    hindi_mass = dist.get(LanguageCode.HI.value, 0.0) + dist.get(LanguageCode.HI_LATN.value, 0.0)
    english_mass = dist.get(LanguageCode.EN.value, 0.0)

    if min(hindi_mass, english_mass) >= mixed_threshold and n_lexical >= 4:
        primary = LanguageCode.MIXED
        confidence = round(min(hindi_mass, english_mass) / max(hindi_mass, english_mass), 4)
    else:
        primary_key = max(dist, key=lambda k: dist[k])
        primary = LanguageCode(primary_key)
        confidence = round(dist[primary_key], 4)

    return LidResult(primary, confidence, dist, tags)


def normalize_text(text: str) -> str:
    """Silver-layer text normalisation.

    Kept conservative on purpose: we lower-case, unify Devanagari encoding,
    collapse stutters and whitespace, and strip filler. We do *not* stem, expand
    contractions or spell-correct, because those steps destroy the evidence a
    compliance auditor needs to see the exact words spoken."""
    if not text:
        return ""
    out = normalize_devanagari(text)
    out = _REPEAT_RE.sub(r"\1", out)
    out = _DISFLUENCY_RE.sub(" ", out)
    out = re.sub(r"[^\p{L}\p{Nd}\s'’%.,?-]", " ", out)
    out = re.sub(r"\s+", " ", out).strip().lower()
    return out


def devanagari_projection(text: str, lang: LanguageCode) -> str | None:
    """Single canonical representation for Hindi content regardless of script.
    Returns None for pure English (nothing to project)."""
    if lang == LanguageCode.EN:
        return None
    if lang == LanguageCode.HI:
        return normalize_devanagari(text)
    if lang in (LanguageCode.HI_LATN, LanguageCode.MIXED):
        return normalize_devanagari(back_transliterate(text, only_hindi_tokens=True))
    return None


def token_key(token: str) -> str:
    return normalize_key(token)
