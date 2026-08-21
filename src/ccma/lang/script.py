"""Unicode-script detection.

Script is the one language signal that is *free and exact*. Any token containing
Devanagari is Hindi with certainty — no model needed. The hard problem is the
Latin-script residue, which is where everything downstream spends its effort."""

from __future__ import annotations

import unicodedata
from enum import Enum

import regex as re

DEVANAGARI = re.compile(r"[\p{Devanagari}]")
LATIN = re.compile(r"[\p{Latin}]")
DIGIT = re.compile(r"[\p{Nd}]")

TOKEN_RE = re.compile(r"[\p{L}\p{Nd}][\p{L}\p{Nd}'’\-]*", re.UNICODE)


class Script(str, Enum):
    DEVANAGARI = "Deva"
    LATIN = "Latn"
    MIXED = "Mixed"
    NUMERIC = "Nume"
    OTHER = "Zyyy"


def tokenize(text: str) -> list[str]:
    """Word tokenizer that keeps Devanagari clusters and Latin words intact and
    drops punctuation. Intentionally script-agnostic — a Hinglish utterance is a
    single token stream, not two."""
    return TOKEN_RE.findall(text or "")


def token_script(token: str) -> Script:
    has_deva = bool(DEVANAGARI.search(token))
    has_latin = bool(LATIN.search(token))
    if has_deva and has_latin:
        return Script.MIXED
    if has_deva:
        return Script.DEVANAGARI
    if has_latin:
        return Script.LATIN
    if DIGIT.search(token):
        return Script.NUMERIC
    return Script.OTHER


def script_profile(text: str) -> dict[Script, float]:
    """Share of tokens by script. Cheap enough to run on every utterance."""
    tokens = tokenize(text)
    if not tokens:
        return {}
    counts: dict[Script, int] = {}
    for tok in tokens:
        s = token_script(tok)
        counts[s] = counts.get(s, 0) + 1
    total = len(tokens)
    return {k: v / total for k, v in counts.items()}


def strip_diacritics(text: str) -> str:
    """Fold ISO-15919 style romanisation (`nahī̃`, `kyā`) onto plain ASCII so a
    single lexicon entry matches every diacritic convention agents type."""
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def normalize_devanagari(text: str) -> str:
    """NFC + nukta folding + chandrabindu→anusvara.

    Hindi has several visually identical encodings (क़ as one codepoint vs.
    क + nukta). Without folding, exact-match phrase banks silently miss."""
    text = unicodedata.normalize("NFC", text)
    nukta_map = {
        "ऩ": "न",  # ऩ -> न
        "ऱ": "र",  # ऱ -> र
        "ऴ": "ळ",  # ऴ -> ळ
        "क़": "क",  # क़ -> क
        "ख़": "ख",  # ख़ -> ख
        "ग़": "ग",  # ग़ -> ग
        "ज़": "ज",  # ज़ -> ज
        "ड़": "ड",  # ड़ -> ड
        "ढ़": "ढ",  # ढ़ -> ढ
        "फ़": "फ",  # फ़ -> फ
        "य़": "य",  # य़ -> य
        "ँ": "ं",  # chandrabindu -> anusvara
    }
    return "".join(nukta_map.get(ch, ch) for ch in text).replace("़", "")
