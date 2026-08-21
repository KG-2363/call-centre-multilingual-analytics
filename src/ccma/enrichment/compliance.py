"""Hybrid compliance detection.

Compliance is the one output an auditor or a regulator may act on, so it gets a
different treatment from every other enrichment field: **deterministic first,
LLM second, and disagreement surfaced rather than resolved silently.**

Three detectors, combined:

1. **Multilingual phrase bank** (this module) — normalised, script-folded,
   transliteration-aware regex over both the Latin and Devanagari projections.
   Precise, auditable, zero cost, and it works identically for `yeh call record
   ki ja rahi hai` and `यह कॉल रिकॉर्ड की जा रही है`.
2. **Embedding similarity** (optional, via Ollama `bge-m3`) — catches paraphrases
   the phrase bank misses. Multilingual embeddings are essential here: an English
   query must retrieve a Hindi disclosure.
3. **LLM adjudication** — invoked *only* where 1 and 2 disagree or both are
   borderline. This is the cost-control lever: on our corpus the deterministic
   layer settles the large majority of rule/call pairs outright.

Where the deterministic detector says PRESENT and the LLM says ABSENT (or vice
versa), we emit the result with `detector_disagreement=True` and route it to
human review. An automated compliance *failure* on a disputed detection is worse
than no detection at all — it destroys trust in the whole scorecard.
"""

from __future__ import annotations

from dataclasses import dataclass

import regex as re

from ..lang.lid import normalize_text
from ..lang.script import normalize_devanagari
from ..lang.translit import back_transliterate
from ..models.enums import ComplianceOutcome, ComplianceRule

# Each rule carries patterns in three surface forms because that is how the text
# actually arrives: English, Devanagari Hindi, and romanised Hindi/Hinglish.
PHRASE_BANK: dict[ComplianceRule, dict[str, list[str]]] = {
    ComplianceRule.RECORDING_DISCLOSURE: {
        "en": [
            r"\bcall (is|may be|is being) (recorded|monitored)\b",
            r"\brecorded for (quality|training|verification)\b",
            r"\bthis (call|conversation) is being recorded\b",
            r"\bmonitored for quality\b",
        ],
        "hi": [
            r"कॉल\s+रिकॉर्ड\s+की\s+जा\s+रही",
            r"रिकॉर्ड\s+किया\s+जा\s+रहा",
            r"गुणवत्ता\s+.{0,20}(के\s+लिए|हेतु).{0,20}रिकॉर्ड",
        ],
        "hi_latn": [
            r"\bcall record ki ja rahi\b",
            r"\bkiya ja raha hai record\b",
            r"\byeh call record\b",
            r"\bquality .{0,25}record\b",
        ],
    },
    ComplianceRule.IDENTITY_VERIFICATION: {
        "en": [
            r"\b(may i have|can you (confirm|provide)|could you confirm) your\b",
            r"\b(verify|verification) (your|purposes?)\b",
            r"\bdate of birth\b",
            r"\bregistered (mobile )?(number|email)\b",
        ],
        "hi": [
            r"सत्यापन\s+के\s+लिए",
            r"पंजीकृत\s+(मोबाइल\s+)?(नंबर|संख्या)",
            r"जन्म\s+तिथि",
            r"अपना\s+(नाम|पता)\s+बता",
        ],
        "hi_latn": [
            r"\bverification ke liye\b",
            r"\bregistered (mobile )?number bata\b",
            r"\bjanm tithi\b",
            r"\bapna (naam|pata|number) bata\b",
            r"\bdate of birth bata\b",
        ],
    },
    ComplianceRule.DPDP_CONSENT: {
        "en": [
            r"\b(do you|may we have your) consent\b",
            r"\bpermission to (share|use|store) your (data|information|details)\b",
            r"\bshare .{0,30}with .{0,25}team\b.{0,40}\b(consent|permission|agree)\b",
        ],
        "hi": [
            r"आपकी\s+सहमति",
            r"जानकारी\s+.{0,25}साझा\s+कर",
            r"अनुमति\s+है",
        ],
        "hi_latn": [
            r"\bsahmati hai\b",
            r"\bshare kar sakte hain\b.{0,30}\?",
            r"\bijazat\b",
            r"\banumati\b",
        ],
    },
    ComplianceRule.NO_FULL_CARD_READBACK: {
        # A *prohibition*: matching it is a VIOLATION, not compliance.
        "en": [
            r"\b(\d[ -]?){13,16}\b",
            r"\bfull card number\b",
            r"\bcvv\b",
            r"\b(read|tell) me your (complete|full) (card|account) number\b",
        ],
        "hi": [r"पूरा\s+कार्ड\s+नंबर", r"सीवीवी"],
        "hi_latn": [r"\bpoora card number bata\b", r"\bcvv bata\b"],
    },
    ComplianceRule.MIS_SELLING_DISCLAIMER: {
        "en": [
            r"\bno hidden (charges?|costs?|fees?)\b",
            r"\b(inclusive of|including) (gst|taxes)\b",
            r"\byou can (cancel|revert|change) (any ?time|at any time)\b",
            r"\bterms and conditions apply\b",
        ],
        "hi": [
            r"कोई\s+छुपा\s+(हुआ\s+)?(शुल्क|खर्च)",
            r"कर\s+सहित",
            r"कभी\s+भी\s+बदल\s+सकते",
        ],
        "hi_latn": [
            r"\bkoi hidden charge nahi\b",
            r"\bgst inclusive\b",
            r"\bkabhi bhi change kar sakte\b",
            r"\bkoi penalty nahi\b",
        ],
    },
    ComplianceRule.GRIEVANCE_REDRESSAL_INFO: {
        "en": [
            r"\b(trai|consumer forum|ombudsman|grievance) \b",
            r"\bcomplaint (number|reference|id) is\b",
            r"\bescalation (matrix|reference)\b",
        ],
        "hi": [
            r"शिकायत\s+(संख्या|क्रमांक)",
            r"ट्राई",
            r"शिकायत\s+पोर्टल",
            r"उपभोक्ता\s+फ़?ोरम",
        ],
        "hi_latn": [
            r"\bshikayat (number|sankhya)\b",
            r"\btrai (ke|ki|portal)\b",
            r"\bcomplaint number\b",
            r"\bconsumer forum\b",
        ],
    },
    ComplianceRule.PROFESSIONAL_CLOSING: {
        "en": [
            r"\bthank you for calling\b",
            r"\bhave a (great|good|nice|pleasant) day\b",
            r"\banything else i can help\b",
            r"\bmy pleasure\b",
        ],
        "hi": [
            r"धन्यवाद",
            r"आपका\s+दिन\s+(शुभ|मंगलमय)",
        ],
        "hi_latn": [
            r"\bdhanyavaad\b", r"\bdhanyavad\b", r"\bshukriya\b",
            r"\baapka din shubh\b",
            r"\bhave a (great|good) day\b",
        ],
    },
    ComplianceRule.DND_ACKNOWLEDGEMENT: {
        "en": [r"\bdo not disturb\b", r"\bdnd (status|registration|activated?)\b"],
        "hi": [r"डी\s?एन\s?डी", r"परेशान\s+न\s+कर"],
        "hi_latn": [r"\bdnd (chalu|activate|register)\b", r"\bdnd status\b"],
    },
}

# Rules that only apply when the call touched the relevant topic. Marking an
# inapplicable rule ABSENT is the fastest way to make a compliance dashboard lie.
APPLICABILITY_TRIGGERS: dict[ComplianceRule, list[str]] = {
    ComplianceRule.DPDP_CONSENT: [
        r"\bshare\b", r"\bthird party\b", r"\bpartner\b", r"\bteam ke saath\b",
        r"साझा", r"\bdocument\b", r"\bkyc\b", r"\bfraud\b",
    ],
    ComplianceRule.NO_FULL_CARD_READBACK: [
        r"\bcard\b", r"\bकार्ड\b", r"\bdebit\b", r"\bcredit\b", r"\bbank\b", r"\bcvv\b",
    ],
    ComplianceRule.MIS_SELLING_DISCLAIMER: [
        r"\bplan\b", r"\boffer\b", r"\bupgrade\b", r"\bdiscount\b", r"\bactivate\b",
        r"\bप्लान\b", r"\bछूट\b",
    ],
    ComplianceRule.DND_ACKNOWLEDGEMENT: [
        r"\bdnd\b", r"डी\s?एन\s?डी", r"\bpromotional\b", r"\bspam\b", r"\bप्रचार\b",
    ],
    ComplianceRule.GRIEVANCE_REDRESSAL_INFO: [
        r"\bcomplaint\b", r"\bshikayat\b", r"शिकायत", r"\bescalat\w+\b", r"\btrai\b",
    ],
}

_PROHIBITION_RULES = {ComplianceRule.NO_FULL_CARD_READBACK}


@dataclass
class PhraseMatch:
    rule: ComplianceRule
    outcome: ComplianceOutcome
    confidence: float
    matched_text: str | None
    matched_language: str | None
    turn_index: int | None


def _searchable_forms(text: str) -> dict[str, str]:
    """Every utterance is searched in three projections so a phrase bank entry
    written in one script matches content written in another."""
    normalized = normalize_text(text)
    return {
        "raw": normalized,
        "deva": normalize_devanagari(back_transliterate(normalized, only_hindi_tokens=True)),
        "latin": normalized,
    }


def _rule_applies(rule: ComplianceRule, full_text: str) -> bool:
    triggers = APPLICABILITY_TRIGGERS.get(rule)
    if triggers is None:
        return True
    return any(re.search(p, full_text, re.IGNORECASE) for p in triggers)


def detect_rule(
    rule: ComplianceRule, turns: list[tuple[int, str, str]]
) -> PhraseMatch:
    """Scan agent turns for one rule. `turns` is (turn_index, role, text)."""
    full_text = " ".join(t[2] for t in turns)
    agent_turns = [(i, txt) for i, role, txt in turns if role == "AGENT"]

    if not _rule_applies(rule, full_text):
        return PhraseMatch(rule, ComplianceOutcome.NOT_APPLICABLE, 0.9, None, None, None)

    patterns = PHRASE_BANK[rule]
    for turn_index, text in agent_turns:
        forms = _searchable_forms(text)
        for lang, pattern_list in patterns.items():
            haystack = forms["deva"] if lang == "hi" else forms["raw"]
            for pattern in pattern_list:
                match = re.search(pattern, haystack, re.IGNORECASE)
                if match:
                    outcome = (
                        ComplianceOutcome.VIOLATED
                        if rule in _PROHIBITION_RULES
                        else ComplianceOutcome.PRESENT
                    )
                    return PhraseMatch(
                        rule, outcome, 0.93, match.group(0)[:400], lang, turn_index
                    )

    # A prohibition that never matched is satisfied, not absent.
    if rule in _PROHIBITION_RULES:
        return PhraseMatch(rule, ComplianceOutcome.PRESENT, 0.85, None, None, None)
    return PhraseMatch(rule, ComplianceOutcome.ABSENT, 0.80, None, None, None)


def detect_all(turns: list[tuple[int, str, str]]) -> list[PhraseMatch]:
    return [detect_rule(rule, turns) for rule in ComplianceRule]


ABSTENTION_CONFIDENCE = 0.5


def reconcile(
    deterministic: list[PhraseMatch], llm_checks: list[dict]
) -> list[dict]:
    """Merge the two detectors.

    Policy: the deterministic detector wins on PRESENT (it matched literal text
    an auditor can see), the LLM wins on NOT_APPLICABLE (it understands context),
    and every genuine conflict is flagged for review rather than auto-resolved.

    A *low-confidence* LLM assessment is treated as an **abstention**, not a
    disagreement. This distinction matters operationally: `detector_disagreement`
    sizes the human review queue, and counting every hedge as a conflict floods
    that queue with cases where the LLM simply had no opinion — which is exactly
    how a review queue gets ignored."""
    llm_by_rule = {c.get("rule"): c for c in llm_checks if c.get("rule")}
    out: list[dict] = []

    for det in deterministic:
        llm = llm_by_rule.get(det.rule.value)
        if llm is not None and float(llm.get("confidence", 0.0)) < ABSTENTION_CONFIDENCE:
            llm = None

        if llm is None:
            out.append(
                {
                    "rule": det.rule.value,
                    "outcome": det.outcome.value,
                    "confidence": det.confidence,
                    "matched_text": det.matched_text,
                    "matched_language": det.matched_language,
                    "evidence_turn_indices": [det.turn_index] if det.turn_index is not None else [],
                    "detector_disagreement": False,
                }
            )
            continue

        llm_outcome = llm.get("outcome")
        agree = llm_outcome == det.outcome.value

        if agree:
            outcome, confidence = det.outcome.value, min(0.99, det.confidence + 0.06)
        elif det.outcome == ComplianceOutcome.PRESENT:
            outcome, confidence = det.outcome.value, 0.75
        elif llm_outcome == ComplianceOutcome.NOT_APPLICABLE.value:
            outcome, confidence = llm_outcome, 0.70
        else:
            # Deterministic says ABSENT, LLM claims PRESENT -> trust the LLM only
            # far enough to mark it PARTIAL and send it to review.
            outcome, confidence = ComplianceOutcome.PARTIAL.value, 0.55

        out.append(
            {
                "rule": det.rule.value,
                "outcome": outcome,
                "confidence": round(confidence, 4),
                "matched_text": det.matched_text or llm.get("matched_text"),
                "matched_language": det.matched_language or llm.get("matched_language"),
                "evidence_turn_indices": (
                    [det.turn_index] if det.turn_index is not None
                    else llm.get("evidence_turn_indices", [])
                ),
                "detector_disagreement": not agree,
            }
        )
    return out
