"""Offline LLM backend.

Not a stub that returns fixed JSON — it parses the rendered transcript out of the
prompt and applies a bilingual rule engine to produce a genuinely input-dependent
enrichment. That matters because it means the whole downstream stack (validation,
repair, Gold aggregation, escalation features, dashboards) is exercised against
varying, plausible data with no server running.

It also serves as the **weak-supervision labeller** described in
docs/04-evaluation-framework.md: these same rules generate the noisy labels used
to bootstrap evaluation before any human annotation exists, and they are the
baseline that the real LLM has to beat.
"""

from __future__ import annotations

import json
import re
from typing import Any

from ..config import Settings
from ..models.enums import (
    ComplianceOutcome,
    ComplianceRule,
    DissatisfactionSignal,
    EscalationIndicator,
    ResolutionStatus,
    Sentiment,
)
from .base import LlmBackend, LlmResponse

_TURN_RE = re.compile(
    r"^\[(?P<idx>\d+)\]\s+(?P<role>AGENT|CUSTOMER|IVR|UNKNOWN)\s+\((?P<meta>[^)]*)\)\s+"
    r"@(?P<sec>\d+)s:\s*(?P<text>.*)$"
)

# --- bilingual signal lexicons ---------------------------------------------
_NEGATIVE = [
    "galat", "ग़लत", "गलत", "kharab", "ख़राब", "खराब", "bekar", "bekaar", "बेकार",
    "pareshan", "परेशान", "gussa", "ग़ुस्सा", "naraz", "नाराज़", "dikkat", "दिक्कत",
    "samasya", "समस्या", "shikayat", "शिकायत", "nahi hua", "नहीं हुआ", "mazaak",
    "dhokha", "धोखा", "worst", "terrible", "useless", "unacceptable", "frustrated",
    "angry", "disappointed", "poor service", "not working", "still not", "again",
    "delay", "delayed", "failed", "wrong", "issue", "problem", "complaint",
]
_STRONG_NEGATIVE = [
    "bekar", "bekaar", "बेकार", "dhokha", "धोखा", "mazaak", "unacceptable",
    "worst", "consumer forum", "trai", "court", "kanoon", "legal", "social media",
    "band karwana", "cancel my", "बंद करवाना",
]
_POSITIVE = [
    "thank", "thanks", "dhanyavaad", "dhanyavad", "धन्यवाद", "shukriya", "शुक्रिया",
    "perfect", "great", "helpful", "achha", "अच्छा", "theek hai", "ठीक है",
    "samajh gaya", "समझ गया", "appreciate", "good", "nice", "bahut achhe",
]
_ESCALATION_PATTERNS: list[tuple[str, EscalationIndicator]] = [
    (r"\bsenior\b|\bsupervisor\b|\bmanager\b|\badhikari\b|अधिकारी|सीनियर", EscalationIndicator.SUPERVISOR_REQUEST),
    (r"consumer forum|उपभोक्ता|trai|ट्राई|court|कोर्ट|kanoon|क़ानून|legal|adalat", EscalationIndicator.LEGAL_OR_REGULATOR_THREAT),
    (r"social media|twitter|facebook|post karunga|पोस्ट कर", EscalationIndicator.SOCIAL_MEDIA_THREAT),
    (r"band kar|बंद कर|cancel|deactivat|port out|dusri company|jio|airtel de raha", EscalationIndicator.CHURN_THREAT),
    (r"teesri baar|तीसरी बार|paanchvi baar|पाँचवी बार|baar call|बार कॉल|again and again|third time|fifth time", EscalationIndicator.REPEAT_CONTACT),
    (r"bekar|बेकार|bakwas|बकवास|nonsense|rubbish", EscalationIndicator.ABUSIVE_LANGUAGE),
    (r"sla|promised|vada kiya|वादा किया|24 hours mein|committed", EscalationIndicator.SLA_BREACH_CLAIM),
]
_DISSATISFACTION_PATTERNS: list[tuple[str, DissatisfactionSignal]] = [
    (r"shikayat|शिकायत|complaint|complain", DissatisfactionSignal.EXPLICIT_COMPLAINT),
    (r"pareshan|परेशान|frustrat|gussa|ग़ुस्सा|naraz|नाराज़|angry", DissatisfactionSignal.FRUSTRATION_MARKER),
    (r"phir se|फिर से|dobara|दोबारा|again|baar baar|बार बार|repeat", DissatisfactionSignal.REPETITION_OF_ISSUE),
    (r"yeh kya|यह क्या|kya mazaak|क्या मज़ाक|really\?|seriously", DissatisfactionSignal.SARCASM),
    (r"dekhte hain|देखते हैं|theek hai chalo|whatever|as you say", DissatisfactionSignal.RESIGNED_ACCEPTANCE),
]
_RESOLUTION_PATTERNS: list[tuple[str, ResolutionStatus]] = [
    (r"escalat|senior se|सीनियर से|supervisor|transfer kar|फ़ॉरवर्ड", ResolutionStatus.ESCALATED),
    (r"callback|call back|kal .{0,15}call|update call|अपडेट कॉल", ResolutionStatus.CALLBACK_SCHEDULED),
    (r"process kar diya|कर दिया|activate ho gaya|हो गया|done sir|updated|waiver|refund .{0,15}initiat|block ho gayi", ResolutionStatus.RESOLVED),
    (r"do din|दो दिन|48 hours|अड़तालीस|next billing|shikayat darj|दर्ज कर", ResolutionStatus.PARTIALLY_RESOLVED),
]
_INTENT_PATTERNS: list[tuple[str, tuple[str, str, str | None]]] = [
    (r"sim swap|फ्रॉड|fraud|unauthorized|paise nikal|अनधिकृत|nikal gaye", ("FRAUD_SECURITY", "UNAUTHORIZED_TRANSACTION", "SIM_SWAP_SUSPECTED")),
    (r"kyc|आधार|aadhaar|document reject|dastavez", ("ACCOUNT_MANAGEMENT", "KYC_UPDATE", "DOCUMENT_REJECTED")),
    (r"dnd|डी एन डी|प्रचार|promotional|spam", ("COMPLAINT", "DND_VIOLATION", "PROMOTIONAL_SPAM")),
    (r"delivery|डिलीवरी|courier|shipment|order id|deliver nahi", ("LOGISTICS", "DELIVERY_DELAY", "SIM_NOT_DELIVERED")),
    (r"band karwana|बंद करवाना|cancel.{0,20}connection|jio.{0,20}half price|competitor", ("RETENTION", "CANCELLATION_REQUEST", "COMPETITOR_OFFER")),
    (r"refund|वापसी|wapsi|paisa wapas", ("BILLING", "REFUND_STATUS", "REFUND_DELAY")),
    (r"kat gaye|कट गए|debited|payment fail|recharge hua hi nahi|transaction pending", ("BILLING", "PAYMENT_FAILURE", "AMOUNT_DEBITED_NOT_CREDITED")),
    (r"bill.{0,25}(zyada|ज़्यादा|high|extra)|overcharge|extra charge|847|शुल्क", ("BILLING", "OVERCHARGE_DISPUTE", "UNEXPECTED_DATA_CHARGE")),
    (r"network nahi|नेटवर्क नहीं|signal|सिग्नल|tower|टावर", ("TECHNICAL_SUPPORT", "NETWORK_CONNECTIVITY", "NO_SIGNAL_AREA")),
    (r"call.{0,15}(kat|कट)|call drop|disconnect", ("TECHNICAL_SUPPORT", "CALL_DROP", "FREQUENT_DISCONNECTION")),
    (r"router|राउटर|wifi|set up nahi", ("TECHNICAL_SUPPORT", "DEVICE_SETUP", "ROUTER_CONFIG")),
    (r"plan.{0,20}(upgrade|better|3gb|499)|ott|addon", ("SALES", "PLAN_UPGRADE", "DATA_ADDON")),
    (r"repeated failure|paanchvi baar|पाँचवी बार|internet chalta hi nahi", ("COMPLAINT", "SERVICE_QUALITY", "REPEATED_FAILURE")),
    (r"balance|बैलेंस|data balance|validity", ("INFORMATION", "BALANCE_ENQUIRY", None)),
    (r"naya plan|नया प्लान|tariff|199|unlimited calling", ("INFORMATION", "PLAN_DETAILS", "TARIFF_ENQUIRY")),
]


def _hits(text: str, patterns: list[str]) -> int:
    lowered = text.lower()
    return sum(1 for p in patterns if p in lowered)


class MockLlmBackend(LlmBackend):
    name = "mock"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def chat(
        self,
        messages: list[dict[str, str]],
        json_schema: dict[str, Any] | None = None,
        model: str | None = None,
        temperature: float = 0.0,
    ) -> LlmResponse:
        user_text = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
        call_id = self._extract(user_text, r"call_id:\s*(\S+)") or "UNKNOWN"
        turns = self._parse_turns(user_text)
        payload = self._analyse(call_id, turns)
        return LlmResponse(
            text=json.dumps(payload, ensure_ascii=False),
            model="mock:rule-engine-v1",
            prompt_tokens=len(user_text) // 4,
            completion_tokens=220,
            latency_ms=5,
        )

    def embed(self, texts: list[str], model: str | None = None) -> list[list[float]]:
        """Deterministic hashed bag-of-characters embedding. Enough for the
        similarity plumbing to be exercised; not a semantic model."""
        dim = 256
        out: list[list[float]] = []
        for text in texts:
            vec = [0.0] * dim
            for i, ch in enumerate(text.lower()):
                vec[(ord(ch) * 31 + i) % dim] += 1.0
            norm = sum(v * v for v in vec) ** 0.5 or 1.0
            out.append([v / norm for v in vec])
        return out

    # -- parsing / analysis --------------------------------------------------

    @staticmethod
    def _extract(text: str, pattern: str) -> str | None:
        m = re.search(pattern, text)
        return m.group(1) if m else None

    @staticmethod
    def _parse_turns(text: str) -> list[tuple[int, str, int, str]]:
        turns: list[tuple[int, str, int, str]] = []
        for line in text.splitlines():
            m = _TURN_RE.match(line.strip())
            if m:
                turns.append(
                    (int(m.group("idx")), m.group("role"), int(m.group("sec")), m.group("text"))
                )
        return turns

    def _analyse(self, call_id: str, turns: list[tuple[int, str, int, str]]) -> dict[str, Any]:
        customer_turns = [(i, s, t) for i, role, s, t in turns if role == "CUSTOMER"]
        all_text = " ".join(t[3] for t in turns)
        customer_text = " ".join(t[2] for t in customer_turns)

        intent = self._classify_intent(all_text)
        trajectory = self._trajectory(customer_turns)
        dissatisfaction = self._dissatisfaction(customer_turns)
        escalation = self._escalation(customer_text, all_text, dissatisfaction, trajectory)
        resolution, res_conf = self._resolution(all_text, escalation["escalated_in_call"])

        start_sent = trajectory[0]["sentiment"] if trajectory else Sentiment.NEUTRAL.value
        end_sent = trajectory[-1]["sentiment"] if trajectory else Sentiment.NEUTRAL.value

        lang_request = None
        accommodation = None
        if re.search(r"hindi mein baat|हिंदी में बात|english .{0,15}samajh nahi", all_text, re.I):
            lang_request = "hi"
            accommodation = bool(
                re.search(r"bilkul sir|main hindi mein|कोई बात नहीं|hindi mein hi baat", all_text, re.I)
            )

        return {
            "primary_intent": {
                "l1": intent[0], "l2": intent[1], "l3": intent[2],
                "confidence": intent[3],
                "evidence_turn_indices": [t[0] for t in customer_turns[:2]],
            },
            "secondary_intents": [],
            "resolution_status": resolution,
            "resolution_confidence": res_conf,
            "fcr_candidate": resolution == ResolutionStatus.RESOLVED.value
            and not escalation["escalated_in_call"],
            "sentiment_trajectory": trajectory,
            "customer_sentiment_start": start_sent,
            "customer_sentiment_end": end_sent,
            "dissatisfaction_signals": dissatisfaction,
            "escalation": escalation,
            "compliance": self._compliance_stub(),
            "summary_en": self._summary(intent, resolution, len(turns)),
            "customer_requested_language": lang_request,
            "agent_language_accommodation": accommodation,
            "action_items": self._action_items(resolution, escalation["escalated_in_call"]),
        }

    @staticmethod
    def _classify_intent(text: str) -> tuple[str, str, str | None, float]:
        lowered = text.lower()
        for pattern, (l1, l2, l3) in _INTENT_PATTERNS:
            if re.search(pattern, lowered, re.IGNORECASE):
                return l1, l2, l3, 0.78
        return "OTHER", "UNCLASSIFIED", None, 0.35

    @staticmethod
    def _score_sentiment(text: str) -> tuple[str, float]:
        neg = _hits(text, _NEGATIVE)
        strong = _hits(text, _STRONG_NEGATIVE)
        pos = _hits(text, _POSITIVE)
        raw = (pos * 0.5) - (neg * 0.35) - (strong * 0.5)
        score = max(-1.0, min(1.0, raw))
        if score <= -0.65:
            return Sentiment.VERY_NEGATIVE.value, score
        if score <= -0.2:
            return Sentiment.NEGATIVE.value, score
        if score < 0.2:
            return Sentiment.NEUTRAL.value, score
        if score < 0.6:
            return Sentiment.POSITIVE.value, score
        return Sentiment.VERY_POSITIVE.value, score

    def _trajectory(self, customer_turns: list[tuple[int, int, str]]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for idx, sec, text in customer_turns:
            label, score = self._score_sentiment(text)
            drivers: list[str] = []
            if _hits(text, _STRONG_NEGATIVE):
                drivers.append("strong_negative_language")
            if _hits(text, _NEGATIVE):
                drivers.append("negative_language")
            if _hits(text, _POSITIVE):
                drivers.append("positive_language")
            out.append(
                {
                    "turn_index": idx,
                    "at_ms": sec * 1000,
                    "speaker_role": "CUSTOMER",
                    "sentiment": label,
                    "score": round(score, 3),
                    "drivers": drivers[:5],
                }
            )
        return out

    @staticmethod
    def _dissatisfaction(customer_turns: list[tuple[int, int, str]]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for idx, _sec, text in customer_turns:
            for pattern, signal in _DISSATISFACTION_PATTERNS:
                if re.search(pattern, text, re.IGNORECASE):
                    out.append(
                        {
                            "signal": signal.value,
                            "turn_index": idx,
                            "quote": text[:300],
                            "language": None,
                            "severity": 0.7 if _hits(text, _STRONG_NEGATIVE) else 0.45,
                        }
                    )
                    break
        return out[:12]

    @staticmethod
    def _escalation(
        customer_text: str,
        all_text: str,
        dissatisfaction: list[dict[str, Any]],
        trajectory: list[dict[str, Any]],
    ) -> dict[str, Any]:
        indicators: list[str] = []
        evidence: list[int] = []
        for pattern, indicator in _ESCALATION_PATTERNS:
            if re.search(pattern, customer_text, re.IGNORECASE):
                indicators.append(indicator.value)
        indicators = sorted(set(indicators))

        escalated = bool(
            re.search(r"escalat|senior|supervisor|सीनियर|फ़्लोर|floor supervisor", all_text, re.I)
        )
        min_score = min((p["score"] for p in trajectory), default=0.0)

        risk = min(
            1.0,
            0.12
            + 0.14 * len(indicators)
            + 0.05 * len(dissatisfaction)
            + max(0.0, -min_score) * 0.35
            + (0.18 if escalated else 0.0),
        )
        if dissatisfaction:
            evidence = [d["turn_index"] for d in dissatisfaction[:3]]

        return {
            "escalated_in_call": escalated,
            "risk_score": round(risk, 3),
            "indicators": indicators,
            "rationale": (
                f"{len(indicators)} escalation indicator(s), "
                f"{len(dissatisfaction)} dissatisfaction signal(s), "
                f"minimum sentiment {min_score:.2f}."
            )[:600],
            "evidence_turn_indices": evidence,
        }

    @staticmethod
    def _resolution(all_text: str, escalated: bool) -> tuple[str, float]:
        if escalated:
            return ResolutionStatus.ESCALATED.value, 0.8
        for pattern, status in _RESOLUTION_PATTERNS:
            if re.search(pattern, all_text, re.IGNORECASE):
                return status.value, 0.7
        return ResolutionStatus.UNRESOLVED.value, 0.45

    @staticmethod
    def _compliance_stub() -> list[dict[str, Any]]:
        """The deterministic phrase-bank detector owns compliance; the mock LLM
        abstains rather than guessing, and reconcile() fills these in."""
        return [
            {
                "rule": rule.value,
                "outcome": ComplianceOutcome.NOT_APPLICABLE.value,
                "confidence": 0.3,
                "matched_text": None,
                "matched_language": None,
                "evidence_turn_indices": [],
                "detector_disagreement": False,
            }
            for rule in ComplianceRule
        ]

    @staticmethod
    def _summary(intent: tuple[str, str, str | None, float], resolution: str, n_turns: int) -> str:
        leaf = intent[2] or intent[1]
        return (
            f"Customer contacted support regarding {leaf.replace('_', ' ').lower()} "
            f"({intent[0].replace('_', ' ').title()}). "
            f"The call ran {n_turns} turns and ended with status {resolution}."
        )[:800]

    @staticmethod
    def _action_items(resolution: str, escalated: bool) -> list[str]:
        items: list[str] = []
        if escalated:
            items.append("Supervisor to review escalation and contact customer")
        if resolution in (
            ResolutionStatus.UNRESOLVED.value,
            ResolutionStatus.PARTIALLY_RESOLVED.value,
        ):
            items.append("Follow up to confirm the issue was closed")
        if resolution == ResolutionStatus.CALLBACK_SCHEDULED.value:
            items.append("Honour the scheduled callback within SLA")
        return items[:6]
