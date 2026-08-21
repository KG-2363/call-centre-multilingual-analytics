"""Controlled vocabularies. Every enum here is a contract: the LLM is constrained
to these values, the warehouse stores them as-is, and dashboards group on them.
Adding a value is a schema change (see docs/05-mlops-production.md)."""

from __future__ import annotations

from enum import Enum


class LanguageCode(str, Enum):
    """BCP-47-ish. `hi_latn` is the important one — romanised Hindi in Latin
    script, which off-the-shelf LID almost always mislabels as English."""

    EN = "en"
    HI = "hi"
    HI_LATN = "hi_latn"
    MIXED = "mixed"
    UNKNOWN = "und"


class SpeakerRole(str, Enum):
    AGENT = "AGENT"
    CUSTOMER = "CUSTOMER"
    IVR = "IVR"
    UNKNOWN = "UNKNOWN"


class ResolutionStatus(str, Enum):
    RESOLVED = "RESOLVED"
    PARTIALLY_RESOLVED = "PARTIALLY_RESOLVED"
    UNRESOLVED = "UNRESOLVED"
    ESCALATED = "ESCALATED"
    CALLBACK_SCHEDULED = "CALLBACK_SCHEDULED"
    ABANDONED = "ABANDONED"


class Sentiment(str, Enum):
    VERY_NEGATIVE = "VERY_NEGATIVE"
    NEGATIVE = "NEGATIVE"
    NEUTRAL = "NEUTRAL"
    POSITIVE = "POSITIVE"
    VERY_POSITIVE = "VERY_POSITIVE"


SENTIMENT_SCORE: dict[str, float] = {
    Sentiment.VERY_NEGATIVE: -1.0,
    Sentiment.NEGATIVE: -0.5,
    Sentiment.NEUTRAL: 0.0,
    Sentiment.POSITIVE: 0.5,
    Sentiment.VERY_POSITIVE: 1.0,
}


class EscalationIndicator(str, Enum):
    SUPERVISOR_REQUEST = "SUPERVISOR_REQUEST"
    REPEAT_CONTACT = "REPEAT_CONTACT"
    LEGAL_OR_REGULATOR_THREAT = "LEGAL_OR_REGULATOR_THREAT"
    SOCIAL_MEDIA_THREAT = "SOCIAL_MEDIA_THREAT"
    CHURN_THREAT = "CHURN_THREAT"
    ABUSIVE_LANGUAGE = "ABUSIVE_LANGUAGE"
    LONG_HOLD = "LONG_HOLD"
    AGENT_TRANSFER_LOOP = "AGENT_TRANSFER_LOOP"
    SLA_BREACH_CLAIM = "SLA_BREACH_CLAIM"


class DissatisfactionSignal(str, Enum):
    EXPLICIT_COMPLAINT = "EXPLICIT_COMPLAINT"
    FRUSTRATION_MARKER = "FRUSTRATION_MARKER"
    REPETITION_OF_ISSUE = "REPETITION_OF_ISSUE"
    INTERRUPTION = "INTERRUPTION"
    SARCASM = "SARCASM"
    RESIGNED_ACCEPTANCE = "RESIGNED_ACCEPTANCE"
    PRAISE_WITHHELD = "PRAISE_WITHHELD"


class ComplianceRule(str, Enum):
    RECORDING_DISCLOSURE = "RECORDING_DISCLOSURE"
    IDENTITY_VERIFICATION = "IDENTITY_VERIFICATION"
    DPDP_CONSENT = "DPDP_CONSENT"
    NO_FULL_CARD_READBACK = "NO_FULL_CARD_READBACK"
    MIS_SELLING_DISCLAIMER = "MIS_SELLING_DISCLAIMER"
    GRIEVANCE_REDRESSAL_INFO = "GRIEVANCE_REDRESSAL_INFO"
    PROFESSIONAL_CLOSING = "PROFESSIONAL_CLOSING"
    DND_ACKNOWLEDGEMENT = "DND_ACKNOWLEDGEMENT"


class ComplianceOutcome(str, Enum):
    PRESENT = "PRESENT"
    ABSENT = "ABSENT"
    PARTIAL = "PARTIAL"
    NOT_APPLICABLE = "NOT_APPLICABLE"
    VIOLATED = "VIOLATED"


class CallDirection(str, Enum):
    INBOUND = "INBOUND"
    OUTBOUND = "OUTBOUND"


class ValidationStatus(str, Enum):
    VALID = "VALID"
    REPAIRED = "REPAIRED"
    FAILED = "FAILED"
    QUARANTINED = "QUARANTINED"
