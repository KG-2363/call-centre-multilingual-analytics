"""Intent taxonomy.

A closed 3-level hierarchy, not free-text labels. Free-text intents from an LLM
look great in a demo and are worthless in a warehouse: "billing issue",
"Billing Issue", "bill problem" and "बिलिंग समस्या" become four rows in a GROUP BY
and the intent-distribution chart is meaningless.

The tree below is the enum the model must choose from. Validation rejects any
path not present here, so the warehouse only ever sees canonical values.

Evolving the taxonomy is a deliberate, versioned act (see docs/05):
  - additive leaf changes -> minor version, no backfill
  - L1/L2 changes -> major version, backfill required, dashboards pinned to a
    taxonomy_version column so historical charts do not silently change shape.
"""

from __future__ import annotations

TAXONOMY_VERSION = "intent-taxonomy/1.1.0"

INTENT_TREE: dict[str, dict[str, list[str]]] = {
    "BILLING": {
        "OVERCHARGE_DISPUTE": [
            "UNEXPECTED_DATA_CHARGE", "ROAMING_CHARGE", "VAS_SUBSCRIPTION",
            "PLAN_PRICE_MISMATCH", "LATE_FEE_DISPUTE",
        ],
        "PAYMENT_FAILURE": [
            "AMOUNT_DEBITED_NOT_CREDITED", "UPI_FAILURE", "CARD_DECLINED",
            "AUTOPAY_NOT_TRIGGERED",
        ],
        "REFUND_STATUS": ["REFUND_DELAY", "REFUND_FAILED", "REFUND_AMOUNT_DISPUTE"],
        "INVOICE_REQUEST": ["GST_INVOICE", "DUPLICATE_BILL", "BILL_EXPLANATION"],
        "PLAN_PRICING": ["TARIFF_CHANGE_QUERY", "PRORATION_QUERY"],
    },
    "TECHNICAL_SUPPORT": {
        "NETWORK_CONNECTIVITY": ["NO_SIGNAL_AREA", "SLOW_DATA", "INTERMITTENT_DROP"],
        "CALL_DROP": ["FREQUENT_DISCONNECTION", "ONE_WAY_AUDIO"],
        "DEVICE_SETUP": ["ROUTER_CONFIG", "APN_SETTINGS", "SIM_ACTIVATION"],
        "SERVICE_OUTAGE": ["PLANNED_MAINTENANCE", "UNPLANNED_OUTAGE"],
        "APP_ISSUE": ["LOGIN_FAILURE", "APP_CRASH", "OTP_NOT_RECEIVED"],
    },
    "ACCOUNT_MANAGEMENT": {
        "KYC_UPDATE": ["DOCUMENT_REJECTED", "DOCUMENT_PENDING", "ADDRESS_CHANGE"],
        "PROFILE_CHANGE": ["NAME_CORRECTION", "EMAIL_UPDATE", "NUMBER_PORT"],
        "PLAN_CHANGE": ["UPGRADE", "DOWNGRADE", "PLAN_INFO"],
        "ACCOUNT_CLOSURE": ["DEACTIVATION_REQUEST", "TEMPORARY_SUSPENSION"],
    },
    "SALES": {
        "PLAN_UPGRADE": ["DATA_ADDON", "OTT_BUNDLE", "FAMILY_PLAN"],
        "NEW_CONNECTION": ["POSTPAID", "PREPAID", "BROADBAND", "FIBER"],
        "CROSS_SELL": ["DEVICE_OFFER", "INSURANCE_ADDON"],
    },
    "RETENTION": {
        "CANCELLATION_REQUEST": ["COMPETITOR_OFFER", "SERVICE_DISSATISFACTION", "RELOCATION"],
        "WIN_BACK": ["REACTIVATION_OFFER"],
        "LOYALTY_BENEFIT": ["TENURE_DISCOUNT"],
    },
    "COMPLAINT": {
        "SERVICE_QUALITY": ["REPEATED_FAILURE", "SLA_BREACH", "UNRESOLVED_TICKET"],
        "AGENT_BEHAVIOUR": ["RUDENESS", "MISINFORMATION", "CALL_DISCONNECTED_BY_AGENT"],
        "DND_VIOLATION": ["PROMOTIONAL_SPAM", "TELEMARKETING_CALL"],
        "MIS_SELLING": ["UNDISCLOSED_CHARGES", "FALSE_PROMISE"],
    },
    "FRAUD_SECURITY": {
        "UNAUTHORIZED_TRANSACTION": ["SIM_SWAP_SUSPECTED", "PHISHING_VICTIM", "UNKNOWN_DEBIT"],
        "ACCOUNT_COMPROMISE": ["PASSWORD_RESET_REQUEST", "SUSPICIOUS_LOGIN"],
        "SIM_BLOCK": ["LOST_DEVICE", "STOLEN_DEVICE"],
    },
    "LOGISTICS": {
        "DELIVERY_DELAY": ["SIM_NOT_DELIVERED", "DEVICE_NOT_DELIVERED"],
        "INSTALLATION": ["TECHNICIAN_NO_SHOW", "RESCHEDULE_REQUEST"],
        "RETURN_PICKUP": ["PICKUP_PENDING"],
    },
    "INFORMATION": {
        "BALANCE_ENQUIRY": [],
        "PLAN_DETAILS": ["TARIFF_ENQUIRY", "VALIDITY_ENQUIRY"],
        "STORE_LOCATION": [],
        "PROCESS_ENQUIRY": ["PORTING_PROCESS", "REFUND_PROCESS"],
    },
    "OTHER": {"UNCLASSIFIED": []},
}


def l1_values() -> list[str]:
    return sorted(INTENT_TREE)


def l2_values(l1: str | None = None) -> list[str]:
    if l1:
        return sorted(INTENT_TREE.get(l1, {}))
    return sorted({l2 for subtree in INTENT_TREE.values() for l2 in subtree})


def l3_values(l1: str, l2: str) -> list[str]:
    return sorted(INTENT_TREE.get(l1, {}).get(l2, []))


def is_valid_path(l1: str, l2: str, l3: str | None) -> bool:
    subtree = INTENT_TREE.get(l1)
    if subtree is None or l2 not in subtree:
        return False
    if l3 is None:
        return True
    return l3 in subtree[l2]


def coerce_path(l1: str, l2: str, l3: str | None) -> tuple[str, str, str | None, bool]:
    """Best-effort repair of a near-miss path before we give up on it.

    Returns (l1, l2, l3, was_coerced). Handles the two failure modes we actually
    see from local models: a valid L2 placed under the wrong L1, and an
    invented L3 under an otherwise valid L1/L2."""
    l1_u = (l1 or "").strip().upper().replace(" ", "_")
    l2_u = (l2 or "").strip().upper().replace(" ", "_")
    l3_u = (l3 or "").strip().upper().replace(" ", "_") or None

    if is_valid_path(l1_u, l2_u, l3_u):
        return l1_u, l2_u, l3_u, False

    # Valid L1/L2 but a hallucinated leaf -> drop the leaf, keep the branch.
    if l1_u in INTENT_TREE and l2_u in INTENT_TREE[l1_u]:
        return l1_u, l2_u, None, True

    # Valid L2 under the wrong parent -> re-parent it.
    for parent, subtree in INTENT_TREE.items():
        if l2_u in subtree:
            leaf = l3_u if l3_u in subtree[l2_u] else None
            return parent, l2_u, leaf, True

    return "OTHER", "UNCLASSIFIED", None, True


def flat_paths() -> list[str]:
    out: list[str] = []
    for l1, subtree in INTENT_TREE.items():
        for l2, leaves in subtree.items():
            out.append(f"{l1}/{l2}")
            out.extend(f"{l1}/{l2}/{leaf}" for leaf in leaves)
    return sorted(out)


def taxonomy_prompt_block() -> str:
    """Compact rendering for the system prompt. Kept terse — a 3000-token
    taxonomy dump crowds out the transcript in an 8k context window."""
    lines: list[str] = []
    for l1, subtree in sorted(INTENT_TREE.items()):
        for l2, leaves in sorted(subtree.items()):
            leaf_str = f" [{', '.join(leaves)}]" if leaves else ""
            lines.append(f"- {l1} > {l2}{leaf_str}")
    return "\n".join(lines)
