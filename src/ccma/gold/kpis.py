"""KPI definitions.

Every KPI lives here as a single named definition with its formula, its
eligibility filter and its caveat. The point is that "AHT" means exactly one
thing across the dashboard, the alerting rules and any ad-hoc query — the most
common way analytics platforms lose credibility is three teams computing the same
metric three ways.

The `caveat` field is not documentation garnish. It is rendered next to the
number in the dashboard, because a KPI derived from imperfect ASR that presents
itself as exact is actively misleading.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class KpiDefinition:
    key: str
    label: str
    unit: str
    formula: str
    eligibility: str
    caveat: str
    higher_is_better: bool | None
    target: float | None = None


ELIGIBLE = "fact_call.is_kpi_eligible = TRUE (quality_band IN (GOOD, FAIR), enrichment validated, both speakers present, >= 3 turns)"

KPIS: dict[str, KpiDefinition] = {
    "call_volume": KpiDefinition(
        key="call_volume",
        label="Call Volume",
        unit="calls",
        formula="count(*) grouped by date_key",
        eligibility="All calls, including quality-excluded ones",
        caveat="Volume is the one metric that must never be quality-filtered — "
               "excluding unusable calls would understate real demand.",
        higher_is_better=None,
    ),
    "aht": KpiDefinition(
        key="aht",
        label="Average Handling Time",
        unit="seconds",
        formula="avg(talk_time + hold_time); queue wait excluded",
        eligibility=ELIGIBLE,
        caveat="Excludes queue wait, which measures staffing rather than agent "
               "performance. Report the median alongside: AHT is right-skewed and "
               "the mean is dragged by a handful of very long calls.",
        higher_is_better=False,
        target=300.0,
    ),
    "fcr_claimed": KpiDefinition(
        key="fcr_claimed",
        label="First Contact Resolution (claimed)",
        unit="ratio",
        formula="share of eligible calls where the LLM set fcr_candidate = true",
        eligibility=ELIGIBLE,
        caveat="This is the model's reading of the conversation, not an outcome. "
               "Always show it next to measured FCR.",
        higher_is_better=True,
        target=0.75,
    ),
    "fcr_measured": KpiDefinition(
        key="fcr_measured",
        label="First Contact Resolution (measured)",
        unit="ratio",
        formula="share of fcr_candidate calls with no repeat contact on the same "
                "L1 intent by the same customer within 7 days",
        eligibility=ELIGIBLE,
        caveat="Needs a 7-day settling window, so the most recent week is always "
               "provisional and must be rendered as such.",
        higher_is_better=True,
        target=0.70,
    ),
    "escalation_rate": KpiDefinition(
        key="escalation_rate",
        label="In-Call Escalation Rate",
        unit="ratio",
        formula="share of eligible calls with escalated_in_call = true",
        eligibility=ELIGIBLE,
        caveat="Counts escalations that happened on the call. Post-call escalation "
               "risk is a separate, predicted metric.",
        higher_is_better=False,
        target=0.08,
    ),
    "compliance_adherence": KpiDefinition(
        key="compliance_adherence",
        label="Compliance Adherence",
        unit="ratio",
        formula="PRESENT / (assessments where outcome <> NOT_APPLICABLE)",
        eligibility=ELIGIBLE,
        caveat="Disputed detections (phrase bank and LLM disagree) are counted as "
               "PARTIAL and queued for human review — they are never auto-failed.",
        higher_is_better=True,
        target=0.95,
    ),
    "recording_disclosure_rate": KpiDefinition(
        key="recording_disclosure_rate",
        label="Recording Disclosure Rate",
        unit="ratio",
        formula="share of calls with RECORDING_DISCLOSURE = PRESENT",
        eligibility=ELIGIBLE,
        caveat="Regulatory floor metric — treat any value below 1.0 as an incident, "
               "not a trend.",
        higher_is_better=True,
        target=1.0,
    ),
    "sentiment_delta": KpiDefinition(
        key="sentiment_delta",
        label="Sentiment Delta",
        unit="score (-2..2)",
        formula="mean(sentiment at close - sentiment at open)",
        eligibility=ELIGIBLE,
        caveat="Direction of travel, not absolute mood. A negative delta on a "
               "complaint queue is expected; a negative delta on an information "
               "queue is a problem.",
        higher_is_better=True,
        target=0.2,
    ),
    "negative_close_rate": KpiDefinition(
        key="negative_close_rate",
        label="Negative Close Rate",
        unit="ratio",
        formula="share of eligible calls ending NEGATIVE or VERY_NEGATIVE",
        eligibility=ELIGIBLE,
        caveat="The strongest single leading indicator of a repeat contact in this "
               "dataset.",
        higher_is_better=False,
        target=0.12,
    ),
    "agent_talk_ratio": KpiDefinition(
        key="agent_talk_ratio",
        label="Agent Talk Ratio",
        unit="ratio",
        formula="agent_talk_time / total_talk_time",
        eligibility=ELIGIBLE + "; overlapped turns excluded",
        caveat="Both extremes are bad — below ~0.35 the agent is disengaged, above "
               "~0.65 they are talking over the customer. Judge against a band, "
               "never as higher-is-better.",
        higher_is_better=None,
        target=0.50,
    ),
    "silence_ratio": KpiDefinition(
        key="silence_ratio",
        label="Silence Ratio",
        unit="ratio",
        formula="1 - (total talk time / call duration)",
        eligibility=ELIGIBLE,
        caveat="Inflated on calls with hold music that VAD did not strip; correlate "
               "with hold_count before acting on it.",
        higher_is_better=False,
        target=0.25,
    ),
    "language_match": KpiDefinition(
        key="language_match",
        label="Language Accommodation",
        unit="ratio",
        formula="distribution overlap between agent and customer language mix",
        eligibility=ELIGIBLE,
        caveat="An India-specific quality metric: an agent answering Hindi in "
               "English is a measurable CSAT risk that monolingual analytics "
               "cannot see.",
        higher_is_better=True,
        target=0.80,
    ),
    "estimated_wer": KpiDefinition(
        key="estimated_wer",
        label="Estimated WER",
        unit="ratio",
        formula="calibrated reference-free WER estimate (ccma.quality.asr_quality)",
        eligibility="All calls with at least one ASR segment",
        caveat="An estimate with a real prediction interval, not a measurement. "
               "Trust movement over time more than the absolute level, and "
               "re-calibrate whenever the ASR model changes.",
        higher_is_better=False,
        target=0.20,
    ),
    "kpi_eligibility_rate": KpiDefinition(
        key="kpi_eligibility_rate",
        label="KPI Eligibility Rate",
        unit="ratio",
        formula="eligible calls / total calls",
        eligibility="All calls",
        caveat="The honesty metric. If this drops, every other KPI on the page is "
               "computed on a smaller and possibly biased sample.",
        higher_is_better=True,
        target=0.85,
    ),
}


def get(key: str) -> KpiDefinition:
    return KPIS[key]


def all_kpis() -> list[KpiDefinition]:
    return list(KPIS.values())


EXEC_SUMMARY_SQL = """
SELECT
    count(*)                                                        AS total_calls,
    count(*) FILTER (WHERE is_kpi_eligible)                         AS eligible_calls,
    count(*) FILTER (WHERE is_kpi_eligible)::DOUBLE / nullif(count(*), 0) AS kpi_eligibility_rate,
    avg(handling_time_s) FILTER (WHERE is_kpi_eligible)             AS aht_s,
    median(handling_time_s) FILTER (WHERE is_kpi_eligible)          AS median_handling_time_s,
    avg(CASE WHEN fcr_candidate THEN 1.0 ELSE 0.0 END) FILTER (WHERE is_kpi_eligible) AS fcr_claimed,
    avg(CASE WHEN escalated_in_call THEN 1.0 ELSE 0.0 END) FILTER (WHERE is_kpi_eligible) AS escalation_rate,
    avg(compliance_score) FILTER (WHERE is_kpi_eligible)            AS compliance_adherence,
    avg(CASE WHEN has_recording_disclosure THEN 1.0 ELSE 0.0 END) FILTER (WHERE is_kpi_eligible) AS recording_disclosure_rate,
    avg(sentiment_delta) FILTER (WHERE is_kpi_eligible)             AS sentiment_delta,
    avg(CASE WHEN customer_sentiment_end IN ('NEGATIVE','VERY_NEGATIVE') THEN 1.0 ELSE 0.0 END)
        FILTER (WHERE is_kpi_eligible)                              AS negative_close_rate,
    avg(agent_talk_ratio) FILTER (WHERE is_kpi_eligible)            AS agent_talk_ratio,
    avg(silence_ratio) FILTER (WHERE is_kpi_eligible)               AS silence_ratio,
    avg(agent_customer_language_match) FILTER (WHERE is_kpi_eligible) AS language_match,
    avg(estimated_wer)                                              AS estimated_wer,
    sum(rules_disputed)                                             AS compliance_review_queue
FROM fact_call
"""
