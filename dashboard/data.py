"""Warehouse access for the dashboard.

Every query is cached and read-only. The dashboard is a *consumer* of Gold — it
contains no business logic and computes no KPI itself, because a metric
recomputed in the presentation layer is a metric that will eventually disagree
with the one in the warehouse.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from ccma.config import get_settings
from ccma.gold.build import query, table_exists


@st.cache_resource
def settings():
    return get_settings()


def warehouse_ready() -> bool:
    return table_exists(settings(), "fact_call")


@st.cache_data(ttl=300)
def run(sql: str, params: tuple = ()) -> pd.DataFrame:
    return query(settings(), sql, list(params))


@st.cache_data(ttl=300)
def fact_call() -> pd.DataFrame:
    return run("SELECT * FROM fact_call ORDER BY recorded_at")


@st.cache_data(ttl=300)
def exec_summary() -> pd.DataFrame:
    from ccma.gold.kpis import EXEC_SUMMARY_SQL

    return run(EXEC_SUMMARY_SQL)


@st.cache_data(ttl=300)
def utterances(call_id: str) -> pd.DataFrame:
    return run(
        "SELECT * FROM fact_utterance WHERE call_id = ? ORDER BY turn_index", (call_id,)
    )


@st.cache_data(ttl=300)
def call_ids() -> pd.DataFrame:
    return run(
        """
        SELECT call_id, recorded_at, agent_id, queue, intent_l1, intent_l2,
               dominant_language, resolution_status, quality_band,
               escalated_in_call, llm_escalation_risk, duration_s
        FROM fact_call
        ORDER BY recorded_at DESC
        """
    )


@st.cache_data(ttl=300)
def sentiment_trajectory(call_id: str) -> pd.DataFrame:
    """Read the trajectory out of the Silver enrichment file.

    Kept out of Gold deliberately: it is a variable-length array per call, useful
    for one call at a time and never aggregated across calls, so unnesting it
    into the star schema would bloat every scan for no analytical gain."""
    s = settings()
    pattern = (s.silver_dir / "dt=*" / "enrichment.jsonl").as_posix()
    return run(
        f"""
        SELECT p.turn_index, p.at_ms, p.sentiment, p.score, p.drivers
        FROM read_json_auto('{pattern}', union_by_name := true) e,
             UNNEST(e.sentiment_trajectory) AS t(p)
        WHERE e.call_id = ?
        ORDER BY p.turn_index
        """,
        (call_id,),
    )


@st.cache_data(ttl=300)
def compliance_detail(call_id: str) -> pd.DataFrame:
    s = settings()
    pattern = (s.silver_dir / "dt=*" / "enrichment.jsonl").as_posix()
    return run(
        f"""
        SELECT c.rule, c.outcome, c.confidence, c.matched_text,
               c.matched_language, c.detector_disagreement
        FROM read_json_auto('{pattern}', union_by_name := true) e,
             UNNEST(e.compliance) AS t(c)
        WHERE e.call_id = ?
        ORDER BY c.rule
        """,
        (call_id,),
    )


@st.cache_data(ttl=300)
def agg(table: str) -> pd.DataFrame:
    return run(f"SELECT * FROM {table}")


@st.cache_data(ttl=300)
def escalation_risk() -> pd.DataFrame:
    if not table_exists(settings(), "fact_escalation_risk"):
        return pd.DataFrame()
    return run(
        """
        SELECT r.*, f.recorded_at, f.queue, f.resolution_status,
               f.customer_sentiment_end, f.duration_s, f.summary_en,
               f.dominant_language
        FROM fact_escalation_risk r
        JOIN fact_call f ON f.call_id = r.call_id
        ORDER BY r.risk_score DESC
        """
    )
