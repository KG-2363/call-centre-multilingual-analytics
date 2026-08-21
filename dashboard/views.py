"""The seven dashboard views.

Shared rules applied in every view, per the visualisation method:
  * legend whenever there are >= 2 series, plus direct labels on <= 4 series;
  * a "Show data table" toggle everywhere — this is the required relief for the
    two categorical slots that sit below 3:1 contrast on the light surface, and
    it doubles as the accessible view;
  * hover tooltips on every mark (unified crosshair on time series);
  * never two y-scales on one chart;
  * every KPI is rendered with the caveat from ccma.gold.kpis, because a number
    derived from imperfect ASR that presents itself as exact is misleading.
"""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import data as ds
import theme as th
from ccma.gold.kpis import KPIS


def _table_toggle(df: pd.DataFrame, key: str, label: str = "Show data table") -> None:
    if df is None or df.empty:
        return
    with st.expander(label):
        st.dataframe(df, use_container_width=True, hide_index=True)


def _fmt(value, kind: str = "num", digits: int = 2) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "—"
    if kind == "pct":
        return f"{value * 100:.1f}%"
    if kind == "sec":
        return f"{value:,.0f}s"
    if kind == "int":
        return f"{int(value):,}"
    return f"{value:,.{digits}f}"


# ---------------------------------------------------------------- 1. exec
def executive() -> None:
    st.subheader("Executive KPI view")
    summary = ds.exec_summary()
    if summary.empty:
        st.warning("No data. Run `ccma run-all`.")
        return
    row = summary.iloc[0]

    eligibility = row["kpi_eligibility_rate"]
    if eligibility is not None and eligibility < 0.85:
        st.warning(
            f"Only {_fmt(eligibility, 'pct')} of calls are KPI-eligible "
            f"({_fmt(row['eligible_calls'], 'int')} of {_fmt(row['total_calls'], 'int')}). "
            "Every metric below is computed on that subset."
        )

    tiles = [
        ("call_volume", "Calls", _fmt(row["total_calls"], "int"), f"{_fmt(row['eligible_calls'], 'int')} KPI-eligible"),
        ("aht", "AHT", _fmt(row["aht_s"], "sec"), f"median {_fmt(row['median_handling_time_s'], 'sec')}"),
        ("fcr_claimed", "FCR (claimed)", _fmt(row["fcr_claimed"], "pct"), ""),
        ("escalation_rate", "Escalation rate", _fmt(row["escalation_rate"], "pct"), ""),
        ("compliance_adherence", "Compliance", _fmt(row["compliance_adherence"], "pct"), f"{_fmt(row['compliance_review_queue'], 'int')} in review"),
        ("estimated_wer", "Est. WER", _fmt(row["estimated_wer"], "pct"), ""),
    ]
    for chunk in (tiles[:3], tiles[3:]):
        for col, (key, label, value, sub) in zip(st.columns(3), chunk):
            with col:
                st.markdown(
                    th.tile(label, value, sub, KPIS[key].caveat if key in KPIS else ""),
                    unsafe_allow_html=True,
                )
    st.write("")

    volume = ds.agg("agg_volume_daily")
    if not volume.empty:
        daily = (
            volume.groupby(["date_key", "dominant_language"], as_index=False)["call_count"]
            .sum()
            .sort_values("date_key")
        )
        daily["language"] = daily["dominant_language"].map(th.LANGUAGE_LABEL).fillna(daily["dominant_language"])

        fig = px.area(
            daily, x="date_key", y="call_count", color="dominant_language",
            color_discrete_map=th.LANGUAGE_COLOR,
            title="Call volume by day and dominant language",
            labels={"date_key": "", "call_count": "Calls", "dominant_language": "Language"},
        )
        fig.for_each_trace(
            lambda t: t.update(name=th.LANGUAGE_LABEL.get(t.name, t.name), line={"width": 2})
        )
        fig.update_layout(hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)
        _table_toggle(daily, "vol")

    intents = ds.agg("agg_intent_daily")
    if not intents.empty:
        by_intent = (
            intents.groupby("intent_l1", as_index=False)
            .agg(calls=("call_count", "sum"), aht=("aht_s", "mean"))
            .sort_values("calls", ascending=False)
        )
        col_a, col_b = st.columns(2)
        with col_a:
            fig = px.bar(
                by_intent, x="calls", y="intent_l1", orientation="h",
                title="Intent distribution", text="calls",
                labels={"calls": "Calls", "intent_l1": ""},
            )
            fig.update_traces(
                marker_color=th.CATEGORICAL[0], textposition="outside",
                marker_line={"width": 2, "color": th.SURFACE},
                cliponaxis=False,
            )
            fig.update_layout(yaxis={"categoryorder": "total ascending"}, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        with col_b:
            fig = px.bar(
                by_intent.dropna(subset=["aht"]), x="aht", y="intent_l1", orientation="h",
                title="Average handling time by intent",
                text=by_intent.dropna(subset=["aht"])["aht"].round(0).astype("Int64"),
                labels={"aht": "Seconds", "intent_l1": ""},
            )
            fig.update_traces(
                marker_color=th.CATEGORICAL[1], textposition="outside",
                marker_line={"width": 2, "color": th.SURFACE}, cliponaxis=False,
            )
            fig.update_layout(yaxis={"categoryorder": "total ascending"}, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        _table_toggle(by_intent, "intent")


# ------------------------------------------------- 2. transcript explorer
def transcript_explorer() -> None:
    st.subheader("Call transcript explorer")
    calls = ds.call_ids()
    if calls.empty:
        st.warning("No calls found.")
        return

    col_a, col_b, col_c = st.columns([2, 1, 1])
    with col_b:
        language_filter = st.selectbox(
            "Language", ["All", *sorted(calls["dominant_language"].dropna().unique())]
        )
    with col_c:
        quality_filter = st.selectbox("Quality band", ["All", "GOOD", "FAIR", "POOR", "UNUSABLE"])

    filtered = calls.copy()
    if language_filter != "All":
        filtered = filtered[filtered["dominant_language"] == language_filter]
    if quality_filter != "All":
        filtered = filtered[filtered["quality_band"] == quality_filter]
    if filtered.empty:
        st.info("No calls match these filters.")
        return

    with col_a:
        options = {
            f"{r.call_id} · {r.intent_l1}/{r.intent_l2} · {r.dominant_language} · {r.resolution_status}": r.call_id
            for r in filtered.itertuples()
        }
        selected = st.selectbox("Call", list(options))
    call_id = options[selected]

    meta = calls[calls["call_id"] == call_id].iloc[0]
    for col, (label, value) in zip(
        st.columns(5),
        [
            ("Agent", meta["agent_id"]),
            ("Queue", meta["queue"]),
            ("Duration", _fmt(meta["duration_s"], "sec")),
            ("Resolution", meta["resolution_status"]),
            ("Transcript quality", meta["quality_band"]),
        ],
    ):
        col.markdown(th.tile(label, str(value)), unsafe_allow_html=True)
    st.write("")

    turns = ds.utterances(call_id)
    if turns.empty:
        st.info("No utterances for this call.")
        return

    show_low_confidence = st.checkbox("Highlight low-confidence turns", value=True)

    for turn in turns.itertuples():
        lang_color = th.LANGUAGE_COLOR.get(turn.lang_primary, th.INK_MUTED)
        chips = [
            th.chip(turn.speaker_role, th.SPEAKER_COLOR.get(turn.speaker_role, th.INK_MUTED)),
            th.chip(th.LANGUAGE_LABEL.get(turn.lang_primary, turn.lang_primary), lang_color),
        ]
        if turn.is_code_switched:
            chips.append(th.chip(f"CMI {turn.cmi:.2f}", th.CATEGORICAL[3]))
        if show_low_confidence and turn.asr_confidence < 0.5:
            chips.append(th.chip(f"ASR {turn.asr_confidence:.2f}", th.STATUS["serious"]))
        for flag in (turn.quality_flags or []):
            if flag in ("hallucination_suspect", "speaker_flip_repaired", "overlapped_speech"):
                chips.append(th.chip(flag.replace("_", " "), th.STATUS["critical"]))

        deva = ""
        if turn.text_devanagari and turn.lang_primary in ("hi_latn", "mixed"):
            deva = f'<div class="ccma-turn-deva">देवनागरी: {turn.text_devanagari}</div>'

        st.markdown(
            f'<div class="ccma-turn">'
            f'<div class="ccma-turn-meta">[{turn.turn_index}] @{turn.start_ms // 1000}s '
            f'{"".join(chips)}</div>'
            f'<div class="ccma-turn-text">{turn.text_raw}</div>{deva}</div>',
            unsafe_allow_html=True,
        )

    _table_toggle(
        turns[["turn_index", "speaker_role", "lang_primary", "lang_confidence",
               "asr_confidence", "cmi", "text_raw"]],
        "turns",
    )


# ------------------------------------------------ 3. language switching
def language_view() -> None:
    st.subheader("Language & code-switching")

    quality = ds.agg("agg_language_quality_daily")
    if quality.empty:
        st.warning("No language data.")
        return

    by_lang = (
        quality.groupby("language", as_index=False)
        .agg(
            utterances=("utterance_count", "sum"),
            mean_asr=("mean_asr_confidence", "mean"),
            mean_lid=("mean_lang_confidence", "mean"),
            switch_rate=("code_switch_rate", "mean"),
            low_conf=("low_confidence_rate", "mean"),
        )
        .sort_values("utterances", ascending=False)
    )
    by_lang["label"] = by_lang["language"].map(th.LANGUAGE_LABEL).fillna(by_lang["language"])

    col_a, col_b = st.columns(2)
    with col_a:
        fig = px.bar(
            by_lang, x="utterances", y="label", orientation="h",
            color="language", color_discrete_map=th.LANGUAGE_COLOR,
            title="Utterance share by detected language", text="utterances",
            labels={"utterances": "Utterances", "label": ""},
        )
        fig.update_traces(
            textposition="outside", cliponaxis=False,
            marker_line={"width": 2, "color": th.SURFACE},
        )
        fig.update_layout(showlegend=False, yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        # Per-language ASR confidence: the label-free regression signal. Movement
        # across languages matters more than the absolute level.
        fig = px.bar(
            by_lang, x="mean_asr", y="label", orientation="h",
            color="language", color_discrete_map=th.LANGUAGE_COLOR,
            title="Mean ASR confidence by language",
            text=by_lang["mean_asr"].round(3),
            labels={"mean_asr": "Confidence", "label": ""},
        )
        fig.update_traces(
            textposition="outside", cliponaxis=False,
            marker_line={"width": 2, "color": th.SURFACE},
        )
        fig.update_layout(showlegend=False, yaxis={"categoryorder": "total ascending"}, xaxis={"range": [0, 1]})
        st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "Romanised Hindi and code-switched turns consistently score lower than "
        "monolingual English. That gap is the multilingual quality problem this "
        "platform exists to measure — not an artefact."
    )
    _table_toggle(by_lang, "lang")

    st.markdown("##### Language-switch ribbon")
    st.caption("Per-turn language for one call — where the conversation switches, and how often.")

    calls = ds.call_ids()
    switch_candidates = calls[calls["dominant_language"].isin(["mixed", "hi_latn", "hi"])]
    pool = switch_candidates if not switch_candidates.empty else calls
    call_id = st.selectbox("Call", pool["call_id"].tolist(), key="ribbon_call")

    turns = ds.utterances(call_id)
    if turns.empty:
        return

    fig = go.Figure()
    for lang in turns["lang_primary"].unique():
        subset = turns[turns["lang_primary"] == lang]
        fig.add_trace(
            go.Bar(
                x=subset["duration_ms"] / 1000,
                y=subset["speaker_role"],
                base=subset["start_ms"] / 1000,
                orientation="h",
                name=th.LANGUAGE_LABEL.get(lang, lang),
                marker={"color": th.LANGUAGE_COLOR.get(lang, th.INK_MUTED),
                        "line": {"width": 2, "color": th.SURFACE}},
                customdata=subset[["turn_index", "cmi", "asr_confidence", "text_raw"]],
                hovertemplate=(
                    "Turn %{customdata[0]}<br>%{y}<br>"
                    "CMI %{customdata[1]:.2f} · ASR %{customdata[2]:.2f}"
                    "<br>%{customdata[3]}<extra>%{fullData.name}</extra>"
                ),
            )
        )
    fig.update_layout(
        barmode="overlay",
        title=f"Language timeline — {call_id}",
        xaxis_title="Seconds into call",
        yaxis_title="",
        height=260,
    )
    st.plotly_chart(fig, use_container_width=True)

    switches = int((turns["lang_primary"] != turns["lang_primary"].shift()).sum() - 1)
    for col, (label, value) in zip(
        st.columns(4),
        [
            ("Turns", _fmt(len(turns), "int")),
            ("Language switches", _fmt(max(0, switches), "int")),
            ("Mean CMI", _fmt(turns["cmi"].mean(), digits=3)),
            ("Code-switched turns", _fmt(int(turns["is_code_switched"].sum()), "int")),
        ],
    ):
        col.markdown(th.tile(label, value), unsafe_allow_html=True)


# ----------------------------------------------------- 4. sentiment
def sentiment_view() -> None:
    st.subheader("Sentiment")

    daily = ds.agg("agg_sentiment_daily")
    if daily.empty:
        st.warning("No sentiment data.")
        return

    trend = (
        daily.groupby("date_key", as_index=False)
        .agg(
            delta=("mean_sentiment_delta", "mean"),
            worsened=("calls_worsened", "sum"),
            improved=("calls_improved", "sum"),
        )
        .sort_values("date_key")
    )

    fig = px.line(
        trend, x="date_key", y="delta", markers=True,
        title="Mean sentiment delta by day (call close minus call open)",
        labels={"date_key": "", "delta": "Sentiment delta"},
    )
    fig.update_traces(
        line={"width": 2, "color": th.CATEGORICAL[0]},
        marker={"size": 8, "line": {"width": 2, "color": th.SURFACE}},
    )
    fig.add_hline(y=0, line={"color": th.AXIS, "width": 1, "dash": "dot"})
    fig.update_layout(hovermode="x unified", showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
    st.markdown(f'<div class="ccma-caveat">{KPIS["sentiment_delta"].caveat}</div>', unsafe_allow_html=True)
    _table_toggle(trend, "senttrend")

    st.markdown("##### Sentiment trajectory for one call")
    calls = ds.call_ids()
    call_id = st.selectbox("Call", calls["call_id"].tolist(), key="sent_call")
    trajectory = ds.sentiment_trajectory(call_id)

    if trajectory.empty:
        st.info("No trajectory recorded for this call (no readable customer turns).")
    else:
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=trajectory["turn_index"], y=trajectory["score"],
                mode="lines+markers", name="Customer sentiment",
                line={"width": 2, "color": th.CATEGORICAL[0]},
                marker={
                    "size": 11,
                    "color": trajectory["score"],
                    "colorscale": th.DIVERGING_SENTIMENT,
                    "cmin": -1, "cmax": 1,
                    "line": {"width": 2, "color": th.SURFACE},
                },
                customdata=trajectory[["sentiment", "drivers"]],
                hovertemplate="Turn %{x}<br>%{customdata[0]} (%{y:.2f})<br>%{customdata[1]}<extra></extra>",
            )
        )
        fig.add_hline(y=0, line={"color": th.AXIS, "width": 1, "dash": "dot"})
        fig.update_layout(
            title=f"Customer sentiment across turns — {call_id}",
            xaxis_title="Turn index", yaxis_title="Sentiment (-1 to +1)",
            yaxis={"range": [-1.1, 1.1]}, height=320, showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)
        _table_toggle(trajectory, "traj")

    by_intent = (
        daily.groupby("intent_l1", as_index=False)
        .agg(delta=("mean_sentiment_delta", "mean"), negative_close=("negative_close_rate", "mean"))
        .sort_values("delta")
    )
    fig = px.bar(
        by_intent, x="delta", y="intent_l1", orientation="h",
        title="Sentiment delta by intent",
        text=by_intent["delta"].round(2),
        labels={"delta": "Sentiment delta", "intent_l1": ""},
    )
    fig.update_traces(
        marker={
            "color": by_intent["delta"],
            "colorscale": th.DIVERGING_SENTIMENT,
            "cmin": -1, "cmax": 1,
            "line": {"width": 2, "color": th.SURFACE},
        },
        textposition="outside", cliponaxis=False,
    )
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
    _table_toggle(by_intent, "sentintent")


# ------------------------------------------------------ 5. agents
def agent_scorecards() -> None:
    st.subheader("Agent scorecards")

    scorecards = ds.agg("agg_agent_scorecard_daily")
    if scorecards.empty:
        st.warning("No scorecard data.")
        return

    rolled = (
        scorecards.groupby(["agent_id", "agent_name"], as_index=False)
        .agg(
            calls=("calls_handled", "sum"),
            scored=("scored_calls", "sum"),
            aht=("aht_s", "mean"),
            fcr=("fcr_rate", "mean"),
            escalation=("escalation_rate", "mean"),
            compliance=("compliance_score", "mean"),
            sentiment=("mean_sentiment_delta", "mean"),
            talk_ratio=("mean_talk_ratio", "mean"),
            language_match=("language_match_score", "mean"),
            transcript_conf=("mean_transcript_confidence", "mean"),
        )
        .sort_values("calls", ascending=False)
    )

    st.caption(
        "Scored on KPI-eligible calls only. `scored` vs `calls` shows how much of "
        "each agent's volume had a transcript good enough to judge — an agent with "
        "low coverage is not a low performer, they are an unmeasured one."
    )

    display = rolled.copy()
    for col, kind in [("aht", "sec"), ("fcr", "pct"), ("escalation", "pct"),
                      ("compliance", "pct"), ("language_match", "pct")]:
        display[col] = display[col].map(lambda v, k=kind: _fmt(v, k))
    display["sentiment"] = display["sentiment"].map(lambda v: _fmt(v, digits=2))
    display["talk_ratio"] = display["talk_ratio"].map(lambda v: _fmt(v, digits=2))
    display["transcript_conf"] = display["transcript_conf"].map(lambda v: _fmt(v, digits=2))
    st.dataframe(
        display.rename(columns={
            "agent_name": "Agent", "calls": "Calls", "scored": "Scored",
            "aht": "AHT", "fcr": "FCR", "escalation": "Escalation",
            "compliance": "Compliance", "sentiment": "Sentiment Δ",
            "talk_ratio": "Talk ratio", "language_match": "Language match",
            "transcript_conf": "Transcript conf.",
        }).drop(columns=["agent_id"]),
        use_container_width=True, hide_index=True,
    )

    metric_options = {
        "AHT (seconds)": ("aht", False),
        "FCR rate": ("fcr", True),
        "Escalation rate": ("escalation", False),
        "Compliance score": ("compliance", True),
        "Language accommodation": ("language_match", True),
        "Sentiment delta": ("sentiment", True),
    }
    choice = st.selectbox("Compare agents on", list(metric_options))
    column, higher_better = metric_options[choice]

    chart_df = rolled.dropna(subset=[column]).sort_values(column, ascending=higher_better)
    fig = px.bar(
        chart_df, x=column, y="agent_name", orientation="h",
        title=f"{choice} by agent",
        text=chart_df[column].round(2),
        labels={column: choice, "agent_name": ""},
    )
    fig.update_traces(
        marker_color=th.CATEGORICAL[0] if higher_better else th.CATEGORICAL[1],
        textposition="outside", cliponaxis=False,
        marker_line={"width": 2, "color": th.SURFACE},
    )
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    # Talk ratio is judged against a band, never higher-is-better.
    st.markdown("##### Agent talk ratio against the healthy band")
    band = rolled.dropna(subset=["talk_ratio"]).sort_values("talk_ratio")
    fig = go.Figure()
    fig.add_vrect(x0=0.35, x1=0.65, fillcolor=th.STATUS["good"], opacity=0.10, line_width=0)
    fig.add_trace(
        go.Scatter(
            x=band["talk_ratio"], y=band["agent_name"], mode="markers",
            marker={"size": 13, "color": th.CATEGORICAL[0],
                    "line": {"width": 2, "color": th.SURFACE}},
            name="Talk ratio",
            hovertemplate="%{y}<br>Talk ratio %{x:.2f}<extra></extra>",
        )
    )
    fig.update_layout(
        xaxis={"range": [0, 1], "title": "Agent share of talk time"},
        yaxis_title="", height=340, showlegend=False,
        title="Healthy band 0.35–0.65 shaded; both extremes are problems",
    )
    st.plotly_chart(fig, use_container_width=True)
    st.markdown(f'<div class="ccma-caveat">{KPIS["agent_talk_ratio"].caveat}</div>', unsafe_allow_html=True)
    _table_toggle(rolled, "agents")


# --------------------------------------------------- 6. compliance
def compliance_view() -> None:
    st.subheader("Compliance monitoring")

    compliance = ds.agg("agg_compliance_daily")
    if compliance.empty:
        st.warning("No compliance data.")
        return

    by_rule = (
        compliance.groupby("rule", as_index=False)
        .agg(
            assessed=("assessed_calls", "sum"),
            present=("present_count", "sum"),
            absent=("absent_count", "sum"),
            partial=("partial_count", "sum"),
            violated=("violated_count", "sum"),
            not_applicable=("not_applicable_count", "sum"),
            disputed=("disputed_count", "sum"),
        )
    )
    # NOT_APPLICABLE is excluded from the denominator; rules that were never
    # applicable produce NaN rather than a misleading 0%.
    applicable = (by_rule["assessed"] - by_rule["not_applicable"]).astype(float)
    by_rule["adherence"] = by_rule["present"].astype(float).div(applicable.where(applicable > 0))
    by_rule = by_rule.sort_values("adherence", na_position="first")

    disclosure = by_rule[by_rule["rule"] == "RECORDING_DISCLOSURE"]
    if not disclosure.empty:
        rate = disclosure.iloc[0]["adherence"]
        if pd.notna(rate) and rate < 1.0:
            st.error(
                f"Recording disclosure present on only {_fmt(rate, 'pct')} of applicable calls. "
                "This is a regulatory floor — treat it as an incident, not a trend."
            )

    total_disputed = int(by_rule["disputed"].sum())
    if total_disputed:
        st.info(
            f"{total_disputed} assessments where the phrase-bank detector and the LLM "
            "disagreed. These are queued for human review and are never auto-failed."
        )

    fig = px.bar(
        by_rule.dropna(subset=["adherence"]), x="adherence", y="rule", orientation="h",
        title="Compliance adherence by rule (NOT_APPLICABLE excluded from the denominator)",
        text=by_rule.dropna(subset=["adherence"])["adherence"].map(lambda v: f"{v * 100:.0f}%"),
        labels={"adherence": "Adherence", "rule": ""},
    )
    fig.update_traces(
        marker_color=th.CATEGORICAL[0], textposition="outside", cliponaxis=False,
        marker_line={"width": 2, "color": th.SURFACE},
    )
    fig.add_vline(x=0.95, line={"color": th.STATUS["warning"], "width": 2, "dash": "dash"})
    fig.update_layout(xaxis={"range": [0, 1.08]}, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Dashed line marks the 95% adherence target.")

    # Sequential single-hue heatmap: adherence is a magnitude, not an identity.
    pivot = compliance.pivot_table(
        index="rule", columns="date_key", values="adherence_rate", aggfunc="mean"
    )
    if not pivot.empty:
        fig = px.imshow(
            pivot, color_continuous_scale=th.SEQUENTIAL_BLUE, zmin=0, zmax=1,
            aspect="auto", title="Adherence by rule and day",
            labels={"x": "", "y": "", "color": "Adherence"},
        )
        fig.update_traces(
            xgap=2, ygap=2,
            hovertemplate="%{y}<br>%{x}<br>Adherence %{z:.0%}<extra></extra>",
        )
        st.plotly_chart(fig, use_container_width=True)
    _table_toggle(by_rule, "comp")

    st.markdown("##### Per-call compliance detail")
    calls = ds.call_ids()
    call_id = st.selectbox("Call", calls["call_id"].tolist(), key="comp_call")
    detail = ds.compliance_detail(call_id)
    if detail.empty:
        st.info("No compliance assessment for this call.")
        return

    for row in detail.itertuples():
        color = th.COMPLIANCE_OUTCOME_COLOR.get(row.outcome, th.INK_MUTED)
        icon = {"PRESENT": "✓", "ABSENT": "✗", "PARTIAL": "~",
                "VIOLATED": "!", "NOT_APPLICABLE": "–"}.get(row.outcome, "?")
        dispute = th.chip("needs review", th.STATUS["warning"]) if row.detector_disagreement else ""
        matched = f'<div class="ccma-turn-deva">matched ({row.matched_language}): "{row.matched_text}"</div>' if row.matched_text else ""
        st.markdown(
            f'<div class="ccma-turn"><div class="ccma-turn-meta">'
            f'{th.chip(f"{icon} {row.outcome}", color)}{dispute}'
            f'confidence {row.confidence:.2f}</div>'
            f'<div class="ccma-turn-text">{row.rule.replace("_", " ").title()}</div>{matched}</div>',
            unsafe_allow_html=True,
        )


# --------------------------------------------------- 7. escalation
def escalation_view() -> None:
    st.subheader("Escalation-risk alerts")

    risk = ds.escalation_risk()
    if risk.empty:
        st.warning("No escalation scores. Run `ccma gold`.")
        return

    source = risk.iloc[0]["model_source"]
    threshold = float(risk.iloc[0]["threshold"])
    st.caption(
        f"Scored by **{source}** at threshold {threshold:.2f}. The threshold is tuned "
        "for recall: a missed escalation costs a customer, a false positive costs a "
        "supervisor two minutes."
    )

    counts = risk["risk_band"].value_counts()
    for col, band in zip(st.columns(4), ["CRITICAL", "HIGH", "MEDIUM", "LOW"]):
        col.markdown(
            th.tile(
                f"{th.RISK_BAND_ICON[band]} {band}",
                _fmt(counts.get(band, 0), "int"),
                "calls",
            ),
            unsafe_allow_html=True,
        )
    st.write("")

    band_df = (
        counts.rename_axis("risk_band").reset_index(name="calls")
        .sort_values("calls", ascending=True)
    )
    fig = px.bar(
        band_df, x="calls", y="risk_band", orientation="h",
        title="Calls by risk band", text="calls",
        labels={"calls": "Calls", "risk_band": ""},
    )
    fig.update_traces(
        marker_color=[th.RISK_BAND_COLOR.get(b, th.INK_MUTED) for b in band_df["risk_band"]],
        textposition="outside", cliponaxis=False,
        marker_line={"width": 2, "color": th.SURFACE},
    )
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("##### Flagged calls")
    flagged = risk[risk["is_flagged"]].sort_values("risk_score", ascending=False)
    if flagged.empty:
        st.success("No calls above the alert threshold.")
    for row in flagged.head(20).itertuples():
        color = th.RISK_BAND_COLOR.get(row.risk_band, th.INK_MUTED)
        icon = th.RISK_BAND_ICON.get(row.risk_band, "●")
        band_chip = th.chip(f"{icon} {row.risk_band} {row.risk_score:.2f}", color)
        st.markdown(
            f'<div class="ccma-turn">'
            f'<div class="ccma-turn-meta">{band_chip}'
            f'{row.call_id} · {row.agent_id} · {row.intent_l1} · {row.queue} · '
            f'ends {row.customer_sentiment_end}</div>'
            f'<div class="ccma-turn-text">{row.top_drivers}</div>'
            f'<div class="ccma-turn-deva">{row.summary_en}</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown("##### Learned model vs rule baseline")
    st.caption(
        "Both scores ship on every call. If the learned model cannot beat seven "
        "weighted rules it is not worth operating, so the comparison stays visible."
    )
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=risk["rule_score"], y=risk["risk_score"], mode="markers",
            marker={"size": 10, "color": th.CATEGORICAL[0],
                    "line": {"width": 2, "color": th.SURFACE}},
            customdata=risk[["call_id", "risk_band"]],
            hovertemplate="%{customdata[0]}<br>rule %{x:.2f} · model %{y:.2f}<br>%{customdata[1]}<extra></extra>",
            name="Calls",
        )
    )
    fig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1,
                  line={"color": th.AXIS, "width": 1, "dash": "dot"})
    fig.update_layout(
        xaxis={"title": "Rule baseline score", "range": [0, 1]},
        yaxis={"title": "Shipped risk score", "range": [0, 1]},
        showlegend=False, height=380,
    )
    st.plotly_chart(fig, use_container_width=True)
    _table_toggle(
        risk[["call_id", "agent_id", "intent_l1", "rule_score", "risk_score",
              "risk_band", "is_flagged", "top_drivers"]],
        "risk",
    )
