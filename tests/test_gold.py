"""Gold layer: the star schema builds, KPIs are internally consistent, and the
escalation model behaves sanely on the mock corpus."""

from __future__ import annotations

import pandas as pd
import pytest

from ccma.gold import escalation
from ccma.gold.kpis import KPIS


@pytest.fixture(scope="module")
def built_warehouse(tmp_path_factory):
    """Full pipeline into a throwaway warehouse. Slow-ish but it is the only
    test that proves the layers actually compose."""
    from ccma.config import Settings
    from ccma.pipeline import run_all

    root = tmp_path_factory.mktemp("gold-run")
    settings = Settings(
        data_root=root,
        asr_backend="mock",
        llm_backend="mock",
        tts_backend="none",
        corpus_size=30,
        random_seed=20260731,
        log_level="ERROR",
    )
    results = run_all(settings)
    return settings, {r.stage: r for r in results}


class TestPipelineComposition:
    def test_all_stages_produced_rows(self, built_warehouse):
        _settings, stages = built_warehouse
        for stage in ("seed", "transcribe", "enrich", "gold"):
            assert stages[stage].rows > 0, f"{stage} produced no rows"

    def test_every_call_reaches_gold(self, built_warehouse):
        settings, _stages = built_warehouse
        from ccma.gold.build import query

        df = query(settings, "SELECT count(*) AS n FROM fact_call")
        assert int(df.iloc[0]["n"]) == settings.corpus_size

    def test_expected_tables_exist(self, built_warehouse):
        settings, stages = built_warehouse
        expected = {
            "fact_call", "fact_utterance", "dim_agent", "dim_date", "dim_intent",
            "agg_intent_daily", "agg_volume_daily", "agg_agent_scorecard_daily",
            "agg_compliance_daily", "agg_language_quality_daily",
            "agg_quality_daily", "agg_sentiment_daily", "fact_repeat_contact",
            "fact_escalation_risk",
        }
        assert expected.issubset(set(stages["gold"].partitions))


class TestFactCallIntegrity:
    def test_no_duplicate_calls(self, built_warehouse):
        settings, _ = built_warehouse
        from ccma.gold.build import query

        df = query(settings, "SELECT call_id, count(*) n FROM fact_call GROUP BY call_id HAVING n > 1")
        assert df.empty

    def test_ratios_in_range(self, built_warehouse):
        settings, _ = built_warehouse
        from ccma.gold.build import query

        df = query(
            settings,
            """
            SELECT count(*) AS bad FROM fact_call
            WHERE (agent_talk_ratio IS NOT NULL AND (agent_talk_ratio < 0 OR agent_talk_ratio > 1))
               OR (silence_ratio    IS NOT NULL AND (silence_ratio    < 0 OR silence_ratio    > 1))
               OR (compliance_score IS NOT NULL AND (compliance_score < 0 OR compliance_score > 1))
            """,
        )
        assert int(df.iloc[0]["bad"]) == 0

    def test_handling_time_includes_hold(self, built_warehouse):
        settings, _ = built_warehouse
        from ccma.gold.build import query

        df = query(
            settings,
            "SELECT count(*) AS bad FROM fact_call WHERE handling_time_s < duration_s - 0.001",
        )
        assert int(df.iloc[0]["bad"]) == 0

    def test_kpi_eligibility_excludes_quarantined(self, built_warehouse):
        settings, _ = built_warehouse
        from ccma.gold.build import query

        df = query(
            settings,
            "SELECT count(*) AS bad FROM fact_call "
            "WHERE is_kpi_eligible AND validation_status = 'QUARANTINED'",
        )
        assert int(df.iloc[0]["bad"]) == 0

    def test_language_detected_for_most_calls(self, built_warehouse):
        settings, _ = built_warehouse
        from ccma.gold.build import query

        df = query(
            settings,
            "SELECT avg(CASE WHEN dominant_language IN ('und') OR dominant_language IS NULL "
            "THEN 1.0 ELSE 0.0 END) AS unknown_rate FROM fact_call",
        )
        assert float(df.iloc[0]["unknown_rate"]) < 0.10


class TestAggregates:
    def test_intent_daily_totals_match_fact_call(self, built_warehouse):
        settings, _ = built_warehouse
        from ccma.gold.build import query

        agg = query(settings, "SELECT sum(call_count) AS n FROM agg_intent_daily")
        fact = query(settings, "SELECT count(*) AS n FROM fact_call WHERE intent_l1 IS NOT NULL")
        assert int(agg.iloc[0]["n"]) == int(fact.iloc[0]["n"])

    def test_compliance_adherence_bounded(self, built_warehouse):
        settings, _ = built_warehouse
        from ccma.gold.build import query

        df = query(
            settings,
            "SELECT count(*) AS bad FROM agg_compliance_daily "
            "WHERE adherence_rate IS NOT NULL AND (adherence_rate < 0 OR adherence_rate > 1)",
        )
        assert int(df.iloc[0]["bad"]) == 0

    def test_exec_summary_returns_one_row(self, built_warehouse):
        settings, _ = built_warehouse
        from ccma.gold.build import query
        from ccma.gold.kpis import EXEC_SUMMARY_SQL

        df = query(settings, EXEC_SUMMARY_SQL)
        assert len(df) == 1
        assert int(df.iloc[0]["total_calls"]) == settings.corpus_size

    def test_language_quality_covers_multiple_languages(self, built_warehouse):
        settings, _ = built_warehouse
        from ccma.gold.build import query

        df = query(settings, "SELECT DISTINCT language FROM agg_language_quality_daily")
        assert len(df) >= 2, "corpus should exercise more than one language"


class TestKpiDefinitions:
    def test_every_kpi_documents_its_caveat(self):
        for kpi in KPIS.values():
            assert kpi.caveat.strip(), f"{kpi.key} has no caveat"
            assert kpi.formula.strip()

    def test_talk_ratio_is_not_higher_is_better(self):
        assert KPIS["agent_talk_ratio"].higher_is_better is None

    def test_volume_is_not_quality_filtered(self):
        assert "All calls" in KPIS["call_volume"].eligibility


class TestEscalationModel:
    def _frame(self) -> pd.DataFrame:
        rows = []
        for i in range(60):
            escalating = i % 3 == 0
            rows.append(
                {
                    "call_id": f"C{i}",
                    "date_key": "2026-07-20",
                    "agent_id": f"A{i % 5}",
                    "intent_l1": "BILLING",
                    "sentiment_delta": -0.8 if escalating else 0.3,
                    "sentiment_min": -0.9 if escalating else -0.1,
                    "dissatisfaction_signal_count": 4 if escalating else 0,
                    "escalation_indicator_count": 3 if escalating else 0,
                    "llm_escalation_risk": 0.8 if escalating else 0.1,
                    "hold_count": 3 if escalating else 0,
                    "hold_time_s": 150 if escalating else 10,
                    "transfer_count": 1 if escalating else 0,
                    "queue_wait_s": 90,
                    "handling_time_s": 600 if escalating else 200,
                    "turn_count": 14,
                    "agent_talk_ratio": 0.5,
                    "silence_ratio": 0.2,
                    "overlap_count": 2 if escalating else 0,
                    "compliance_score": 0.6 if escalating else 1.0,
                    "prior_contacts_7d": 2 if escalating else 0,
                    "subsequent_contacts_7d": 1 if escalating else 0,
                    "mean_asr_confidence": 0.8,
                    "call_cmi": 0.3,
                    "agent_customer_language_match": 0.9,
                    "escalated_in_call": escalating,
                    "customer_sentiment_end": "VERY_NEGATIVE" if escalating else "POSITIVE",
                }
            )
        return pd.DataFrame(rows)

    def test_weak_labels_fire_on_escalating_calls(self):
        df = self._frame()
        labels = escalation.build_weak_labels(df)
        assert labels.sum() == 20
        assert labels.mean() < 1.0

    def test_rule_score_separates_classes(self):
        df = self._frame()
        scores = escalation.rule_score(df)
        labels = escalation.build_weak_labels(df)
        assert scores[labels == 1].mean() > scores[labels == 0].mean() + 0.2

    def test_rule_score_bounded(self):
        assert escalation.rule_score(self._frame()).between(0, 1).all()

    def test_predict_emits_both_scores_and_drivers(self):
        df = self._frame()
        model = escalation.train(df)
        out = escalation.predict(df, model)
        assert {"rule_score", "risk_score", "risk_band", "is_flagged", "top_drivers"} <= set(out.columns)
        assert out["risk_band"].isin(["LOW", "MEDIUM", "HIGH", "CRITICAL"]).all()
        assert (out["top_drivers"].str.len() > 0).all()

    def test_falls_back_to_rules_when_training_impossible(self):
        df = self._frame().head(6)
        model = escalation.train(df)
        assert not model.trained
        out = escalation.predict(df, model)
        assert out["model_source"].iloc[0] in ("rule_baseline", "rule_fallback")

    def test_scores_exist_for_every_call_in_warehouse(self, built_warehouse):
        settings, _ = built_warehouse
        from ccma.gold.build import query

        df = query(
            settings,
            "SELECT (SELECT count(*) FROM fact_escalation_risk) AS scored, "
            "(SELECT count(*) FROM fact_call) AS total",
        )
        assert int(df.iloc[0]["scored"]) == int(df.iloc[0]["total"])


class TestEvaluation:
    def test_evaluation_report_has_every_section(self, built_warehouse):
        settings, _ = built_warehouse
        from ccma.evaluate import run_evaluation

        report = run_evaluation(settings)
        for section in ("asr", "language_id", "diarization", "enrichment",
                        "compliance", "asr_quality_calibration"):
            assert section in report

    def test_asr_sliced_by_language_mode(self, built_warehouse):
        settings, _ = built_warehouse
        from ccma.evaluate import run_evaluation

        report = run_evaluation(settings)
        assert len(report["asr"]["by_language_mode"]) >= 2

    def test_transliteration_invariant_wer_is_not_worse(self, built_warehouse):
        """Projecting both sides to Devanagari can only merge spelling variants,
        so it must never score worse than plain WER."""
        settings, _ = built_warehouse
        from ccma.evaluate import run_evaluation

        report = run_evaluation(settings)
        overall = report["asr"]["overall"]
        assert overall["translit_invariant_wer"] <= overall["wer"] + 1e-6

    def test_speaker_attribution_is_high(self, built_warehouse):
        settings, _ = built_warehouse
        from ccma.evaluate import run_evaluation

        report = run_evaluation(settings)
        accuracy = report["diarization"]["speaker_attribution_accuracy"]
        assert accuracy is not None and accuracy > 0.85

    def test_calibration_fits_and_ranks(self, built_warehouse):
        settings, _ = built_warehouse
        from ccma.evaluate import run_evaluation

        report = run_evaluation(settings)
        calibration = report["asr_quality_calibration"]
        assert calibration["fitted"]
        assert calibration["spearman_rho"] > 0.0
