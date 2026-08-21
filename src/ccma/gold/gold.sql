-- =====================================================================
-- GOLD LAYER
--
-- Star schema over DuckDB (local) / the same SQL runs on Snowflake,
-- BigQuery or Databricks SQL with only type-name changes.
--
-- Conventions:
--   dim_*   conformed dimensions, slowly changing where it matters
--   fact_*  one row per business event (a call, a risk score)
--   agg_*   pre-aggregated marts the dashboard reads directly
--
-- Every aggregate is filtered on transcript quality. This is the single
-- most important design decision in this layer: a KPI computed over
-- unusable transcripts is worse than a missing KPI, because someone will
-- act on it. `fact_call.is_kpi_eligible` is the gate, and the excluded
-- volume is itself published (agg_quality_daily) so the exclusion can
-- never hide silently.
-- =====================================================================

-- ---------------------------------------------------------------------
-- Staging views over the Silver JSONL files
-- ---------------------------------------------------------------------
-- `recorded_at` is cast explicitly rather than trusting JSON type inference:
-- read_json_auto may land an ISO-8601 string with an offset as VARCHAR or as
-- TIMESTAMPTZ depending on the sample it inspects, and fact_repeat_contact's
-- RANGE ... INTERVAL window requires a genuine temporal type.
CREATE OR REPLACE VIEW stg_calls AS
SELECT
    * EXCLUDE (recorded_at, ingested_at),
    CAST(recorded_at AS TIMESTAMP) AS recorded_at,
    CAST(ingested_at AS TIMESTAMP) AS ingested_at
FROM read_json_auto($bronze_calls_glob, union_by_name := true);

CREATE OR REPLACE VIEW stg_utterances AS
SELECT * FROM read_json_auto($silver_utterances_glob, union_by_name := true);

CREATE OR REPLACE VIEW stg_enrichment AS
SELECT * FROM read_json_auto($silver_enrichment_glob, union_by_name := true);

CREATE OR REPLACE VIEW stg_quality AS
SELECT * FROM read_json_auto($silver_quality_glob, union_by_name := true);

CREATE OR REPLACE VIEW stg_language_profile AS
SELECT * FROM read_json_auto($silver_langprofile_glob, union_by_name := true);

-- ---------------------------------------------------------------------
-- Dimensions
-- ---------------------------------------------------------------------
CREATE OR REPLACE TABLE dim_date AS
SELECT DISTINCT
    CAST(recorded_at AS DATE)                   AS date_key,
    EXTRACT(year    FROM recorded_at)           AS year,
    EXTRACT(month   FROM recorded_at)           AS month,
    EXTRACT(day     FROM recorded_at)           AS day,
    EXTRACT(dow     FROM recorded_at)           AS day_of_week,
    EXTRACT(dow     FROM recorded_at) IN (0, 6) AS is_weekend
FROM stg_calls;

CREATE OR REPLACE TABLE dim_agent AS
SELECT
    agent_id,
    any_value(agent_name)                       AS agent_name,
    mode(queue)                                 AS primary_queue,
    count(*)                                    AS lifetime_calls,
    min(CAST(recorded_at AS DATE))              AS first_call_date,
    max(CAST(recorded_at AS DATE))              AS last_call_date
FROM stg_calls
GROUP BY agent_id;

CREATE OR REPLACE TABLE dim_intent AS
SELECT DISTINCT
    primary_intent.l1                                                   AS intent_l1,
    primary_intent.l2                                                   AS intent_l2,
    primary_intent.l3                                                   AS intent_l3,
    concat_ws(' / ', primary_intent.l1, primary_intent.l2, primary_intent.l3) AS intent_path
FROM stg_enrichment
WHERE validation_status <> 'QUARANTINED';

-- ---------------------------------------------------------------------
-- fact_call — the conformed one-row-per-call table everything else uses
-- ---------------------------------------------------------------------
CREATE OR REPLACE TABLE fact_call AS
WITH utterance_agg AS (
    SELECT
        call_id,
        count(*)                                                       AS turn_count,
        sum(CASE WHEN speaker_role = 'AGENT'    THEN end_ms - start_ms ELSE 0 END) AS agent_talk_ms,
        sum(CASE WHEN speaker_role = 'CUSTOMER' THEN end_ms - start_ms ELSE 0 END) AS customer_talk_ms,
        sum(end_ms - start_ms)                                         AS total_talk_ms,
        avg(asr_confidence)                                            AS mean_asr_confidence,
        avg(lang_confidence)                                           AS mean_lang_confidence,
        sum(CASE WHEN overlapped_with_next THEN 1 ELSE 0 END)          AS overlap_count,
        sum(CASE WHEN is_code_switched      THEN 1 ELSE 0 END)         AS code_switched_turns,
        avg(cmi)                                                       AS mean_cmi,
        sum(CASE WHEN list_contains(quality_flags, 'hallucination_suspect')
                 THEN 1 ELSE 0 END)                                    AS hallucination_flags
    FROM stg_utterances
    GROUP BY call_id
),
compliance_agg AS (
    -- NOT_APPLICABLE rules are excluded from the denominator; see
    -- CallEnrichment.compliance_score for why this matters.
    SELECT
        call_id,
        count(*) FILTER (WHERE c.outcome <> 'NOT_APPLICABLE')          AS applicable_rules,
        count(*) FILTER (WHERE c.outcome = 'PRESENT')                  AS rules_present,
        count(*) FILTER (WHERE c.outcome = 'PARTIAL')                  AS rules_partial,
        count(*) FILTER (WHERE c.outcome = 'VIOLATED')                 AS rules_violated,
        count(*) FILTER (WHERE c.detector_disagreement)                AS rules_disputed,
        bool_or(c.rule = 'RECORDING_DISCLOSURE' AND c.outcome = 'PRESENT') AS has_recording_disclosure
    FROM stg_enrichment e, UNNEST(e.compliance) AS t(c)
    GROUP BY call_id
)
SELECT
    c.call_id,
    c.tenant_id,
    CAST(c.recorded_at AS DATE)                                        AS date_key,
    c.recorded_at,
    c.agent_id,
    c.queue,
    c.campaign,
    c.direction,
    c.customer_id_hash,
    c.customer_city,
    c.customer_state,

    -- Handling time. AHT here is talk + hold, deliberately excluding queue wait:
    -- queue wait measures staffing, not the agent, and blending them produces a
    -- scorecard agents can neither control nor trust.
    c.duration_ms / 1000.0                                             AS duration_s,
    (c.duration_ms + c.hold_total_ms) / 1000.0                         AS handling_time_s,
    c.hold_count,
    c.hold_total_ms / 1000.0                                           AS hold_time_s,
    c.transfer_count,
    c.queue_wait_ms / 1000.0                                           AS queue_wait_s,

    u.turn_count,
    u.agent_talk_ms / 1000.0                                           AS agent_talk_s,
    u.customer_talk_ms / 1000.0                                        AS customer_talk_s,
    CASE WHEN u.total_talk_ms > 0
         THEN u.agent_talk_ms::DOUBLE / u.total_talk_ms END            AS agent_talk_ratio,
    -- Silence share: high values usually mean dead air the customer experienced
    -- as being ignored.
    CASE WHEN c.duration_ms > 0
         THEN greatest(0.0, 1.0 - (u.total_talk_ms::DOUBLE / c.duration_ms)) END AS silence_ratio,
    u.overlap_count,
    u.code_switched_turns,
    u.mean_cmi,
    u.mean_asr_confidence,
    u.mean_lang_confidence,

    lp.dominant_language,
    lp.call_cmi,
    lp.switch_point_fraction,
    lp.agent_customer_language_match,
    lp.transliterated_token_ratio,

    e.primary_intent.l1                                                AS intent_l1,
    e.primary_intent.l2                                                AS intent_l2,
    e.primary_intent.l3                                                AS intent_l3,
    e.primary_intent.confidence                                        AS intent_confidence,
    e.resolution_status,
    e.resolution_confidence,
    e.fcr_candidate,
    e.customer_sentiment_start,
    e.customer_sentiment_end,
    e.sentiment_delta,
    e.sentiment_min,
    e.escalation.escalated_in_call                                     AS escalated_in_call,
    e.escalation.risk_score                                            AS llm_escalation_risk,
    len(e.escalation.indicators)                                       AS escalation_indicator_count,
    len(e.dissatisfaction_signals)                                     AS dissatisfaction_signal_count,
    e.compliance_score,
    e.summary_en,
    e.validation_status,
    e.llm_model,
    e.prompt_version,

    comp.applicable_rules,
    comp.rules_present,
    comp.rules_violated,
    comp.rules_disputed,
    coalesce(comp.has_recording_disclosure, FALSE)                     AS has_recording_disclosure,

    q.estimated_wer,
    q.quality_band,
    q.low_confidence_utterance_ratio,
    u.hallucination_flags,

    -- The KPI gate. A call is eligible only if the transcript is usable, the
    -- enrichment validated, and both speakers were actually captured.
    (
        coalesce(q.quality_band, 'UNUSABLE') IN ('GOOD', 'FAIR')
        AND e.validation_status <> 'QUARANTINED'
        AND u.turn_count >= 3
        AND u.agent_talk_ms > 0
        AND u.customer_talk_ms > 0
    )                                                                  AS is_kpi_eligible
FROM stg_calls c
LEFT JOIN utterance_agg      u    ON u.call_id    = c.call_id
LEFT JOIN stg_enrichment     e    ON e.call_id    = c.call_id
LEFT JOIN stg_quality        q    ON q.call_id    = c.call_id
LEFT JOIN stg_language_profile lp ON lp.call_id   = c.call_id
LEFT JOIN compliance_agg     comp ON comp.call_id = c.call_id;

-- ---------------------------------------------------------------------
-- fact_utterance — powers the transcript explorer and language ribbon
-- ---------------------------------------------------------------------
CREATE OR REPLACE TABLE fact_utterance AS
SELECT
    u.call_id,
    u.turn_index,
    u.speaker_role,
    u.start_ms,
    u.end_ms,
    u.end_ms - u.start_ms                       AS duration_ms,
    u.text_raw,
    u.text_normalized,
    u.text_devanagari,
    u.lang_primary,
    u.lang_confidence,
    u.is_code_switched,
    u.cmi,
    u.switch_points,
    u.asr_confidence,
    u.quality_flags,
    u.diarization_confidence,
    u.overlapped_with_next,
    f.date_key,
    f.agent_id,
    f.queue,
    f.intent_l1
FROM stg_utterances u
JOIN fact_call f ON f.call_id = u.call_id;

-- ---------------------------------------------------------------------
-- Aggregates
-- ---------------------------------------------------------------------

-- Intent distribution + AHT by intent.
CREATE OR REPLACE TABLE agg_intent_daily AS
SELECT
    date_key,
    intent_l1,
    intent_l2,
    count(*)                                                    AS call_count,
    count(*) FILTER (WHERE is_kpi_eligible)                     AS eligible_call_count,
    avg(handling_time_s)     FILTER (WHERE is_kpi_eligible)     AS aht_s,
    median(handling_time_s)  FILTER (WHERE is_kpi_eligible)     AS median_handling_time_s,
    quantile_cont(handling_time_s, 0.90) FILTER (WHERE is_kpi_eligible) AS p90_handling_time_s,
    avg(CASE WHEN fcr_candidate THEN 1.0 ELSE 0.0 END) FILTER (WHERE is_kpi_eligible) AS fcr_rate,
    avg(CASE WHEN escalated_in_call THEN 1.0 ELSE 0.0 END) FILTER (WHERE is_kpi_eligible) AS escalation_rate,
    avg(sentiment_delta)     FILTER (WHERE is_kpi_eligible)     AS mean_sentiment_delta,
    avg(intent_confidence)                                      AS mean_intent_confidence,
    avg(compliance_score)    FILTER (WHERE is_kpi_eligible)     AS mean_compliance_score
FROM fact_call
GROUP BY date_key, intent_l1, intent_l2;

-- Call volume by day / queue / language.
CREATE OR REPLACE TABLE agg_volume_daily AS
SELECT
    date_key,
    queue,
    dominant_language,
    direction,
    count(*)                                                    AS call_count,
    sum(duration_s) / 3600.0                                    AS audio_hours,
    avg(handling_time_s) FILTER (WHERE is_kpi_eligible)         AS aht_s,
    avg(queue_wait_s)                                           AS mean_queue_wait_s
FROM fact_call
GROUP BY date_key, queue, dominant_language, direction;

-- Agent scorecard.
CREATE OR REPLACE TABLE agg_agent_scorecard_daily AS
SELECT
    f.date_key,
    f.agent_id,
    a.agent_name,
    f.queue,
    count(*)                                                    AS calls_handled,
    count(*) FILTER (WHERE f.is_kpi_eligible)                   AS scored_calls,
    avg(f.handling_time_s)  FILTER (WHERE f.is_kpi_eligible)    AS aht_s,
    avg(f.hold_time_s)      FILTER (WHERE f.is_kpi_eligible)    AS mean_hold_s,
    avg(f.agent_talk_ratio) FILTER (WHERE f.is_kpi_eligible)    AS mean_talk_ratio,
    avg(f.silence_ratio)    FILTER (WHERE f.is_kpi_eligible)    AS mean_silence_ratio,
    avg(CASE WHEN f.fcr_candidate THEN 1.0 ELSE 0.0 END)    FILTER (WHERE f.is_kpi_eligible) AS fcr_rate,
    avg(CASE WHEN f.escalated_in_call THEN 1.0 ELSE 0.0 END) FILTER (WHERE f.is_kpi_eligible) AS escalation_rate,
    avg(f.sentiment_delta)  FILTER (WHERE f.is_kpi_eligible)    AS mean_sentiment_delta,
    avg(f.compliance_score) FILTER (WHERE f.is_kpi_eligible)    AS compliance_score,
    -- Language accommodation: did the agent meet the customer in their language?
    avg(f.agent_customer_language_match) FILTER (WHERE f.is_kpi_eligible) AS language_match_score,
    sum(CASE WHEN f.rules_violated > 0 THEN 1 ELSE 0 END)       AS calls_with_violation,
    avg(f.mean_asr_confidence)                                  AS mean_transcript_confidence
FROM fact_call f
LEFT JOIN dim_agent a ON a.agent_id = f.agent_id
GROUP BY f.date_key, f.agent_id, a.agent_name, f.queue;

-- Compliance adherence, per rule.
CREATE OR REPLACE TABLE agg_compliance_daily AS
SELECT
    f.date_key,
    f.queue,
    c.rule                                                      AS rule,
    count(*)                                                    AS assessed_calls,
    count(*) FILTER (WHERE c.outcome = 'PRESENT')               AS present_count,
    count(*) FILTER (WHERE c.outcome = 'ABSENT')                AS absent_count,
    count(*) FILTER (WHERE c.outcome = 'PARTIAL')               AS partial_count,
    count(*) FILTER (WHERE c.outcome = 'VIOLATED')              AS violated_count,
    count(*) FILTER (WHERE c.outcome = 'NOT_APPLICABLE')        AS not_applicable_count,
    count(*) FILTER (WHERE c.detector_disagreement)             AS disputed_count,
    -- Adherence excludes NOT_APPLICABLE from the denominator.
    CASE WHEN count(*) FILTER (WHERE c.outcome <> 'NOT_APPLICABLE') > 0
         THEN count(*) FILTER (WHERE c.outcome = 'PRESENT')::DOUBLE
              / count(*) FILTER (WHERE c.outcome <> 'NOT_APPLICABLE')
    END                                                         AS adherence_rate,
    avg(c.confidence)                                           AS mean_detection_confidence
FROM stg_enrichment e, UNNEST(e.compliance) AS t(c)
JOIN fact_call f ON f.call_id = e.call_id
WHERE f.is_kpi_eligible
GROUP BY f.date_key, f.queue, c.rule;

-- Per-language transcript quality — the multilingual monitoring surface.
CREATE OR REPLACE TABLE agg_language_quality_daily AS
SELECT
    date_key,
    lang_primary                                                AS language,
    count(*)                                                    AS utterance_count,
    count(DISTINCT call_id)                                     AS call_count,
    avg(asr_confidence)                                         AS mean_asr_confidence,
    quantile_cont(asr_confidence, 0.10)                         AS p10_asr_confidence,
    avg(lang_confidence)                                        AS mean_lang_confidence,
    avg(CASE WHEN is_code_switched THEN 1.0 ELSE 0.0 END)       AS code_switch_rate,
    avg(cmi)                                                    AS mean_cmi,
    avg(CASE WHEN list_contains(quality_flags, 'low_asr_confidence') THEN 1.0 ELSE 0.0 END) AS low_confidence_rate,
    avg(CASE WHEN list_contains(quality_flags, 'hallucination_suspect') THEN 1.0 ELSE 0.0 END) AS hallucination_rate
FROM fact_utterance
GROUP BY date_key, lang_primary;

-- Transcript-quality funnel: how much data each stage discarded, and why.
CREATE OR REPLACE TABLE agg_quality_daily AS
SELECT
    date_key,
    count(*)                                                    AS total_calls,
    count(*) FILTER (WHERE is_kpi_eligible)                     AS kpi_eligible_calls,
    count(*) FILTER (WHERE quality_band = 'GOOD')               AS good_calls,
    count(*) FILTER (WHERE quality_band = 'FAIR')               AS fair_calls,
    count(*) FILTER (WHERE quality_band = 'POOR')               AS poor_calls,
    count(*) FILTER (WHERE quality_band = 'UNUSABLE')           AS unusable_calls,
    count(*) FILTER (WHERE validation_status = 'QUARANTINED')   AS quarantined_calls,
    count(*) FILTER (WHERE validation_status = 'REPAIRED')      AS repaired_calls,
    avg(estimated_wer)                                          AS mean_estimated_wer,
    avg(mean_asr_confidence)                                    AS mean_asr_confidence,
    sum(rules_disputed)                                         AS compliance_review_queue
FROM fact_call
GROUP BY date_key;

-- Sentiment trend, sampled at the turn level so shape is preserved.
CREATE OR REPLACE TABLE agg_sentiment_daily AS
SELECT
    date_key,
    queue,
    intent_l1,
    count(*)                                                    AS call_count,
    avg(sentiment_delta)                                        AS mean_sentiment_delta,
    avg(sentiment_min)                                          AS mean_sentiment_min,
    count(*) FILTER (WHERE sentiment_delta < -0.3)              AS calls_worsened,
    count(*) FILTER (WHERE sentiment_delta >  0.3)              AS calls_improved,
    avg(CASE WHEN customer_sentiment_end IN ('NEGATIVE', 'VERY_NEGATIVE')
             THEN 1.0 ELSE 0.0 END)                             AS negative_close_rate
FROM fact_call
WHERE is_kpi_eligible
GROUP BY date_key, queue, intent_l1;

-- Repeat-contact detection: same customer, same L1 intent, within 7 days.
-- The empirical proxy for "first contact resolution actually failed".
CREATE OR REPLACE TABLE fact_repeat_contact AS
SELECT
    f.call_id,
    f.customer_id_hash,
    f.date_key,
    f.intent_l1,
    f.fcr_candidate,
    count(*) OVER (
        PARTITION BY f.customer_id_hash, f.intent_l1
        ORDER BY f.recorded_at
        RANGE BETWEEN INTERVAL 7 DAYS PRECEDING AND INTERVAL 1 SECOND PRECEDING
    )                                                           AS prior_contacts_7d,
    count(*) OVER (
        PARTITION BY f.customer_id_hash, f.intent_l1
        ORDER BY f.recorded_at
        RANGE BETWEEN INTERVAL 1 SECOND FOLLOWING AND INTERVAL 7 DAYS FOLLOWING
    )                                                           AS subsequent_contacts_7d
FROM fact_call f;

-- Measured FCR: claimed FCR that was NOT followed by a repeat contact.
-- The LLM's fcr_candidate is a hypothesis; this table is the verdict.
CREATE OR REPLACE VIEW vw_fcr_measured AS
SELECT
    f.date_key,
    f.intent_l1,
    f.agent_id,
    count(*)                                                    AS eligible_calls,
    avg(CASE WHEN f.fcr_candidate THEN 1.0 ELSE 0.0 END)        AS claimed_fcr_rate,
    avg(CASE WHEN f.fcr_candidate AND r.subsequent_contacts_7d = 0
             THEN 1.0 ELSE 0.0 END)                             AS measured_fcr_rate
FROM fact_call f
JOIN fact_repeat_contact r ON r.call_id = f.call_id
WHERE f.is_kpi_eligible
GROUP BY f.date_key, f.intent_l1, f.agent_id;
