# MLOps & production readiness

## Scaling: batch and streaming

### Batch — the throughput path

The bottleneck is ASR, and it is embarrassingly parallel per call.

| Volume | Audio hours/day | Architecture | Est. monthly GPU cost |
|---|---|---|---|
| 5k calls/day | ~400 | 2× g5.xlarge Spot, 6h window | ~$180 |
| 50k calls/day | ~4,000 | 12× g5.xlarge Spot, autoscaled on queue depth | ~$1,800 |
| 500k calls/day | ~40,000 | 60× g5.2xlarge Spot + distilled models on the easy majority | ~$14,000 |

Sizing basis: faster-whisper large-v3, int8_float16, VAD-stripped audio runs at
roughly 12–15× real time on an A10G. VAD removes 30–60% of duration before
decoding, so effective throughput is ~25× wall-clock audio.

What makes this scale:

- **Per-call checkpointing** — a Spot interruption loses one call, not a batch.
  This is what makes Spot (≈70% saving) safe, and Spot is the single largest cost
  lever in the system.
- **Work queue, not a file scan** — SQS with visibility timeout gives retry and
  dead-lettering for free.
- **Partition-scoped tasks** — the unit of work is `dt=YYYY-MM-DD`, so backfill is
  a parameter, not a code path.
- **Stage isolation** — Silver rebuilds from Bronze without touching audio; Gold
  rebuilds from Silver without touching the LLM.

### Streaming — the latency path

[`pipelines/streaming/consumer.py`](../pipelines/streaming/consumer.py).
Only escalation alerting runs in the stream: partial ASR → language ID →
keyword hits → **rule** score → supervisor alert. Target p95 under 3 seconds from
utterance to alert.

The discipline that matters: **the stream never writes to Gold.** Two systems
computing the same metric two ways is the classic lambda-architecture failure.
Batch remains the single source of truth for every reported number; the stream is
an operational trigger with a different correctness bar.

The learned model is deliberately **not** used in-stream. It needs features (hold
time, repeat contacts, final sentiment) that do not exist until the call ends.
Scoring a partial call with a model trained on complete calls is a silent
distribution shift, and it would be wrong in exactly the cases we care about.

At-least-once delivery with idempotent alert keys (`call_id:rule_version:bucket`).
Deduplicating in the alert store is far cheaper than transactional exactly-once
across Kafka and Redis, and a duplicate alert is a much smaller problem than a
missing one.

## Orchestration

[`pipelines/airflow/dags/ccma_batch.py`](../pipelines/airflow/dags/ccma_batch.py).

- **Pools separate contention domains.** `gpu_pool` for ASR, `llm_pool` for
  enrichment. Without them a burst of enrichment starves ASR and neither finishes.
  It also makes each stage's cost visible as a line item.
- **DQ gates are tasks, not assertions inside transforms.** A failed gate should
  be a named red box in the Airflow UI, not a stack trace buried in a log.
- **Gold publishes atomically** — built into a staging schema, swapped in after
  its gate passes, so a dashboard never reads a half-built star.
- **`FileSensor` in `reschedule` mode**, not a time assumption. The recorder drops
  files when it drops them; a time-based assumption produces silent empty
  partitions.
- **Retries with exponential backoff** on the LLM task — a busy model server
  should be waited out, not hammered.

## Data quality

[`quality/dq.py`](../src/ccma/quality/dq.py). Three severities, and the
distinction is operational, not cosmetic:

| Severity | Behaviour | Examples |
|---|---|---|
| **BLOCKING** | Partition is not published | duplicate `call_id`, negative duration, unhashed customer id, orphan segments, non-contiguous turn index, enrichment coverage < 95% |
| **WARNING** | Publishes, alerts | quarantine rate > 5%, hallucination rate > 8%, single-speaker calls > 10%, `OTHER` intent > 20%, collapsed confidence spread |
| **INFO** | Recorded for trend | disputed compliance count, review queue size |

**Block on corruption, alert on drift.** A pipeline that blocks on every anomaly
gets its checks disabled within a month, and then it has no checks at all.

Two checks worth highlighting because they catch silent model failure rather than
data corruption:

- `enrichment.intent_specificity` — a model classifying everything as `OTHER` is
  broken, not cautious.
- `enrichment.confidence_spread` — confidence collapsed to a near-constant value
  means calibration is gone, even if accuracy looks fine.

## Model monitoring

| Signal | Detects | Response |
|---|---|---|
| Per-language estimated WER (`agg_language_quality_daily`) | ASR regression, **especially on the hardest language** | Alert on *divergence between languages*, not absolute level — no labels needed |
| LID confidence distribution | New dialect, new agent cohort, script drift | Sample for review; extend lexicon |
| Intent distribution PSI vs 30-day baseline | Real demand shift vs model drift | PSI > 0.2 → investigate; correlate with campaign changes |
| Intent ECE on the labelled sample | Calibration decay | Recalibrate; block auto-approval until fixed |
| Quarantine + repair rates | Prompt/model regression | Roll back prompt version (it is a column, so this is a query) |
| Judge score distribution | Judge degeneration | `judge_degenerate` invalidates the eval |
| Escalation PR-AUC on lagged outcomes | Concept drift | Retrain monthly; ship only on lift over baseline |
| `is_kpi_eligible` rate | Upstream audio degradation | The honesty metric — if it drops, every KPI is on a smaller, possibly biased sample |

**Everything is versioned as a column**, not a config file: `asr_model`,
`asr_revision`, `llm_model`, `prompt_version`, `schema_version`,
`taxonomy_version`. When a metric shifts, "did the prompt change?" is answerable
with `GROUP BY prompt_version` rather than by archaeology in Git.

## Schema evolution

| Change | Version bump | Backfill | Notes |
|---|---|---|---|
| New optional field | minor | No | Readers tolerate absence |
| New enum value (e.g. new compliance rule) | minor | No | Historical rows show `NOT_APPLICABLE` |
| New intent **leaf** (L3) | taxonomy minor | No | Dashboards roll up to L1/L2 |
| Intent **L1/L2** change | taxonomy **major** | **Yes** | Charts pinned to `taxonomy_version` so historical shapes do not silently change |
| Field removed / retyped | major | Yes | Dual-write window, then cut |
| Prompt change altering semantics | `prompt_version` | Selective | Re-run affected partitions from Silver — no ASR re-run |

Enforcement mechanisms:

- Pydantic `extra="forbid"` on every model — unknown fields fail at the boundary.
- `jsonschema` validation on LLM output before it becomes a row.
- Parquet in Gold with `union_by_name` on read, so an added column does not break
  a historical partition.
- `ccma schemas` exports every contract as JSON Schema for consumer teams; those
  files belong in code review.

## Cost optimisation

Ordered by impact, largest first:

1. **VAD before ASR** — 30–60% of contact-centre audio is silence and hold music.
   Not decoding it is the biggest single saving and it *improves* quality by
   suppressing hallucination.
2. **Spot instances with per-call checkpointing** — ~70% off on-demand for the
   dominant cost line.
3. **Model tiering.** Bulk enrichment on a 7B; escalate to 14B only where
   confidence falls below threshold (`escalate_to_heavy_below`). On this corpus
   the heavy tier fires on a small minority of calls.
4. **Distillation.** Pseudo-labels from high-confidence agreement train a small
   MuRIL intent classifier that handles the routine ~70%, leaving the LLM for the
   tail — an order-of-magnitude cut in enrichment cost.
5. **Deterministic-first compliance.** The phrase bank settles most rule/call
   pairs at zero inference cost; the LLM only adjudicates.
6. **Send transcripts, never audio, to the LLM.** Obvious, and routinely got
   wrong.
7. **Head/tail transcript sampling** on very long calls — openings carry the
   disclosure and the stated problem, closings carry the resolution. Beats naive
   truncation, which discards the resolution.
8. **Sampled shadow decoding** (~10%) — enough to track the quality distribution
   without doubling GPU spend.
9. **Scale-to-zero LLM nodes** — nightly batch does not justify an always-on GPU.
10. **Storage lifecycle** — audio to Glacier IR at 90 days; transcripts stay hot
    indefinitely at ~0.1% of the size.

Self-hosting is itself a cost decision: enrichment is high-volume and
latency-tolerant, which is exactly where owned/reserved GPU beats per-token
pricing. The trade-off is model capability, and it is measured rather than
assumed — see [04-evaluation-framework.md](04-evaluation-framework.md).

## Observability & alerting

**Structured logs** ([`logging_conf.py`](../src/ccma/logging_conf.py)) — JSON in
production with `call_id`, `partition`, `stage`, `model`, `latency_ms` as indexed
fields. Never a formatted string that has to be regex-parsed later.

**Metrics** (Prometheus): per-stage throughput and latency percentiles, GPU
utilisation, LLM tokens/sec and queue depth, DQ pass rate by check, quarantine
rate, per-language confidence percentiles.

**Traces** (OpenTelemetry): one span per call across stages, so "why did this call
take 40 minutes" is answerable.

Alert policy — page only on things a human must fix tonight:

| Severity | Condition | Route |
|---|---|---|
| **Page** | Pipeline failed; recording-disclosure rate < 0.98; KPI eligibility < 0.70 | On-call |
| **Ticket** | Quarantine > 5%; per-language WER estimate +15% WoW; PSI > 0.2 | Team queue |
| **Digest** | Judge scores, calibration drift, review queue size | Daily email |

Recording disclosure pages because it is a regulatory floor, not a trend — any
value below target is an incident.

## Security & compliance (India-specific)

- **Data residency**: `ap-south-1`. Local LLM inference means no PII egress at
  all, which removes the hardest DPDP conversation rather than answering it.
- **DPDP Act**: purpose limitation encoded as retention policy per layer; consent
  status carried on Bronze; `DPDP_CONSENT` is an audited compliance rule.
- **PII**: MSISDN hashed with a KMS-held tenant salt before the lake. Card numbers,
  Aadhaar and OTPs redacted at Silver, before enrichment. The
  `NO_FULL_CARD_READBACK` rule doubles as a redaction trigger.
- **Access**: row-level security by `tenant_id`; agent-identifying columns
  restricted to supervisors; every dashboard query logged.
- **Audit**: S3 Object Lock on raw audio; `matched_text` retained on every
  compliance detection so an auditor sees the literal words, not a model's
  opinion of them.
