# Evaluation framework

Starting assumption: **labelled data is scarce and expensive**, and the people who
could label it (bilingual QA analysts who understand Hinglish) are the same people
whose time the platform is supposed to save.

So evaluation is designed as a **ladder**, cheapest rung first, with human effort
spent only where nothing cheaper works.

```
        cost
         ▲
   human │  ████  stratified human review (~200 utterances, quarterly)
         │  ███   dual-labelled judge/human sample (κ measurement)
    LLM  │  ██    LLM-as-judge screening (50/day, different model family)
         │  █     pseudo-labelling from high-confidence agreement
   free  │  ▌     weak supervision + reference-free QE + agreement signals
         └──────────────────────────────────────────────────► coverage
```

## Rung 1 — Weak supervision (free, 100% coverage)

Noisy labels from sources whose errors are largely independent. Implemented in
[`enrichment/mock_backend.py`](../src/ccma/enrichment/mock_backend.py) — that
module is not only an offline stub, it is the weak labeller and the baseline the
real LLM must beat.

| Source | Signal | Precision | Recall |
|---|---|---|---|
| Multilingual keyword lexicons | dissatisfaction, escalation, sentiment | Moderate | Moderate |
| Deterministic phrase bank | compliance | High | Moderate |
| Telephony metadata | transfer ⇒ escalation, hold ⇒ friction | High | Low |
| Outcome data | repeat contact in 7 days ⇒ FCR failed | High | Complete (lagged) |
| Agent disposition codes | intent (as logged) | Low | High |

Labels are combined by disjunction for escalation
([`build_weak_labels`](../src/ccma/gold/escalation.py)) — deliberately
over-generating positives, because a false positive costs a supervisor two minutes
and a false negative costs a customer.

The principled next step is a **label model** (Snorkel-style) that learns each
source's accuracy from their agreement pattern rather than treating them as
equally reliable. Worth it once there are >5 sources; at 5 the disjunction is
within noise of the fitted model and is far easier to explain.

## Rung 2 — Reference-free quality estimation (free, 100% coverage)

Covered in depth in [03-multilingual-strategy.md](03-multilingual-strategy.md#estimating-per-language-asr-quality-without-ground-truth).
Summary: dual-decode disagreement + decoder telemetry + LM perplexity, fused and
calibrated isotonically against the seed set, producing a per-call and
per-language WER estimate with a prediction interval.

## Rung 3 — Pseudo-labelling (cheap, high precision)

Where **two independent systems agree with high confidence**, treat the agreement
as a label:

- deterministic phrase bank and LLM agree on a compliance outcome → pseudo-label
- primary and shadow ASR produce identical text for a segment → treat as a
  reference transcript for that segment
- rule-based intent and LLM intent agree → pseudo-label

These bootstrap a training set for a **distilled classifier** — a small
MuRIL-based intent model that replaces the LLM on the ~70% of calls that are
routine, cutting inference cost by an order of magnitude while the LLM handles the
tail.

The trap to avoid: **confirmation bias**. Agreement between two systems that share
a failure mode is not evidence. Pseudo-labels are therefore only taken from
sources that are architecturally independent (a lexicon and a neural model, not
two neural models), and pseudo-labelled data is always kept separable from
human-labelled data so a model can be trained on either.

## Rung 4 — LLM-as-judge (moderate cost, sampled)

[`quality/judge.py`](../src/ccma/quality/judge.py). A rubric-scored audit of the
extraction against the transcript, on a stratified daily sample.

Guardrails, each addressing a documented pathology:

| Pathology | Mitigation |
|---|---|
| Self-preference (a model rates its own output higher) | Judge is a **different model family** — `gemma2` judging `qwen2.5`. Never the same weights |
| Verbosity/position bias | Per-dimension rubric with **anchored** 1–5 scales, not a bare score. Anchors describe observable properties |
| Miscalibration / score collapse | The judge's own score distribution is monitored; `judge_degenerate` fires when spread ≤ 1, invalidating the eval |
| Judge ≠ truth | **Cohen's κ against human labels** on a dual-labelled sample. Below ~0.6 the judge may not gate a release, and we say so |

The judge is a **screening instrument**: it ranks releases, sizes the human review
queue, and catches regressions. It runs *downstream of publication* in the DAG —
it informs the next release, it does not gate this one.

A dimension worth calling out: `language_handling`, anchored so that "misreads
romanised Hindi as English and draws a wrong conclusion from it" scores 2. That
is the failure this platform exists to prevent, and the eval measures it directly.

## Rung 5 — Human-in-the-loop (expensive, targeted)

Human effort goes only where it is worth the most:

1. **The seed set** (~200 utterances, refreshed quarterly). Stratified across
   language mode × acoustic profile × region — *never uniform*, because uniform
   sampling spends the budget on the easy majority. This is what calibrates the
   reference-free estimator and grounds all absolute numbers.
2. **The disagreement queue.** Every compliance detection where the phrase bank
   and LLM disagree (`detector_disagreement = true`). These are the highest
   information-per-minute reviews available: each one either fixes a phrase bank
   gap or reveals an LLM failure mode.
3. **Quarantined enrichments.** Records that failed validation after all repair
   attempts. Low volume, high diagnostic value.
4. **Judge-flagged hallucinations.** Where the judge asserts an unsupported claim,
   a human confirms and the case becomes a regression test.
5. **Active learning on the decision boundary.** Sample where the escalation
   model's calibrated probability sits near its threshold — labels there move the
   decision boundary; labels at 0.02 and 0.98 do not.

Review tooling is the transcript explorer itself: per-turn language tags,
confidence, Devanagari projection, and the matched compliance text. A reviewer who
cannot see *why* the system decided something cannot correct it efficiently.

## Success metrics

Targets are for the production system on real audio. The repo's own numbers on the
synthetic corpus are produced by `ccma evaluate`; they validate the *harness*, not
the models.

### ASR

| Metric | Target | Note |
|---|---|---|
| WER — English | ≤ 0.12 | Indian-accented English |
| WER — Hindi (Devanagari) | ≤ 0.20 | |
| WER — Hinglish / romanised | ≤ 0.30 | The honest ceiling with off-the-shelf models |
| **Transliteration-invariant WER** | ≤ 0.22 | Both sides projected to Devanagari — measures *understanding* rather than *spelling* |
| CER (Devanagari) | ≤ 0.10 | More stable where word segmentation is itself noisy |
| Hallucination rate | < 3% of segments | |
| Reference-free WER estimator MAE | ≤ 0.08 | vs. held-out true WER |
| Estimator Spearman ρ | ≥ 0.70 | **Ranking matters more than absolute accuracy** — the job is to order calls for gating and routing |

Reporting a single blended WER is forbidden by convention here. A headline 0.22
hides the only fact that matters: probably 0.09 English, 0.40 Hinglish, and those
demand completely different responses.

### Language identification

| Metric | Target |
|---|---|
| Utterance-level accuracy (collapsed hi/hi_latn) | ≥ 0.92 |
| **Romanised-Hindi recall** | ≥ 0.85 |
| Code-switch detection F1 | ≥ 0.80 |
| False "English" rate on Hindi content | **< 0.10** ← the metric that matters most |
| Token-level accuracy on homographs | ≥ 0.75 |

### Intent classification

| Metric | Target |
|---|---|
| L1 accuracy | ≥ 0.85 |
| L2 accuracy | ≥ 0.72 |
| Macro-F1 over L1 (guards rare intents) | ≥ 0.75 |
| `OTHER/UNCLASSIFIED` rate | < 0.10 |
| **Expected Calibration Error** | ≤ 0.10 |
| Accuracy gap English vs Hinglish | **≤ 0.08** ← fairness across languages |

ECE is non-negotiable for selective automation: a model that is accurate but
miscalibrated cannot be thresholded, so nothing can be auto-approved.

### Sentiment

| Metric | Target |
|---|---|
| 3-class agreement with human (neg/neu/pos) | ≥ 0.75 |
| Cohen's κ vs human | ≥ 0.60 |
| Trajectory direction accuracy (improved/worsened/flat) | ≥ 0.80 |
| Correlation of `sentiment_delta` with post-call CSAT | ρ ≥ 0.45 |

The last one is the real test — sentiment that does not correlate with the
customer's own rating is a number, not a signal.

### Compliance

| Metric | Target |
|---|---|
| Recording-disclosure recall | **≥ 0.98** |
| Recording-disclosure precision | ≥ 0.95 |
| Macro precision across rules | ≥ 0.90 |
| Macro recall across rules | ≥ 0.85 |
| Disputed rate (needing review) | ≤ 0.10 |
| Recall parity English vs Hindi | ratio ≥ 0.90 |

Recall is weighted above precision for disclosure: a missed violation is a
regulatory exposure; a false flag costs a two-minute review. Parity across
languages is explicit because a compliance system that only works in English is a
compliance system that does not work.

### Escalation prediction

| Metric | Target |
|---|---|
| PR-AUC | ≥ 0.55 (base rate ~0.15) |
| Recall at operating threshold | ≥ 0.85 |
| Precision at that threshold | ≥ 0.35 |
| Brier score | ≤ 0.12 |
| **Lift over the rule baseline** | ≥ +0.05 PR-AUC, else the baseline ships |

PR-AUC, not ROC-AUC — the classes are imbalanced and ROC-AUC flatters imbalanced
classifiers. The lift requirement is enforced in code
([`escalation.train`](../src/ccma/gold/escalation.py)): a gradient-boosted model
that cannot beat seven weighted rules is not worth operating.

### Pipeline health

| Metric | Target |
|---|---|
| KPI eligibility rate | ≥ 0.85 |
| Enrichment quarantine rate | ≤ 0.05 |
| Schema validation pass rate (first attempt) | ≥ 0.90 |
| Repair-loop success rate | ≥ 0.80 |
| Grounding-error rate (citations to non-existent turns) | ≤ 0.02 |

## Running it

```bash
ccma evaluate            # full report, fits the WER calibration
ccma evaluate --json     # machine-readable
ccma judge --sample 20   # LLM-as-judge screening
```

`ccma evaluate` writes `data/eval/evaluation_report.json` and prints a sliced
summary. Ground truth lives in `data/eval/`, which **no pipeline stage has a path
to** — that isolation is what stops label leakage.
