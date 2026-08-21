# Multilingual Call-Centre Analytics

End-to-end speech + conversation analytics for Indian contact centres, handling
**English, Hindi (Devanagari), Hinglish (code-switched), and romanised Hindi in
Latin script**.

Bronze → Silver → Gold over faster-whisper ASR and a local Ollama LLM, with
compliance monitoring, agent scorecards, escalation prediction, and a
quality-gated KPI layer.

> **Status of this repository.** The code is complete and internally consistent,
> but **it has not been executed** — the machine it was authored on has no working
> Python interpreter (Windows Store stub only), no ffmpeg, and no running Docker
> daemon. Follow the steps below to run it; treat the first `make test` as the
> real verification. Nothing here reports results that were not measured.

---

## Quick start (offline — no GPU, no API key, no model downloads)

```bash
# 1. Environment
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .

# 2. Configuration (defaults are already the offline mock backends)
cp .env.example .env

# 3. Check the environment
ccma doctor

# 4. Run the whole pipeline: 30 calls -> bronze -> silver -> gold
ccma run-all

# 5. Score it against ground truth (also fits the ASR quality calibration)
ccma evaluate

# 6. Dashboard
streamlit run dashboard/app.py
```

`ccma run-all` needs no network, no GPU and no API key: the ASR and LLM layers
have deterministic offline backends. Expect it to take well under a minute.

### Verify

```bash
pytest                    # ~90 tests across language, pipeline, gold, evaluation
ccma kpis                 # executive KPI summary in the terminal
ccma query "SELECT intent_l1, count(*) FROM fact_call GROUP BY 1 ORDER BY 2 DESC"
```

---

## Running it for real

### Real ASR (faster-whisper)

```bash
pip install -r requirements-heavy.txt      # faster-whisper, edge-tts
# ffmpeg must be on PATH — https://ffmpeg.org/download.html

# Generate real TTS audio for the corpus, then transcribe it
ccma seed --tts edge_tts
CCMA_ASR_BACKEND=faster_whisper ccma transcribe
```

First run downloads the Whisper weights (~3 GB for `large-v3`). On CPU, set
`CCMA_WHISPER_MODEL=small` unless you have time to spare.

### Real LLM enrichment (Ollama)

```bash
# https://ollama.com/download
ollama serve
ollama pull qwen2.5:7b-instruct
ollama pull gemma2:9b            # judge — deliberately a different model family
ollama pull bge-m3               # multilingual embeddings

CCMA_LLM_BACKEND=ollama ccma enrich
ccma gold
ccma judge --sample 20
```

`ccma doctor` reports which models are present and prints the `ollama pull`
command for any that are missing.

---

## What it does

| Stage | Command | Output |
|---|---|---|
| **Seed** | `ccma seed` | 30 synthetic calls from a 15-scenario template bank; Bronze metadata + isolated ground truth |
| **Transcribe** | `ccma transcribe` | ASR segments, speaker turns, per-token language, code-switch metrics, quality estimates |
| **Enrich** | `ccma enrich` | Intent hierarchy, sentiment trajectory, resolution, escalation, compliance — schema-validated |
| **Gold** | `ccma gold` | Star schema + 8 aggregates + escalation risk scores |
| **Evaluate** | `ccma evaluate` | Sliced metrics vs ground truth; fits the reference-free WER calibration |

Other commands: `ccma doctor`, `ccma kpis`, `ccma query <sql>`, `ccma judge`,
`ccma schemas`.

## Dashboard

Seven views: executive KPIs · transcript explorer · language & code-switching ·
sentiment · agent scorecards · compliance monitoring · escalation alerts.

Every KPI is rendered with its caveat from `ccma.gold.kpis`, because a number
derived from imperfect ASR that presents itself as exact is misleading. Every view
has a data-table toggle.

---

## The problem this solves

Off-the-shelf speech analytics fails on Indian call audio in a specific, silent
way. Whisper transcribes **romanised Hindi** fluently and labels it `en` roughly
three times in four. Nothing errors. The utterance then flows into English
sentiment lexicons, English compliance phrase banks, and an English intent
classifier. The dashboard shows a neutral, compliant call. The customer was
furious.

Three design responses:

1. **Language is resolved before meaning is extracted** — its own layer, its own
   tests, producing a *Devanagari projection* that gives Hindi content one
   canonical form regardless of the script it arrived in.
2. **Deterministic where it must be auditable, LLM where it must be flexible** —
   compliance is decided by a multilingual phrase bank an auditor can read; the
   LLM adjudicates, and disagreements are published for review rather than
   silently resolved.
3. **Quality is a first-class column** — every row carries confidence and flags,
   every call carries a calibrated WER estimate, and `is_kpi_eligible` gates every
   aggregate while `agg_quality_daily` publishes what was excluded.

---

## Documentation

| Doc | Contents |
|---|---|
| [01-architecture.md](docs/01-architecture.md) | End-to-end diagram, technology choices with rejected alternatives, cloud deployment |
| [02-data-model.md](docs/02-data-model.md) | Bronze/Silver/Gold schemas, KPI eligibility gate, retention & privacy |
| [03-multilingual-strategy.md](docs/03-multilingual-strategy.md) | **Code-switching, Hinglish, transliteration, embeddings, dialect, per-language monitoring — and estimating ASR quality with no ground truth** |
| [04-evaluation-framework.md](docs/04-evaluation-framework.md) | Weak supervision, pseudo-labelling, LLM-as-judge with guardrails, human-in-the-loop, full metric targets |
| [05-mlops-production.md](docs/05-mlops-production.md) | Scaling, orchestration, DQ, monitoring, schema evolution, cost, observability, DPDP |
| [06-tradeoffs.md](docs/06-tradeoffs.md) | Ten decisions, what each costs, and what would change my mind |

## Repository layout

```
src/ccma/
  models/        Pydantic contracts (Bronze, Silver, enrichment) — the schemas
  ingestion/     Template bank (the corpus), TTS, Bronze writer
  asr/           faster-whisper + mock backends, diarization & repair
  lang/          script · lexicon · transliteration · LID fusion · code-switch
  silver/        Bronze -> conformed conversation
  enrichment/    Ollama + mock, prompts, taxonomy, compliance, validate/repair
  quality/       Reference-free WER estimation, DQ checks, LLM-as-judge
  gold/          DuckDB star schema (gold.sql), KPI definitions, escalation model
  pipeline.py    Stage functions — shared by the CLI and the DAG
  evaluate.py    Evaluation harness + calibration fitting
  cli.py         `ccma`
dashboard/       Streamlit: theme, data access, 7 views
pipelines/       Airflow DAG · Kafka streaming consumer
tests/           Language, pipeline, gold, evaluation
docs/            Design documentation
```

## Configuration

All via environment / `.env` (see `.env.example`). The ones that matter:

| Variable | Default | Purpose |
|---|---|---|
| `CCMA_ASR_BACKEND` | `mock` | `mock` \| `faster_whisper` |
| `CCMA_LLM_BACKEND` | `mock` | `mock` \| `ollama` |
| `CCMA_OLLAMA_MODEL` | `qwen2.5:7b-instruct` | Bulk enrichment tier |
| `CCMA_OLLAMA_MODEL_HEAVY` | `qwen2.5:14b-instruct` | Low-confidence escalation tier |
| `CCMA_OLLAMA_MODEL_JUDGE` | `gemma2:9b` | Judge — different family, to avoid self-preference |
| `CCMA_TTS_BACKEND` | `none` | `none` \| `edge_tts` |
| `CCMA_RANDOM_SEED` | `20260731` | Corpus determinism — the mock ASR needs the same seed as `ccma seed` |
| `CCMA_DATA_ROOT` | `./data` | Point at `s3://…` in production |

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `not in the generated corpus` | Seed changed between `ccma seed` and `ccma transcribe` | Keep `CCMA_RANDOM_SEED` fixed, or re-run `ccma seed` |
| `Ollama is not reachable` | Server not running | `ollama serve`, or `CCMA_LLM_BACKEND=mock` |
| `Silver layer incomplete` | Ran `ccma gold` too early | `ccma transcribe && ccma enrich` first |
| `audio missing for CALL-…` | Real ASR with no audio rendered | `ccma seed --tts edge_tts`, or `CCMA_ASR_BACKEND=mock` |
| Warehouse locked | Dashboard holds a read connection | Stop Streamlit before re-running `ccma gold` |
| ffmpeg errors during `--tts` | ffmpeg not on PATH | Install it; without it a silent placeholder track is written and logged |
