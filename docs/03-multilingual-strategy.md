# Multilingual strategy

The core claim of this document: **in an Indian contact centre, language handling
is not a preprocessing detail — it is the primary determinant of whether every
downstream metric is true.** What follows is what actually breaks, and what this
repo does about each.

## The four inputs, and why the fourth is the hard one

| Mode | Example | Difficulty |
|---|---|---|
| English | "I cancelled my broadband two weeks ago" | Baseline |
| Hindi (Devanagari) | "मेरा बिल बहुत ज़्यादा आ गया है" | Moderate — good models exist |
| Hinglish (intra-sentential) | "Sir aapka refund process ho gaya hai, next billing cycle mein credit ho jayega" | Hard |
| Romanised Hindi | "mujhe apna paisa wapas chahiye" | **Hardest — and invisible** |

The fourth is worst because it fails *silently*. Whisper transcribes it fluently
and confidently labels it `en`. Nothing errors. The utterance flows into an
English sentiment lexicon that finds no negative words, an English compliance
phrase bank that finds no disclosure, and an intent classifier that sees an
unfamiliar English sentence. The dashboard shows a neutral, compliant,
low-confidence call. The customer was furious.

This is measurable in this repo: `tests/test_pipeline.py::test_whisper_mislabels_romanised_hindi_as_english`
asserts the failure mode exists in the corpus, so the fix is never quietly lost.

## 1. Code-switch detection and segmentation

Detection is **token-level, not utterance-level**. An utterance label cannot
distinguish "mixed" from "ambiguous", and the mixing is intra-sentential — often
mid-clause — so a single label per turn discards the phenomenon.

The cascade in [`lang/lid.py`](../src/ccma/lang/lid.py) resolves each token by the
cheapest sufficient signal:

| Stage | Signal | Confidence | Cost |
|---|---|---|---|
| 1 | Devanagari script | 0.99 | free, exact |
| 2 | Curated romanised lexicon | 0.92 | free |
| 3 | English stopword list | 0.90 | free |
| 4 | Homograph → ±2-token neighbour vote | ≤0.88 | free |
| 5 | Orthographic scorer (OOV) | ≤0.80 | free |

Stage 4 exists because roughly 40 tokens are valid in both languages. `to` is
English in "I want **to** check my bill" and Hindi (तो) in "**to** main aapko bata
deta hun". Same for `me/में`, `is/इस`, `us/उस`, `bat/बात`. Claiming these for
either side by lookup produces *confidently wrong* labels, which are worse than
low-confidence ones. Neighbour voting settles almost all of them.

Segmentation into same-language spans ([`codeswitch.segment_by_language`](../src/ccma/lang/codeswitch.py))
lets each span be routed to whichever model handles it best, rather than forcing
one model to cope with the whole mixed utterance.

### Measuring the mixing

Three metrics, because any one alone misleads:

- **CMI** (Das & Gambäck) — *how much* mixing: share of tokens outside the
  dominant language.
- **M-index** (Barnett et al.) — how *evenly* the languages are distributed.
- **Switch-point fraction** (Guzmán et al.) — how *often* the speaker alternates.

An utterance with one long English clause inside a Hindi turn has high CMI and low
SPF, and is easy for ASR. Word-level alternation has moderate CMI and high SPF, and
is hard. **SPF is what predicts ASR difficulty**, so per-language quality
monitoring buckets on SPF, not CMI.

Critically, `hi` and `hi_latn` **collapse to one language** for these metrics.
They are one language in two scripts, and counting a script change as a language
switch would report every romanised Hindi call as heavily code-mixed.

## 2. Hinglish normalisation

Normalisation is deliberately **conservative** ([`lid.normalize_text`](../src/ccma/lang/lid.py)):
lower-case, unify Devanagari encoding, collapse stutters, strip fillers. We do
*not* stem, expand contractions, or spell-correct — those destroy the exact words
a compliance auditor must be able to see.

Two Unicode issues that silently break exact matching and are fixed in
[`script.normalize_devanagari`](../src/ccma/lang/script.py):

- **Nukta encoding.** क़ exists as one codepoint and as क + U+093C. Visually
  identical, different bytes. A phrase bank written one way misses the other.
- **Chandrabindu vs anusvara.** हाँ / हां. Both occur; both must match.

## 3. Transliterated Hindi

The key move is the **Devanagari projection**: every Hindi-bearing utterance gets
a canonical Devanagari form stored alongside the raw text.

Without it, `paisa kat gaya` and `पैसा कट गया` are unrelated strings to every
embedding model, every phrase bank, and every `GROUP BY` in the warehouse. With
it, one phrase-bank entry covers both, and compliance detection over Hindi works
regardless of which script the ASR happened to emit — which, at code-switch
boundaries, is genuinely unpredictable.

Implementation is two-tier:

1. **Lexicon lookup** — correct, covers the head of the distribution, ~300 surface
   forms mapped to canonical Devanagari, with variants (`nahi/nahin/nhi/nahee`)
   folded together. Variants matter enormously: agents and ASR both produce all
   four spellings, and a lexicon knowing only one is worse than useless.
2. **Syllabic rule mapping** — approximate, covers the tail (names, inflections,
   ASR misspellings).

In production the second tier becomes **AI4Bharat IndicXlit** (seq2seq, trained on
Dakshina), with the rules retained as the zero-dependency fallback and the
deterministic path for unit tests.

## 4. Mixed-language embeddings

Where embeddings are used (compliance paraphrase matching, intent clustering,
retrieval), the encoder must be genuinely multilingual and preferably
Indic-aware. Ranked:

1. **MuRIL** (AI4Bharat) — trained on Indian languages *including transliterated
   text*, which is exactly our failure case. Best choice where an Indic-specific
   encoder is affordable.
2. **bge-m3** — strong multilingual retrieval, available in Ollama, our default.
3. **LaBSE** — good cross-lingual alignment, weaker on romanised Indic.

Anything English-only is disqualified: an English query must retrieve a Hindi
disclosure, and a monolingual encoder puts them in unrelated regions of the space.

**Always embed the Devanagari projection, not the raw Latin text.** Even a
multilingual encoder handles `पैसा` better than `paisa`, because the romanisation
is unstable and the Devanagari is not.

## 5. Language-aware intent classification

Three mechanisms, in the order they apply:

1. **Language-conditioned prompting.** The LLM receives the dominant language,
   the language mix, and per-turn language tags. It is explicitly instructed to
   treat romanised Hindi as Hindi and **not to translate before analysing** —
   translation loses register, politeness level, and the switch itself. Indian
   customers switch to Hindi when they become emotional; that switch *is* a
   dissatisfaction feature, and translating it away discards the signal.
2. **Dual projection in the prompt.** Turns are rendered with the original text
   and, where they differ, the Devanagari projection.
3. **Closed taxonomy with constrained decoding.** Language variation must not
   produce label variation. `billing issue` / `Billing Issue` / `बिलिंग समस्या`
   are four rows in a `GROUP BY` and one meaningless chart. The taxonomy is an
   enum in the JSON schema; near-misses are coerced (and marked as coerced) by
   [`taxonomy.coerce_path`](../src/ccma/enrichment/taxonomy.py).

An India-specific quality metric falls out of this layer:
**language accommodation** — did the agent meet the customer in their language?
An agent answering Hindi in English is a measurable CSAT risk that monolingual
analytics literally cannot see. It is computed per call
([`_language_match`](../src/ccma/silver/build.py)) and appears on the agent
scorecard.

## 6. Dialect and accent variation

India is not one accent. Bhojpuri-influenced Hindi from eastern UP, Deccani from
Hyderabad, Punjabi-influenced from the north, and Dravidian-L1 English from the
south all degrade Whisper differently — and unevenly across languages.

The strategy is **measure, then target**:

1. **Instrument the geography.** `customer_state` / `customer_city` are on every
   Bronze row, so quality can be sliced regionally without new collection.
2. **Report per-region estimated WER** from the reference-free estimator. If
   Patna traffic shows a consistently worse estimate than Bengaluru traffic, that
   is a real finding and it is visible without a single labelled transcript.
3. **Stratify the annotation budget by the gap.** The human-labelled seed set is
   sampled across region × language mode × acoustic profile, not uniformly.
   Uniform sampling spends the budget on the easy majority.
4. **Fine-tune where the gap justifies it.** LoRA adaptation of Whisper on
   ~50 hours of in-domain audio for the worst region, evaluated per-region and
   only shipped if it does not regress others. A single global model tuned on the
   majority accent is how you make the minority worse.
5. **Accent-aware routing as the cheap intervention.** Where a region shows
   persistently poor quality, route to a human QA sample at a higher rate rather
   than pretending the automated score is trustworthy.

`ACOUSTIC_PROFILES` in the corpus generator deliberately includes
`heavy_accent_regional` and `landline_8k_codec` so this path is exercised end to
end rather than being a paragraph in a design doc.

## 7. Per-language quality monitoring

`agg_language_quality_daily` reports, per language per day: mean and p10 ASR
confidence, mean LID confidence, code-switch rate, mean CMI, low-confidence rate,
and hallucination rate.

**The alert is on divergence, not on level.** Absolute WER needs labels;
*relative* movement between languages does not. If `hi_latn` confidence drifts
down 8% while `en` stays flat after an ASR upgrade, that is an unambiguous
regression on the hardest segment — detected with no ground truth at all. This is
the single most valuable monitoring signal in the system.

---

## Estimating per-language ASR quality without ground truth

This deserves its own treatment because it is the question that decides whether
the platform is trustworthy: **you will never have reference transcripts for
production traffic.**

The approach is reference-free quality estimation — predict WER from signals
available at inference time, then calibrate against a small labelled seed.
Implemented in [`quality/asr_quality.py`](../src/ccma/quality/asr_quality.py).

### The signals, in descending order of value

**1. Dual-decode disagreement** (weight 0.45 — the dominant term).
Decode the same audio with two models (large-v3 and small) and take the normalised
edit distance between hypotheses. Where two independently-erring systems agree,
both are usually right; where they diverge, at least one is wrong. This correlates
with true WER far better than any single-model confidence and needs **zero
labels**. Sampled at ~10% of traffic, because the second decode is pure added GPU
cost and the estimator only needs enough coverage to track the distribution.

**2. Mean word probability** (0.34). Free from `word_timestamps=True`. Well-ordered
but badly *scaled* — Whisper is systematically overconfident. Ordering is what
calibration needs; scale is what calibration fixes.

**3. Decoder telemetry** — `avg_logprob` (0.16), `no_speech_prob` (0.10),
`compression_ratio` (0.06). Their distinct value is catching **hallucination**,
which mean word probability does not: hallucinated text is emitted *confidently*.
The signature is high `no_speech_prob` + low `avg_logprob` + extreme compression
ratio; any two of three flags it ([`_is_hallucination`](../src/ccma/silver/build.py)).

**4. LM perplexity** of the hypothesis under MuRIL/IndicBERT (0.04). Garbled ASR
is improbable text. Weighted low because it mis-fires on legitimate
code-switching — precisely where we need it most, so it cannot be trusted.

### Calibration

Raw fused signal is not WER. Take a **stratified seed set of ~200 utterances**
balanced across language mode × acoustic profile × region, transcribe once by
humans, compute true WER, and fit an **isotonic** (monotone, piecewise-constant)
mapping from signal to WER.

Isotonic rather than linear because the relationship is reliably monotone but
definitely not linear, and isotonic will not extrapolate nonsense outside the
observed range. PAVA is implemented directly
([`_pool_adjacent_violators`](../src/ccma/quality/asr_quality.py)) to keep the
calibration path dependency-free.

Output is a **point estimate with an ~80% prediction interval** from the
calibration residuals — good enough to gate on, which raw confidence never is.

### Per-language estimates

Because the language layer already tags every utterance, the same calibrated
estimator is applied per language group, plus a **per-language residual bias**
term fitted where ≥5 samples exist. That bias term captures what the shared
signal fusion cannot express — e.g. we are systematically optimistic on `hi_latn`
because word probabilities there are misleadingly high (the model is confident and
wrong).

### Honest limits

- The absolute level is only as good as the seed set; re-calibrate whenever the
  ASR model changes, and the Airflow DAG does this on every run
  (`evaluate_and_calibrate`).
- On a corpus with little quality variance, rank correlation is unstable — the
  estimator ranks well when there is a real spread to rank.
- It estimates *recognition* quality, not *understanding* quality. A perfect
  transcript of an inaudible mumble is still useless, which is why
  `unintelligible_ratio` is reported separately.
