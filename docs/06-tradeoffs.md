# Key implementation trade-offs

Each entry: the decision, what it costs, and what would change my mind.

## 1. Local Ollama models over a hosted frontier LLM

**Chose:** self-hosted qwen2.5 7B/14B via Ollama.

**Cost:** a 7B is materially weaker at nuanced multilingual reasoning than a
frontier model. Sarcasm in Hinglish, implied dissatisfaction, and multi-issue
calls are where the gap shows.

**Why anyway:** call recordings are PII under the DPDP Act, and keeping inference
in-region removes a legal conversation rather than answering it. Enrichment is
high-volume and latency-tolerant — precisely the shape where owned GPU beats
per-token pricing. And backfilling a year of calls after a prompt change becomes a
scheduling problem rather than a quota negotiation.

**Compensations:** constrained JSON decoding (which matters far more for small
models than large ones), a deterministic detector owning the precision-critical
compliance path, two-tier escalation to a larger local model, and LLM-as-judge
sampling to *measure* the gap rather than assume it away.

**Would change my mind:** if judge scores or the labelled sample showed the 7B
below ~0.75 L1 intent accuracy on Hinglish, or if a hosted provider offered an
in-region endpoint with contractual no-retention. `LlmBackend` makes the swap one
class.

## 2. A hand-curated lexicon alongside neural LID

**Chose:** ~300-entry romanised-Hindi lexicon as the second stage of a cascade.

**Cost:** it is manual, domain-specific, and needs maintenance as slang shifts.
It feels unfashionable next to "just fine-tune a classifier".

**Why anyway:** off-the-shelf LID operates on script statistics and labels
romanised Hindi as English roughly three times in four. That single error poisons
sentiment, compliance and intent for the utterance. A lexicon covering the words
that actually occur in support calls fixes the head of the distribution for zero
inference cost, is inspectable when it is wrong, and gives the neural model a
prior to fuse with rather than replacing it.

**The subtlety:** variants matter more than headwords. A lexicon that knows `nahi`
but not `nahin`/`nhi` produces *confidently wrong* labels on the others, which is
worse than no lexicon. Hence variant sets throughout.

**Would change my mind:** a fine-tuned MuRIL token classifier trained on
Dakshina + in-domain data reaching ≥0.90 romanised-Hindi recall. Then the lexicon
becomes a high-precision override for the domain terms the model still misses,
not the primary mechanism.

## 3. Deterministic-first compliance, LLM as adjudicator

**Chose:** multilingual phrase bank decides; LLM only adjudicates; disagreements
are published, not resolved.

**Cost:** the phrase bank misses paraphrases. An agent who says "just so you know,
we keep a record of this conversation" is not matched.

**Why anyway:** compliance is the one output a regulator may act on. An auditor
must be able to read the rule and see the literal matched text. An LLM saying
"disclosure absent" with no quotable evidence is not defensible, and an automated
compliance *failure* on a disputed detection destroys trust in the entire
scorecard — including the parts that are right.

**The design consequence:** `detector_disagreement` is a first-class column, not
an error. Those rows are the human review queue, and they are the
highest-information reviews available — each either fixes a phrase-bank gap or
reveals an LLM failure mode.

**Would change my mind:** nothing about the ordering. But embedding-similarity
retrieval over the phrase bank (bge-m3) is the right way to close the paraphrase
gap without giving the LLM the deciding vote.

## 4. Channel separation over diarization

**Chose:** insist on stereo recording; use the channel index for speaker
attribution; pyannote only for legacy mono.

**Cost:** requires a recorder configuration change at the customer site, which is
a commercial conversation, not a technical one.

**Why anyway:** channel attribution is *exact* and free. Diarization on 8 kHz
Indian call audio with overlapping speech runs 10–20% DER, and every point of that
error propagates into agent talk ratio, per-speaker sentiment, and compliance
attribution — the metrics agents are measured on. Running a diarizer over stereo
audio is choosing a worse answer for no reason.

**Would change my mind:** nothing. This is the cheapest accuracy win in the whole
system. Where mono is unavoidable, the repair functions
([`asr/diarization.py`](../src/ccma/asr/diarization.py)) target the specific
failures — overlap, speaker flip, over-segmentation — and mark rather than hide
them.

## 5. Quality gating over maximum coverage

**Chose:** `is_kpi_eligible` excludes poor-quality calls from every aggregate, and
publishes how much it excluded.

**Cost:** KPIs are computed on a subset. On noisy traffic that subset can be
meaningfully smaller, and it may be biased — the excluded calls are
disproportionately the hard, noisy, accented ones.

**Why anyway:** a KPI computed over unusable transcripts is worse than a missing
KPI, because someone will act on it. Once an operations director discovers that an
AHT number included calls the system could not transcribe, nothing on the
dashboard is believed again.

**The mitigation, and it is essential:** `agg_quality_daily` publishes the
exclusion, the dashboard warns when eligibility drops below 85%, and
`kpi_eligibility_rate` is itself a monitored KPI. The bias is real; hiding it
would be the actual failure.

## 6. Two escalation models shipped side by side

**Chose:** rule baseline and GBM both scored on every call; the learned model only
adopted if it beats the baseline by ≥0.02 PR-AUC.

**Cost:** two code paths, two things to explain.

**Why anyway:** the rule baseline is what ships on day one before there are
labels, what we fall back to when the model degrades, and what makes "is the model
actually earning its keep" a visible number rather than an argument. A
gradient-boosted model that cannot beat seven weighted rules is not worth
operating, and the comparison scatter is on the dashboard so nobody has to take my
word for it.

**Secondary reason:** supervisors act on these scores. `top_drivers` gives every
score a human-readable reason, and a rule score is trivially explainable when
someone asks why a call was flagged.

## 7. Synthetic corpus over real openly-licensed audio

**Chose:** a 15-scenario template bank expanded deterministically to 30 calls,
with a modelled ASR degradation layer.

**Cost:** it is not real audio. Real accents, real overlapping speech, real
codec artefacts and real hold music are all approximated.

**Why anyway:** openly-licensed *Indian contact-centre* audio with code-switching
does not meaningfully exist. IndicVoices and Shrutilipi are read speech; Common
Voice `hi` is single-speaker prompts. None has agent/customer turn structure, hold
events, or compliance language. Building on read-speech corpora and calling it
call data would be a worse dishonesty than a clearly-labelled synthetic set.

**What the synthetic set buys:** known ground truth for every call, which makes
the calibration in `ccma.quality` possible at all; deterministic tests; and a
degradation model that reproduces the error *types* that matter (romanisation
drift, script flip, near-homophone substitution, hallucination over silence)
rather than random noise.

**Would change my mind:** immediately, given real recordings under an appropriate
agreement. Nothing downstream depends on the corpus module — swapping the loader
is the only change.

## 8. DuckDB over a warehouse

**Chose:** DuckDB locally, with SQL written to port.

**Cost:** single-node. Concurrency and multi-tenant governance are not there.

**Why anyway:** at this data size DuckDB is the *correct* answer, not a
compromise — it reads Silver files directly with no load step and needs no
cluster. The migration path is deliberate: only the five staging views are
dialect-specific, so moving to Snowflake or BigQuery means replacing five view
definitions.

**Trigger to move:** >100 GB Gold, or more than a handful of concurrent dashboard
users, or when row-level multi-tenant security becomes a requirement.

## 9. Streamlit over React

**Chose:** Streamlit.

**Cost:** less polished, weaker interactivity, and it re-runs the script on every
interaction.

**Why anyway:** this is an internal analytics tool. Streamlit shares the Pydantic
models and the DuckDB connection with the pipeline, so there is no API layer to
keep in sync and no drift between "what the pipeline computed" and "what the
dashboard shows". Days instead of weeks.

**Would change my mind:** the moment it is customer-facing or embedded in a
client's portal. Then it is React against the metrics API, and the dashboard views
here become the specification for it.

## 10. Enrichment at the call level, not the turn level

**Chose:** one LLM call per conversation.

**Cost:** long calls must be truncated (head/tail sampled), and per-turn
attributions are the model's own bookkeeping rather than independently derived.

**Why anyway:** sentiment trajectory, resolution status and escalation are
properties of the *whole conversation*. A per-turn approach would cost 10–15× more
and produce worse answers, because no individual turn contains the information.

**The mitigation:** every claim carries `evidence_turn_indices`, so per-turn
attribution is recoverable and — more importantly — *checkable*. A citation
pointing at a turn that does not support the claim is caught by the validator
without a human in the loop.

---

## What I would do next, in order

1. **Real audio.** Everything else is speculation until the models meet genuine
   Indian call recordings.
2. **Fine-tune Whisper on in-domain code-switched audio.** The largest single
   accuracy lever, and the one the whole quality-estimation apparatus exists to
   let us measure honestly.
3. **Distil the intent classifier.** Pseudo-labels from high-confidence agreement
   → a MuRIL model handling the routine majority. Order-of-magnitude cost cut.
4. **Snorkel-style label model.** Once there are >5 weak supervision sources,
   learn their accuracies from their agreement pattern instead of taking a
   disjunction.
5. **Embedding retrieval over the compliance phrase bank.** Closes the paraphrase
   gap without giving the LLM the deciding vote.
6. **Agent-assist in the stream.** The alerting path already computes most of what
   a next-best-action prompt needs; it is the highest-value thing built on this
   foundation.
