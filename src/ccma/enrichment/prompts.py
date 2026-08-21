"""Prompt construction for call enrichment.

Prompting decisions that matter for multilingual call data, and why:

* **Do not translate before analysing.** Translating Hinglish to English first
  loses exactly the signal we need — register, politeness level, and the
  code-switch itself (Indian customers switch to Hindi when they get emotional,
  which is a dissatisfaction feature, not noise). We ask the model to reason over
  the original text and to *output* in English so the warehouse stays uniform.
* **Give the model both projections.** Each turn is rendered with its original
  text and, where they differ, the Devanagari projection. A model that only sees
  `paisa kat gaya` may under-weight it; seeing `पैसा कट गया` alongside anchors it.
* **Number every turn.** Every extracted claim must cite `turn_index` values.
  This makes the output auditable, and — more usefully — it makes hallucination
  *detectable*: a citation pointing at a turn that does not support the claim is
  caught by the validator without a human in the loop.
* **Constrain, don't instruct.** Categorical fields go through a JSON schema
  passed to the model's structured-output mode, not through prose asking nicely
  for valid values. Prose constraints fail on small local models.
* **One call, one request.** Per-turn LLM calls would be 10-15x the cost for
  worse results — sentiment trajectory and resolution status are properties of
  the whole conversation.
"""

from __future__ import annotations

from ..models.enums import (
    ComplianceRule,
    DissatisfactionSignal,
    EscalationIndicator,
    ResolutionStatus,
    Sentiment,
)
from ..models.silver import Utterance
from .taxonomy import taxonomy_prompt_block

SYSTEM_PROMPT = """You are a conversation-analytics engine for an Indian contact centre.

You analyse support calls that mix English, Hindi (Devanagari), and romanised \
Hindi/Hinglish written in Latin script. Treat romanised Hindi as Hindi, not as \
English. Never translate the conversation before analysing it; reason over the \
original words, then write your output in English.

Rules you must follow:
1. Output ONLY a single JSON object matching the provided schema. No prose, no \
markdown fences, no commentary.
2. Choose intents ONLY from the taxonomy provided. Never invent a label.
3. Every claim you make must cite the turn indices that support it in the \
corresponding evidence field. If you cannot cite a turn, do not make the claim.
4. Confidence values must be calibrated: use < 0.5 when the transcript is \
ambiguous or the ASR text is garbled. Do not default to 0.9.
5. The transcript comes from automatic speech recognition and contains errors. \
Do not treat a recognition error as a factual statement by the speaker.
6. Judge compliance only on what the AGENT actually said in the transcript.

INTENT TAXONOMY (L1 > L2 [L3 options]):
{taxonomy}
"""

USER_PROMPT = """Analyse the following call and return the JSON object.

CALL METADATA
call_id: {call_id}
queue: {queue}
direction: {direction}
duration_seconds: {duration_s}
hold_count: {hold_count}
transfer_count: {transfer_count}
dominant_language: {dominant_language}
language_mix: {language_mix}
mean_asr_confidence: {mean_asr_confidence}

TRANSCRIPT
{transcript}

FIELD GUIDANCE
- resolution_status: one of {resolutions}
- customer sentiment values: one of {sentiments}
- escalation.indicators: zero or more of {escalation_indicators}
- dissatisfaction_signals[].signal: zero or more of {dissatisfaction_signals}
- compliance[].rule: assess EACH of {compliance_rules}
  Use NOT_APPLICABLE when the call never touched the rule's subject matter.
  Use VIOLATED only for NO_FULL_CARD_READBACK when the agent solicited or \
repeated a full card number or CVV.
- sentiment_trajectory: one entry per CUSTOMER turn where sentiment is \
readable, in ascending turn_index order. score is -1.0..1.0.
- fcr_candidate: true only if the issue appears fully closed on this call with \
no follow-up promised.
- customer_requested_language: set when the customer explicitly asks to switch \
language, else null.
- agent_language_accommodation: true if the agent then switched, false if they \
did not, null if never requested.
"""


def render_transcript(utterances: list[Utterance], max_turns: int = 120) -> str:
    """Render numbered, speaker-tagged turns with language tags.

    On very long calls we keep the head and tail and drop the middle: openings
    carry the disclosure and the stated problem, closings carry the resolution
    and the sign-off, and the middle is usually procedural back-and-forth. This
    beats naive truncation, which silently discards the resolution."""
    if len(utterances) > max_turns:
        head = utterances[: max_turns // 2]
        tail = utterances[-(max_turns // 2) :]
        dropped = len(utterances) - len(head) - len(tail)
        selected = head + tail
        marker_after = len(head)
    else:
        selected = utterances
        dropped = 0
        marker_after = -1

    lines: list[str] = []
    for pos, utt in enumerate(selected):
        if pos == marker_after and dropped:
            lines.append(f"... [{dropped} turns omitted] ...")

        tag = utt.lang_primary.value
        conf_note = "" if utt.asr_confidence >= 0.6 else f" asr_conf={utt.asr_confidence:.2f}"
        flag_note = " LOW_QUALITY" if "hallucination_suspect" in utt.quality_flags else ""

        lines.append(
            f"[{utt.turn_index}] {utt.speaker_role.value} ({tag}{conf_note}{flag_note}) "
            f"@{utt.start_ms // 1000}s: {utt.text_raw}"
        )
        # Second projection only when transliteration actually changed something.
        if utt.text_devanagari and utt.text_devanagari.strip() and utt.lang_primary.value in (
            "hi_latn",
            "mixed",
        ):
            lines.append(f"      [devanagari] {utt.text_devanagari}")

    return "\n".join(lines)


def build_messages(
    call_id: str,
    queue: str,
    direction: str,
    duration_ms: int,
    hold_count: int,
    transfer_count: int,
    dominant_language: str,
    language_mix: dict[str, float],
    mean_asr_confidence: float,
    utterances: list[Utterance],
) -> list[dict[str, str]]:
    system = SYSTEM_PROMPT.format(taxonomy=taxonomy_prompt_block())
    user = USER_PROMPT.format(
        call_id=call_id,
        queue=queue,
        direction=direction,
        duration_s=round(duration_ms / 1000, 1),
        hold_count=hold_count,
        transfer_count=transfer_count,
        dominant_language=dominant_language,
        language_mix=", ".join(f"{k}={v:.2f}" for k, v in sorted(language_mix.items())),
        mean_asr_confidence=f"{mean_asr_confidence:.2f}",
        transcript=render_transcript(utterances),
        resolutions=[r.value for r in ResolutionStatus],
        sentiments=[s.value for s in Sentiment],
        escalation_indicators=[e.value for e in EscalationIndicator],
        dissatisfaction_signals=[d.value for d in DissatisfactionSignal],
        compliance_rules=[c.value for c in ComplianceRule],
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def repair_message(errors: list[str], previous_output: str) -> dict[str, str]:
    """Follow-up turn for the repair loop.

    We show the model its own output and the exact validation errors rather than
    re-asking from scratch: it is cheaper, and on local models it converges far
    more reliably than a bare retry."""
    error_block = "\n".join(f"- {e}" for e in errors[:15])
    return {
        "role": "user",
        "content": (
            "Your previous response failed schema validation.\n\n"
            f"ERRORS:\n{error_block}\n\n"
            f"YOUR PREVIOUS OUTPUT:\n{previous_output[:4000]}\n\n"
            "Return the corrected JSON object only. Fix every listed error. "
            "Do not add fields that are not in the schema. Do not include markdown fences."
        ),
    }
