"""Bronze -> Silver: conformed, language-resolved conversation.

Per call this stage:
  1. groups ASR segments into speaker turns (channel-based, with repair),
  2. normalises text and resolves language per token and per utterance,
  3. projects Hindi content to a single Devanagari representation,
  4. computes code-switch metrics at utterance and call level,
  5. attaches an ASR confidence and quality flags to every row.

Everything here is deterministic and model-free apart from the optional decoder
prior, which keeps Silver cheap to rebuild — an important property, because a
lexicon update should trigger a Silver rebuild without re-running ASR.
"""

from __future__ import annotations

from ..lang import codeswitch, lid
from ..logging_conf import get_logger
from ..models.bronze import AsrSegment, CallMetadata
from ..models.enums import LanguageCode, SpeakerRole
from ..models.silver import ConversationLanguageProfile, TokenLanguage, Utterance
from ..asr.diarization import SpeakerTurn, assign_roles_from_channel, build_turns

log = get_logger(__name__)

# Utterance-level quality flags. Kept as strings (not an enum) because this list
# grows often and a new flag must not require a schema migration.
FLAG_LOW_ASR_CONFIDENCE = "low_asr_confidence"
FLAG_HALLUCINATION_SUSPECT = "hallucination_suspect"
FLAG_OVERLAPPED_SPEECH = "overlapped_speech"
FLAG_VERY_SHORT = "very_short_utterance"
FLAG_NO_LEXICAL_CONTENT = "no_lexical_content"
FLAG_LANG_UNCERTAIN = "language_uncertain"
FLAG_HIGH_SWITCH_RATE = "high_switch_rate"


def _segment_confidence(seg: AsrSegment) -> float:
    """Fuse the decoder's signals into one 0-1 confidence.

    Word probabilities are the primary term when present. avg_logprob is the
    fallback, mapped through exp() since it is a mean log-probability. Both are
    then penalised by no_speech_prob, which is the strongest single predictor of
    a hallucinated segment."""
    if seg.words:
        base = sum(w.probability for w in seg.words) / len(seg.words)
    elif seg.avg_logprob is not None:
        base = min(1.0, max(0.0, 2.718281828 ** seg.avg_logprob))
    else:
        base = 0.5

    if seg.no_speech_prob is not None:
        base *= 1.0 - min(0.85, seg.no_speech_prob)
    return round(max(0.0, min(1.0, base)), 4)


def _is_hallucination(seg: AsrSegment, confidence: float) -> bool:
    """Whisper hallucinations over silence share a signature: high no_speech_prob,
    low avg_logprob, and an extreme compression ratio (repetitive text) — plus
    they are short. Any two of the three is enough to flag."""
    votes = 0
    if (seg.no_speech_prob or 0) > 0.5:
        votes += 1
    if (seg.avg_logprob or 0) < -1.0:
        votes += 1
    cr = seg.compression_ratio
    if cr is not None and (cr > 2.4 or cr < 0.9):
        votes += 1
    return votes >= 2 and confidence < 0.55


def _turn_confidence(turn: SpeakerTurn, segments_by_id: dict[int, AsrSegment]) -> float:
    """Duration-weighted mean of the constituent segments' confidences."""
    segs = [segments_by_id[i] for i in turn.segment_ids if i in segments_by_id]
    if not segs:
        return 0.5
    total_ms = sum(max(1, s.end_ms - s.start_ms) for s in segs)
    return round(
        sum(_segment_confidence(s) * max(1, s.end_ms - s.start_ms) for s in segs) / total_ms, 4
    )


def _dominant_prior(turn: SpeakerTurn, segments_by_id: dict[int, AsrSegment]) -> tuple[str | None, float | None]:
    """Longest constituent segment's decoder LID guess, used as the fusion prior."""
    segs = [segments_by_id[i] for i in turn.segment_ids if i in segments_by_id]
    if not segs:
        return None, None
    longest = max(segs, key=lambda s: s.end_ms - s.start_ms)
    return longest.whisper_lang, longest.whisper_lang_prob


def build_utterances(
    call: CallMetadata, segments: list[AsrSegment]
) -> tuple[list[Utterance], ConversationLanguageProfile]:
    segments_by_id = {s.segment_id: s for s in segments}
    role_by_channel = assign_roles_from_channel(segments)
    turns = build_turns(segments, role_by_channel, call.call_id)

    utterances: list[Utterance] = []
    per_utterance_metrics: list[codeswitch.CodeSwitchMetrics] = []

    for turn in turns:
        normalized = lid.normalize_text(turn.text)
        prior_lang, prior_prob = _dominant_prior(turn, segments_by_id)
        result = lid.identify(normalized, prior_lang=prior_lang, prior_prob=prior_prob)

        cs = codeswitch.compute([t.lang for t in result.tokens])
        per_utterance_metrics.append(cs)

        asr_conf = _turn_confidence(turn, segments_by_id)
        flags = list(dict.fromkeys(turn.flags))

        constituent = [segments_by_id[i] for i in turn.segment_ids if i in segments_by_id]
        if any(_is_hallucination(s, _segment_confidence(s)) for s in constituent):
            flags.append(FLAG_HALLUCINATION_SUSPECT)
        if asr_conf < 0.5:
            flags.append(FLAG_LOW_ASR_CONFIDENCE)
        if turn.overlapped_with_next:
            flags.append(FLAG_OVERLAPPED_SPEECH)
        if (turn.end_ms - turn.start_ms) < 400:
            flags.append(FLAG_VERY_SHORT)
        if not result.tokens:
            flags.append(FLAG_NO_LEXICAL_CONTENT)
        if result.confidence < 0.55:
            flags.append(FLAG_LANG_UNCERTAIN)
        if cs.switch_point_fraction > 0.4:
            flags.append(FLAG_HIGH_SWITCH_RATE)

        utterances.append(
            Utterance(
                call_id=call.call_id,
                turn_index=turn.turn_index,
                segment_ids=turn.segment_ids,
                speaker_role=turn.speaker_role,
                speaker_id=turn.speaker_id,
                start_ms=turn.start_ms,
                end_ms=turn.end_ms,
                text_raw=turn.text,
                text_normalized=normalized,
                text_devanagari=lid.devanagari_projection(normalized, result.lang_primary),
                lang_primary=result.lang_primary,
                lang_confidence=result.confidence,
                lang_distribution=result.distribution,
                token_langs=[
                    TokenLanguage(
                        token=t.token,
                        lang=t.lang,
                        confidence=t.confidence,
                        is_transliterated=t.is_transliterated,
                    )
                    for t in result.tokens
                ],
                is_code_switched=cs.is_code_switched,
                cmi=cs.cmi,
                switch_points=cs.switch_points,
                asr_confidence=asr_conf,
                quality_flags=sorted(set(flags)),
                diarization_confidence=turn.confidence,
                overlapped_with_next=turn.overlapped_with_next,
            )
        )

    profile = _language_profile(call.call_id, utterances, per_utterance_metrics)
    return utterances, profile


def _language_profile(
    call_id: str,
    utterances: list[Utterance],
    metrics: list[codeswitch.CodeSwitchMetrics],
) -> ConversationLanguageProfile:
    cmi, m_index, spf = codeswitch.call_level_profile(metrics)

    total_tokens = sum(len(u.token_langs) for u in utterances)
    share: dict[str, float] = {}
    translit_tokens = 0
    for utt in utterances:
        for tok in utt.token_langs:
            share[tok.lang.value] = share.get(tok.lang.value, 0.0) + 1
            if tok.is_transliterated:
                translit_tokens += 1
    if total_tokens:
        share = {k: round(v / total_tokens, 4) for k, v in share.items()}

    dominant = (
        LanguageCode(max(share, key=lambda k: share[k])) if share else LanguageCode.UNKNOWN
    )

    return ConversationLanguageProfile(
        call_id=call_id,
        dominant_language=dominant,
        language_share=share,
        call_cmi=cmi,
        m_index=m_index,
        switch_point_fraction=spf,
        agent_customer_language_match=_language_match(utterances),
        transliterated_token_ratio=round(translit_tokens / total_tokens, 4) if total_tokens else 0.0,
    )


def _language_match(utterances: list[Utterance]) -> float:
    """How well the agent mirrored the customer's language choice.

    This is a genuine quality metric in Indian contact centres — an agent
    answering Hindi in English is a common, measurable CSAT killer, and it is
    invisible to any monolingual analytics stack."""
    def _collapse(lang: LanguageCode) -> str:
        if lang in (LanguageCode.HI, LanguageCode.HI_LATN):
            return "hi"
        if lang == LanguageCode.EN:
            return "en"
        return "mixed"

    agent = [_collapse(u.lang_primary) for u in utterances if u.speaker_role == SpeakerRole.AGENT]
    customer = [
        _collapse(u.lang_primary) for u in utterances if u.speaker_role == SpeakerRole.CUSTOMER
    ]
    if not agent or not customer:
        return 1.0

    def _dist(seq: list[str]) -> dict[str, float]:
        out: dict[str, float] = {}
        for x in seq:
            out[x] = out.get(x, 0.0) + 1
        return {k: v / len(seq) for k, v in out.items()}

    a_dist, c_dist = _dist(agent), _dist(customer)
    # Bhattacharyya-style overlap: 1.0 = identical language mix, 0.0 = disjoint.
    overlap = sum(min(a_dist.get(k, 0.0), c_dist.get(k, 0.0)) for k in set(a_dist) | set(c_dist))
    return round(overlap, 4)
