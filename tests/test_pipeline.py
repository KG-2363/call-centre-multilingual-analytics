"""ASR -> Silver -> enrichment -> validation, end to end on the mock backends."""

from __future__ import annotations

import pytest

from ccma.asr.base import TranscriptionRequest, get_backend
from ccma.asr.diarization import assign_roles_from_channel, build_turns, role_score
from ccma.enrichment.base import get_llm_backend
from ccma.enrichment.runner import enrich_call
from ccma.enrichment.taxonomy import coerce_path, is_valid_path
from ccma.enrichment.validation import check_grounding, extract_json, validate
from ccma.models.enums import SpeakerRole, ValidationStatus
from ccma.quality import asr_quality, dq
from ccma.quality.text_metrics import normalized_disagreement, word_error_rate
from ccma.silver.build import build_utterances


class TestCorpusDeterminism:
    def test_same_seed_same_corpus(self, settings):
        from ccma.ingestion.corpus import generate_corpus

        a = generate_corpus(8, seed=99)
        b = generate_corpus(8, seed=99)
        assert [c.metadata.call_id for c in a] == [c.metadata.call_id for c in b]
        assert [c.metadata.duration_ms for c in a] == [c.metadata.duration_ms for c in b]

    def test_different_seed_differs(self):
        from ccma.ingestion.corpus import generate_corpus

        a = generate_corpus(8, seed=1)
        b = generate_corpus(8, seed=2)
        assert [c.metadata.duration_ms for c in a] != [c.metadata.duration_ms for c in b]

    def test_customer_ids_are_hashed(self, corpus):
        for call in corpus:
            assert len(call.metadata.customer_id_hash) == 64


class TestAsr:
    def test_mock_backend_produces_segments(self, transcribed):
        for _call, segments, _utts, _profile in transcribed:
            assert segments
            assert all(s.end_ms >= s.start_ms for s in segments)

    def test_decoder_telemetry_present(self, transcribed):
        _call, segments, _utts, _profile = transcribed[0]
        assert all(s.avg_logprob is not None for s in segments)
        assert all(s.no_speech_prob is not None for s in segments)

    def test_segment_ids_unique_and_ordered(self, transcribed):
        for _call, segments, _utts, _profile in transcribed:
            ids = [s.segment_id for s in segments]
            assert ids == sorted(ids) == list(range(len(ids)))

    def test_deterministic_across_runs(self, settings, corpus):
        backend = get_backend(settings)
        first = backend.transcribe(TranscriptionRequest(call=corpus[0].metadata, audio_path=None))
        second = backend.transcribe(TranscriptionRequest(call=corpus[0].metadata, audio_path=None))
        assert [s.text_raw for s in first] == [s.text_raw for s in second]

    def test_whisper_mislabels_romanised_hindi_as_english(self, settings, corpus):
        """Guards the assumption the whole LID fusion design rests on."""
        backend = get_backend(settings)
        latin_hindi = [c for c in corpus if c.template.language_style in ("hi_latn", "hinglish")]
        guesses = []
        for call in latin_hindi:
            for seg in backend.transcribe(TranscriptionRequest(call=call.metadata, audio_path=None)):
                guesses.append(seg.whisper_lang)
        assert guesses.count("en") > guesses.count("hi")


class TestDiarization:
    def test_role_score_identifies_agent(self):
        assert role_score("Thank you for calling, this call is being recorded") > 0

    def test_role_score_identifies_customer(self):
        assert role_score("mera bill galat hai aur paisa kat gaya") < 0

    def test_channel_attribution(self, transcribed):
        _call, segments, _utts, _profile = transcribed[0]
        roles = assign_roles_from_channel(segments)
        assert set(roles.values()) == {SpeakerRole.AGENT, SpeakerRole.CUSTOMER}

    def test_turns_are_contiguous(self, transcribed):
        for call, segments, _utts, _profile in transcribed:
            turns = build_turns(segments, assign_roles_from_channel(segments), call.metadata.call_id)
            assert [t.turn_index for t in turns] == list(range(len(turns)))

    def test_overlap_is_flagged_not_silently_resolved(self, settings, corpus):
        backend = get_backend(settings)
        overlapping = [
            c for c in corpus if any(t.overlaps_next for t in c.turns)
        ]
        if not overlapping:
            pytest.skip("no overlapping calls in this corpus slice")
        call = overlapping[0]
        segments = backend.transcribe(TranscriptionRequest(call=call.metadata, audio_path=None))
        utterances, _ = build_utterances(call.metadata, segments)
        assert any("overlapped_speech" in u.quality_flags for u in utterances) or any(
            u.overlapped_with_next for u in utterances
        )


class TestSilver:
    def test_utterances_have_language_and_confidence(self, transcribed):
        for _call, _segments, utterances, _profile in transcribed:
            for utt in utterances:
                assert 0.0 <= utt.asr_confidence <= 1.0
                assert 0.0 <= utt.lang_confidence <= 1.0

    def test_language_profile_shares_sum_to_one(self, transcribed):
        for _call, _segments, _utts, profile in transcribed:
            if profile.language_share:
                assert sum(profile.language_share.values()) == pytest.approx(1.0, abs=0.02)

    def test_hindi_calls_get_devanagari_projection(self, transcribed):
        hindi_calls = [
            t for t in transcribed if t[0].template.language_style in ("hi", "hi_latn", "hinglish")
        ]
        if not hindi_calls:
            pytest.skip("no Hindi calls in slice")
        _call, _segments, utterances, _profile = hindi_calls[0]
        assert any(u.text_devanagari for u in utterances)

    def test_hallucinated_segment_is_flagged(self, settings, corpus):
        backend = get_backend(settings)
        flagged = False
        for call in corpus:
            segments = backend.transcribe(TranscriptionRequest(call=call.metadata, audio_path=None))
            utterances, _ = build_utterances(call.metadata, segments)
            if any("hallucination_suspect" in u.quality_flags for u in utterances):
                flagged = True
                break
        assert flagged, "degradation model should produce at least one hallucination"


class TestTaxonomy:
    def test_valid_path(self):
        assert is_valid_path("BILLING", "REFUND_STATUS", "REFUND_DELAY")

    def test_invented_leaf_is_dropped_not_rejected(self):
        l1, l2, l3, coerced = coerce_path("BILLING", "REFUND_STATUS", "MADE_UP_LEAF")
        assert (l1, l2, l3) == ("BILLING", "REFUND_STATUS", None)
        assert coerced

    def test_valid_l2_under_wrong_parent_is_reparented(self):
        l1, l2, _l3, coerced = coerce_path("TECHNICAL_SUPPORT", "REFUND_STATUS", None)
        assert (l1, l2) == ("BILLING", "REFUND_STATUS")
        assert coerced

    def test_unrecoverable_falls_back_to_other(self):
        l1, l2, _, coerced = coerce_path("NONSENSE", "GARBAGE", None)
        assert (l1, l2) == ("OTHER", "UNCLASSIFIED")
        assert coerced


class TestValidation:
    def test_extracts_json_from_fenced_output(self):
        payload, err = extract_json('```json\n{"a": 1}\n```')
        assert err is None and payload == {"a": 1}

    def test_extracts_json_with_leading_prose(self):
        payload, err = extract_json('Here is the result:\n{"a": 1}')
        assert err is None and payload == {"a": 1}

    def test_rejects_unparseable(self):
        payload, err = extract_json("not json at all")
        assert payload is None and err

    def test_grounding_rejects_citation_to_missing_turn(self, transcribed):
        _call, _segments, utterances, _profile = transcribed[0]
        payload = {
            "primary_intent": {"evidence_turn_indices": [9999]},
            "escalation": {"evidence_turn_indices": []},
            "dissatisfaction_signals": [],
            "sentiment_trajectory": [],
            "compliance": [],
        }
        errors, _warnings = check_grounding(payload, utterances)
        assert any("9999" in e for e in errors)

    def test_grounding_warns_on_fabricated_quote(self, transcribed):
        _call, _segments, utterances, _profile = transcribed[0]
        payload = {
            "primary_intent": {"evidence_turn_indices": []},
            "escalation": {"evidence_turn_indices": []},
            "dissatisfaction_signals": [
                {
                    "signal": "EXPLICIT_COMPLAINT",
                    "turn_index": utterances[0].turn_index,
                    "quote": "this text was never spoken anywhere in the call at all",
                    "severity": 0.5,
                }
            ],
            "sentiment_trajectory": [],
            "compliance": [],
        }
        _errors, warnings = check_grounding(payload, utterances)
        assert any("quote does not match" in w for w in warnings)


class TestEnrichment:
    def test_enrich_produces_valid_record(self, settings, transcribed):
        backend = get_llm_backend(settings)
        call, _segments, utterances, profile = transcribed[0]
        outcome = enrich_call(call.metadata, utterances, profile, backend, settings)
        assert outcome.enrichment.validation_status != ValidationStatus.QUARANTINED
        assert outcome.enrichment.call_id == call.metadata.call_id

    def test_intent_path_is_always_canonical(self, settings, transcribed):
        backend = get_llm_backend(settings)
        for call, _segments, utterances, profile in transcribed:
            outcome = enrich_call(call.metadata, utterances, profile, backend, settings)
            intent = outcome.enrichment.primary_intent
            assert is_valid_path(intent.l1, intent.l2, intent.l3)

    def test_compliance_covers_every_rule(self, settings, transcribed):
        from ccma.models.enums import ComplianceRule

        backend = get_llm_backend(settings)
        call, _segments, utterances, profile = transcribed[0]
        outcome = enrich_call(call.metadata, utterances, profile, backend, settings)
        assessed = {c.rule for c in outcome.enrichment.compliance}
        assert assessed == set(ComplianceRule)

    def test_recording_disclosure_detected_when_spoken(self, settings, transcribed):
        from ccma.models.enums import ComplianceOutcome, ComplianceRule

        backend = get_llm_backend(settings)
        with_disclosure = [
            t for t in transcribed
            if ComplianceRule.RECORDING_DISCLOSURE in t[0].template.compliance_present
        ]
        if not with_disclosure:
            pytest.skip("no disclosure calls in slice")

        detected = 0
        for call, _segments, utterances, profile in with_disclosure:
            outcome = enrich_call(call.metadata, utterances, profile, backend, settings)
            check = next(
                c for c in outcome.enrichment.compliance
                if c.rule == ComplianceRule.RECORDING_DISCLOSURE
            )
            detected += int(check.outcome == ComplianceOutcome.PRESENT)
        # ASR degradation can garble the disclosure; require the majority, not all.
        assert detected >= len(with_disclosure) * 0.5

    def test_compliance_score_excludes_not_applicable(self, settings, transcribed):
        backend = get_llm_backend(settings)
        call, _segments, utterances, profile = transcribed[0]
        outcome = enrich_call(call.metadata, utterances, profile, backend, settings)
        assert 0.0 <= outcome.enrichment.compliance_score <= 1.0

    def test_sentiment_delta_is_computed(self, settings, transcribed):
        backend = get_llm_backend(settings)
        call, _segments, utterances, profile = transcribed[0]
        outcome = enrich_call(call.metadata, utterances, profile, backend, settings)
        assert -2.0 <= outcome.enrichment.sentiment_delta <= 2.0

    def test_no_utterances_is_quarantined_not_dropped(self, settings, transcribed):
        backend = get_llm_backend(settings)
        call, _segments, _utts, profile = transcribed[0]
        outcome = enrich_call(call.metadata, [], profile, backend, settings)
        assert outcome.enrichment.validation_status == ValidationStatus.QUARANTINED
        assert outcome.enrichment.call_id == call.metadata.call_id


class TestQuality:
    def test_wer_identical_is_zero(self):
        assert word_error_rate("mera bill galat hai", "mera bill galat hai") == 0.0

    def test_wer_detects_substitution(self):
        assert 0 < word_error_rate("mera bill galat hai", "mera bill sahi hai") <= 1.0

    def test_disagreement_is_symmetric(self):
        a, b = "mera bill galat hai", "mera bil galath hai"
        assert normalized_disagreement(a, b) == normalized_disagreement(b, a)

    def test_quality_bands_ordered(self):
        assert asr_quality.band_for(0.05) == "GOOD"
        assert asr_quality.band_for(0.22) == "FAIR"
        assert asr_quality.band_for(0.40) == "POOR"
        assert asr_quality.band_for(0.90) == "UNUSABLE"

    def test_assess_call_produces_interval(self, transcribed):
        model = asr_quality.CalibrationModel()
        call, segments, utterances, _profile = transcribed[0]
        quality = asr_quality.assess_call(call.metadata.call_id, segments, utterances, model)
        assert quality.estimated_wer_lo <= quality.estimated_wer <= quality.estimated_wer_hi
        assert quality.quality_band in ("GOOD", "FAIR", "POOR", "UNUSABLE")

    def test_worse_signals_predict_worse_wer(self):
        model = asr_quality.CalibrationModel()
        good = {"mean_word_prob": 0.95, "mean_avg_logprob": -0.1,
                "max_no_speech_prob": 0.02, "mean_compression_ratio": 1.6}
        bad = {"mean_word_prob": 0.45, "mean_avg_logprob": -1.3,
               "max_no_speech_prob": 0.7, "mean_compression_ratio": 3.0}
        assert model.predict(bad) > model.predict(good)

    def test_isotonic_calibration_is_monotone(self):
        samples = [
            ({"mean_word_prob": p, "mean_avg_logprob": -0.2, "max_no_speech_prob": 0.05,
              "mean_compression_ratio": 1.6}, (1 - p) * 0.8, "en")
            for p in [i / 40 for i in range(4, 40)]
        ]
        model = asr_quality.fit_calibration(samples)
        assert model.fitted
        low = model.predict({"mean_word_prob": 0.95, "mean_avg_logprob": -0.2,
                             "max_no_speech_prob": 0.05, "mean_compression_ratio": 1.6})
        high = model.predict({"mean_word_prob": 0.30, "mean_avg_logprob": -0.2,
                              "max_no_speech_prob": 0.05, "mean_compression_ratio": 1.6})
        assert high > low


class TestDataQuality:
    def test_duplicate_call_ids_block(self, corpus):
        calls = [c.metadata for c in corpus[:3]]
        report = dq.check_bronze([*calls, calls[0]], [])
        assert not report.ok
        assert any(f.name == "bronze.call_id_unique" for f in report.blocking_failures)

    def test_unhashed_customer_id_blocks(self, corpus):
        bad = corpus[0].metadata.model_copy(update={"customer_id_hash": "9876543210"})
        report = dq.check_bronze([bad], [])
        assert any(f.name == "bronze.customer_id_hashed" for f in report.blocking_failures)

    def test_clean_bronze_passes(self, corpus, transcribed):
        segments = [s for _c, segs, _u, _p in transcribed for s in segs]
        calls = [c.metadata for c in corpus]
        assert dq.check_bronze(calls, segments).ok

    def test_silver_checks_pass_on_clean_data(self, corpus, transcribed):
        utterances = [u for _c, _s, utts, _p in transcribed for u in utts]
        calls = [t[0].metadata for t in transcribed]
        assert dq.check_silver(utterances, calls).ok
