"""The language layer carries the most product risk, so it gets the most tests."""

from __future__ import annotations

import pytest

from ccma.lang import codeswitch, lid
from ccma.lang.lexicon import is_english_word, is_homograph, is_romanized_hindi
from ccma.lang.script import Script, normalize_devanagari, token_script, tokenize
from ccma.lang.translit import back_transliterate, romanization_score
from ccma.models.enums import LanguageCode


class TestScript:
    def test_devanagari_detected(self):
        assert token_script("नहीं") == Script.DEVANAGARI

    def test_latin_detected(self):
        assert token_script("billing") == Script.LATIN

    def test_tokenizer_drops_punctuation_keeps_both_scripts(self):
        assert tokenize("Sir, मेरा bill galat hai!") == [
            "Sir", "मेरा", "bill", "galat", "hai"
        ]

    def test_nukta_folding_makes_variants_comparable(self):
        # Same word, two legal encodings; without folding these never match.
        assert normalize_devanagari("ज़्यादा") == normalize_devanagari("ज़्यादा")

    def test_chandrabindu_folds_to_anusvara(self):
        assert "ँ" not in normalize_devanagari("हाँ")


class TestLexicon:
    @pytest.mark.parametrize("token", ["nahi", "nahin", "nhi", "bahut", "bohot", "paisa", "shikayat"])
    def test_romanised_hindi_variants_recognised(self, token):
        assert is_romanized_hindi(token)

    @pytest.mark.parametrize("token", ["billing", "refund", "customer", "network"])
    def test_english_recognised(self, token):
        assert is_english_word(token)

    @pytest.mark.parametrize("token", ["to", "me", "is", "us", "bat"])
    def test_homographs_are_not_claimed_by_either_side(self, token):
        assert is_homograph(token)
        assert not is_romanized_hindi(token)
        assert not is_english_word(token)


class TestRomanizationScore:
    @pytest.mark.parametrize("token", ["karenge", "batayiye", "samjhaya", "chahiye"])
    def test_hindi_morphology_scores_high(self, token):
        assert romanization_score(token) >= 0.5

    @pytest.mark.parametrize("token", ["information", "processing", "management", "acknowledgement"])
    def test_english_morphology_scores_low(self, token):
        assert romanization_score(token) < 0.5


class TestLid:
    def test_pure_english(self):
        result = lid.identify("I want to check my bill for this month")
        assert result.lang_primary == LanguageCode.EN

    def test_devanagari_hindi(self):
        result = lid.identify("मेरा बिल बहुत ज़्यादा आ गया है")
        assert result.lang_primary == LanguageCode.HI

    def test_romanised_hindi_is_not_labelled_english(self):
        """The headline failure this project exists to fix."""
        result = lid.identify("mera bill bahut zyada aa gaya hai")
        assert result.lang_primary in (LanguageCode.HI_LATN, LanguageCode.MIXED)
        assert result.lang_primary != LanguageCode.EN

    def test_hinglish_detected_as_mixed(self):
        result = lid.identify(
            "Sir aapka refund process ho gaya hai, next billing cycle mein credit ho jayega"
        )
        assert result.lang_primary in (LanguageCode.MIXED, LanguageCode.HI_LATN)
        assert result.distribution.get("en", 0) > 0

    def test_homograph_resolved_by_hindi_context(self):
        tags = {t.token.lower(): t for t in lid.tag_tokens("to main aapko bata deta hun")}
        assert tags["to"].lang == LanguageCode.HI_LATN

    def test_homograph_resolved_by_english_context(self):
        tags = {t.token.lower(): t for t in lid.tag_tokens("I want to check my account balance")}
        assert tags["to"].lang == LanguageCode.EN

    def test_decoder_prior_does_not_override_strong_lexical_evidence(self):
        """Whisper claiming 'en' on obviously romanised Hindi must not win."""
        result = lid.identify(
            "mujhe apna paisa wapas chahiye bahut dikkat ho rahi hai",
            prior_lang="en",
            prior_prob=0.95,
        )
        assert result.lang_primary != LanguageCode.EN

    def test_empty_input(self):
        assert lid.identify("").lang_primary == LanguageCode.UNKNOWN


class TestTransliteration:
    def test_lexicon_hit_is_exact(self):
        assert back_transliterate("nahi") == "नहीं"

    def test_english_left_alone(self):
        assert back_transliterate("my billing account") == "my billing account"

    def test_mixed_converts_only_hindi_side(self):
        out = back_transliterate("mera bill galat hai")
        assert "bill" in out
        assert "मेरा" in out or "बिल" not in out

    def test_projection_unifies_scripts(self):
        """`paisa kat gaya` and `पैसा कट गया` must converge on one representation."""
        latin = lid.devanagari_projection(lid.normalize_text("paisa kat gaya"), LanguageCode.HI_LATN)
        deva = lid.devanagari_projection(lid.normalize_text("पैसा कट गया"), LanguageCode.HI)
        assert latin is not None and deva is not None
        assert "पैसा" in latin and "पैसा" in deva


class TestCodeSwitch:
    def test_monolingual_scores_zero(self):
        metrics = codeswitch.compute([LanguageCode.EN] * 8)
        assert metrics.cmi == 0.0
        assert metrics.m_index == 0.0
        assert not metrics.is_code_switched

    def test_hindi_scripts_collapse_to_one_language(self):
        """hi and hi_latn are one language in two scripts, not two languages."""
        metrics = codeswitch.compute([LanguageCode.HI, LanguageCode.HI_LATN] * 4)
        assert metrics.cmi == 0.0
        assert not metrics.is_code_switched

    def test_even_mix_maximises_m_index(self):
        metrics = codeswitch.compute([LanguageCode.EN, LanguageCode.HI] * 5)
        assert metrics.m_index == pytest.approx(1.0, abs=0.01)
        assert metrics.cmi == pytest.approx(0.5, abs=0.01)

    def test_switch_point_fraction_separates_clause_from_word_switching(self):
        clause = codeswitch.compute([LanguageCode.EN] * 5 + [LanguageCode.HI] * 5)
        word = codeswitch.compute([LanguageCode.EN, LanguageCode.HI] * 5)
        assert clause.cmi == pytest.approx(word.cmi, abs=0.01)
        assert word.switch_point_fraction > clause.switch_point_fraction

    def test_segment_by_language(self):
        tokens = ["main", "aapko", "refund", "process", "kar", "deta", "hun"]
        langs = [
            LanguageCode.HI_LATN, LanguageCode.HI_LATN, LanguageCode.EN,
            LanguageCode.EN, LanguageCode.HI_LATN, LanguageCode.HI_LATN, LanguageCode.HI_LATN,
        ]
        spans = codeswitch.segment_by_language(tokens, langs)
        assert [lang for lang, _ in spans] == ["hi", "en", "hi"]

    def test_call_rollup_is_token_weighted(self):
        short = codeswitch.compute([LanguageCode.EN, LanguageCode.HI])
        long = codeswitch.compute([LanguageCode.EN] * 40)
        cmi, _, _ = codeswitch.call_level_profile([short, long])
        assert cmi < 0.1  # the 40-token monolingual turn must dominate


class TestNormalization:
    def test_disfluencies_removed(self):
        assert "uh" not in lid.normalize_text("uh I uh need help")

    def test_stutter_collapsed(self):
        assert lid.normalize_text("the the bill").count("the") == 1

    def test_devanagari_preserved(self):
        assert "मेरा" in lid.normalize_text("मेरा बिल")
