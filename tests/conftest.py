from __future__ import annotations

import pytest

from ccma.config import Settings
from ccma.ingestion.corpus import generate_corpus


@pytest.fixture(scope="session")
def settings(tmp_path_factory) -> Settings:
    root = tmp_path_factory.mktemp("ccma-data")
    s = Settings(
        data_root=root,
        asr_backend="mock",
        llm_backend="mock",
        tts_backend="none",
        corpus_size=12,
        random_seed=20260731,
        log_level="WARNING",
    )
    s.ensure_dirs()
    return s


@pytest.fixture(scope="session")
def corpus(settings):
    return generate_corpus(n_calls=settings.corpus_size, seed=settings.random_seed)


@pytest.fixture(scope="session")
def transcribed(settings, corpus):
    """One call carried through ASR -> Silver, reused across tests."""
    from ccma.asr.base import TranscriptionRequest, get_backend
    from ccma.silver.build import build_utterances

    backend = get_backend(settings)
    out = []
    for call in corpus[:6]:
        segments = backend.transcribe(TranscriptionRequest(call=call.metadata, audio_path=None))
        utterances, profile = build_utterances(call.metadata, segments)
        out.append((call, segments, utterances, profile))
    return out
