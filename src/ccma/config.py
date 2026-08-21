"""Central configuration. All knobs are env-driven so the same image runs in
local-mock, staging and production without code changes."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="CCMA_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    data_root: Path = Path("./data")

    asr_backend: str = "mock"
    whisper_model: str = "large-v3"
    whisper_device: str = "auto"
    whisper_compute_type: str = "int8"
    whisper_shadow_model: str = "small"

    llm_backend: str = "mock"
    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5:7b-instruct"
    ollama_model_heavy: str = "qwen2.5:14b-instruct"
    ollama_model_judge: str = "gemma2:9b"
    ollama_embed_model: str = "bge-m3"
    ollama_timeout_s: int = 180
    ollama_num_ctx: int = 8192
    llm_max_retries: int = 3
    llm_concurrency: int = 4

    tts_backend: str = "none"

    corpus_size: int = 30
    random_seed: int = 20260731

    log_level: str = "INFO"
    log_json: bool = False

    # Enrichment below this confidence is re-run on the heavy model.
    escalate_to_heavy_below: float = Field(0.55, ge=0.0, le=1.0)
    # Utterances below this ASR confidence are excluded from KPI aggregates.
    min_asr_confidence_for_kpi: float = Field(0.40, ge=0.0, le=1.0)

    @property
    def bronze_dir(self) -> Path:
        return self.data_root / "bronze"

    @property
    def silver_dir(self) -> Path:
        return self.data_root / "silver"

    @property
    def gold_dir(self) -> Path:
        return self.data_root / "gold"

    @property
    def audio_dir(self) -> Path:
        return self.data_root / "audio"

    @property
    def warehouse_path(self) -> Path:
        return self.data_root / "warehouse.duckdb"

    def ensure_dirs(self) -> None:
        for d in (self.bronze_dir, self.silver_dir, self.gold_dir, self.audio_dir):
            d.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
