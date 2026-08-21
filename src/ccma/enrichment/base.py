"""LLM backend interface + the Ollama client.

Ollama (local open-weight models) rather than a hosted API. Rationale for this
workload:

* **Data residency.** Call recordings contain customer PII and, under India's
  DPDP Act, moving them to a third-party US endpoint is a compliance
  conversation nobody wants to have for a batch enrichment job. Local inference
  removes the question entirely.
* **Unit economics.** Enrichment is high-volume and latency-tolerant — the exact
  shape where a self-hosted 7B on an owned/reserved GPU beats per-token pricing
  by a wide margin. Cost becomes GPU-hours, which we control by batching.
* **No rate limits on backfill.** Reprocessing a year of calls after a prompt
  change is a scheduling problem, not a quota negotiation.

The trade-off is real and stated plainly: a 7B local model is weaker at nuanced
multilingual reasoning than a frontier hosted model. We compensate with (a) a
constrained JSON schema, (b) the deterministic compliance detector doing the
work that actually needs precision, (c) a two-tier escalation to a larger local
model on low confidence, and (d) LLM-as-judge sampling to measure the gap. The
`LlmBackend` interface means swapping in a hosted provider is one class.
"""

from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from ..config import Settings
from ..logging_conf import get_logger

log = get_logger(__name__)


@dataclass
class LlmResponse:
    text: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: int = 0
    raw: dict[str, Any] = field(default_factory=dict)


class LlmBackend(ABC):
    name: str = "base"

    @abstractmethod
    def chat(
        self,
        messages: list[dict[str, str]],
        json_schema: dict[str, Any] | None = None,
        model: str | None = None,
        temperature: float = 0.0,
    ) -> LlmResponse: ...

    def embed(self, texts: list[str], model: str | None = None) -> list[list[float]]:
        raise NotImplementedError

    def health(self) -> bool:
        return True


class OllamaBackend(LlmBackend):
    name = "ollama"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._client = httpx.Client(
            base_url=settings.ollama_host.rstrip("/"),
            timeout=httpx.Timeout(settings.ollama_timeout_s, connect=10.0),
        )

    def health(self) -> bool:
        try:
            resp = self._client.get("/api/tags")
            return resp.status_code == 200
        except httpx.HTTPError:
            return False

    def available_models(self) -> list[str]:
        try:
            resp = self._client.get("/api/tags")
            resp.raise_for_status()
            return [m["name"] for m in resp.json().get("models", [])]
        except httpx.HTTPError:
            return []

    @retry(
        retry=retry_if_exception_type((httpx.HTTPError, json.JSONDecodeError)),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        stop=stop_after_attempt(3),
        reraise=True,
    )
    def chat(
        self,
        messages: list[dict[str, str]],
        json_schema: dict[str, Any] | None = None,
        model: str | None = None,
        temperature: float = 0.0,
    ) -> LlmResponse:
        model = model or self.settings.ollama_model
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_ctx": self.settings.ollama_num_ctx,
                # Deterministic decoding: enrichment must be reproducible for a
                # given input, otherwise re-runs churn the warehouse.
                "seed": 42,
                "top_p": 0.9,
            },
            # Ollama structured outputs: constrains generation to the schema
            # rather than hoping the prompt is obeyed. This is the single biggest
            # reliability win when using small local models for extraction.
            "keep_alive": "10m",
        }
        if json_schema is not None:
            payload["format"] = json_schema

        started = time.perf_counter()
        resp = self._client.post("/api/chat", json=payload)
        resp.raise_for_status()
        body = resp.json()
        elapsed_ms = int((time.perf_counter() - started) * 1000)

        return LlmResponse(
            text=body.get("message", {}).get("content", ""),
            model=model,
            prompt_tokens=body.get("prompt_eval_count", 0),
            completion_tokens=body.get("eval_count", 0),
            latency_ms=elapsed_ms,
            raw=body,
        )

    def embed(self, texts: list[str], model: str | None = None) -> list[list[float]]:
        model = model or self.settings.ollama_embed_model
        resp = self._client.post("/api/embed", json={"model": model, "input": texts})
        resp.raise_for_status()
        return resp.json().get("embeddings", [])

    def close(self) -> None:
        self._client.close()


def get_llm_backend(settings: Settings | None = None) -> LlmBackend:
    from ..config import get_settings

    settings = settings or get_settings()
    if settings.llm_backend == "ollama":
        backend = OllamaBackend(settings)
        if not backend.health():
            raise RuntimeError(
                f"Ollama is not reachable at {settings.ollama_host}. "
                "Start it with `ollama serve`, or set CCMA_LLM_BACKEND=mock."
            )
        return backend
    if settings.llm_backend == "mock":
        from .mock_backend import MockLlmBackend

        return MockLlmBackend(settings)
    raise ValueError(f"unknown LLM backend: {settings.llm_backend}")
