"""Shared model base.

`extra="forbid"` is deliberate everywhere in this codebase — an unexpected key
should fail loudly rather than be silently dropped. But it collides with
`@computed_field`: computed fields are *written* by `model_dump()` and *rejected*
on read, so a model that round-trips through JSONL would fail validation on its
own output.

Rather than weaken `extra` (which is the protection we actually want against
hallucinated LLM keys and upstream schema drift), we strip the model's own
computed-field names before validation. Genuinely unknown keys still fail.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, model_validator


class RoundTripModel(BaseModel):
    """Base for models that carry computed fields and are persisted as JSON."""

    @model_validator(mode="before")
    @classmethod
    def _drop_own_computed_fields(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        computed = getattr(cls, "model_computed_fields", None) or {}
        if not computed:
            return data
        return {k: v for k, v in data.items() if k not in computed}
