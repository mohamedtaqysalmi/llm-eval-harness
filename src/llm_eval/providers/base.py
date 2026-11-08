"""Provider protocol and registry helpers."""

from __future__ import annotations

from typing import Protocol

from llm_eval.models import ModelResponse


class Provider(Protocol):
    name: str

    async def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 512,
    ) -> ModelResponse: ...

    async def aclose(self) -> None: ...
