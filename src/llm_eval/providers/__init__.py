"""Model providers (OpenAI-compatible, Ollama, mock)."""

from __future__ import annotations

import os

from llm_eval.providers.base import Provider
from llm_eval.providers.mock import MockProvider
from llm_eval.providers.openai_compat import OpenAICompatProvider


def create_provider(model: str, **kwargs) -> Provider:
    """Create a provider for a model id.

    Conventions:
      - ``mock`` / ``mock:<name>`` → MockProvider
      - ``ollama/<model>`` → OpenAICompatProvider against OLLAMA_BASE_URL
      - anything else → OpenAICompatProvider against OPENAI_BASE_URL
    """
    raw = model.strip()
    lower = raw.lower()

    if lower == "mock" or lower.startswith("mock:"):
        name = raw.split(":", 1)[1] if ":" in raw else "mock"
        return MockProvider(model=name, fail_rate=float(kwargs.get("fail_rate", 0.0)))

    if lower.startswith("ollama/"):
        ollama_model = raw.split("/", 1)[1]
        base = kwargs.get("base_url") or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
        return OpenAICompatProvider(
            ollama_model,
            api_key=kwargs.get("api_key", os.getenv("OLLAMA_API_KEY", "ollama")),
            base_url=base,
            timeout=float(kwargs.get("timeout", 120.0)),
            max_retries=int(kwargs.get("max_retries", 3)),
        )

    return OpenAICompatProvider(
        raw,
        api_key=kwargs.get("api_key"),
        base_url=kwargs.get("base_url"),
        timeout=float(kwargs.get("timeout", 60.0)),
        max_retries=int(kwargs.get("max_retries", 3)),
    )


__all__ = ["Provider", "MockProvider", "OpenAICompatProvider", "create_provider"]
