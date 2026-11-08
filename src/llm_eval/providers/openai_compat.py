"""OpenAI-compatible chat completions provider (OpenAI, Ollama, Groq, etc.)."""

from __future__ import annotations

import asyncio
import os
import time
from typing import Any

import httpx

from llm_eval.models import ModelResponse


class OpenAICompatProvider:
    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 60.0,
        max_retries: int = 3,
        retry_backoff: float = 0.8,
    ) -> None:
        self.model = model
        self.name = model
        self.api_key = api_key if api_key is not None else os.getenv("OPENAI_API_KEY", "")
        self.base_url = (
            base_url or os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1"
        ).rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_backoff = retry_backoff
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        self._client = httpx.AsyncClient(base_url=self.base_url, headers=headers, timeout=timeout)

    async def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 512,
    ) -> ModelResponse:
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        last_err: str | None = None
        t0 = time.perf_counter()
        for attempt in range(self.max_retries + 1):
            try:
                resp = await self._client.post("/chat/completions", json=payload)
                if resp.status_code in {429, 500, 502, 503, 504}:
                    last_err = f"HTTP {resp.status_code}: {resp.text[:200]}"
                    if attempt < self.max_retries:
                        await asyncio.sleep(self.retry_backoff * (2**attempt))
                        continue
                    latency = (time.perf_counter() - t0) * 1000
                    return ModelResponse(
                        text="",
                        latency_ms=latency,
                        model=self.model,
                        error=last_err,
                    )
                resp.raise_for_status()
                data: dict[str, Any] = resp.json()
                choice = (data.get("choices") or [{}])[0]
                message = choice.get("message") or {}
                text = str(message.get("content") or "")
                usage = data.get("usage") or {}
                latency = (time.perf_counter() - t0) * 1000
                prompt_tokens = int(usage.get("prompt_tokens") or 0)
                completion_tokens = int(usage.get("completion_tokens") or 0)
                total_tokens = int(usage.get("total_tokens") or (prompt_tokens + completion_tokens))
                return ModelResponse(
                    text=text,
                    latency_ms=latency,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=total_tokens,
                    model=self.model,
                    raw={"id": data.get("id"), "usage": usage},
                )
            except (httpx.HTTPError, ValueError, KeyError) as exc:
                last_err = str(exc)
                if attempt < self.max_retries:
                    await asyncio.sleep(self.retry_backoff * (2**attempt))
                    continue
                latency = (time.perf_counter() - t0) * 1000
                return ModelResponse(
                    text="",
                    latency_ms=latency,
                    model=self.model,
                    error=last_err,
                )

        latency = (time.perf_counter() - t0) * 1000
        return ModelResponse(
            text="", latency_ms=latency, model=self.model, error=last_err or "unknown"
        )

    async def aclose(self) -> None:
        await self._client.aclose()
