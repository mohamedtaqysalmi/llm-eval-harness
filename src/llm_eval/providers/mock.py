"""Deterministic offline mock provider for tests and demos."""

from __future__ import annotations

import hashlib
import json
import re
import time
from typing import Any

from llm_eval.models import ModelResponse


class MockProvider:
    """Produces predictable answers for sample suites without network I/O."""

    name = "mock"

    def __init__(self, model: str = "mock", *, fail_rate: float = 0.0) -> None:
        self.model = model
        self.fail_rate = fail_rate

    async def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 512,
    ) -> ModelResponse:
        t0 = time.perf_counter()
        digest = int(hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:8], 16)
        if self.fail_rate > 0 and (digest % 1000) / 1000 < self.fail_rate:
            latency = (time.perf_counter() - t0) * 1000
            return ModelResponse(
                text="",
                latency_ms=latency,
                model=self.model,
                error="mock induced failure",
            )

        text = self._answer(prompt, system)
        # Fake token counts from length
        prompt_tokens = max(1, len(prompt.split()))
        completion_tokens = max(1, len(text.split()))
        latency = (time.perf_counter() - t0) * 1000 + (digest % 17)
        return ModelResponse(
            text=text[: max_tokens * 8],
            latency_ms=latency,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            model=self.model,
            raw={"provider": "mock", "system": system, "temperature": temperature},
        )

    async def aclose(self) -> None:
        return None

    def _answer(self, prompt: str, system: str | None) -> str:
        p = prompt.strip()
        lower = p.lower()

        # LLM-as-judge prompts (used offline with --judge-model mock)
        if "you are an evaluation judge" in lower:
            return self._judge(p)

        # Explicit expected markers used in suites for deterministic offline demos
        m = re.search(r"\[MOCK_ANSWER:\s*(.*?)\]", p, flags=re.DOTALL)
        if m:
            return m.group(1).strip()

        m = re.search(r"\[MOCK_JSON:\s*(\{.*?\})\]", p, flags=re.DOTALL)
        if m:
            return m.group(1).strip()

        # Translation shortcuts
        if "translate to french" in lower or "traduis en français" in lower:
            if "hello" in lower:
                return "Bonjour"
            if "thank you" in lower or "thanks" in lower:
                return "Merci"
            if "good morning" in lower:
                return "Bonjour"
        if "translate to english" in lower:
            if "bonjour" in lower:
                return "Hello"
            if "merci" in lower:
                return "Thank you"
            if "salam" in lower or "salam 3likoum" in lower:
                return "Hello"
        if "translate to darija" in lower or "en darija" in lower:
            if "hello" in lower or "bonjour" in lower:
                return "Salam"
            if "thank you" in lower or "merci" in lower:
                return "Choukran"

        # JSON extraction heuristics
        if "extract" in lower and "json" in lower:
            payload: dict[str, Any] = {}
            name = re.search(r"name[:\s]+([A-Za-z][A-Za-z\s\-']+)", p, re.I)
            age = re.search(r"age[:\s]+(\d+)", p, re.I)
            email = re.search(r"[\w.+-]+@[\w.-]+\.\w+", p)
            city = re.search(r"city[:\s]+([A-Za-z][A-Za-z\s\-]+)", p, re.I)
            if name:
                payload["name"] = name.group(1).strip().rstrip(".")
            if age:
                payload["age"] = int(age.group(1))
            if email:
                payload["email"] = email.group(0)
            if city:
                payload["city"] = city.group(1).strip().rstrip(".")
            if payload:
                return json.dumps(payload, ensure_ascii=False)

        # General QA hard-coded facts for sample suite
        qa_map = {
            "capital of france": "Paris",
            "capital of morocco": "Rabat",
            "capital of japan": "Tokyo",
            "capital of spain": "Madrid",
            "capital of italy": "Rome",
            "capital of germany": "Berlin",
            "capital of portugal": "Lisbon",
            "capital of egypt": "Cairo",
            "2 + 2": "4",
            "what is 2+2": "4",
            "largest planet": "Jupiter",
            "chemical symbol for water": "H2O",
            "chemical formula for water": "H2O",
            "how many days in a leap year": "366",
            "primary colors": "red, blue, yellow",
            "who wrote hamlet": "William Shakespeare",
            "speed of light": "299792458",
            "boiling point of water at sea level": "100",
            "freezing point of water": "0",
            "closest to the sun": "Mercury",
        }
        for key, val in qa_map.items():
            if key in lower:
                return val

        # Fallback: echo a short deterministic snippet
        words = re.findall(r"[A-Za-z0-9]+", p)
        if words:
            return " ".join(words[-5:])
        return "OK"

    def _judge(self, prompt: str) -> str:
        expected_m = re.search(r"Expected:\s*(.*?)\nModel output:\s*(.*)$", prompt, re.S | re.I)
        if not expected_m:
            return json.dumps({"score": 0.0, "passed": False, "reason": "unparseable judge prompt"})
        expected = expected_m.group(1).strip()
        output = expected_m.group(2).strip()
        # Normalize loose equality for offline demos
        same = expected.lower().strip().strip('"') == output.lower().strip().strip('"')
        partial = expected.lower() in output.lower() or output.lower() in expected.lower()
        if same:
            return json.dumps({"score": 1.0, "passed": True, "reason": "exact match"})
        if partial and len(output) > 0:
            return json.dumps({"score": 0.7, "passed": True, "reason": "partial match"})
        return json.dumps({"score": 0.0, "passed": False, "reason": "mismatch"})
