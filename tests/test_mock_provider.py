from __future__ import annotations

import pytest

from llm_eval.providers.mock import MockProvider


@pytest.mark.asyncio
async def test_mock_capitals() -> None:
    p = MockProvider()
    r = await p.complete("What is the capital of France? Reply with only the city name.")
    assert r.error is None
    assert r.text == "Paris"


@pytest.mark.asyncio
async def test_mock_answer_marker() -> None:
    p = MockProvider()
    r = await p.complete("Anything [MOCK_ANSWER: Zebra]")
    assert r.text == "Zebra"


@pytest.mark.asyncio
async def test_mock_json_marker() -> None:
    p = MockProvider()
    r = await p.complete('[MOCK_JSON: {"ok": true}]')
    assert r.text == '{"ok": true}'


@pytest.mark.asyncio
async def test_mock_judge_exact() -> None:
    p = MockProvider()
    prompt = (
        "You are an evaluation judge. Reply with ONLY a JSON object "
        '{"score": <0..1 float>, "passed": <bool>, "reason": "<short>"}.\n'
        "Rubric: Pass if correct.\n"
        "Expected: Hello\n"
        "Model output: Hello\n"
    )
    r = await p.complete(prompt)
    compact = r.text.lower().replace(" ", "")
    assert '"passed":true' in compact
