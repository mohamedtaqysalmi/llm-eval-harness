from __future__ import annotations

import pytest

from llm_eval.scorers import score_case


@pytest.mark.asyncio
async def test_exact_match_case_insensitive() -> None:
    r = await score_case("exact", "Paris", "paris", {})
    assert r.passed and r.score == 1.0


@pytest.mark.asyncio
async def test_regex() -> None:
    r = await score_case("regex", "answer: 42", None, {"pattern": r"\b42\b"})
    assert r.passed


@pytest.mark.asyncio
async def test_json_schema_valid() -> None:
    schema = {
        "type": "object",
        "required": ["name"],
        "properties": {"name": {"type": "string"}},
    }
    r = await score_case("json_schema", '{"name": "Ada"}', schema, {})
    assert r.passed


@pytest.mark.asyncio
async def test_json_schema_invalid_json() -> None:
    r = await score_case("json_schema", "not-json", {"type": "object"}, {})
    assert not r.passed


@pytest.mark.asyncio
async def test_contains_and_not_contains() -> None:
    ok = await score_case("contains", "red blue yellow", None, {"values": ["red", "blue"]})
    assert ok.passed
    bad = await score_case("not_contains", "as an AI I cannot", None, {"values": ["as an AI"]})
    assert not bad.passed


@pytest.mark.asyncio
async def test_similarity_threshold() -> None:
    r = await score_case("similarity", "Paris", "Paris", {"threshold": 0.9})
    assert r.passed and r.score >= 0.9
