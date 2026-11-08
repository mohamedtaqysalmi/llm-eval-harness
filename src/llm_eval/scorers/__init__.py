"""Scoring methods for case outputs."""

from __future__ import annotations

import json
import re
from typing import Any

from jsonschema import Draft202012Validator
from rapidfuzz import fuzz

from llm_eval.models import ScoreResult, ScoringMethod
from llm_eval.providers.base import Provider


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def score_exact(output: str, expected: Any, params: dict[str, Any]) -> ScoreResult:
    ignore_case = bool(params.get("ignore_case", True))
    strip = bool(params.get("strip", True))
    a = output.strip() if strip else output
    b = _as_text(expected)
    b = b.strip() if strip else b
    if ignore_case:
        passed = a.lower() == b.lower()
    else:
        passed = a == b
    return ScoreResult(passed=passed, score=1.0 if passed else 0.0, detail="exact match")


def score_regex(output: str, expected: Any, params: dict[str, Any]) -> ScoreResult:
    pattern = params.get("pattern") or _as_text(expected)
    flags = re.IGNORECASE if params.get("ignore_case", False) else 0
    try:
        matched = re.search(pattern, output, flags=flags) is not None
    except re.error as exc:
        return ScoreResult(passed=False, score=0.0, detail=f"invalid regex: {exc}")
    return ScoreResult(
        passed=matched,
        score=1.0 if matched else 0.0,
        detail=f"regex {pattern!r}",
        raw={"pattern": pattern},
    )


def score_json_schema(output: str, expected: Any, params: dict[str, Any]) -> ScoreResult:
    schema = params.get("schema") or expected
    if not isinstance(schema, dict):
        return ScoreResult(passed=False, score=0.0, detail="json_schema requires a schema object")
    try:
        data = json.loads(output)
    except json.JSONDecodeError as exc:
        return ScoreResult(passed=False, score=0.0, detail=f"invalid JSON: {exc}")
    validator = Draft202012Validator(schema)
    errors = sorted(validator.iter_errors(data), key=lambda e: list(e.path))
    if errors:
        msg = "; ".join(e.message for e in errors[:3])
        return ScoreResult(passed=False, score=0.0, detail=msg, raw={"errors": [e.message for e in errors]})
    return ScoreResult(passed=True, score=1.0, detail="valid against schema")


def score_contains(output: str, expected: Any, params: dict[str, Any]) -> ScoreResult:
    needles = params.get("values") or params.get("contains") or expected
    if isinstance(needles, str):
        needles = [needles]
    if not isinstance(needles, list):
        needles = [_as_text(needles)]
    ignore_case = bool(params.get("ignore_case", True))
    hay = output.lower() if ignore_case else output
    missing = []
    for n in needles:
        text = str(n)
        probe = text.lower() if ignore_case else text
        if probe not in hay:
            missing.append(text)
    passed = not missing
    return ScoreResult(
        passed=passed,
        score=1.0 if passed else max(0.0, 1.0 - len(missing) / max(1, len(needles))),
        detail="contains all" if passed else f"missing: {missing}",
    )


def score_not_contains(output: str, expected: Any, params: dict[str, Any]) -> ScoreResult:
    needles = params.get("values") or params.get("not_contains") or expected
    if isinstance(needles, str):
        needles = [needles]
    if not isinstance(needles, list):
        needles = [_as_text(needles)]
    ignore_case = bool(params.get("ignore_case", True))
    hay = output.lower() if ignore_case else output
    found = []
    for n in needles:
        text = str(n)
        probe = text.lower() if ignore_case else text
        if probe in hay:
            found.append(text)
    passed = not found
    return ScoreResult(
        passed=passed,
        score=1.0 if passed else 0.0,
        detail="no forbidden strings" if passed else f"found: {found}",
    )


def score_similarity(output: str, expected: Any, params: dict[str, Any]) -> ScoreResult:
    threshold = float(params.get("threshold", 0.8))
    method = str(params.get("method", "ratio"))
    target = _as_text(expected)
    if method == "partial":
        ratio = fuzz.partial_ratio(output, target) / 100.0
    elif method == "token_set":
        ratio = fuzz.token_set_ratio(output, target) / 100.0
    else:
        ratio = fuzz.ratio(output, target) / 100.0
    passed = ratio >= threshold
    return ScoreResult(
        passed=passed,
        score=ratio,
        detail=f"similarity={ratio:.3f} threshold={threshold}",
        raw={"similarity": ratio, "threshold": threshold, "method": method},
    )


async def score_llm_judge(
    output: str,
    expected: Any,
    params: dict[str, Any],
    judge: Provider,
) -> ScoreResult:
    rubric = params.get("rubric") or (
        "Score 1 if the answer is correct and complete relative to the expected answer; else 0."
    )
    pass_threshold = float(params.get("pass_threshold", 0.5))
    prompt = (
        "You are an evaluation judge. Reply with ONLY a JSON object "
        '{"score": <0..1 float>, "passed": <bool>, "reason": "<short>"}.\n'
        f"Rubric: {rubric}\n"
        f"Expected: {_as_text(expected)}\n"
        f"Model output: {output}\n"
    )
    resp = await judge.complete(prompt, temperature=0.0, max_tokens=200)
    if resp.error:
        return ScoreResult(passed=False, score=0.0, detail=f"judge error: {resp.error}")
    try:
        data = json.loads(resp.text)
        score = float(data.get("score", 0.0))
        passed = bool(data.get("passed", score >= pass_threshold))
        reason = str(data.get("reason", ""))
        return ScoreResult(
            passed=passed,
            score=score,
            detail=reason or "llm judge",
            raw={"judge_raw": resp.text},
        )
    except (json.JSONDecodeError, TypeError, ValueError):
        # Heuristic fallback if judge did not return JSON
        lower = resp.text.lower()
        passed = "pass" in lower and "fail" not in lower
        return ScoreResult(
            passed=passed,
            score=1.0 if passed else 0.0,
            detail="llm judge (non-JSON fallback)",
            raw={"judge_raw": resp.text},
        )


async def score_case(
    method: str,
    output: str,
    expected: Any,
    params: dict[str, Any] | None = None,
    *,
    judge: Provider | None = None,
) -> ScoreResult:
    params = params or {}
    key = str(method).lower()
    if key == ScoringMethod.EXACT:
        return score_exact(output, expected, params)
    if key == ScoringMethod.REGEX:
        return score_regex(output, expected, params)
    if key == ScoringMethod.JSON_SCHEMA:
        return score_json_schema(output, expected, params)
    if key == ScoringMethod.CONTAINS:
        return score_contains(output, expected, params)
    if key == ScoringMethod.NOT_CONTAINS:
        return score_not_contains(output, expected, params)
    if key == ScoringMethod.SIMILARITY:
        return score_similarity(output, expected, params)
    if key == ScoringMethod.LLM_JUDGE:
        if judge is None:
            return ScoreResult(passed=False, score=0.0, detail="llm_judge requires a judge provider")
        return await score_llm_judge(output, expected, params, judge)
    return ScoreResult(passed=False, score=0.0, detail=f"unknown scoring method: {method}")


__all__ = ["score_case", "ScoreResult"]
