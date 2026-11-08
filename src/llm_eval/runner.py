"""Async evaluation runner with concurrency + rate limiting."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from llm_eval.cost import estimate_cost_usd, load_price_table
from llm_eval.models import CaseResult, RunResult, Suite
from llm_eval.providers import create_provider
from llm_eval.providers.base import Provider
from llm_eval.scorers import score_case


class RateLimiter:
    def __init__(self, rate: float) -> None:
        self._interval = 1.0 / rate if rate > 0 else 0.0
        self._lock = asyncio.Lock()
        self._last = 0.0

    async def acquire(self) -> None:
        if self._interval <= 0:
            return
        async with self._lock:
            loop = asyncio.get_running_loop()
            now = loop.time()
            wait = self._last + self._interval - now
            if wait > 0:
                await asyncio.sleep(wait)
            self._last = loop.time()


async def _eval_one(
    *,
    suite: Suite,
    case_id: str,
    prompt: str,
    expected: Any,
    scoring: str,
    scoring_params: dict[str, Any],
    system: str | None,
    metadata: dict[str, Any],
    model: str,
    provider: Provider,
    limiter: RateLimiter,
    price_table: dict[str, dict[str, float]],
    judge: Provider | None,
    temperature: float,
    max_tokens: int,
) -> CaseResult:
    await limiter.acquire()
    resp = await provider.complete(
        prompt, system=system, temperature=temperature, max_tokens=max_tokens
    )
    cost = estimate_cost_usd(
        model if model != "mock" else "mock",
        resp.prompt_tokens,
        resp.completion_tokens,
        price_table,
    )
    if resp.error:
        return CaseResult(
            case_id=case_id,
            suite=suite.name,
            model=model,
            prompt=prompt,
            expected=expected,
            output=resp.text,
            passed=False,
            score=0.0,
            scoring=scoring,
            scoring_detail=f"provider error: {resp.error}",
            latency_ms=resp.latency_ms,
            prompt_tokens=resp.prompt_tokens,
            completion_tokens=resp.completion_tokens,
            total_tokens=resp.total_tokens,
            cost_usd=cost,
            error=resp.error,
            metadata=metadata,
        )

    scored = await score_case(scoring, resp.text, expected, scoring_params, judge=judge)
    return CaseResult(
        case_id=case_id,
        suite=suite.name,
        model=model,
        prompt=prompt,
        expected=expected,
        output=resp.text,
        passed=scored.passed,
        score=scored.score,
        scoring=scoring,
        scoring_detail=scored.detail,
        latency_ms=resp.latency_ms,
        prompt_tokens=resp.prompt_tokens,
        completion_tokens=resp.completion_tokens,
        total_tokens=resp.total_tokens,
        cost_usd=cost,
        error=None,
        metadata=metadata,
    )


async def run_evaluation(
    suites: list[Suite],
    models: list[str],
    *,
    concurrency: int = 8,
    rate_limit: float = 0.0,
    price_table_path: str | Path | None = None,
    judge_model: str | None = None,
    temperature: float = 0.0,
    max_tokens: int = 512,
    config: dict[str, Any] | None = None,
) -> RunResult:
    price_table = load_price_table(price_table_path)
    providers: dict[str, Provider] = {m: create_provider(m) for m in models}
    needs_judge = any(
        str(case.scoring or suite.default_scoring) == "llm_judge"
        for suite in suites
        for case in suite.cases
    )
    judge: Provider | None = None
    if needs_judge:
        judge = create_provider(judge_model or "mock")

    limiter = RateLimiter(rate_limit)
    sem = asyncio.Semaphore(concurrency)
    tasks: list[asyncio.Task[CaseResult]] = []

    async def wrapped(**kwargs: Any) -> CaseResult:
        async with sem:
            return await _eval_one(**kwargs)

    for suite in suites:
        for case in suite.cases:
            scoring = str(case.scoring or suite.default_scoring)
            for model in models:
                case_judge = judge if scoring == "llm_judge" else None
                tasks.append(
                    asyncio.create_task(
                        wrapped(
                            suite=suite,
                            case_id=case.id,
                            prompt=case.prompt,
                            expected=case.expected,
                            scoring=scoring,
                            scoring_params=dict(case.scoring_params),
                            system=case.system,
                            metadata=dict(case.metadata),
                            model=model,
                            provider=providers[model],
                            limiter=limiter,
                            price_table=price_table,
                            judge=case_judge,
                            temperature=temperature,
                            max_tokens=max_tokens,
                        )
                    )
                )

    results = await asyncio.gather(*tasks)

    for p in providers.values():
        await p.aclose()
    if judge is not None:
        await judge.aclose()

    run = RunResult(
        models=list(models),
        suites=[s.name for s in suites],
        cases=list(results),
        config=config
        or {
            "concurrency": concurrency,
            "rate_limit": rate_limit,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "judge_model": judge_model,
        },
    )
    run.recompute_summaries()
    return run
