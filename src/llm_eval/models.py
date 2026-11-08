"""Pydantic schemas for suites, cases, scores, and run results."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


class ScoringMethod(StrEnum):
    EXACT = "exact"
    REGEX = "regex"
    JSON_SCHEMA = "json_schema"
    CONTAINS = "contains"
    NOT_CONTAINS = "not_contains"
    SIMILARITY = "similarity"
    LLM_JUDGE = "llm_judge"


class Case(BaseModel):
    id: str
    prompt: str
    expected: Any | None = None
    system: str | None = None
    scoring: ScoringMethod | str = ScoringMethod.EXACT
    scoring_params: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class Suite(BaseModel):
    name: str
    description: str = ""
    cases: list[Case]
    default_scoring: ScoringMethod | str = ScoringMethod.EXACT
    version: str = "1"


class ScoreResult(BaseModel):
    passed: bool
    score: float = 0.0
    detail: str = ""
    raw: dict[str, Any] = Field(default_factory=dict)


class ModelResponse(BaseModel):
    text: str
    latency_ms: float
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    model: str
    error: str | None = None
    raw: dict[str, Any] = Field(default_factory=dict)


class CaseResult(BaseModel):
    case_id: str
    suite: str
    model: str
    prompt: str
    expected: Any | None = None
    output: str = ""
    passed: bool = False
    score: float = 0.0
    scoring: str = "exact"
    scoring_detail: str = ""
    latency_ms: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0
    error: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ModelSummary(BaseModel):
    model: str
    total: int = 0
    passed: int = 0
    failed: int = 0
    errors: int = 0
    accuracy: float = 0.0
    mean_score: float = 0.0
    mean_latency_ms: float = 0.0
    total_tokens: int = 0
    total_cost_usd: float = 0.0


class RunResult(BaseModel):
    run_id: str = Field(default_factory=lambda: uuid4().hex[:12])
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    models: list[str] = Field(default_factory=list)
    suites: list[str] = Field(default_factory=list)
    cases: list[CaseResult] = Field(default_factory=list)
    summaries: list[ModelSummary] = Field(default_factory=list)
    config: dict[str, Any] = Field(default_factory=dict)

    def recompute_summaries(self) -> None:
        by_model: dict[str, list[CaseResult]] = {}
        for c in self.cases:
            by_model.setdefault(c.model, []).append(c)
        summaries: list[ModelSummary] = []
        for model, rows in by_model.items():
            total = len(rows)
            passed = sum(1 for r in rows if r.passed and not r.error)
            errors = sum(1 for r in rows if r.error)
            failed = total - passed
            mean_score = sum(r.score for r in rows) / total if total else 0.0
            mean_latency = sum(r.latency_ms for r in rows) / total if total else 0.0
            summaries.append(
                ModelSummary(
                    model=model,
                    total=total,
                    passed=passed,
                    failed=failed,
                    errors=errors,
                    accuracy=(passed / total) if total else 0.0,
                    mean_score=mean_score,
                    mean_latency_ms=mean_latency,
                    total_tokens=sum(r.total_tokens for r in rows),
                    total_cost_usd=sum(r.cost_usd for r in rows),
                )
            )
        self.summaries = sorted(summaries, key=lambda s: (-s.accuracy, s.model))
