"""Pydantic schemas for suites, cases, scores, and run results (skeleton)."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class Case(BaseModel):
    id: str
    prompt: str
    expected: Any | None = None
    scoring: str = "exact"
    scoring_params: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class Suite(BaseModel):
    name: str
    description: str = ""
    cases: list[Case]
    default_scoring: str = "exact"
