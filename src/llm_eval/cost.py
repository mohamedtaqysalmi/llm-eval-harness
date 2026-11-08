"""Estimate USD cost from a configurable price table."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

_DEFAULT_PRICES: dict[str, dict[str, float]] = {
    "mock": {"input": 0.0, "output": 0.0},
    "default": {"input": 0.50, "output": 1.50},
}


def load_price_table(path: str | Path | None = None) -> dict[str, dict[str, float]]:
    candidates: list[Path] = []
    if path:
        candidates.append(Path(path))
    candidates.extend(
        [
            Path("config/prices.yaml"),
            Path(__file__).resolve().parents[2] / "config" / "prices.yaml",
        ]
    )
    for candidate in candidates:
        if candidate.is_file():
            raw: dict[str, Any] = yaml.safe_load(candidate.read_text(encoding="utf-8")) or {}
            models = raw.get("models") or raw
            table: dict[str, dict[str, float]] = {}
            for name, prices in models.items():
                if isinstance(prices, dict):
                    table[str(name)] = {
                        "input": float(prices.get("input", 0.0)),
                        "output": float(prices.get("output", 0.0)),
                    }
            if table:
                return table
    return dict(_DEFAULT_PRICES)


def estimate_cost_usd(
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    price_table: dict[str, dict[str, float]] | None = None,
) -> float:
    table = price_table or _DEFAULT_PRICES
    prices = table.get(model) or table.get(_strip_provider(model)) or table.get("default")
    if not prices:
        prices = {"input": 0.0, "output": 0.0}
    # prices are USD per 1M tokens
    return (prompt_tokens * prices["input"] + completion_tokens * prices["output"]) / 1_000_000


def _strip_provider(model: str) -> str:
    if "/" in model:
        return model.split("/", 1)[1]
    return model
