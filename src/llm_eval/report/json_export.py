"""JSON export for run results."""

from __future__ import annotations

import json
from pathlib import Path

from llm_eval.models import RunResult


def export_json(run: RunResult, path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = run.model_dump(mode="json")
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return out
