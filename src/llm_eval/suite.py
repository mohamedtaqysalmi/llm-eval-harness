"""Load YAML / JSONL evaluation suites."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from llm_eval.models import Case, Suite


def _normalize_case(raw: dict[str, Any], default_scoring: str, index: int) -> Case:
    data = dict(raw)
    if "id" not in data:
        data["id"] = f"case-{index + 1:03d}"
    if "scoring" not in data:
        data["scoring"] = default_scoring
    return Case.model_validate(data)


def load_suite_file(path: Path) -> Suite:
    text = path.read_text(encoding="utf-8")
    suffix = path.suffix.lower()

    if suffix in {".yaml", ".yml"}:
        raw = yaml.safe_load(text)
        if not isinstance(raw, dict):
            raise ValueError(f"Suite YAML must be a mapping: {path}")
        name = raw.get("name") or path.stem
        default_scoring = raw.get("default_scoring", "exact")
        cases_raw = raw.get("cases") or []
        cases = [
            _normalize_case(c, default_scoring, i)
            for i, c in enumerate(cases_raw)
            if isinstance(c, dict)
        ]
        return Suite(
            name=name,
            description=raw.get("description", ""),
            cases=cases,
            default_scoring=default_scoring,
            version=str(raw.get("version", "1")),
        )

    if suffix == ".jsonl":
        cases: list[Case] = []
        meta: dict[str, Any] = {"name": path.stem, "description": "", "default_scoring": "exact"}
        for i, line in enumerate(text.splitlines()):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if obj.get("_meta") is True or obj.get("type") == "meta":
                meta.update({k: v for k, v in obj.items() if k not in {"_meta", "type"}})
                continue
            cases.append(_normalize_case(obj, meta.get("default_scoring", "exact"), i))
        return Suite(
            name=str(meta.get("name") or path.stem),
            description=str(meta.get("description") or ""),
            cases=cases,
            default_scoring=str(meta.get("default_scoring") or "exact"),
            version=str(meta.get("version") or "1"),
        )

    if suffix == ".json":
        raw = json.loads(text)
        if not isinstance(raw, dict):
            raise ValueError(f"Suite JSON must be an object: {path}")
        name = raw.get("name") or path.stem
        default_scoring = raw.get("default_scoring", "exact")
        cases = [
            _normalize_case(c, default_scoring, i)
            for i, c in enumerate(raw.get("cases") or [])
            if isinstance(c, dict)
        ]
        return Suite(
            name=name,
            description=raw.get("description", ""),
            cases=cases,
            default_scoring=default_scoring,
            version=str(raw.get("version", "1")),
        )

    raise ValueError(f"Unsupported suite format: {path}")


def discover_suite_paths(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(f"Suite path not found: {path}")
    files: list[Path] = []
    for pattern in ("*.yaml", "*.yml", "*.jsonl", "*.json"):
        files.extend(sorted(path.rglob(pattern)))
    # de-dupe while preserving order
    seen: set[Path] = set()
    out: list[Path] = []
    for f in files:
        if f not in seen:
            seen.add(f)
            out.append(f)
    return out


def load_suites(path: str | Path) -> list[Suite]:
    root = Path(path)
    suite_paths = discover_suite_paths(root)
    if not suite_paths:
        raise FileNotFoundError(f"No suite files found under {root}")
    suites = [load_suite_file(p) for p in suite_paths]
    return suites
