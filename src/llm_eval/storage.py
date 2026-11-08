"""SQLite persistence for evaluation runs."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from llm_eval.models import CaseResult, RunResult

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    models_json TEXT NOT NULL,
    suites_json TEXT NOT NULL,
    config_json TEXT NOT NULL,
    summaries_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS case_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    case_id TEXT NOT NULL,
    suite TEXT NOT NULL,
    model TEXT NOT NULL,
    prompt TEXT NOT NULL,
    expected_json TEXT,
    output TEXT,
    passed INTEGER NOT NULL,
    score REAL NOT NULL,
    scoring TEXT NOT NULL,
    scoring_detail TEXT,
    latency_ms REAL,
    prompt_tokens INTEGER,
    completion_tokens INTEGER,
    total_tokens INTEGER,
    cost_usd REAL,
    error TEXT,
    metadata_json TEXT,
    FOREIGN KEY(run_id) REFERENCES runs(run_id)
);

CREATE INDEX IF NOT EXISTS idx_case_results_run ON case_results(run_id);
CREATE INDEX IF NOT EXISTS idx_case_results_model ON case_results(model);
"""


class ResultStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(SCHEMA)
        self._conn.commit()

    def save_run(self, run: RunResult) -> None:
        self._conn.execute(
            """
            INSERT OR REPLACE INTO runs
            (run_id, created_at, models_json, suites_json, config_json, summaries_json)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                run.run_id,
                run.created_at.isoformat(),
                json.dumps(run.models),
                json.dumps(run.suites),
                json.dumps(run.config),
                json.dumps([s.model_dump() for s in run.summaries]),
            ),
        )
        self._conn.execute("DELETE FROM case_results WHERE run_id = ?", (run.run_id,))
        for c in run.cases:
            self._conn.execute(
                """
                INSERT INTO case_results (
                    run_id, case_id, suite, model, prompt, expected_json, output,
                    passed, score, scoring, scoring_detail, latency_ms,
                    prompt_tokens, completion_tokens, total_tokens, cost_usd,
                    error, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run.run_id,
                    c.case_id,
                    c.suite,
                    c.model,
                    c.prompt,
                    json.dumps(c.expected),
                    c.output,
                    1 if c.passed else 0,
                    c.score,
                    c.scoring,
                    c.scoring_detail,
                    c.latency_ms,
                    c.prompt_tokens,
                    c.completion_tokens,
                    c.total_tokens,
                    c.cost_usd,
                    c.error,
                    json.dumps(c.metadata),
                ),
            )
        self._conn.commit()

    def load_run(self, run_id: str) -> RunResult | None:
        row = self._conn.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
        if row is None:
            return None
        cases_rows = self._conn.execute(
            "SELECT * FROM case_results WHERE run_id = ? ORDER BY id", (run_id,)
        ).fetchall()
        cases = [
            CaseResult(
                case_id=r["case_id"],
                suite=r["suite"],
                model=r["model"],
                prompt=r["prompt"],
                expected=json.loads(r["expected_json"]) if r["expected_json"] is not None else None,
                output=r["output"] or "",
                passed=bool(r["passed"]),
                score=float(r["score"]),
                scoring=r["scoring"],
                scoring_detail=r["scoring_detail"] or "",
                latency_ms=float(r["latency_ms"] or 0),
                prompt_tokens=int(r["prompt_tokens"] or 0),
                completion_tokens=int(r["completion_tokens"] or 0),
                total_tokens=int(r["total_tokens"] or 0),
                cost_usd=float(r["cost_usd"] or 0),
                error=r["error"],
                metadata=json.loads(r["metadata_json"] or "{}"),
            )
            for r in cases_rows
        ]
        from datetime import datetime

        return RunResult(
            run_id=row["run_id"],
            created_at=datetime.fromisoformat(row["created_at"]),
            models=json.loads(row["models_json"]),
            suites=json.loads(row["suites_json"]),
            cases=cases,
            summaries=json.loads(row["summaries_json"]),
            config=json.loads(row["config_json"]),
        )

    def list_runs(self, limit: int = 20) -> list[dict]:
        rows = self._conn.execute(
            "SELECT run_id, created_at, models_json, suites_json FROM runs "
            "ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [
            {
                "run_id": r["run_id"],
                "created_at": r["created_at"],
                "models": json.loads(r["models_json"]),
                "suites": json.loads(r["suites_json"]),
            }
            for r in rows
        ]

    def close(self) -> None:
        self._conn.close()
