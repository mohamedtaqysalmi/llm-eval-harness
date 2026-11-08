from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from llm_eval.cli import app
from llm_eval.models import RunResult
from llm_eval.report import compare_runs

ROOT = Path(__file__).resolve().parents[1]
runner = CliRunner()


def test_cli_version() -> None:
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert "0.1.0" in result.stdout


def test_mock_e2e_run(tmp_path: Path) -> None:
    out = tmp_path / "reports" / "mock-e2e"
    db = tmp_path / "evals.db"
    result = runner.invoke(
        app,
        [
            "run",
            str(ROOT / "suites"),
            "--models",
            "mock",
            "--out",
            str(out),
            "--db",
            str(db),
            "--judge-model",
            "mock",
        ],
    )
    assert result.exit_code == 0, result.output
    html = out / "report.html"
    js = out / "results.json"
    assert html.is_file()
    assert js.is_file()
    assert "llm-eval report" in html.read_text(encoding="utf-8")
    payload = json.loads(js.read_text(encoding="utf-8"))
    assert payload["models"] == ["mock"]
    assert len(payload["cases"]) >= 50
    assert payload["summaries"][0]["accuracy"] > 0.8


def test_compare_identical_runs(tmp_path: Path) -> None:
    out = tmp_path / "r1"
    db = tmp_path / "evals.db"
    result = runner.invoke(
        app,
        [
            "run",
            str(ROOT / "suites" / "general_qa.yaml"),
            "--models",
            "mock",
            "--out",
            str(out),
            "--db",
            str(db),
            "--judge-model",
            "mock",
        ],
    )
    assert result.exit_code == 0, result.output
    data = json.loads((out / "results.json").read_text(encoding="utf-8"))
    run = RunResult.model_validate(data)
    report = compare_runs(run, run)
    assert report.regressions == []
    assert report.improvements == []
