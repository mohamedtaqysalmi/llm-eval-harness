"""Typer CLI for llm-eval."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Optional

import typer
from dotenv import load_dotenv
from rich.console import Console

from llm_eval import __version__
from llm_eval.report import (
    compare_runs,
    export_json,
    format_compare,
    print_run_table,
    render_html,
)
from llm_eval.runner import run_evaluation
from llm_eval.storage import ResultStore
from llm_eval.suite import load_suites

load_dotenv()

app = typer.Typer(
    name="llm-eval",
    help="Compare LLM models on your own prompt suites.",
    no_args_is_help=True,
)
console = Console()


def _db_path(explicit: Optional[str] = None) -> Path:
    return Path(explicit or os.getenv("LLM_EVAL_DB_PATH", "results/evals.db"))


@app.callback()
def _main() -> None:
    """llm-eval harness."""


@app.command("version")
def version() -> None:
    """Print package version."""
    typer.echo(__version__)


@app.command("run")
def run(
    suites: str = typer.Argument(..., help="Path to suite file or directory"),
    models: str = typer.Option(
        "mock",
        "--models",
        "-m",
        help="Comma-separated model ids (mock, gpt-4o-mini, ollama/llama3.2, ...)",
    ),
    out: str = typer.Option("reports/latest", "--out", "-o", help="Output directory"),
    concurrency: int = typer.Option(8, "--concurrency", "-c", help="Max concurrent requests"),
    rate_limit: float = typer.Option(
        0.0, "--rate-limit", help="Max requests/sec (0 = unlimited)"
    ),
    judge_model: Optional[str] = typer.Option(
        None, "--judge-model", help="Model for llm_judge scoring"
    ),
    db: Optional[str] = typer.Option(None, "--db", help="SQLite DB path"),
    price_table: Optional[str] = typer.Option(
        None, "--prices", help="Path to prices YAML"
    ),
    temperature: float = typer.Option(0.0, "--temperature"),
    max_tokens: int = typer.Option(512, "--max-tokens"),
) -> None:
    """Run evaluation suites against one or more models."""
    model_list = [m.strip() for m in models.split(",") if m.strip()]
    if not model_list:
        raise typer.BadParameter("Provide at least one model via --models")

    loaded = load_suites(suites)
    console.print(
        f"Loaded {len(loaded)} suite(s), "
        f"{sum(len(s.cases) for s in loaded)} case(s); models={model_list}"
    )

    result = asyncio.run(
        run_evaluation(
            loaded,
            model_list,
            concurrency=concurrency,
            rate_limit=rate_limit,
            price_table_path=price_table,
            judge_model=judge_model or os.getenv("LLM_EVAL_JUDGE_MODEL"),
            temperature=temperature,
            max_tokens=max_tokens,
        )
    )

    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = export_json(result, out_dir / "results.json")
    html_path = render_html(result, out_dir / "report.html")

    store = ResultStore(_db_path(db))
    store.save_run(result)
    store.close()

    print_run_table(result, console)
    console.print(f"[green]JSON[/green]  {json_path}")
    console.print(f"[green]HTML[/green]  {html_path}")
    console.print(f"[green]Run ID[/green] {result.run_id}")


@app.command("compare")
def compare(
    baseline: str = typer.Argument(..., help="Baseline run id OR results.json path"),
    candidate: str = typer.Argument(..., help="Candidate run id OR results.json path"),
    db: Optional[str] = typer.Option(None, "--db", help="SQLite DB path"),
) -> None:
    """Compare two runs and show regressions / improvements."""
    from llm_eval.models import RunResult

    def load(ref: str) -> RunResult:
        path = Path(ref)
        if path.is_file():
            return RunResult.model_validate(json.loads(path.read_text(encoding="utf-8")))
        store = ResultStore(_db_path(db))
        run = store.load_run(ref)
        store.close()
        if run is None:
            raise typer.BadParameter(f"Run not found: {ref}")
        return run

    report = compare_runs(load(baseline), load(candidate))
    console.print(format_compare(report))
    if report.regressions:
        raise typer.Exit(code=1)


@app.command("list-runs")
def list_runs(
    db: Optional[str] = typer.Option(None, "--db", help="SQLite DB path"),
    limit: int = typer.Option(20, "--limit", "-n"),
) -> None:
    """List recent runs stored in SQLite."""
    store = ResultStore(_db_path(db))
    rows = store.list_runs(limit=limit)
    store.close()
    if not rows:
        console.print("No runs stored yet.")
        return
    for r in rows:
        console.print(
            f"{r['run_id']}  {r['created_at']}  "
            f"models={','.join(r['models'])}  suites={','.join(r['suites'])}"
        )


if __name__ == "__main__":
    app()
