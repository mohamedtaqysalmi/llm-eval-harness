"""Rich terminal summary table."""

from __future__ import annotations

from rich.console import Console
from rich.table import Table

from llm_eval.models import RunResult


def print_run_table(run: RunResult, console: Console | None = None) -> None:
    console = console or Console()
    console.print(
        f"[bold]Run[/bold] {run.run_id}  "
        f"[dim]{run.created_at.isoformat()}[/dim]  "
        f"suites={', '.join(run.suites)}"
    )
    table = Table(title="Model comparison")
    table.add_column("Model")
    table.add_column("Accuracy", justify="right")
    table.add_column("Passed", justify="right")
    table.add_column("Failed", justify="right")
    table.add_column("Errors", justify="right")
    table.add_column("Mean score", justify="right")
    table.add_column("Mean latency ms", justify="right")
    table.add_column("Tokens", justify="right")
    table.add_column("Cost USD", justify="right")

    for s in run.summaries:
        table.add_row(
            s.model,
            f"{s.accuracy:.1%}",
            str(s.passed),
            str(s.failed),
            str(s.errors),
            f"{s.mean_score:.3f}",
            f"{s.mean_latency_ms:.1f}",
            str(s.total_tokens),
            f"{s.total_cost_usd:.6f}",
        )
    console.print(table)
