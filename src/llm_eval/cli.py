"""Typer CLI entrypoint (skeleton — commands implemented in core phase)."""

from __future__ import annotations

import typer

from llm_eval import __version__

app = typer.Typer(
    name="llm-eval",
    help="Compare LLM models on your own prompt suites.",
    no_args_is_help=True,
)


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
    models: str = typer.Option("mock", "--models", "-m", help="Comma-separated model ids"),
    out: str = typer.Option("reports/latest", "--out", "-o", help="Output directory"),
) -> None:
    """Run evaluation suites against one or more models."""
    typer.echo(
        f"[skeleton] run suites={suites!r} models={models!r} out={out!r} "
        "(core not wired yet — see PROGRESS.md)"
    )
    raise typer.Exit(code=2)


if __name__ == "__main__":
    app()
