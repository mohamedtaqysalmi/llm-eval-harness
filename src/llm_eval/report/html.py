"""Self-contained HTML report via Jinja2."""

from __future__ import annotations

from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape

from llm_eval.models import RunResult


def _template_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "templates"


def render_html(run: RunResult, path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    env = Environment(
        loader=FileSystemLoader(str(_template_dir())),
        autoescape=select_autoescape(["html", "xml"]),
    )
    template = env.get_template("report.html.j2")
    # Convert to plain dicts for template friendliness
    payload = run.model_dump(mode="json")
    html = template.render(run=payload)
    out.write_text(html, encoding="utf-8")
    return out
