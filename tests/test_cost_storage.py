from __future__ import annotations

from pathlib import Path

from llm_eval.cost import estimate_cost_usd, load_price_table
from llm_eval.models import CaseResult, RunResult
from llm_eval.storage import ResultStore


def test_price_table_loads() -> None:
    root = Path(__file__).resolve().parents[1]
    table = load_price_table(root / "config" / "prices.yaml")
    assert "mock" in table
    assert estimate_cost_usd("mock", 1000, 1000, table) == 0.0
    cost = estimate_cost_usd("gpt-4o-mini", 1_000_000, 0, table)
    assert abs(cost - 0.15) < 1e-9


def test_sqlite_roundtrip(tmp_path: Path) -> None:
    run = RunResult(
        models=["mock"],
        suites=["demo"],
        cases=[
            CaseResult(
                case_id="c1",
                suite="demo",
                model="mock",
                prompt="hi",
                expected="hi",
                output="hi",
                passed=True,
                score=1.0,
                scoring="exact",
            )
        ],
    )
    run.recompute_summaries()
    store = ResultStore(tmp_path / "t.db")
    store.save_run(run)
    loaded = store.load_run(run.run_id)
    store.close()
    assert loaded is not None
    assert loaded.run_id == run.run_id
    assert loaded.cases[0].passed is True
    assert loaded.summaries[0].accuracy == 1.0
