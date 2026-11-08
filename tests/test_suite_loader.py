from __future__ import annotations

from pathlib import Path

from llm_eval.suite import load_suites

ROOT = Path(__file__).resolve().parents[1]


def test_load_all_sample_suites() -> None:
    suites = load_suites(ROOT / "suites")
    assert len(suites) == 3
    total = sum(len(s.cases) for s in suites)
    assert total >= 50
    names = {s.name for s in suites}
    assert names == {"general_qa", "json_extraction", "translation_sanity"}


def test_case_ids_unique_within_suite() -> None:
    for suite in load_suites(ROOT / "suites"):
        ids = [c.id for c in suite.cases]
        assert len(ids) == len(set(ids))
