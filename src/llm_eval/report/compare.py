"""Compare two runs and highlight regressions."""

from __future__ import annotations

from dataclasses import dataclass

from llm_eval.models import RunResult


@dataclass
class Regression:
    case_id: str
    suite: str
    model: str
    baseline_passed: bool
    candidate_passed: bool
    baseline_score: float
    candidate_score: float
    kind: str  # regression | improvement | unchanged


@dataclass
class CompareReport:
    baseline_id: str
    candidate_id: str
    regressions: list[Regression]
    improvements: list[Regression]
    unchanged: int
    accuracy_delta: dict[str, float]


def compare_runs(baseline: RunResult, candidate: RunResult) -> CompareReport:
    base_idx = {(c.suite, c.case_id, c.model): c for c in baseline.cases}
    cand_idx = {(c.suite, c.case_id, c.model): c for c in candidate.cases}
    keys = sorted(set(base_idx) | set(cand_idx))

    regressions: list[Regression] = []
    improvements: list[Regression] = []
    unchanged = 0

    for key in keys:
        b = base_idx.get(key)
        c = cand_idx.get(key)
        if b is None or c is None:
            continue
        if b.passed and not c.passed:
            regressions.append(
                Regression(
                    case_id=c.case_id,
                    suite=c.suite,
                    model=c.model,
                    baseline_passed=b.passed,
                    candidate_passed=c.passed,
                    baseline_score=b.score,
                    candidate_score=c.score,
                    kind="regression",
                )
            )
        elif (not b.passed) and c.passed:
            improvements.append(
                Regression(
                    case_id=c.case_id,
                    suite=c.suite,
                    model=c.model,
                    baseline_passed=b.passed,
                    candidate_passed=c.passed,
                    baseline_score=b.score,
                    candidate_score=c.score,
                    kind="improvement",
                )
            )
        else:
            unchanged += 1

    base_acc = {s.model: s.accuracy for s in baseline.summaries}
    cand_acc = {s.model: s.accuracy for s in candidate.summaries}
    models = sorted(set(base_acc) | set(cand_acc))
    accuracy_delta = {
        m: cand_acc.get(m, 0.0) - base_acc.get(m, 0.0) for m in models
    }
    return CompareReport(
        baseline_id=baseline.run_id,
        candidate_id=candidate.run_id,
        regressions=regressions,
        improvements=improvements,
        unchanged=unchanged,
        accuracy_delta=accuracy_delta,
    )


def format_compare(report: CompareReport) -> str:
    lines = [
        f"Compare {report.baseline_id} → {report.candidate_id}",
        f"Regressions: {len(report.regressions)}  Improvements: {len(report.improvements)}  "
        f"Unchanged: {report.unchanged}",
        "Accuracy delta:",
    ]
    for model, delta in report.accuracy_delta.items():
        sign = "+" if delta >= 0 else ""
        lines.append(f"  {model}: {sign}{delta:.1%}")
    if report.regressions:
        lines.append("Regressions:")
        for r in report.regressions[:50]:
            lines.append(
                f"  - [{r.suite}/{r.case_id}] {r.model}: "
                f"{r.baseline_score:.2f}→{r.candidate_score:.2f}"
            )
    return "\n".join(lines)
