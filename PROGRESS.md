# Progress — llm-eval-harness (P4)

## Done
- Cloned private repo; MIT LICENSE (Mohamed-Taqy Salmi)
- Skeleton: package layout, Docker, Makefile, `.env.example`, `.gitignore`
- Core: suite loader (YAML/JSON/JSONL), mock + OpenAI-compatible providers, scorers, async runner, cost table, SQLite storage
- Reports: Rich terminal table, self-contained HTML (sortable + charts + drill-down), JSON export, run compare / regressions
- Sample suites: general QA, JSON extraction, Darija/FR/EN translation — **70 cases**
- Tests: unit + mock e2e (**17 passed** on Python 3.12)
- GitHub Actions CI: ruff + pytest + mock CLI smoke
- README (12 sections) with metrics from real mock run `6bfda768435d` on 2026-11-08 (70/70, 100%)
- Verified: `llm-eval run suites/ --models mock` produces HTML + JSON

## In progress
- None (P4 deliverable complete)

## Next
- Optional: PyPI publish, more suites, live OpenAI/Ollama comparison runs
- Optional: keep GitHub topics/description in sync

## Known issues
- Host default `python` may be 3.15; use `py -3.12` (PyYAML needs wheels/MSVC on 3.15)
- LLM-as-judge is biased by design — see README warning
- Cost figures are estimates from `config/prices.yaml`, not billing truth
