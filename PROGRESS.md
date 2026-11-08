# Progress — llm-eval-harness (P4)

## Done
- Cloned private repo `mohamedtaqysalmi/llm-eval-harness`
- MIT LICENSE (Mohamed-Taqy Salmi)
- `.gitignore`, `.env.example`
- Project skeleton: `pyproject.toml`, package layout, Makefile, Docker stubs, PROGRESS.md

## In progress
- Core: suite loader, providers (mock + OpenAI-compat), scorers, async runner, SQLite storage, reports

## Next
- Sample suites (general QA, JSON extraction, translation sanity) — 50+ cases
- Tests + mock end-to-end
- GitHub Actions CI (ruff + mock e2e)
- Full README (12 sections) with metrics from a real mock run
- Polish: HTML report charts, run comparison / regressions

## Known issues
- Repo was empty on clone; no prior commits
- README metrics intentionally absent until a real `llm-eval run` completes
