# Progress — llm-eval-harness (P4)

## Done
- Private → **public** repo: https://github.com/mohamedtaqysalmi/llm-eval-harness
- MIT LICENSE (Mohamed-Taqy Salmi)
- Core CLI: suite loader, mock + OpenAI-compatible providers, scorers, async runner, SQLite, HTML/JSON/terminal reports, compare
- Sample suites: general QA, JSON extraction, Darija/FR/EN — **70 cases**
- Tests + GitHub Actions CI (ruff, pytest, mock smoke) — latest green; failed historical run deleted
- README with real mock metrics from `6bfda768435d` (2026-11-08)
- GitHub Pages landing + committed sample report under `docs/`

## In progress
- Nothing for P4 core delivery

## Next
- Optional: live OpenAI/DeepSeek comparison run (secrets already available on other repos if desired)
- Optional: pin repo / add GitHub topics

## Known issues
- Host default `python` may be 3.15; use `py -3.12`
- LLM-as-judge is biased by design — see README warning
- Cost figures are estimates from `config/prices.yaml`
