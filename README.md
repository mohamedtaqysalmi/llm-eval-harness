# llm-eval-harness

Tiny, honest framework to compare LLM models on **your own** prompt suites — not someone else's leaderboard.

Installable CLI: `llm-eval`. Offline-friendly via a deterministic `mock` provider.

## 1. Why this exists

Most eval tooling either assumes huge public benchmarks or hides scoring behind opaque dashboards. This project is deliberately small: bring YAML/JSONL cases, run N OpenAI-compatible models (including Ollama), score with clear methods, and get a terminal table + self-contained HTML report + JSON export you can diff.

## 2. Features

- Suite formats: YAML, JSON, JSONL
- Scorers: exact, regex, JSON Schema, contains / not-contains, fuzzy similarity, LLM-as-judge (configurable judge model)
- Async runner with concurrency, optional rate limiting, retries
- Latency, token, and estimated cost capture (configurable price table)
- SQLite persistence of runs
- Terminal summary, sortable HTML report (per-case drill-down + charts), JSON export
- Compare two runs for regressions / improvements
- Mock provider for offline tests and demos
- Docker + Makefile + GitHub Actions CI (mock e2e + ruff)

## 3. Install

Requires **Python 3.12+**.

```bash
# from the repo root
python -m pip install -e ".[dev]"

# or with the Windows py launcher
py -3.12 -m pip install -e ".[dev]"
```

Copy env template (never commit secrets):

```bash
cp .env.example .env
```

## 4. Quick start (offline mock)

```bash
llm-eval run suites/ --models mock --out reports/mock-run --judge-model mock
```

Open `reports/mock-run/report.html`. Equivalent Make target: `make run-mock` (add `--judge-model mock` if your Make target does not already).

## 5. Real providers

OpenAI-compatible APIs:

```bash
export OPENAI_API_KEY=sk-...
export OPENAI_BASE_URL=https://api.openai.com/v1   # optional
llm-eval run suites/ --models gpt-4o-mini -o reports/openai
```

Ollama (OpenAI-compatible endpoint):

```bash
export OLLAMA_BASE_URL=http://localhost:11434/v1
llm-eval run suites/ --models ollama/llama3.2 -o reports/ollama
```

Multiple models:

```bash
llm-eval run suites/ --models mock,gpt-4o-mini -c 4 --rate-limit 2
```

## 6. Writing suites

Each case needs at least `id`, `prompt`, and usually `expected` + `scoring`.

```yaml
name: demo
default_scoring: exact
cases:
  - id: capital-fr
    prompt: "Capital of France?"
    expected: "Paris"
    scoring: exact
  - id: json-user
    prompt: "Return JSON {\"name\": \"Ada\"}"
    scoring: json_schema
    expected:
      type: object
      required: [name]
      properties:
        name: { type: string }
```

Shipped sample suites under `suites/` (70 cases total):

| Suite | Focus |
| --- | --- |
| `general_qa.yaml` | Short factual QA |
| `json_extraction.yaml` | Structured JSON extraction |
| `translation_sanity.yaml` | Darija / French / English sanity checks |

## 7. Scoring methods

| Method | Meaning |
| --- | --- |
| `exact` | String equality (optional case fold / strip) |
| `regex` | `re.search` against output |
| `json_schema` | Parse JSON + validate Draft 2020-12 schema |
| `contains` / `not_contains` | Required / forbidden substrings |
| `similarity` | RapidFuzz ratio / partial / token_set vs threshold |
| `llm_judge` | Separate judge model scores with a rubric |

**LLM-as-judge bias warning:** Judge models inherit their own preferences and can favor verbose or stylistically similar answers. Treat judge scores as noisy secondary signals, keep rubrics short and binary when possible, and always inspect failures in the HTML drill-down. Prefer deterministic scorers for regression gates.

## 8. CLI reference

```bash
llm-eval version
llm-eval run SUITES --models mock[,gpt-4o-mini] [--out DIR] [--db PATH]
                   [--concurrency N] [--rate-limit R] [--judge-model MODEL]
                   [--prices config/prices.yaml] [--temperature T] [--max-tokens N]
llm-eval list-runs [--db PATH]
llm-eval compare BASELINE CANDIDATE   # run id or results.json path
```

## 9. Reports & compare

- Terminal: Rich accuracy / latency / cost table
- HTML: self-contained, sortable columns, filters, per-case expand, bar charts
- JSON: full machine-readable dump
- SQLite: `results/evals.db` by default (`LLM_EVAL_DB_PATH`)

```bash
llm-eval compare reports/a/results.json reports/b/results.json
```

Exit code `1` if any pass→fail regressions are found.

## 10. Docker

```bash
docker compose up --build
# or
docker build -t llm-eval-harness:local .
docker run --rm -v ${PWD}/reports:/app/reports llm-eval-harness:local \
  run suites/ --models mock --out reports/docker-mock --judge-model mock
```

## 11. Development

```bash
make dev      # pip install -e ".[dev]"
make test     # pytest
make lint     # ruff check
make build    # python -m build
```

CI (`.github/workflows/ci.yml`) runs ruff, pytest, and a mock CLI smoke on every push/PR to `main`.

### Verified mock run (real numbers)

Command (2026-11-08):

```bash
py -3.12 -m llm_eval run suites/ --models mock --out reports/mock-run --judge-model mock --db results/evals.db
```

| Metric | Value |
| --- | --- |
| Run id | `6bfda768435d` |
| Timestamp (UTC) | `2026-11-08T20:34:15Z` |
| Suites / cases | 3 / 70 |
| Accuracy | **100.0%** (70/70) |
| Mean score | 1.000 |
| Mean latency | ~8.0 ms |
| Total tokens (fake mock counts) | 898 |
| Estimated cost | $0.000000 |

> Numbers above come from that exact local command/date. Re-run after suite or mock changes; never invent benchmark figures.

## 12. License & topics

MIT © Mohamed-Taqy Salmi

Suggested GitHub topics: `llm`, `evaluation`, `benchmark`, `openai-compatible`, `ollama`, `typer`, `pydantic`, `sqlite`, `html-report`
