.PHONY: help install dev test lint format build run-mock docker-build docker-up clean

help:
	@echo "Targets: install | dev | test | lint | format | build | run-mock | docker-build | docker-up | clean"

install:
	python -m pip install -e .

dev:
	python -m pip install -e ".[dev]"

test:
	pytest -q

lint:
	ruff check src tests

format:
	ruff format src tests
	ruff check --fix src tests

build:
	python -m pip install build
	python -m build

run-mock:
	llm-eval run suites/ --models mock --out reports/mock-run

docker-build:
	docker build -t llm-eval-harness:local .

docker-up:
	docker compose up --build

clean:
	rm -rf build dist *.egg-info .pytest_cache .ruff_cache htmlcov .coverage results reports out
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
