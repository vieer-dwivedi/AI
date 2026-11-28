.PHONY: setup seed run test eval eval-safety lint clean

# Provider extra: make setup EXTRA=openai (default: anthropic)
EXTRA ?= anthropic

setup:
	uv sync --extra $(EXTRA)

seed:
	uv run python data/seed.py
	uv run python -m ingest.contracts

run:
	docker compose up --build

test:
	uv run pytest

lint:
	uv run ruff check .

eval:
	uv run python evals/run_evals.py

eval-safety:
	uv run python evals/run_evals.py --suite safety

clean:
	rm -rf data/erp.db data/chroma traces .pytest_cache .ruff_cache
