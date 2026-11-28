.PHONY: setup seed run test eval eval-cases eval-safety lint clean

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

# eval / eval-safety need the mock carrier running: uv run uvicorn mock_services.carrier_api.main:app --port 8001
# Each case runs on a fresh seeded database. Reports go to evals/reports/. Exit code 1 if the gate fails.
eval-cases:
	uv run python evals/build_cases.py

eval:
	uv run python evals/run_evals.py

eval-safety:
	uv run python evals/run_evals.py --suite safety

clean:
	rm -rf data/erp.db data/chroma traces .pytest_cache .ruff_cache
