.PHONY: setup seed run test eval eval-safety clean

setup:
	pip install -e ".[dev]"

seed:
	python data/seed.py
	python -m ingest.contracts

run:
	docker compose up --build

test:
	pytest

eval:
	python evals/run_evals.py

eval-safety:
	python evals/run_evals.py --suite safety

clean:
	rm -rf data/erp.db data/chroma traces .pytest_cache
