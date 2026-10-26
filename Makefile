.PHONY: install run test lint index scrape refresh-kb docker eval robustness load

install:
	python -m pip install -e ".[dev]"

run:
	python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

test:
	python -m pytest -q

lint:
	python -m ruff check app scripts tests

index:
	python -m scripts.build_index

scrape:
	python -m scripts.scrape_knowledge_base

refresh-kb: scrape index

docker:
	docker compose up --build

eval:
	python -m scripts.evaluate_voice_agent

robustness:
	python -m scripts.evaluate_voice_agent --dataset evals/robustness_cases.jsonl

load:
	python -m scripts.load_test_websocket
