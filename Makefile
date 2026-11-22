PY ?= python3
VENV ?= .venv
BIN = $(VENV)/bin

.PHONY: install lint typecheck test smoke dashboard api acceptance clean-data

install:
	$(PY) -m venv $(VENV)
	$(BIN)/pip install -U pip
	$(BIN)/pip install -e ".[dev]"

lint:
	$(BIN)/ruff check .

typecheck:
	$(BIN)/mypy

test:
	$(BIN)/pytest -q

smoke:
	bash scripts/smoke_test.sh

acceptance:
	$(BIN)/python scripts/acceptance_test.py

dashboard:
	$(BIN)/streamlit run apps/dashboard/streamlit_app.py

api:
	$(BIN)/uvicorn quantlab.api.app:app --host 127.0.0.1 --port 8000

clean-data:
	rm -rf data
