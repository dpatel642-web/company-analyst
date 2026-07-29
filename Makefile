PY ?= /opt/homebrew/bin/python3.14
VENV := .venv
BIN := $(VENV)/bin

.PHONY: venv test run clean

venv:
	$(PY) -m venv $(VENV)
	$(BIN)/pip install --quiet --upgrade pip
	$(BIN)/pip install --quiet -e '.[dev]'
	@$(BIN)/python -c "import sys; print('python', sys.version.split()[0])"

test:
	$(BIN)/pytest -q

run:
	$(BIN)/python scripts/fin642_run.py

clean:
	rm -rf $(VENV) out/* .pytest_cache
	find . -name __pycache__ -type d -exec rm -rf {} +
