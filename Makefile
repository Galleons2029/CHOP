PY ?= .venv/bin/python

.PHONY: test test-fast lint smoke dry-run

test:
	$(PY) -m pytest

test-fast:
	$(PY) -m pytest -m "not slow"

lint:
	$(PY) -m ruff check .

smoke:
	$(PY) -m chop.reference_trainer --config configs/smoke/tiny_math.yaml

dry-run:
	$(PY) scripts/reproduce.py --table all --dry-run
