.PHONY: install test lint fmt e01 e02 e03

install:
	pip install -e ".[dev]"

test:
	pytest

lint:
	ruff check .

fmt:
	ruff check --fix . && ruff format .

e01:
	python -m experiments.E01_layden_repro.run

e02:
	python -m experiments.E02_tunnelvision_demo.run

e03:
	python -m experiments.E03_maxwells_daemon.run
