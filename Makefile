.PHONY: install test lint fmt e01 e02 e03

install:
	uv sync --all-extras

test:
	uv run pytest

lint:
	uv run ruff check .

fmt:
	uv run ruff check --fix . && uv run ruff format .

e01:
	uv run python -m experiments.E01_layden_repro.run

e02:
	uv run python -m experiments.E02_tunnelvision_demo.run

e03:
	uv run python -m experiments.E03_maxwells_daemon.run
