.PHONY: install test lint fmt e01 e02 e02-rho e02-ablations e03

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

e02-rho:
	uv run python -m experiments.E02_tunnelvision_demo.run_rho_sweep

e02-ablations:
	uv run python -m experiments.E02_tunnelvision_demo.run_ablations

e03:
	uv run python -m experiments.E03_maxwells_daemon.run
