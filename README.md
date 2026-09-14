# TunnelVision

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22758619.svg)](https://doi.org/10.5281/zenodo.22758619)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> Exact Bayesian variable selection with quantum tunneling proposals. Quantum proposes; Metropolis accepts. The mixing advantage, on this target, is not there.

TunnelVision is a research codebase that puts **quantum-enhanced MCMC** (Layden et al., [Nature 619, 282 (2023)](https://arxiv.org/abs/2203.12497)) inside an exact Metropolis–Hastings sampler for **spike-and-slab Bayesian variable selection**. The quantum device only *proposes*; accept/reject uses the exact g-prior. Unital noise can slow the chain without biasing it. Non-unital noise (amplitude damping / T1) can bias it if you still pretend the proposal is symmetric — we measured that.

The two claims we actually tested:

1. **N1.** A Layden quench driven by a 2-local Ising surrogate mixes faster than add-delete-swap on correlated-design variable selection. **It does not** (diabetes gap 0.40× ADS; ρ-sweep never crosses 1).
2. **N2.** Scheduled depolarizing noise is useful proposal-tempering. **It heats, and it does not help.**

Write-up: [`docs/paper/paper.md`](docs/paper/paper.md). Background and roadmap in `docs/`.

## Layout

```
src/tunnelvision/        the package
  engine.py              Metropolis–Hastings engine, kernel-agnostic (the exactness boundary)
  surrogate.py           2-local Ising surrogate construction from (X, y)
  diagnostics.py         spectral gap (exact, n<=~14), ESS, split-R̂, PIP error
  kernels/               proposal kernels: classical baselines + quantum quench + noise ladder
  targets/               target distributions: Ising Boltzmann, spike-and-slab posterior
  data/                  dataset loaders (diabetes p=10 first)
tests/                   exactness invariants + unit tests (see docs/ARCHITECTURE.md)
experiments/             E01 Layden reproduction, E02 diabetes demo, E03 noise ladder
docs/                    background, architecture, roadmap
results/                 experiment outputs (gitignored except summaries)
```

## Quickstart (once implemented)

```bash
uv sync --all-extras         # creates .venv and installs everything (uv.lock pins it)
uv run pytest                # exactness invariants must pass before anything else
uv run python -m experiments.E01_layden_repro.run   # reproduce the Layden speedup curve
```

This is a [uv](https://docs.astral.sh/uv/)-managed project — commit `uv.lock`; it is the
reproducibility record for every experiment. (`pip install -e ".[dev]"` still works if you must.)

## Ground rules

- **The engine never trusts a kernel.** Correctness lives entirely in `engine.py`; kernels are allowed to be wrong, slow, or noisy.
- **Every experiment is a script + config + seed**, reproducible from a clean checkout.
- **Baselines are the real tools** (add-delete-swap), never strawmen.

## Status

E01–E03 and the Aer bias audit are in. The experiment spine is done.
The remaining electives are a live IBM TV bound and Ferguson-style
coarse-graining. Draft: `docs/paper/paper.md`.

Update (v1.0.0): the **P01 PORTAL race** (`experiments/P01_portal/`) tested an
amortized architecture — a device- or solver-mined proposal library consumed by exact
mode-jumping — under pre-registered kill criteria; both kills fired (classical miners
match the quench's mode coverage at matched budget; darting gain < 2× over parallel
tempering at n = 50–200). A **free-certificate theorem** (`docs/free_certificate.md`)
shows conserved charges confine measurements to a symmetry sector at zero cost while
preserving evaluation-free exactness. This repo is archived alongside
[goldilocks](https://github.com/fishygeek91/goldilocks) at
**[10.5281/zenodo.22758619](https://doi.org/10.5281/zenodo.22758619)**.

