# TunnelVision

> Exact Bayesian variable selection with quantum tunneling proposals. Quantum dreams up the moves; Metropolis keeps them honest.

TunnelVision is a research codebase exploring **quantum-enhanced MCMC** (Layden et al., [Nature 619, 282 (2023)](https://arxiv.org/abs/2203.12497)) as a proposal engine for **spike-and-slab Bayesian variable selection**. The quantum device only *proposes* moves; a classical Metropolis–Hastings accept/reject step evaluates the *exact* posterior, so the sampler remains unbiased no matter how noisy the hardware is. Noise can shrink the speedup — it can never corrupt the answer.

## The two research claims

1. **TunnelVision (N1):** Layden-style quench proposals, driven by a cheap 2-local Ising *surrogate* of the posterior, mix faster than the field-standard add-delete-swap kernel on correlated-design variable selection — while sampling the exact posterior.
2. **Maxwell's Daemon (N2):** hardware noise, deliberately *scheduled* (dynamical decoupling on/off, twirling levels, idle insertion), acts as a proposal-temperature ladder — hardware-native tempering with exactness intact.

See `docs/` for the research background (DEEPDIVE-01, NOVELTY-CHECK-01), architecture, and roadmap.

## Layout

```
src/tunnelvision/        the package
  engine.py              Metropolis–Hastings engine, kernel-agnostic (the exactness boundary)
  surrogate.py           2-local Ising surrogate construction from (X, y)
  diagnostics.py         spectral gap (exact, n<=~14), ESS, autocorrelation, PIP error
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
pip install -e ".[dev]"
pytest                       # exactness invariants must pass before anything else
python -m experiments.E01_layden_repro.run    # reproduce the Layden speedup curve
```

## Ground rules

- **The engine never trusts a kernel.** Correctness lives entirely in `engine.py`; kernels are allowed to be wrong, slow, or noisy.
- **Every experiment is a script + config + seed**, reproducible from a clean checkout.
- **Baselines are the real tools** (add-delete-swap), never strawmen.

## Status

Scaffold. Implementation begins with `engine.py` + classical kernels + exact spectral-gap diagnostics (Rung 1 of the roadmap).
