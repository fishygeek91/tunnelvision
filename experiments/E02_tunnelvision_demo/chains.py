"""Independent-chain scoring for E02 (exact-tier ρ-sweep and sampled tier).

Split-R̂ is the stuck-mode detector: a single long chain parked in one
mode looks converged; four chains that disagree do not. Both E02b and
E02c need the same accounting, so it lives here instead of being copied.

This module never builds a transition matrix. Callers that have an
enumerable Q (p ≤ 14) compute the gap themselves, then pass a tabulated
kernel into ``run_replicated_chains`` so ESS matches that Q. Callers
above the exact-tier cutoff pass the live kernel.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Any

import numpy as np

from tunnelvision.bits import states_to_indices
from tunnelvision.diagnostics import ess, pip_error, rhat
from tunnelvision.engine import MetropolisEngine


def rho_chain_seed(base: int, rho: float, chain_index: int) -> int:
    """Seed used by the E02b ρ-sweep.

    Integer ρ-blocks so adding a correlation does not reshuffle the
    other cells. Do not change: existing ρ-sweep numbers were drawn
    from this map.
    """
    return int(base) + 10_000 * int(round(float(rho) * 100.0)) + int(chain_index)


def sampled_chain_seed(
    base: int,
    *,
    p: int,
    rho: float,
    kernel_id: int,
    chain_index: int,
) -> int:
    """Seed for a sampled-tier cell.

    p / ρ / kernel each get their own block so lengthening one chain
    or adding a kernel does not reshuffle the others.
    """
    return (
        int(base)
        + 1_000_000 * int(p)
        + 10_000 * int(round(float(rho) * 100.0))
        + 100 * int(kernel_id)
        + int(chain_index)
    )


def run_replicated_chains(
    engine: MetropolisEngine,
    *,
    n_steps: int,
    burn_in: int,
    seeds: Sequence[int],
    x0: np.ndarray | Sequence[np.ndarray],
    exact_pips: np.ndarray | None = None,
    exact_pi: np.ndarray | None = None,
) -> dict[str, Any]:
    """Run independent MH chains and score ESS, R̂, and optional PIP / TV.

    ``x0`` is either one shared start or one start per chain. The
    sampled tier uses independent random starts so four chains parked
    in different modes cannot hide behind a shared empty-model launch.
    Need ≥2 chains so split-R̂ is defined. ``exact_pi`` is the
    enumerated posterior; TV is how the E03 bias audit scores a
    non-unital proposal against the exact target.
    """
    n_chains = len(seeds)
    if n_chains < 2:
        raise ValueError(f"n_chains must be at least 2 for R̂, got {n_chains}")
    if burn_in >= n_steps:
        raise ValueError(f"burn_in ({burn_in}) must be < n_steps ({n_steps})")
    if n_steps < 1:
        raise ValueError(f"n_steps must be at least 1, got {n_steps}")

    if isinstance(x0, np.ndarray) and x0.ndim == 1:
        starts = [np.asarray(x0, dtype=np.uint8) for _ in seeds]
    else:
        starts = [np.asarray(start, dtype=np.uint8) for start in x0]
        if len(starts) != n_chains:
            raise ValueError(f"expected {n_chains} starts, got {len(starts)}")

    n_vars = int(engine.target.n_vars)
    size_series: list[np.ndarray] = []
    kept_states: list[np.ndarray] = []
    pip_errors: list[float] = []
    accepts: list[float] = []
    ess_size_vals: list[float] = []
    ess_pip_vals: list[float] = []
    elapsed_vals: list[float] = []
    n_kept = n_steps - burn_in

    for seed, start in zip(seeds, starts, strict=True):
        t0 = time.perf_counter()
        result = engine.run(n_steps=n_steps, x0=start, seed=int(seed))
        elapsed = time.perf_counter() - t0
        kept = result.states[burn_in:]
        size = kept.sum(axis=1).astype(np.float64)
        size_series.append(size)
        kept_states.append(kept)
        ess_size = float(ess(size))
        pip_ess = [float(ess(kept[:, j].astype(np.float64))) for j in range(n_vars)]
        ess_size_vals.append(ess_size)
        ess_pip_vals.append(float(np.mean(pip_ess)))
        if exact_pips is not None:
            pip_errors.append(float(pip_error(kept, exact_pips)))
        accepts.append(float(result.acceptance_rate))
        elapsed_vals.append(elapsed)

    sizes = np.stack(size_series, axis=0)
    pooled = np.concatenate(kept_states, axis=0)
    pip_estimate = pooled.astype(np.float64).mean(axis=0)
    pip_rhat = [
        float(rhat(np.stack([chain[:, j].astype(np.float64) for chain in kept_states], axis=0)))
        for j in range(n_vars)
    ]
    total_ess = float(np.sum(ess_size_vals))
    total_time = float(np.sum(elapsed_vals))
    scored: dict[str, Any] = {
        "acceptance_rate": float(np.mean(accepts)),
        "pip_estimate": pip_estimate.astype(np.float64).tolist(),
        "ess_size": float(np.mean(ess_size_vals)),
        "ess_size_per_step": float(np.mean(ess_size_vals)) / float(n_kept),
        "ess_size_per_sec": total_ess / total_time if total_time > 0.0 else float("inf"),
        "ess_pip_mean": float(np.mean(ess_pip_vals)),
        "ess_pip_per_step": float(np.mean(ess_pip_vals)) / float(n_kept),
        "ess_pip_per_sec": (
            float(np.sum(ess_pip_vals)) / total_time if total_time > 0.0 else float("inf")
        ),
        "rhat_size": float(rhat(sizes)),
        "rhat_pip_max": float(np.max(pip_rhat)),
        "wall_seconds": total_time,
        "n_steps": n_steps,
        "burn_in": burn_in,
        "n_chains": n_chains,
        "n_kept": n_kept,
    }
    if exact_pips is not None:
        scored["pip_error"] = float(np.mean(pip_errors))
        scored["pip_error_pooled"] = float(pip_error(pooled, exact_pips))
    else:
        scored["pip_error"] = None
        scored["pip_error_pooled"] = None
    if exact_pi is not None:
        pi = np.asarray(exact_pi, dtype=np.float64)
        if pi.ndim != 1 or pi.size != (1 << n_vars):
            raise ValueError(f"exact_pi must have length 2^{n_vars}, got {pi.shape}")
        counts = np.bincount(states_to_indices(pooled), minlength=int(pi.size))
        empirical = counts.astype(np.float64) / float(counts.sum())
        scored["tv_distance"] = 0.5 * float(np.abs(empirical - pi).sum())
    else:
        scored["tv_distance"] = None
    return scored
