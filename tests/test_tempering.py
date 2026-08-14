"""Parallel tempering stays on the exact side of the wall.

Single-replica PT is ordinary MH. Multi-replica PT changes mixing, not
the cold-chain target. If either invariant fails, the swap ratio or the
tempered Hastings path is wrong — do not loosen the tolerances.
"""

from __future__ import annotations

import numpy as np
import pytest

from tunnelvision.bits import states_to_indices
from tunnelvision.diagnostics import pip_error
from tunnelvision.engine import MetropolisEngine
from tunnelvision.kernels.classical import AddDeleteSwap
from tunnelvision.targets.ising import random_spin_glass
from tunnelvision.targets.spike_slab import SpikeSlabTarget
from tunnelvision.tempering import (
    ParallelTempering,
    TemperedTarget,
    geometric_beta_ladder,
    replica_exchange_log_alpha,
)


def _small_spike_slab(n_vars: int = 6) -> SpikeSlabTarget:
    rng = np.random.Generator(np.random.PCG64(1))
    X = rng.standard_normal((30, n_vars))
    y = X[:, 0] + X[:, 1] + 0.2 * rng.standard_normal(30)
    return SpikeSlabTarget(X, y, prior_inclusion=0.4)


def _tv_distance(p: np.ndarray, q: np.ndarray) -> float:
    return 0.5 * float(np.abs(p - q).sum())


def _empirical_distribution(states: np.ndarray, n_vars: int) -> np.ndarray:
    counts = np.bincount(states_to_indices(states), minlength=1 << n_vars)
    return counts.astype(np.float64) / counts.sum()


def test_geometric_beta_ladder_endpoints_and_ratio() -> None:
    single = geometric_beta_ladder(1)
    np.testing.assert_array_equal(single, np.array([1.0]))
    betas = geometric_beta_ladder(5, beta_min=0.2)
    assert betas[0] == 1.0
    np.testing.assert_allclose(betas[-1], 0.2)
    ratios = betas[1:] / betas[:-1]
    np.testing.assert_allclose(ratios, ratios[0])
    with pytest.raises(ValueError, match="n_replicas"):
        geometric_beta_ladder(0)
    with pytest.raises(ValueError, match="beta_min"):
        geometric_beta_ladder(3, beta_min=0.0)


def test_replica_exchange_log_alpha_hand_computed() -> None:
    """Two-replica swap ratio, worked by hand.

    β_i=1, β_j=1/2, log p(x_i)=−2, log p(x_j)=−1:
        (1 − 1/2)(−1 − (−2)) = 1/2
    β_i=1, β_j=1/2, log p(x_i)=0, log p(x_j)=−4:
        (1 − 1/2)(−4 − 0) = −2
    """
    np.testing.assert_allclose(replica_exchange_log_alpha(1.0, 0.5, -2.0, -1.0), 0.5)
    np.testing.assert_allclose(replica_exchange_log_alpha(1.0, 0.5, 0.0, -4.0), -2.0)
    # Equal temperatures: swap is always accepted (log α = 0).
    np.testing.assert_allclose(replica_exchange_log_alpha(0.4, 0.4, 3.0, -8.0), 0.0)


def test_tempered_target_scales_exact_log_prob() -> None:
    target = random_spin_glass(4, topology="all-to-all", seed=0, temperature=1.0)
    x = np.array([1, 0, 1, 0], dtype=np.uint8)
    half = TemperedTarget(target, 0.5)
    np.testing.assert_allclose(half.log_prob(x), 0.5 * target.log_prob(x))
    np.testing.assert_allclose(
        TemperedTarget(target, 1.0).log_prob(x),
        target.log_prob(x),
    )
    states = np.array([[0, 0, 0, 0], [1, 1, 1, 1]], dtype=np.uint8)
    np.testing.assert_allclose(
        half.log_prob_batch(states),
        0.5 * target.log_prob_batch(states),
    )
    with pytest.raises(ValueError, match="beta"):
        TemperedTarget(target, 0.0)
    with pytest.raises(ValueError, match="beta"):
        TemperedTarget(target, -1.0)


def test_parallel_tempering_rejects_bad_ladders() -> None:
    target = _small_spike_slab(4)
    with pytest.raises(ValueError, match="cold replica"):
        ParallelTempering(target, betas=np.array([0.8, 0.4]))
    with pytest.raises(ValueError, match="strictly decreasing"):
        ParallelTempering(target, betas=np.array([1.0, 0.5, 0.5]))
    with pytest.raises(ValueError, match="positive"):
        ParallelTempering(target, betas=np.array([1.0, -0.2]))


def test_single_replica_matches_plain_mh() -> None:
    """K=1, β=[1] must be the same RNG stream as MetropolisEngine.run."""
    target = _small_spike_slab(6)
    kernel = AddDeleteSwap()
    x0 = np.zeros(target.n_vars, dtype=np.uint8)
    mh = MetropolisEngine(target, kernel).run(n_steps=800, x0=x0, seed=7)
    pt = ParallelTempering(target, kernel, betas=np.array([1.0])).run(
        n_steps=800, x0=x0, seed=7
    )
    np.testing.assert_array_equal(pt.states, mh.states)
    np.testing.assert_allclose(pt.log_probs, mh.log_probs)
    np.testing.assert_array_equal(pt.accepted, mh.accepted)


def test_single_replica_pips_match_enumeration() -> None:
    target = _small_spike_slab(6)
    kernel = AddDeleteSwap()
    x0 = np.zeros(target.n_vars, dtype=np.uint8)
    result = ParallelTempering(target, kernel, betas=np.array([1.0])).run(
        n_steps=25_000, x0=x0, seed=11
    )
    kept = result.states[4_000:]
    err = pip_error(kept, target.posterior_inclusion_probs_exact())
    assert err < 0.12, f"single-replica PIP error {err:.4f} exceeded tolerance"


def test_multi_replica_cold_chain_is_exact() -> None:
    """Swaps change mixing, not the β=1 target. PIP error is Monte Carlo."""
    target = _small_spike_slab(6)
    kernel = AddDeleteSwap()
    x0 = np.zeros(target.n_vars, dtype=np.uint8)
    pt = ParallelTempering(target, kernel, n_replicas=4, beta_min=0.25)
    result = pt.run(n_steps=25_000, x0=x0, seed=5)
    kept = result.states[4_000:]
    exact_pips = target.posterior_inclusion_probs_exact()
    err = pip_error(kept, exact_pips)
    assert err < 0.12, f"multi-replica PIP error {err:.4f} exceeded tolerance"
    empirical = _empirical_distribution(kept, target.n_vars)
    tv = _tv_distance(empirical, target.enumerate_exact())
    assert tv < 0.12, f"multi-replica TV distance {tv:.4f} exceeded tolerance"
    assert result.meta["n_replicas"] == 4
    assert result.meta["n_target_evals"] == 4 * 25_000 + 4


def test_parallel_tempering_is_deterministic() -> None:
    target = _small_spike_slab(5)
    x0 = np.zeros(target.n_vars, dtype=np.uint8)
    first = ParallelTempering(target, n_replicas=3, beta_min=0.3).run(
        n_steps=400, x0=x0, seed=19
    )
    second = ParallelTempering(target, n_replicas=3, beta_min=0.3).run(
        n_steps=400, x0=x0, seed=19
    )
    np.testing.assert_array_equal(first.states, second.states)
    np.testing.assert_allclose(first.log_probs, second.log_probs)
    other = ParallelTempering(target, n_replicas=3, beta_min=0.3).run(
        n_steps=400, x0=x0, seed=20
    )
    assert not np.array_equal(first.states, other.states)
