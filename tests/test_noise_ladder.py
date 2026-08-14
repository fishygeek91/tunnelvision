"""NoiseLadderKernel: random-scan mixture stays exact.

A uniform mixture of fixed symmetric kernels is symmetric. If these
invariants fail, either a rung is lying about log-q or the ladder
is averaging the wrong Q. Do not loosen the tolerances.
"""

from __future__ import annotations

import numpy as np
import pytest

from tunnelvision.bits import states_to_indices
from tunnelvision.engine import MetropolisEngine
from tunnelvision.kernels.classical import AddDeleteSwap, SingleFlip, UniformFlip
from tunnelvision.kernels.noise_ladder import (
    AdaptiveMixture,
    NoiseLadderKernel,
    boltzmann_mean_energy,
    invert_temperature,
)
from tunnelvision.kernels.quantum import DepolarizedQuenchKernel, QuenchKernel
from tunnelvision.targets.ising import random_spin_glass


def _tv_distance(p: np.ndarray, q: np.ndarray) -> float:
    return 0.5 * float(np.abs(p - q).sum())


def _empirical_distribution(states: np.ndarray, n_vars: int) -> np.ndarray:
    counts = np.bincount(states_to_indices(states), minlength=1 << n_vars)
    return counts.astype(np.float64) / counts.sum()


def test_ladder_rejects_empty_and_adaptive_scan() -> None:
    with pytest.raises(ValueError, match="at least one rung"):
        NoiseLadderKernel([])
    with pytest.raises(ValueError, match="scan"):
        NoiseLadderKernel([UniformFlip()], scan="round-robin")


def test_adaptive_mixture_is_still_a_stub() -> None:
    with pytest.raises(NotImplementedError):
        AdaptiveMixture([UniformFlip()])


def test_ladder_proposal_is_the_rung_average() -> None:
    rungs = [UniformFlip(), SingleFlip()]
    ladder = NoiseLadderKernel(rungs)
    n_vars = 4
    expected = 0.5 * (rungs[0].proposal_matrix(n_vars) + rungs[1].proposal_matrix(n_vars))
    q = ladder.proposal_matrix(n_vars)
    np.testing.assert_allclose(q, expected, atol=1e-12, rtol=0.0)
    np.testing.assert_allclose(q.sum(axis=1), 1.0, atol=1e-12, rtol=0.0)
    np.testing.assert_allclose(q, q.T, atol=1e-12, rtol=0.0)


def test_depolarized_ladder_is_symmetric_and_stochastic() -> None:
    target = random_spin_glass(4, topology="all-to-all", seed=0, temperature=1.0)
    inner = QuenchKernel(target.h, target.J, n_gamma=3, n_t=3, evolution="exact")
    rungs = [DepolarizedQuenchKernel(inner, lam) for lam in (0.0, 0.02, 0.1)]
    ladder = NoiseLadderKernel(rungs)
    q = ladder.proposal_matrix(4)
    np.testing.assert_allclose(q.sum(axis=1), 1.0, atol=1e-12, rtol=0.0)
    np.testing.assert_allclose(q, q.T, atol=1e-10, rtol=0.0)


def test_ladder_stationarity_and_detailed_balance() -> None:
    target = random_spin_glass(5, topology="all-to-all", seed=1, temperature=1.0)
    inner = QuenchKernel(target.h, target.J, n_gamma=3, n_t=3, evolution="exact")
    ladder = NoiseLadderKernel(
        [DepolarizedQuenchKernel(inner, lam) for lam in (0.0, 0.05, 0.2)]
    )
    P = MetropolisEngine(target, ladder).transition_matrix()
    pi = target.enumerate_exact()
    np.testing.assert_allclose(pi @ P, pi, atol=1e-12, rtol=0.0)
    balance = pi[:, None] * P
    np.testing.assert_allclose(balance, balance.T, atol=1e-12, rtol=0.0)


def test_ladder_sampled_distribution_matches_enumeration() -> None:
    target = random_spin_glass(4, topology="all-to-all", seed=2, temperature=1.0)
    inner = QuenchKernel(target.h, target.J, n_gamma=3, n_t=3, evolution="exact")
    ladder = NoiseLadderKernel(
        [DepolarizedQuenchKernel(inner, lam) for lam in (0.0, 0.1)]
    )
    result = MetropolisEngine(target, ladder).run(
        n_steps=30_000, x0=np.zeros(target.n_vars, dtype=np.uint8), seed=5
    )
    empirical = _empirical_distribution(result.states[4_000:], target.n_vars)
    tv = _tv_distance(empirical, target.enumerate_exact())
    assert tv < 0.12, f"ladder TV distance {tv:.4f} exceeded tolerance"


def test_invert_temperature_recovers_boltzmann_mean() -> None:
    energies = np.array([0.0, 1.0, 2.0, 5.0], dtype=np.float64)
    for temperature in (0.5, 1.0, 2.0, 10.0):
        mean = boltzmann_mean_energy(energies, temperature)
        recovered = invert_temperature(energies, mean)
        np.testing.assert_allclose(recovered, temperature, rtol=1e-4, atol=1e-4)
    assert invert_temperature(energies, float(energies.min())) == 0.0
    assert invert_temperature(energies, float(energies.mean())) == float("inf")


def test_ladder_refuses_asymmetric_rung_log_q() -> None:
    """ADS is asymmetric at the empty model; the mixture cannot inherit its log-q."""
    ladder = NoiseLadderKernel([AddDeleteSwap()])
    rng = np.random.Generator(np.random.PCG64(0))
    with pytest.raises(RuntimeError, match="not symmetric"):
        ladder.propose(np.zeros(4, dtype=np.uint8), rng)
