"""Unit tests for the mixing scoreboard (spectral gap, IAT, ESS, PIP error)."""

from __future__ import annotations

import numpy as np
import pytest

from tunnelvision.diagnostics import (
    ess,
    integrated_autocorrelation_time,
    pip_error,
    rhat,
    spectral_gap,
)
from tunnelvision.engine import MetropolisEngine
from tunnelvision.kernels.classical import UniformFlip
from tunnelvision.targets.ising import random_spin_glass


def test_spectral_gap_two_state_known() -> None:
    """P = [[0.2, 0.8], [0.4, 0.6]] has eigenvalues 1 and -0.2, so δ = 0.8."""
    P = np.array([[0.2, 0.8], [0.4, 0.6]], dtype=np.float64)
    pi = np.array([1.0 / 3.0, 2.0 / 3.0])
    np.testing.assert_allclose(spectral_gap(P, pi), 0.8, atol=1e-12)


def test_spectral_gap_identity_is_zero() -> None:
    P = np.eye(4)
    pi = np.full(4, 0.25)
    assert spectral_gap(P, pi) == 0.0


def test_spectral_gap_rejects_failed_stationarity() -> None:
    P = np.array([[0.2, 0.8], [0.4, 0.6]], dtype=np.float64)
    with pytest.raises(ValueError, match="stationarity failed"):
        spectral_gap(P, np.array([0.9, 0.1]))


def test_spectral_gap_on_engine_matrix() -> None:
    target = random_spin_glass(5, topology="all-to-all", seed=0, temperature=1.0)
    P = MetropolisEngine(target, UniformFlip()).transition_matrix()
    gap = spectral_gap(P, target.enumerate_exact())
    assert 0.0 < gap <= 1.0


def test_iat_of_iid_is_near_one() -> None:
    rng = np.random.Generator(np.random.PCG64(0))
    x = rng.standard_normal(8_000)
    iat = integrated_autocorrelation_time(x)
    assert 1.0 <= iat < 1.4


def test_iat_of_ar1_matches_theory() -> None:
    """AR(1) with φ=0.8 has τ = (1+φ)/(1-φ) = 9. Windowing must recover that."""
    rng = np.random.Generator(np.random.PCG64(1))
    phi = 0.8
    n = 20_000
    eps = rng.standard_normal(n)
    x = np.empty(n, dtype=np.float64)
    x[0] = eps[0] / np.sqrt(1.0 - phi**2)
    for t in range(1, n):
        x[t] = phi * x[t - 1] + eps[t]
    iat = integrated_autocorrelation_time(x)
    assert 6.0 < iat < 14.0


def test_iat_constant_series_is_one() -> None:
    assert integrated_autocorrelation_time(np.ones(50)) == 1.0


def test_ess_iid() -> None:
    rng = np.random.Generator(np.random.PCG64(2))
    x = rng.standard_normal(4_000)
    n_eff = ess(x)
    assert 0.7 * x.size < n_eff <= x.size


def test_rhat_iid_is_near_one() -> None:
    rng = np.random.Generator(np.random.PCG64(3))
    chains = rng.standard_normal((4, 4_000))
    value = rhat(chains)
    assert 0.99 < value < 1.05


def test_rhat_stuck_modes_is_large() -> None:
    """Four chains parked at two different constants — the high-ρ failure mode."""
    chains = np.array(
        [
            np.full(200, 0.0),
            np.full(200, 0.0),
            np.full(200, 10.0),
            np.full(200, 10.0),
        ],
        dtype=np.float64,
    )
    assert rhat(chains) > 5.0


def test_rhat_identical_constants_is_one() -> None:
    assert rhat(np.ones((3, 50))) == 1.0


def test_rhat_rejects_too_few_chains() -> None:
    with pytest.raises(ValueError, match="at least 2 chains"):
        rhat(np.zeros((1, 20)))


def test_rhat_rejects_short_chains() -> None:
    with pytest.raises(ValueError, match="at least 4 draws"):
        rhat(np.zeros((2, 3)))


def test_pip_error_zero_when_empirical_matches() -> None:
    states = np.array([[1, 0], [1, 0], [0, 1]], dtype=np.uint8)
    exact = np.array([2.0 / 3.0, 1.0 / 3.0])
    assert pip_error(states, exact) == 0.0


def test_pip_error_max_abs() -> None:
    states = np.ones((10, 2), dtype=np.uint8)
    exact = np.array([0.5, 0.25])
    np.testing.assert_allclose(pip_error(states, exact), 0.75)
