"""Global depolarizing channel and Aer per-gate validation.

qiskit stays in this file and in kernels/quantum.py. The analytic
Q_lambda formula is the exact-tier noise model; Aer is only here to
show that formula is a credible proxy for per-gate depolarizing.
"""

from __future__ import annotations

import numpy as np
import pytest

from tunnelvision.engine import MetropolisEngine
from tunnelvision.kernels.quantum import (
    DepolarizedQuenchKernel,
    QuenchKernel,
    aer_noisy_proposal_matrix,
    depolarize_proposal,
    fit_global_depolarize,
)
from tunnelvision.targets.ising import random_spin_glass


def _small_quench(n: int = 4, seed: int = 0, **kwargs: object) -> QuenchKernel:
    target = random_spin_glass(n, topology="all-to-all", seed=seed, temperature=1.0)
    defaults: dict[str, object] = {
        "n_gamma": 3,
        "n_t": 3,
        "evolution": "exact",
        "backend": "statevector",
    }
    defaults.update(kwargs)
    return QuenchKernel(target.h, target.J, **defaults)  # type: ignore[arg-type]


def test_depolarize_zero_is_identity() -> None:
    kernel = _small_quench()
    q = kernel.proposal_matrix(4)
    np.testing.assert_allclose(depolarize_proposal(q, 0.0), q, atol=1e-12, rtol=0.0)


def test_depolarize_one_is_uniform() -> None:
    kernel = _small_quench()
    q = depolarize_proposal(kernel.proposal_matrix(4), 1.0)
    np.testing.assert_allclose(q, np.full((16, 16), 1.0 / 16.0), atol=1e-12, rtol=0.0)


def test_depolarize_is_symmetric_and_stochastic() -> None:
    kernel = _small_quench()
    q = depolarize_proposal(kernel.proposal_matrix(4), 0.02)
    np.testing.assert_allclose(q.sum(axis=1), 1.0, atol=1e-12, rtol=0.0)
    np.testing.assert_allclose(q, q.T, atol=1e-10, rtol=0.0)
    assert np.all(q >= -1e-15)


def test_fit_global_depolarize_recovers_lambda() -> None:
    kernel = _small_quench()
    q = kernel.proposal_matrix(4)
    for lam in (0.0, 1e-3, 0.02, 0.1, 1.0):
        mixed = depolarize_proposal(q, lam)
        fitted = fit_global_depolarize(q, mixed)
        np.testing.assert_allclose(fitted, lam, atol=1e-12, rtol=0.0)


def test_depolarized_kernel_proposal_matches_formula() -> None:
    inner = _small_quench()
    kernel = DepolarizedQuenchKernel(inner, 0.05)
    expected = depolarize_proposal(inner.proposal_matrix(4), 0.05)
    np.testing.assert_allclose(kernel.proposal_matrix(4), expected, atol=1e-12, rtol=0.0)


def test_depolarized_kernel_rejects_bad_lambda() -> None:
    inner = _small_quench()
    with pytest.raises(ValueError, match="depolarize"):
        DepolarizedQuenchKernel(inner, -0.1)
    with pytest.raises(ValueError, match="depolarize"):
        DepolarizedQuenchKernel(inner, 1.5)


def test_depolarized_kernel_stationarity() -> None:
    target = random_spin_glass(5, topology="all-to-all", seed=1, temperature=1.0)
    inner = QuenchKernel(target.h, target.J, n_gamma=3, n_t=3, evolution="exact")
    kernel = DepolarizedQuenchKernel(inner, 0.02)
    P = MetropolisEngine(target, kernel).transition_matrix()
    pi = target.enumerate_exact()
    np.testing.assert_allclose(pi @ P, pi, atol=1e-12, rtol=0.0)
    balance = pi[:, None] * P
    np.testing.assert_allclose(balance, balance.T, atol=1e-12, rtol=0.0)


def test_depolarized_propose_is_symmetric_log_q() -> None:
    inner = _small_quench(n=3)
    kernel = DepolarizedQuenchKernel(inner, 0.4)
    rng = np.random.Generator(np.random.PCG64(0))
    y, log_fwd, log_rev = kernel.propose(np.zeros(3, dtype=np.uint8), rng)
    assert y.shape == (3,)
    assert set(y.tolist()) <= {0, 1}
    assert log_fwd == 0.0 and log_rev == 0.0


def test_aer_noiseless_tracks_trotter_q() -> None:
    inner = _small_quench(n=3, seed=2, evolution="trotter", n_gamma=2, n_t=2)
    q_sv = inner.proposal_matrix(3)
    q_aer = aer_noisy_proposal_matrix(inner, 0.0, n_gamma=2, n_t=2)
    np.testing.assert_allclose(q_aer, q_sv, atol=1e-8, rtol=0.0)
    np.testing.assert_allclose(fit_global_depolarize(q_sv, q_aer), 0.0, atol=1e-8)


def test_aer_noise_increases_effective_lambda() -> None:
    inner = _small_quench(n=3, seed=3, evolution="trotter", n_gamma=2, n_t=2)
    q_sv = inner.proposal_matrix(3)
    q_noisy = aer_noisy_proposal_matrix(inner, 0.02, n_gamma=2, n_t=2)
    fitted = fit_global_depolarize(q_sv, q_noisy)
    assert fitted > 0.01
    np.testing.assert_allclose(q_noisy.sum(axis=1), 1.0, atol=1e-10, rtol=0.0)
    # Per-gate noise after a palindromic circuit is unital but only
    # approximately symmetric. The analytic λ model is exactly Q = Q.T;
    # that is why the p=10 sweep uses it, not raw Aer Q.
    assert float(np.max(np.abs(q_noisy - q_noisy.T))) < 5e-3
