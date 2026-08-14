"""Exactness invariants — the tests that guard the project's core claim.

These must pass for EVERY kernel, including deliberately broken ones.
If a new kernel fails these, the kernel is fine and the engine is broken,
or the kernel is lying about its proposal probabilities. Either way: stop.
"""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

from tunnelvision.bits import all_binary_states, states_to_indices
from tunnelvision.engine import MetropolisEngine
from tunnelvision.kernels.base import Kernel
from tunnelvision.kernels.classical import AddDeleteSwap, SingleFlip, UniformFlip
from tunnelvision.targets.ising import IsingTarget, random_spin_glass
from tunnelvision.targets.spike_slab import SpikeSlabTarget

REPO_ROOT = Path(__file__).resolve().parents[1]
EXACT_SIDE_PATHS = [
    REPO_ROOT / "src" / "tunnelvision" / "engine.py",
    REPO_ROOT / "src" / "tunnelvision" / "diagnostics.py",
    REPO_ROOT / "src" / "tunnelvision" / "surrogate.py",
    REPO_ROOT / "src" / "tunnelvision" / "design.py",
    REPO_ROOT / "src" / "tunnelvision" / "bits.py",
    REPO_ROOT / "src" / "tunnelvision" / "tempering.py",
    *sorted((REPO_ROOT / "src" / "tunnelvision" / "targets").glob("*.py")),
    *sorted((REPO_ROOT / "src" / "tunnelvision" / "data").glob("*.py")),
    REPO_ROOT / "tests" / "test_exactness.py",
]


class IndependentBernoulli(Kernel):
    """Proposes from Bern(p) bits, independent of x and of the target.

    This is a terrible proposal for any interesting posterior. It is also
    the engine-bug detector: honest log-q on a wrong distribution must
    still leave the chain exact. If it does not, stop — the engine is
    broken (roadmap WP4, test 4).
    """

    name = "independent-bernoulli"

    def __init__(self, p: float = 0.8) -> None:
        if not 0.0 < p < 1.0:
            raise ValueError(f"p must be in (0, 1), got {p}")
        self.p = float(p)
        self._log_p = float(np.log(self.p))
        self._log_1mp = float(np.log(1.0 - self.p))

    def _log_q_of(self, z: np.ndarray) -> float:
        ones = float(np.asarray(z).sum())
        n_vars = float(z.shape[0])
        return ones * self._log_p + (n_vars - ones) * self._log_1mp

    def propose(
        self, x: np.ndarray, rng: np.random.Generator
    ) -> tuple[np.ndarray, float, float]:
        n_vars = int(np.asarray(x).shape[0])
        y = (rng.random(n_vars) < self.p).astype(np.uint8)
        return y, self._log_q_of(y), self._log_q_of(x)

    def proposal_matrix(self, n_vars: int) -> np.ndarray:
        states = all_binary_states(n_vars)
        ones = states.sum(axis=1).astype(np.float64)
        q = (self.p**ones) * ((1.0 - self.p) ** (n_vars - ones))
        return np.tile(q, (1 << n_vars, 1))


def _classical_kernels() -> list[Kernel]:
    return [UniformFlip(), SingleFlip(), AddDeleteSwap()]


def _all_test_kernels() -> list[Kernel]:
    return [*_classical_kernels(), IndependentBernoulli(0.8)]


def _small_ising() -> IsingTarget:
    return random_spin_glass(6, topology="all-to-all", seed=0, temperature=1.0)


def _tv_distance(p: np.ndarray, q: np.ndarray) -> float:
    return 0.5 * float(np.abs(p - q).sum())


def _empirical_distribution(states: np.ndarray, n_vars: int) -> np.ndarray:
    counts = np.bincount(states_to_indices(states), minlength=1 << n_vars)
    return counts.astype(np.float64) / counts.sum()


@pytest.mark.parametrize("kernel", _all_test_kernels(), ids=lambda k: k.name)
def test_stationarity_exact(kernel: Kernel) -> None:
    """Invariant 1: π @ P = π to 1e-12 on a small target."""
    target = _small_ising()
    P = MetropolisEngine(target, kernel).transition_matrix()
    pi = target.enumerate_exact()
    residual = pi @ P - pi
    np.testing.assert_allclose(residual, 0.0, atol=1e-12, rtol=0.0)


def test_stationarity_exact_spike_slab() -> None:
    """Same invariant on the g-prior target, not just Ising."""
    rng = np.random.Generator(np.random.PCG64(1))
    X = rng.standard_normal((20, 4))
    y = X[:, 0] + 0.1 * rng.standard_normal(20)
    target = SpikeSlabTarget(X, y, prior_inclusion=0.4)
    for kernel in _classical_kernels():
        P = MetropolisEngine(target, kernel).transition_matrix()
        pi = target.enumerate_exact()
        np.testing.assert_allclose(pi @ P, pi, atol=1e-12, rtol=0.0)


@pytest.mark.parametrize("kernel", _all_test_kernels(), ids=lambda k: k.name)
def test_detailed_balance_exact(kernel: Kernel) -> None:
    """Invariant 2: π_x P[x,y] = π_y P[y,x] for reversible MH kernels."""
    target = _small_ising()
    P = MetropolisEngine(target, kernel).transition_matrix()
    pi = target.enumerate_exact()
    balance = pi[:, None] * P
    np.testing.assert_allclose(balance, balance.T, atol=1e-12, rtol=0.0)


@pytest.mark.parametrize("kernel", _all_test_kernels(), ids=lambda k: k.name)
def test_sampled_distribution_matches_enumeration(kernel: Kernel) -> None:
    """Invariant 3: a long chain's TV distance to the exact law is small."""
    target = _small_ising()
    engine = MetropolisEngine(target, kernel)
    x0 = np.zeros(target.n_vars, dtype=np.uint8)
    result = engine.run(n_steps=40_000, x0=x0, seed=7)
    burned = result.states[5_000:]
    empirical = _empirical_distribution(burned, target.n_vars)
    tv = _tv_distance(empirical, target.enumerate_exact())
    assert tv < 0.12, f"{kernel.name} TV distance {tv:.4f} exceeded tolerance"


def test_broken_kernel_still_exact() -> None:
    """Invariant 4: honest log-q on a wrong proposal is still exact.

    This is the test that catches engine bugs. A failure here is not a
    kernel problem — the engine's Hastings ratio is wrong, full stop.
    """
    kernel = IndependentBernoulli(0.8)
    target = _small_ising()
    engine = MetropolisEngine(target, kernel)
    P = engine.transition_matrix()
    pi = target.enumerate_exact()
    np.testing.assert_allclose(pi @ P, pi, atol=1e-12, rtol=0.0)
    balance = pi[:, None] * P
    np.testing.assert_allclose(balance, balance.T, atol=1e-12, rtol=0.0)

    result = engine.run(n_steps=40_000, x0=np.zeros(target.n_vars, dtype=np.uint8), seed=3)
    empirical = _empirical_distribution(result.states[5_000:], target.n_vars)
    assert _tv_distance(empirical, pi) < 0.12


def test_asymmetric_kernel_logq() -> None:
    """Invariant 5: AddDeleteSwap log-q at empty / full / |γ|=1."""
    kernel = AddDeleteSwap()
    n_vars = 5
    rng = np.random.Generator(np.random.PCG64(0))

    empty = np.zeros(n_vars, dtype=np.uint8)
    y, log_q_fwd, log_q_rev = kernel.propose(empty, rng)
    assert int(y.sum()) == 1
    # Only add is available: q(y|empty) = 1/p.
    np.testing.assert_allclose(log_q_fwd, -np.log(n_vars))
    # Reverse is a delete from |γ|=1, one of three types, one included bit.
    np.testing.assert_allclose(log_q_rev, -np.log(3.0))
    np.testing.assert_allclose(log_q_fwd, kernel.log_q(empty, y))
    np.testing.assert_allclose(log_q_rev, kernel.log_q(y, empty))

    full = np.ones(n_vars, dtype=np.uint8)
    y, log_q_fwd, log_q_rev = kernel.propose(full, rng)
    assert int(y.sum()) == n_vars - 1
    np.testing.assert_allclose(log_q_fwd, -np.log(n_vars))
    # Reverse is an add from |γ|=p-1, one of three types, one excluded bit.
    np.testing.assert_allclose(log_q_rev, -np.log(3.0))

    one = np.zeros(n_vars, dtype=np.uint8)
    one[0] = 1
    for _ in range(40):
        y, log_q_fwd, log_q_rev = kernel.propose(one, rng)
        np.testing.assert_allclose(log_q_fwd, kernel.log_q(one, y))
        np.testing.assert_allclose(log_q_rev, kernel.log_q(y, one))
        delta = int(y.sum()) - 1
        if delta == 1:
            # add: (1/3) / (p-1)
            np.testing.assert_allclose(log_q_fwd, -np.log(3.0) - np.log(n_vars - 1))
            np.testing.assert_allclose(log_q_rev, -np.log(3.0) - np.log(2.0))
        elif delta == -1:
            # delete back to empty: (1/3) / 1  vs  reverse add 1/p
            np.testing.assert_allclose(log_q_fwd, -np.log(3.0))
            np.testing.assert_allclose(log_q_rev, -np.log(n_vars))
        else:
            # swap: |γ| unchanged, q is symmetric
            np.testing.assert_allclose(log_q_fwd, -np.log(3.0) - np.log(n_vars - 1))
            np.testing.assert_allclose(log_q_fwd, log_q_rev)

    target = random_spin_glass(n_vars, topology="all-to-all", seed=2, temperature=1.0)
    P = MetropolisEngine(target, kernel).transition_matrix()
    pi = target.enumerate_exact()
    np.testing.assert_allclose(pi @ P, pi, atol=1e-12, rtol=0.0)


def test_ads_proposal_matrix_matches_log_q() -> None:
    """proposal_matrix and log_q are the same accounting, two ways."""
    kernel = AddDeleteSwap()
    n_vars = 4
    Q = kernel.proposal_matrix(n_vars)
    np.testing.assert_allclose(Q.sum(axis=1), 1.0, atol=1e-12)
    states = all_binary_states(n_vars)
    for i, x in enumerate(states):
        for j, y in enumerate(states):
            log_q = kernel.log_q(x, y)
            if Q[i, j] == 0.0:
                assert log_q == -np.inf
            else:
                np.testing.assert_allclose(log_q, np.log(Q[i, j]), atol=1e-12)


def test_transition_matrix_rejection_mass_on_diagonal() -> None:
    """The classic bug: forgetting the self-loop, rows fail to sum to 1."""
    target = _small_ising()
    P = MetropolisEngine(target, SingleFlip()).transition_matrix()
    np.testing.assert_allclose(P.sum(axis=1), 1.0, atol=1e-12, rtol=0.0)
    assert np.all(np.diag(P) >= -1e-15)


def test_exact_side_imports_no_qiskit() -> None:
    """Correctness modules must stay numpy/scipy — no quantum imports."""
    for path in EXACT_SIDE_PATHS:
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            names: list[str] = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module is not None:
                names = [node.module]
            for name in names:
                assert "qiskit" not in name.split("."), f"{path} imports {name}"
