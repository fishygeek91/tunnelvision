"""WP5: the surrogate may be wrong, but it must be the *right kind* of wrong.

Scale, correlation with the exact log-posterior, and the exact/heuristic
wall. A surrogate that points the quench at the wrong landscape is a
failed experiment, not a failed sampler — catch it here, before E02.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tunnelvision.bits import all_binary_states, state_to_index
from tunnelvision.design import center_and_scale
from tunnelvision.diagnostics import pip_error
from tunnelvision.engine import MetropolisEngine
from tunnelvision.kernels.base import Kernel
from tunnelvision.surrogate import (
    IsingSurrogate,
    corrupt_surrogate,
    diagnose_surrogate,
    ising_surrogate_from_data,
    learned_surrogate,
    problem_frobenius_norm,
    write_energy_logprob_scatter,
)
from tunnelvision.targets.ising import random_spin_glass
from tunnelvision.targets.spike_slab import SpikeSlabTarget


def _clear_signal(n_obs: int = 80, n_vars: int = 8, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Near-orthogonal design, y driven by the first two columns."""
    rng = np.random.Generator(np.random.PCG64(seed))
    X = rng.standard_normal((n_obs, n_vars))
    y = X[:, 0] + X[:, 1] + 0.15 * rng.standard_normal(n_obs)
    return X, y


class SurrogateIndependence(Kernel):
    """Independence MH from the surrogate Boltzmann law.

    A terrible or wonderful proposal, depending on the surrogate — and
    the engine-bug detector for the wall: two different (h, J) must
    leave the posterior alone.
    """

    name = "surrogate-independence"

    def __init__(self, surrogate: IsingSurrogate) -> None:
        self.n_vars = surrogate.n_vars
        self._states = all_binary_states(self.n_vars)
        logq = -surrogate.energies(self._states)
        logq = logq - np.max(logq)
        mass = np.exp(logq)
        mass /= mass.sum()
        self._mass = mass
        self._logq = np.log(mass)

    def propose(self, x: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, float, float]:
        idx_y = int(rng.choice(self._mass.size, p=self._mass))
        y = self._states[idx_y]
        idx_x = state_to_index(x)
        return y, float(self._logq[idx_y]), float(self._logq[idx_x])

    def proposal_matrix(self, n_vars: int) -> np.ndarray:
        if n_vars != self.n_vars:
            raise ValueError(f"expected n_vars={self.n_vars}, got {n_vars}")
        return np.tile(self._mass, (1 << n_vars, 1))


def test_design_lockstep_with_spike_slab() -> None:
    """If centering drifts, the surrogate approximates the wrong posterior."""
    X, y = _clear_signal()
    target = SpikeSlabTarget(X, y)
    X_c, y_c = center_and_scale(X, y)
    np.testing.assert_allclose(target._X, X_c, atol=1e-12, rtol=0.0)
    np.testing.assert_allclose(target._y, y_c, atol=1e-12, rtol=0.0)


def test_analytic_shapes_symmetry_and_unpack() -> None:
    X, y = _clear_signal()
    surrogate = ising_surrogate_from_data(X, y, prior_inclusion=0.5)
    h, J = surrogate
    assert surrogate.kind == "analytic"
    assert h.shape == (8,)
    assert J.shape == (8, 8)
    np.testing.assert_allclose(J, J.T, atol=1e-12)
    np.testing.assert_array_equal(np.diag(J), 0.0)
    np.testing.assert_array_equal(h, surrogate.h)


def test_analytic_couplings_are_antiferromagnetic() -> None:
    X, y = _clear_signal()
    surrogate = ising_surrogate_from_data(X, y)
    off = surrogate.J.copy()
    np.fill_diagonal(off, 0.0)
    assert np.all(off <= 1e-15)


def test_analytic_is_deterministic() -> None:
    X, y = _clear_signal()
    a = ising_surrogate_from_data(X, y, prior_inclusion=0.3)
    b = ising_surrogate_from_data(X, y, prior_inclusion=0.3)
    np.testing.assert_array_equal(a.h, b.h)
    np.testing.assert_array_equal(a.J, b.J)


def test_frobenius_pin_makes_alpha_one() -> None:
    """Pinned ||H_prob||_F = √p is the convention that makes quench α = 1."""
    X, y = _clear_signal()
    surrogate = ising_surrogate_from_data(X, y)
    np.testing.assert_allclose(
        problem_frobenius_norm(surrogate.h, surrogate.J),
        np.sqrt(surrogate.n_vars),
        atol=1e-12,
        rtol=0.0,
    )


def test_sparsity_field_favors_empty_model_when_prior_is_sparse() -> None:
    X, y = _clear_signal()
    sparse = ising_surrogate_from_data(X, y, prior_inclusion=0.1)
    dense = ising_surrogate_from_data(X, y, prior_inclusion=0.9)
    empty = np.zeros(X.shape[1], dtype=np.uint8)
    assert sparse.energy(empty) < dense.energy(empty)
    assert sparse.meta["h_sparsity"] < 0.0
    assert dense.meta["h_sparsity"] > 0.0


def test_rejects_bad_prior_and_ratio() -> None:
    X, y = _clear_signal()
    with pytest.raises(ValueError, match="prior_inclusion"):
        ising_surrogate_from_data(X, y, prior_inclusion=0.0)
    with pytest.raises(ValueError, match="field_coupling_ratio"):
        ising_surrogate_from_data(X, y, field_coupling_ratio=0.0)


def test_zero_hamiltonian_is_refused() -> None:
    """Constant y and orthogonal columns: no fields, no couplings, α undefined."""
    X = np.array(
        [
            [1.0, 1.0],
            [1.0, -1.0],
            [-1.0, 1.0],
            [-1.0, -1.0],
        ]
    )
    y = np.ones(4)
    with pytest.raises(ValueError, match="identically zero"):
        ising_surrogate_from_data(X, y, prior_inclusion=0.5)


def test_ground_state_is_a_high_posterior_model() -> None:
    X, y = _clear_signal(n_vars=8, seed=1)
    target = SpikeSlabTarget(X, y, prior_inclusion=0.5)
    surrogate = ising_surrogate_from_data(X, y, prior_inclusion=0.5)
    diagnosis = diagnose_surrogate(surrogate, target)
    # 256 models. The analytic Hamiltonian is 2-local, so it will not
    # hit the MAP; it must still land on a high-posterior model that
    # includes the two variables that actually enter y.
    assert diagnosis.ground_state_rank <= 16
    assert diagnosis.top_log_prob - diagnosis.ground_state_log_prob < 5.0
    assert diagnosis.ground_state[0] == 1
    assert diagnosis.ground_state[1] == 1


def test_energy_correlates_with_exact_log_posterior() -> None:
    X, y = _clear_signal(n_vars=8, seed=2)
    target = SpikeSlabTarget(X, y)
    analytic = ising_surrogate_from_data(X, y)
    learned = learned_surrogate(target, seed=0)
    analytic_d = diagnose_surrogate(analytic, target)
    learned_d = diagnose_surrogate(learned, target)
    assert analytic_d.spearman > 0.45
    assert learned_d.spearman > 0.85
    assert learned_d.spearman >= analytic_d.spearman - 0.02


def test_learned_recovers_an_ising_target() -> None:
    """The 2-local family is exact for Ising: Spearman must be ~1 after the fit."""
    target = random_spin_glass(
        5, topology="all-to-all", seed=4, temperature=1.0, random_fields=True
    )
    surrogate = learned_surrogate(target, seed=0)
    diagnosis = diagnose_surrogate(surrogate, target)
    assert diagnosis.spearman > 0.999
    assert diagnosis.ground_state_rank == 1
    assert surrogate.kind == "learned"
    assert surrogate.meta["enumerated"] is True
    np.testing.assert_allclose(
        problem_frobenius_norm(surrogate.h, surrogate.J),
        np.sqrt(5.0),
        atol=1e-12,
    )


def test_learned_is_seeded() -> None:
    target = random_spin_glass(6, seed=0, random_fields=True)
    a = learned_surrogate(target, n_samples=64, seed=11, ridge=1e-2)
    b = learned_surrogate(target, n_samples=64, seed=11, ridge=1e-2)
    np.testing.assert_allclose(a.h, b.h)
    np.testing.assert_allclose(a.J, b.J)


def test_wall_swapping_surrogates_leaves_the_posterior_alone() -> None:
    """P0: a different (h, J) may change IAT, never the invariant distribution."""
    rng = np.random.Generator(np.random.PCG64(3))
    X = rng.standard_normal((40, 5))
    y = X[:, 0] + 0.25 * rng.standard_normal(40)
    target = SpikeSlabTarget(X, y, prior_inclusion=0.4)
    exact_pips = target.posterior_inclusion_probs_exact()

    analytic = ising_surrogate_from_data(X, y, prior_inclusion=0.4)
    noise = np.random.Generator(np.random.PCG64(99))
    J_noise = noise.standard_normal((5, 5))
    J_noise = 0.5 * (J_noise + J_noise.T)
    np.fill_diagonal(J_noise, 0.0)
    garbage = IsingSurrogate(h=noise.standard_normal(5), J=J_noise, kind="analytic")

    x0 = np.zeros(5, dtype=np.uint8)
    errors = []
    for surrogate in (analytic, garbage):
        kernel = SurrogateIndependence(surrogate)
        P = MetropolisEngine(target, kernel).transition_matrix()
        pi = target.enumerate_exact()
        np.testing.assert_allclose(pi @ P, pi, atol=1e-12, rtol=0.0)
        result = MetropolisEngine(target, kernel).run(n_steps=25_000, x0=x0, seed=7)
        errors.append(pip_error(result.states[5_000:], exact_pips))

    assert errors[0] < 0.12
    assert errors[1] < 0.12


def test_corrupt_surrogate_re_pins_and_is_seeded() -> None:
    X, y = _clear_signal(n_vars=6, seed=0)
    clean = ising_surrogate_from_data(X, y)
    a = corrupt_surrogate(clean, relative_sigma=1.0, seed=4)
    b = corrupt_surrogate(clean, relative_sigma=1.0, seed=4)
    c = corrupt_surrogate(clean, relative_sigma=1.0, seed=5)
    np.testing.assert_allclose(a.h, b.h)
    np.testing.assert_allclose(a.J, b.J)
    assert not np.allclose(a.h, c.h)
    np.testing.assert_allclose(
        problem_frobenius_norm(a.h, a.J),
        np.sqrt(a.n_vars),
        atol=1e-12,
        rtol=0.0,
    )


def test_corrupt_sigma_zero_is_a_pinned_clone() -> None:
    X, y = _clear_signal(n_vars=6, seed=1)
    clean = ising_surrogate_from_data(X, y)
    clone = corrupt_surrogate(clean, relative_sigma=0.0, seed=0)
    np.testing.assert_allclose(clone.h, clean.h, atol=1e-12, rtol=0.0)
    np.testing.assert_allclose(clone.J, clean.J, atol=1e-12, rtol=0.0)


def test_corrupt_destroys_spearman() -> None:
    X, y = _clear_signal(n_vars=6, seed=2)
    target = SpikeSlabTarget(X, y)
    learned = learned_surrogate(target, seed=0)
    clean = diagnose_surrogate(learned, target)
    wrecked = diagnose_surrogate(corrupt_surrogate(learned, 4.0, seed=1), target)
    assert clean.spearman > 0.85
    assert wrecked.spearman < clean.spearman - 0.2


def test_wall_corrupted_surrogate_leaves_posterior_alone() -> None:
    """P0: destroying the surrogate may change speed, never the posterior."""
    rng = np.random.Generator(np.random.PCG64(3))
    X = rng.standard_normal((40, 5))
    y = X[:, 0] + 0.25 * rng.standard_normal(40)
    target = SpikeSlabTarget(X, y, prior_inclusion=0.4)
    exact_pips = target.posterior_inclusion_probs_exact()
    learned = learned_surrogate(target, seed=0)
    wrecked = corrupt_surrogate(learned, relative_sigma=4.0, seed=9)

    x0 = np.zeros(5, dtype=np.uint8)
    errors = []
    for surrogate in (learned, wrecked):
        kernel = SurrogateIndependence(surrogate)
        P = MetropolisEngine(target, kernel).transition_matrix()
        pi = target.enumerate_exact()
        np.testing.assert_allclose(pi @ P, pi, atol=1e-12, rtol=0.0)
        result = MetropolisEngine(target, kernel).run(n_steps=25_000, x0=x0, seed=7)
        errors.append(pip_error(result.states[5_000:], exact_pips))

    assert errors[0] < 0.12
    assert errors[1] < 0.12


def test_energy_logprob_scatter_writes_a_figure(tmp_path: Path) -> None:
    X, y = _clear_signal(n_vars=5, seed=0)
    target = SpikeSlabTarget(X, y)
    surrogate = ising_surrogate_from_data(X, y)
    path = write_energy_logprob_scatter(surrogate, target, tmp_path / "scatter.png")
    assert path.is_file()
    assert path.stat().st_size > 0
