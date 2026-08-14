"""Target numerics: Ising enumeration and the g-prior 3-predictor case."""

from __future__ import annotations

import numpy as np
import pytest

from tunnelvision.bits import all_binary_states
from tunnelvision.targets.ising import IsingTarget, random_spin_glass
from tunnelvision.targets.spike_slab import SpikeSlabTarget


def test_ising_log_prob_matches_batch() -> None:
    target = random_spin_glass(5, topology="all-to-all", seed=4, temperature=0.7)
    states = all_binary_states(target.n_vars)
    batch = target.log_prob_batch(states)
    single = np.array([target.log_prob(row) for row in states])
    np.testing.assert_allclose(batch, single, atol=1e-12, rtol=0.0)


def test_ising_enumerate_normalized_and_positive() -> None:
    target = random_spin_glass(6, topology="all-to-all", seed=0, temperature=1.0)
    pi = target.enumerate_exact()
    assert pi.shape == (1 << 6,)
    np.testing.assert_allclose(pi.sum(), 1.0, atol=1e-12)
    assert np.all(pi > 0.0)


def test_ising_low_temperature_concentrates_on_ground_state() -> None:
    h = np.array([1.0, 0.0], dtype=np.float64)
    J = np.zeros((2, 2))
    cold = IsingTarget(h, J, temperature=0.05)
    pi = cold.enumerate_exact()
    # s = 2x-1; field h=[1,0] prefers s0=+1 ⇒ x0=1, both x1 values close.
    states = all_binary_states(2)
    assert states[int(np.argmax(pi)), 0] == 1


def test_random_spin_glass_all_to_all_is_symmetric() -> None:
    target = random_spin_glass(7, topology="all-to-all", seed=3)
    np.testing.assert_allclose(target.J, target.J.T)
    np.testing.assert_array_equal(np.diag(target.J), 0.0)
    n_edges = int(np.count_nonzero(np.triu(target.J, k=1)))
    assert n_edges == 7 * 6 // 2


def test_random_spin_glass_2d_periodic_degree() -> None:
    target = random_spin_glass(9, topology="2d", seed=1)
    degree = np.count_nonzero(target.J, axis=1)
    np.testing.assert_array_equal(degree, 4)


def test_random_spin_glass_2d_rejects_non_square() -> None:
    with pytest.raises(ValueError, match="perfect square"):
        random_spin_glass(5, topology="2d")


def test_random_spin_glass_is_seeded() -> None:
    a = random_spin_glass(6, seed=11)
    b = random_spin_glass(6, seed=11)
    c = random_spin_glass(6, seed=12)
    np.testing.assert_array_equal(a.J, b.J)
    assert not np.allclose(a.J, c.J)


def test_random_spin_glass_fields_break_inversion_and_keep_j() -> None:
    """Layden Fig. 2 draws h ~ N(0,1); fields are drawn after J."""
    bare = random_spin_glass(6, seed=11, random_fields=False)
    glass = random_spin_glass(6, seed=11, random_fields=True)
    np.testing.assert_array_equal(bare.J, glass.J)
    np.testing.assert_array_equal(bare.h, 0.0)
    assert not np.allclose(glass.h, 0.0)
    again = random_spin_glass(6, seed=11, random_fields=True)
    np.testing.assert_array_equal(glass.h, again.h)
    lattice = random_spin_glass(9, topology="2d", seed=2, random_fields=True)
    assert lattice.h.shape == (9,)
    assert not np.allclose(lattice.h, 0.0)


def test_spike_slab_three_predictor_hand_worked() -> None:
    """Orthogonal 8-row design; y = x0 exactly. Closed-form g-prior values.

    After centering/scaling (already mean-0, unit std) we have XtX = 8 I,
    y'y = 8, g = n = 8, df = 7, π = 1/2 (prior term vanishes):

        log m(∅)     = 0
        log m({0})   = 3 log 9          (R² = 1)
        log m({1})   = -½ log 9         (R² = 0)
        log m({0,1}) = 2.5 log 9        (R² = 1)
        log m({1,2}) = -log 9           (R² = 0)
        log m({0,1,2}) = 2 log 9        (R² = 1)
    """
    x0 = np.array([1, 1, 1, 1, -1, -1, -1, -1], dtype=np.float64)
    x1 = np.array([1, 1, -1, -1, 1, 1, -1, -1], dtype=np.float64)
    x2 = np.array([1, -1, 1, -1, 1, -1, 1, -1], dtype=np.float64)
    X = np.column_stack([x0, x1, x2])
    y = x0.copy()
    target = SpikeSlabTarget(X, y, g=8.0, prior_inclusion=0.5)
    log9 = float(np.log(9.0))

    cases = {
        (0, 0, 0): 0.0,
        (1, 0, 0): 3.0 * log9,
        (0, 1, 0): -0.5 * log9,
        (0, 0, 1): -0.5 * log9,
        (1, 1, 0): 2.5 * log9,
        (1, 0, 1): 2.5 * log9,
        (0, 1, 1): -1.0 * log9,
        (1, 1, 1): 2.0 * log9,
    }
    for bits, expected in cases.items():
        gamma = np.array(bits, dtype=np.uint8)
        np.testing.assert_allclose(target.log_prob(gamma), expected, atol=1e-12, rtol=0.0)


def test_spike_slab_empty_model_is_zero_at_equal_prior() -> None:
    rng = np.random.Generator(np.random.PCG64(0))
    X = rng.standard_normal((12, 3))
    y = rng.standard_normal(12)
    target = SpikeSlabTarget(X, y, prior_inclusion=0.5)
    assert target.log_prob(np.zeros(3, dtype=np.uint8)) == 0.0


def test_spike_slab_independent_r_squared_path() -> None:
    """Recompute R² with lstsq on the stored design — a second code path."""
    rng = np.random.Generator(np.random.PCG64(5))
    X = rng.standard_normal((30, 3))
    y = X @ np.array([1.2, 0.0, -0.4]) + 0.3 * rng.standard_normal(30)
    target = SpikeSlabTarget(X, y, g=30.0, prior_inclusion=0.5)
    gamma = np.array([1, 0, 1], dtype=np.uint8)
    idx = np.flatnonzero(gamma)
    beta, residuals, *_ = np.linalg.lstsq(target._X[:, idx], target._y, rcond=None)
    del beta
    sse = float(residuals[0]) if residuals.size else 0.0
    r2 = 1.0 - sse / target._yty
    shrinkage = target.g / (1.0 + target.g)
    expected = -0.5 * 2 * np.log(1.0 + target.g) - 0.5 * target._df * np.log(1.0 - shrinkage * r2)
    np.testing.assert_allclose(target.log_prob(gamma), expected, atol=1e-10)


def test_spike_slab_pips_sum_bounds() -> None:
    rng = np.random.Generator(np.random.PCG64(6))
    X = rng.standard_normal((25, 4))
    y = X[:, :2].sum(axis=1) + 0.2 * rng.standard_normal(25)
    target = SpikeSlabTarget(X, y)
    pips = target.posterior_inclusion_probs_exact()
    assert pips.shape == (4,)
    assert np.all((pips >= 0.0) & (pips <= 1.0))
    # The two true predictors should outrank the noise columns.
    assert pips[0] > pips[2]
    assert pips[1] > pips[3]


def test_spike_slab_prior_odds_penalize_size() -> None:
    rng = np.random.Generator(np.random.PCG64(7))
    X = rng.standard_normal((16, 2))
    y = rng.standard_normal(16)
    sparse = SpikeSlabTarget(X, y, prior_inclusion=0.1)
    dense = SpikeSlabTarget(X, y, prior_inclusion=0.9)
    both = np.ones(2, dtype=np.uint8)
    assert sparse.log_prob(both) < dense.log_prob(both)


def test_spike_slab_rejects_bad_prior() -> None:
    X = np.ones((4, 2))
    y = np.arange(4, dtype=np.float64)
    with pytest.raises(ValueError, match="prior_inclusion"):
        SpikeSlabTarget(X, y, prior_inclusion=0.0)
