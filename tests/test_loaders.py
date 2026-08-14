"""E02 loaders: shape, scaling, seeded covariance, generative SNR."""

from __future__ import annotations

import numpy as np
import pytest

from tunnelvision.data.loaders import (
    correlated_synthetic,
    load_diabetes,
    load_diabetes_standardized,
)
from tunnelvision.design import center_and_scale


def test_diabetes_is_p10_and_scaled() -> None:
    X, y = load_diabetes()
    assert X.shape == (442, 10)
    assert y.shape == (442,)
    np.testing.assert_allclose(X.mean(axis=0), 0.0, atol=1e-12)
    np.testing.assert_allclose(X.std(axis=0, ddof=0), 1.0, atol=1e-12)
    np.testing.assert_allclose(y.mean(), 0.0, atol=1e-12)
    np.testing.assert_allclose(y.std(ddof=0), 1.0, atol=1e-12)


def test_diabetes_alias_matches() -> None:
    X_a, y_a = load_diabetes()
    X_b, y_b = load_diabetes_standardized()
    np.testing.assert_array_equal(X_a, X_b)
    np.testing.assert_array_equal(y_a, y_b)


def test_diabetes_matches_shared_convention() -> None:
    from sklearn.datasets import load_diabetes as sk_load

    bunch = sk_load()
    X_c, y_c = center_and_scale(bunch.data, bunch.target)
    X, y = load_diabetes()
    np.testing.assert_allclose(X, X_c)
    np.testing.assert_allclose(y, y_c)


def test_synthetic_is_seeded_and_scaled() -> None:
    a = correlated_synthetic(n=80, p=8, rho=0.5, k_true=2, seed=4)
    b = correlated_synthetic(n=80, p=8, rho=0.5, k_true=2, seed=4)
    c = correlated_synthetic(n=80, p=8, rho=0.5, k_true=2, seed=5)
    for X, y, support in (a, b, c):
        assert X.shape == (80, 8)
        assert y.shape == (80,)
        assert support.shape == (8,)
        assert int(support.sum()) == 2
        np.testing.assert_array_equal(support[:2], 1)
        np.testing.assert_allclose(X.mean(axis=0), 0.0, atol=1e-12)
        np.testing.assert_allclose(X.std(axis=0, ddof=0), 1.0, atol=1e-12)
    np.testing.assert_array_equal(a[0], b[0])
    assert not np.allclose(a[0], c[0])


def test_equicorrelated_off_diagonals_match_rho() -> None:
    X, _y, _support = correlated_synthetic(
        n=4000, p=6, rho=0.7, k_true=2, seed=0, structure="equicorrelated"
    )
    corr = np.corrcoef(X, rowvar=False)
    off = corr[np.triu_indices(6, k=1)]
    np.testing.assert_allclose(off.mean(), 0.7, atol=0.05)


def test_ar1_lag1_matches_rho() -> None:
    X, _y, _support = correlated_synthetic(
        n=4000, p=6, rho=0.6, k_true=2, seed=1, structure="ar1"
    )
    corr = np.corrcoef(X, rowvar=False)
    lag1 = np.array([corr[i, i + 1] for i in range(5)])
    np.testing.assert_allclose(lag1.mean(), 0.6, atol=0.05)


def test_high_snr_recovers_true_support() -> None:
    """A sanity check on the generative model, not on the sampler."""
    from tunnelvision.targets.spike_slab import SpikeSlabTarget

    X, y, support = correlated_synthetic(
        n=200, p=6, rho=0.1, k_true=2, snr=8.0, seed=2
    )
    pips = SpikeSlabTarget(X, y).posterior_inclusion_probs_exact()
    assert pips[0] > 0.8
    assert pips[1] > 0.8
    assert np.all(pips[2:] < 0.4)
    np.testing.assert_array_equal(support, np.array([1, 1, 0, 0, 0, 0], dtype=np.uint8))


def test_synthetic_rejects_bad_args() -> None:
    with pytest.raises(ValueError, match="k_true"):
        correlated_synthetic(p=4, k_true=5)
    with pytest.raises(ValueError, match="snr"):
        correlated_synthetic(snr=0.0)
    with pytest.raises(ValueError, match="equicorrelated"):
        correlated_synthetic(p=4, rho=1.0, structure="equicorrelated")
    with pytest.raises(ValueError, match="ar1"):
        correlated_synthetic(p=4, rho=1.0, structure="ar1")
    with pytest.raises(ValueError, match="structure"):
        correlated_synthetic(structure="wishart")  # type: ignore[arg-type]
