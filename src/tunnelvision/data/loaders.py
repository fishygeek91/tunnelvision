"""Dataset loaders for E02. Both sides of the wall see the same (X, y).

diabetes: sklearn's 10-predictor series, 2^10 = 1024 models — exact
posterior and exact spectral gaps are enumerable. The E02a headline.

correlated_synthetic: linear-model data with a tunable predictor
correlation ρ and a sparse true support. That ρ is the knob for
"does quantum advantage grow with multimodality?" Correlation makes
predictors interchangeable, which splits the posterior into competing
modes; tunneling is supposed to pay there.

Returned designs are passed through ``center_and_scale`` so a loader
and ``SpikeSlabTarget`` cannot silently disagree about the matrix the
surrogate was built from. Generative SNR is computed *before* that
rescaling — it is a property of the simulation, not of the units.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

from tunnelvision.design import center_and_scale

CovarianceStructure = Literal["equicorrelated", "ar1"]


def load_diabetes() -> tuple[np.ndarray, np.ndarray]:
    """sklearn diabetes (n=442, p=10), under the shared scaling convention."""
    from sklearn.datasets import load_diabetes as _sk_load

    bunch = _sk_load()
    X = np.asarray(bunch.data, dtype=np.float64)
    y = np.asarray(bunch.target, dtype=np.float64)
    if X.shape[1] != 10:
        raise RuntimeError(f"expected sklearn diabetes to have 10 predictors, got {X.shape[1]}")
    return center_and_scale(X, y)


def load_diabetes_standardized() -> tuple[np.ndarray, np.ndarray]:
    """Alias kept for the Rung-1 stub name."""
    return load_diabetes()


def correlated_synthetic(
    n: int = 100,
    p: int = 20,
    rho: float = 0.7,
    k_true: int = 4,
    snr: float = 2.0,
    seed: int = 0,
    structure: CovarianceStructure = "equicorrelated",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Gaussian design with sparse truth. Returns ``(X, y, true_support)``.

    ``true_support`` is a length-p binary vector: the first ``k_true``
    predictors, with seeded random ±1 coefficients. First-k (not a
    random subset) so adding a column does not reshuffle the truth.

    ``structure='equicorrelated'`` is the exchangeable design
    (Σ = (1−ρ)I + ρ 11ᵀ). ``'ar1'`` is the decaying-correlation design
    (Σ_ij = ρ^{|i−j|}) — nearby predictors compete, distant ones do not.
    """
    if n < 2:
        raise ValueError(f"n must be at least 2, got {n}")
    if p < 1:
        raise ValueError(f"p must be at least 1, got {p}")
    if not 1 <= k_true <= p:
        raise ValueError(f"k_true must be in [1, p], got k_true={k_true}, p={p}")
    if snr <= 0.0 or not np.isfinite(snr):
        raise ValueError(f"snr must be a positive finite value, got {snr}")

    cov = _covariance(p, rho, structure)
    rng = np.random.Generator(np.random.PCG64(seed))
    try:
        factor = np.linalg.cholesky(cov)
    except np.linalg.LinAlgError as exc:
        raise ValueError(
            f"covariance is not SPD for structure={structure!r}, rho={rho}, p={p}"
        ) from exc

    X_raw = rng.standard_normal((n, p)) @ factor.T
    support = np.zeros(p, dtype=np.uint8)
    support[:k_true] = 1
    signs = rng.choice(np.array([-1.0, 1.0]), size=k_true)
    beta = np.zeros(p, dtype=np.float64)
    beta[:k_true] = signs
    signal = X_raw @ beta
    signal_sd = float(signal.std(ddof=0))
    noise_sd = (signal_sd / snr) if signal_sd > 0.0 else 1.0
    y_raw = signal + noise_sd * rng.standard_normal(n)
    X, y = center_and_scale(X_raw, y_raw)
    return X, y, support


def _covariance(p: int, rho: float, structure: str) -> np.ndarray:
    key = structure.lower()
    if key == "equicorrelated":
        # SPD iff −1/(p−1) < ρ < 1. ρ = 1 is rank-1 and unusable.
        lower = -1.0 / (p - 1) if p > 1 else -1.0
        if not lower < rho < 1.0:
            raise ValueError(
                f"equicorrelated rho must be in ({lower:.6g}, 1), got {rho}"
            )
        return (1.0 - rho) * np.eye(p) + rho * np.ones((p, p))
    if key == "ar1":
        if not -1.0 < rho < 1.0:
            raise ValueError(f"ar1 rho must be in (-1, 1), got {rho}")
        idx = np.arange(p)
        return np.power(rho, np.abs(idx[:, None] - idx[None, :]))
    raise ValueError(f"structure must be 'equicorrelated' or 'ar1', got {structure!r}")
