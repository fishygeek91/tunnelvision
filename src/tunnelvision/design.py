"""Shared design-matrix convention for the exact and heuristic sides.

Both the g-prior target and the Ising surrogate must see the same
centered-and-scaled (X, y). If they drift, the surrogate is approximating
a different posterior than the engine is targeting — the energy-vs-logp
scatter will look like a bad Hamiltonian when it is really a bookkeeping
bug. Column scaling is numerical, not a modeling choice: the intercept is
absorbed by centering (the BAS convention).
"""

from __future__ import annotations

import numpy as np


def as_design(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Validate a design and return float64 copies without transforming."""
    X_arr = np.asarray(X, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    if X_arr.ndim != 2:
        raise ValueError("X must be a 2-d array of shape (n_obs, n_vars)")
    if y_arr.ndim != 1 or y_arr.shape[0] != X_arr.shape[0]:
        raise ValueError("y must be 1-d with length X.shape[0]")
    if X_arr.shape[0] < 2:
        raise ValueError("need at least 2 observations to center the design")
    if X_arr.shape[1] < 1:
        raise ValueError("X must have at least one predictor")
    return X_arr, y_arr


def center_and_scale(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Center X and y; scale columns to unit standard deviation (ddof=0).

    Constant columns are left at zero (scale is taken to be 1) so they
    contribute no field and no coupling, rather than exploding.
    """
    X_arr, y_arr = as_design(X, y)
    X_c = X_arr - X_arr.mean(axis=0)
    y_c = y_arr - y_arr.mean()
    x_scale = X_c.std(axis=0, ddof=0)
    x_scale = np.where(x_scale == 0.0, 1.0, x_scale)
    X_c = X_c / x_scale
    y_scale = float(y_c.std(ddof=0))
    if y_scale > 0.0:
        y_c = y_c / y_scale
    return X_c, y_c
