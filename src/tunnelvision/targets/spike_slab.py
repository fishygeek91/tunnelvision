"""Spike-and-slab marginal posterior over inclusion vectors — the E02 target.

Model: y = X_γ β_γ + ε with a Zellner g-prior on β_γ and independent
Bernoulli(π) inclusions. Integrating β and σ² (flat intercept, p(σ²)∝1/σ²)
gives a closed-form log marginal posterior (Liang, Paulo, Molina, Clyde &
Berger, JASA 2008, eq. 3; the BAS/BMS default):

    log p(γ | y) = log m(y | γ) + |γ| log(π/(1-π)) + const
    log m(y | γ) = -(q/2) log(1+g) - ((n-1)/2) log(1 - [g/(1+g)] R²_γ)

with q = |γ| and R²_γ the OLS coefficient of determination on centered
data. Default g = n is the unit-information prior (Kass & Wasserman 1995).

Numerics, not statistics, is the hazard: never form det(·); cache X'X and
X'y so each state is a Cholesky of a q×q block. The empty model is its
own closed form (R²=0). This is not a 2-local Ising energy — that is why
surrogate.py exists. Keep the exact/surrogate boundary sacred.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import solve_triangular

from tunnelvision.bits import all_binary_states, as_binary_vector
from tunnelvision.design import center_and_scale
from tunnelvision.targets.base import Target


class SpikeSlabTarget(Target):
    """Exact g-prior marginal posterior over binary inclusion vectors."""

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        g: float | None = None,
        prior_inclusion: float = 0.5,
    ) -> None:
        X_c, y_c = center_and_scale(X, y)
        n_obs, n_vars = int(X_c.shape[0]), int(X_c.shape[1])
        if not (0.0 < prior_inclusion < 1.0):
            raise ValueError(f"prior_inclusion must be in (0, 1), got {prior_inclusion}")

        g_val = float(n_obs) if g is None else float(g)
        if g_val <= 0.0:
            raise ValueError(f"g-prior scale must be positive, got {g_val}")

        self.n_obs = n_obs
        self.n_vars = n_vars
        self.g = g_val
        self.prior_inclusion = float(prior_inclusion)
        self._df = float(n_obs - 1)
        self._log_prior_odds = float(np.log(prior_inclusion / (1.0 - prior_inclusion)))
        self._shrinkage = self.g / (1.0 + self.g)
        self._log_one_plus_g = float(np.log(1.0 + self.g))
        # Cached so per-state work is a q×q Cholesky, not another X'X.
        self._XtX = X_c.T @ X_c
        self._Xty = X_c.T @ y_c
        self._yty = float(y_c @ y_c)
        self._X = X_c
        self._y = y_c

    def log_prob(self, gamma: np.ndarray) -> float:
        gamma_bin = as_binary_vector(gamma, self.n_vars)
        idx = np.flatnonzero(gamma_bin)
        q = int(idx.size)
        if q == 0 or self._yty <= 0.0:
            # Empty model: R²=0 so log m = 0 (up to γ-independent const).
            # Zero-variance y: every model is equivalent; only the prior remains.
            return float(q * self._log_prior_odds)

        xtx = self._XtX[np.ix_(idx, idx)]
        xty = self._Xty[idx]
        try:
            chol = np.linalg.cholesky(xtx)
        except np.linalg.LinAlgError:
            # Rank-deficient X_γ: the g-prior needs (X'X)^{-1}, so this
            # model has posterior mass zero.
            return -np.inf

        # y' P_γ y = ||L^{-1} X_γ' y||² — no determinant, no explicit inverse.
        residual = solve_triangular(chol, xty, lower=True, check_finite=False)
        y_proj = float(residual @ residual)
        r_squared = y_proj / self._yty
        # A perfect fit is legal (one_minus = 1/(1+g) > 0). Only clip
        # the floating-point overshoot that would make one_minus negative.
        if r_squared > 1.0:
            r_squared = 1.0 - 1e-15
        elif r_squared < 0.0:
            r_squared = 0.0
        one_minus = 1.0 - self._shrinkage * r_squared
        if one_minus <= 0.0:
            return -np.inf

        log_m = -0.5 * q * self._log_one_plus_g - 0.5 * self._df * np.log(one_minus)
        return float(log_m + q * self._log_prior_odds)

    def posterior_inclusion_probs_exact(self) -> np.ndarray:
        """Exact PIPs by enumeration (p <= ~20). Evaluation ground truth for E02."""
        if self.n_vars > 20:
            raise ValueError(f"exact PIPs require n_vars <= 20, got {self.n_vars}")
        states = all_binary_states(self.n_vars)
        logp = self.log_prob_batch(states)
        weights = np.exp(logp - np.max(logp))
        weights /= weights.sum()
        return weights @ states.astype(np.float64)
