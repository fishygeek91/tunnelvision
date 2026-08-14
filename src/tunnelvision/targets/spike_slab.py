"""Spike-and-slab marginal posterior over inclusion vectors — the E02 target.

Model: y = X_gamma beta_gamma + eps, g-prior on beta_gamma, Bernoulli(pi)
prior on inclusions. Marginalizing beta and sigma^2 analytically gives a
closed-form log marginal posterior log p(gamma | y, X) up to a constant:

    log p(gamma) = log m(y | gamma) + |gamma| log(pi/(1-pi)) + const

with m(y|gamma) involving (X_gamma' X_gamma) determinants — cheap for
p <= ~30 per evaluation, and EXACT. This is what the engine sees.

NOTE: this is not a 2-local Ising energy — that's the whole reason
surrogate.py exists. Keep the exact/surrogate boundary sacred.
"""

from __future__ import annotations

import numpy as np

from tunnelvision.targets.base import Target


class SpikeSlabTarget(Target):
    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        g: float | None = None,      # g-prior scale; default g = n (unit information)
        prior_inclusion: float = 0.5,
    ) -> None:
        raise NotImplementedError  # Rung 1

    def log_prob(self, gamma: np.ndarray) -> float:
        raise NotImplementedError

    def posterior_inclusion_probs_exact(self) -> np.ndarray:
        """Exact PIPs by enumeration (p <= ~20). Evaluation ground truth for E02."""
        raise NotImplementedError
