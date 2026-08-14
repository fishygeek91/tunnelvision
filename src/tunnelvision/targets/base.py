"""Target distribution interface."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from tunnelvision.bits import all_binary_states


class Target(ABC):
    """An unnormalized log-density over binary vectors of length n_vars.

    log_prob must be EXACT — this is what the engine's accept/reject
    evaluates, and it is the source of the sampler's unbiasedness.
    Surrogates for shaping proposals live in surrogate.py, never here.
    """

    n_vars: int

    @abstractmethod
    def log_prob(self, x: np.ndarray) -> float: ...

    def log_prob_batch(self, X: np.ndarray) -> np.ndarray:
        """Default row-wise wrapper. Override when a vectorized path exists."""
        arr = np.asarray(X)
        if arr.ndim == 1:
            return np.array([self.log_prob(arr)], dtype=np.float64)
        if arr.ndim != 2:
            raise ValueError("X must have shape (n_vars,) or (n_states, n_vars)")
        return np.array([self.log_prob(row) for row in arr], dtype=np.float64)

    def enumerate_exact(self) -> np.ndarray:
        """Normalized probabilities over all 2^n states (n <= ~20). Ground truth."""
        if self.n_vars > 20:
            raise ValueError(
                f"exact enumeration is only supported for n_vars <= 20, got {self.n_vars}"
            )
        states = all_binary_states(self.n_vars)
        logp = np.asarray(self.log_prob_batch(states), dtype=np.float64)
        logp = logp - np.max(logp)
        probs = np.exp(logp)
        total = float(probs.sum())
        if not np.isfinite(total) or total <= 0.0:
            raise RuntimeError("enumerate_exact: probabilities are not finite")
        return probs / total
