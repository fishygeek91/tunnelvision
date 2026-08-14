"""Target distribution interface."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class Target(ABC):
    """An unnormalized log-density over binary vectors of length n_vars.

    log_prob must be EXACT — this is what the engine's accept/reject
    evaluates, and it is the source of the sampler's unbiasedness.
    Surrogates for shaping proposals live in surrogate.py, never here.
    """

    n_vars: int

    @abstractmethod
    def log_prob(self, x: np.ndarray) -> float:
        ...

    def log_prob_batch(self, X: np.ndarray) -> np.ndarray:
        return np.array([self.log_prob(x) for x in X])

    def enumerate_exact(self) -> np.ndarray:
        """Normalized probabilities over all 2^n states (n <= ~20). Ground truth."""
        raise NotImplementedError
