"""Kernel interface. Kernels propose; they never accept."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class Kernel(ABC):
    """A proposal distribution q(y | x) over binary vectors.

    Contract:
    - propose() returns (y, log_q_forward, log_q_reverse). If the kernel is
      symmetric (q(y|x) == q(x|y) for all pairs), return 0.0 for both — the
      engine cancels them. Quantum quench kernels with time-symmetric
      evolution and identical measurement are symmetric (Layden 2023, SI).
    - A kernel MUST NOT look at target log-probabilities to filter its own
      proposals. Shaping the proposal with a surrogate is fine (the surrogate
      is fixed data); peeking at the target breaks detailed-balance accounting.
    - proposal_matrix() is optional; implement for n <= ~14 to enable exact
      spectral-gap diagnostics.
    """

    name: str = "kernel"

    @abstractmethod
    def propose(
        self, x: np.ndarray, rng: np.random.Generator
    ) -> tuple[np.ndarray, float, float]: ...

    def proposal_matrix(self, n_vars: int) -> np.ndarray:
        """Row ``x`` is q(·|x) over the bit-packed index convention in bits.py."""
        raise NotImplementedError(f"{self.name} has no exact proposal matrix")
