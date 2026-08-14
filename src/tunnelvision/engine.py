"""Metropolis–Hastings engine — the exactness boundary.

Everything quantum, noisy, or heuristic lives on the OTHER side of the
Kernel interface. This module must stay boring, classical, and provably
correct. If you are editing this file to make a kernel's life easier,
you are probably about to break the project's core guarantee.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from tunnelvision.kernels.base import Kernel
from tunnelvision.targets.base import Target


@dataclass
class ChainResult:
    """Samples plus bookkeeping from one chain run."""

    states: np.ndarray  # (n_steps, n_vars) uint8
    log_probs: np.ndarray  # (n_steps,) target log-prob of each state
    accepted: np.ndarray  # (n_steps,) bool
    kernel_name: str
    seed: int
    meta: dict = field(default_factory=dict)

    @property
    def acceptance_rate(self) -> float:
        return float(self.accepted.mean())


class MetropolisEngine:
    """Kernel-agnostic Metropolis–Hastings over binary state spaces.

    Accept probability: min(1, exp(logp(y) - logp(x)) * q(x|y)/q(y|x)).
    Kernels must implement propose(x, rng) -> (y, log_q_forward, log_q_reverse).
    Symmetric kernels return 0.0 for both log-q terms.
    """

    def __init__(self, target: Target, kernel: Kernel) -> None:
        self.target = target
        self.kernel = kernel

    def run(self, n_steps: int, x0: np.ndarray, seed: int = 0) -> ChainResult:
        raise NotImplementedError  # Rung 1

    def transition_matrix(self) -> np.ndarray:
        """Exact transition matrix over all 2^n states (n <= ~14 only).

        Requires the kernel to implement proposal_matrix(). Used by
        diagnostics.spectral_gap for airtight kernel comparison.
        """
        raise NotImplementedError  # Rung 1
