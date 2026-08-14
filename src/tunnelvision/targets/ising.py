"""Ising Boltzmann targets — for E01, the Layden reproduction.

E(s) = -sum_ij J_ij s_i s_j - sum_i h_i s_i,  p(s) ∝ exp(-E(s)/T)
Instance generators: 2D lattice and fully-connected spin glasses with
J ~ N(0,1), matching Layden et al. benchmark instances.
"""

from __future__ import annotations

import numpy as np

from tunnelvision.targets.base import Target


class IsingTarget(Target):
    def __init__(self, h: np.ndarray, J: np.ndarray, temperature: float = 1.0) -> None:
        raise NotImplementedError  # Rung 1


def random_spin_glass(n: int, topology: str = "all-to-all", seed: int = 0) -> IsingTarget:
    raise NotImplementedError  # Rung 1
