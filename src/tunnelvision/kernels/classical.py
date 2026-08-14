"""Classical baseline kernels.

These are the bar to beat — especially AddDeleteSwap, the kernel
statisticians actually use for variable selection. Never benchmark
against a strawman.
"""

from __future__ import annotations

import numpy as np

from tunnelvision.bits import as_binary_vector
from tunnelvision.kernels.base import Kernel


def _available_ads_types(k: int, n_vars: int) -> list[str]:
    """Move types that are legal at Hamming weight k.

    Illegal types are dropped and the remainder is renormalized. The
    alternative (attempt a type and stay on failure) injects a large
    self-loop at the empty/full models and is a different kernel.
    """
    types: list[str] = []
    if k < n_vars:
        types.append("add")
    if k > 0:
        types.append("delete")
    if 0 < k < n_vars:
        types.append("swap")
    return types


class UniformFlip(Kernel):
    """Propose a uniformly random state. Symmetric. The weakest baseline."""

    name = "uniform"

    def propose(self, x: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, float, float]:
        x_bin = as_binary_vector(x)
        y = rng.integers(0, 2, size=x_bin.shape[0], dtype=np.uint8)
        return y, 0.0, 0.0

    def proposal_matrix(self, n_vars: int) -> np.ndarray:
        if n_vars < 1:
            raise ValueError("n_vars must be at least 1")
        n_states = 1 << n_vars
        return np.full((n_states, n_states), 1.0 / n_states, dtype=np.float64)


class SingleFlip(Kernel):
    """Flip one uniformly chosen bit. Symmetric. The classic local kernel."""

    name = "single-flip"

    def propose(self, x: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, float, float]:
        y = as_binary_vector(x)
        if y.size == 0:
            raise ValueError("SingleFlip requires at least one variable")
        idx = int(rng.integers(0, y.size))
        y[idx] = np.uint8(1 - y[idx])
        return y, 0.0, 0.0

    def proposal_matrix(self, n_vars: int) -> np.ndarray:
        if n_vars < 1:
            raise ValueError("n_vars must be at least 1")
        n_states = 1 << n_vars
        Q = np.zeros((n_states, n_states), dtype=np.float64)
        idx = np.arange(n_states)
        mass = 1.0 / n_vars
        for bit in range(n_vars):
            Q[idx, idx ^ (1 << bit)] = mass
        return Q


class AddDeleteSwap(Kernel):
    """Field-standard variable-selection kernel.

    With equal probability among the *available* types: add a variable,
    delete a variable, or swap an included variable with an excluded one.

    NOT symmetric in general. From the empty model delete/swap are
    unavailable; from the full model add/swap are unavailable; swap
    counts depend on |γ|·(p-|γ|). Forward and reverse proposal
    probabilities differ whenever the move changes |γ| (George &
    McCulloch, JASA 1993 conventions; see also the MC³ add/delete/swap
    family). Do not "simplify" this to a symmetric kernel.
    """

    name = "add-delete-swap"

    def propose(self, x: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, float, float]:
        x_bin = as_binary_vector(x)
        n_vars = int(x_bin.size)
        if n_vars < 1:
            raise ValueError("AddDeleteSwap requires at least one variable")

        included = np.flatnonzero(x_bin)
        excluded = np.flatnonzero(1 - x_bin)
        types = _available_ads_types(int(included.size), n_vars)
        move = types[int(rng.integers(0, len(types)))]

        y = x_bin.copy()
        if move == "add":
            y[int(excluded[int(rng.integers(0, excluded.size))])] = 1
        elif move == "delete":
            y[int(included[int(rng.integers(0, included.size))])] = 0
        else:
            y[int(included[int(rng.integers(0, included.size))])] = 0
            y[int(excluded[int(rng.integers(0, excluded.size))])] = 1

        return y, self.log_q(x_bin, y), self.log_q(y, x_bin)

    def log_q(self, x: np.ndarray, y: np.ndarray) -> float:
        """log q(y|x) with the available-type renormalization made explicit."""
        x_bin = as_binary_vector(x)
        y_bin = as_binary_vector(y, n_vars=x_bin.size)
        n_vars = int(x_bin.size)
        k = int(x_bin.sum())
        types = _available_ads_types(k, n_vars)
        if not types:
            return -np.inf

        n_add = int(np.count_nonzero((y_bin == 1) & (x_bin == 0)))
        n_del = int(np.count_nonzero((y_bin == 0) & (x_bin == 1)))
        log_type = -np.log(len(types))

        if n_add == 1 and n_del == 0:
            if "add" not in types:
                return -np.inf
            return float(log_type - np.log(n_vars - k))
        if n_add == 0 and n_del == 1:
            if "delete" not in types:
                return -np.inf
            return float(log_type - np.log(k))
        if n_add == 1 and n_del == 1:
            if "swap" not in types:
                return -np.inf
            return float(log_type - np.log(k * (n_vars - k)))
        return -np.inf

    def proposal_matrix(self, n_vars: int) -> np.ndarray:
        if n_vars < 1:
            raise ValueError("n_vars must be at least 1")
        n_states = 1 << n_vars
        Q = np.zeros((n_states, n_states), dtype=np.float64)
        for idx in range(n_states):
            bits = [(idx >> bit) & 1 for bit in range(n_vars)]
            included = [b for b, flag in enumerate(bits) if flag]
            excluded = [b for b, flag in enumerate(bits) if not flag]
            k = len(included)
            types = _available_ads_types(k, n_vars)
            p_type = 1.0 / len(types)
            if "add" in types:
                mass = p_type / (n_vars - k)
                for bit in excluded:
                    Q[idx, idx | (1 << bit)] = mass
            if "delete" in types:
                mass = p_type / k
                for bit in included:
                    Q[idx, idx & ~(1 << bit)] = mass
            if "swap" in types:
                mass = p_type / (k * (n_vars - k))
                for bit_in in included:
                    for bit_out in excluded:
                        Q[idx, (idx & ~(1 << bit_in)) | (1 << bit_out)] = mass
        return Q
