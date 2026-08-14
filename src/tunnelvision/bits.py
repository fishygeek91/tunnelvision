"""Bit-packing convention shared by the exact-tier routines.

State index ``i`` has bit ``k`` (value ``2**k``) equal to variable ``k``.
``proposal_matrix``, ``enumerate_exact``, and ``transition_matrix`` must
all use this layout — a mismatch here looks like a kernel bug but is
really an indexing bug, and it silently wrecks spectral gaps.
"""

from __future__ import annotations

import numpy as np

# Enumeration above this is a memory/time foot-gun (2^24 rows).
_MAX_ENUMERATE_VARS = 24


def as_binary_vector(x: np.ndarray, n_vars: int | None = None) -> np.ndarray:
    """Copy ``x`` as uint8 and reject anything that is not a 0/1 vector."""
    arr = np.asarray(x)
    if arr.ndim != 1:
        raise ValueError("state must be a 1-d binary vector")
    if n_vars is not None and arr.shape[0] != n_vars:
        raise ValueError(f"expected length {n_vars}, got {arr.shape[0]}")
    if arr.size > 0 and not np.all((arr == 0) | (arr == 1)):
        raise ValueError("state must be binary (0/1)")
    return arr.astype(np.uint8, copy=True)


def all_binary_states(n_vars: int) -> np.ndarray:
    """All ``2**n`` binary vectors; row ``i`` is the bit pattern of ``i``.

    Vectorized on purpose: n=20 is a million rows and a Python loop over
    those rows is the difference between "fine" and "unusable".
    """
    if n_vars < 0:
        raise ValueError(f"n_vars must be non-negative, got {n_vars}")
    if n_vars > _MAX_ENUMERATE_VARS:
        raise ValueError(
            f"refusing to allocate 2^{n_vars} states (max {_MAX_ENUMERATE_VARS})"
        )
    n_states = 1 << n_vars
    idx = np.arange(n_states, dtype=np.uint32)
    shifts = np.arange(n_vars, dtype=np.uint32)
    return ((idx[:, None] >> shifts[None, :]) & np.uint32(1)).astype(np.uint8)


def state_to_index(x: np.ndarray) -> int:
    """Inverse of ``all_binary_states``: pack a binary vector into an int."""
    x = as_binary_vector(x)
    if x.size > 62:
        raise ValueError("state_to_index supports at most 62 variables")
    powers = np.left_shift(1, np.arange(x.size, dtype=np.int64))
    return int(x.astype(np.int64) @ powers)


def states_to_indices(states: np.ndarray) -> np.ndarray:
    """Pack a batch of binary rows into integer indices."""
    arr = np.asarray(states)
    if arr.ndim != 2:
        raise ValueError("states must have shape (n_samples, n_vars)")
    n_vars = arr.shape[1]
    if n_vars > 62:
        raise ValueError("states_to_indices supports at most 62 variables")
    if arr.size > 0 and not np.all((arr == 0) | (arr == 1)):
        raise ValueError("states must be binary (0/1)")
    powers = np.left_shift(1, np.arange(n_vars, dtype=np.int64))
    return arr.astype(np.int64) @ powers
