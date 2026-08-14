"""Mixing diagnostics. The scoreboard for every kernel comparison.

Two evaluation tiers:
    exact (n <= ~14): spectral gap of the full transition matrix — airtight.
    sampled (any n):  integrated autocorrelation time, ESS per step and per
                      QPU-second, PIP error vs. exact enumeration (p <= ~20)
                      or vs. long-run pooled reference (p > 20).
"""

from __future__ import annotations

import numpy as np


def spectral_gap(P: np.ndarray, pi: np.ndarray) -> float:
    """delta = 1 - |lambda_2| of the transition matrix P with stationary pi.

    Verifies stationarity (pi @ P == pi) before computing — a failed check
    means a kernel is lying about its proposal probabilities.
    """
    raise NotImplementedError  # Rung 1


def integrated_autocorrelation_time(x: np.ndarray) -> float:
    raise NotImplementedError  # Rung 1


def ess(x: np.ndarray) -> float:
    raise NotImplementedError  # Rung 1


def pip_error(states: np.ndarray, exact_pips: np.ndarray) -> float:
    """Max-abs error of estimated posterior inclusion probabilities."""
    raise NotImplementedError  # Rung 1
