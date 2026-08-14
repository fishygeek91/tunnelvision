"""Mixing diagnostics. The scoreboard for every kernel comparison.

Two evaluation tiers:
    exact (n <= ~14): spectral gap of the full transition matrix — airtight.
    sampled (any n):  integrated autocorrelation time, ESS per step and per
                      QPU-second, PIP error vs. exact enumeration (p <= ~20)
                      or vs. long-run pooled reference (p > 20).
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import eigvals

# Sokal's adaptive window constant. c≈5 is the usual choice: smaller c
# under-smooths (noise-dominated IAT); much larger c wastes lag range.
_SOKAL_C = 5.0
_STATIONARITY_ATOL = 1e-8


def spectral_gap(P: np.ndarray, pi: np.ndarray) -> float:
    """delta = 1 - |lambda_2| of the transition matrix P with stationary pi.

    The MH matrix is not symmetric, so this uses a dense general eigen-
    solver (fine at n≤14). Stationarity πP = π is checked first and
    raised on loudly: a failure here means a kernel is lying about its
    log-q, and that must be caught at the diagnostic, not in a paper draft.
    """
    P_arr = np.asarray(P, dtype=np.float64)
    if P_arr.ndim != 2 or P_arr.shape[0] != P_arr.shape[1]:
        raise ValueError("P must be a square 2-d array")
    n_states = P_arr.shape[0]
    if n_states == 0:
        raise ValueError("P must not be empty")

    pi_arr = np.asarray(pi, dtype=np.float64).reshape(-1)
    if pi_arr.shape[0] != n_states:
        raise ValueError(f"pi length {pi_arr.shape[0]} does not match P shape {n_states}")
    if np.any(pi_arr < -1e-15):
        raise ValueError("pi must be non-negative")
    total = float(pi_arr.sum())
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError("pi must sum to a positive finite value")
    pi_arr = pi_arr / total

    residual = pi_arr @ P_arr - pi_arr
    max_abs = float(np.max(np.abs(residual)))
    if max_abs > _STATIONARITY_ATOL:
        raise ValueError(
            f"stationarity failed (max |pi @ P - pi| = {max_abs:.3e}); "
            "the kernel is lying about its proposal probabilities"
        )

    if n_states == 1:
        return 0.0

    # Dense eig is the honest choice for a non-symmetric MH matrix.
    # (A π^{1/2} similarity transform is only valid for reversible kernels.)
    eigenvalues = eigvals(P_arr)
    moduli = np.abs(eigenvalues)
    order = np.argsort(-moduli)
    gap = 1.0 - float(moduli[order[1]])
    return max(gap, 0.0)


def _normalized_acf(x: np.ndarray) -> np.ndarray:
    """FFT autocorrelation, zero-padded so the sum is not circular."""
    n = x.size
    centered = x - x.mean()
    spectrum = np.fft.rfft(centered, n=2 * n)
    acf = np.fft.irfft(spectrum * np.conjugate(spectrum), n=2 * n)[:n].real
    if acf[0] == 0.0:
        raise RuntimeError("autocorrelation lag-0 is zero")
    return acf / acf[0]


def integrated_autocorrelation_time(x: np.ndarray, c: float = _SOKAL_C) -> float:
    """Sokal windowed IAT. Naive full-length sums are noise-dominated.

    τ = 1 + 2 Σ_{t=1}^M ρ(t), with M the smallest lag such that
    M ≥ c τ(M) (Sokal, 'Monte Carlo Methods in Statistical Mechanics').
    """
    series = np.asarray(x, dtype=np.float64)
    if series.ndim != 1:
        raise ValueError("x must be a 1-d series")
    n = int(series.size)
    if n < 2:
        raise ValueError("need at least 2 samples to estimate IAT")
    if c <= 0.0:
        raise ValueError(f"Sokal window constant c must be positive, got {c}")
    if not np.isfinite(series).all():
        raise ValueError("x contains non-finite values")
    if float(np.var(series)) <= 0.0:
        return 1.0

    acf = _normalized_acf(series)
    # τ(M) = 1 + 2 Σ_{t=1}^M ρ(t) = 2·cumsum(ρ)[M] − 1.
    taus = 2.0 * np.cumsum(acf) - 1.0
    for lag in range(1, n):
        tau = float(taus[lag])
        if lag >= c * tau:
            return max(tau, 1.0)
    return max(float(taus[-1]), 1.0)


def ess(x: np.ndarray) -> float:
    """Effective sample size n / τ_int. Requires the windowed IAT."""
    series = np.asarray(x, dtype=np.float64)
    if series.ndim != 1:
        raise ValueError("x must be a 1-d series")
    tau = integrated_autocorrelation_time(series)
    return float(series.size / tau)


def pip_error(states: np.ndarray, exact_pips: np.ndarray) -> float:
    """Max-abs error of estimated posterior inclusion probabilities."""
    arr = np.asarray(states)
    pips = np.asarray(exact_pips, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError("states must have shape (n_steps, n_vars)")
    if pips.ndim != 1 or pips.shape[0] != arr.shape[1]:
        raise ValueError("exact_pips must have shape (n_vars,)")
    if arr.size > 0 and not np.all((arr == 0) | (arr == 1)):
        raise ValueError("states must be binary (0/1)")
    estimated = arr.astype(np.float64).mean(axis=0)
    return float(np.max(np.abs(estimated - pips)))
