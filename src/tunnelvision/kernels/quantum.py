"""Quantum quench proposal kernel (Layden et al., Nature 619, 282 (2023)).

Procedure per proposal:
    1. Prepare |x> (computational basis state of current chain state).
    2. Evolve under H = alpha * H_prob + (1 - alpha) * H_mix for time t,
       where H_prob is the (surrogate) Ising Hamiltonian and H_mix is a
       transverse field. Trotterized for gate-based devices.
    3. Measure in the computational basis -> proposal y.

Symmetry: with time-symmetric evolution, |<y|U|x>|^2 == |<x|U|y>|^2, so
the proposal is symmetric and log-q terms are exactly 0.0 — even under
depolarizing noise (unital channels preserve this; see docs/ARCHITECTURE.md
for the argument and its limits — amplitude damping is NOT unital, and the
symmetric-kernel assumption must be validated per backend in E03).

Randomize (alpha, t) per proposal within ranges (Layden's approach) to
avoid pathological resonances.
"""

from __future__ import annotations

import numpy as np

from tunnelvision.kernels.base import Kernel


class QuenchKernel(Kernel):
    """Simulator-backed quench proposals via Qiskit Aer (statevector or noisy)."""

    name = "quench"

    def __init__(
        self,
        h: np.ndarray,          # surrogate Ising fields, shape (n,)
        J: np.ndarray,          # surrogate Ising couplings, shape (n, n) upper-tri
        alpha_range: tuple[float, float] = (0.5, 0.9),
        t_range: tuple[float, float] = (2.0, 20.0),
        trotter_steps: int = 20,
        noise_model=None,        # qiskit-aer NoiseModel or None
        shots_per_proposal: int = 1,
    ) -> None:
        raise NotImplementedError  # Rung 1 (simulator), Rung 2 (hardware backend)

    def propose(self, x, rng):
        raise NotImplementedError

    def proposal_matrix(self, n_vars: int) -> np.ndarray:
        """Exact |<y|U|x>|^2 averaged over (alpha, t) draws — n <= ~14."""
        raise NotImplementedError


class HardwareQuenchKernel(QuenchKernel):
    """IBM Runtime backend. Batches proposals: pre-samples a pool of
    (state -> proposal) pairs per job to amortize queue latency.
    Credentials via environment only (see .cursor/rules) — never in code.
    """

    name = "quench-hw"
