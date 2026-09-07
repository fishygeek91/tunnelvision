"""DEMON — Creutz-demon involutive proposals ("discrete HMC").

Idea (IDEATION-01, 2026-08-25). The quench proposal wins on Ising because
unitary dynamics approximately conserves the target energy while moving
far, so the Metropolis filter rarely rejects. That mechanism is available
classically, exactly, and without a QPU: Creutz's microcanonical demon
(Phys. Rev. Lett. 50, 1411 (1983)).

Extended state (x, d) with d ≥ 0 a "demon" energy, joint target
π(x)·exp(−d) (energies measured in units of the target temperature, i.e.
E(x) = −log π(x)). One proposal:

    1. refresh d ~ Exp(1); draw a site order σ from a family closed under
       reversal (each order and its reverse equally likely);
    2. sweep σ deterministically: at site i compute ΔE for flipping;
       flip iff T_d·d ≥ ΔE, then d ← d − ΔE/T_d.  E(x) + T_d·d is conserved.

The reverse sweep from (y, d') undoes the forward sweep exactly, so the
map (x, d, σ) ↦ (y, d', σᴿ) is an involution on a discrete×continuous
space with unit Jacobian. Hence

    q(x|y)/q(y|x) = ρ(d')/ρ(d) = exp(d − d'),

and the MH log-acceptance is Δlog π + d − d' = −ΔE(1 − 1/T_d), which is
identically 0 at T_d = 1: rejection-free and exact. T_d ≠ 1 is a knob
that buys a bigger energy budget at the cost of Metropolis rejections;
the engine's Hastings term keeps it exact either way.

This module reads the target energy (it must, to move the demon). The
Hastings term reported to the engine depends on (d, d') only.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

from tunnelvision.bits import as_binary_vector
from tunnelvision.kernels.base import Kernel
from tunnelvision.targets.base import Target
from tunnelvision.targets.ising import IsingTarget

Order = Literal["cyclic", "random"]


class DemonKernel(Kernel):
    """Deterministic energy-conserving multi-flip proposal with a demon."""

    name = "demon"

    def __init__(
        self,
        target: Target,
        sweep_len: int | None = None,
        demon_temperature: float = 1.0,
        order: Order = "cyclic",
    ) -> None:
        n = int(target.n_vars)
        if n < 1:
            raise ValueError("DemonKernel requires at least one variable")
        L = n if sweep_len is None else int(sweep_len)
        if L < 1:
            raise ValueError(f"sweep_len must be at least 1, got {L}")
        if demon_temperature <= 0.0:
            raise ValueError("demon_temperature must be positive")
        if order not in ("cyclic", "random"):
            raise ValueError(f"order must be 'cyclic' or 'random', got {order!r}")
        if order == "random" and L > n:
            raise ValueError("random order visits each site at most once: sweep_len <= n_vars")
        self.target = target
        self.n_vars = n
        self.sweep_len = L
        self.demon_temperature = float(demon_temperature)
        self.order = order
        self.n_energy_evals = 0  # honest cost accounting
        self.name = f"demon(L={L},Td={self.demon_temperature:g},{order})"

        # Fast local ΔE for Ising: O(n) per site instead of a full log_prob.
        self._ising = isinstance(target, IsingTarget)

    # ---- energy ---------------------------------------------------------

    def _delta_energy(self, x: np.ndarray, i: int, logp_x: float | None) -> tuple[float, float]:
        """ΔE = E(x ⊕ e_i) − E(x) with E = −log π. Returns (ΔE, logp of flipped)."""
        self.n_energy_evals += 1
        if self._ising:
            tgt: IsingTarget = self.target  # type: ignore[assignment]
            s = 2.0 * x.astype(np.float64) - 1.0
            # E = -½ sᵀJs - h·s ; flipping s_i changes E by 2 s_i (J_i·s + h_i)
            local = float(tgt.J[i] @ s + tgt.h[i])
            dE = 2.0 * s[i] * local / tgt.temperature
            return dE, (logp_x - dE) if logp_x is not None else -np.inf
        y = x.copy()
        y[i] ^= 1
        logp_y = float(self.target.log_prob(y))
        if logp_x is None:
            logp_x = float(self.target.log_prob(x))
        return logp_x - logp_y, logp_y

    # ---- orders ---------------------------------------------------------

    def _draw_order(self, rng: np.random.Generator) -> np.ndarray:
        n, L = self.n_vars, self.sweep_len
        if self.order == "cyclic":
            start = int(rng.integers(0, n))
            direction = 1 if rng.random() < 0.5 else -1
            return (start + direction * np.arange(L)) % n
        return rng.permutation(n)[:L]

    def _all_orders(self) -> list[np.ndarray]:
        """Every order in the family with equal probability (exact tier)."""
        n, L = self.n_vars, self.sweep_len
        if self.order == "cyclic":
            out = []
            for start in range(n):
                for direction in (1, -1):
                    out.append((start + direction * np.arange(L)) % n)
            return out
        from itertools import permutations

        return [np.asarray(p) for p in permutations(range(n), L)]

    # ---- dynamics -------------------------------------------------------

    def sweep(
        self, x: np.ndarray, d: float, order: np.ndarray, logp_x: float | None = None
    ) -> tuple[np.ndarray, float, float | None]:
        """Deterministic demon sweep. Conserves E(x) + T_d·d exactly."""
        Td = self.demon_temperature
        y = x.copy()
        for i in order:
            dE, logp_flip = self._delta_energy(y, int(i), logp_x)
            if Td * d >= dE:
                y[int(i)] ^= 1
                d -= dE / Td
                logp_x = logp_flip if logp_x is not None else None
        return y, d, logp_x

    def propose(self, x: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, float, float]:
        x_bin = as_binary_vector(x, self.n_vars)
        d0 = float(rng.exponential())
        order = self._draw_order(rng)
        logp0 = None if self._ising else float(self.target.log_prob(x_bin))
        y, d1, _ = self.sweep(x_bin, d0, order, logp0)
        # log q(x|y) − log q(y|x) = log ρ(d1) − log ρ(d0) = d0 − d1.
        return y, 0.0, d0 - d1

    # ---- exact tier -----------------------------------------------------

    def proposal_matrix(self, n_vars: int) -> np.ndarray:
        """Exact q(y|x): integrate the piecewise-constant sweep over d ~ Exp(1).

        For a fixed order the outcome is constant on d-intervals; we
        recurse over intervals [lo, hi) splitting at each site's threshold.
        """
        if n_vars != self.n_vars:
            raise ValueError("n_vars must match the target")
        if n_vars > 14:
            raise ValueError("exact proposal matrix limited to n_vars <= 14")
        Td = self.demon_temperature
        n_states = 1 << n_vars
        Q = np.zeros((n_states, n_states), dtype=np.float64)
        orders = self._all_orders()
        p_order = 1.0 / len(orders)
        # Cache ΔE table: dE[x, i]
        from tunnelvision.bits import all_binary_states

        states = all_binary_states(n_vars)
        logp = np.asarray(self.target.log_prob_batch(states), dtype=np.float64)
        idx = np.arange(n_states)
        dE = np.empty((n_states, n_vars), dtype=np.float64)
        for i in range(n_vars):
            dE[:, i] = logp[idx] - logp[idx ^ (1 << i)]

        def mass(lo: float, hi: float) -> float:
            return float(np.exp(-lo) - (0.0 if np.isinf(hi) else np.exp(-hi)))

        for x in range(n_states):
            if not np.isfinite(logp[x]):
                Q[x, x] = 1.0
                continue
            for order in orders:
                # stack of (state, lo, hi, shift, position) — d ∈ [lo,hi) is
                # the *initial* demon; current demon = d − shift.
                stack = [(x, 0.0, np.inf, 0.0, 0)]
                while stack:
                    s, lo, hi, shift, pos = stack.pop()
                    if pos == len(order):
                        Q[x, s] += p_order * mass(lo, hi)
                        continue
                    i = int(order[pos])
                    de = dE[s, i]
                    if not np.isfinite(de):  # flipping into zero mass: never
                        stack.append((s, lo, hi, shift, pos + 1))
                        continue
                    # flip iff Td·(d − shift) ≥ de  ⇔  d ≥ shift + de/Td
                    thr = shift + de / Td
                    if thr <= lo:
                        stack.append((s ^ (1 << i), lo, hi, shift + de / Td, pos + 1))
                    elif thr >= hi:
                        stack.append((s, lo, hi, shift, pos + 1))
                    else:
                        stack.append((s, lo, thr, shift, pos + 1))
                        stack.append((s ^ (1 << i), thr, hi, shift + de / Td, pos + 1))
        return Q
