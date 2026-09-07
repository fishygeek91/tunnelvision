"""Discrete Langevin Proposal (Zhang, Liu & Liu, ICML 2022) with exact local
differences — the classical "soft-spin multi-flip" shell candidate.

Each site flips independently with probability
    p_i(x) = sigmoid( −ΔE_i(x)/2 − 1/(2α) ),   ΔE_i = E(x⊕e_i) − E(x),
so q(y|x) = ∏_i p_i^{m_i} (1−p_i)^{1−m_i}, m = x⊕y. The Hastings term is
computable from the local differences at x and at y; the kernel is exact.
α sets the typical flip count. Cost: two local-difference vectors per step
(O(n²) for Ising, i.e. ≈ 2 energy evaluations) plus the engine's log_prob(y).

Physics motivation (D01): the transverse field flips weak-local-field spins
preferentially; this is the classical proposal that does the same thing.
"""

from __future__ import annotations

import numpy as np

from tunnelvision.bits import all_binary_states, as_binary_vector
from tunnelvision.kernels.base import Kernel
from tunnelvision.targets.base import Target
from tunnelvision.targets.ising import IsingTarget


class DLPKernel(Kernel):
    def __init__(self, target: Target, alpha: float = 1.0, temper: float = 0.5) -> None:
        self.target = target
        self.n = int(target.n_vars)
        self.alpha = float(alpha)
        self.temper = float(temper)  # coefficient on −ΔE (½ is the DLP/locally-balanced choice)
        self.n_energy_evals = 0
        self._ising = isinstance(target, IsingTarget)
        self.name = f"dlp(alpha={self.alpha:g},t={self.temper:g})"

    def local_deltas(self, x: np.ndarray) -> np.ndarray:
        """ΔE_i(x) in log-prob units for every site."""
        self.n_energy_evals += 1
        if self._ising:
            tgt: IsingTarget = self.target  # type: ignore[assignment]
            s = 2.0 * x.astype(np.float64) - 1.0
            return 2.0 * s * (tgt.J @ s + tgt.h) / tgt.temperature
        lp0 = float(self.target.log_prob(x))
        out = np.empty(self.n)
        for i in range(self.n):
            y = x.copy()
            y[i] ^= 1
            out[i] = lp0 - float(self.target.log_prob(y))
        return out

    def _logit(self, dE: np.ndarray) -> np.ndarray:
        return -self.temper * dE - 1.0 / (2.0 * self.alpha)

    @staticmethod
    def _log_bern(logit: np.ndarray, m: np.ndarray) -> float:
        # log p for m=1, log(1-p) for m=0, with p = sigmoid(logit)
        return float(np.sum(np.where(m == 1, -np.logaddexp(0.0, -logit), -np.logaddexp(0.0, logit))))

    def propose(self, x: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, float, float]:
        x_bin = as_binary_vector(x, self.n)
        lx = self._logit(self.local_deltas(x_bin))
        p = 1.0 / (1.0 + np.exp(-lx))
        m = (rng.random(self.n) < p).astype(np.uint8)
        y = x_bin ^ m
        ly = self._logit(self.local_deltas(y))
        return y, self._log_bern(lx, m), self._log_bern(ly, m)

    def proposal_matrix(self, n_vars: int) -> np.ndarray:
        S = all_binary_states(n_vars)
        N = 1 << n_vars
        idx = np.arange(N)
        lp = np.asarray(self.target.log_prob_batch(S), dtype=np.float64)
        dE = np.stack([lp - lp[idx ^ (1 << i)] for i in range(n_vars)], axis=1)  # (N, n)
        logit = self._logit(dE)
        log_p = -np.logaddexp(0.0, -logit)
        log_1mp = -np.logaddexp(0.0, logit)
        masks = S.astype(bool)  # row m = bit pattern of m
        logQ = masks.astype(float) @ log_p.T + (~masks).astype(float) @ log_1mp.T  # (m, x)
        Q = np.zeros((N, N))
        for mi in range(N):
            Q[idx, idx ^ mi] = np.exp(logQ[mi])
        return Q
