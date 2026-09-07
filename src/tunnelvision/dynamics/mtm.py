"""Multiple-try Metropolis with an isoenergetic ("shell") selection weight.

Motivation (D01): an oracle kernel proposing uniformly from the target's
energy shell {y : |E(y) − E(x)| ≤ ε} matches or beats the quantum quench's
spectral gap, T-independently. The quench is an approximate global shell
sampler; a classical competitor must be one too. MTM (Liu, Liang & Wong,
JASA 2000) lets us *select* shell-like candidates among K cheap symmetric
ones while staying exact.

MTM with symmetric candidate generator T and symmetric λ(x,y):
    w(y|x) = π(y) λ(x,y).
    1. draw y_1..y_K ~ T(·|x); pick y_j ∝ w(y_j|x);
    2. draw reference x*_1..x*_{K-1} ~ T(·|y), set x*_K = x;
    3. accept y with prob min(1, Σ_j w(y_j|x) / Σ_j w(x*_j|y)).

We use λ_{a,ε}(x,y) = (π(x)π(y))^{-a} · exp(−((E(y)−E(x))/ε)²), symmetric,
so the selection weight is  w(y|x) ∝ π(y)^{1−a} · exp(−(ΔE/ε)²).
  a=0, ε=∞ : standard (greedy) MTM.
  a=1      : pure shell selection — π drops out of the choice.
Cost per step: 2K − 1 target evaluations.
"""

from __future__ import annotations

import numpy as np

from tunnelvision.bits import as_binary_vector
from tunnelvision.targets.base import Target


class ShellMTM:
    """Exact MTM sampler with shell-weighted candidate selection."""

    def __init__(
        self,
        target: Target,
        n_try: int = 8,
        eps: float = np.inf,
        a: float = 1.0,
        max_weight: int | None = None,
    ) -> None:
        self.target = target
        self.n = int(target.n_vars)
        self.K = int(n_try)
        self.eps = float(eps)
        self.a = float(a)
        self.wmax = self.n if max_weight is None else int(max_weight)
        self.n_energy_evals = 0
        self.name = f"mtm(K={self.K},eps={self.eps:g},a={self.a:g},w<={self.wmax})"

    def _candidates(self, x: np.ndarray, rng: np.random.Generator, k: int) -> np.ndarray:
        """Symmetric generator: flip a uniformly random mask of weight w~U{1..wmax}."""
        Y = np.repeat(x[None, :], k, axis=0)
        for r in range(k):
            w = int(rng.integers(1, self.wmax + 1))
            idx = rng.choice(self.n, size=w, replace=False)
            Y[r, idx] ^= 1
        return Y

    def _log_w(self, logp_ref: float, logp_c: np.ndarray) -> np.ndarray:
        dE = logp_ref - logp_c  # E(y) − E(x) in log-prob units
        lw = (1.0 - self.a) * logp_c
        if np.isfinite(self.eps):
            lw = lw - (dE / self.eps) ** 2
        # λ's (π(x)π(y))^{-a} factor contributes −a·logp_x, common to all
        # candidates from the same x; it cancels within the selection but
        # NOT between numerator and denominator of the acceptance ratio.
        return lw - self.a * logp_ref

    def step(self, x: np.ndarray, logp_x: float, rng: np.random.Generator):
        Y = self._candidates(x, rng, self.K)
        logp_Y = np.asarray(self.target.log_prob_batch(Y), dtype=np.float64)
        self.n_energy_evals += self.K
        lw_y = self._log_w(logp_x, logp_Y)
        m = lw_y.max()
        if not np.isfinite(m):
            return x, logp_x, False
        p = np.exp(lw_y - m)
        j = int(rng.choice(self.K, p=p / p.sum()))
        y, logp_y = Y[j], float(logp_Y[j])
        Xs = self._candidates(y, rng, self.K - 1)
        logp_Xs = np.asarray(self.target.log_prob_batch(Xs), dtype=np.float64)
        self.n_energy_evals += self.K - 1
        lw_x = np.concatenate([self._log_w(logp_y, logp_Xs), self._log_w(logp_y, np.array([logp_x]))])
        log_num = m + np.log(p.sum())
        mx = lw_x.max()
        log_den = mx + np.log(np.exp(lw_x - mx).sum())
        if np.log(rng.random()) < log_num - log_den:
            return y, logp_y, True
        return x, logp_x, False

    def run(self, n_steps: int, x0: np.ndarray, seed: int = 0):
        rng = np.random.default_rng(seed)
        x = as_binary_vector(x0, self.n)
        logp = float(self.target.log_prob(x))
        states = np.empty((n_steps, self.n), dtype=np.uint8)
        lps = np.empty(n_steps)
        acc = np.zeros(n_steps, dtype=bool)
        for t in range(n_steps):
            x, logp, took = self.step(x, logp, rng)
            states[t] = x
            lps[t] = logp
            acc[t] = took
        return states, lps, acc
