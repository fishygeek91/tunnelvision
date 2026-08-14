"""Classical parallel tempering — exact-side replica exchange.

This is the third arm of E03: a *target*-temperature ladder, not a
proposal-temperature ladder. Every replica's accept/reject uses the
exact log-density (scaled by β). Kernels still never see log-probs.

The swap ratio is textbook (Geyer 1991; Earl & Deem, PCCP 2005):

    α = min(1, exp((β_i − β_j) (log p(x_j) − log p(x_i))))

with untempered log p. A mixture of *proposal* noise (NoiseLadderKernel)
is a different object and lives on the heuristic side.

Cost is honest: K replicas cost K target evaluations per sweep. ESS
per evaluation, not just per cold-chain step, is the fair scoreboard
column against a single-chain baseline.
"""

from __future__ import annotations

import numpy as np

from tunnelvision.bits import as_binary_vector
from tunnelvision.engine import ChainResult, MetropolisEngine
from tunnelvision.kernels.base import Kernel
from tunnelvision.kernels.classical import AddDeleteSwap
from tunnelvision.targets.base import Target


def geometric_beta_ladder(n_replicas: int, beta_min: float = 0.2) -> np.ndarray:
    """Geometric inverse-temperature schedule from 1 down to ``beta_min``.

    Equal log-spacing is the usual default when the energy scale is
    unknown; a hand-tuned ladder can be passed to ``ParallelTempering``
    instead. ``n_replicas=1`` is the degenerate MH case (β = 1 only).
    """
    if n_replicas < 1:
        raise ValueError(f"n_replicas must be at least 1, got {n_replicas}")
    if n_replicas == 1:
        return np.array([1.0], dtype=np.float64)
    lo = float(beta_min)
    if not np.isfinite(lo) or lo <= 0.0 or lo > 1.0:
        raise ValueError(f"beta_min must lie in (0, 1], got {beta_min}")
    return np.geomspace(1.0, lo, n_replicas)


def replica_exchange_log_alpha(
    beta_i: float,
    beta_j: float,
    logp_i: float,
    logp_j: float,
) -> float:
    """Log Hastings ratio for swapping untempered states between β_i and β_j.

    Joint target π_i(x_i) π_j(x_j) with π_k(x) ∝ exp(β_k log p(x)).
    The swap (x_i, x_j) → (x_j, x_i) has ratio
    exp((β_i − β_j)(log p(x_j) − log p(x_i))).
    """
    return (float(beta_i) - float(beta_j)) * (float(logp_j) - float(logp_i))


def _validate_betas(betas: np.ndarray) -> np.ndarray:
    arr = np.asarray(betas, dtype=np.float64)
    if arr.ndim != 1 or arr.size < 1:
        raise ValueError("betas must be a non-empty 1-d array")
    if not np.isfinite(arr).all():
        raise ValueError("betas must be finite")
    if np.any(arr <= 0.0):
        raise ValueError("betas must be positive")
    if not np.isclose(arr[0], 1.0, atol=1e-12, rtol=0.0):
        raise ValueError(f"cold replica must have beta=1, got {arr[0]}")
    if arr.size > 1 and np.any(arr[1:] >= arr[:-1] - 1e-15):
        raise ValueError("betas must be strictly decreasing")
    out = arr.copy()
    out[0] = 1.0
    return out


def _untemper(tempered_logp: float, beta: float) -> float:
    """Invert β · log p. −∞ stays −∞; β=0 is rejected at construction."""
    if not np.isfinite(tempered_logp):
        return float(tempered_logp)
    return float(tempered_logp) / float(beta)


class TemperedTarget(Target):
    """Exact target with log-density scaled by an inverse temperature β.

    β = 1 is the original posterior. β < 1 flattens it. This is still
    the exact side: the numbers are the real log-prob, just multiplied.
    Surrogates do not belong here.
    """

    def __init__(self, inner: Target, beta: float) -> None:
        if not isinstance(inner, Target):
            raise TypeError("inner must be a Target")
        value = float(beta)
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"beta must be positive and finite, got {beta}")
        self.inner = inner
        self.beta = value
        self.n_vars = inner.n_vars

    def log_prob(self, x: np.ndarray) -> float:
        logp = float(self.inner.log_prob(x))
        if not np.isfinite(logp):
            return logp
        return self.beta * logp

    def log_prob_batch(self, X: np.ndarray) -> np.ndarray:
        logp = np.asarray(self.inner.log_prob_batch(X), dtype=np.float64)
        out = self.beta * logp
        nonfinite = ~np.isfinite(logp)
        if np.any(nonfinite):
            out = out.copy()
            out[nonfinite] = logp[nonfinite]
        return out


class ParallelTempering:
    """Replica-exchange MCMC against a geometric (or supplied) β ladder.

    Local moves go through ``MetropolisEngine.step`` on a
    ``TemperedTarget``, so the Hastings ratio is the same audited
    code path as a single chain. Swaps use untempered log p and the
    adjacent even/odd pairing (no state-dependent rung choice).
    """

    target: Target
    kernel: Kernel
    betas: np.ndarray
    name: str = "parallel-tempering"

    def __init__(
        self,
        target: Target,
        kernel: Kernel | None = None,
        betas: np.ndarray | None = None,
        n_replicas: int = 5,
        beta_min: float = 0.2,
    ) -> None:
        if n_replicas < 1:
            raise ValueError(f"n_replicas must be at least 1, got {n_replicas}")
        self.target = target
        self.kernel = AddDeleteSwap() if kernel is None else kernel
        if betas is None:
            self.betas = geometric_beta_ladder(n_replicas, beta_min)
        else:
            self.betas = _validate_betas(betas)
        self.name = "parallel-tempering"
        self._engines = [
            MetropolisEngine(TemperedTarget(target, float(beta)), self.kernel)
            for beta in self.betas
        ]

    @property
    def n_replicas(self) -> int:
        return int(self.betas.size)

    def run(self, n_steps: int, x0: np.ndarray, seed: int = 0) -> ChainResult:
        if n_steps < 1:
            raise ValueError(f"n_steps must be at least 1, got {n_steps}")

        n_vars = self.target.n_vars
        n_rep = self.n_replicas
        start = as_binary_vector(x0, n_vars)
        rng = np.random.Generator(np.random.PCG64(seed))

        states = [start.copy() for _ in range(n_rep)]
        untempered = [float(self.target.log_prob(start)) for _ in range(n_rep)]
        tempered = [float(beta) * lp for beta, lp in zip(self.betas, untempered, strict=True)]
        n_target_evals = n_rep

        cold_states = np.empty((n_steps, n_vars), dtype=np.uint8)
        cold_logp = np.empty(n_steps, dtype=np.float64)
        cold_accepted = np.empty(n_steps, dtype=bool)
        local_accepts = np.zeros(n_rep, dtype=np.int64)
        n_pairs = max(n_rep - 1, 0)
        swap_attempts = np.zeros(n_pairs, dtype=np.int64)
        swap_accepts = np.zeros(n_pairs, dtype=np.int64)

        for sweep in range(n_steps):
            cold_took = False
            for i in range(n_rep):
                nxt, nxt_tempered, took = self._engines[i].step(states[i], tempered[i], rng)
                n_target_evals += 1
                if took:
                    states[i] = nxt
                    tempered[i] = nxt_tempered
                    untempered[i] = _untemper(nxt_tempered, float(self.betas[i]))
                    local_accepts[i] += 1
                    if i == 0:
                        cold_took = True
            cold_accepted[sweep] = cold_took

            if n_rep >= 2:
                offset = sweep % 2
                for i in range(offset, n_rep - 1, 2):
                    swap_attempts[i] += 1
                    log_alpha = replica_exchange_log_alpha(
                        float(self.betas[i]),
                        float(self.betas[i + 1]),
                        untempered[i],
                        untempered[i + 1],
                    )
                    if log_alpha >= 0.0 or rng.random() < np.exp(log_alpha):
                        states[i], states[i + 1] = states[i + 1], states[i]
                        untempered[i], untempered[i + 1] = untempered[i + 1], untempered[i]
                        tempered[i] = float(self.betas[i]) * untempered[i]
                        tempered[i + 1] = float(self.betas[i + 1]) * untempered[i + 1]
                        swap_accepts[i] += 1

            cold_states[sweep] = states[0]
            cold_logp[sweep] = untempered[0]

        swap_rates = np.full(n_pairs, np.nan, dtype=np.float64)
        attempted = swap_attempts > 0
        swap_rates[attempted] = swap_accepts[attempted] / swap_attempts[attempted]
        local_rates = local_accepts.astype(np.float64) / float(n_steps)
        return ChainResult(
            states=cold_states,
            log_probs=cold_logp,
            accepted=cold_accepted,
            kernel_name=self.name,
            seed=seed,
            meta={
                "n_steps": n_steps,
                "n_vars": n_vars,
                "n_replicas": n_rep,
                "betas": self.betas.tolist(),
                "inner_kernel": self.kernel.name,
                "n_target_evals": int(n_target_evals),
                "swap_attempts": swap_attempts.tolist(),
                "swap_accepts": swap_accepts.tolist(),
                "swap_accept_rates": swap_rates.tolist(),
                "local_accept_rates": local_rates.tolist(),
            },
        )
