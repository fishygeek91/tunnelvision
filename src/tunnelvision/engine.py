"""Metropolis–Hastings engine — the exactness boundary.

Everything quantum, noisy, or heuristic lives on the OTHER side of the
Kernel interface. This module must stay boring, classical, and provably
correct. If you are editing this file to make a kernel's life easier,
you are probably about to break the project's core guarantee.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from tunnelvision.bits import all_binary_states, as_binary_vector
from tunnelvision.kernels.base import Kernel
from tunnelvision.targets.base import Target

# Exact diagonalization is O(8^n) memory in the naive sense (2^n × 2^n).
# n=14 is 16k states / 2GB float64; above that is a different tier.
_MAX_TRANSITION_VARS = 14
_ROW_SUM_ATOL = 1e-12


@dataclass
class ChainResult:
    """Samples plus bookkeeping from one chain run."""

    states: np.ndarray  # (n_steps, n_vars) uint8
    log_probs: np.ndarray  # (n_steps,) target log-prob of each state
    accepted: np.ndarray  # (n_steps,) bool
    kernel_name: str
    seed: int
    meta: dict = field(default_factory=dict)

    @property
    def acceptance_rate(self) -> float:
        return float(self.accepted.mean())


class MetropolisEngine:
    """Kernel-agnostic Metropolis–Hastings over binary state spaces.

    Accept probability: min(1, exp(logp(y) - logp(x) + log q(x|y) - log q(y|x))).
    Kernels must implement propose(x, rng) -> (y, log_q_forward, log_q_reverse).
    Symmetric kernels return 0.0 for both log-q terms.

    The Hastings ratio lives only here. A kernel that proposes from the
    wrong distribution but reports honest log-q is still exact — that is
    the invariant tests/test_exactness.py::test_broken_kernel_still_exact
    exists to protect.
    """

    def __init__(self, target: Target, kernel: Kernel) -> None:
        self.target = target
        self.kernel = kernel

    def run(self, n_steps: int, x0: np.ndarray, seed: int = 0) -> ChainResult:
        if n_steps < 1:
            raise ValueError(f"n_steps must be at least 1, got {n_steps}")

        n_vars = self.target.n_vars
        x = as_binary_vector(x0, n_vars)
        rng = np.random.Generator(np.random.PCG64(seed))

        states = np.empty((n_steps, n_vars), dtype=np.uint8)
        log_probs = np.empty(n_steps, dtype=np.float64)
        accepted = np.empty(n_steps, dtype=bool)

        logp_x = float(self.target.log_prob(x))
        for step in range(n_steps):
            proposal, log_q_fwd, log_q_rev = self.kernel.propose(x, rng)
            y = as_binary_vector(proposal, n_vars)
            logp_y = float(self.target.log_prob(y))
            # MH log-accept: Δlogp + log q(x|y) − log q(y|x).
            log_alpha = (logp_y - logp_x) + float(log_q_rev) - float(log_q_fwd)
            if log_alpha >= 0.0 or rng.random() < np.exp(log_alpha):
                x = y
                logp_x = logp_y
                accepted[step] = True
            else:
                accepted[step] = False
            states[step] = x
            log_probs[step] = logp_x

        return ChainResult(
            states=states,
            log_probs=log_probs,
            accepted=accepted,
            kernel_name=self.kernel.name,
            seed=seed,
            meta={"n_steps": n_steps, "n_vars": n_vars},
        )

    def transition_matrix(self) -> np.ndarray:
        """Exact transition matrix over all 2^n states (n <= ~14 only).

        P[x, y] = Q[x, y] A[x, y] for y ≠ x, with Hastings accept
        A[x, y] = min(1, exp(Δlogp + log Q[y, x] − log Q[x, y])).
        Rejection mass belongs on the diagonal: P[x, x] = 1 − Σ_{y≠x} P[x, y].
        Forgetting that self-loop is the classic bug — rows will not sum
        to 1 and the spectral gap comes out wrong but not obviously wrong.
        """
        n_vars = self.target.n_vars
        if n_vars > _MAX_TRANSITION_VARS:
            raise ValueError(
                f"transition_matrix supports n_vars <= {_MAX_TRANSITION_VARS}, got {n_vars}"
            )
        n_states = 1 << n_vars
        Q = np.asarray(self.kernel.proposal_matrix(n_vars), dtype=np.float64)
        if Q.shape != (n_states, n_states):
            raise ValueError(
                f"proposal_matrix must return shape {(n_states, n_states)}, got {Q.shape}"
            )
        if np.any(Q < -1e-15):
            raise ValueError("proposal_matrix contains negative entries")

        states = all_binary_states(n_vars)
        logp = np.asarray(self.target.log_prob_batch(states), dtype=np.float64)
        if logp.shape != (n_states,):
            raise ValueError("log_prob_batch must return one value per state")

        log_Q = np.full((n_states, n_states), -np.inf, dtype=np.float64)
        positive = Q > 0.0
        log_Q[positive] = np.log(Q[positive])

        # Hastings ratio only on proposed pairs. Evaluating log Q[y,x] −
        # log Q[x,y] on the structural zeros gives −∞ − (−∞) = NaN and is
        # the classic "invalid value in subtract" warning; those entries
        # are already A = 0 because Q[x,y] = 0.
        log_hastings = np.full((n_states, n_states), -np.inf, dtype=np.float64)
        finite_logp = np.isfinite(logp)
        defined = positive & finite_logp[:, None] & finite_logp[None, :]
        log_hastings[defined] = (
            (logp[None, :] - logp[:, None])[defined] + log_Q.T[defined] - log_Q[defined]
        )
        # Leave a zero-mass state toward any finite-mass proposal (A = 1).
        escape = positive & ~finite_logp[:, None] & finite_logp[None, :]
        log_hastings[escape] = 0.0

        accept = np.zeros((n_states, n_states), dtype=np.float64)
        finite = np.isfinite(log_hastings)
        accept[finite] = np.exp(np.minimum(0.0, log_hastings[finite]))

        P = np.zeros((n_states, n_states), dtype=np.float64)
        np.copyto(P, Q * accept, where=Q > 0.0)
        np.fill_diagonal(P, 0.0)
        off_diag = P.sum(axis=1)
        if np.any(off_diag > 1.0 + _ROW_SUM_ATOL):
            raise RuntimeError(
                "off-diagonal mass exceeds 1; proposal_matrix rows may not be a "
                f"sub-probability (max row mass {float(off_diag.max()):.16f})"
            )
        np.fill_diagonal(P, 1.0 - off_diag)

        row_sums = P.sum(axis=1)
        if not np.allclose(row_sums, 1.0, atol=_ROW_SUM_ATOL, rtol=0.0):
            err = float(np.max(np.abs(row_sums - 1.0)))
            raise RuntimeError(
                f"transition matrix rows must sum to 1 within {_ROW_SUM_ATOL}, "
                f"max abs error {err:.3e}"
            )
        return P
