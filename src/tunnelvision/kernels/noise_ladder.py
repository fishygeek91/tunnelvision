"""Maxwell's Daemon — scheduled noise as a proposal-temperature ladder (N2).

Idea: run the same quench circuit at K deliberately different noise
levels. Each level is one rung of a ladder of effectively hotter
proposal distributions. Compose rungs like parallel tempering — but
the replicas differ in PROPOSAL heat, not target temperature, so
every rung samples the same exact target and no replica-exchange
bias correction is needed.

Open scientific question this module exists to answer (experiment E03):
does noise-heating help mixing, or does it destroy the tunneling
structure that creates the quantum advantage? Either answer is a result.

Caution on exactness: noise levels must be FIXED per rung (a fixed
unital channel keeps the kernel symmetric). Adapting the rung based
on chain state or history requires diminishing-adaptation conditions
— see AdaptiveMixture. Random-scan over a fixed list is a uniform
mixture of symmetric kernels, hence symmetric: log-q is 0.0.
"""

from __future__ import annotations

import numpy as np

from tunnelvision.kernels.base import Kernel


def boltzmann_mean_energy(energies: np.ndarray, temperature: float) -> float:
    """Mean energy of exp(−E/T) / Z. T ≤ 0 or non-finite is the uniform law."""
    e = np.asarray(energies, dtype=np.float64)
    if e.size == 0:
        raise ValueError("energies must not be empty")
    if not np.isfinite(temperature) or temperature <= 0.0:
        return float(e.mean())
    shifted = -(e - float(e.min())) / temperature
    shifted = shifted - float(shifted.max())
    weights = np.exp(shifted)
    return float(np.dot(weights, e) / weights.sum())


def invert_temperature(energies: np.ndarray, target_mean: float) -> float:
    """T such that Boltzmann mean energy equals ``target_mean``.

    The image of T ∈ (0, ∞) is (min E, mean E). At or below the
    ground state we return 0; at or above the uniform mean, +∞.
    Used by E03 to turn ⟨E(y)⟩ of a proposal into T_eff.
    """
    e = np.asarray(energies, dtype=np.float64)
    e_min = float(e.min())
    e_hot = float(e.mean())
    target = float(target_mean)
    if not np.isfinite(target):
        raise ValueError("target_mean must be finite")
    if target <= e_min + 1e-12:
        return 0.0
    if target >= e_hot - 1e-12:
        return float("inf")
    lo, hi = 1e-8, 1e8
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if boltzmann_mean_energy(e, mid) < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


class NoiseLadderKernel(Kernel):
    """Uniform random-scan mixture of fixed, symmetric rungs.

    ``proposal_matrix`` is the average of the rung matrices — exact
    for a state-independent scan. Round-robin would also be symmetric
    if the period is independent of the chain; it is not implemented
    because the exact-tier object we score is the random-scan mixture.
    """

    name = "maxwells-daemon"

    def __init__(self, rungs: list[Kernel], scan: str = "random") -> None:
        if not rungs:
            raise ValueError("NoiseLadderKernel requires at least one rung")
        if scan != "random":
            raise ValueError(
                "only scan='random' is implemented; a state- or "
                "history-dependent rung choice needs explicit log-q "
                "or diminishing-adaptation theory"
            )
        self.rungs = list(rungs)
        self.scan = scan
        self._proposal_cache: np.ndarray | None = None
        self._cached_n_vars: int | None = None

    def propose(self, x: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, float, float]:
        index = int(rng.integers(len(self.rungs)))
        y, log_fwd, log_rev = self.rungs[index].propose(x, rng)
        # Mixture of symmetric kernels is symmetric. If a rung ever
        # returns nonzero log-q, the mixture log-q is not the rung's
        # log-q — refuse rather than silently break the Hastings ratio.
        if log_fwd != 0.0 or log_rev != 0.0:
            raise RuntimeError(
                f"rung {self.rungs[index].name} is not symmetric; "
                "NoiseLadderKernel does not account for mixture log-q"
            )
        return y, 0.0, 0.0

    def proposal_matrix(self, n_vars: int) -> np.ndarray:
        if self._proposal_cache is not None and self._cached_n_vars == n_vars:
            return self._proposal_cache
        matrices = [
            np.asarray(rung.proposal_matrix(n_vars), dtype=np.float64) for rung in self.rungs
        ]
        shape = matrices[0].shape
        if any(mat.shape != shape for mat in matrices):
            raise ValueError("all rungs must return proposal matrices of the same shape")
        mixed = np.mean(np.stack(matrices, axis=0), axis=0)
        np.clip(mixed, 0.0, None, out=mixed)
        row_sums = mixed.sum(axis=1, keepdims=True)
        if np.any(row_sums <= 0.0):
            raise ValueError("ladder proposal has a zero row")
        mixed = mixed / row_sums
        self._proposal_cache = mixed
        self._cached_n_vars = n_vars
        return mixed


class AdaptiveMixture(Kernel):
    """N3: online-adapted mixture of quantum and classical kernels.

    Weights adapt from acceptance-rate feedback under diminishing
    adaptation (Roberts & Rosenthal 2007) to preserve ergodicity.
    """

    name = "adaptive-mixture"

    def __init__(self, kernels: list[Kernel]) -> None:
        raise NotImplementedError  # Rung 3 / WP7

    def propose(
        self, x: np.ndarray, rng: np.random.Generator
    ) -> tuple[np.ndarray, float, float]:
        raise NotImplementedError  # Rung 3 / WP7
