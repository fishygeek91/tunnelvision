"""2-local Ising surrogate of the spike-and-slab posterior.

The quench Hamiltonian has to be 2-local. The exact g-prior marginal is
not (it has a log-determinant of an |γ|×|γ| Gram block). So we build a
cheap (h, J) whose *low-energy* states are *high-posterior* models, and
we let Metropolis grade the proposals against the exact log-posterior.

Three additive pieces, then one scale-fixing step:

    fields     h_j  ~ |x_j' y|          marginal evidence for including j
    couplings  J_jk ~ −| (X'X)_jk |     collinear predictors compete (AF)
    sparsity   h_j += ½ log(π/(1−π))    the Bernoulli prior, as a field

Energy convention is the same as ``IsingTarget`` / the quench
(``E = −½ sᵀ J s − h·s``, ``s = 2γ−1``). Negative J is antiferromagnetic:
opposite spins (one in, one out) are cheaper than both-in, which is the
"competing models" story E02 is built to test.

Scale is the whole game. α in the quench already matches ||H_prob||_F to
||H_mix||_F, so an overall factor is invisible. What is *not* invisible
is the relative size of fields vs couplings: a 100×-too-hot J makes the
response a spectator and the quench tunnels between sparse models that
ignore y. We balance RMS(h·s) against RMS(½ sᵀ J s) over random ±1
spins, then pin the Frobenius problem-norm to √p so α = 1.

EXACTNESS NOTE: this module may look at (X, y) and, for the learned
path, at target.log_prob — but only at *construction*. The returned
(h, J) shape proposals. They must never reach accept/reject. If swapping
surrogates changes posterior estimates beyond Monte Carlo error, that is
a P0 wall breach, not a "better surrogate."
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
from scipy.stats import spearmanr

from tunnelvision.bits import all_binary_states
from tunnelvision.design import center_and_scale
from tunnelvision.targets.base import Target
from tunnelvision.targets.ising import IsingTarget, symmetrize_couplings

SurrogateKind = Literal["analytic", "learned"]

# Enumerating 2^n energies is the cheap de-risking plot; above this the
# caller should pass a subsample, not ask for a full scatter.
_MAX_ENUMERATE_VARS = 20


@dataclass(frozen=True)
class IsingSurrogate:
    """A 2-local proposal Hamiltonian. Never an accept/reject input.

    Iterates as ``(h, J)`` so ``h, J = ising_surrogate_from_data(...)``
    still works. Prefer the named fields — E02 needs ``kind`` and ``meta``.
    """

    h: np.ndarray
    J: np.ndarray
    kind: SurrogateKind
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        h_arr = np.asarray(self.h, dtype=np.float64).reshape(-1)
        J_arr = symmetrize_couplings(self.J)
        if J_arr.shape != (h_arr.size, h_arr.size):
            raise ValueError(f"J shape {J_arr.shape} does not match h length {h_arr.size}")
        object.__setattr__(self, "h", h_arr.copy())
        object.__setattr__(self, "J", J_arr.copy())
        object.__setattr__(self, "kind", str(self.kind))
        object.__setattr__(self, "meta", dict(self.meta))

    @property
    def n_vars(self) -> int:
        return int(self.h.size)

    def __iter__(self) -> Iterator[np.ndarray]:
        yield self.h
        yield self.J

    def as_ising(self, temperature: float = 1.0) -> IsingTarget:
        """The same Hamiltonian the quench and the Boltzmann target use."""
        return IsingTarget(self.h, self.J, temperature=temperature)

    def energy(self, gamma: np.ndarray) -> float:
        return self.as_ising().energy(gamma)

    def energies(self, states: np.ndarray) -> np.ndarray:
        """Vectorized E(γ). ``IsingTarget.log_prob_batch`` is −E at T=1."""
        return -self.as_ising().log_prob_batch(states)

    def ground_state(self) -> np.ndarray:
        """Exact argmin of E over {0,1}^p. p ≤ 20 only — this is a check, not a sampler."""
        if self.n_vars > _MAX_ENUMERATE_VARS:
            raise ValueError(
                f"ground_state enumerates 2^n; n_vars <= {_MAX_ENUMERATE_VARS}, got {self.n_vars}"
            )
        states = all_binary_states(self.n_vars)
        return states[int(np.argmin(self.energies(states)))].copy()


@dataclass(frozen=True)
class SurrogateDiagnosis:
    """The cheapest possible de-risking of the whole project.

    If ``spearman`` is weak or ``ground_state_rank`` is near 2^p, the
    surrogate is pointing the quench at the wrong landscape. Fix scale
    (or the features) before any chain runs.
    """

    spearman: float
    ground_state: np.ndarray
    ground_state_log_prob: float
    ground_state_rank: int  # 1 = MAP under the exact target
    n_states: int
    top_log_prob: float


def problem_frobenius_norm(h: np.ndarray, J: np.ndarray) -> float:
    """√(Σ_{i<j} J_ij² + Σ_i h_i²). Same quantity ``frobenius_alpha`` inverts."""
    h_arr = np.asarray(h, dtype=np.float64).reshape(-1)
    J_arr = symmetrize_couplings(J)
    return float(np.sqrt(0.5 * np.sum(J_arr * J_arr) + h_arr @ h_arr))


def ising_surrogate_from_data(
    X: np.ndarray,
    y: np.ndarray,
    prior_inclusion: float = 0.5,
    *,
    field_coupling_ratio: float = 1.0,
) -> IsingSurrogate:
    """Analytic (h, J) from the design. No posterior evaluations.

    Parameters
    ----------
    field_coupling_ratio:
        Target RMS(h·s) / RMS(½ sᵀ J s) over random ±1 spins. 1.0
        balances the response against collinearity; larger values trust
        the marginals more. This is the knob if the scatter plot is flat.
    """
    if not (0.0 < prior_inclusion < 1.0):
        raise ValueError(f"prior_inclusion must be in (0, 1), got {prior_inclusion}")
    if field_coupling_ratio <= 0.0 or not np.isfinite(field_coupling_ratio):
        raise ValueError(
            f"field_coupling_ratio must be a positive finite value, got {field_coupling_ratio}"
        )

    X_c, y_c = center_and_scale(X, y)
    n_obs, n_vars = int(X_c.shape[0]), int(X_c.shape[1])
    # After unit-scaling, these *are* correlations (ddof=0).
    gram = (X_c.T @ X_c) / n_obs
    corr = (X_c.T @ y_c) / n_obs

    h_marginal = np.abs(corr)
    J_raw = -np.abs(gram)
    np.fill_diagonal(J_raw, 0.0)
    h_sparsity = 0.5 * float(np.log(prior_inclusion / (1.0 - prior_inclusion)))
    h_prior = np.full(n_vars, h_sparsity, dtype=np.float64)

    # Balance fields vs couplings on the *response* piece only. The
    # sparsity field is a real prior bias; using it to inflate J would
    # invent collinearity the design does not have.
    J_bal, balance_scale = _balance_couplings(h_marginal, J_raw, field_coupling_ratio)
    h = h_marginal + h_prior
    h, J, fro_scale = _pin_frobenius(h, J_bal)

    meta = {
        "prior_inclusion": float(prior_inclusion),
        "field_coupling_ratio": float(field_coupling_ratio),
        "n_obs": n_obs,
        "n_vars": n_vars,
        "balance_scale": balance_scale,
        "frobenius_scale": fro_scale,
        "frobenius_norm": problem_frobenius_norm(h, J),
        "h_sparsity": h_sparsity,
        "max_abs_corr": float(np.max(h_marginal)),
        "max_abs_collinearity": float(np.max(np.abs(J_raw))),
    }
    return IsingSurrogate(h=h, J=J, kind="analytic", meta=meta)


def learned_surrogate(
    target: Target,
    n_samples: int = 2000,
    seed: int = 0,
    ridge: float = 1e-3,
) -> IsingSurrogate:
    """Ridge-fit (h, J) so E_θ(γ) ≈ −log p(γ) + const on a sample of models.

    The 2-local Ising family is the linear span of {s_j} ∪ {s_j s_k}_{j<k}.
    At p ≤ 12 we project onto that span with a full 2^p design (the
    unique least-squares 2-local approximation). Above that we draw a
    seeded sample — the posterior is too big to enumerate, which is why
    the sampler exists.

    ``ridge`` is a diagonal jitter on θ = (h, J), not on the intercept
    (the intercept is removed by centering log p). It is not a prior on
    the spike-and-slab; it is numerical insurance when two predictors
    are interchangeable.
    """
    if n_samples < 1:
        raise ValueError(f"n_samples must be at least 1, got {n_samples}")
    if ridge < 0.0 or not np.isfinite(ridge):
        raise ValueError(f"ridge must be a non-negative finite value, got {ridge}")

    n_vars = int(target.n_vars)
    if n_vars < 1:
        raise ValueError("learned_surrogate requires at least one variable")

    states = _fit_states(n_vars, n_samples, seed)
    logp = np.asarray(target.log_prob_batch(states), dtype=np.float64)
    finite = np.isfinite(logp)
    if int(finite.sum()) < n_vars + 1:
        raise ValueError(
            "learned_surrogate: too few finite log-probabilities to fit a 2-local Hamiltonian"
        )
    states = states[finite]
    y = logp[finite]
    y = y - y.mean()

    features, pairs = _ising_features(states)
    n_coef = features.shape[1]
    gram = features.T @ features
    gram.flat[:: n_coef + 1] += float(ridge)
    rhs = features.T @ y
    try:
        theta = np.linalg.solve(gram, rhs)
    except np.linalg.LinAlgError as exc:
        raise ValueError("learned_surrogate: ridge system is singular") from exc

    h = theta[:n_vars].copy()
    J = np.zeros((n_vars, n_vars), dtype=np.float64)
    for coef, (i, j) in zip(theta[n_vars:], pairs, strict=True):
        J[i, j] = coef
        J[j, i] = coef

    # The fit recovers −E up to the affine transform we discarded.
    # Pin overall scale so analytic and learned quench kernels share α = 1;
    # do *not* re-balance fields vs couplings — that is what was learned.
    residual = y - features @ theta
    r_squared = 1.0 - float(np.sum(residual**2) / np.sum(y**2)) if np.sum(y**2) > 0.0 else 1.0
    h, J, fro_scale = _pin_frobenius(h, J)
    meta = {
        "n_samples": int(states.shape[0]),
        "n_vars": n_vars,
        "seed": int(seed),
        "ridge": float(ridge),
        "enumerated": n_vars <= 12 or states.shape[0] == (1 << n_vars),
        "fit_r_squared": r_squared,
        "frobenius_scale": fro_scale,
        "frobenius_norm": problem_frobenius_norm(h, J),
    }
    return IsingSurrogate(h=h, J=J, kind="learned", meta=meta)


def corrupt_surrogate(
    surrogate: IsingSurrogate,
    relative_sigma: float,
    seed: int,
) -> IsingSurrogate:
    """Add relative Gaussian noise to (h, J), then re-pin ||H_prob||_F.

    Isolates landscape *shape* from overall scale: after this, quench α
    is still 1. σ = 0 is a pinned clone. The E02 ablation asks where
    the gap dies as this noise grows; accept/reject must not notice.
    """
    if relative_sigma < 0.0 or not np.isfinite(relative_sigma):
        raise ValueError(
            f"relative_sigma must be a non-negative finite value, got {relative_sigma}"
        )

    h = np.asarray(surrogate.h, dtype=np.float64).copy()
    j_mat = np.asarray(surrogate.J, dtype=np.float64).copy()
    if relative_sigma > 0.0:
        rng = np.random.Generator(np.random.PCG64(seed))
        h_rms = float(np.sqrt(np.mean(h * h)))
        if h_rms <= 0.0:
            h_rms = 1.0
        j_off = j_mat.copy()
        np.fill_diagonal(j_off, 0.0)
        j_rms = float(np.sqrt(np.mean(j_off * j_off)))
        if j_rms <= 0.0:
            j_rms = 1.0
        h = h + relative_sigma * h_rms * rng.standard_normal(h.shape)
        j_mat = j_mat + relative_sigma * j_rms * rng.standard_normal(j_mat.shape)
        j_mat = 0.5 * (j_mat + j_mat.T)
        np.fill_diagonal(j_mat, 0.0)

    h, j_mat, fro_scale = _pin_frobenius(h, j_mat)
    meta = {
        **surrogate.meta,
        "corrupt_relative_sigma": float(relative_sigma),
        "corrupt_seed": int(seed),
        "corrupt_frobenius_scale": fro_scale,
        "frobenius_norm": problem_frobenius_norm(h, j_mat),
    }
    return IsingSurrogate(h=h, J=j_mat, kind=surrogate.kind, meta=meta)


def diagnose_surrogate(surrogate: IsingSurrogate, target: Target) -> SurrogateDiagnosis:
    """Ground-state rank and Spearman(−E, log p) over the full 2^p cube."""
    if surrogate.n_vars != target.n_vars:
        raise ValueError(
            f"surrogate has n_vars={surrogate.n_vars}, target has n_vars={target.n_vars}"
        )
    if target.n_vars > _MAX_ENUMERATE_VARS:
        raise ValueError(
            f"diagnose_surrogate enumerates 2^n; n_vars <= {_MAX_ENUMERATE_VARS}, "
            f"got {target.n_vars}"
        )
    states = all_binary_states(target.n_vars)
    energy = surrogate.energies(states)
    logp = np.asarray(target.log_prob_batch(states), dtype=np.float64)
    finite = np.isfinite(logp) & np.isfinite(energy)
    if int(finite.sum()) < 3:
        raise ValueError("diagnose_surrogate: fewer than 3 finite (energy, logp) pairs")

    # Low energy should be high posterior: correlate −E with log p.
    corr = spearmanr(-energy[finite], logp[finite])
    rho = float(corr.statistic)

    ground = surrogate.ground_state()
    ground_logp = float(target.log_prob(ground))
    # Rank 1 = unique MAP. Ties take the optimistic (best) rank.
    better = int(np.sum(logp > ground_logp + 1e-12))
    rank = better + 1
    finite_logp = logp[np.isfinite(logp)]
    top = float(np.max(finite_logp)) if finite_logp.size else -np.inf
    return SurrogateDiagnosis(
        spearman=rho,
        ground_state=ground,
        ground_state_log_prob=ground_logp,
        ground_state_rank=rank,
        n_states=int(states.shape[0]),
        top_log_prob=top,
    )


def write_energy_logprob_scatter(
    surrogate: IsingSurrogate,
    target: Target,
    path: str | Path,
) -> Path:
    """The one plot that de-risks E02 before any chain runs."""
    diagnosis = diagnose_surrogate(surrogate, target)
    states = all_binary_states(target.n_vars)
    energy = surrogate.energies(states)
    logp = np.asarray(target.log_prob_batch(states), dtype=np.float64)
    finite = np.isfinite(logp) & np.isfinite(energy)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(4.4, 4.0))
    ax.scatter(energy[finite], logp[finite], s=8, alpha=0.45, c="#2c3e50", linewidths=0)
    ax.set_xlabel(r"surrogate energy $E_{h,J}(\gamma)$")
    ax.set_ylabel(r"exact $\log p(\gamma \mid y)$")
    ax.set_title(
        f"{surrogate.kind} surrogate, p={target.n_vars}\n"
        f"Spearman(−E, log p) = {diagnosis.spearman:.3f}, "
        f"ground-state rank {diagnosis.ground_state_rank}/{diagnosis.n_states}"
    )
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def _balance_couplings(
    h: np.ndarray,
    J: np.ndarray,
    field_coupling_ratio: float,
) -> tuple[np.ndarray, float]:
    """Scale J so RMS(h·s) / RMS(½ sᵀ J s) = ``field_coupling_ratio``.

    For random s ∈ {±1}^p the cross terms vanish and
    E[(h·s)²] = ||h||²,  E[(½ sᵀ J s)²] = ||J||_F² / 2.
    """
    lin_rms = float(np.linalg.norm(h))
    quad_rms = float(np.linalg.norm(J) / np.sqrt(2.0))
    if lin_rms <= 0.0 or quad_rms <= 0.0:
        return J.copy(), 1.0
    target_quad = lin_rms / field_coupling_ratio
    scale = target_quad / quad_rms
    return J * scale, float(scale)


def _pin_frobenius(h: np.ndarray, J: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Pin ||H_prob||_F so the quench's α equals 1. Overall scale only."""
    n_vars = int(np.asarray(h).size)
    norm = problem_frobenius_norm(h, J)
    if norm <= 0.0:
        raise ValueError(
            "surrogate Hamiltonian is identically zero; the quench Frobenius α is undefined"
        )
    scale = float(np.sqrt(n_vars) / norm)
    return h * scale, J * scale, scale


def _fit_states(n_vars: int, n_samples: int, seed: int) -> np.ndarray:
    n_full = 1 << n_vars if n_vars <= 24 else None
    if n_vars <= 12 or (n_full is not None and n_samples >= n_full):
        return all_binary_states(n_vars)
    rng = np.random.Generator(np.random.PCG64(seed))
    return rng.integers(0, 2, size=(n_samples, n_vars), dtype=np.uint8)


def _ising_features(states: np.ndarray) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """Φ such that Φ θ = h·s + Σ_{j<k} J_jk s_j s_k = −E without the constant."""
    bits = np.asarray(states)
    if bits.ndim != 2:
        raise ValueError("states must have shape (n_states, n_vars)")
    if bits.size > 0 and not np.all((bits == 0) | (bits == 1)):
        raise ValueError("states must be binary (0/1)")
    spins = 2.0 * bits.astype(np.float64) - 1.0
    n_states, n_vars = spins.shape
    pairs = [(i, j) for i in range(n_vars) for j in range(i + 1, n_vars)]
    features = np.empty((n_states, n_vars + len(pairs)), dtype=np.float64)
    features[:, :n_vars] = spins
    for col, (i, j) in enumerate(pairs, start=n_vars):
        features[:, col] = spins[:, i] * spins[:, j]
    return features, pairs
