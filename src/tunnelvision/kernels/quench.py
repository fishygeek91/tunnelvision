"""Layden quench evolution — the physics, with no Kernel interface.

H(γ) = (1−γ)·α·H_prob + γ·H_mix,  H_mix = Σ_j X_j,  α = ||H_mix||_F / ||H_prob||_F

(Layden et al., Nature 619, 282 (2023), Eqs. 5–8). Two realizations of
U = e^{−i H t} live here:

- ``exact_unitary``: dense ``expm``. This is what Nature Fig. 2 measures.
- ``apply_symmetric_trotter``: second-order (Strang) product formula,
  Δt = 0.8. First-order Trotter is *not* implemented — it breaks
  U = Uᵀ and silently invalidates the MH ratio.

Both act in the ``bits.py`` computational basis, and H_prob's diagonal
is the same Ising energy the Boltzmann target uses. A mismatch here
would make the kernel explore the wrong landscape while still looking
symmetric.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from scipy.linalg import expm

from tunnelvision.bits import all_binary_states
from tunnelvision.targets.ising import symmetrize_couplings

Evolution = Literal["exact", "trotter"]

_MAX_EXACT_VARS = 12


def frobenius_alpha(h: np.ndarray, J: np.ndarray) -> float:
    """α = √n / √(Σ_{i<j} J_ij² + Σ_i h_i²). Layden Eq. 8.

    The 2^{n/2} factors in the two Frobenius norms cancel, so this is
    O(n²) and well-defined at any n. Zero H_prob is refused: α would
    be undefined and the quench would be pure mixing.
    """
    h_arr = np.asarray(h, dtype=np.float64).reshape(-1)
    J_arr = symmetrize_couplings(J)
    if J_arr.shape != (h_arr.size, h_arr.size):
        raise ValueError(f"J shape {J_arr.shape} does not match h length {h_arr.size}")
    denom = float(np.sqrt(0.5 * np.sum(J_arr * J_arr) + h_arr @ h_arr))
    if denom <= 0.0:
        raise ValueError("H_prob is identically zero; Frobenius α is undefined")
    return float(np.sqrt(h_arr.size) / denom)


def problem_energies(h: np.ndarray, J: np.ndarray) -> np.ndarray:
    """Diagonal of H_prob: E(x) = −½ sᵀ J s − h·s with s = 2x−1.

    Vectorized over the full 2^n basis so the quench and the Ising
    target cannot disagree about which computational basis state is
    which energy. Must stay in lockstep with ``IsingTarget.energy``.
    """
    h_arr = np.asarray(h, dtype=np.float64).reshape(-1)
    J_arr = symmetrize_couplings(J)
    n = int(h_arr.size)
    if J_arr.shape != (n, n):
        raise ValueError(f"J shape {J_arr.shape} does not match h length {n}")
    spins = 2.0 * all_binary_states(n).astype(np.float64) - 1.0
    quad = np.einsum("ij,jk,ik->i", spins, J_arr, spins)
    return -0.5 * quad - spins @ h_arr


def mixing_z_eigenvalues(n_vars: int) -> np.ndarray:
    """Eigenvalues of Σ_j Z_j on the computational basis: n − 2 wt(x).

    Used to apply e^{−i θ Σ X} as H^{⊗n} e^{−i θ Σ Z} H^{⊗n}.
    """
    if n_vars < 1:
        raise ValueError(f"n_vars must be at least 1, got {n_vars}")
    return n_vars - 2.0 * all_binary_states(n_vars).sum(axis=1).astype(np.float64)


def parameter_grid(
    gamma_range: tuple[float, float],
    t_range: tuple[float, float],
    n_gamma: int,
    n_t: int,
) -> np.ndarray:
    """Equal-weight (γ, t) grid on the Layden rectangle.

    ``proposal_matrix`` averages |⟨y|U|x⟩|² over this grid — a
    deterministic quadrature of the same U[0.25, 0.6] × U[2, 20]
    measure ``propose`` draws from continuously. Midpoints of equal
    bins, not endpoints: the continuous density has no extra mass on
    the boundary.
    """
    g_lo, g_hi = gamma_range
    t_lo, t_hi = t_range
    if n_gamma < 1 or n_t < 1:
        raise ValueError("n_gamma and n_t must be at least 1")
    if g_hi <= g_lo or t_hi <= t_lo:
        raise ValueError("parameter ranges must be non-empty intervals")

    def _bin_centers(lo: float, hi: float, n: int) -> np.ndarray:
        edges = np.linspace(lo, hi, n + 1, dtype=np.float64)
        return 0.5 * (edges[:-1] + edges[1:])

    gammas = _bin_centers(g_lo, g_hi, n_gamma)
    times = _bin_centers(t_lo, t_hi, n_t)
    grid = np.stack(np.meshgrid(gammas, times, indexing="ij"), axis=-1)
    return grid.reshape(-1, 2)


def trotter_partition(t: float, dt_max: float) -> tuple[int, float]:
    """Smallest r ≥ 1 with t/r ≤ dt_max. Actual step is t/r, not dt_max.

    Using a leftover partial step of a different size would break the
    palindrome that makes the product formula symmetric.
    """
    if t <= 0.0:
        raise ValueError(f"evolution time must be positive, got {t}")
    if dt_max <= 0.0:
        raise ValueError(f"trotter_dt must be positive, got {dt_max}")
    n_steps = max(1, int(np.ceil(t / dt_max)))
    return n_steps, t / n_steps


def exact_unitary(
    energies: np.ndarray,
    alpha: float,
    gamma: float,
    t: float,
) -> np.ndarray:
    """U = exp(−i H(γ) t) by dense expm. n ≤ ~12 only."""
    dim = int(energies.shape[0])
    n_vars = int(np.log2(dim))
    if (1 << n_vars) != dim:
        raise ValueError("energies length must be a power of two")
    if n_vars > _MAX_EXACT_VARS:
        raise ValueError(f"exact_unitary supports n_vars <= {_MAX_EXACT_VARS}, got {n_vars}")

    hamiltonian = np.zeros((dim, dim), dtype=np.float64)
    np.fill_diagonal(hamiltonian, (1.0 - gamma) * alpha * energies)
    idx = np.arange(dim)
    for bit in range(n_vars):
        hamiltonian[idx, idx ^ (1 << bit)] += gamma
    return np.asarray(expm(-1.0j * hamiltonian * t), dtype=np.complex128)


def apply_symmetric_trotter(
    psi: np.ndarray,
    energies: np.ndarray,
    mix_z: np.ndarray,
    alpha: float,
    gamma: float,
    t: float,
    dt_max: float,
) -> np.ndarray:
    """In-place second-order Trotter on a state or a batch of columns.

    Splitting H = A + B with A = γ H_mix, B = (1−γ) α H_prob:

        [e^{−i A Δt/2} e^{−i B Δt} e^{−i A Δt/2}]^r

    Adjacent half-steps merge, so the applied sequence is a palindrome
    and the approximate U stays complex-symmetric (U = Uᵀ). That is
    the whole reason this formula exists in this repo.
    """
    state = np.ascontiguousarray(np.asarray(psi, dtype=np.complex128))
    if state.shape[0] != energies.shape[0]:
        raise ValueError("psi leading dimension must match energies")
    n_steps, dt = trotter_partition(t, dt_max)
    mix_half = gamma * (dt / 2.0)
    mix_full = gamma * dt
    problem_full = (1.0 - gamma) * alpha * dt

    _apply_mixing(state, mix_half, mix_z)
    for _ in range(n_steps - 1):
        _apply_problem(state, problem_full, energies)
        _apply_mixing(state, mix_full, mix_z)
    _apply_problem(state, problem_full, energies)
    _apply_mixing(state, mix_half, mix_z)
    return state


def evolve_basis(
    n_vars: int,
    energies: np.ndarray,
    mix_z: np.ndarray,
    alpha: float,
    gamma: float,
    t: float,
    evolution: Evolution,
    trotter_dt: float,
) -> np.ndarray:
    """Apply U to every computational-basis state (columns of the identity)."""
    if evolution == "exact":
        return exact_unitary(energies, alpha, gamma, t)
    dim = 1 << n_vars
    psi = np.eye(dim, dtype=np.complex128)
    return apply_symmetric_trotter(psi, energies, mix_z, alpha, gamma, t, trotter_dt)


def evolve_state(
    psi: np.ndarray,
    energies: np.ndarray,
    mix_z: np.ndarray,
    alpha: float,
    gamma: float,
    t: float,
    evolution: Evolution,
    trotter_dt: float,
) -> np.ndarray:
    """Apply U to a single statevector."""
    if evolution == "exact":
        return exact_unitary(energies, alpha, gamma, t) @ psi
    return apply_symmetric_trotter(psi, energies, mix_z, alpha, gamma, t, trotter_dt)


def _apply_hadamard(psi: np.ndarray) -> None:
    """In-place H^{⊗n} along axis 0. Bit k of the index is qubit k."""
    n_states = psi.shape[0]
    scale = 1.0 / np.sqrt(n_states)
    rest = psi.shape[1:]
    h = 1
    while h < n_states:
        view = psi.reshape(-1, 2, h, *rest)
        even = view[:, 0].copy()
        odd = view[:, 1].copy()
        view[:, 0] = even + odd
        view[:, 1] = even - odd
        h *= 2
    psi *= scale


def _apply_mixing(psi: np.ndarray, theta: float, mix_z: np.ndarray) -> None:
    """e^{−i θ Σ X_j} = H^{⊗n} e^{−i θ Σ Z_j} H^{⊗n}."""
    if theta == 0.0:
        return
    _apply_hadamard(psi)
    phase = np.exp(-1.0j * theta * mix_z)
    psi *= phase.reshape(-1, *([1] * (psi.ndim - 1)))
    _apply_hadamard(psi)


def _apply_problem(psi: np.ndarray, phi: float, energies: np.ndarray) -> None:
    """e^{−i φ H_prob}, H_prob diagonal."""
    if phi == 0.0:
        return
    phase = np.exp(-1.0j * phi * energies)
    psi *= phase.reshape(-1, *([1] * (psi.ndim - 1)))
