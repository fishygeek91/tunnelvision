"""Ising Boltzmann targets — for E01, the Layden reproduction.

E(s) = -sum_{i<j} J_ij s_i s_j - sum_i h_i s_i,  p(s) ∝ exp(-E(s)/T)

J is stored as a symmetric matrix with zero diagonal, so the quadratic
form is E = -½ sᵀ J s - h·s. Instance generators match the Layden
benchmark family: 2D lattice and fully-connected spin glasses with
J ~ N(0,1) on each undirected edge (Layden et al., Nature 619, 282 (2023)).
"""

from __future__ import annotations

import numpy as np

from tunnelvision.bits import as_binary_vector
from tunnelvision.targets.base import Target

_VALID_TOPOLOGIES = {"all-to-all", "2d"}


def symmetrize_couplings(J: np.ndarray) -> np.ndarray:
    """Canonicalize J to a symmetric zero-diagonal matrix.

    Accepts a symmetric matrix or a strictly triangular one. Averaging a
    triangular input would silently halve the couplings; a generic
    asymmetric J is almost certainly a caller bug, not a convention.
    Shared by the Boltzmann target and the quench Hamiltonian so they
    cannot drift apart.
    """
    J_arr = np.asarray(J, dtype=np.float64)
    if J_arr.ndim != 2 or J_arr.shape[0] != J_arr.shape[1]:
        raise ValueError(f"J must be a square matrix, got shape {J_arr.shape}")
    lower = np.tril(J_arr, k=-1)
    upper = np.triu(J_arr, k=1)
    if np.allclose(J_arr, J_arr.T, atol=1e-12, rtol=0.0):
        J_sym = 0.5 * (J_arr + J_arr.T)
    elif not np.any(lower) or not np.any(upper):
        J_sym = J_arr + J_arr.T
    else:
        raise ValueError("J must be symmetric or (upper/lower) triangular")
    np.fill_diagonal(J_sym, 0.0)
    return J_sym


class IsingTarget(Target):
    """Boltzmann distribution of a ±1 Ising model on binary (0/1) states.

    Chain states stay in {0,1}; they are mapped to spins via s = 2x-1
    only inside the energy. That keeps the engine kernel-agnostic.
    """

    def __init__(
        self,
        h: np.ndarray,
        J: np.ndarray,
        temperature: float = 1.0,
    ) -> None:
        h_arr = np.asarray(h, dtype=np.float64)
        J_arr = np.asarray(J, dtype=np.float64)
        if h_arr.ndim != 1:
            raise ValueError("h must be a 1-d array")
        n = int(h_arr.shape[0])
        if J_arr.shape != (n, n):
            raise ValueError(f"J must have shape {(n, n)}, got {J_arr.shape}")
        if temperature <= 0.0:
            raise ValueError(f"temperature must be positive, got {temperature}")

        J_sym = symmetrize_couplings(J_arr)

        self.h = h_arr
        self.J = J_sym
        self.temperature = float(temperature)
        self.n_vars = n

    def energy(self, x: np.ndarray) -> float:
        """Ising energy of a single binary state."""
        s = 2.0 * as_binary_vector(x, self.n_vars).astype(np.float64) - 1.0
        return float(-0.5 * (s @ self.J @ s) - self.h @ s)

    def log_prob(self, x: np.ndarray) -> float:
        return float(-self.energy(x) / self.temperature)

    def log_prob_batch(self, X: np.ndarray) -> np.ndarray:
        """Vectorized energy: n=14 is 16k states and must not Python-loop."""
        arr = np.asarray(X)
        if arr.ndim == 1:
            arr = arr[None, :]
        if arr.ndim != 2 or arr.shape[1] != self.n_vars:
            raise ValueError(f"X must have shape (n_states, {self.n_vars})")
        if arr.size > 0 and not np.all((arr == 0) | (arr == 1)):
            raise ValueError("states must be binary (0/1)")
        spins = 2.0 * arr.astype(np.float64) - 1.0
        quad = np.einsum("ij,jk,ik->i", spins, self.J, spins)
        lin = spins @ self.h
        energy = -0.5 * quad - lin
        return -energy / self.temperature


def random_spin_glass(
    n: int,
    topology: str = "all-to-all",
    seed: int = 0,
    temperature: float = 1.0,
    random_fields: bool = False,
) -> IsingTarget:
    """Seeded spin-glass instance; J ~ N(0,1) on each undirected edge.

    ``random_fields=False`` (default) keeps h = 0, matching the WP3
    generator. Layden's average-case ensemble (Nature Fig. 2) draws
    h_j ~ N(0,1) as well — the fields break inversion symmetry — so
    E01 passes ``random_fields=True``. Fields are drawn *after* the
    couplings so a given seed shares J with the h = 0 instance.

    ``topology='2d'`` requires ``n`` to be a perfect square (periodic
    square lattice). Periodic BC is the usual finite-size convention
    for these benchmarks; open BC would change the spectrum at n=8–12.
    """
    if n < 1:
        raise ValueError(f"n must be at least 1, got {n}")
    key = topology.lower()
    if key not in _VALID_TOPOLOGIES:
        raise ValueError(f"topology must be one of {sorted(_VALID_TOPOLOGIES)}, got {topology!r}")

    rng = np.random.Generator(np.random.PCG64(seed))
    J = np.zeros((n, n), dtype=np.float64)

    if key == "all-to-all":
        iu = np.triu_indices(n, k=1)
        couplings = rng.standard_normal(size=iu[0].shape[0])
        J[iu] = couplings
        J[(iu[1], iu[0])] = couplings
        h = rng.standard_normal(n) if random_fields else np.zeros(n, dtype=np.float64)
        return IsingTarget(h, J, temperature=temperature)

    side = int(round(np.sqrt(n)))
    if side * side != n:
        raise ValueError(f"2d topology requires n to be a perfect square, got {n}")

    # Undirected edges, generated once so periodic images are not doubled.
    # Sorted before drawing weights: iterating a set would tie the seeded
    # instance to CPython hash-iteration order, which is not a contract.
    edges: set[tuple[int, int]] = set()
    for row in range(side):
        for col in range(side):
            i = row * side + col
            right = row * side + (col + 1) % side
            down = ((row + 1) % side) * side + col
            for j in (right, down):
                if i == j:
                    continue
                edges.add((min(i, j), max(i, j)))
    for i, j in sorted(edges):
        weight = float(rng.standard_normal())
        J[i, j] = weight
        J[j, i] = weight
    h = rng.standard_normal(n) if random_fields else np.zeros(n, dtype=np.float64)
    return IsingTarget(h, J, temperature=temperature)
