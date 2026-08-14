"""Quantum quench proposal kernel (Layden et al., Nature 619, 282 (2023)).

Procedure per proposal:
    1. Prepare |x⟩ in the computational basis (bits.py indexing).
    2. Evolve under H(γ) = (1−γ)·α·H_prob + γ·H_mix for time t.
       α is the Frobenius ratio (Eq. 8); γ ~ U[0.25, 0.6] and
       t ~ U[2, 20] are drawn fresh — there is no parameter-optimization
       loop anywhere in this kernel.
    3. Measure in the computational basis → proposal y.

Symmetry: H is real symmetric, so U = e^{−iHt} satisfies U = Uᵀ and
|⟨y|U|x⟩|² = |⟨x|U|y⟩|². The Trotter path uses only the second-order
symmetric splitting for the same reason. log-q terms are therefore
exactly 0.0. Unital noise (depolarizing, dephasing) preserves that;
amplitude damping does not — a non-unital ``noise_model`` is an
approximation, and E03's bias audit must quantify it before any
hardware claim.

Two backends, one class: ``statevector`` (exact amplitudes, n ≤ ~14)
and ``aer`` (the same Trotter circuit, sampled). ``proposal_matrix``
always uses the statevector amplitudes averaged over a fixed (γ, t)
grid so E01–E03 do not care which backend is proposing.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np

from tunnelvision.bits import as_binary_vector, state_to_index
from tunnelvision.kernels.base import Kernel
from tunnelvision.kernels.quench import (
    Evolution,
    apply_symmetric_trotter,
    evolve_basis,
    exact_unitary,
    frobenius_alpha,
    mixing_z_eigenvalues,
    parameter_grid,
    problem_energies,
    trotter_partition,
)
from tunnelvision.targets.ising import symmetrize_couplings

Backend = Literal["statevector", "aer"]


def _unpack_index(index: int, n_vars: int) -> np.ndarray:
    bits = np.empty(n_vars, dtype=np.uint8)
    for bit in range(n_vars):
        bits[bit] = (index >> bit) & 1
    return bits


def _validate_range(name: str, bounds: tuple[float, float]) -> tuple[float, float]:
    lo, hi = float(bounds[0]), float(bounds[1])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        raise ValueError(f"{name} must be a finite non-empty interval, got {bounds}")
    return lo, hi


class QuenchKernel(Kernel):
    """Layden quench proposals, simulator-backed.

    Parameters
    ----------
    evolution:
        ``exact`` uses dense expm (Nature Fig. 2). ``trotter`` uses the
        second-order product formula with ``trotter_dt`` (the circuit
        that Aer / hardware actually run). Aer requires ``trotter``.
    """

    name = "quench"

    def __init__(
        self,
        h: np.ndarray,
        J: np.ndarray,
        gamma_range: tuple[float, float] = (0.25, 0.6),
        t_range: tuple[float, float] = (2.0, 20.0),
        trotter_dt: float = 0.8,
        backend: Backend = "statevector",
        evolution: Evolution = "trotter",
        noise_model: Any | None = None,
        shots_per_proposal: int = 1,
        n_gamma: int = 8,
        n_t: int = 8,
    ) -> None:
        h_arr = np.asarray(h, dtype=np.float64).reshape(-1)
        if h_arr.size < 1:
            raise ValueError("QuenchKernel requires at least one variable")
        J_arr = symmetrize_couplings(J)
        if J_arr.shape != (h_arr.size, h_arr.size):
            raise ValueError(f"J shape {J_arr.shape} does not match h length {h_arr.size}")
        if backend not in ("statevector", "aer"):
            raise ValueError(f"backend must be 'statevector' or 'aer', got {backend!r}")
        if evolution not in ("exact", "trotter"):
            raise ValueError(f"evolution must be 'exact' or 'trotter', got {evolution!r}")
        if backend == "aer" and evolution != "trotter":
            raise ValueError("Aer shots realize the Trotter circuit; use evolution='trotter'")
        if trotter_dt <= 0.0:
            raise ValueError(f"trotter_dt must be positive, got {trotter_dt}")
        if shots_per_proposal < 1:
            raise ValueError("shots_per_proposal must be at least 1")
        if noise_model is not None and backend != "aer":
            raise ValueError("noise_model requires backend='aer'")

        self.h = h_arr
        self.J = J_arr
        self.n_vars = int(h_arr.size)
        self.gamma_range = _validate_range("gamma_range", gamma_range)
        self.t_range = _validate_range("t_range", t_range)
        self.trotter_dt = float(trotter_dt)
        self.backend: Backend = backend
        self.evolution: Evolution = evolution
        self.noise_model = noise_model
        self.shots_per_proposal = int(shots_per_proposal)
        self.n_gamma = int(n_gamma)
        self.n_t = int(n_t)
        self.alpha = frobenius_alpha(self.h, self.J)
        self.energies = problem_energies(self.h, self.J)
        self._mix_z = mixing_z_eigenvalues(self.n_vars)
        self._proposal_cache: np.ndarray | None = None
        self._aer_sim: Any | None = None

    def propose(self, x: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, float, float]:
        x_bin = as_binary_vector(x, self.n_vars)
        gamma = float(rng.uniform(*self.gamma_range))
        t = float(rng.uniform(*self.t_range))
        if self.backend == "aer":
            y = self._propose_aer(x_bin, gamma, t, rng)
        else:
            y = self._propose_statevector(x_bin, gamma, t, rng)
        return y, 0.0, 0.0

    def proposal_matrix(self, n_vars: int) -> np.ndarray:
        """Average |⟨y|U(γ,t)|x⟩|² over the fixed (γ, t) quadrature grid.

        The average is over the *same* rectangle ``propose`` draws from.
        Cached: Q depends only on (h, J) and the grid, never on the
        target temperature — E01 reuses one matrix across T.
        """
        if n_vars != self.n_vars:
            raise ValueError(f"expected n_vars={self.n_vars}, got {n_vars}")
        if self._proposal_cache is not None:
            return self._proposal_cache

        grid = parameter_grid(self.gamma_range, self.t_range, self.n_gamma, self.n_t)
        dim = 1 << self.n_vars
        accumulated = np.zeros((dim, dim), dtype=np.float64)
        for gamma, t in grid:
            unitary = evolve_basis(
                self.n_vars,
                self.energies,
                self._mix_z,
                self.alpha,
                float(gamma),
                float(t),
                self.evolution,
                self.trotter_dt,
            )
            accumulated += np.abs(unitary) ** 2
        accumulated /= grid.shape[0]
        # Tiny imaginary-roundoff negatives would fail the engine's Q ≥ 0 check.
        np.clip(accumulated, 0.0, None, out=accumulated)
        row_sums = accumulated.sum(axis=1, keepdims=True)
        accumulated /= row_sums
        self._proposal_cache = accumulated
        return accumulated

    def _propose_statevector(
        self,
        x: np.ndarray,
        gamma: float,
        t: float,
        rng: np.random.Generator,
    ) -> np.ndarray:
        psi = np.zeros(1 << self.n_vars, dtype=np.complex128)
        psi[state_to_index(x)] = 1.0
        if self.evolution == "exact":
            psi = exact_unitary(self.energies, self.alpha, gamma, t) @ psi
        else:
            psi = apply_symmetric_trotter(
                psi,
                self.energies,
                self._mix_z,
                self.alpha,
                gamma,
                t,
                self.trotter_dt,
            )
        probs = np.abs(psi) ** 2
        total = float(probs.sum())
        if total <= 0.0 or not np.isfinite(total):
            raise RuntimeError("quench statevector has no probability mass")
        probs /= total
        y_idx = int(rng.choice(probs.size, p=probs))
        return _unpack_index(y_idx, self.n_vars)

    def _propose_aer(
        self,
        x: np.ndarray,
        gamma: float,
        t: float,
        rng: np.random.Generator,
    ) -> np.ndarray:
        # Aer is imported only here so the statevector path (and the
        # exact-side import linter) never see qiskit.
        circuit = self._trotter_circuit(x, gamma, t, measure=True)
        sim = self._simulator()
        seed = int(rng.integers(0, 2**31 - 1))
        result = sim.run(
            circuit,
            shots=self.shots_per_proposal,
            seed_simulator=seed,
        ).result()
        counts = result.get_counts()
        bitstring = max(counts, key=counts.get)
        # measure_all bitstrings are qubit n-1 … qubit 0; int(., 2) matches bits.py.
        return _unpack_index(int(bitstring.replace(" ", ""), 2), self.n_vars)

    def _trotter_circuit(
        self,
        x: np.ndarray | None,
        gamma: float,
        t: float,
        measure: bool,
    ) -> Any:
        from qiskit import QuantumCircuit

        n = self.n_vars
        circuit = QuantumCircuit(n)
        if x is not None:
            for qubit, bit in enumerate(as_binary_vector(x, n)):
                if bit:
                    circuit.x(qubit)

        n_steps, dt = trotter_partition(t, self.trotter_dt)
        # Same palindrome as apply_symmetric_trotter.
        self._append_mix(circuit, gamma * (dt / 2.0))
        for _ in range(n_steps - 1):
            self._append_problem(circuit, (1.0 - gamma) * self.alpha * dt)
            self._append_mix(circuit, gamma * dt)
        self._append_problem(circuit, (1.0 - gamma) * self.alpha * dt)
        self._append_mix(circuit, gamma * (dt / 2.0))

        if measure:
            circuit.measure_all()
        return circuit

    def _append_mix(self, circuit: Any, theta: float) -> None:
        """e^{−i θ Σ X_j} = ⊗ RX(2θ)."""
        if theta == 0.0:
            return
        angle = 2.0 * theta
        for qubit in range(self.n_vars):
            circuit.rx(angle, qubit)

    def _append_problem(self, circuit: Any, phi: float) -> None:
        """e^{−i φ H_prob} as RZ / RZZ. See module docstring for angles.

        E = −Σ_{i<j} J_ij Z_i Z_j − Σ_i h_i Z_i, so e^{−i φ E}
        = e^{i (φ J) ZZ} e^{i (φ h) Z}. RZ(λ) = e^{−i λ Z/2} ⇒
        e^{i ψ Z} = RZ(−2ψ); likewise RZZ(−2φ) for the couplings.
        """
        if phi == 0.0:
            return
        for i in range(self.n_vars):
            field = phi * self.h[i]
            if field != 0.0:
                circuit.rz(-2.0 * field, i)
        for i in range(self.n_vars):
            for j in range(i + 1, self.n_vars):
                coupling = phi * self.J[i, j]
                if coupling != 0.0:
                    circuit.rzz(-2.0 * coupling, i, j)

    def _simulator(self) -> Any:
        if self._aer_sim is None:
            from qiskit_aer import AerSimulator

            # Single-threaded: Aer’s OpenMP backend aborts in some sandboxes.
            kwargs: dict[str, Any] = {"max_parallel_threads": 1}
            if self.noise_model is not None:
                # Non-unital noise (amplitude damping) breaks Q = Qᵀ.
                # Unital models keep the symmetric-q claim; E03 audits the rest.
                kwargs["noise_model"] = self.noise_model
            self._aer_sim = AerSimulator(**kwargs)
        return self._aer_sim


class HardwareQuenchKernel(QuenchKernel):
    """IBM Runtime backend. Batches proposals: pre-samples a pool of
    (state → proposal) pairs per job to amortize queue latency.
    Credentials via environment only (see .cursor/rules) — never in code.
    """

    name = "quench-hw"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError("WP6: HardwareQuenchKernel via qiskit-ibm-runtime")
