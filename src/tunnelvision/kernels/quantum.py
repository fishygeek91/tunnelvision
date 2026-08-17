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

Three backends, one circuit: ``statevector`` (exact amplitudes, n ≤ ~14),
``aer`` (the same Trotter circuit, sampled), and ``hardware`` (batched
pools via an injected sampler or IBM Runtime). ``proposal_matrix``
always uses the statevector amplitudes averaged over a fixed (γ, t)
grid so E01–E03 do not care which backend is proposing.
"""

from __future__ import annotations

from collections.abc import Sequence
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

Backend = Literal["statevector", "aer", "hardware"]


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


def _validate_unit_interval(name: str, value: float) -> float:
    number = float(value)
    if not np.isfinite(number) or number < 0.0 or number > 1.0:
        raise ValueError(f"{name} must be in [0, 1], got {value}")
    return number


def depolarize_proposal(proposal: np.ndarray, depolarize: float) -> np.ndarray:
    """Global depolarizing channel on a proposal matrix.

    Q_lambda = (1-lambda) Q + (lambda / 2^n) 11^T. The channel is unital
    and commutes with the (gamma, t) average, so this is the exact-tier
    noise model at p=10: cheap, symmetric, and row-stochastic whenever Q is.
    """
    q = np.asarray(proposal, dtype=np.float64)
    if q.ndim != 2 or q.shape[0] != q.shape[1]:
        raise ValueError("proposal matrix must be square")
    lam = _validate_unit_interval("depolarize", depolarize)
    dim = int(q.shape[0])
    if dim < 1:
        raise ValueError("proposal matrix must not be empty")
    mixed = (1.0 - lam) * q + lam / dim
    np.clip(mixed, 0.0, None, out=mixed)
    row_sums = mixed.sum(axis=1, keepdims=True)
    if np.any(row_sums <= 0.0):
        raise ValueError("depolarized proposal has a zero row")
    return mixed / row_sums


def fit_global_depolarize(reference: np.ndarray, observed: np.ndarray) -> float:
    """Least-squares λ for ``observed ≈ (1−λ) reference + λ/2^n 11ᵀ``.

    The global channel is a one-parameter family, so this is the exact
    Frobenius projection onto it. Used to map Aer per-gate error onto
    the analytic λ the p=10 sweep actually runs.
    """
    ref = np.asarray(reference, dtype=np.float64)
    obs = np.asarray(observed, dtype=np.float64)
    if ref.shape != obs.shape or ref.ndim != 2 or ref.shape[0] != ref.shape[1]:
        raise ValueError("reference and observed must be square matrices of the same shape")
    dim = int(ref.shape[0])
    if dim < 1:
        raise ValueError("proposal matrix must not be empty")
    direction = np.full((dim, dim), 1.0 / dim, dtype=np.float64) - ref
    residual = obs - ref
    denom = float(np.sum(direction * direction))
    if denom <= 0.0:
        return 0.0
    fitted = float(np.sum(direction * residual) / denom)
    return float(np.clip(fitted, 0.0, 1.0))


def depolarizing_noise_model(gate_error: float) -> Any:
    """Aer ``NoiseModel``: depolarizing after every RX / RZ / RZZ.

    Same ``gate_error`` on 1q and 2q gates — the 2q channel is still
    the harsher one (it replaces a two-qubit state with the maximally
    mixed state). Unital, so the symmetric-q claim survives. Amplitude
    damping is a different, non-unital story; that is the hardware audit.
    """
    from qiskit_aer.noise import NoiseModel, depolarizing_error

    p_err = _validate_unit_interval("gate_error", gate_error)
    model = NoiseModel()
    if p_err == 0.0:
        return model
    model.add_all_qubit_quantum_error(depolarizing_error(p_err, 1), ["rx", "rz"])
    model.add_all_qubit_quantum_error(depolarizing_error(p_err, 2), ["rzz"])
    return model


def amplitude_damping_noise_model(gamma: float) -> Any:
    """Aer ``NoiseModel``: amplitude damping after every RX / RZ / RZZ.

    Depolarizing is unital and keeps ``Q ≈ Q.T``. This channel is not —
    it exists only so the E03 bias audit can bound the symmetric-q
    approximation before any hardware claim. The Hastings ratio still
    treats proposals as symmetric; the audit, not accept/reject, is
    where the residual bias is allowed to show up.
    """
    from qiskit_aer.noise import NoiseModel, amplitude_damping_error

    damp = _validate_unit_interval("gamma", gamma)
    model = NoiseModel()
    if damp == 0.0:
        return model
    one_qubit = amplitude_damping_error(damp)
    two_qubit = one_qubit.tensor(one_qubit)
    model.add_all_qubit_quantum_error(one_qubit, ["rx", "rz"])
    model.add_all_qubit_quantum_error(two_qubit, ["rzz"])
    return model


def aer_noisy_proposal_matrix(
    kernel: QuenchKernel,
    gate_error: float,
    *,
    n_gamma: int | None = None,
    n_t: int | None = None,
) -> np.ndarray:
    """Density-matrix ``Q`` for the Trotter circuit under per-gate depolarizing.

    Rows are computational-basis inputs; each is the diagonal of the
    noisy density matrix, averaged over the same (γ, t) quadrature
    ``proposal_matrix`` uses. p ≤ 7 and a reduced grid only — a full
    Aer ``Q`` at p=10 is a wall-clock decision, not a correctness one.

    Per-gate noise is unital but only approximately symmetric (noise
    sits after each gate, not in a palindrome of its own). The
    exact-tier sweep uses ``depolarize_proposal``, which is exactly
    ``Q = Q.T``. This function exists to show that model is a fair
    proxy, not to replace it.
    """
    if kernel.evolution != "trotter":
        raise ValueError("aer_noisy_proposal_matrix realizes the Trotter circuit")
    n_g = int(kernel.n_gamma if n_gamma is None else n_gamma)
    n_times = int(kernel.n_t if n_t is None else n_t)
    if n_g < 1 or n_times < 1:
        raise ValueError("n_gamma and n_t must be at least 1")

    from qiskit_aer import AerSimulator

    grid = parameter_grid(kernel.gamma_range, kernel.t_range, n_g, n_times)
    dim = 1 << kernel.n_vars
    noise_model = depolarizing_noise_model(gate_error)
    sim_kwargs: dict[str, Any] = {
        "method": "density_matrix",
        "max_parallel_threads": 1,
    }
    if gate_error > 0.0:
        sim_kwargs["noise_model"] = noise_model
    sim = AerSimulator(**sim_kwargs)

    circuits = []
    for gamma, t in grid:
        for x_idx in range(dim):
            circuit = kernel._trotter_circuit(
                _unpack_index(x_idx, kernel.n_vars),
                float(gamma),
                float(t),
                measure=False,
            )
            circuit.save_density_matrix()
            circuits.append(circuit)

    accumulated = np.zeros((dim, dim), dtype=np.float64)
    batch_size = 32
    for start in range(0, len(circuits), batch_size):
        chunk = circuits[start : start + batch_size]
        result = sim.run(chunk, shots=1).result()
        for offset, _circuit in enumerate(chunk):
            data = result.data(offset)
            rho = np.asarray(data["density_matrix"], dtype=np.complex128)
            probs = np.real(np.diag(rho))
            total = float(probs.sum())
            if total <= 0.0 or not np.isfinite(total):
                raise RuntimeError("noisy density matrix has no probability mass")
            accumulated[(start + offset) % dim] += probs / total

    accumulated /= grid.shape[0]
    np.clip(accumulated, 0.0, None, out=accumulated)
    row_sums = accumulated.sum(axis=1, keepdims=True)
    if np.any(row_sums <= 0.0):
        raise RuntimeError("aer noisy proposal has a zero row")
    return accumulated / row_sums


class QuenchKernel(Kernel):
    """Layden quench proposals, simulator-backed.

    Parameters
    ----------
    evolution:
        ``exact`` uses dense expm (Nature Fig. 2). ``trotter`` uses the
        second-order product formula with ``trotter_dt`` (the circuit
        that Aer / hardware actually run). Aer and hardware require
        ``trotter``.
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
        if backend not in ("statevector", "aer", "hardware"):
            raise ValueError(
                f"backend must be 'statevector', 'aer', or 'hardware', got {backend!r}"
            )
        if evolution not in ("exact", "trotter"):
            raise ValueError(f"evolution must be 'exact' or 'trotter', got {evolution!r}")
        if backend in ("aer", "hardware") and evolution != "trotter":
            raise ValueError(
                "Aer/hardware shots realize the Trotter circuit; use evolution='trotter'"
            )
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
        elif self.backend == "hardware":
            raise RuntimeError("use HardwareQuenchKernel.propose for the hardware backend")
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
        """e^{−i φ H_prob} as RZ / RZZ.

        Numpy energies use s = 2x−1, so computational |0⟩ is s = −1.
        Qiskit has Z|0⟩ = +|0⟩, therefore s = −Z and s_i s_j = Z_i Z_j.
        The field term flips; the couplings do not:

            E = −Σ_{i<j} J_ij Z_i Z_j + Σ_i h_i Z_i

        Then e^{−i φ E} = e^{i (φ J) ZZ} e^{−i (φ h) Z}. RZ(λ) = e^{−i λ Z/2}
        ⇒ e^{−i ψ Z} = RZ(2ψ); RZZ(−2 φ J) for the couplings. Tests with
        h = 0 cannot catch a field-sign error — the surrogate (and any
        Layden instance with random_fields=True) can.
        """
        if phi == 0.0:
            return
        for i in range(self.n_vars):
            field = phi * self.h[i]
            if field != 0.0:
                circuit.rz(2.0 * field, i)
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


class DepolarizedQuenchKernel(Kernel):
    """Quench proposals mixed with a global depolarizing channel.

    With probability ``depolarize`` the proposal is uniform on {0,1}^n;
    otherwise it is a fresh quench draw. That mixture is exactly
    ``depolarize_proposal(Q, λ)``, so ``propose`` and ``proposal_matrix``
    agree with the same (γ, t) averaging convention the inner kernel
    already uses. Symmetric (unital) ⇒ log-q is 0.0.

    Share one inner ``QuenchKernel`` across λ values — ``Q`` is cached
    on the inner object and the channel is a rank-1 update.
    """

    name = "quench-depolarized"

    def __init__(self, inner: QuenchKernel, depolarize: float) -> None:
        if not isinstance(inner, QuenchKernel):
            raise TypeError("DepolarizedQuenchKernel wraps a QuenchKernel")
        self.inner = inner
        self.depolarize = _validate_unit_interval("depolarize", depolarize)
        self.n_vars = inner.n_vars
        self.name = f"{inner.name}-depol-{self.depolarize:g}"
        self._proposal_cache: np.ndarray | None = None

    def propose(self, x: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, float, float]:
        if self.depolarize >= 1.0 or (self.depolarize > 0.0 and rng.random() < self.depolarize):
            y_idx = int(rng.integers(0, 1 << self.n_vars))
            return _unpack_index(y_idx, self.n_vars), 0.0, 0.0
        return self.inner.propose(x, rng)

    def proposal_matrix(self, n_vars: int) -> np.ndarray:
        if n_vars != self.n_vars:
            raise ValueError(f"expected n_vars={self.n_vars}, got {n_vars}")
        if self._proposal_cache is None:
            self._proposal_cache = depolarize_proposal(
                self.inner.proposal_matrix(n_vars), self.depolarize
            )
        return self._proposal_cache


class AerCircuitSampler:
    """One Aer job per ``run`` call — the session path for hardware pools.

    Matches the ``run(circuits) -> bitstrings`` contract
    ``HardwareQuenchKernel`` uses. IBM Runtime is the same contract
    behind ``backend_name``; tests never import it.
    """

    def __init__(self, noise_model: Any | None = None, shots: int = 1) -> None:
        if shots < 1:
            raise ValueError(f"shots must be at least 1, got {shots}")
        from qiskit_aer import AerSimulator

        kwargs: dict[str, Any] = {"max_parallel_threads": 1}
        if noise_model is not None:
            kwargs["noise_model"] = noise_model
        self._sim = AerSimulator(**kwargs)
        self.shots = int(shots)
        self.n_jobs = 0

    def run(self, circuits: Sequence[Any]) -> list[str]:
        if not circuits:
            raise ValueError("sampler.run requires at least one circuit")
        self.n_jobs += 1
        result = self._sim.run(list(circuits), shots=self.shots).result()
        bitstrings: list[str] = []
        for index in range(len(circuits)):
            counts = result.get_counts(index)
            if not counts:
                raise RuntimeError("Aer job returned empty counts")
            bitstrings.append(max(counts, key=counts.get))
        return bitstrings


def _runtime_sampler(backend_name: str) -> Any:
    """Lazy IBM Runtime primitive wrapped to the injected-sampler contract.

    Transpile-to-backend ISA is a live-hardware concern (next session).
    This exists so ``backend_name`` is wired; tests inject a sampler.
    Credentials come from the Runtime account file / env, never from
    configs or notebooks.
    """
    try:
        from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
    except ImportError as exc:
        raise ImportError(
            "HardwareQuenchKernel(backend_name=...) requires the 'hardware' extra "
            "(uv sync --extra hardware). Inject a sampler for simulator tests."
        ) from exc

    try:
        service = QiskitRuntimeService()
        backend = service.backend(backend_name)
        primitive = SamplerV2(mode=backend)
    except Exception as exc:
        raise RuntimeError(
            "No resolvable IBM Quantum account for backend "
            f"{backend_name!r}. Use IBM_QUANTUM_TOKEN or qiskit-ibm.json "
            "(never configs or notebooks), or inject a sampler."
        ) from exc
    return _RuntimeSamplerAdapter(primitive)


class _RuntimeSamplerAdapter:
    """Map SamplerV2 results onto ``run(circuits) -> bitstrings``."""

    def __init__(self, primitive: Any) -> None:
        self._primitive = primitive

    def run(self, circuits: Sequence[Any]) -> list[str]:
        job = self._primitive.run(list(circuits))
        result = job.result()
        bitstrings: list[str] = []
        for index, _circuit in enumerate(circuits):
            data = result[index].data
            meas = getattr(data, "meas", None)
            if meas is not None and hasattr(meas, "get_counts"):
                counts = meas.get_counts()
            elif hasattr(data, "get_counts"):
                counts = data.get_counts()
            else:
                raise RuntimeError("Runtime result has no counts; cannot decode proposals")
            if not counts:
                raise RuntimeError("Runtime job returned empty counts")
            bitstrings.append(max(counts, key=counts.get))
        return bitstrings


class HardwareQuenchKernel(QuenchKernel):
    """Batched quench proposals for a QPU-shaped sampler.

    Queue latency dominates wall clock, so a cache miss submits
    ``pool_size`` Trotter circuits from the current state in one job
    and later visits pop from that pool. States revisit constantly at
    p=10; the hit-rate counters are how we know the batching paid off.

    ``propose`` returns ``(y, 0.0, 0.0)``. That is the symmetric-q
    *approximation*: time-symmetric Trotter plus unital noise would
    make it exact; amplitude damping (and real hardware) is non-unital.
    The E03 bias audit bounds the residual. Accept/reject still uses
    the exact target — only the proposal symmetry is approximate.

    Credentials via environment / ``qiskit-ibm.json`` only. The session
    path injects a sampler (usually ``AerCircuitSampler``) so default
    ``uv sync`` never imports ``qiskit-ibm-runtime``.
    """

    name = "quench-hw"

    def __init__(
        self,
        h: np.ndarray,
        J: np.ndarray,
        gamma_range: tuple[float, float] = (0.25, 0.6),
        t_range: tuple[float, float] = (2.0, 20.0),
        trotter_dt: float = 0.8,
        shots_per_proposal: int = 1,
        n_gamma: int = 8,
        n_t: int = 8,
        *,
        sampler: Any | None = None,
        backend_name: str | None = None,
        pool_size: int = 16,
    ) -> None:
        if pool_size < 1:
            raise ValueError(f"pool_size must be at least 1, got {pool_size}")
        if sampler is None and backend_name is None:
            raise ValueError(
                "HardwareQuenchKernel requires an injected sampler or backend_name. "
                "Credentials via env / qiskit-ibm.json only — never configs."
            )
        resolved = sampler if sampler is not None else _runtime_sampler(str(backend_name))
        super().__init__(
            h,
            J,
            gamma_range=gamma_range,
            t_range=t_range,
            trotter_dt=trotter_dt,
            backend="hardware",
            evolution="trotter",
            shots_per_proposal=shots_per_proposal,
            n_gamma=n_gamma,
            n_t=n_t,
        )
        self.sampler = resolved
        self.backend_name = backend_name
        self.pool_size = int(pool_size)
        self._pools: dict[int, list[np.ndarray]] = {}
        self.cache_hits = 0
        self.cache_misses = 0
        self.jobs_submitted = 0

    def propose(self, x: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, float, float]:
        x_bin = as_binary_vector(x, self.n_vars)
        key = state_to_index(x_bin)
        pool = self._pools.get(key)
        if not pool:
            self.cache_misses += 1
            self._fill_pool(x_bin, key, rng)
            pool = self._pools[key]
        else:
            self.cache_hits += 1
        # Symmetric-q approximation: hardware / amplitude damping is
        # non-unital, so q(y|x) = q(x|y) is not exact. The bias audit
        # measures the residual; the Hastings ratio does not try to.
        return pool.pop(), 0.0, 0.0

    def _fill_pool(self, x: np.ndarray, key: int, rng: np.random.Generator) -> None:
        circuits = []
        for _ in range(self.pool_size):
            gamma = float(rng.uniform(*self.gamma_range))
            t = float(rng.uniform(*self.t_range))
            circuits.append(self._trotter_circuit(x, gamma, t, measure=True))
        bitstrings = self.sampler.run(circuits)
        if len(bitstrings) != self.pool_size:
            raise RuntimeError(
                f"sampler returned {len(bitstrings)} bitstrings, expected {self.pool_size}"
            )
        proposals = [_bitstring_to_state(raw, self.n_vars) for raw in bitstrings]
        self._pools[key] = proposals
        self.jobs_submitted += 1


def _bitstring_to_state(bitstring: str, n_vars: int) -> np.ndarray:
    cleaned = str(bitstring).replace(" ", "")
    if len(cleaned) != n_vars:
        raise RuntimeError(f"expected {n_vars}-bit string, got {bitstring!r}")
    return _unpack_index(int(cleaned, 2), n_vars)
