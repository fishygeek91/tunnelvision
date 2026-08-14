"""Quench kernel: symmetry, bit-ordering, exactness, Aer agreement.

qiskit stays in this file and in kernels/quantum.py — never in
test_exactness.py. A failure of Q ≈ Q.T is a Trotter-ordering bug,
not a statistics bug; stop and fix the product formula.
"""

from __future__ import annotations

import numpy as np
import pytest

from tunnelvision.bits import all_binary_states, state_to_index, states_to_indices
from tunnelvision.engine import MetropolisEngine
from tunnelvision.kernels.quantum import HardwareQuenchKernel, QuenchKernel
from tunnelvision.kernels.quench import (
    exact_unitary,
    frobenius_alpha,
    parameter_grid,
    problem_energies,
)
from tunnelvision.targets.ising import IsingTarget, random_spin_glass


def _tv_distance(p: np.ndarray, q: np.ndarray) -> float:
    return 0.5 * float(np.abs(p - q).sum())


def _empirical_distribution(states: np.ndarray, n_vars: int) -> np.ndarray:
    counts = np.bincount(states_to_indices(states), minlength=1 << n_vars)
    return counts.astype(np.float64) / counts.sum()


def _small_glass(n: int = 4, seed: int = 0) -> IsingTarget:
    return random_spin_glass(n, topology="all-to-all", seed=seed, temperature=1.0)


def _quench(target: IsingTarget, **kwargs: object) -> QuenchKernel:
    defaults: dict[str, object] = {
        "n_gamma": 4,
        "n_t": 4,
        "evolution": "trotter",
        "backend": "statevector",
    }
    defaults.update(kwargs)
    return QuenchKernel(target.h, target.J, **defaults)  # type: ignore[arg-type]


def test_frobenius_alpha_matches_layden_eq8() -> None:
    h = np.array([0.5, -1.0, 0.25])
    J = np.array(
        [
            [0.0, 0.3, -0.2],
            [0.3, 0.0, 0.4],
            [-0.2, 0.4, 0.0],
        ]
    )
    expected = np.sqrt(3.0) / np.sqrt(0.3**2 + 0.2**2 + 0.4**2 + 0.5**2 + 1.0 + 0.25**2)
    np.testing.assert_allclose(frobenius_alpha(h, J), expected, atol=1e-12)


def test_problem_energies_match_ising_target() -> None:
    target = _small_glass(5, seed=3)
    energies = problem_energies(target.h, target.J)
    states = all_binary_states(target.n_vars)
    expected = np.array([target.energy(row) for row in states])
    np.testing.assert_allclose(energies, expected, atol=1e-12, rtol=0.0)


def test_zero_hamiltonian_alpha_is_undefined() -> None:
    with pytest.raises(ValueError, match="identically zero"):
        frobenius_alpha(np.zeros(3), np.zeros((3, 3)))


def test_parameter_grid_is_bin_centers() -> None:
    grid = parameter_grid((0.0, 1.0), (0.0, 2.0), n_gamma=2, n_t=2)
    gammas = set(np.round(grid[:, 0], 12))
    times = set(np.round(grid[:, 1], 12))
    assert gammas == {0.25, 0.75}
    assert times == {0.5, 1.5}
    assert grid.shape == (4, 2)


def test_bit_ordering_pure_mixing() -> None:
    """γ = 1 ⇒ U = ⊗ e^{−i t X}. Flip amplitudes must land on bit k = 2^k."""
    n_vars = 2
    h = np.ones(n_vars)
    J = np.zeros((n_vars, n_vars))
    # Non-zero H_prob so α is defined; γ=1 turns it off.
    t = 0.4
    energies = problem_energies(h, J)
    alpha = frobenius_alpha(h, J)
    unitary = exact_unitary(energies, alpha, gamma=1.0, t=t)
    # e^{−itX}|0⟩ = cos(t)|0⟩ − i sin(t)|1⟩, so
    # |⟨e_k|U|0⟩|² = sin²(t) cos²(t) for a single-bit flip.
    single = (np.sin(t) * np.cos(t)) ** 2
    both = np.sin(t) ** 4
    stay = np.cos(t) ** 4
    np.testing.assert_allclose(np.abs(unitary[0, 0]) ** 2, stay, atol=1e-12)
    np.testing.assert_allclose(np.abs(unitary[1, 0]) ** 2, single, atol=1e-12)
    np.testing.assert_allclose(np.abs(unitary[2, 0]) ** 2, single, atol=1e-12)
    np.testing.assert_allclose(np.abs(unitary[3, 0]) ** 2, both, atol=1e-12)


def test_single_qubit_flip_index_is_bit_zero() -> None:
    """|1⟩ is index 1, not index 2^{n-1}. The classic endianness trap."""
    h = np.array([1.0])
    J = np.zeros((1, 1))
    t = 0.3
    U = exact_unitary(problem_energies(h, J), frobenius_alpha(h, J), gamma=1.0, t=t)
    # Pure mixing on one qubit: |⟨1|e^{−itX}|0⟩|² = sin²(t).
    np.testing.assert_allclose(np.abs(U[1, 0]) ** 2, np.sin(t) ** 2, atol=1e-12)
    assert state_to_index(np.array([1], dtype=np.uint8)) == 1


@pytest.mark.parametrize("evolution", ["exact", "trotter"])
def test_proposal_matrix_is_symmetric_and_stochastic(evolution: str) -> None:
    kernel = _quench(_small_glass(4, seed=1), evolution=evolution)
    Q = kernel.proposal_matrix(4)
    np.testing.assert_allclose(Q.sum(axis=1), 1.0, atol=1e-12, rtol=0.0)
    np.testing.assert_allclose(Q, Q.T, atol=1e-10, rtol=0.0)
    assert np.all(Q >= -1e-15)


def test_trotter_unitary_is_symmetric() -> None:
    """The product-formula trap: first-order Trotter would fail this."""
    target = _small_glass(4, seed=2)
    kernel = _quench(target, evolution="trotter")
    from tunnelvision.kernels.quench import evolve_basis

    U = evolve_basis(
        4,
        kernel.energies,
        kernel._mix_z,
        kernel.alpha,
        gamma=0.4,
        t=3.2,
        evolution="trotter",
        trotter_dt=0.8,
    )
    np.testing.assert_allclose(U, U.T, atol=1e-12, rtol=0.0)
    np.testing.assert_allclose(U.conj().T @ U, np.eye(16), atol=1e-12, rtol=0.0)


def test_trotter_tracks_exact_at_moderate_time() -> None:
    target = _small_glass(3, seed=4)
    kernel = _quench(target)
    from tunnelvision.kernels.quench import evolve_basis

    kwargs = {
        "n_vars": 3,
        "energies": kernel.energies,
        "mix_z": kernel._mix_z,
        "alpha": kernel.alpha,
        "gamma": 0.4,
        "t": 2.4,
        "trotter_dt": 0.4,
    }
    exact = evolve_basis(evolution="exact", **kwargs)
    trotter = evolve_basis(evolution="trotter", **kwargs)
    # Same (γ, t); a well-resolved second-order step should be close.
    assert float(np.max(np.abs(np.abs(exact) ** 2 - np.abs(trotter) ** 2))) < 0.05


@pytest.mark.parametrize("evolution", ["exact", "trotter"])
def test_quench_stationarity_and_detailed_balance(evolution: str) -> None:
    target = _small_glass(5, seed=0)
    kernel = _quench(target, evolution=evolution, n_gamma=3, n_t=3)
    P = MetropolisEngine(target, kernel).transition_matrix()
    pi = target.enumerate_exact()
    np.testing.assert_allclose(pi @ P, pi, atol=1e-12, rtol=0.0)
    balance = pi[:, None] * P
    np.testing.assert_allclose(balance, balance.T, atol=1e-12, rtol=0.0)


def test_quench_sampled_distribution_matches_enumeration() -> None:
    target = _small_glass(4, seed=7)
    kernel = _quench(target, evolution="exact", n_gamma=3, n_t=3)
    engine = MetropolisEngine(target, kernel)
    result = engine.run(n_steps=30_000, x0=np.zeros(target.n_vars, dtype=np.uint8), seed=5)
    empirical = _empirical_distribution(result.states[4_000:], target.n_vars)
    tv = _tv_distance(empirical, target.enumerate_exact())
    assert tv < 0.12, f"quench TV distance {tv:.4f} exceeded tolerance"


def test_qiskit_trotter_circuit_matches_numpy() -> None:
    """Aer/Qiskit must implement the same palindrome as apply_symmetric_trotter."""
    from qiskit.quantum_info import Operator

    from tunnelvision.kernels.quench import evolve_basis

    target = _small_glass(3, seed=1)
    kernel = _quench(target, evolution="trotter")
    gamma, t = 0.4, 2.4
    numpy_U = evolve_basis(
        3,
        kernel.energies,
        kernel._mix_z,
        kernel.alpha,
        gamma,
        t,
        evolution="trotter",
        trotter_dt=0.8,
    )
    circuit = kernel._trotter_circuit(x=None, gamma=gamma, t=t, measure=False)
    qiskit_U = np.asarray(Operator(circuit).data)
    np.testing.assert_allclose(qiskit_U, numpy_U, atol=1e-10, rtol=0.0)


def test_aer_shots_match_statevector_histogram() -> None:
    target = _small_glass(3, seed=2)
    kernel = _quench(target, backend="aer", evolution="trotter")
    x = np.array([1, 0, 1], dtype=np.uint8)
    gamma, t = 0.35, 3.2
    from tunnelvision.kernels.quench import evolve_state

    psi = np.zeros(8, dtype=np.complex128)
    psi[state_to_index(x)] = 1.0
    psi = evolve_state(
        psi,
        kernel.energies,
        kernel._mix_z,
        kernel.alpha,
        gamma,
        t,
        evolution="trotter",
        trotter_dt=0.8,
    )
    exact = np.abs(psi) ** 2

    circuit = kernel._trotter_circuit(x, gamma, t, measure=True)
    result = kernel._simulator().run(circuit, shots=4000, seed_simulator=0).result()
    counts = result.get_counts()
    empirical = np.zeros(8, dtype=np.float64)
    for bitstring, n_hits in counts.items():
        empirical[int(bitstring.replace(" ", ""), 2)] = n_hits
    empirical /= empirical.sum()
    assert _tv_distance(empirical, exact) < 0.08


def test_aer_propose_returns_binary_vector() -> None:
    target = _small_glass(3, seed=0)
    kernel = _quench(target, backend="aer", evolution="trotter")
    rng = np.random.Generator(np.random.PCG64(0))
    y, log_fwd, log_rev = kernel.propose(np.zeros(3, dtype=np.uint8), rng)
    assert y.shape == (3,)
    assert set(y.tolist()) <= {0, 1}
    assert log_fwd == 0.0 and log_rev == 0.0


def test_hardware_kernel_is_still_a_stub() -> None:
    with pytest.raises(NotImplementedError, match="WP6"):
        HardwareQuenchKernel(np.ones(2), np.zeros((2, 2)))


def test_proposal_matrix_is_cached_across_calls() -> None:
    kernel = _quench(_small_glass(3, seed=0), evolution="exact", n_gamma=2, n_t=2)
    first = kernel.proposal_matrix(3)
    second = kernel.proposal_matrix(3)
    assert first is second
