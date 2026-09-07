"""Exactness tests for the auxiliary-variable dynamics (D01)."""

import numpy as np
import pytest

from tunnelvision.bits import all_binary_states
from tunnelvision.diagnostics import spectral_gap
from tunnelvision.dynamics import DemonKernel
from tunnelvision.dynamics.mtm import ShellMTM
from tunnelvision.engine import MetropolisEngine
from tunnelvision.kernels.classical import SingleFlip
from tunnelvision.targets.ising import random_spin_glass


@pytest.mark.parametrize("Td", [1.0, 4.0])
@pytest.mark.parametrize("L", [1, 3, 6, 9])
def test_demon_involution(Td, L):
    tgt = random_spin_glass(6, seed=3, temperature=0.5, random_fields=True)
    k = DemonKernel(tgt, sweep_len=L, demon_temperature=Td)
    rng = np.random.default_rng(0)
    for _ in range(100):
        x = rng.integers(0, 2, 6).astype(np.uint8)
        d = rng.exponential()
        order = k._draw_order(rng)
        y, d1, _ = k.sweep(x, d, order)
        x2, d2, _ = k.sweep(y, d1, order[::-1])
        assert np.array_equal(x, x2)
        assert abs(d - d2) < 1e-12


@pytest.mark.parametrize("Td", [1.0, 3.0])
def test_demon_stationary_and_rejection_free_at_Td1(Td):
    tgt = random_spin_glass(6, seed=1, temperature=0.4, random_fields=True)
    pi = tgt.enumerate_exact()
    k = DemonKernel(tgt, demon_temperature=Td)
    P = MetropolisEngine(tgt, k).transition_matrix()
    spectral_gap(P, pi)  # raises if pi P != pi
    Q = k.proposal_matrix(6)
    assert np.allclose(Q.sum(1), 1.0)
    if Td == 1.0:
        assert np.allclose(P, Q)  # acceptance identically 1


def test_demon_sampled_matches_exact_energy_prob():
    """The propose() path and the exact proposal_matrix agree."""
    tgt = random_spin_glass(5, seed=2, temperature=0.7, random_fields=True)
    k = DemonKernel(tgt, sweep_len=5, demon_temperature=2.0)
    Q = k.proposal_matrix(5)
    rng = np.random.default_rng(1)
    x = np.array([1, 0, 1, 1, 0], dtype=np.uint8)
    xi = int((x * (1 << np.arange(5))).sum())
    counts = np.zeros(32)
    N = 40000
    for _ in range(N):
        y, _, _ = k.propose(x, rng)
        counts[int((y * (1 << np.arange(5))).sum())] += 1
    assert np.max(np.abs(counts / N - Q[xi])) < 0.01


def test_shell_mtm_exact_moments():
    tgt = random_spin_glass(6, seed=4, temperature=0.5, random_fields=True)
    S = all_binary_states(6)
    pi = tgt.enumerate_exact()
    m_exact = pi @ S.astype(float)
    for a, eps in [(0.0, np.inf), (1.0, 4.0), (0.5, 2.0)]:
        s = ShellMTM(tgt, n_try=6, eps=eps, a=a)
        st, _, _ = s.run(60000, np.zeros(6, dtype=np.uint8), seed=7)
        m_hat = st[5000:].mean(0)
        assert np.max(np.abs(m_hat - m_exact)) < 0.03, (a, eps, m_hat, m_exact)


def test_dlp_logq_matches_matrix_and_stationary():
    from tunnelvision.dynamics.dlp import DLPKernel

    tgt = random_spin_glass(6, seed=1, temperature=0.3, random_fields=True)
    k = DLPKernel(tgt, alpha=1.0)
    Q = k.proposal_matrix(6)
    assert np.allclose(Q.sum(1), 1.0)
    rng = np.random.default_rng(0)
    pw = 1 << np.arange(6)
    for _ in range(50):
        x = rng.integers(0, 2, 6).astype(np.uint8)
        y, lf, lr = k.propose(x, rng)
        assert abs(np.log(Q[x @ pw, y @ pw]) - lf) < 1e-9
        assert abs(np.log(Q[y @ pw, x @ pw]) - lr) < 1e-9
    spectral_gap(MetropolisEngine(tgt, k).transition_matrix(), tgt.enumerate_exact())


# ----------------------------------------------------------------- M02: ICM


def test_icm_faithful_is_swap_on_all_to_all():
    """theta=0 on all-to-all: cluster = whole disagreement set, move = swap.

    This is the percolation degeneracy the paper's Sec. VII.A paragraph
    asserts; certify it rather than assume it. Pair energy change is
    exactly zero (fields cancel across replicas on D).
    """
    from tunnelvision.dynamics.icm import HoudayerICM

    tgt = random_spin_glass(8, seed=1, temperature=0.3, random_fields=True)
    icm = HoudayerICM(tgt)
    rng = np.random.default_rng(0)
    for _ in range(25):
        x1 = rng.integers(0, 2, 8).astype(np.uint8)
        x2 = rng.integers(0, 2, 8).astype(np.uint8)
        if np.all(x1 == x2):
            continue
        y1, y2, C = icm.propose_pair(x1, x2, rng)
        assert np.array_equal(y1, x2) and np.array_equal(y2, x1)
        assert abs(icm.pair_log_accept(x1, x2, y1, y2)) < 1e-10


def test_icm_full_component_conserves_pair_energy_sparse():
    """On a sparse (2d) topology with random fields, flipping a full
    connected component of the disagreement subgraph changes the pair
    energy by exactly zero — the Houdayer identity, fields included."""
    from tunnelvision.dynamics.icm import HoudayerICM

    tgt = random_spin_glass(9, topology="2d", seed=2, temperature=0.5,
                            random_fields=True)
    icm = HoudayerICM(tgt)  # theta=0, no cap: full components
    rng = np.random.default_rng(1)
    checked = 0
    for _ in range(200):
        x1 = rng.integers(0, 2, 9).astype(np.uint8)
        x2 = rng.integers(0, 2, 9).astype(np.uint8)
        if np.all(x1 == x2):
            continue
        y1, y2, C = icm.propose_pair(x1, x2, rng)
        assert abs(icm.pair_log_accept(x1, x2, y1, y2)) < 1e-9
        checked += 1
    assert checked > 100


def test_icm_symmetry_and_disagreement_invariance():
    """q(pair'|pair) == q(pair|pair') for every strengthened variant, via
    explicit enumeration over seeds; and D is invariant under the move."""
    from tunnelvision.dynamics.icm import HoudayerICM

    tgt = random_spin_glass(7, seed=3, temperature=0.3, random_fields=True)
    rng = np.random.default_rng(2)
    for theta, cap in [(0.0, None), (0.7, None), (0.0, 3), (1.0, 2)]:
        icm = HoudayerICM(tgt, theta=theta, size_cap=cap)
        for _ in range(30):
            x1 = rng.integers(0, 2, 7).astype(np.uint8)
            x2 = rng.integers(0, 2, 7).astype(np.uint8)
            D = np.flatnonzero(x1 != x2)
            if D.size == 0:
                continue
            # forward proposal distribution over resulting pairs
            fwd = {}
            for s in D:
                C = icm.cluster(D, int(s))
                key = tuple(C.tolist())
                fwd[key] = fwd.get(key, 0.0) + 1.0 / D.size
            for key, p_f in fwd.items():
                C = np.array(key, dtype=np.int64)
                y1, y2 = x1.copy(), x2.copy()
                y1[C] = 1 - y1[C]
                y2[C] = 1 - y2[C]
                D2 = np.flatnonzero(y1 != y2)
                assert np.array_equal(D2, D)  # invariance
                # reverse probability of proposing the same cluster
                p_r = 0.0
                for s in D2:
                    if np.array_equal(icm.cluster(D2, int(s)), C):
                        p_r += 1.0 / D2.size
                assert abs(p_f - p_r) < 1e-12, (theta, cap, key)


def test_icm_pair_chain_stationary_and_sf_baseline():
    """Exact pair chain: rows sum to 1, pi x pi is stationary; and the
    pure single-flip pair chain has gap = (single-chain SF gap) / 2,
    the tensor-mixture identity — certifies the sparse machinery."""
    from tunnelvision.dynamics.icm import HoudayerICM, sparse_pair_gap

    n = 6
    tgt = random_spin_glass(n, seed=1, temperature=0.3, random_fields=True)
    icm = HoudayerICM(tgt, theta=0.8)
    P, pi_pair = icm.pair_transition_matrix(n, w_icm=0.5)
    row_sums = np.asarray(P.sum(axis=1)).ravel()
    assert np.max(np.abs(row_sums - 1.0)) < 1e-12
    resid = np.max(np.abs(pi_pair @ P - pi_pair))
    assert resid < 1e-12, resid

    P_sf, pi_pair2 = icm.pair_transition_matrix(n, w_icm=0.0)
    gap_pair_sf = sparse_pair_gap(P_sf, pi_pair2)
    gap_single = spectral_gap(
        MetropolisEngine(tgt, SingleFlip()).transition_matrix(),
        tgt.enumerate_exact(),
    )
    assert abs(gap_pair_sf - gap_single / 2.0) < 1e-8, (gap_pair_sf, gap_single)


# ------------------------------------------------------------ M03: SHELLHASH


def test_shellhash_encoding_matches_enumeration():
    """Solver-enumerated shell == numpy-enumerated integer shell, exactly."""
    from tunnelvision.dynamics.shellhash import ShellHashSampler

    tgt = random_spin_glass(8, seed=1, temperature=1.0, random_fields=True)
    sh = ShellHashSampler(tgt.J, tgt.h, eps=2.0, seed=0)
    rng = np.random.default_rng(3)
    pw = 1 << np.arange(8)
    for _ in range(4):
        x = rng.integers(0, 2, 8).astype(np.uint8)
        clauses, _ = sh._shell_cnf(x)
        sols, _ab = sh._enumerate_cell(clauses, cap=1 << 9)
        got = sorted(int(y @ pw) for y in sols)
        want = sorted(int(i) for i in sh.exact_shell_indices(x))
        assert got == want, (len(got), len(want))


def test_shellhash_min_dh_constraint():
    from tunnelvision.dynamics.shellhash import ShellHashSampler

    tgt = random_spin_glass(8, seed=2, temperature=1.0, random_fields=True)
    sh = ShellHashSampler(tgt.J, tgt.h, eps=2.0, min_dh=3, seed=0)
    x = np.zeros(8, dtype=np.uint8)
    clauses, _ = sh._shell_cnf(x)
    sols, _ab = sh._enumerate_cell(clauses, cap=1 << 9)
    pw = 1 << np.arange(8)
    got = sorted(int(y @ pw) for y in sols)
    want = sorted(int(i) for i in sh.exact_shell_indices(x))
    assert got == want and all(int((y != x).sum()) >= 3 for y in sols)


def test_shellhash_near_uniform_on_shell():
    """Empirical TV distance to uniform-on-shell < 0.1 at 2000 samples."""
    from tunnelvision.dynamics.shellhash import ShellHashSampler

    tgt = random_spin_glass(8, seed=1, temperature=1.0, random_fields=True)
    sh = ShellHashSampler(tgt.J, tgt.h, eps=2.0, seed=7)
    x = np.zeros(8, dtype=np.uint8)
    shell = sh.exact_shell_indices(x)
    pw = 1 << np.arange(8)
    counts = {int(i): 0 for i in shell}
    N = 2000
    for _ in range(N):
        y, _info = sh.sample(x)
        counts[int(y @ pw)] += 1
    tv = 0.5 * sum(abs(c / N - 1.0 / shell.size) for c in counts.values())
    assert tv < 0.10, tv


def test_shellhash_bin_mode_stays_in_bin():
    """Bin-mode proposals land in the proposer's energy bin (the property
    that makes the proposal exactly symmetric under Metropolis)."""
    from tunnelvision.dynamics.shellhash import ShellHashSampler

    tgt = random_spin_glass(8, seed=3, temperature=1.0, random_fields=True)
    sh = ShellHashSampler(tgt.J, tgt.h, eps=2.0, seed=1)
    rng = np.random.default_rng(9)
    w2 = 2 * round(sh.scale * sh.eps)
    done = 0
    for _ in range(20):
        x = rng.integers(0, 2, 8).astype(np.uint8)
        y, info = sh.sample(x, mode="bin")
        if y is None:
            continue                      # bin can contain only x itself
        bx = (sh.energy_int(x) - sh.C0) // w2
        by = (sh.energy_int(y) - sh.C0) // w2
        assert bx == by
        done += 1
    assert done >= 10
