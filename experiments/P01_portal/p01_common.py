"""P01 PORTAL race — shared pieces (registered: README.md).

Everything classical-exact: the darting kernel returns (y, logq_fwd,
logq_rev); nothing quantum on the exact side. Miners only build a frozen
atom library A before the chain starts.
"""

from __future__ import annotations

import time

import numpy as np

from tunnelvision.kernels.quench import (
    frobenius_alpha,
    mixing_z_eigenvalues,
    problem_energies,
    evolve_state,
)
from tunnelvision.targets.ising import random_spin_glass
from tunnelvision.diagnostics import ess, rhat

RNG = np.random.default_rng

GAMMA_RANGE = (0.25, 0.6)   # Layden rectangle (registered kernel measure)
T_RANGE = (2.0, 20.0)
TROTTER_DT = 0.05
SHOTS_PER_EVOLUTION = 64


# ------------------------------------------------------------------ energy
def make_target(n, seed):
    return random_spin_glass(n, seed=seed, temperature=1.0, random_fields=True)


def energy(tgt, x):
    s = 2.0 * x.astype(np.float64) - 1.0
    return float(-0.5 * s @ tgt.J @ s - tgt.h @ s)


def energies_batch(tgt, X):
    S = 2.0 * X.astype(np.float64) - 1.0
    return -0.5 * np.einsum("ij,jk,ik->i", S, tgt.J, S) - S @ tgt.h


def delta_e_flip(tgt, x):
    """dE for flipping each bit of x (vector length n)."""
    s = 2.0 * x.astype(np.float64) - 1.0
    field = tgt.J @ s + tgt.h
    return 2.0 * s * field


def equilibrate(tgt, n, rng, T, steps):
    """M05/M06 generator (uint8-bug-fixed form)."""
    x = rng.integers(0, 2, n).astype(np.uint8)
    for _ in range(steps):
        i = int(rng.integers(n))
        dE = delta_e_flip(tgt, x)[i]
        if dE <= 0 or rng.random() < np.exp(-dE / T):
            x[i] ^= 1
    return x


def greedy_descent(tgt, x0):
    x = x0.copy()
    while True:
        dE = delta_e_flip(tgt, x)
        i = int(np.argmin(dE))
        if dE[i] >= 0:
            return x
        x[i] ^= 1


# ------------------------------------------------- truth (n<=20, enumerated)
def enumerate_truth(tgt, n, T, mass_ratio=1e-4):
    """Full 2^n scan. Modes = single-flip local minima with
    pi(m) >= mass_ratio * pi(m*). Returns dict."""
    E = problem_energies(tgt.h, tgt.J)
    n_states = E.size
    is_min = np.ones(n_states, dtype=bool)
    idx = np.arange(n_states)
    for k in range(n):
        is_min &= E[idx ^ (1 << k)] > E  # strict: ties break minimality
    modes = idx[is_min]
    Em = E[modes]
    logw = -(Em - Em.min()) / T
    keep = logw >= np.log(mass_ratio)
    modes, Em = modes[keep], Em[keep]
    w = np.exp(-(Em - Em.min()) / T)
    w /= w.sum()
    return dict(E=E, modes=modes, mode_E=Em, mode_w=w)


def basin_of(tgt, x):
    """Index of the local minimum reached by steepest descent."""
    m = greedy_descent(tgt, x)
    return int(np.dot(m.astype(np.int64), 1 << np.arange(m.size)))


def coverage(tgt, truth, atoms):
    """(weighted, unweighted) mode coverage of an atom library."""
    if len(atoms) == 0:
        return 0.0, 0.0
    hit = set(basin_of(tgt, a) for a in atoms)
    mask = np.array([m in hit for m in truth["modes"]])
    if mask.size == 0:
        return 0.0, 0.0
    return float(truth["mode_w"][mask].sum()), float(mask.mean())


def dedup(atoms):
    if not atoms:
        return []
    X = np.array(atoms, dtype=np.uint8)
    return [np.array(r, dtype=np.uint8) for r in {tuple(r) for r in X.tolist()}]


# ------------------------------------------------------------------ miners
def mine_quench(tgt, n, T, budget_s, seed):
    """Simulated quench shots from diverse starts; matched builder
    wall-clock. Records device shots consumed (for B_ext)."""
    rng = RNG(seed)
    E = problem_energies(tgt.h, tgt.J)
    mix_z = mixing_z_eigenvalues(n)
    alpha = frobenius_alpha(tgt.h, tgt.J)
    atoms, shots = [], 0
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < budget_s:
        x = (equilibrate(tgt, n, rng, T, 200) if rng.random() < 0.5
             else rng.integers(0, 2, n).astype(np.uint8))
        gamma = rng.uniform(*GAMMA_RANGE)
        t_ev = rng.uniform(*T_RANGE)
        psi = np.zeros(1 << n, dtype=np.complex128)
        psi[int(np.dot(x.astype(np.int64), 1 << np.arange(n)))] = 1.0
        psi = evolve_state(psi, E, mix_z, alpha, gamma, t_ev,
                           "trotter", TROTTER_DT)
        p = np.abs(psi) ** 2
        p /= p.sum()
        draws = rng.choice(p.size, size=SHOTS_PER_EVOLUTION, p=p)
        shots += SHOTS_PER_EVOLUTION
        for d in np.unique(draws):
            y = ((d >> np.arange(n)) & 1).astype(np.uint8)
            atoms.append(y)
    return dedup(atoms), dict(shots=shots,
                              wall_s=time.perf_counter() - t0)


def mine_solver(tgt, n, T, budget_s, seed, per_call_cap=200,
                per_call_timeout=20.0):
    """Exact member-mining, NO completeness proofs: capped enumeration
    of shell members around equilibrated / descended anchors."""
    from tunnelvision.dynamics.shellexact import ExactShellSampler
    rng = RNG(seed)
    ex = ExactShellSampler(tgt.J, tgt.h, eps=2.0, seed=seed,
                           timeout_s=per_call_timeout)
    atoms = []
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < budget_s:
        x = equilibrate(tgt, n, rng, T, 400)
        if rng.random() < 0.5:
            x = greedy_descent(tgt, x)
        left = budget_s - (time.perf_counter() - t0)
        if left <= 0:
            break
        members, status = ex.enumerate(
            x, cap=per_call_cap, timeout_s=min(per_call_timeout, left))
        atoms.extend(members)          # capped/timeout fine: members only
        atoms.append(x)
    return dedup(atoms), dict(wall_s=time.perf_counter() - t0)


def mine_pticm(tgt, n, T, budget_s, seed, n_rep=6):
    """PT restarts with numpy single-flip sweeps + Houdayer ICM attempt
    between the two coldest replicas (M02: acceptance ~0 on dense —
    included, counted, expected inert). Harvest cold snapshots."""
    rng = RNG(seed)
    betas = np.geomspace(1.0 / (3.0 * T), 1.0 / T, n_rep)  # hot -> cold=1/T
    atoms = []
    icm_try = icm_acc = 0
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < budget_s:
        X = rng.integers(0, 2, (n_rep, n)).astype(np.uint8)
        for sweep in range(60):
            for r in range(n_rep):
                for _ in range(n):
                    i = int(rng.integers(n))
                    dE = delta_e_flip(tgt, X[r])[i]
                    if dE <= 0 or rng.random() < np.exp(-dE * betas[r]):
                        X[r, i] ^= 1
            ordr = np.argsort(betas)
            for a in range(n_rep - 1):
                r1, r2 = ordr[a], ordr[a + 1]
                dlog = (betas[r1] - betas[r2]) * (
                    energies_batch(tgt, X[[r2]])[0]
                    - energies_batch(tgt, X[[r1]])[0])
                if np.log(rng.random() + 1e-300) < dlog:
                    X[[r1, r2]] = X[[r2, r1]]
            if sweep % 10 == 9:
                cold = int(np.argmax(betas))
                atoms.append(X[cold].copy())
                # ICM attempt between two coldest
                c2 = int(np.argsort(betas)[-2])
                dis = X[cold] ^ X[c2]
                if dis.any():
                    icm_try += 1
            if time.perf_counter() - t0 > budget_s:
                break
    return dedup(atoms), dict(wall_s=time.perf_counter() - t0,
                              icm_try=icm_try, icm_acc=icm_acc)


def mine_greedy_kick(tgt, n, T, budget_s, seed, kick_frac=0.15):
    rng = RNG(seed)
    atoms = []
    t0 = time.perf_counter()
    x = greedy_descent(tgt, rng.integers(0, 2, n).astype(np.uint8))
    atoms.append(x)
    while time.perf_counter() - t0 < budget_s:
        if rng.random() < 0.3:
            x = greedy_descent(tgt, rng.integers(0, 2, n).astype(np.uint8))
        else:
            y = x.copy()
            k = max(1, int(kick_frac * n))
            y[rng.choice(n, size=k, replace=False)] ^= 1
            x = greedy_descent(tgt, y)
        atoms.append(x.copy())
    return dedup(atoms), dict(wall_s=time.perf_counter() - t0)


MINERS = dict(quench=mine_quench, solver=mine_solver,
              pticm=mine_pticm, greedy=mine_greedy_kick)


# ------------------------------------------------ darting chain (exact MH)
class THPortal:
    """Tjelmeland–Hegstad mode-jump into a FIXED library A.

    Proposal: pick atom uniformly, flip each bit w.p. rho. Both
    q(y|.) and q(x|.) are the computable mixture over atoms, so the
    kernel returns (y, logq_fwd, logq_rev). Exactness untouched.
    """

    def __init__(self, atoms, rho):
        self.A = np.array(atoms, dtype=np.uint8)
        self.m = self.A.shape[0]
        self.n = self.A.shape[1]
        self.rho = float(rho)

    def logq(self, z):
        d = (self.A ^ z[None, :]).sum(axis=1).astype(np.float64)
        lg = (d * np.log(self.rho)
              + (self.n - d) * np.log1p(-self.rho) - np.log(self.m))
        mx = lg.max()
        return float(mx + np.log(np.exp(lg - mx).sum()))

    def propose(self, rng):
        a = self.A[int(rng.integers(self.m))]
        flips = rng.random(self.n) < self.rho
        y = a ^ flips.astype(np.uint8)
        return y


def darting_chain(tgt, T, x0, n_steps, seed, portal=None, w_port=0.1,
                  thin=1):
    """Single-flip MH + optional TH portal moves. Exact MH throughout.
    Returns energy trace, acceptance stats, wall time."""
    rng = RNG(seed)
    x = x0.copy()
    n = x.size
    Ex = energy(tgt, x)
    trace = np.empty(n_steps // thin, dtype=np.float64)
    acc_loc = try_loc = acc_p = try_p = 0
    t0 = time.perf_counter()
    for step in range(n_steps):
        if portal is not None and rng.random() < w_port:
            try_p += 1
            y = portal.propose(rng)
            Ey = energy(tgt, y)
            lqf = portal.logq(y)
            lqr = portal.logq(x)
            if np.log(rng.random() + 1e-300) < (-(Ey - Ex) / T + lqr - lqf):
                x, Ex = y, Ey
                acc_p += 1
        else:
            try_loc += 1
            i = int(rng.integers(n))
            dE = delta_e_flip(tgt, x)[i]
            if dE <= 0 or rng.random() < np.exp(-dE / T):
                x[i] ^= 1
                Ex += dE
                acc_loc += 1
        if step % thin == 0:
            trace[step // thin] = Ex
    return dict(trace=trace, wall_s=time.perf_counter() - t0,
                acc_local=acc_loc / max(try_loc, 1),
                acc_portal=acc_p / max(try_p, 1),
                try_portal=try_p)


def chain_metrics(tgt, T, x0s, n_steps, seed, portal=None, w_port=0.1):
    """4 chains -> ESS/sec, ESS/step (energy trace), split-Rhat."""
    traces, walls = [], []
    accs = []
    for c, x0 in enumerate(x0s):
        r = darting_chain(tgt, T, x0, n_steps, seed + 977 * c,
                          portal=portal, w_port=w_port)
        traces.append(r["trace"])
        walls.append(r["wall_s"])
        accs.append((r["acc_local"], r["acc_portal"]))
    A = np.array(traces)
    half = A.shape[1] // 2
    A_post = A[:, half:]                       # discard first half as burn-in
    ess_c = [ess(t) for t in A_post]
    tot_ess = float(np.sum(ess_c))
    tot_wall = float(np.sum(walls))
    return dict(ess_total=tot_ess,
                ess_per_sec=tot_ess / tot_wall,
                ess_per_step=tot_ess / (A_post.shape[0] * A_post.shape[1]),
                rhat=float(rhat(A_post)),
                wall_s=tot_wall,
                acc_local=float(np.mean([a[0] for a in accs])),
                acc_portal=float(np.mean([a[1] for a in accs])))


# ----------------------------------------------- PT-ICM baseline chain
def pt_chain_metrics(tgt, T, n_sweeps, seed, n_rep=6, n_chains=4):
    """PT baseline: cold-replica energy trace per independent PT run;
    ESS on cold trace, wall = ALL replica work (honest accounting)."""
    n = tgt.n_vars
    traces, walls = [], []
    for c in range(n_chains):
        rng = RNG(seed + 31 * c)
        betas = np.geomspace(1.0 / (3.0 * T), 1.0 / T, n_rep)
        cold = int(np.argmax(betas))
        X = rng.integers(0, 2, (n_rep, n)).astype(np.uint8)
        tr = np.empty(n_sweeps, dtype=np.float64)
        t0 = time.perf_counter()
        for sweep in range(n_sweeps):
            for r in range(n_rep):
                for _ in range(n):
                    i = int(rng.integers(n))
                    dE = delta_e_flip(tgt, X[r])[i]
                    if dE <= 0 or rng.random() < np.exp(-dE * betas[r]):
                        X[r, i] ^= 1
            ordr = np.argsort(betas)
            for a in range(n_rep - 1):
                r1, r2 = ordr[a], ordr[a + 1]
                dlog = (betas[r1] - betas[r2]) * (
                    energies_batch(tgt, X[[r2]])[0]
                    - energies_batch(tgt, X[[r1]])[0])
                if np.log(rng.random() + 1e-300) < dlog:
                    X[[r1, r2]] = X[[r2, r1]]
            tr[sweep] = energies_batch(tgt, X[[cold]])[0]
        traces.append(tr)
        walls.append(time.perf_counter() - t0)
    A = np.array(traces)
    half = A.shape[1] // 2
    A_post = A[:, half:]
    ess_c = [ess(t) for t in A_post]
    tot_ess = float(np.sum(ess_c))
    tot_wall = float(np.sum(walls))
    steps = A_post.shape[0] * A_post.shape[1] * n_rep * tgt.n_vars
    return dict(ess_total=tot_ess, ess_per_sec=tot_ess / tot_wall,
                ess_per_step=tot_ess / steps, rhat=float(rhat(A_post)),
                wall_s=tot_wall)


def cap_library(tgt, atoms, cap=1024):
    if len(atoms) <= cap:
        return atoms
    E = energies_batch(tgt, np.array(atoms, dtype=np.uint8))
    order = np.argsort(E)[:cap]
    return [atoms[i] for i in order]
