"""M01 — the Sec. 7 test: does energy concentration C(q) predict advantage?

For a zoo of exact-tier kernels, compute the off-diagonal energy
concentration under two window conventions and the exact MH spectral gap:

    C_T(q; kappa) = sum_{x != y} pi(x) q(y|x) 1[|E(y)-E(x)| <= kappa*T]
    C_J(q; eps)   = sum_{x != y} pi(x) q(y|x) 1[|E(y)-E(x)| <= eps]   (raw units)

Also records mean proposal Hamming distance (weighted the same way).
Output: CSV, one row per (instance, T, kernel).
"""

from __future__ import annotations

import csv
import sys

import numpy as np

from tunnelvision.bits import all_binary_states
from tunnelvision.diagnostics import spectral_gap
from tunnelvision.dynamics import DemonKernel
from tunnelvision.dynamics.dlp import DLPKernel
from tunnelvision.engine import MetropolisEngine
from tunnelvision.kernels.base import Kernel
from tunnelvision.kernels.classical import AddDeleteSwap, SingleFlip, UniformFlip
from tunnelvision.kernels.quantum import QuenchKernel
from tunnelvision.targets.ising import random_spin_glass


class ShellOracle(Kernel):
    def __init__(self, tgt, eps_raw):
        self.tgt = tgt
        self.eps = eps_raw / tgt.temperature  # log-prob units
        self.name = f"shell(eps={eps_raw:g})"

    def propose(self, x, rng):
        raise NotImplementedError

    def proposal_matrix(self, n):
        lp = self.tgt.log_prob_batch(all_binary_states(n))
        N = 1 << n
        M = (np.abs(lp[:, None] - lp[None, :]) <= self.eps).astype(float)
        np.fill_diagonal(M, 0)
        cnt = M.sum(1)
        Q = np.zeros((N, N))
        nz = cnt > 0
        Q[nz] = M[nz] / cnt[nz, None]
        Q[~nz, ~nz] = 1.0
        return Q


class SoftSpin(DLPKernel):
    def _logit(self, dE):
        return -self.temper * np.abs(dE) - 1.0 / (2.0 * self.alpha)


def concentration(Q, pi, dE_raw, ham, thresholds_raw):
    """C and mean Hamming distance for each raw-unit threshold."""
    W = pi[:, None] * Q
    np.fill_diagonal(W, 0.0)
    tot = W.sum()
    out = {}
    for eps in thresholds_raw:
        out[eps] = float(W[np.abs(dE_raw) <= eps].sum())
    mean_ham = float((W * ham).sum() / tot) if tot > 0 else 0.0
    return out, float(tot), mean_ham


def main(out_path: str):
    kappas = [1.0, 2.0, 4.0]          # C_T windows (in units of T)
    eps_raw_list = [1.0, 2.0, 3.0]    # C_J windows (raw)
    rows = []
    for n in [8, 10]:
        S = all_binary_states(n)
        ham_base = (S[:, None, :] != S[None, :, :]).sum(2)
        for seed in [1, 2, 3]:
            quench_Q = None
            for T in [1.0, 0.3, 0.1]:
                tgt = random_spin_glass(n, seed=seed, temperature=T, random_fields=True)
                pi = tgt.enumerate_exact()
                lp = tgt.log_prob_batch(S)
                dE_raw = np.abs(lp[:, None] - lp[None, :]) * T  # raw energy units
                if quench_Q is None:
                    quench_Q = QuenchKernel(
                        tgt.h, tgt.J, evolution="exact", n_gamma=6, n_t=8
                    ).proposal_matrix(n)
                zoo: list[tuple[str, np.ndarray]] = [
                    ("uniform", UniformFlip().proposal_matrix(n)),
                    ("single-flip", SingleFlip().proposal_matrix(n)),
                    ("add-delete-swap", AddDeleteSwap().proposal_matrix(n)),
                    ("quench", quench_Q),
                    ("demon(L=4,Td=1)", DemonKernel(tgt, sweep_len=4).proposal_matrix(n)),
                    ("demon(L=4,Td=8)", DemonKernel(tgt, sweep_len=4, demon_temperature=8.0).proposal_matrix(n)),
                    ("dlp(t=.5,a=3)", DLPKernel(tgt, alpha=3.0, temper=0.5).proposal_matrix(n)),
                    ("dlp(t=.05,a=3)", DLPKernel(tgt, alpha=3.0, temper=0.05).proposal_matrix(n)),
                    ("dlp(t=0,a=3)", DLPKernel(tgt, alpha=3.0, temper=0.0).proposal_matrix(n)),
                    ("softspin(c=.1,a=30)", SoftSpin(tgt, alpha=30.0, temper=0.1 * T).proposal_matrix(n)),
                    ("softspin(c=1,a=30)", SoftSpin(tgt, alpha=30.0, temper=1.0 * T).proposal_matrix(n)),
                ] + [(f"shell(eps={e:g})", ShellOracle(tgt, e).proposal_matrix(n)) for e in [1, 2, 3, 4, 6]]
                for name, Q in zoo:
                    kern = type("K", (Kernel,), {
                        "name": name,
                        "propose": lambda *a: (_ for _ in ()).throw(NotImplementedError()),
                        "proposal_matrix": lambda self, m, _Q=Q: _Q,
                    })()
                    try:
                        gap = spectral_gap(MetropolisEngine(tgt, kern).transition_matrix(), pi)
                    except Exception as exc:  # disconnected shells etc.
                        print(f"skip {n} {seed} {T} {name}: {exc}")
                        continue
                    cj, offdiag, mh = concentration(Q, pi, dE_raw, ham_base, eps_raw_list)
                    ct, _, _ = concentration(Q, pi, dE_raw, ham_base, [k * T for k in kappas])
                    row = dict(n=n, seed=seed, T=T, kernel=name, gap=gap,
                               offdiag_mass=offdiag, mean_hamming=mh)
                    for e in eps_raw_list:
                        row[f"C_J{e:g}"] = cj[e]
                    for k in kappas:
                        row[f"C_T{k:g}"] = ct[k * T]
                    rows.append(row)
                    print(f"{n} {seed} {T:4g} {name:22s} gap={gap:.3e} C_J2={row['C_J2']:.3f} "
                          f"C_T2={row['C_T2']:.3f} ham={mh:.2f}", flush=True)
    with open(out_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} rows -> {out_path}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "m01_results.csv")
