"""M02a — does Houdayer ICM reach concentration-at-distance (C_far)?

For each M01 cell (n in {8,10}, seed 1-3, T in {1, .3, .1}) and a grid of
ICM variants, Monte-Carlo estimate the single-replica proposal statistics
under stationarity: pairs (x1, x2) ~ pi x pi (exact, via enumeration),
seed site uniform on the disagreement set, deterministic cluster C, and
the replica-1 proposal y1 = flip(x1, C).

Reported per (cell, variant), matching M01's C_far convention
(d_H >= 3 AND |dE| <= 2 raw units, proposal mass BEFORE acceptance):

    C_far_hat   E[ 1(|C| >= 3) 1(|dE_1| <= 2) ]
    C_J2_hat    E[ 1(|dE_1| <= 2) ]  (off-diagonal only)
    mean_dH     E[ |C| ]
    mean_acc    E[ min(1, exp(dlp_pair)) ]   (pair-Metropolis acceptance)
    frac_id     fraction of draws with empty D (identity move)

Interpretation caveat (goes in the writeup): for the FAITHFUL variant the
move is a replica swap, whose single-chain marginal is the perfect
independence proposal q(y|x) = pi(y) — its C_far is aspirational, not
realizable, because the pair move is a statistical no-op. C_far here
measures where ICM proposals LAND; whether the pair chain actually mixes
faster is M02b's exact-gap question.
"""

from __future__ import annotations

import csv
import sys

import numpy as np

from tunnelvision.bits import all_binary_states
from tunnelvision.dynamics.icm import HoudayerICM
from tunnelvision.targets.ising import random_spin_glass

VARIANTS = [
    ("faithful", 0.0, None),
    ("theta=0.5", 0.5, None),
    ("theta=1.0", 1.0, None),
    ("theta=1.5", 1.5, None),
    ("theta=2.0", 2.0, None),
    ("cap=2", 0.0, 2),
    ("cap=3", 0.0, 3),
    ("cap=5", 0.0, 5),
]

M_DRAWS = 10_000


def run_cell(n: int, seed: int, T: float, rng: np.random.Generator):
    tgt = random_spin_glass(n, seed=seed, temperature=T, random_fields=True)
    S = all_binary_states(n)
    lp = tgt.log_prob_batch(S)
    pi = np.exp(lp - lp.max())
    pi /= pi.sum()
    pw = 1 << np.arange(n)
    icms = [(v, HoudayerICM(tgt, theta=th, size_cap=cap)) for v, th, cap in VARIANTS]

    i1s = rng.choice(S.shape[0], size=M_DRAWS, p=pi)
    i2s = rng.choice(S.shape[0], size=M_DRAWS, p=pi)
    stats = {v: dict(far=0.0, cj2=0.0, dh=0.0, acc=0.0, ident=0, moves=0)
             for v, _ in icms}
    for i1, i2 in zip(i1s, i2s):
        x1, x2 = S[i1], S[i2]
        D = np.flatnonzero(x1 != x2)
        if D.size == 0:
            for v, _ in icms:
                stats[v]["ident"] += 1
            continue
        seed_site = int(D[rng.integers(0, D.size)])
        for v, icm in icms:
            C = icm.cluster(D, seed_site)
            mask = int(np.sum(pw[C]))
            j1, j2 = i1 ^ mask, i2 ^ mask
            dE1 = abs(lp[j1] - lp[i1]) * T
            dlp_pair = (lp[j1] + lp[j2]) - (lp[i1] + lp[i2])
            st = stats[v]
            st["moves"] += 1
            st["dh"] += C.size
            st["acc"] += min(1.0, float(np.exp(dlp_pair)))
            if dE1 <= 2.0:
                st["cj2"] += 1.0
                if C.size >= 3:
                    st["far"] += 1.0
    rows = []
    for (v, _), (name, theta, cap) in zip(icms, VARIANTS):
        st = stats[v]
        m = max(st["moves"], 1)
        rows.append(dict(
            n=n, seed=seed, T=T, variant=name, theta=theta,
            cap=("" if cap is None else cap), mc_draws=M_DRAWS,
            C_far_hat=st["far"] / M_DRAWS,     # identity draws count 0, like diag
            C_J2_hat=st["cj2"] / M_DRAWS,
            mean_dH=st["dh"] / m,
            mean_acc=st["acc"] / m,
            frac_id=st["ident"] / M_DRAWS,
        ))
        print(f"n={n} s={seed} T={T:4g} {name:10s} "
              f"C_far={rows[-1]['C_far_hat']:.3f} CJ2={rows[-1]['C_J2_hat']:.3f} "
              f"dH={rows[-1]['mean_dH']:5.2f} acc={rows[-1]['mean_acc']:.3f} "
              f"id={rows[-1]['frac_id']:.2f}", flush=True)
    return rows


def main(out_path: str):
    rng = np.random.default_rng(20260831)
    rows = []
    for n in [8, 10]:
        for seed in [1, 2, 3]:
            for T in [1.0, 0.3, 0.1]:
                rows.extend(run_cell(n, seed, T, rng))
    with open(out_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} rows -> {out_path}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "m02a_icm_cfar.csv")
