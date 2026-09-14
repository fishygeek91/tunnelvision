"""P01 Step 3 — Tier S scale race, classical arms only (registered: README.md).

Pre-screen for measured classical metastability first; then solver-mined
PORTAL vs PT-ICM baseline. Decides K2. Quench arm absent (statevector cap
~n<=28; MPS truncation corrupts shell physics): K1 at scale is undecidable
without hardware.
"""

from __future__ import annotations

import csv
import sys

import numpy as np

from p01_common import (RNG, THPortal, cap_library, chain_metrics,
                        equilibrate, make_target, mine_solver,
                        pt_chain_metrics)
from tunnelvision.diagnostics import rhat

NS = [50, 100, 200]
SEEDS = [1, 2]
TFACT = [0.55, 0.35]
SCREEN_STEPS = 60_000
SCREEN_RHAT = 1.2
BUILD_BUDGET = 30.0
RHO_TIMES_N = 0.5
W_PORT = 0.1
N_STEPS = 100_000
PT_SWEEPS_BY_N = {50: 300, 100: 200, 200: 120}


def screen(tgt, n, T, seed):
    """4 dispersed single-flip chains; metastable iff split-Rhat > 1.2."""
    from p01_common import darting_chain
    rng = RNG(seed)
    traces = []
    for c in range(4):
        x0 = rng.integers(0, 2, n).astype(np.uint8)
        r = darting_chain(tgt, T, x0, SCREEN_STEPS, seed + 13 * c,
                          portal=None)
        traces.append(r["trace"])
    A = np.array(traces)[:, SCREEN_STEPS // 2:]
    return float(rhat(A))


def main(out="p01_tier_s.csv"):
    rows = []
    for n in NS:
        for seed in SEEDS:
            tgt = make_target(n, seed)
            for tf in TFACT:
                T = tf * np.sqrt(n)
                rh = screen(tgt, n, T, 8800 + n + seed)
                row = dict(n=n, seed=seed, T_factor=tf,
                           screen_rhat=round(rh, 4),
                           valid_for_k2=int(rh > SCREEN_RHAT))
                if rh <= SCREEN_RHAT:
                    row.update(note="baseline mixes; INVALID for K2")
                    rows.append(row)
                    print(row, flush=True)
                    continue
                rng = RNG(4400 + n + seed)
                x0s = [rng.integers(0, 2, n).astype(np.uint8)
                       for _ in range(4)]
                atoms, info = mine_solver(tgt, n, T, BUILD_BUDGET,
                                          seed=5500 + n + seed,
                                          per_call_cap=100,
                                          per_call_timeout=10.0)
                lib = cap_library(tgt, atoms)
                portal = THPortal(lib, RHO_TIMES_N / n) if lib else None
                mp = chain_metrics(tgt, T, x0s, N_STEPS,
                                   seed=6600 + n + seed,
                                   portal=portal, w_port=W_PORT)
                mb = pt_chain_metrics(tgt, T, PT_SWEEPS_BY_N[n],
                                      seed=6600 + n + seed)
                gain = mp["ess_per_sec"] / mb["ess_per_sec"]
                gain_amort = (mp["ess_total"]
                              / (mp["wall_s"] + info["wall_s"])
                              / mb["ess_per_sec"])
                row.update(n_atoms=len(lib), build_s=round(info["wall_s"], 1),
                           portal_ess_sec=round(mp["ess_per_sec"], 4),
                           portal_ess_step=round(mp["ess_per_step"], 8),
                           portal_rhat=round(mp["rhat"], 3),
                           portal_acc=round(mp["acc_portal"], 4),
                           pticm_ess_sec=round(mb["ess_per_sec"], 4),
                           pticm_ess_step=round(mb["ess_per_step"], 8),
                           pticm_rhat=round(mb["rhat"], 3),
                           gain=round(gain, 3),
                           gain_amortized=round(gain_amort, 3))
                rows.append(row)
                print(row, flush=True)
                keys = sorted({k for r in rows for k in r})
                with open(out, "w", newline="") as fh:
                    w = csv.DictWriter(fh, fieldnames=keys)
                    w.writeheader()
                    w.writerows(rows)
    print("done")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "p01_tier_s.csv")
