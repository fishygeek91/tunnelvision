"""P01 Step 2 — Tier T truth race, n <= 20 (registered: README.md).

Miner arms at matched builder wall-clock; mode coverage vs enumerated
truth; downstream darting-chain ESS/sec and ESS/step with split-Rhat
gating. Decides K1.
"""

from __future__ import annotations

import csv
import sys

import numpy as np

from p01_common import (RNG, MINERS, THPortal, cap_library, chain_metrics,
                        coverage, enumerate_truth, equilibrate, make_target,
                        pt_chain_metrics)

NS = [12, 16, 20]
SEEDS = [1, 2, 3]
TFACT = [0.55, 0.35]
BUDGETS = [2.0, 10.0]
RHO_TIMES_N = 0.5          # best from A3 gate scan
W_PORT = 0.1
N_STEPS = 40_000
PT_SWEEPS = 400            # ~comparable wall to darting chains


def main(out="p01_tier_t.csv"):
    mine_rows, chain_rows = [], []
    for n in NS:
        for seed in SEEDS:
            tgt = make_target(n, seed)
            for tf in TFACT:
                T = tf * np.sqrt(n)
                truth = enumerate_truth(tgt, n, T)
                rng = RNG(4000 + 10 * n + seed)
                x0s = [equilibrate(tgt, n, rng, T, 2000) for _ in range(4)]
                libs = {}
                for bud in BUDGETS:
                    for arm, fn in MINERS.items():
                        atoms, info = fn(tgt, n, T, bud,
                                         seed=5000 + 7 * n + seed + int(bud))
                        cov_w, cov_u = coverage(tgt, truth, atoms)
                        mine_rows.append(dict(
                            n=n, seed=seed, T_factor=tf, budget_s=bud,
                            arm=arm, n_atoms=len(atoms),
                            cov_weighted=round(cov_w, 4),
                            cov_unweighted=round(cov_u, 4),
                            n_modes=len(truth["modes"]),
                            shots=info.get("shots", 0),
                            wall_s=round(info["wall_s"], 2)))
                        print(mine_rows[-1], flush=True)
                        if bud == BUDGETS[-1]:
                            libs[arm] = atoms
                # downstream chains: 4 portal arms + 2 baselines
                for arm, atoms in libs.items():
                    lib = cap_library(tgt, atoms)
                    portal = THPortal(lib, RHO_TIMES_N / n) if lib else None
                    m = chain_metrics(tgt, T, x0s, N_STEPS,
                                      seed=6100 + n + seed,
                                      portal=portal, w_port=W_PORT)
                    chain_rows.append(dict(
                        n=n, seed=seed, T_factor=tf, chain=f"portal-{arm}",
                        n_atoms=len(lib), **{k: round(v, 6)
                                             for k, v in m.items()}))
                    print(chain_rows[-1], flush=True)
                m = chain_metrics(tgt, T, x0s, N_STEPS,
                                  seed=6100 + n + seed, portal=None)
                chain_rows.append(dict(n=n, seed=seed, T_factor=tf,
                                       chain="local-MH", n_atoms=0,
                                       **{k: round(v, 6)
                                          for k, v in m.items()}))
                print(chain_rows[-1], flush=True)
                m = pt_chain_metrics(tgt, T, PT_SWEEPS,
                                     seed=6100 + n + seed)
                m["acc_local"] = m["acc_portal"] = -1
                chain_rows.append(dict(n=n, seed=seed, T_factor=tf,
                                       chain="PT-ICM", n_atoms=0,
                                       **{k: round(v, 6)
                                          for k, v in m.items()}))
                print(chain_rows[-1], flush=True)
                for rows, path in ((mine_rows, out),
                                   (chain_rows, out.replace(".csv",
                                                            "_chains.csv"))):
                    with open(path, "w", newline="") as fh:
                        w = csv.DictWriter(fh,
                                           fieldnames=list(rows[0].keys()))
                        w.writeheader()
                        w.writerows(rows)
    print("done")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "p01_tier_t.csv")
