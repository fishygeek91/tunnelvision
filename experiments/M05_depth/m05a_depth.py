"""M05a — depth-resolved hardness curve at n=20 (registered: README.md)."""

from __future__ import annotations

import csv
import sys
import time

import numpy as np

from tunnelvision.dynamics.shellhash import ShellHashSampler
from tunnelvision.targets.ising import random_spin_glass

from m05_common import (approxmc_count, energy_per_n, greedy_descent,
                        metropolis_chain)

N = 20
EPS = 2.0
SNAPSHOTS = [0, 200, 1000, 5000, 20000]
N_SAMP = 5
BUDGET = 300_000
STATE_CAP_S = 400.0
COUNT_CAP_S = 300.0


def main(out="m05a_depth.csv"):
    rows = []
    for seed in (1, 2, 3):
        tgt = random_spin_glass(N, seed=seed, temperature=1.0,
                                random_fields=True)
        rng = np.random.default_rng(5000 + seed)
        snaps = metropolis_chain(tgt, N, rng, SNAPSHOTS)
        states = [(str(k), v) for k, v in snaps.items()]
        states.append(("greedy", greedy_descent(tgt, snaps[SNAPSHOTS[-1]])))
        seen = set()
        for label, x in states:
            key = x.tobytes()
            dup = key in seen
            seen.add(key)
            sh = ShellHashSampler(tgt.J, tgt.h, eps=EPS, seed=seed,
                                  conf_budget=BUDGET)
            clauses, _ = sh._shell_cnf(x)
            log2s, count_s, cstat = approxmc_count(
                [list(map(int, c)) for c in clauses],
                [int(v) for v in sh.yvar], COUNT_CAP_S)
            times, aborts, empties = [], 0, 0
            t0 = time.perf_counter()
            for _ in range(N_SAMP):
                if time.perf_counter() - t0 > STATE_CAP_S:
                    break
                y, info = sh.sample(x)
                times.append(info["seconds"])
                if info.get("aborted"):
                    aborts += 1
                elif y is None:
                    empties += 1
            arr = np.array(times) if times else np.array([np.nan])
            rows.append(dict(seed=seed, ladder=label, dup=int(dup),
                             E_per_n=round(energy_per_n(tgt, x), 4),
                             log2S=log2s, count_s=count_s, count_status=cstat,
                             samples=len(times), aborts=aborts,
                             empties=empties,
                             t_med=round(float(np.nanmedian(arr)), 3),
                             censored=int(len(times) < N_SAMP)))
            print(f"s={seed} {label:>6s}: E/n={rows[-1]['E_per_n']:+.3f} "
                  f"log2S={log2s} t_med={rows[-1]['t_med']}s "
                  f"aborts={aborts} cens={rows[-1]['censored']} dup={int(dup)}",
                  flush=True)
            with open(out, "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
                w.writeheader()
                w.writerows(rows)
    print("done")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "m05a_depth.csv")
