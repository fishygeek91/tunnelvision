import csv
import sys

sys.path.insert(0, "/home/user/m03/src")
sys.path.insert(0, "/home/user/m03/experiments/M04_separation")
import numpy as np

from tunnelvision.dynamics.shellhash import ShellHashSampler
from tunnelvision.targets.ising import random_spin_glass

import importlib
m04a = importlib.import_module("m04a_scaling")

rows = []
for n in (24, 32):
    tgt = random_spin_glass(n, seed=1, temperature=1.0, random_fields=True)
    rng = np.random.default_rng(4000 + 100 * n + 1)
    x = rng.integers(0, 2, n).astype(np.uint8)
    sh = ShellHashSampler(tgt.J, tgt.h, eps=2.0, seed=1, conf_budget=300_000)
    clauses, _ = sh._shell_cnf(x)
    log2s, count_s, status = m04a.approxmc_count(
        [list(map(int, c)) for c in clauses], [int(v) for v in sh.yvar],
        300.0)
    times = []
    for _ in range(8):
        y, info = sh.sample(x)
        times.append(info["seconds"])
    rows.append(dict(n=n, seed=1, regime="rand",
                     E_per_n=round(m04a.energy_per_n(tgt, x), 4), redraws=0,
                     log2S=log2s, count_s=count_s, count_status=status,
                     samples=len(times), aborts=0, empties=0,
                     t_med=round(float(np.median(times)), 3),
                     t_p90=round(float(np.quantile(times, 0.9)), 3),
                     censored=0))
    print(f"n={n} rand: log2S={log2s} ({status}, {count_s}s) "
          f"med={rows[-1]['t_med']}s p90={rows[-1]['t_p90']}s", flush=True)
with open("m04a_rand.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
print("done")
