"""M03c — per-sample diagnostic spot-check (NOT the registered run).

Purpose: (1) characterize the n=16 seed=3 'rand' cell where the registered
M03b run stalls >55 min; (2) get the kill-metric number (n=20 cold) with
per-sample logging so slow samples are visible as they happen.

Lower max_tries (12 vs 40) and budget 300k: labeled deviation, reported
as the diagnostic it is. Every sample prints wall-clock, m, cell size,
aborted flag.
"""

from __future__ import annotations

import sys
import time

import numpy as np

sys.path.insert(0, "/home/user/m03/src")
from tunnelvision.dynamics.shellhash import ShellHashSampler  # noqa: E402
from tunnelvision.targets.ising import random_spin_glass  # noqa: E402

sys.path.insert(0, "/home/user/m03/experiments/M03_shellhash")
from m03b_scale import metropolis_cold  # noqa: E402

CELLS = [
    (16, 3, "rand"),
    (20, 1, "cold"),
    (20, 2, "cold"),
    (20, 3, "cold"),
    (20, 1, "rand"),
    (27, 1, "cold"),
]
N_SAMP = 8


def main():
    for n, seed, regime in CELLS:
        tgt = random_spin_glass(n, seed=seed, temperature=1.0,
                                random_fields=True)
        rng = np.random.default_rng(1000 * n + seed)
        x = (rng.integers(0, 2, n).astype(np.uint8) if regime == "rand"
             else metropolis_cold(tgt, n, rng))
        sh = ShellHashSampler(tgt.J, tgt.h, eps=2.0, seed=seed,
                              conf_budget=300_000)
        print(f"--- n={n} seed={seed} {regime} ---", flush=True)
        times = []
        for k in range(N_SAMP):
            t0 = time.perf_counter()
            y, info = sh.sample(x, max_tries=12)
            dt = time.perf_counter() - t0
            times.append(dt)
            print(f"  sample {k}: {dt:7.2f}s m={info['m']} cell={info['cell']} "
                  f"calls={info['sat_calls']} aborted={info.get('aborted')} "
                  f"none={y is None}", flush=True)
        print(f"  => median {np.median(times):.2f}s mean {np.mean(times):.2f}s",
              flush=True)


if __name__ == "__main__":
    main()
