"""M03b — the kill test: wall-clock per shell sample beyond enumeration.

For n in {16, 20, 27} (seeds 1-3, all-to-all, random fields), time
SHELLHASH from two state regimes:

  * 'rand' — x uniform at random (hot chain / early exploration);
  * 'cold' — x after 4000 single-flip Metropolis steps at T=0.1 (the
    low-energy states an actual chain occupies, where shells are tight
    and the M02a mechanism says the interesting proposals live).

N_SAMP samples per x, 2 x's per (n, seed, regime). Conflict budget
500k per SAT call (~tens of seconds worst case) so a hard cell aborts
and is COUNTED rather than hanging the run; aborted samples report as
failures, not as missing data.

Pre-registered (README.md): median warm time > 1 s at n=20 in the
MCMC-relevant regime kills the deploy lane; quench hardware reference
1e-2 s/proposal. Host recorded per row (container timings are ~1.7x
slower than the dev laptop VM on identical work — measured, M03a log).
"""

from __future__ import annotations

import csv
import platform
import sys
import time

import numpy as np

from tunnelvision.dynamics.shellhash import ShellHashSampler
from tunnelvision.targets.ising import random_spin_glass

EPS = 2.0
N_SAMP = 30
N_X = 2
BUDGET = 500_000


def metropolis_cold(tgt, n, rng, steps=4000, T=0.1):
    x = rng.integers(0, 2, n).astype(np.uint8)
    lp = tgt.log_prob(x) * tgt.temperature / T  # rescale to T
    s = 2 * x.astype(np.float64) - 1
    J, h = tgt.J, tgt.h
    E = float(-(np.triu(J, 1) * np.outer(s, s)).sum() - h @ s)
    for _ in range(steps):
        i = int(rng.integers(n))
        s_i = 2 * int(x[i]) - 1
        dE = 2 * s_i * (J[i] @ (2 * x.astype(np.int64) - 1)  # FIXED 2026-09-07: uint8 wraparound (2*x-1 -> 255) corrupted the chain; see M05_RESULTS.md - J[i, i] * s_i + h[i])
        if dE <= 0 or rng.random() < np.exp(-dE / T):
            x[i] ^= 1
            E += dE
    return x


def main(out_path: str, ns=(16, 20, 27)):
    host = "container" if "claude" in platform.node() else platform.node()
    rows = []
    for n in ns:
        for seed in [1, 2, 3]:
            tgt = random_spin_glass(n, seed=seed, temperature=1.0,
                                    random_fields=True)
            rng = np.random.default_rng(1000 * n + seed)
            for regime in ["rand", "cold"]:
                for k in range(N_X):
                    x = (rng.integers(0, 2, n).astype(np.uint8)
                         if regime == "rand"
                         else metropolis_cold(tgt, n, rng))
                    sh = ShellHashSampler(tgt.J, tgt.h, eps=EPS,
                                          seed=seed * 10 + k,
                                          conf_budget=BUDGET)
                    times, aborts, cells, ms = [], 0, [], []
                    empty = 0
                    for _ in range(N_SAMP):
                        y, info = sh.sample(x)
                        if info.get("aborted"):
                            aborts += 1
                            times.append(info["seconds"])
                            continue
                        if y is None:
                            empty += 1
                            times.append(info["seconds"])
                            continue
                        times.append(info["seconds"])
                        cells.append(info["cell"])
                        ms.append(info["m"])
                    warm = np.array(times[1:]) if len(times) > 1 else np.array(times)
                    rows.append(dict(
                        n=n, seed=seed, regime=regime, x_idx=k, host=host,
                        n_samp=N_SAMP, aborts=aborts, empty=empty,
                        t_cold=round(times[0], 3),
                        t_med=round(float(np.median(warm)), 4),
                        t_p90=round(float(np.quantile(warm, 0.9)), 4),
                        t_mean=round(float(warm.mean()), 4),
                        m_mode=(int(np.median(ms)) if ms else -1),
                        cell_med=(float(np.median(cells)) if cells else 0)))
                    print(f"n={n} s={seed} {regime} x{k}: "
                          f"med={rows[-1]['t_med']*1e3:.0f}ms "
                          f"p90={rows[-1]['t_p90']*1e3:.0f}ms "
                          f"aborts={aborts} empty={empty} m~{rows[-1]['m_mode']}",
                          flush=True)
                    with open(out_path, "w", newline="") as fh:
                        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
                        w.writeheader()
                        w.writerows(rows)
    print(f"wrote {len(rows)} rows -> {out_path}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "m03b_scale.csv")
