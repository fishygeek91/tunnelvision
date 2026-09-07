"""M06 — PB-native deep-shell proposal cost vs n (registered: README.md)."""

from __future__ import annotations

import csv
import sys
import time

import numpy as np

from tunnelvision.dynamics.shellexact import ExactShellSampler
from tunnelvision.dynamics.shellhash import ShellHashSampler
from tunnelvision.targets.ising import random_spin_glass

sys.path.insert(0, "/root/m05/experiments/M05_depth")
from m05_unified import equilibrate  # fixed generator, T ladder  # noqa: E402
from m05_common import energy_per_n  # noqa: E402

NS = [16, 20, 24, 28, 32]
SEEDS = [1, 2]
TFACT = [0.55, 0.35]
N_SAMP = 5
EX_TIMEOUT = 300.0
CMS_CAP = 120.0


def main(out="m06_scan.csv"):
    rows = []
    for n in NS:
        for seed in SEEDS:
            tgt = random_spin_glass(n, seed=seed, temperature=1.0,
                                    random_fields=True)
            for tf in TFACT:
                T = tf * np.sqrt(n)
                rng = np.random.default_rng(6000 + 100 * n + seed + int(tf * 100))
                x = equilibrate(tgt, n, rng, T, 8000)
                ex = ExactShellSampler(tgt.J, tgt.h, eps=2.0, seed=seed,
                                       timeout_s=EX_TIMEOUT)
                times, modes, sizes = [], [], []
                for _ in range(N_SAMP):
                    y, info = ex.sample(x)
                    times.append(info["seconds"])
                    modes.append(info["mode"])
                    sizes.append(info["size"])
                    if info["mode"] == "abort":
                        break            # censored: one 300 s timeout suffices
                cms_t = float("nan")
                if n <= 20:
                    sh = ShellHashSampler(tgt.J, tgt.h, eps=2.0, seed=seed,
                                          conf_budget=300_000)
                    t0 = time.perf_counter()
                    sh.sample(x)
                    cms_t = time.perf_counter() - t0
                arr = np.array(times)
                rows.append(dict(
                    n=n, seed=seed, T_factor=tf, T=round(float(T), 3),
                    E_per_n=round(energy_per_n(tgt, x), 4),
                    mode=modes[-1], size=sizes[-1] if sizes else -1,
                    samples=len(times),
                    t_med=round(float(np.median(arr)), 3),
                    t_max=round(float(arr.max()), 3),
                    censored=int(modes[-1] == "abort"),
                    cms_1sample_s=(round(cms_t, 3) if n <= 20 else "")))
                print(f"n={n} s={seed} T={tf}√n: E/n={rows[-1]['E_per_n']:+.3f} "
                      f"mode={modes[-1]} |S|={sizes[-1]} "
                      f"t_med={rows[-1]['t_med']}s max={rows[-1]['t_max']}s "
                      f"cens={rows[-1]['censored']} cms={cms_t:.1f}s",
                      flush=True)
                with open(out, "w", newline="") as fh:
                    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
                    w.writeheader()
                    w.writerows(rows)
    print("done")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "m06_scan.csv")
