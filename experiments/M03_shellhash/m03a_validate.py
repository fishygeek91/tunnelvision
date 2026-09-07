"""M03a — correctness, uniformity, and wall-clock at enumerable sizes.

For each M01 instance (n in {8,10}, seed 1-3), draw stationary states x
from the exact Gibbs distribution at T in {1.0, 0.1}, and for each x:

  * enumerate the exact integer shell (eps=2 raw) and the real-J shell,
    reporting sizes and their symmetric difference (quantization audit);
  * draw N samples with SHELLHASH; report TV distance to uniform-on-shell
    and per-sample wall-clock (first sample = cold cache, reported apart).

Pass gates (pre-registered in README.md): TV < 0.10 at >= 1000 samples on
every cell; solver shell == integer shell is enforced by the test suite.
"""

from __future__ import annotations

import csv
import sys
import time

import numpy as np

from tunnelvision.bits import all_binary_states
from tunnelvision.dynamics.shellhash import ShellHashSampler
from tunnelvision.targets.ising import random_spin_glass

EPS = 2.0
N_SAMPLES = 2000
N_X = 1  # states per (instance, T)


def real_shell(J, h, x, eps):
    n = x.size
    S = all_binary_states(n)
    sp = 2 * S.astype(np.float64) - 1
    Ju = np.triu(J, 1)
    E = -np.einsum("ki,ij,kj->k", sp, Ju, sp) - sp @ h
    s0 = 2 * x.astype(np.float64) - 1
    e0 = float(-(Ju * np.outer(s0, s0)).sum() - h @ s0)
    ok = np.abs(E - e0) <= eps
    pw = 1 << np.arange(n)
    ok[int(x @ pw)] = False
    return np.flatnonzero(ok)


def main(out_path: str, ns=(8, 10)):
    rng = np.random.default_rng(303)
    rows = []
    for n in ns:
        S = all_binary_states(n)
        pw = 1 << np.arange(n)
        for seed in [1, 2, 3]:
            for T in [1.0, 0.1]:
                tgt = random_spin_glass(n, seed=seed, temperature=T,
                                        random_fields=True)
                pi = tgt.enumerate_exact()
                sh = ShellHashSampler(tgt.J, tgt.h, eps=EPS, seed=seed)
                xs = rng.choice(S.shape[0], size=N_X, p=pi)
                for xi in xs:
                    x = S[xi]
                    shell_int = sh.exact_shell_indices(x)
                    shell_real = real_shell(tgt.J, tgt.h, x, EPS)
                    sym_diff = np.setxor1d(shell_int, shell_real).size
                    if shell_int.size == 0:
                        rows.append(dict(n=n, seed=seed, T=T, x=int(xi),
                                         shell=0, shell_real=shell_real.size,
                                         sym_diff=sym_diff, tv=np.nan,
                                         t_cold=np.nan, t_med=np.nan,
                                         t_mean=np.nan, calls_mean=np.nan))
                        continue
                    counts = {int(i): 0 for i in shell_int}
                    times, calls = [], []
                    for k in range(N_SAMPLES):
                        y, info = sh.sample(x)
                        counts[int(y @ pw)] += 1
                        times.append(info["seconds"])
                        calls.append(info["sat_calls"])
                    tv = 0.5 * sum(abs(c / N_SAMPLES - 1.0 / shell_int.size)
                                   for c in counts.values())
                    warm = np.array(times[1:])
                    rows.append(dict(
                        n=n, seed=seed, T=T, x=int(xi),
                        shell=int(shell_int.size),
                        shell_real=int(shell_real.size), sym_diff=int(sym_diff),
                        tv=round(float(tv), 4),
                        t_cold=round(times[0], 4),
                        t_med=round(float(np.median(warm)), 5),
                        t_mean=round(float(warm.mean()), 5),
                        calls_mean=round(float(np.mean(calls)), 2)))
                    print(f"n={n} s={seed} T={T:g} x={xi}: shell={shell_int.size} "
                          f"symdiff={sym_diff} TV={tv:.3f} "
                          f"t_med={np.median(warm)*1e3:.1f}ms", flush=True)
    with open(out_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} rows -> {out_path}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "m03a_validate.csv")
