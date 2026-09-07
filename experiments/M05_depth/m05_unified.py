"""M05 (unified, revised) — depth curve + PB-native flank on identical states.

Design amendment 2026-09-07, before any valid results: the registered
step-ladder collapses to two depths (at n=20, T=0.1 the single-flip chain
reaches its minimum within ~200 steps), so it cannot resolve a curve. It is
replaced by a TEMPERATURE ladder: for each seed and each T in a grid, burn a
single-flip chain to equilibrium and take the state; E/n is then a controlled,
reproducible depth axis. Everything else (metric, caps-as-censoring, kill
criteria) is unchanged. This file also folds in the M05b PB-native flank so
Exact and CryptoMiniSat race on the SAME states.

(A prior run used chain states corrupted by a uint8 wraparound in 2*x-1; that
same bug affected the m03b/m04 cold-state GENERATORS — the sampler, the shell
encoding, and the M03a/M04 rand controls were always correct. See M05_RESULTS.)
"""

from __future__ import annotations

import csv
import sys
import time

import numpy as np

from tunnelvision.dynamics.shellhash import ShellHashSampler
from tunnelvision.targets.ising import random_spin_glass

from m05_common import approxmc_count, energy_per_n

N = 20
EPS = 2.0
TEMPS = [30.0, 15.0, 8.0, 5.0, 3.5, 2.5, 1.5, 0.5]   # spans T >> T_c(~sqrt(n)=4.5) to deep cold
BURN = 8000
N_SAMP = 3
BUDGET = 300_000
CAP = 120.0            # per-tool wall cap (censoring), seconds


def equilibrate(tgt, n, rng, T, steps):
    x = rng.integers(0, 2, n).astype(np.uint8)
    J, h = tgt.J, tgt.h
    for _ in range(steps):
        i = int(rng.integers(n))
        s_i = 2 * int(x[i]) - 1
        dE = 2 * s_i * (J[i] @ (2 * x.astype(np.int64) - 1)
                        - J[i, i] * s_i + h[i])
        if dE <= 0 or rng.random() < np.exp(-dE / T):
            x[i] ^= 1
    return x


def build_exact(sh, x):
    import exact
    s = exact.Exact()
    n = sh.n
    ynames = [f"y{i}" for i in range(n)]
    for nm in ynames:
        s.addVariable(nm, 0, 1)
    terms = []
    for i in range(n):
        a_i = 2 * int(sh.Jint[i].sum() - sh.Jint[i, i]) - 2 * int(sh.hint[i])
        if a_i:
            terms.append((a_i, f"y{i}"))
    for i in range(n):
        for j in range(i + 1, n):
            if sh.Jint[i, j] == 0:
                continue
            p = f"p{i}_{j}"
            s.addVariable(p, 0, 1)
            s.addConstraint([(1, p), (-1, f"y{i}")], False, 0, True, 0)
            s.addConstraint([(1, p), (-1, f"y{j}")], False, 0, True, 0)
            s.addConstraint([(1, p), (-1, f"y{i}"), (-1, f"y{j}")],
                            True, -1, False, 0)
            terms.append((-4 * int(sh.Jint[i, j]), p))
    e0 = sh.energy_int(x) - sh.C0
    w = round(sh.scale * sh.eps)
    s.addConstraint(terms, True, e0 - w, True, e0 + w)
    diff = [(1 if x[i] == 0 else -1, f"y{i}") for i in range(n)]
    s.addConstraint(diff, True, 1 - int(x.sum()), False, 0)
    return s, ynames


def exact_enum(sh, x, cap):
    s, ynames = build_exact(sh, x)
    t0 = time.perf_counter()
    found, status = 0, "ok"
    while found <= 16:
        left = cap - (time.perf_counter() - t0)
        if left <= 0:
            status = "timeout"
            break
        r = s.runOnce(left)
        if r == "PAUSED":
            continue
        if r == "SAT" or s.hasSolution():
            found += 1
            s.invalidateLastSol(ynames)
        elif r in ("UNSAT", "INCONSISTENT"):
            break
        else:
            status = f"stop:{r}"
            break
    return found, status, round(time.perf_counter() - t0, 2)


def exact_count(sh, x, cap):
    s, ynames = build_exact(sh, x)
    t0 = time.perf_counter()
    try:
        cstat, cnt = s.count(ynames, cap)
        return f"{cstat}:{cnt}", round(time.perf_counter() - t0, 2)
    except Exception as ex:  # noqa: BLE001
        return f"error:{type(ex).__name__}", round(time.perf_counter() - t0, 2)


def main(out="m05_unified.csv"):
    rows = []
    for seed in (1, 2, 3):
        tgt = random_spin_glass(N, seed=seed, temperature=1.0,
                                random_fields=True)
        for T in TEMPS:
            rng = np.random.default_rng(5000 + seed * 100 + int(T * 100))
            x = equilibrate(tgt, N, rng, T, BURN)
            sh = ShellHashSampler(tgt.J, tgt.h, eps=EPS, seed=seed,
                                  conf_budget=BUDGET)
            clauses, _ = sh._shell_cnf(x)
            log2s, count_s, cstat = approxmc_count(
                [list(map(int, c)) for c in clauses],
                [int(v) for v in sh.yvar], CAP)
            # CMS sampler timing
            times, aborts, empties = [], 0, 0
            t0 = time.perf_counter()
            for _ in range(N_SAMP):
                if time.perf_counter() - t0 > CAP:
                    break
                y, info = sh.sample(x)
                times.append(info["seconds"])
                if info.get("aborted"):
                    aborts += 1
                elif y is None:
                    empties += 1
            cms_med = round(float(np.nanmedian(times if times else [np.nan])),
                            3)
            # Exact PB-native
            ef, es, et = exact_enum(sh, x, CAP)
            ecnt, ect = exact_count(sh, x, CAP)
            rows.append(dict(
                seed=seed, T=T, E_per_n=round(energy_per_n(tgt, x), 4),
                amc_log2S=log2s, amc_count_s=count_s, amc_status=cstat,
                cms_med=cms_med, cms_aborts=aborts, cms_empties=empties,
                cms_censored=int(len(times) < N_SAMP),
                exact_enum=ef, exact_enum_status=es, exact_enum_s=et,
                exact_count=ecnt, exact_count_s=ect))
            print(f"s={seed} T={T:>4}: E/n={rows[-1]['E_per_n']:+.3f} "
                  f"log2S={log2s} | CMS med={cms_med}s ab={aborts} "
                  f"cens={rows[-1]['cms_censored']} | Exact enum={ef}"
                  f"[{es}]{et}s count={ecnt}({ect}s)", flush=True)
            with open(out, "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
                w.writeheader()
                w.writerows(rows)
    print("done")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "m05_unified.csv")
