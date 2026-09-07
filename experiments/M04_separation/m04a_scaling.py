"""M04a — cold-shell hardness scaling + ApproxMC certification."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
import time

import numpy as np

from tunnelvision.dynamics.shellhash import ShellHashSampler
from tunnelvision.targets.ising import random_spin_glass

EPS = 2.0
N_SAMP = 8
BUDGET = 300_000
CELL_CAP_S = 900.0
COUNT_TIMEOUT_S = 900.0


def metropolis_cold(tgt, n, rng, T=0.1):
    x = rng.integers(0, 2, n).astype(np.uint8)
    J, h = tgt.J, tgt.h
    for _ in range(250 * n):
        i = int(rng.integers(n))
        s_i = 2 * int(x[i]) - 1
        dE = 2 * s_i * (J[i] @ (2 * x.astype(np.int64) - 1)  # FIXED 2026-09-07: uint8 wraparound (2*x-1 -> 255) corrupted the chain; see M05_RESULTS.md - J[i, i] * s_i + h[i])
        if dE <= 0 or rng.random() < np.exp(-dE / T):
            x[i] ^= 1
    return x


def energy_per_n(tgt, x):
    s = 2 * x.astype(np.float64) - 1
    return float((-(np.triu(tgt.J, 1) * np.outer(s, s)).sum()
                  - tgt.h @ s) / x.size)


def approxmc_count(clauses, yvars, timeout_s):
    """Projected count in a subprocess (ApproxMC can hang on hard CNFs)."""
    payload = json.dumps(dict(clauses=clauses, proj=yvars))
    code = (
        "import json,sys;d=json.load(sys.stdin);import pyapproxmc;"
        "c=pyapproxmc.Counter();[c.add_clause(cl) for cl in d['clauses']];"
        "cells,hashes=c.count(d['proj']);print(cells,hashes)"
    )
    try:
        t0 = time.perf_counter()
        r = subprocess.run([sys.executable, "-c", code], input=payload,
                           capture_output=True, text=True, timeout=timeout_s)
        dt = time.perf_counter() - t0
        cells, hashes = map(int, r.stdout.split())
        log2s = (np.log2(cells) + hashes) if cells > 0 else -1.0
        return round(float(log2s), 2), round(dt, 1), "ok"
    except subprocess.TimeoutExpired:
        return -1.0, timeout_s, "timeout"
    except Exception as ex:  # noqa: BLE001
        return -1.0, -1.0, f"error:{type(ex).__name__}"


def main(out="m04a_scaling.csv"):
    cells = ([(n, sd, "cold") for n in (16, 20, 24, 28, 32) for sd in (1, 2)]
             + [(24, 1, "rand"), (32, 1, "rand")])
    rows = []
    for n, seed, regime in cells:
        tgt = random_spin_glass(n, seed=seed, temperature=1.0,
                                random_fields=True)
        rng = np.random.default_rng(4000 + 100 * n + seed)
        sh = ShellHashSampler(tgt.J, tgt.h, eps=EPS, seed=seed,
                              conf_budget=BUDGET)
        redraws = 0
        while True:
            x = (rng.integers(0, 2, n).astype(np.uint8) if regime == "rand"
                 else metropolis_cold(tgt, n, rng))
            clauses, _ = sh._shell_cnf(x)
            log2s, count_s, count_status = approxmc_count(
                [list(map(int, c)) for c in clauses],
                [int(v) for v in sh.yvar], COUNT_TIMEOUT_S)
            if count_status != "ok" or log2s >= 0 or redraws >= 3:
                break
            redraws += 1                      # certified-empty shell: redraw
        times, aborts, nones = [], 0, 0
        t_cell = time.perf_counter()
        for _ in range(N_SAMP):
            if time.perf_counter() - t_cell > CELL_CAP_S:
                break
            y, info = sh.sample(x)
            times.append(info["seconds"])
            if info.get("aborted"):
                aborts += 1
            elif y is None:
                nones += 1
        arr = np.array(times) if times else np.array([np.nan])
        rows.append(dict(n=n, seed=seed, regime=regime,
                         E_per_n=round(energy_per_n(tgt, x), 4),
                         redraws=redraws, log2S=log2s,
                         count_s=count_s, count_status=count_status,
                         samples=len(times), aborts=aborts, empties=nones,
                         t_med=round(float(np.nanmedian(arr)), 3),
                         t_p90=round(float(np.nanquantile(arr, 0.9)), 3),
                         censored=int(len(times) < N_SAMP)))
        print(f"n={n} s={seed} {regime}: log2S={log2s} ({count_status},"
              f" {count_s}s) med={rows[-1]['t_med']}s "
              f"p90={rows[-1]['t_p90']}s aborts={aborts} "
              f"done={len(times)}/{N_SAMP}", flush=True)
        with open(out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    print("done")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "m04a_scaling.csv")
