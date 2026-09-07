"""M05 shared pieces: depth ladders, ApproxMC subprocess count."""

from __future__ import annotations

import json
import subprocess
import sys
import time

import numpy as np


def metropolis_chain(tgt, n, rng, snapshots, T=0.1):
    """Snapshots of a T=0.1 single-flip chain at given step counts."""
    x = rng.integers(0, 2, n).astype(np.uint8)
    out = {}
    J, h = tgt.J, tgt.h
    steps = 0
    for target in sorted(snapshots):
        while steps < target:
            i = int(rng.integers(n))
            s_i = 2 * int(x[i]) - 1
            dE = 2 * s_i * (J[i] @ (2 * x.astype(np.int64) - 1)  # FIXED 2026-09-07: uint8 wraparound (2*x-1 -> 255) corrupted the chain; see M05_RESULTS.md - J[i, i] * s_i + h[i])
            if dE <= 0 or rng.random() < np.exp(-dE / T):
                x[i] ^= 1
            steps += 1
        out[target] = x.copy()
    return out


def greedy_descent(tgt, x0):
    """Steepest descent to a local minimum."""
    x = x0.copy()
    J, h = tgt.J, tgt.h
    n = x.size
    while True:
        s = 2 * x.astype(np.float64) - 1
        dE = 2 * s * (J @ s - np.diag(J) * s + h)
        i = int(np.argmin(dE))
        if dE[i] >= 0:
            return x
        x[i] ^= 1


def energy_per_n(tgt, x):
    s = 2 * x.astype(np.float64) - 1
    return float((-(np.triu(tgt.J, 1) * np.outer(s, s)).sum()
                  - tgt.h @ s) / x.size)


def approxmc_count(clauses, yvars, timeout_s, python=sys.executable):
    payload = json.dumps(dict(clauses=clauses, proj=yvars))
    code = ("import json,sys;d=json.load(sys.stdin);import pyapproxmc;"
            "c=pyapproxmc.Counter();[c.add_clause(cl) for cl in d['clauses']];"
            "cells,hashes=c.count(d['proj']);print(cells,hashes)")
    try:
        t0 = time.perf_counter()
        r = subprocess.run([python, "-c", code], input=payload,
                           capture_output=True, text=True, timeout=timeout_s)
        dt = time.perf_counter() - t0
        cells, hashes = map(int, r.stdout.split())
        log2s = (np.log2(cells) + hashes) if cells > 0 else -1.0
        return round(float(log2s), 2), round(dt, 1), "ok"
    except subprocess.TimeoutExpired:
        return -2.0, timeout_s, "timeout"
    except Exception as ex:  # noqa: BLE001
        return -2.0, -1.0, f"error:{type(ex).__name__}"
