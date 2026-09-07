"""M05b — the PB-native flank: Exact solver on raw energy windows.

Same ladder states as M05a. For each: pose the shell natively as PB
(binary y_i, linearized products, one two-sided window constraint),
then time (a) feasibility, (b) enumeration of up to 16 solutions
projected on y, (c) projected count — each with native 300 s timeouts.
CMS reference numbers come from M05a on identical shells.
"""

from __future__ import annotations

import csv
import sys
import time

import numpy as np

from tunnelvision.dynamics.shellhash import ShellHashSampler
from tunnelvision.targets.ising import random_spin_glass

from m05_common import energy_per_n, greedy_descent, metropolis_chain

N = 20
EPS = 2.0
SNAPSHOTS = [0, 200, 1000, 5000, 20000]
TIMEOUT = 300.0


def build_exact(sh, x):
    """Native PB model of the shell around x. Returns (solver, yvar names)."""
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
            # p = y_i AND y_j, standard linearization
            s.addConstraint([(1, p), (-1, f"y{i}")], False, 0, True, 0)
            s.addConstraint([(1, p), (-1, f"y{j}")], False, 0, True, 0)
            s.addConstraint([(1, p), (-1, f"y{i}"), (-1, f"y{j}")],
                            True, -1, False, 0)
            terms.append((-4 * int(sh.Jint[i, j]), p))
    e0 = sh.energy_int(x) - sh.C0
    w = round(sh.scale * sh.eps)
    s.addConstraint(terms, True, e0 - w, True, e0 + w)
    # y != x
    diff = [(1 if x[i] == 0 else -1, f"y{i}") for i in range(n)]
    s.addConstraint(diff, True, 1 - int(x.sum()), False, 0)
    return s, ynames


def main(out="m05b_exact.csv"):
    rows = []
    for seed in (1, 2, 3):
        tgt = random_spin_glass(N, seed=seed, temperature=1.0,
                                random_fields=True)
        rng = np.random.default_rng(5000 + seed)  # same stream as M05a
        snaps = metropolis_chain(tgt, N, rng, SNAPSHOTS)
        states = [(str(k), v) for k, v in snaps.items()]
        states.append(("greedy", greedy_descent(tgt, snaps[SNAPSHOTS[-1]])))
        for label, x in states:
            sh = ShellHashSampler(tgt.J, tgt.h, eps=EPS, seed=seed)
            # (a) feasibility
            s, ynames = build_exact(sh, x)
            t0 = time.perf_counter()
            res = s.runOnce(TIMEOUT)
            t_feas = time.perf_counter() - t0
            # (b) enumerate up to 16 projected solutions
            s2, ynames = build_exact(sh, x)
            t0 = time.perf_counter()
            found = 0
            enum_status = "ok"
            while found <= 16:
                left = TIMEOUT - (time.perf_counter() - t0)
                if left <= 0:
                    enum_status = "timeout"
                    break
                r = s2.runOnce(left)
                if r == "SAT" or s2.hasSolution():
                    found += 1
                    s2.invalidateLastSol(ynames)
                elif r in ("UNSAT", "INCONSISTENT"):
                    break
                else:
                    enum_status = f"stop:{r}"
                    break
            t_enum = time.perf_counter() - t0
            # (c) projected count
            s3, ynames = build_exact(sh, x)
            t0 = time.perf_counter()
            try:
                cnt, cstat_i = s3.count(ynames, TIMEOUT)
                t_count = time.perf_counter() - t0
                cnt_str = str(cnt)
            except Exception as ex:  # noqa: BLE001
                t_count = time.perf_counter() - t0
                cnt_str = f"error:{type(ex).__name__}"
            rows.append(dict(seed=seed, ladder=label,
                             E_per_n=round(energy_per_n(tgt, x), 4),
                             feas_res=res, t_feas=round(t_feas, 2),
                             enum_found=found, enum_status=enum_status,
                             t_enum=round(t_enum, 2),
                             count=cnt_str, t_count=round(t_count, 2)))
            print(f"s={seed} {label:>6s}: feas={res}({t_feas:.1f}s) "
                  f"enum={found}[{enum_status}]({t_enum:.1f}s) "
                  f"count={cnt_str}({t_count:.1f}s)", flush=True)
            with open(out, "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
                w.writeheader()
                w.writerows(rows)
    print("done")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "m05b_exact.csv")
