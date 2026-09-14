"""P01 Step 1 — A3 portal-acceptance gate (registered: README.md).

Energy-matched atom pairs (dE <~ T) from shell states at the M06 ladder
temperatures. Library = capped shell enumeration around an equilibrated
state (members only, no completeness). Measure portal-jump acceptance in
a darting chain seeded on the shell. Gate: mean portal acceptance must
reach ~0.2 at ladder temperatures, else the rung dies here.
"""

from __future__ import annotations

import csv
import sys

import numpy as np

from p01_common import (RNG, make_target, equilibrate, energies_batch,
                        energy, THPortal, darting_chain)
from tunnelvision.dynamics.shellexact import ExactShellSampler

NS = [16, 20]
SEEDS = [1, 2]
TFACT = [0.55, 0.35]
RHO_LIST = [0.5, 1.0, 2.0]      # rho = RHO/n
GATE = 0.2


def main(out="p01_gate.csv"):
    rows = []
    for n in NS:
        for seed in SEEDS:
            tgt = make_target(n, seed)
            for tf in TFACT:
                T = tf * np.sqrt(n)
                rng = RNG(9000 + 100 * n + seed + int(tf * 100))
                x = equilibrate(tgt, n, rng, T, 8000)
                ex = ExactShellSampler(tgt.J, tgt.h, eps=2.0, seed=seed,
                                       timeout_s=30.0)
                members, status = ex.enumerate(x, cap=400, timeout_s=30.0)
                atoms = [x] + members
                E = energies_batch(tgt, np.array(atoms))
                span = float(E.max() - E.min())
                for rr in RHO_LIST:
                    rho = rr / n
                    portal = THPortal(atoms, rho)
                    r = darting_chain(tgt, T, x, 4000,
                                      seed=7700 + n + seed,
                                      portal=portal, w_port=0.5)
                    rows.append(dict(
                        n=n, seed=seed, T_factor=tf, T=round(float(T), 3),
                        n_atoms=len(atoms), enum_status=status,
                        atom_E_span=round(span, 3),
                        rho_times_n=rr,
                        acc_portal=round(r["acc_portal"], 4),
                        acc_local=round(r["acc_local"], 4),
                        try_portal=r["try_portal"]))
                    print(rows[-1], flush=True)
                with open(out, "w", newline="") as fh:
                    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
                    w.writeheader()
                    w.writerows(rows)
    best = {}
    for r in rows:
        key = (r["n"], r["seed"], r["T_factor"])
        best[key] = max(best.get(key, 0.0), r["acc_portal"])
    n_pass = sum(v >= GATE for v in best.values())
    print(f"\nGATE: {n_pass}/{len(best)} cells reach acc >= {GATE} "
          f"at best rho; verdict = {'PASS' if n_pass >= len(best) / 2 else 'FAIL'}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "p01_gate.csv")
