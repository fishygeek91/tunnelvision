"""M04b — real UniGen3 on the same cold shells (tool-choice defense)."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
import time

import numpy as np

from tunnelvision.dynamics.shellhash import ShellHashSampler
from tunnelvision.targets.ising import random_spin_glass

from m04a_scaling import metropolis_cold  # same burn-in recipe

EPS = 2.0
N_SAMP = 5
TIMEOUT_S = 900.0


def unigen_sample(clauses, yvars, num, timeout_s):
    payload = json.dumps(dict(clauses=clauses, proj=yvars, num=num))
    code = (
        "import json,sys,time;d=json.load(sys.stdin);import pyunigen;"
        "s=pyunigen.Sampler();[s.add_clause(cl) for cl in d['clauses']];"
        "t0=time.perf_counter();"
        "out=s.sample(num=d['num'],sampling_set=d['proj']);"
        "print(time.perf_counter()-t0, len(out[2]))"
    )
    try:
        r = subprocess.run([sys.executable, "-c", code], input=payload,
                           capture_output=True, text=True, timeout=timeout_s)
        dt, got = r.stdout.split()
        return float(dt), int(got), "ok"
    except subprocess.TimeoutExpired:
        return timeout_s, 0, "timeout"
    except Exception as ex:  # noqa: BLE001
        return -1.0, 0, f"error:{type(ex).__name__}"


def main(out="m04b_unigen.csv"):
    rows = []
    for n in (16, 20, 24):
        for seed in (1, 2):
            for regime in (["cold"] if n > 16 else ["cold", "rand"]):
                tgt = random_spin_glass(n, seed=seed, temperature=1.0,
                                        random_fields=True)
                rng = np.random.default_rng(4000 + 100 * n + seed)
                x = (rng.integers(0, 2, n).astype(np.uint8)
                     if regime == "rand" else metropolis_cold(tgt, n, rng))
                sh = ShellHashSampler(tgt.J, tgt.h, eps=EPS, seed=seed)
                clauses, _ = sh._shell_cnf(x)
                dt, got, status = unigen_sample(
                    [list(map(int, c)) for c in clauses],
                    [int(v) for v in sh.yvar], N_SAMP, TIMEOUT_S)
                per = dt / got if got else float("nan")
                rows.append(dict(n=n, seed=seed, regime=regime,
                                 status=status, total_s=round(dt, 2),
                                 samples=got,
                                 s_per_sample=round(per, 3) if got else -1))
                print(f"n={n} s={seed} {regime}: unigen {status} "
                      f"{got}/{N_SAMP} in {dt:.1f}s "
                      f"({per if got else float('nan'):.2f} s/sample)",
                      flush=True)
                with open(out, "w", newline="") as fh:
                    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
                    w.writeheader()
                    w.writerows(rows)
    print("done")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "m04b_unigen.csv")
