"""M02b — exact pair-space spectral gaps: does ICM actually buy mixing?

For n=8 cells (seeds 1-3, T in {1, .3, .1}), the exact gap of the
two-replica chain P = w*P_ICM + (1-w)*P_SF against the w=0 single-flip
baseline. The tensor identity gap(P_SF-pair) = gap(P_SF-single)/2 is
certified in tests/test_dynamics.py, so pair gaps here can be read
against M01 single-chain gaps via that factor of 2.

Cost model for per-eval comparison: SF move = 1 target evaluation, ICM
move = 2 (one delta-E per replica); mixture at w: evals/step = 1 + w.
"""

from __future__ import annotations

import csv
import sys

import numpy as np

from tunnelvision.dynamics.icm import HoudayerICM, sparse_pair_gap
from tunnelvision.targets.ising import random_spin_glass

VARIANTS = [
    ("baseline", None, None, 0.0),
    ("faithful", 0.0, None, 0.5),
    ("theta=1.0", 1.0, None, 0.5),
    ("cap=3", 0.0, 3, 0.5),
    ("cap=5", 0.0, 5, 0.5),
]


def main(out_path: str, n: int = 8):
    rows = []
    for seed in [1, 2, 3]:
        for T in [1.0, 0.3, 0.1]:
            tgt = random_spin_glass(n, seed=seed, temperature=T, random_fields=True)
            base_gap = None
            for name, theta, cap, w in VARIANTS:
                icm = HoudayerICM(tgt, theta=(theta or 0.0), size_cap=cap)
                P, pi_pair = icm.pair_transition_matrix(n, w_icm=w)
                gap = sparse_pair_gap(P, pi_pair)
                evals = 1.0 + w
                if name == "baseline":
                    base_gap = gap
                ratio = gap / base_gap if base_gap else float("nan")
                ratio_ev = (gap / evals) / base_gap if base_gap else float("nan")
                rows.append(dict(n=n, seed=seed, T=T, variant=name, w_icm=w,
                                 gap=gap, vs_baseline=ratio,
                                 vs_baseline_per_eval=ratio_ev))
                print(f"n={n} s={seed} T={T:4g} {name:10s} gap={gap:.4e} "
                      f"x{ratio:6.3f} (per-eval x{ratio_ev:6.3f})", flush=True)
                with open(out_path, "w", newline="") as fh:
                    wtr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
                    wtr.writeheader()
                    wtr.writerows(rows)
    print(f"wrote {len(rows)} rows -> {out_path}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "m02b_icm_gap.csv")
