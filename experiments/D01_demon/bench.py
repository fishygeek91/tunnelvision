"""D01 sampled-tier benchmark: ESS per target-eval and per second, n=10 Ising.

Exactness gate: mean-magnetization error vs enumeration for every arm.
"""

from __future__ import annotations

import sys
import time

import numpy as np

from tunnelvision.bits import all_binary_states
from tunnelvision.diagnostics import integrated_autocorrelation_time as iat
from tunnelvision.dynamics import DemonKernel
from tunnelvision.dynamics.mtm import ShellMTM
from tunnelvision.engine import MetropolisEngine
from tunnelvision.kernels.classical import AddDeleteSwap, SingleFlip, UniformFlip
from tunnelvision.kernels.quantum import QuenchKernel
from tunnelvision.targets.ising import random_spin_glass

n = 10
T = float(sys.argv[1]) if len(sys.argv) > 1 else 0.3
steps = int(sys.argv[2]) if len(sys.argv) > 2 else 100_000
seeds = [1, 2, 3]

print(f"n={n} T={T} steps={steps}")
hdr = f"{'arm':34s} {'seed':>4s} {'acc':>5s} {'tauE':>8s} {'tauM':>8s} {'evals/step':>10s} {'ESS_E/eval':>10s} {'ESS_E/sec':>10s} {'|dM|':>6s}"
print(hdr)
for seed in seeds:
    tgt = random_spin_glass(n, seed=seed, temperature=T, random_fields=True)
    S = all_binary_states(n)
    pi = tgt.enumerate_exact()
    M_exact = pi @ (2.0 * S.mean(1) - 1.0)
    arms = {
        "single-flip": ("mh", SingleFlip()),
        "uniform": ("mh", UniformFlip()),
        "add-delete-swap": ("mh", AddDeleteSwap()),
        "demon(L=4,Td=8)": ("mh", DemonKernel(tgt, sweep_len=4, demon_temperature=8.0)),
        "mtm(K=8,greedy a=0)": ("mtm", ShellMTM(tgt, n_try=8, eps=np.inf, a=0.0)),
        "mtm(K=8,shell eps=3,a=1)": ("mtm", ShellMTM(tgt, n_try=8, eps=3.0 / T, a=1.0)),
        "mtm(K=8,shell eps=3,a=0.5)": ("mtm", ShellMTM(tgt, n_try=8, eps=3.0 / T, a=0.5)),
        "mtm(K=32,shell eps=3,a=1)": ("mtm", ShellMTM(tgt, n_try=32, eps=3.0 / T, a=1.0)),
        "mtm(K=8,shell eps=3,a=1,w<=3)": ("mtm", ShellMTM(tgt, n_try=8, eps=3.0 / T, a=1.0, max_weight=3)),
        "quench(statevector)": ("mh", QuenchKernel(tgt.h, tgt.J, evolution="exact")),
    }
    for name, (kind, obj) in arms.items():
        ns = steps if name != "quench(statevector)" else min(steps, 5000)
        x0 = np.zeros(n, dtype=np.uint8)
        t0 = time.perf_counter()
        if kind == "mh":
            res = MetropolisEngine(tgt, obj).run(ns, x0, seed=seed)
            st, lp, acc = res.states, res.log_probs, res.accepted
            evals = ns * (1 if not hasattr(obj, "n_energy_evals") else max(1, obj.n_energy_evals / ns))
            if hasattr(obj, "n_energy_evals"):
                evals = obj.n_energy_evals + ns  # sweep evals + the engine's log_prob(y)
            else:
                evals = ns
        else:
            st, lp, acc = obj.run(ns, x0, seed=seed)
            evals = obj.n_energy_evals
        dt = time.perf_counter() - t0
        E = -lp
        Mg = 2.0 * st.mean(1) - 1.0
        tE, tM = iat(E), iat(Mg)
        essE = ns / tE
        print(
            f"{name:34s} {seed:4d} {acc.mean():5.2f} {tE:8.1f} {tM:8.1f} {evals/ns:10.1f} "
            f"{essE/evals:10.2e} {essE/dt:10.2e} {abs(Mg.mean()-M_exact):6.3f}"
        )
