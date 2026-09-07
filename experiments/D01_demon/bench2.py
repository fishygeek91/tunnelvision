"""D01 gated benchmark: 4 chains from random inits, split-Rhat gate, ESS per eval.
Quench proposals are drawn from its exact proposal matrix (identical kernel, fast)."""
from __future__ import annotations

import sys
import time

import numpy as np

from tunnelvision.bits import all_binary_states
from tunnelvision.diagnostics import integrated_autocorrelation_time as iat
from tunnelvision.diagnostics import rhat
from tunnelvision.dynamics import DemonKernel
from tunnelvision.dynamics.mtm import ShellMTM
from tunnelvision.engine import MetropolisEngine
from tunnelvision.kernels.base import Kernel
from tunnelvision.kernels.classical import AddDeleteSwap, SingleFlip, UniformFlip
from tunnelvision.kernels.quantum import QuenchKernel
from tunnelvision.targets.ising import random_spin_glass


class TabulatedQuench(Kernel):
    name = "quench(exact Q)"

    def __init__(self, tgt):
        self.Q = QuenchKernel(tgt.h, tgt.J, evolution="exact").proposal_matrix(tgt.n_vars)
        self.n = tgt.n_vars
        self.pw = 1 << np.arange(self.n)
        self.cdf = np.cumsum(self.Q, axis=1)

    def propose(self, x, rng):
        xi = int((x * self.pw).sum())
        yi = min(int(np.searchsorted(self.cdf[xi], rng.random())), (1 << self.n) - 1)
        return ((yi >> np.arange(self.n)) & 1).astype(np.uint8), 0.0, 0.0

    def proposal_matrix(self, n):
        return self.Q


n = 10
T = float(sys.argv[1])
steps = int(sys.argv[2])
C = 4
print(f"n={n} T={T} steps={steps} chains={C}")
print(f"{'arm':30s} {'seed':>4s} {'acc':>5s} {'Rhat_E':>6s} {'tauE':>7s} {'ev/step':>7s} "
      f"{'ESS_E/eval':>10s} {'ESS_E/sec':>9s} {'|dM|':>6s}")
for seed in [1, 2, 3]:
    tgt = random_spin_glass(n, seed=seed, temperature=T, random_fields=True)
    S = all_binary_states(n)
    pi = tgt.enumerate_exact()
    M_exact = pi @ (2.0 * S.mean(1) - 1.0)
    arms = {
        "single-flip": lambda: ("mh", SingleFlip()),
        "uniform": lambda: ("mh", UniformFlip()),
        "add-delete-swap": lambda: ("mh", AddDeleteSwap()),
        "demon(L=4,Td=8)": lambda: ("mh", DemonKernel(tgt, sweep_len=4, demon_temperature=8.0)),
        "quench(exact Q)": lambda: ("mh", TabulatedQuench(tgt)),
        "mtm(K=8,greedy)": lambda: ("mtm", ShellMTM(tgt, n_try=8, eps=np.inf, a=0.0)),
        "mtm(K=8,shell3,a=1)": lambda: ("mtm", ShellMTM(tgt, n_try=8, eps=3.0 / T, a=1.0)),
        "mtm(K=8,shell3,a=.5)": lambda: ("mtm", ShellMTM(tgt, n_try=8, eps=3.0 / T, a=0.5)),
        "mtm(K=4,shell3,a=.5)": lambda: ("mtm", ShellMTM(tgt, n_try=4, eps=3.0 / T, a=0.5)),
        "mtm(K=8,shell3,a=.5,w<=5)": lambda: (
            "mtm", ShellMTM(tgt, n_try=8, eps=3.0 / T, a=0.5, max_weight=5)),
    }
    for name, mk in arms.items():
        Es, Ms, accs, evals, dt = [], [], [], 0, 0.0
        for c in range(C):
            kind, obj = mk()
            rng0 = np.random.default_rng(1000 * seed + c)
            x0 = rng0.integers(0, 2, n).astype(np.uint8)
            t0 = time.perf_counter()
            if kind == "mh":
                r = MetropolisEngine(tgt, obj).run(steps, x0, seed=10 * seed + c)
                st, lp, acc = r.states, r.log_probs, r.accepted
                evals += steps + getattr(obj, "n_energy_evals", 0)
            else:
                st, lp, acc = obj.run(steps, x0, seed=10 * seed + c)
                evals += obj.n_energy_evals
            dt += time.perf_counter() - t0
            Es.append(-lp)
            Ms.append(2.0 * st.mean(1) - 1.0)
            accs.append(acc.mean())
        Es = np.array(Es)
        Ms = np.array(Ms)
        R = rhat(Es)
        tau = np.mean([iat(e) for e in Es])
        ess = C * steps / tau
        print(f"{name:30s} {seed:4d} {np.mean(accs):5.2f} {R:6.2f} {tau:7.1f} {evals/(C*steps):7.1f} "
              f"{ess/evals:10.2e} {ess/dt:9.1e} {abs(Ms.mean()-M_exact):6.3f}")
