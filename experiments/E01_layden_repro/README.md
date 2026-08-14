# E01 — Layden reproduction

**Question:** does our quench kernel reproduce the spectral-gap speedup of
Layden et al. (Nature 619, 282 (2023), Fig. 3) on random spin glasses?

**Setup:** all-to-all and 2D Ising instances, n = 8–12, low temperature
(T in {0.1, 0.3, 1.0} × typical coupling scale). Kernels: uniform, single-flip,
quench (statevector). Metric: exact spectral gap vs. n, per temperature.

**Success:** quench gap decays with a visibly better exponent than local/uniform,
matching the qualitative Layden scaling. This gates all further work — if we
can't reproduce the known result, stop and debug.

Run: `python -m experiments.E01_layden_repro.run` → results/E01/summary.md
