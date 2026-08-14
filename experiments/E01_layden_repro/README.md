# E01 — Layden reproduction

**Question:** does our quench kernel reproduce the spectral-gap speedup of
Layden et al. (Nature 619, 282 (2023), Fig. 2) on random spin glasses?

**Setup:** fully-connected Ising instances with random fields (the paper's
average-case ensemble), n = 5–10, T ∈ {0.1, 0.3, 1.0}. Kernels: uniform,
single-flip, quench (exact `e^{-iHt}`, not Trotter). Metric: exact spectral
gap, geometric-mean over instances, fit ⟨δ⟩ ∝ 2^{−kn}. Fitting only
n = 8–10 overestimates k; the paper's lever arm is n = 3–10.

**Success:** quench k is clearly smaller than uniform/local at low T
(paper: ≈ 0.29 vs ≈ 1.0 at T = 0.1), and the advantage shrinks as T
grows. This gates all further work — if we cannot reproduce the known
result, stop and debug.

```bash
uv run python -m experiments.E01_layden_repro.run          # full figure
uv run python -m experiments.E01_layden_repro.run --quick  # smoke
```

Writes `results/E01/summary.md` (tracked), plus gitignored `gaps.csv`,
`meta.json`, and `gap_vs_n.png`.
