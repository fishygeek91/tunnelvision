# E02 — TunnelVision demo (the headline)

**Question:** does the quench kernel, driven by the 2-local surrogate, mix
faster than add-delete-swap on real Bayesian variable selection — while
sampling the exact posterior?

**E02a (this PR, exact tier):** diabetes p=10, or a correlated synthetic
of the same size. Enumerate all 2^p models → exact posterior, exact PIPs,
exact spectral gap for every kernel (uniform, single-flip, add-delete-swap,
quench-analytic, quench-learned). One table: gap, ESS/step, ESS/sec.
The per-second column exists so a bigger gap that costs 100× more
wall-clock cannot pose as a win.

**E02b (later):** p = 10 (exact) and p = 20–27 (sampled), predictor
correlation ρ ∈ {0, 0.3, 0.5, 0.7, 0.9}. Hypothesis: quantum advantage
grows with ρ.

```bash
uv run python -m experiments.E02_tunnelvision_demo.run          # diabetes p=10
uv run python -m experiments.E02_tunnelvision_demo.run --quick  # p=5 synthetic smoke
```

Writes `results/E02/summary.md` (tracked on a full run), plus gitignored
`scoreboard.csv`, `meta.json`, and the two surrogate-vs-logp scatter plots.
`--quick` writes to `results/E02_quick/` and is not the gate.
