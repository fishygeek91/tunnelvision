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

**E02b (ρ-sweep, exact tier):** p = 10, predictor correlation
ρ ∈ {0, 0.3, 0.5, 0.7, 0.9}. Hypothesis: quantum advantage grows with ρ.
Four independent chains and split-R̂ catch stuck modes at high ρ.
Sampled p = 20–27 is a later session.

**Ablations (exact tier):** depolarizing-noise sweep (global λ, Aer
check at p=6) and learned-surrogate corruption sweep. Question: where
does the remaining quench/ADS gap die, and does PIP error stay at
Monte Carlo error while it dies?

```bash
uv run python -m experiments.E02_tunnelvision_demo.run              # diabetes p=10
uv run python -m experiments.E02_tunnelvision_demo.run --quick      # p=5 synthetic smoke
uv run python -m experiments.E02_tunnelvision_demo.run_rho_sweep    # ρ-sweep p=10
uv run python -m experiments.E02_tunnelvision_demo.run_rho_sweep --quick
uv run python -m experiments.E02_tunnelvision_demo.run_ablations    # noise + surrogate sweeps
uv run python -m experiments.E02_tunnelvision_demo.run_ablations --quick
```

Writes `results/E02/summary.md` (tracked on a full run), plus gitignored
`scoreboard.csv`, `scoreboard.png`, `meta.json`, and the two
surrogate-vs-logp scatter plots. The ρ-sweep writes
`results/E02/rho_sweep/summary.md` and `gap_vs_rho.png`. Ablations write
`results/E02/ablations/summary.md`. `--quick` writes to
`results/E02_quick/` and is not the gate.
