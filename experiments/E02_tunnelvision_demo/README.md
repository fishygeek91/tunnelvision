# E02 — TunnelVision demo (the headline)

**Question:** does the quench kernel, driven by the 2-local surrogate, mix
faster than add-delete-swap on real Bayesian variable selection — while
sampling the exact posterior?

**E02a (diabetes, p=10):** exact tier. Enumerate all 1024 models → exact
posterior, exact PIPs, exact spectral gap for every kernel (uniform,
single-flip, add-delete-swap, quench-analytic-surrogate, quench-learned-surrogate).
One headline figure: spectral gap per kernel; one table: ESS/step and ESS/sec.

**E02b (synthetic rho-sweep):** p = 10 (exact) and p = 20–27 (sampled),
predictor correlation rho ∈ {0, 0.3, 0.5, 0.7, 0.9}. Hypothesis: quantum
advantage grows with rho (correlation → competing models → multimodal
posterior → tunneling pays). This curve is the scientific story.

**Ablations:** noisy simulator (Aer depolarizing at hardware-realistic rates)
— does the gap advantage survive noise? Surrogate quality sweep — how bad can
the surrogate be before the advantage dies?

Run: `python -m experiments.E02_tunnelvision_demo.run` → results/E02/summary.md
