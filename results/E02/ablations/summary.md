# E02 ablations — noise and surrogate quality

Exact-tier sweeps on the same spike-and-slab posteriors as E02a/b.
Accept/reject uses the exact g-prior; noise and surrogate quality
may only change the proposal, never the target.

## Provenance

- git commit: `72542edb4ae606a2d11f200c25b197e8e16c2aa0`
- started (UTC): 2026-08-14T20:40:08.335872+00:00
- finished (UTC): 2026-08-14T20:43:01.719238+00:00
- package versions: numpy=2.4.6, scipy=1.17.1, scikit-learn=1.9.0, qiskit=2.5.2, qiskit-aer=0.17.2, tunnelvision=0.0.1
- λ grid: 0.0, 0.001, 0.005, 0.02, 0.1
- σ grid: 0.0, 0.25, 0.5, 1.0, 2.0, 4.0
- quench: exact, grid 4×4
- chain: 4000 steps, burn-in 500

## Ablation A — depolarizing noise

Global channel `Q_λ = (1−λ) Q + λ/2^p 11ᵀ` on the cached quench
proposal. Unital, so the kernel stays symmetric. ADS does not
see λ — it is the flat classical baseline.

| dataset | kernel | λ | gap | accept | PIP err | ESS/step (|γ|) |
| --- | --- | --- | --- | --- | --- | --- |
| diabetes | add-delete-swap | 0 | 0.0225 | 0.095 | 0.126 | 0.018 |
| diabetes | quench-analytic-depol-0 | 0 | 0.006838 | 0.036 | 0.073 | 0.004 |
| diabetes | quench-analytic-depol-0.001 | 0.001 | 0.006835 | 0.038 | 0.094 | 0.006 |
| diabetes | quench-analytic-depol-0.005 | 0.005 | 0.006822 | 0.042 | 0.234 | 0.009 |
| diabetes | quench-analytic-depol-0.02 | 0.02 | 0.006774 | 0.032 | 0.064 | 0.005 |
| diabetes | quench-analytic-depol-0.1 | 0.1 | 0.006518 | 0.038 | 0.153 | 0.008 |
| diabetes | quench-learned-depol-0 | 0 | 0.009002 | 0.069 | 0.072 | 0.020 |
| diabetes | quench-learned-depol-0.001 | 0.001 | 0.008997 | 0.067 | 0.099 | 0.010 |
| diabetes | quench-learned-depol-0.005 | 0.005 | 0.008977 | 0.065 | 0.129 | 0.015 |
| diabetes | quench-learned-depol-0.02 | 0.02 | 0.008903 | 0.069 | 0.176 | 0.035 |
| diabetes | quench-learned-depol-0.1 | 0.1 | 0.008509 | 0.052 | 0.203 | 0.016 |
| synthetic-rho-0.9 | add-delete-swap | 0 | 0.06944 | 0.152 | 0.020 | 0.054 |
| synthetic-rho-0.9 | quench-analytic-depol-0 | 0 | 0.0133 | 0.207 | 0.074 | 0.010 |
| synthetic-rho-0.9 | quench-analytic-depol-0.001 | 0.001 | 0.01329 | 0.196 | 0.071 | 0.007 |
| synthetic-rho-0.9 | quench-analytic-depol-0.005 | 0.005 | 0.01326 | 0.172 | 0.051 | 0.006 |
| synthetic-rho-0.9 | quench-analytic-depol-0.02 | 0.02 | 0.01314 | 0.204 | 0.096 | 0.008 |
| synthetic-rho-0.9 | quench-analytic-depol-0.1 | 0.1 | 0.01247 | 0.169 | 0.211 | 0.010 |
| synthetic-rho-0.9 | quench-learned-depol-0 | 0 | 0.02942 | 0.164 | 0.050 | 0.041 |
| synthetic-rho-0.9 | quench-learned-depol-0.001 | 0.001 | 0.0294 | 0.173 | 0.087 | 0.045 |
| synthetic-rho-0.9 | quench-learned-depol-0.005 | 0.005 | 0.02929 | 0.176 | 0.071 | 0.043 |
| synthetic-rho-0.9 | quench-learned-depol-0.02 | 0.02 | 0.02891 | 0.185 | 0.053 | 0.028 |
| synthetic-rho-0.9 | quench-learned-depol-0.1 | 0.1 | 0.02684 | 0.149 | 0.048 | 0.017 |

![gap vs lambda](gap_vs_lambda.png)

Add-delete-swap is the flat baseline. The question is not whether
quench beats it — E02a/b already said no — but where the remaining
gap ratio collapses as proposals are heated toward uniform.

**diabetes** (ADS gap 0.0225):
- `quench-analytic` / ADS is 0.30× at λ=0 and 0.29× at λ=0.1.
- `quench-learned` / ADS is 0.40× at λ=0 and 0.38× at λ=0.1.

**synthetic-rho-0.9** (ADS gap 0.06944):
- `quench-analytic` / ADS is 0.19× at λ=0 and 0.18× at λ=0.1.
- `quench-learned` / ADS is 0.42× at λ=0 and 0.39× at λ=0.1.

λ ≤ 0.1 barely moves the ratio. The gap was already losing and it
stays losing; this slice does not find a noise level that helps.
The Aer panel below says why a larger λ is the physically relevant
one: a Trotter circuit with t ~ 6–15 has enough gates that
per-gate p = 0.005 already maps to λ̂ ≈ 0.65.

## Ablation B — surrogate quality

Relative Gaussian noise on the learned (h, J), then re-pin
Frobenius so α stays 1. Shape is destroyed; scale is not.

| dataset | σ | Spearman | ground rank | gap | PIP err | ESS/step (|γ|) |
| --- | --- | --- | --- | --- | --- | --- |
| diabetes | 0 | 0.983 | 11 | 0.009002 | 0.072 | 0.020 |
| diabetes | 0.25 | 0.963 | 11 | 0.01111 | 0.198 | 0.035 |
| diabetes | 0.5 | 0.919 | 11 | 0.009584 | 0.146 | 0.029 |
| diabetes | 1 | 0.808 | 23 | 0.008173 | 0.346 | 0.040 |
| diabetes | 2 | 0.613 | 312 | 0.006548 | 0.056 | 0.009 |
| diabetes | 4 | 0.414 | 312 | 0.003638 | 0.316 | 0.007 |
| synthetic-rho-0.9 | 0 | 0.994 | 1 | 0.02942 | 0.050 | 0.041 |
| synthetic-rho-0.9 | 0.25 | 0.965 | 16 | 0.02267 | 0.080 | 0.036 |
| synthetic-rho-0.9 | 0.5 | 0.911 | 92 | 0.02458 | 0.054 | 0.044 |
| synthetic-rho-0.9 | 1 | 0.780 | 92 | 0.02882 | 0.049 | 0.039 |
| synthetic-rho-0.9 | 2 | 0.545 | 112 | 0.02671 | 0.064 | 0.023 |
| synthetic-rho-0.9 | 4 | 0.314 | 265 | 0.00895 | 0.070 | 0.015 |

![gap vs sigma](gap_vs_sigma.png)

PIP error is the wall check. Destroying the surrogate may collapse
the gap; it must not move the sampled posterior beyond Monte Carlo
error. A PIP-error column that tracks σ is a P0 bug.

- **diabetes**: Spearman 0.983 → 0.414; gap 0.0090 → 0.0036. PIP err
  wanders in [0.056, 0.346] and does **not** track σ (0.346 at σ=1,
  0.056 at σ=2). That is a short-chain / small-gap Monte Carlo
  column (1/δ ~ 100–300 steps, 3500 kept draws), not a wall breach.
- **synthetic-rho-0.9**: Spearman 0.994 → 0.314; gap 0.0294 → 0.0090
  at σ=4. PIP err stays in [0.049, 0.080] across the whole grid —
  the cleaner wall check, because the gap is larger.

The learned landscape can be wrecked (Spearman 0.99 → 0.3, ground
state falls out of the top 200) and the sampler still targets the
exact posterior. Speed dies; the invariant does not.

## Aer validation — per-gate vs global λ

Trotter circuit, density-matrix Aer, p=6, reduced (γ, t) grid.
Noiseless Aer matches the numpy Trotter Q to 1e-15 (after the
s = −Z field-sign fix). Once the gates are noisy, TV(Aer, Q_λ̂)
stays well below TV(Aer, Q_ideal), so the p=10 analytic sweep is
a fair proxy — and a *colder* one than a real Trotter circuit.

| gate p | λ̂ | TV(Aer, ideal) | TV(Aer, Q_λ̂) | wall (s) |
| --- | --- | --- | --- | --- |
| 0 | 0 | 1.066e-15 | 1.064e-15 | 3.1 |
| 0.005 | 0.6478 | 0.2692 | 0.07137 | 11.7 |
| 0.02 | 0.9619 | 0.4005 | 0.02312 | 11.6 |

![aer validation](aer_validation.png)

## Notes

- Add-delete-swap is the baseline that matters. Beating uniform proves nothing.
- Headline comparison is the exact gap. ESS/sec is tabulated-Q, not QPU time.
- Sampled p=20–27 is still open; these ablations stay at the enumerable tier.

