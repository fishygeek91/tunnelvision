# E03 — Maxwell's Daemon (phase 1)

Fixed depolarizing rungs composed by random-scan, exact tier.
Accept/reject uses the exact g-prior on every rung. Parallel
tempering (the third arm) is deferred.

## Provenance

- git commit: `73f9b69c6d3aa44ef6187eb8231716bd1eee9679`
- started (UTC): 2026-08-14T20:47:01.784761+00:00
- finished (UTC): 2026-08-14T20:47:30.097650+00:00
- package versions: numpy=2.4.6, scipy=1.17.1, scikit-learn=1.9.0, qiskit=2.5.2, qiskit-aer=0.17.2, tunnelvision=0.0.1
- λ grid: 0.0, 0.001, 0.005, 0.02, 0.1
- quench: exact, grid 4×4
- chain: 4000 steps, burn-in 500
- T_eff samples: 4000

## Effective proposal temperature

x ~ exact π, y ~ Q_λ(·|x), E is the *surrogate* energy. T_eff
is the Boltzmann temperature on that landscape whose mean
energy matches ⟨E(y)⟩. T_π is the same invert for ⟨E(x)⟩.
If T_eff rises with λ, noise is heating the proposal.
It does: diabetes 1.86 → 2.28, ρ=0.9  1.70 → 2.04, and
⟨|ΔE|⟩ moves with it. T_eff is already ≫ T_π at λ=0 —
the noiseless quench is a hot proposal relative to where
the exact posterior sits on the surrogate landscape. λ ≤ 0.1
is a modest further heat, consistent with the E02 ablation.

| dataset | λ | T_eff | T_π | ⟨|ΔE|⟩ | ⟨E(y)⟩ |
| --- | --- | --- | --- | --- | --- |
| diabetes | 0 | 1.86 | 0.292 | 1.77 | -2.95 |
| diabetes | 0.001 | 1.93 | 0.292 | 1.83 | -2.9 |
| diabetes | 0.005 | 1.9 | 0.292 | 1.81 | -2.92 |
| diabetes | 0.02 | 2.04 | 0.292 | 1.9 | -2.82 |
| diabetes | 0.1 | 2.28 | 0.292 | 2.04 | -2.67 |
| synthetic-rho-0.9 | 0 | 1.7 | 0.199 | 2.1 | -4.42 |
| synthetic-rho-0.9 | 0.001 | 1.71 | 0.199 | 2.12 | -4.39 |
| synthetic-rho-0.9 | 0.005 | 1.72 | 0.199 | 2.13 | -4.38 |
| synthetic-rho-0.9 | 0.02 | 1.75 | 0.199 | 2.18 | -4.33 |
| synthetic-rho-0.9 | 0.1 | 2.04 | 0.199 | 2.55 | -3.95 |

![teff vs lambda](teff_vs_lambda.png)

## Rung vs ladder vs ADS

Gap is exact. ESS chains sample the same Q the gap used.

| dataset | kernel | gap | accept | PIP err | ESS/step (|γ|) | ESS/sec (|γ|) |
| --- | --- | --- | --- | --- | --- | --- |
| diabetes | add-delete-swap | 0.0225 | 0.095 | 0.126 | 0.018 | 298.99 |
| diabetes | quench-learned-depol-0 | 0.009002 | 0.069 | 0.072 | 0.020 | 318.43 |
| diabetes | quench-learned-depol-0.001 | 0.008997 | 0.067 | 0.099 | 0.010 | 166.74 |
| diabetes | quench-learned-depol-0.005 | 0.008977 | 0.065 | 0.129 | 0.015 | 244.69 |
| diabetes | quench-learned-depol-0.02 | 0.008903 | 0.069 | 0.176 | 0.035 | 579.83 |
| diabetes | quench-learned-depol-0.1 | 0.008509 | 0.052 | 0.203 | 0.016 | 258.52 |
| diabetes | maxwells-daemon | 0.008878 | 0.067 | 0.126 | 0.030 | 490.53 |
| synthetic-rho-0.9 | add-delete-swap | 0.06944 | 0.152 | 0.020 | 0.054 | 867.06 |
| synthetic-rho-0.9 | quench-learned-depol-0 | 0.02942 | 0.164 | 0.050 | 0.041 | 651.60 |
| synthetic-rho-0.9 | quench-learned-depol-0.001 | 0.0294 | 0.173 | 0.087 | 0.045 | 725.26 |
| synthetic-rho-0.9 | quench-learned-depol-0.005 | 0.02929 | 0.176 | 0.071 | 0.043 | 680.33 |
| synthetic-rho-0.9 | quench-learned-depol-0.02 | 0.02891 | 0.185 | 0.053 | 0.028 | 445.53 |
| synthetic-rho-0.9 | quench-learned-depol-0.1 | 0.02684 | 0.149 | 0.048 | 0.017 | 277.18 |
| synthetic-rho-0.9 | maxwells-daemon | 0.02877 | 0.181 | 0.050 | 0.029 | 473.92 |

![scoreboard](scoreboard.png)

## Read

Add-delete-swap is the classical baseline. The ladder wins only
if it beats *both* its best single rung and ADS on the same
column. A bigger gap that costs more wall-clock is not a win.

- **diabetes**: ladder matches its best rung (`quench-learned-depol-0`, 0.99×) and is 0.39× ADS.
- **synthetic-rho-0.9**: ladder matches its best rung (`quench-learned-depol-0`, 0.98×) and is 0.41× ADS.

**Outcome: hurts / no help at the p=10 exact tier.** Noise does
heat the proposal — that part of the Maxwell's Daemon story is
real. A uniform mixture of those rungs does not beat its coldest
(best) rung, and no quench rung beats ADS. The ladder is what a
mixture of already-losing kernels has to be: an average. Whether
a hotter slice (Aer-realistic λ̂ ≈ 0.65) or classical parallel
tempering changes that is the next measurement, not this one.

## Notes

- Add-delete-swap is the baseline that matters.
- The ladder is a uniform mixture of *fixed* rungs. Choosing
  the rung from chain state would need log-q or diminishing
  adaptation — that is AdaptiveMixture (WP7), not this file.
- Classical parallel tempering is the deferred third arm.

