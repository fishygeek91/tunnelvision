# E03 — parallel tempering vs rung vs ladder

Classical replica exchange on the exact g-prior, compared to
the phase-1 noise ladder, its coldest rung, ADS, and the
Aer-realistic hot rung (λ̂ ≈ 0.65). Accept/reject is exact on
every arm. PT has no 2^p spectral gap — that cell is a dash.

## Provenance

- git commit: `6c872d0bcdcfcc994a89e8b98e2036b2477fc802`
- started (UTC): 2026-08-14T21:05:34.328423+00:00
- finished (UTC): 2026-08-14T21:05:58.640854+00:00
- package versions: numpy=2.4.6, scipy=1.17.1, scikit-learn=1.9.0, qiskit=2.5.2, qiskit-aer=0.17.2, tunnelvision=0.0.1
- ladder λ: 0.0, 0.001, 0.005, 0.02, 0.1
- hot λ: 0.65
- PT: K=5, β_min=0.2
- quench: exact, grid 4×4
- chain: 4000 steps, burn-in 500

## Scoreboard

Gap is exact for single-kernel arms. ESS is from chains that
sample the same Q the gap used (tabulated), except PT, which
runs real ADS + swaps. ESS/eval charges K evaluations per PT
sweep.

| dataset | kernel | gap | accept | PIP err | ESS/step | ESS/eval | ESS/sec |
| --- | --- | --- | --- | --- | --- | --- | --- |
| diabetes | add-delete-swap | 0.0225 | 0.095 | 0.126 | 0.018 | 0.018 | 283.87 |
| diabetes | quench-learned-depol-0 | 0.009002 | 0.069 | 0.072 | 0.020 | 0.020 | 304.39 |
| diabetes | maxwells-daemon | 0.008878 | 0.067 | 0.126 | 0.030 | 0.030 | 465.62 |
| diabetes | quench-learned-depol-0.65 | 0.005725 | 0.032 | 0.135 | 0.005 | 0.005 | 71.83 |
| diabetes | parallel-tempering | — | 0.093 | 0.018 | 0.133 | 0.027 | 362.40 |
| synthetic-rho-0.9 | add-delete-swap | 0.06944 | 0.152 | 0.020 | 0.054 | 0.054 | 834.94 |
| synthetic-rho-0.9 | quench-learned-depol-0 | 0.02942 | 0.164 | 0.050 | 0.041 | 0.041 | 615.50 |
| synthetic-rho-0.9 | maxwells-daemon | 0.02877 | 0.181 | 0.050 | 0.029 | 0.029 | 455.80 |
| synthetic-rho-0.9 | quench-learned-depol-0.65 | 0.01256 | 0.064 | 0.053 | 0.015 | 0.015 | 235.50 |
| synthetic-rho-0.9 | parallel-tempering | — | 0.156 | 0.016 | 0.147 | 0.029 | 379.99 |

![scoreboard](scoreboard.png)

## Read

Add-delete-swap is the classical baseline. Parallel tempering
wins only if it beats ADS on ESS/eval (the fair cost) *and*
beats the ladder / best rung on the same column. A larger
ESS/step that spends K target evaluations is not a win.
The hot rung (λ̂ ≈ 0.65) is the Aer-realistic slice.

- **diabetes**: PT beats ADS on ESS/eval (1.44×); 0.88× ladder, 1.34× best rung (`quench-learned-depol-0`). Hot rung is 0.24× the λ=0 rung on ESS/eval. PIP errors: ADS 0.126, PT 0.018, hot 0.135.
- **synthetic-rho-0.9**: PT loses to ADS on ESS/eval (0.54×); 1.00× ladder, 0.72× best rung (`quench-learned-depol-0`). Hot rung is 0.37× the λ=0 rung on ESS/eval. PIP errors: ADS 0.020, PT 0.016, hot 0.053.

**Outcome: still hurts / no help for the quantum story.** The
Aer-realistic hot rung (λ̂ = 0.65) shrinks the already-losing
gap (diabetes 0.40× → 0.25× ADS; ρ=0.9 0.42× → 0.18× ADS)
and cuts ESS/eval. Classical PT does what PT does: ESS/step
jumps (diabetes 0.018 → 0.133; ρ=0.9 0.054 → 0.147) and the
diabetes PIP error drops 0.126 → 0.018, so the cold chain is
actually mixing. Charged per target-eval (K=5) that is 1.44×
ADS on diabetes and 0.54× ADS on the hard ρ=0.9 instance.
PT does not beat the noise ladder on ESS/eval, and nothing
here makes a quench kernel beat ADS on the exact gap. Mean
swap-accept is ~0.71 — the β ∈ [0.2, 1] ladder is
conservative (replicas are talking; they are not far apart).

The three E03 questions at the p=10 exact tier: noise heats
(yes); a proposal-temperature ladder helps (no); classical
target-temperature PT, or a hotter λ, rescues the quench
(no).

## Notes

- Add-delete-swap is the baseline that matters.
- Parallel tempering changes the *target* temperature of
  K replicas and swaps them. The noise ladder changes the
  *proposal* temperature of one chain. They are not the
  same object.
- PT's joint chain has no spectral gap on {0,1}^p. Do not
  invent one.

