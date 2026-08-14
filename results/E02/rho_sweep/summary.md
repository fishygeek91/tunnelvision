# E02b — ρ-sweep, exact tier

Spectral gap vs predictor correlation on equicorrelated spike-and-slab
posteriors. Accept/reject uses the exact g-prior; the surrogate only
shapes proposals. The story question: does quench/ADS advantage grow with ρ?

## Provenance

- git commit: `b2a45c2db03f5a0d9f6c5ddcd2d415b306bd275b`
- started (UTC): 2026-08-14T19:09:20.200178+00:00
- finished (UTC): 2026-08-14T19:11:49.750121+00:00
- package versions: numpy=2.4.6, scipy=1.17.1, scikit-learn=1.9.0, qiskit=2.5.2, qiskit-aer=0.17.2, tunnelvision=0.0.1
- design: equicorrelated n=120 p=10 k_true=3 snr=2.0
- ρ grid: 0, 0.3, 0.5, 0.7, 0.9
- quench: exact, grid 4×4
- chains: 4 × 6000 steps, burn-in 1500

## Surrogate diagnosis

| ρ | surrogate | Spearman(−E, log p) | ground-state rank | Δlogp vs MAP |
| --- | --- | --- | --- | --- |
| 0 | analytic | 0.639 | 46 | 5.81 |
| 0 | learned | 0.999 | 1 | 0.00 |
| 0.3 | analytic | 0.583 | 65 | 6.75 |
| 0.3 | learned | 0.998 | 1 | 0.00 |
| 0.5 | analytic | 0.416 | 422 | 46.87 |
| 0.5 | learned | 0.996 | 1 | 0.00 |
| 0.7 | analytic | 0.395 | 434 | 41.04 |
| 0.7 | learned | 0.996 | 1 | 0.00 |
| 0.9 | analytic | 0.359 | 444 | 24.59 |
| 0.9 | learned | 0.994 | 1 | 0.00 |

## Exact-tier scoreboard

Gap is exact. ESS and R̂ are from independent chains that sample
the same Q the gap used (tabulated, not a fresh expm per propose).
ESS/sec is therefore Q-sampling cost, not unitary assembly — E02a
is the run that prices the simulation. R̂ is split-R̂ on |γ|;
values ≫ 1 mean at least one chain stuck.

| ρ | kernel | gap | accept | PIP err | ESS/step (|γ|) | ESS/sec (|γ|) | R̂ (|γ|) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | uniform | 0.002951 | 0.019 | 0.144 | 0.008 | 128.63 | 1.059 |
| 0 | single-flip | 0.1 | 0.202 | 0.038 | 0.068 | 1023.59 | 1.001 |
| 0 | add-delete-swap | 0.07102 | 0.166 | 0.038 | 0.045 | 682.85 | 1.004 |
| 0 | quench-analytic | 0.01182 | 0.063 | 0.083 | 0.010 | 154.09 | 1.020 |
| 0 | quench-learned | 0.02313 | 0.183 | 0.046 | 0.016 | 232.43 | 1.026 |
| 0.3 | uniform | 0.002847 | 0.016 | 0.165 | 0.007 | 98.56 | 1.057 |
| 0.3 | single-flip | 0.1 | 0.196 | 0.033 | 0.064 | 968.33 | 1.002 |
| 0.3 | add-delete-swap | 0.06959 | 0.154 | 0.045 | 0.036 | 531.50 | 1.007 |
| 0.3 | quench-analytic | 0.01219 | 0.093 | 0.071 | 0.012 | 186.54 | 1.027 |
| 0.3 | quench-learned | 0.02256 | 0.175 | 0.060 | 0.019 | 296.49 | 1.006 |
| 0.5 | uniform | 0.00285 | 0.015 | 0.119 | 0.008 | 123.13 | 1.068 |
| 0.5 | single-flip | 0.1 | 0.193 | 0.027 | 0.068 | 1011.07 | 1.002 |
| 0.5 | add-delete-swap | 0.06946 | 0.151 | 0.039 | 0.040 | 600.40 | 1.004 |
| 0.5 | quench-analytic | 0.01347 | 0.099 | 0.078 | 0.013 | 204.04 | 1.010 |
| 0.5 | quench-learned | 0.02331 | 0.182 | 0.075 | 0.020 | 303.25 | 1.013 |
| 0.7 | uniform | 0.002858 | 0.013 | 0.139 | 0.008 | 121.66 | 1.044 |
| 0.7 | single-flip | 0.1 | 0.195 | 0.038 | 0.057 | 839.33 | 1.003 |
| 0.7 | add-delete-swap | 0.06943 | 0.158 | 0.045 | 0.035 | 520.78 | 1.009 |
| 0.7 | quench-analytic | 0.01529 | 0.144 | 0.094 | 0.013 | 195.43 | 1.015 |
| 0.7 | quench-learned | 0.02444 | 0.171 | 0.060 | 0.024 | 351.95 | 1.009 |
| 0.9 | uniform | 0.002867 | 0.019 | 0.191 | 0.012 | 167.71 | 1.121 |
| 0.9 | single-flip | 0.0999 | 0.191 | 0.033 | 0.062 | 934.38 | 1.004 |
| 0.9 | add-delete-swap | 0.06944 | 0.160 | 0.044 | 0.041 | 618.13 | 1.015 |
| 0.9 | quench-analytic | 0.0133 | 0.184 | 0.091 | 0.012 | 189.13 | 1.062 |
| 0.9 | quench-learned | 0.02942 | 0.164 | 0.036 | 0.031 | 462.30 | 1.005 |

## Gap ratio vs add-delete-swap

| ρ | quench-analytic | quench-learned |
| --- | --- | --- |
| 0 | 0.17× | 0.33× |
| 0.3 | 0.18× | 0.32× |
| 0.5 | 0.19× | 0.34× |
| 0.7 | 0.22× | 0.35× |
| 0.9 | 0.19× | 0.42× |

![gap vs rho](gap_vs_rho.png)

## Read

- `quench-analytic` / ADS gap ratio is roughly flat in ρ (0.17× at ρ=0 → 0.19× at ρ=0.9); beats ADS at 0/5 correlations.
- `quench-learned` / ADS gap ratio grows modestly (0.33× at ρ=0 → 0.42× at ρ=0.9); beats ADS at 0/5 correlations.
- Split-R̂ flagged a mild warning (worst: `uniform` at ρ=0.9, R̂=1.12). ADS and both quench kernels stay below 1.07.

This is a negative result, written as one. Quench never beats add-delete-swap
on the exact gap at p=10. The learned ratio does rise with ρ — quench-learned
gap goes 0.023 → 0.029 while ADS is flat at ≈0.070 — but it never crosses 1.

The analytic surrogate is the other finding: Spearman falls 0.64 → 0.36 and
the ground-state rank goes 46 → 444 as ρ grows. Collinearity-as-antiferromagnet
is the wrong landscape once predictors are interchangeable. The learned
surrogate stays excellent (Spearman ≥ 0.994, rank 1) at every ρ, so the
kernel — not the fit — is what fails to beat ADS here.

Single-flip has the largest gap (≈0.1) on this instance family. That is a
property of a small, not-very-sparse posterior (p=10, prior 0.5, SNR=2),
not a reason to headline it. ADS remains the field baseline.

Sampled p=20–27 is where modes should actually separate. This exact-tier
curve does not license a quantum-advantage claim.

## Notes

- Add-delete-swap is the baseline. Beating uniform proves nothing.
- Sampled tier at p=20–27 is in `results/E02/sampled/summary.md`.

