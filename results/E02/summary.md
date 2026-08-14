# E02a — TunnelVision exact tier

Spectral gaps and mixing diagnostics on a spike-and-slab posterior.
Accept/reject uses the exact g-prior; the surrogate only shapes proposals.

## Provenance

- git commit: `1496da89cd903fdc26cdd9ac04a580d2fefeea73`
- started (UTC): 2026-08-14T19:07:22.670145+00:00
- finished (UTC): 2026-08-14T19:07:47.151722+00:00
- package versions: numpy=2.4.6, scipy=1.17.1, scikit-learn=1.9.0, qiskit=2.5.2, qiskit-aer=0.17.2, tunnelvision=0.0.1
- dataset: diabetes (sklearn, p=10)
- prior inclusion: 0.5
- quench: exact, grid 4×4
- chain: 4000 steps, burn-in 500

## Surrogate diagnosis

If Spearman(−E, log p) is weak, stop — the quench is exploring the wrong landscape.

| surrogate | Spearman(−E, log p) | ground-state rank | Δlogp vs MAP |
| --- | --- | --- | --- |
| analytic | 0.593 | 73 | 8.38 |
| learned | 0.983 | 11 | 3.21 |

![analytic scatter](scatter_analytic.png)

![learned scatter](scatter_learned.png)

## Exact-tier scoreboard

Gap is exact (transition-matrix). ESS chains sample the same Q the
gap used, so ESS/step is the mixing of that kernel. ESS/sec is
Q-sampling + MH, not a fresh expm per propose; `gap_seconds` in
the JSON is the unitary-assembly cost.

| kernel | gap | accept | PIP err | ESS/step (|γ|) | ESS/sec (|γ|) | ESS/sec (PIP) | Q build (s) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| uniform | 0.003475 | 0.008 | 0.236 | 0.007 | 114.71 | 8729.04 | 0.6 |
| single-flip | 0.009314 | 0.124 | 0.105 | 0.015 | 260.59 | 5677.14 | 0.6 |
| add-delete-swap | 0.0225 | 0.095 | 0.126 | 0.018 | 313.38 | 5569.31 | 0.6 |
| quench-analytic | 0.006838 | 0.036 | 0.073 | 0.004 | 66.82 | 5607.70 | 11.2 |
| quench-learned | 0.009002 | 0.069 | 0.072 | 0.020 | 327.23 | 3704.85 | 10.5 |

![scoreboard](scoreboard.png)

## Read

Add-delete-swap is the baseline. A larger gap that costs more wall-clock
is not an advantage — both columns have to move the same way.

- `quench-analytic` loses to ADS on exact gap (0.30×) and loses to ADS on ESS/sec (0.21×).
- `quench-learned` loses to ADS on exact gap (0.40×) and matches ADS on ESS/sec (1.04×).

Diabetes p=10 is mildly correlated, not a multimodal stress test.
A loss here is not a gate failure — E02b asks whether the ratio
grows with ρ. PIP error after a few thousand draws is Monte Carlo
(1/δ is tens to hundreds of steps), not a wall breach.

## Notes

- Add-delete-swap is the baseline that matters. Beating uniform proves nothing.
- Headline comparison is the exact gap. ESS/sec here is tabulated-Q, not QPU time.
- E02b (ρ-sweep, sampled tier at p=20–27) is not in this run.

## E02b — ρ-sweep (exact tier)

Full write-up: [`rho_sweep/summary.md`](rho_sweep/summary.md).

On equicorrelated synthetics (p=10, n=120, k_true=3, SNR=2), quench
does **not** beat add-delete-swap at any ρ ∈ {0, 0.3, 0.5, 0.7, 0.9}.
The learned/ADS gap ratio grows modestly (0.33× → 0.42×) because
quench-learned improves slightly while ADS is flat; it never crosses 1.
The analytic surrogate collapses as ρ grows (Spearman 0.64 → 0.36,
ground-state rank 46 → 444). Split-R̂ is clean for ADS and quench
(R̂ ≤ 1.07); uniform at ρ=0.9 is 1.12.

| ρ | ADS gap | quench-analytic | quench-learned | analytic / ADS | learned / ADS |
| --- | --- | --- | --- | --- | --- |
| 0 | 0.0710 | 0.0118 | 0.0231 | 0.17× | 0.33× |
| 0.3 | 0.0696 | 0.0122 | 0.0226 | 0.18× | 0.32× |
| 0.5 | 0.0695 | 0.0135 | 0.0233 | 0.19× | 0.34× |
| 0.7 | 0.0694 | 0.0153 | 0.0244 | 0.22× | 0.35× |
| 0.9 | 0.0694 | 0.0133 | 0.0294 | 0.19× | 0.42× |

Gate: negative result, written up. Sampled p=20–27 is the remaining
place the story could still turn.

## Ablations

Full write-up: [`ablations/summary.md`](ablations/summary.md).

Depolarizing heat at λ ≤ 0.1 barely moves the already-losing
quench/ADS gap ratio (learned 0.40× → 0.38× on diabetes;
0.42× → 0.39× at ρ=0.9). A Trotter circuit with per-gate
p = 0.005 already maps to λ̂ ≈ 0.65, so the exact-tier slice
is colder than realistic Aer noise. Wrecking the learned
surrogate (Spearman 0.99 → 0.3) kills the gap and leaves PIP
error at Monte Carlo — the wall holds.

