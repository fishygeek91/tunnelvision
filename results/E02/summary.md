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

