# E03 — symmetric-q bias audit (Aer, not a QPU)

This bounds the symmetric-q approximation under a non-unital
channel on the headline dataset (diabetes, p=10). It is not a
hardware result. Accept/reject uses the exact g-prior on every
arm.

## Provenance

- git commit: `b1667043452a7bc8f36984bdab7b98cbdb3508fa-dirty`
- started (UTC): 2026-08-17T22:03:31.851170+00:00
- finished (UTC): 2026-08-17T22:50:10.708482+00:00
- package versions: numpy=2.4.6, scipy=1.17.1, scikit-learn=1.9.0, qiskit=2.5.2, qiskit-aer=0.17.2, tunnelvision=0.0.1
- damping γ: 0.01, 0.05, 0.1
- pool size: 8
- quench: trotter, grid 4×4
- chain: 4 chains, 8000 steps, burn-in 1000

## Scoreboard

TV is the total-variation distance between the pooled sampled
posterior and ``enumerate_exact``. PIP error is max-abs vs the
enumerated inclusion probabilities. R̂ is split-R̂ on |γ|.

| dataset | kernel | TV | PIP err | R̂(|γ|) | accept | cache hit |
| --- | --- | --- | --- | --- | --- | --- |
| diabetes | quench-ideal | 0.067 | 0.026 | 1.002 | 0.089 | — |
| diabetes | quench-ad-0.01 | 0.154 | 0.026 | 1.052 | 0.005 | 0.87 |
| diabetes | quench-ad-0.05 | 0.808 | 0.320 | 2.612 | 0.000 | 0.87 |
| diabetes | quench-ad-0.1 | 1.000 | 0.820 | 3.696 | 0.000 | 0.87 |
| diabetes | add-delete-swap | 0.071 | 0.052 | 1.007 | 0.089 | — |

## Read

ADS and ideal quench mixed (R̂ ≤ 1.01) and sit at the same
Monte Carlo floor (TV 0.071 vs 0.067). The symmetric-q
approximation is therefore fine when the channel is unital /
absent. Amplitude damping is the non-unital stand-in for
hardware T1. TV grows with γ; only γ=0.01 is a clean bias
number.

- **γ=0.01 (mixed, R̂ 1.05):** TV 0.154 = 2.17× ADS. PIP error
  matches ideal (0.026) — the inclusion-probability scoreboard
  can hide a posterior that is already the wrong shape. Accept
  0.5%. This is the residual a live IBM run at mild T1 must
  beat or quote.
- **γ=0.05 (does not mix, R̂ 2.61):** TV 0.808, accept 0.000.
  Not a bias estimate — the chain is dead. Report as "does
  not mix under this channel."
- **γ=0.1 (does not mix, R̂ 3.70):** TV 1.000, accept 0.000.
  The sampled histogram is unrelated to the posterior.

Cache hit 0.87 on every AD arm: state-keyed pools work. They
do not restore mixing.

## Notes

- This is Aer amplitude damping, not a QPU. The live-hardware
  TV bound is still mandatory before any hardware claim.
- Each ``quench-ad-γ`` arm returns log-q = 0 (the
  approximation). The residual lives in TV / PIP error,
  not in the Hastings ratio.
- Physical noise rungs (DD, twirling, idle) are still open.

