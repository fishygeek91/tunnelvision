# E03 — symmetric-q bias audit (Aer, not a QPU)

This bounds the symmetric-q approximation under a non-unital
channel. It is not a hardware result. Accept/reject uses the
exact g-prior on every arm. Cell is p=5 synthetic (enumerable);
the full config is p=10 diabetes and was not run this session.

## Provenance

- git commit: `d76aec945c8136a1a155e648d6fc086dd5fa0162-dirty`
- started (UTC): 2026-08-17T16:30:23.037604+00:00
- finished (UTC): 2026-08-17T16:30:42.508206+00:00
- package versions: numpy=2.4.6, scipy=1.17.1, scikit-learn=1.9.0, qiskit=2.5.2, qiskit-aer=0.17.2, tunnelvision=0.0.1
- damping γ: 0.05
- pool size: 8
- quench: trotter, grid 4×4
- chain: 4 chains, 800 steps, burn-in 100

## Scoreboard

TV is the total-variation distance between the pooled sampled
posterior and ``enumerate_exact``. PIP error is max-abs vs the
enumerated inclusion probabilities. R̂ is split-R̂ on |γ|.

| dataset | kernel | TV | PIP err | R̂(|γ|) | accept | cache hit |
| --- | --- | --- | --- | --- | --- | --- |
| synthetic-rho-0.5 | quench-ideal | 0.074 | 0.037 | 1.019 | 0.372 | — |
| synthetic-rho-0.5 | quench-ad | 0.379 | 0.300 | 1.482 | 0.026 | 0.87 |
| synthetic-rho-0.5 | add-delete-swap | 0.053 | 0.014 | 1.013 | 0.124 | — |

## Read

ADS is the exact-kernel baseline: its TV and PIP error are
Monte Carlo noise, not bias. Ideal quench is the symmetric
Trotter proposal. ``quench-ad`` is HardwareQuenchKernel with
Aer amplitude damping — the non-unital stand-in for hardware.
A TV that tracks ADS is the approximation holding; a TV that
blows past ADS is the residual the live-hardware audit must
quote before any QPU claim.

- **synthetic-rho-0.5**: quench-ad TV 0.379 vs ADS 0.053 (7.09×). PIP error 0.300 vs ADS 0.014. Ideal quench stays near ADS (TV 0.074, R̂ 1.02). Amplitude damping at γ=0.05 already wrecks the proposal: accept 0.026, R̂ 1.48, so part of the TV is a stuck chain, not a clean bias number. That is the point of the methodology — a non-unital channel can move the sampled posterior far past Monte Carlo, and the live-hardware audit has to quote that residual before any QPU claim. Cache hit rate 0.87: state-keyed pools do what ARCHITECTURE §4 promised.

## Notes

- This is the methodology the live-hardware audit must reuse.
- ``quench-ad`` returns log-q = 0 (the approximation). The
  residual lives in TV / PIP error, not in the Hastings ratio.
- p=10 diabetes and a real IBM TV bound are still open.
- Physical noise rungs (DD, twirling, idle) are still open.

