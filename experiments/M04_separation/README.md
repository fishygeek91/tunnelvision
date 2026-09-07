# M04 — the separation study: how classically hard are the quench's shells?

**Registered 2026-09-01, BEFORE any results.** Follow-up to M03; the
candidate core of a separation paper.

## Claim under test

M03 showed de-novo shell sampling around low-energy ("cold") states costs
a median 75 s at n=20 (vs ~1e-1 s mid-spectrum, ~1e-2 s per quench
hardware proposal). M04 asks whether that is a scaling phenomenon or a
constant: measure cold-shell sampling cost at n = 16, 20, 24, 28, 32 with
(a) our sampler (UniGen-lite, CryptoMiniSat native XOR, adder PB), (b)
the state-of-the-art certified tools (ApproxMC for counting, UniGen for
sampling) on the same CNF, so the claim cannot be an artifact of our
implementation.

## Protocol

- Instances: all-to-all spin glass, random fields, seeds {1, 2}.
- Cold x: single-flip Metropolis at T=0.1 for 250*n sweeps from a random
  start (n-scaled burn-in, fixed in advance). Rand controls at n in
  {24, 32}, seed 1.
- Shells: eps = 2 raw units, scale S=1000 (as M03; audited there).
- Per cell: 8 samples, conflict budget 300k per SAT call, cell wall cap
  900 s (partial cells reported as partial, never dropped); ApproxMC
  projected count over the y-variables (subprocess, 900 s timeout)
  certifies the shell nonempty and reports log2|S|. If a cold x has an
  empty shell (the M03a seed-2 pathology), redraw x (recorded).
- M04b: pyunigen (UniGen3) on the same clauses, 5 samples, subprocess
  timeout 900 s, n in {16, 20, 24}.

## Pre-registered readings

- **Separation supported:** cold-shell median time grows by >= 2x per
  n -> n+4 across the grid (superpolynomial-looking), while rand controls
  stay < 1 s; real UniGen is no more than 10x faster than UniGen-lite on
  cold shells (tool-choice defense); ApproxMC certifies the slow shells
  nonempty (slowness is hardness, not emptiness).
- **Separation KILLED:** any tested configuration delivers cold-shell
  samples at n=32 with median < 1 s (then a competent classical
  implementation closes the gap and the paper idea dies).
- Timeouts/aborts are data: a cell where certified tools time out at
  900 s is evidence FOR hardness, reported as a censored measurement
  (median >= cap), never silently excluded.

## Honesty notes

- Cold x's at different n sit at different (unknown) energy percentiles;
  we record E(x)/n and the burn-in recipe rather than pretending to a
  controlled depth. The rand controls anchor the easy end.
- Container host (~1.7x slower than the dev laptop VM, measured M03a);
  irrelevant at the magnitudes in play.
- The quench side of the ledger is not re-run: ~1e-2 s/proposal on IBM
  hardware is E02c's measured number, flat in n at these sizes.
