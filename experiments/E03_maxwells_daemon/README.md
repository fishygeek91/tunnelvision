# E03 — Maxwell's Daemon (noise as tempering)

**Question:** does deliberately scheduling noise strength across proposals
(a proposal-temperature ladder) improve mixing over (a) a single noise
level and (b) classical parallel tempering — with exactness intact?

**Phase 1 (this PR, simulator, exact tier):** quench kernels at K
**fixed** depolarizing levels λ ∈ {0, 1e-3, 5e-3, 2e-2, 1e-1} composed
via `NoiseLadderKernel` (random-scan). Targets: E02a diabetes + hardest
ρ=0.9 synthetic. Also measure: effective proposal temperature vs. λ
(fit proposal energy-change distributions) — the physics figure that
names the paper. Classical parallel tempering is the deferred third arm.

**Phase 2 (hardware):** realize rungs physically — dynamical decoupling
on/off, twirling levels, inserted idle time — on IBM Heron. Includes the
**symmetric-q bias audit**: amplitude damping is non-unital, so measure
sampled-vs-enumerated posterior deviation to bound the residual bias
before claiming anything (see docs/ARCHITECTURE.md §2).

**Honest outcomes:** "noise-heating helps like tempering", "noise washes
out tunneling and hurts", or "helps only in regime X" — all three are
papers.

```bash
uv run python -m experiments.E03_maxwells_daemon.run           # p=10 exact tier
uv run python -m experiments.E03_maxwells_daemon.run --quick   # p=5 smoke
```

Writes `results/E03/summary.md` (tracked on a full run). `--quick`
writes to `results/E03_quick/` and is not the figure.
