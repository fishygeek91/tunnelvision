# E03 — Maxwell's Daemon (noise as tempering)

**Question:** does deliberately scheduling noise strength across proposals
(a proposal-temperature ladder) improve mixing over (a) a single noise level
and (b) classical parallel tempering — with exactness intact?

**Phase 1 (simulator, exact tier):** quench kernels at K noise levels
(Aer depolarizing, p_dep ∈ {0, 1e-3, 5e-3, 2e-2, 1e-1}) composed via
NoiseLadderKernel. Targets: E02a diabetes + hardest rho=0.9 synthetic.
Also measure: effective proposal temperature vs. p_dep (fit proposal energy
distributions) — the physics measurement that names the paper.

**Phase 2 (hardware):** realize rungs physically — dynamical decoupling
on/off, twirling levels, inserted idle time — on IBM Heron. Includes the
**symmetric-q bias audit**: amplitude damping is non-unital, so measure
sampled-vs-enumerated posterior deviation to bound the residual bias before
claiming anything (see docs/ARCHITECTURE.md §2).

**Honest outcomes:** "noise-heating helps like tempering", "noise washes out
tunneling and hurts", or "helps only in regime X" — all three are papers.

Run: `python -m experiments.E03_maxwells_daemon.run` → results/E03/summary.md
