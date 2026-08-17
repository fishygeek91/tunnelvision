# E03 — Maxwell's Daemon (noise as tempering)

**Question:** does deliberately scheduling noise strength across proposals
(a proposal-temperature ladder) improve mixing over (a) a single noise
level and (b) classical parallel tempering — with exactness intact?

**Phase 1 (simulator, exact tier):** quench kernels at K **fixed**
depolarizing levels λ ∈ {0, 1e-3, 5e-3, 2e-2, 1e-1} composed via
`NoiseLadderKernel` (random-scan). Targets: E02a diabetes + hardest
ρ=0.9 synthetic. Also measure: effective proposal temperature vs. λ
(fit proposal energy-change distributions) — the physics figure that
names the paper. Classical parallel tempering is the third arm
(`run_pt.py`): replica exchange on the exact g-prior, plus the
Aer-realistic hot rung (λ̂ ≈ 0.65).

**Bias audit (Aer methodology, not a QPU):** amplitude damping is
non-unital, so the symmetric-q claim is an approximation. The
runner compares sampled posteriors to enumerated truth on ideal
quench, `HardwareQuenchKernel` + Aer AD, and ADS
(`run_bias_audit.py`). Live IBM TV and physical rungs (DD /
twirling / idle) are still open — this audit is mandatory before
any hardware claim (see docs/ARCHITECTURE.md §2).

**Honest outcomes:** "noise-heating helps like tempering", "noise washes
out tunneling and hurts", or "helps only in regime X" — all three are
papers.

```bash
uv run python -m experiments.E03_maxwells_daemon.run              # p=10 exact tier
uv run python -m experiments.E03_maxwells_daemon.run --quick      # p=5 smoke
uv run python -m experiments.E03_maxwells_daemon.run_pt           # PT vs rung/ladder
uv run python -m experiments.E03_maxwells_daemon.run_pt --quick
uv run python -m experiments.E03_maxwells_daemon.run_bias_audit   # p=10 diabetes
uv run python -m experiments.E03_maxwells_daemon.run_bias_audit --quick
```

Writes `results/E03/summary.md` (phase 1), `results/E03/pt/summary.md`
(PT arm), and `results/E03/bias_audit/summary.md` (Aer AD methodology).
`--quick` writes to `results/E03_quick/` and is not the figure.
