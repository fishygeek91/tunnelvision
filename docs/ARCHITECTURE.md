# Architecture

## The one diagram that matters

```
                 ┌────────────────────────────────────────────┐
                 │      EXACT SIDE (correctness lives here)   │
   data (X,y) ──▶│  targets/spike_slab.py   exact log p(γ|y)  │
                 │  engine.py               accept / reject   │
                 │  diagnostics.py          the scoreboard    │
                 └───────────────▲────────────────────────────┘
                                 │ proposals (y, log q_fwd, log q_rev)
                 ┌───────────────┴────────────────────────────┐
                 │   HEURISTIC SIDE (allowed to be wrong)     │
   data (X,y) ──▶│  surrogate.py     2-local Ising (h, J)     │
                 │  kernels/quantum  quench circuit, noisy    │
                 │  kernels/noise_ladder  Maxwell's Daemon    │
                 │  kernels/classical     baselines           │
                 └────────────────────────────────────────────┘
```

The horizontal line is the project. Nothing crosses it except proposals
going up and (X, y) entering both sides independently. The surrogate never
touches accept/reject; the target never shapes proposals; kernels never see
target log-probs.

## Key design decisions (with reasons)

1. **Kernels report (y, log_q_fwd, log_q_rev), not acceptance decisions.**
   Keeps MH accounting in one audited place. Symmetric kernels return zeros.
2. **Quench kernel symmetry**: time-symmetric evolution gives
   |⟨y|U|x⟩|² = |⟨x|U|y⟩|² (Layden SI). Unital noise (depolarizing, dephasing)
   preserves symmetry; amplitude damping does not. Consequence: on real
   hardware the symmetric-q assumption is an approximation — E03 must
   quantify the residual bias empirically (compare vs. enumerated posterior)
   before any hardware result is claimed. This is the project's single
   biggest correctness risk; treat it as such.
3. **Exact tier at small n**: transition-matrix diagonalization at n≤14 gives
   spectral gaps with zero Monte Carlo error. All headline comparisons at
   p=10 use this tier; sampled diagnostics are for scale-up only.
4. **Batched hardware proposals**: QPU round-trips dominate wall clock. The
   hardware kernel pre-generates proposal pools per job; chain-order
   dependence is avoided because quench proposals depend only on the current
   state, and states revisit — cache keyed by state.
5. **Seeds everywhere**: every run takes an explicit seed; results files
   record (git commit, config, seed).

## Experiment map

| ID  | Question | Target | Tier |
|-----|----------|--------|------|
| E01 | Do we reproduce Layden's spectral-gap speedup? | Ising spin glass n=8–12 | exact |
| E02 | Does the quench+surrogate beat add-delete-swap on real data? | diabetes p=10, synthetic rho-sweep p=10–27 | exact + sampled |
| E03 | Is scheduled noise a tempering resource (Maxwell's Daemon)? | E02 targets under Aer noise models, then hardware | exact + sampled |

## Dependencies policy

qiskit + qiskit-aer only in `kernels/quantum*` and `noise_ladder`; the exact
side must import nothing quantum (it should run on a machine with numpy/scipy
alone). Enforced by convention now; add an import-linter contract later.
