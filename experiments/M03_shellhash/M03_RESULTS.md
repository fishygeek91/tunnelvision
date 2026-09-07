# M03 — SHELLHASH results (2026-08-31 → 09-01)

**Code:** `src/tunnelvision/dynamics/shellhash.py` (CryptoMiniSat native-XOR
path + pysat fallback; adder PB encoding; conflict-budget abort accounting).
**Tests:** 4 in `tests/test_dynamics.py` (21/21 pass, device, solver=cms).
**Data:** `m03a_validate.csv` (12 rows), `m03b_scale.csv` (15 rows,
truncated — see log), `m03c_spotcheck.py` (diagnostic), `m03a/b` logs.
Registered outcomes and gates: `README.md`, written before any results.

## Verdict in one line

**SHELLHASH is correct and near-perfectly uniform — and the deploy lane is
dead by the pre-registered criterion: at n=20, low-energy (MCMC-relevant)
shell samples cost a median 74.6 s each, 75× over the kill bar and ~7,500×
the quench's ~10⁻² s hardware proposal.** The open question survives as a
genuine wall-clock separation, now with an evidence-backed classical anchor.

## M03a — correctness and uniformity gates: PASS

All six M01 instances, T ∈ {1.0, 0.1}, ε=2, 2000 samples per cell:
integer-vs-real shell symmetric difference **0 in 12/12 cells** (scale
S=1000 quantization never moved a single member); TV distance to
uniform-on-shell **0.000–0.032**, thirty-fold under the 0.10 gate; solver
enumeration equals numpy enumeration exactly (test-certified). Two side
findings: shells around Gibbs-typical states at n≤10 are tiny (1–5
members), and instance seed=2's ground state is energetically isolated
(empty ε=2 shell) — the same pathology M02a saw as frac_id≈1.

## M03b — the kill test (truncated after the criterion was decided)

Two regimes per instance: `rand` (uniform random x, mid-spectrum) and
`cold` (x after 4000 single-flip steps at T=0.1 — where a chain lives).

| n  | regime | median/sample     | notes                             |
|----|--------|-------------------|-----------------------------------|
| 16 | rand   | 62–192 ms         | m≈10 XORs, big shells, hashing easy |
| 16 | cold   | 0.8–2.5 s (p90 to 10.5 s) | m≈4, tight shells        |
| 20 | rand   | 115–146 ms        | m≈13–14                           |
| 20 | cold   | **74.6 s** (p90 148 s, 0 aborts) | **kill row**       |

Zero aborts anywhere: these are honest solve times, not budget artifacts.
Truncation: after the kill row, remaining cells (n=20 cold s2–s3, n=27)
were redundant for the registered question at ~1 h/row; noted in
`m03b.log`. Kill criterion (>1 s median at n=20, MCMC-relevant regime):
**exceeded 75×. Deploy lane closed.**

## The mechanism — and why this is a result, not just a negative

The hardness is *anti-correlated with hash density*: `rand` states need
m≈13 XORs (huge shells) yet sample in ~0.1 s; `cold` states need only m≈4
(small shells) yet cost 1000× more. The cost is not in the hashing — it is
in the CDCL proofs over tight low-energy energy windows, i.e. in deciding
near-ground-state degeneracy structure. **Classical shell sampling is easy
exactly where MCMC doesn't need it and hard exactly where the quench's
edge lives.** Combined with D01/D02 (shell-oracle sufficiency) this gives
the cleanest statement of the program's separation to date:

> The quench is a physical sampler for precisely the energy shells that
> are classically hard to reach — de novo generation at low energy costs
> ~10⁴ s-scale SAT work per sample at n=20 versus ~10⁻² s per quench
> proposal on hardware, while cheap classical kernels (M01 zoo, M02 ICM)
> cannot reach those shells at all.

Caveats registered in advance and honored: UniGen-lite (guarantees cited,
uniformity validated empirically at enumerable n); quantization audited
(sym_diff=0); container timings ~1.7× a laptop VM on identical work — the
75× kill margin dwarfs the host factor.

## Engineering notes (cost us ~2 h; recorded so they never do again)

1. pysat's default PB encoding ('best'/BDD) is pseudo-polynomial in
   Σ|weights| (~5×10⁵ here): on n=16 seed=3 it took >4 min to *encode*
   (and seqcounter OOM'd). `EncType.adder`: 0.01 s, 14k clauses. Forced.
2. Plain CDCL on chunked-CNF parity constraints hits the known XOR wall
   (>5 min/sample where CMS+native XOR takes 60 ms). CryptoMiniSat via
   pycryptosat is auto-selected when importable; chunked-CNF fallback kept.
3. pysat rejects raw negative PB bounds before its own normalization —
   normalize weights manually (`_atmost_norm`).
4. The PB clauses depend only on the energy window, so they cache across
   samples (and across a whole chain in bin mode, where the proposal is
   also exactly symmetric): encode cost amortizes to ~0.

## What's next for the program

- **Paper:** the sharpest-open-question paragraph can now cite hard
  numbers for the classical anchor (M03 CSV) alongside Christmann's MPS
  cost; consider one sentence + data pointer, as done for M02.
- **The separation framing** (quench = physical sampler for classically
  SAT-hard shells) is a candidate headline for the paper's outlook, or the
  seed of the follow-up paper: quantify how hardness scales (n=24–32 cold
  cells, solver portfolios, #SAT-based |S| certification) and whether ANY
  polynomial classical family (the last untested: TN with energy-resolved
  legs) breaks it.
- IDEATION-02's remaining reaches (NELSON, KZ-SCHED) are untouched by this
  verdict and stay queued.

---
**ERRATUM 2026-09-07 (see M05_depth/M05_RESULTS.md).** The "cold" states
in M03b were generated by a chain with a uint8 wraparound bug (`2*x-1` →
255): they were NOT low-energy states. The 74.6 s figure is a real
timing of an arbitrary state's shell under CNF+CryptoMiniSat, but its
depth attribution is withdrawn, and M05 shows the CNF route itself is the
wrong classical baseline (a PB-native solver is 10–3,500× faster on deep
shells). M03a (correctness, uniformity) stands. Generator fixed in
m03b_scale.py.
