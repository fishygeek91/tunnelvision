# M02 — Houdayer isoenergetic cluster moves vs. the shell mechanism

**Date:** 2026-08-31. **Code:** `src/tunnelvision/dynamics/icm.py` (+4 tests in
`tests/test_dynamics.py`, 17/17 pass). **Data:** `m02a_icm_cfar.csv` (144 rows),
`m02b_icm_gap.csv` (45 rows). **Scripts:** `m02a_icm_cfar.py`, `m02b_icm_gap.py`.

## Question (pre-registered against three outcomes)

IDEATION-02 Idea B: ICM (Houdayer 2001; Zhu–Ochoa–Katzgraber PRL 115, 077201)
is the only O(n), rejection-free, Hamming-global isoenergetic move in the
classical literature. Does it achieve the C_far conjunction — proposal mass
that is simultaneously near-degenerate (|ΔE| ≤ 2) and far (d_H ≥ 3) — and
does it buy mixing on the M01 instances? Outcomes: (i) yes → cheap classical
shell kernel found, no-go weakened; (ii) fails C_far but wins gap → predictor
refinement; (iii) fails both → no-go extends to the strongest classical family.

## Verdict: outcome (iii), with a sharper mechanism than expected

**ICM on dense frustrated instances is the identity in disguise.** Three
structural facts, all certified by tests rather than argued:

1. **Percolation degeneracy (certified).** On all-to-all instances every
   |J_ij| > 0, so the faithful cluster is the whole disagreement set D and the
   move is exactly the replica swap (x1,x2) → (x2,x1): a statistical no-op.
   Pair energy change is exactly zero, random fields included (they cancel
   across replicas on D — the Houdayer identity survives fields).
2. **Restriction destroys conservation (M02a).** Every strengthened variant
   (bond threshold θ, best-first size caps — still exactly symmetric, since D
   is invariant under the move) either barely restricts (θ=0.5 ≈ faithful) or
   loses pair-energy conservation and with it acceptance: at T=0.1 every
   genuinely partial cluster variant has **mean pair-Metropolis acceptance
   0.000** across all six instances. Concentration-without-conservation is
   exactly the informed-local failure mode of M01, now at cluster scale.
3. **The two-replica resource evaporates at low T (M02a).** Under
   stationarity the disagreement set is empty in 52–100% of draws at T=0.1
   (n=8: 52/100/99%; n=10: 72/99/53% for seeds 1/2/3). The regime where the
   quench shines is the regime where ICM has nothing to act on.

**The marginal-proposal mirage (important for the paper's framing).** The
faithful swap's single-replica marginal is q(y|x) = π(y) — the perfect
independence proposal, with C_far up to 0.48 in the M02a table. ICM contains
the ideal proposal as its marginal and delivers none of it, because the pair
move that realizes that marginal is the identity. C_far for pair-kernel
marginals is aspirational, not realizable; the exact pair gap is the honest
metric, hence M02b.

**M02b (exact pair-space gaps, n=8, seeds 1–3, T ∈ {1, 0.3, 0.1}).** Chain
P = ½P_ICM + ½P_SF vs. the pure single-flip pair baseline (whose gap equals
the M01 single-chain SF gap / 2 — the tensor identity holds to all printed
digits in production, cross-certifying the sparse solver against M01's dense
one). In **every resolvable cell (T = 1.0 and 0.3, all seeds, all four
variants): gap ratio 0.500–0.510×** — the pure mixture-dilution floor. The
ICM half of the chain contributes at most ~2% of what the same share of
single-flip moves would have contributed. Per target-evaluation (ICM move = 2
evals) the ratio is 0.33–0.34×. At T=0.1 both baseline and variants sit below
the double-precision floor (gaps ≲ 1e-14; the lone above-floor reading,
seed 2 cap=5 at 2.9e-11, has an unmeasurable baseline so its ratio is
unreliable — and is still ~9 orders below the quench's 6.5e-2 single-chain
gap in that cell). Reference points (M01): quench gap 5.4e-2–8.9e-2 in these
cells; ICM never moves the needle toward it anywhere.

## What this changes

- **Paper:** the §VII.A near-miss paragraph's ICM claims are now measured,
  not asserted: percolation-degeneracy (test), field-cancellation identity
  (test), acceptance collapse of restricted clusters (M02a), and
  dilution-floor gaps (M02b). One sentence + data pointer added to main.tex.
- **No-go strengthened:** the only O(n) global isoenergetic family in the
  classical literature fails the conjunction *for a mechanistic reason* (its
  conservation law lives on the pair, and only on the full component), not
  for lack of tuning. The open question — de-novo, single-chain,
  uniform-on-shell at distance, O(n)–O(n²) — stands.
- **Next:** SHELLHASH (IDEATION-02 Idea A) attacks that open question from
  the NP-oracle side: SAT + XOR hashing generates certified near-uniform
  shell samples de novo; the contest is wall-clock per shell sample, and both
  outcomes are publishable.

## Caveats

- M02a is Monte Carlo (10k draws/cell; σ ≈ 0.005 on C_far); M02b is exact.
- ICM variants tested are symmetric-by-construction (D-invariance +
  deterministic clusters). Randomized cluster growth (KF-style bond
  probabilities) would need its own Hastings accounting; nothing in the
  mechanism suggests it escapes the conservation/acceptance dilemma, but it
  was not tested.
- ICM's home turf (sparse lattices, zero field, PT stacks) is untouched by
  this negative: the claim is specifically about dense frustrated targets at
  the temperatures where the quench's edge lives — the paper's setting.
- n=10 gaps not run (4^10-state build ~hours in pure python); M02a covers
  n=10 proposal statistics, and every n=8 gap cell is at the dilution floor.
