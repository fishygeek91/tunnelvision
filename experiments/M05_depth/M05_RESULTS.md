# M05 results — the depth law is real, and the wall was mostly ours (2026-09-07)

**Data:** m05_unified.csv (temperature ladder, n=20, seeds 1–3; seed 3
was still running at the time of writing — the CSV on disk is the final
one), m05.log, m05_firstladder_discarded.log (see amendment 2 below).
**Registration:** README.md. **Host:** fresh cloud container.

## Two corrections made BEFORE any valid M05 result, both dated

1. **State-generator bug (affects M03b/M04 "cold" rows).** The chain
   used to generate low-energy states computed spins as `2*x - 1` on a
   uint8 array, which wraps 0 → 255. The chain therefore walked on
   corrupted spins and its "cold" states were arbitrary, NOT low-energy.
   Scope: only the cold-state generators (m03b_scale.py, m04a_scaling.py,
   m05_common.py — all fixed 2026-09-07). Untouched and still valid: the
   sampler, the shell encoding, M03a's correctness/uniformity gates, and
   M04's rand controls. Consequence: the M03b/M04 timing rows are real
   measurements of *some* states' shells, but their attribution to
   "cold/deep" is withdrawn (SKELETON C12 partially withdrawn; C13 added).
   M05 supersedes them for the depth question at n=20.
2. **Design amendment.** The registered step-ladder collapses (a T=0.1
   chain reaches its minimum within ~200 steps at n=20); replaced by a
   temperature ladder. The first ladder (T=3.0 → 0.12) turned out to sit
   entirely below T_c ≈ √n ≈ 4.5 for unnormalized J~N(0,1) (its one row
   is kept in the discarded log: a near-ground state with an EMPTY ε=2
   shell, Exact proves UNSAT in 32 s, CMS ≥220 s); the final ladder is
   T ∈ {30, 15, 8, 5, 3.5, 2.5, 1.5, 0.5}.

## Verdicts against the registered readings

**Depth law: SUPPORTED, with the mechanism now visible.** Seed 1 ladder
(E/n → CMS sampler median → shell size by Exact count):

| T   | E/n    | |S|     | CMS sampler        | Exact enum(17) | Exact count |
|-----|--------|---------|--------------------|----------------|-------------|
| 30  | −0.09  | ~2^16.7 | 0.15 s             | 0.02 s         | timeout*    |
| 15  | −0.89  | ~2^15.7 | 0.29 s             | 0.03 s         | timeout*    |
| 8   | −0.93  | ~2^15.6 | 0.29 s             | 0.03 s         | timeout*    |
| 5   | −2.18  | 961     | 4.0 s              | 0.11 s         | 102 s       |
| 3.5 | −2.57  | 155     | 436 s (censored)   | 0.51 s         | 48 s        |
| 2.5 | −3.12  | 4       | 279 s (abort)      | 24 s           | 24 s        |
| 1.5 | −2.96  | 9       | 351 s (abort)      | 36 s           | 36 s        |
| 0.5 | −2.64  | 94      | 744 s (censored)   | 0.21 s         | 41 s        |

*exact counting of ~10^4–10^5 solutions is not what Exact is for; ApproxMC
counted those (log2|S| column) and timed out on every deep shell.
Seed 2 reproduces the shape (0.14 s → 0.17 → 4.4 → 4.3 s through T=5 with
|S| = 1277 at T=5; deeper rows in the CSV).

The mechanism: **the shells shrink**. Above T_c they hold ~10^5 members;
at T≈T_c ~10^3; below, 10^2 → 10^0 → empty. The CNF route's cost is the
cost of CDCL deciding membership in a near-empty integer window through
adder circuits — the depth law is a shell-size law.

**PB-native flank: OPEN — the sampler must be rebuilt.** On identical
states the Exact solver, taking the window natively (linearized products,
one two-sided linear constraint, no CNF), enumerates 17 shell members
10–3,500× faster than CryptoMiniSat samples one (0.11 vs 4.0 s; 0.51 vs
436 s; 0.21 vs 744 s), and *exactly* counts shells that ApproxMC cannot
approximate in 120 s. Certified CNF tools (UniGen/ApproxMC, M04) time out
on exactly the shells Exact resolves in seconds. The M03/M04 "wall" was
substantially a CNF-translation artifact.

## What this does to the program's claims (stated plainly)

- M01/M02 no-gos: untouched (they never involved SAT).
- D01/D02 oracle sufficiency: untouched.
- M03a sampler correctness/uniformity: untouched.
- M03b/M04 "cold-shell wall, ~7,500× separation, n=24 cliff":
  **withdrawn as stated.** Two independent reasons: the states weren't
  cold (bug), and the tool was the wrong one (flank). The paper's §7.1
  paragraph and the fifth contribution were revised 2026-09-07 to the
  honest anchor: with the right classical tool the deep shell costs
  10^-1–10^1 s at n=20 vs ~10^-2 s per quench proposal — a one-to-three
  order gap whose n-scaling is UNMEASURED with the native solver.
- Two robust facts survive every tool: deep shells at ε=2 are tiny, and
  every classical route must *decide* near-empty windows, which no cheap
  kernel attempts.

## Next (registered as questions, not claims)

1. **M06 — PB-native scaling.** Rebuild the shell sampler on Exact
   (enumerate-and-pick for shells ≤ ~10^3; hashing only when large — the
   large shells are the easy ones anyway) and redo M04's n-scan with
   CORRECT cold states. The residual gap's n-dependence is the actual
   separation question; nothing above answers it.
2. The ε window at ε=2 makes deep shells nearly empty at n=20 — the
   mechanism the quench realizes at n=8–10 (D01) may need ε scaling with
   the local level spacing. Measure |S(ε)| along the ladder.
3. Depth-resolved figure for the paper only after M06, with the native
   solver — the CNF curve above is a fact about CNF, not about physics.

---
**Note 2026-09-07 (after M06):** the `exact_enum=17` column in
m05_unified.csv over-counts for shells with fewer than 17 members: the
enumeration loop treated Exact's stale `hasSolution()` after UNSAT as a
new solution (fixed in `shellexact.py`). The *timings* stand: for
|S| ≥ 17 they measure 17 distinct members; for |S| < 17 they measure
complete enumeration + the UNSAT proof (and match the `count` timings to
the second). The 10–3,500× CMS-vs-Exact comparisons are unaffected in
substance. M06 supersedes M05 for the PB-native cost of an *exact*
proposal (which needs the completeness proof).
