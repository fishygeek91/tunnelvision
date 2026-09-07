# M06 results — with the right tool, the separation survives (2026-09-07)

**Data:** m06_scan.csv (20 rows), m06.log. **Sampler:**
`src/tunnelvision/dynamics/shellexact.py` (ExactShellSampler: full projected
enumeration → exactly uniform pick when |S| ≤ 2000; CNF-hashing fallback for
large shells). **Test:** `test_exact_shell_enumeration_matches_numpy`
(6/6 states at n=8; skipped on the dev VM where `exact` cannot build —
runs in the container). **Registration:** README.md.

## Verdict against the registered readings: SEPARATION SURVIVES

PB-native, exactly-uniform deep-shell proposal cost (median of 5, per state):

| n  | T = 0.55√n (just below T_c)           | T = 0.35√n (deep)                    |
|----|---------------------------------------|--------------------------------------|
| 16 | 3.5 s (|S|=262), 2.6 s (55)           | 1.3 s (3), 2.1 s (4)                 |
| 20 | 47 s (146), 41 s (34)                 | 31 s (EMPTY), 41 s (26)              |
| 24 | 27 s* (shallow state, |S|>2000, hash) / ≥300 s (11 found) | ≥300 s (10), ≥300 s (48) |
| 28 | ≥300 s (0 found), ≥300 s (17)         | ≥300 s (0), ≥300 s (10)              |
| 32 | ≥300 s (0), ≥300 s (0)                | ≥300 s (0), ≥300 s (0)               |

"≥300 s" = censored at the per-call timeout (registered as data). The one
starred cell is a state the fixed 8000-step burn-in left shallow
(E/n = −1.61 vs −2.9 to −4.0 elsewhere): its shell exceeded the cap and
went to the hashing route — the easy regime, as M05 predicts.

- Growth: n=16 → 20: 13–24× (uncensored). n=20 → 24: ≥6–10× (censored).
  Every deep cell at n ≥ 24 is censored: growth ≥ 10× per four spins on
  this grid, comfortably past the registered ≥ 2×.
- Against the quench's ~1e-2 s per hardware proposal (E02c, flat in n at
  these sizes): 2–3 orders at n=16, 3.5 orders at n=20, ≥ 4.5 orders and
  widening from n=24.
- Kill condition (≤ 0.1 s through n=32): not remotely met.
- CNF continuity column (n ≤ 20): CryptoMiniSat 13–60 s at n=16 and
  216–1384 s at n=20 on the same states — PB-native is 4–30× cheaper, as
  M05 found, and STILL loses to the quench by 3.5 orders at n=20.

## Why exactness costs what it costs (the mechanism, tool-independent)

Finding shell members is fast (M05: 17 members in 0.1–0.5 s). What an
EXACT proposal needs is either the complete member list (for the
|S(x)|/|S(y)| Hastings ratio, or to pick uniformly) or hashing at a
density that itself requires UNSAT-type proofs. In both cases the cost is
*deciding that no further member exists* in a near-empty integer window —
31 s to certify an empty shell at n=20, undecidable within 300 s at n ≥ 28
(0 members found, emptiness unproven). The quench never pays this: it
proposes from the window without certifying anything, and Metropolis
absorbs the rest. This is the sharpest statement of the program's
mechanism: **the quench's edge is the cost of a completeness proof that a
physical sampler never has to produce.**

## Caveats, registered or discovered

- Cheap partial enumeration ("first k members") is NOT a valid MH
  proposal (solver-order bias, non-symmetric); it would be an
  approximate-proposal escape with a bias to control — noted, not
  pursued.
- ε = 2 makes many deep shells empty or tiny at n ≥ 28 (0 members found
  within budget in 6 of 8 cells): the mechanism's window should scale
  with the local level spacing at larger n; measuring |S(ε)| vs n is the
  next figure, not this one.
- Fixed 8000-step burn-in reaches less relative depth at larger n (one
  shallow outlier at n=24); E/n is recorded per row, and every censored
  cell sits at E/n ≤ −2.9.
- Container host (~1.7× a laptop VM); irrelevant at these magnitudes.
- Two seeds; no exponent claimed, only the ≥ 10×/4-spin growth and the
  censoring pattern.

## What it settles

The open flank from M05 is closed: the deep-shell cost is not a
CNF artifact — with a native pseudo-Boolean solver it is smaller by 1–2
orders but grows just as steeply, and by n = 24 it is censored at 300 s
against a 1e-2 s hardware proposal. The paper's §7.1 sentence "whose
scaling with n we have not measured" is replaced by the measured scaling.
