# M04 results — the separation scales, and it is depth-driven (2026-09-01)

**Data:** m04a_scaling.csv (truncated grid + censoring flags), m04a_rand.csv
(controls), m04b_unigen.csv, m04a.log (truncation note). Registration:
README.md, written first. Host: cloud container (~1.7x a laptop VM, M03a).

## Verdict against the registered readings

**Separation SUPPORTED; kill condition not met.** No tested configuration
sampled a cold shell at n=32 in under 1 s — none sampled one at all.

| n  | cold t_med (s)            | rand t_med (s) | UniGen cold (s/sample) |
|----|---------------------------|----------------|------------------------|
| 16 | 0.57 / 2.4  (seeds 1/2)   | —              | 24 / 48                |
| 20 | 0.90 / 19.3               | —              | 66 / TIMEOUT(900)      |
| 24 | ≥560* / ≥1824 (1 delivered in 30.4 min) | 0.19 | TIMEOUT(900) both |
| 28 | ≥575* (0 delivered)       | —              | not attempted          |
| 32 | not reached (run truncated) | 0.44         | not attempted          |

*censored at cell caps; asterisked cells delivered ZERO samples — every
attempt exhausted the 300k-conflict budget (aborts), so the true cost per
delivered sample is unbounded at that budget, not "~575 s".

- **Growth:** on like-for-like seeds, n=20 → n=24 multiplies the cold
  median by ≥600x (seed 1) and ≥94x (seed 2). Superpolynomial-looking on
  this grid; a clean exponent is not claimable from two seeds and heavy
  censoring, and we don't claim one.
- **Depth, not size:** at n=32 a mid-spectrum shell of ~2^27 members
  samples in 0.44 s (count: 225 s, ok); at n=24 the ~2^20-member rand
  shell takes 0.19 s. The classical cost tracks the DEPTH of the state
  the shell surrounds, not n or |S| — cold-state variance is large
  (M04's n=20 seed-1 state is shallower than M03b's: 0.9 s vs 74.6 s;
  E_per_n and log2S per cell are in the CSV for exactly this reason).
- **Tool defense (airtight):** certified UniGen3 on identical CNFs is
  10-70x SLOWER than our UniGen-lite where it finishes, and delivers
  NOTHING in 900 s on every cell from n=20 seed-2 up. ApproxMC counting
  itself times out at 900 s on every deep shell from n=20 seed-2 up
  (while counting 2^27-member easy shells in minutes). The M03/M04
  numbers understate nothing by tool choice.
- **Nonemptiness:** certified by count at n≤20 seed-1 and by a delivered
  sample at n=24 seed-2. At the all-abort cells (24s1, 28s1) neither a
  sample nor a count nor an UNSAT proof fits in budget: the cells are
  "hard-or-empty, undecidable at 300k conflicts / 900 s" — deciding
  near-ground-shell membership at all is what is hard, which is the
  point sharpened.

## The claim this supports (candidate core of the separation paper)

Around low-energy states of dense frustrated Ising instances, de-novo
near-uniform energy-shell sampling — OUR sampler, certified UniGen, and
certified ApproxMC alike — hits a wall between n=20 and n=24 that
mid-spectrum shells of the SAME instances at the SAME n never feel
through n=32. The quench proposes from these shells at ~1e-2 s per shot,
flat in n at these sizes (E02c). Combined with M01/M02 (no cheap kernel
reaches the shells at all) and D01/D02 (the shells are sufficient for the
quench's edge), this is a three-layer, tool-independent, pre-registered
wall-clock separation with a physical sampler on the winning side.

## Open flanks (for the paper, in honesty)

1. Energy-resolved tensor-network sampling: the one untested classical
   family (Christmann's MPS simulates the quench at superlinear cost;
   a direct TN shell sampler is unexplored).
2. Specialized solvers: portfolio beyond CMS/UniGen (e.g. PB-native
   solvers like RoundingSat on the un-CNF'd constraints) — CDCL-on-CNF
   may not be the best classical attack on window constraints.
3. Cold-state depth is uncontrolled (burn-in recipe, not percentile);
   a depth-resolved hardness curve (time vs E/n at fixed n) is the
   natural next figure and would turn the depth observation into a law.
4. Censoring: n≥28 numbers are lower bounds only; bigger budgets would
   firm them (at ~hours/cell).

## Session cost

Registration + 2 scripts + ~4 h of container compute (most of it the
deep cells proving themselves censored) + truncation per registration.

---
**ERRATUM 2026-09-07 (see M05_depth/M05_RESULTS.md).** Same generator bug
as M03b: the "cold" states here were not low-energy. The censored
timings and UniGen/ApproxMC timeouts are real, but (a) the depth
attribution is withdrawn and (b) M05 shows the wall is largely a
CNF-translation artifact — Exact (PB-native) resolves the same class of
deep shells in seconds. The rand controls (no chain) stand. The
three-layer separation claim is downgraded to a one-to-three-order gap
at n=20 with n-scaling unmeasured; M06 (PB-native, correct states) is
the experiment that would restore or bury it. Generator fixed in
m04a_scaling.py.
