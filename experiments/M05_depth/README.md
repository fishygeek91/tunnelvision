# M05 — the depth law, and the PB-native flank

**Registered 2026-09-07, BEFORE any results.** Follow-up to M04 (open
flanks 2 and 3 of M04_RESULTS.md).

## Questions

1. **M05a (depth curve).** M04 showed cold-shell hardness varies wildly
   with the particular state (0.9 s vs 74.6 s at n=20). Is hardness a
   monotone function of the DEPTH of the state the shell surrounds?
   Protocol: n=20, seeds {1,2,3}; per instance a depth ladder — snapshots
   of a T=0.1 single-flip chain from a random start at 0, 200, 1000,
   5000, 20000 steps, plus a greedy descent to the local minimum from the
   deepest snapshot. Per state: E/n, log2|S| (ApproxMC subprocess, 300 s
   cap), SHELLHASH sample times (<=5 samples, 300k conflict budget,
   400 s state cap; censoring recorded).
2. **M05b (PB-native flank).** Is the M03/M04 wall an artifact of CNF
   translation? The Exact solver (RoundingSat lineage) takes the energy
   window natively: y_i binary, p_ij by standard linearization
   (p<=y_i, p<=y_j, p>=y_i+y_j-1), one two-sided linear window
   constraint — no CNF, no adder encodings. Per state (same ladders):
   feasibility time, 16-solution projected enumeration time, projected
   count (all with 300 s native timeouts), against CryptoMiniSat on the
   same shells. XOR-free enumeration is the fair proxy: M03 established
   hardness lives in the window proofs, not the hashing.

## Pre-registered readings

- **Depth law supported:** within-instance Spearman rho between depth
  (-E/n) and log t_med >= +0.7 on all three instances (censored cells
  enter at their lower bound). log2|S| is recorded so "depth per se vs
  shell size" is decided by data, not narrative.
- **Flank OPEN (sampler must be rebuilt PB-native):** Exact >= 10x
  faster than CMS on the deep half of the ladder. This would weaken
  M03/M04's constants (not the M01/M02 no-gos) and would be reported
  prominently, not buried.
- **Flank CLOSED:** Exact comparable to or slower than CMS on deep
  shells.
- Timeouts are censored data, reported as bounds; nothing is dropped.

## Notes

- Fresh container this session (rebuilt venv; Exact via PyPI). Host
  recorded per row.
- eps=2, scale=1000 throughout (audited in M03a: integer shell == real
  shell in 12/12 cells).
