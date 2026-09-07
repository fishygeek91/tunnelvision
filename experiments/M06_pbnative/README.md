# M06 — PB-native shell sampling: does any separation survive with the right tool?

**Registered 2026-09-07, BEFORE any results.** Closes the flank M05 opened.

## Question

M05 showed the M03/M04 "wall" was largely a CNF-translation artifact and
that a pseudo-Boolean solver (Exact) resolves deep shells 10–3,500x
faster. M06 measures the PB-native cost of a deep-shell proposal as a
function of n, with CORRECT low-energy states, and compares it to the
quench's ~1e-2 s per hardware proposal (E02c).

## Sampler (`dynamics/shellexact.py`)

- Exact model of the shell around x (binary y_i, linearized products,
  one two-sided window constraint, y != x); projected enumeration via
  `invalidateLastSol(y)`.
- If the shell has <= CAP members (CAP = 2000): enumerate fully and pick
  uniformly — EXACTLY uniform, no hashing needed. Deep shells are small
  (M05: 10^3 -> 10^0 below T_c), so this is the regime that matters.
- Else: fall back to the CMS hashing sampler (large shells are the easy
  ones — 0.15 s at 2^17 members, M05).
- Correctness gate: enumeration == numpy enumeration at n=8 (test).

## Protocol

- n in {16, 20, 24, 28, 32}, seeds {1, 2}; states equilibrated by the
  FIXED single-flip chain (8000 steps) at T = 0.55*sqrt(n) (just below
  T_c ~ sqrt(n) for unnormalized J~N(0,1)) and T = 0.35*sqrt(n) (deep).
  Scaling T with sqrt(n) places states at comparable relative depth.
- Per state: E/n, |S| (from full enumeration when <= CAP, else "large"),
  5 proposals, per-Exact-call timeout 300 s (censoring flag), plus the
  CNF sampler on the same state for the n <= 20 rows as continuity.
- eps = 2, scale = 1000 (as M03–M05).

## Pre-registered readings

- **Separation SURVIVES:** PB-native median cost per deep-shell proposal
  exceeds 1 s by n = 32 and grows monotonically with n across the grid
  (>= 2x per n -> n+4 on average) — a gap of >= 2 orders to 1e-2 s that
  widens.
- **Separation KILLED:** PB-native median cost stays <= 0.1 s through
  n = 32 on both depths — within 10x of hardware, the shell edge is
  dequantized in practice; the paper's outlook says so.
- Between: report the numbers, no headline.
- Timeouts are censored data (>= 300 s), never dropped. Empty shells are
  reported as empty (the window may need to scale with level spacing —
  recorded, not fixed here).
