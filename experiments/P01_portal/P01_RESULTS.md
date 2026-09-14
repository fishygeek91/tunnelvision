# P01 PORTAL race — RESULTS (2026-09-14)

**Registered design:** README.md (this directory), written before any run.
**Verdict: BOTH KILLS FIRED. K1 → quantum miner arm dead. K2 → whole rung dead.**
The both-die outcome was pre-registered as possible; it feeds QMCMC-Bench as a
registered negative. A3 gate passed, so the rung died in the race, not at the gate.

## Step 1 — A3 acceptance gate: PASS (p01_gate.csv)

8/8 cells (n ∈ {16,20} × seeds {1,2} × T ∈ {0.55,0.35}√n) reach portal acceptance
≥ 0.2 at the best ρ (best values 0.22–0.54; ρ = 0.5/n generally best). Energy-matched
shell atoms accept as the shell mechanism predicts at n ≤ 20. The race proceeded.

## Step 2 — Tier T truth race, n ≤ 20: K1 FIRED (p01_tier_t.csv, *_chains.csv)

Mode coverage vs enumerated truth (π-mass-weighted; modes = single-flip local minima
with mass ≥ 1e-4 of the top mode):

- n = 12: every arm reaches coverage 1.0 at every budget — non-discriminating.
- n ≥ 16 (24 cells: n ∈ {16,20} × 3 seeds × 2 T × budgets {2 s, 10 s}): **the best
  classical miner matches or beats the quench miner's coverage in 24/24 cells** —
  greedy+kick or PT-ICM hits 1.000 weighted coverage in every cell (solver ties at
  n=16 seed 3); quench spans 0.82–1.00.
- Budget honesty: the quench arm could not even stay inside its budget — a single
  trotterized evolution at n = 20 costs ~30–100 s, so its "2 s" cells actually consumed
  34–109 s (recorded in wall_s). Classical arms stayed on budget. K1 fires *with the
  overspend in the quench's favor.*

**K1 = classical miners match quench-miner mode coverage at matched budget → quantum
arm dead** (registered verbatim). Simulation-side caveat: shots-per-evolution economics
differ on hardware; but coverage parity at unlimited relative shot rate already decides
K1 as registered.

Downstream Tier T chains (diagnostic, n=20 medians, ESS/sec on energy trace,
split-R̂-gated): portal-greedy 525, local-MH 501, PT-ICM 471, portal-quench 360,
portal-solver 349, portal-pticm 174. At n ≤ 20 no portal library beats plain local MH —
these targets are too easy for darting to pay at truth scale (expected; the K2 question
lives at Tier S).

## Step 3 — Tier S scale race, classical arms only: K2 FIRED (p01_tier_s.csv)

Pre-screen (registered): 6/12 (n, seed, T) cells at n ∈ {50,100,200} show classical
metastability (4-chain split-R̂ > 1.2 on plain MH); the other 6 mix already and are
INVALID for K2 (discarded, per the E02c trap amendment A2).

On the 6 valid cells, solver-mined PORTAL vs PT-ICM baseline (raw chain ESS/sec, build
budget excluded → the number most favorable to PORTAL):

| n | seed | T/√n | screen R̂ | atoms | gain | amortized gain |
|---|---|---|---|---|---|---|
| 50 | 1 | 0.35 | 1.72 | 3 | 0.89 | 0.10 |
| 50 | 2 | 0.55 | 1.61 | 3 | 1.16 | 0.14 |
| 50 | 2 | 0.35 | 2.57 | 3 | 1.19 | 0.14 |
| 100 | 1 | 0.35 | 1.27 | 2 | 0.48 | 0.05 |
| 100 | 2 | 0.35 | 1.36 | 2 | 0.46 | 0.05 |
| 200 | 1 | 0.35 | 2.59 | 1 | 1.75 | 0.15 |

**Max gain 1.75× < 2× in every valid cell → K2 = darting gain < 2× over a PT-ICM
baseline chain → whole rung dead** (registered verbatim). Amortized (build included)
gains are 0.05–0.15×.

Mechanism of death (two independent failures at scale):
1. **Miner starvation:** Exact member-mining collapses from hundreds of atoms at n ≤ 20
   to 1–3 atoms in 30 s at n ≥ 50 — on dense all-to-all instances the model build +
   solve cost eats the budget. M06's "member-finding is cheap (0.1–0.5 s)" was measured
   at n ≤ 32; it does NOT extrapolate to n ≥ 50 dense at fixed ε = 2.
2. **Portal-acceptance collapse:** acceptance ~0.00–0.01 at n ≥ 50 (vs ≥ 0.2 at the
   n ≤ 20 gate) — with T fixed per-site and total energies growing like n, mined atoms
   are no longer energy-matched to the chain's typical state within δE ≲ T, and TH
   ρ-noise perturbations cost more energy at larger n. The A3 gate condition is
   n-dependent; passing it at n ≤ 20 did not transfer.

## K1 at scale

Undecidable without quantum hardware (statevector cap ~n ≤ 28; MPS truncation corrupts
the shell physics being measured). No claim is made either way. Noted per registration:
this is a hardware question for a future RECIPROCITY-class run — though with K2 dead,
the darting consumer for a hardware-mined library no longer exists on this problem
class, which lowers that run's priority.

## What survives

- The registered negative itself: a matched-budget miner race with pre-screened targets
  is a reusable QMCMC-Bench protocol piece (B_ext + amortization curve now have a
  measured worked example, outcome negative).
- The Free-Certificate theorem doc (docs/free_certificate.md) — (a)+(b) proven and
  numerically certified; independent of P01's outcome.
- NOVELTY-CHECK-04 (project root) — completed; claim-language constraints recorded
  (smart/generalized darting are the nearest classical neighbors; no solver- or
  quantum-mined fixed-library darting found — and after P01, we now know one reason
  why it's absent: on dense spin glasses it doesn't pay).
- A sharpened M06 caveat: the finding-vs-certifying gap is real at n ≤ 32 but
  member-finding is NOT flat-cheap at n ≥ 50 dense — this belongs in the paper's
  §7.1 wording audit.

## Files

README.md (registration), p01_common.py, p01_gate.py, p01_tier_t.py, p01_tier_s.py,
p01_gate.csv, p01_tier_t.csv, p01_tier_t_chains.csv, p01_tier_s.csv, tier_t.log,
tier_s.log. Seeds and budgets as registered; environment: cloud container venv
(python-sat/pypblib/pycryptosat/exact), single process per tier.
