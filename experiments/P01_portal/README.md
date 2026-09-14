# P01 — PORTAL race (pre-registered design)

**Registered:** 2026-09-14, branch `p01-portal`, BEFORE any run. Amendments to this file
after first data must be dated and labeled AMENDMENT.
**Provenance:** IDEATION-04 memo §4 Rung 1 + §8 red-team amendments A1–A3 (binding).
**Question:** Does a fixed atom library mined OFFLINE (quantum quench or classical solver)
buy honest wall-clock sampling gains via exact darting MCMC — and does the *quantum* miner
earn its place over classical miners at matched budget?

## Architecture under test

- **Offline miner** (paid once, matched wall-clock builder budget B per arm): collects a
  fixed atom library A = {a_1..a_m} of low-energy / shell states.
- **Online chain** (classical, exact): MH with mixture kernel = single-flip local moves +
  portal (mode-jump) proposals into the fixed A. Nothing quantum on the exact side; the
  library is frozen before the chain starts.

## Two-tier structure (amendment A2)

### Tier T — truth, n ≤ 20 (K1 decidable)
- Targets: dense frustrated all-to-all spin glasses, J ~ N(0,1), random fields
  (`random_spin_glass(n, seed, random_fields=True)`), n ∈ {12, 16, 20}, seeds {1, 2, 3}.
- Temperatures: M06 ladder, T = 0.55·√n and 0.35·√n (T_c ≈ √n unnormalized).
- **Miner arms**, each at matched builder wall-clock budget B ∈ {2 s, 10 s} per
  (instance, T):
  1. **Simulated quench** — statevector quench shots (existing E01/D01 kernel) from
     diverse starts; keep distinct low-energy outcomes.
  2. **Exact/CMS member-mining** — solver in member-finding mode ONLY (capped
     enumeration over descending energy windows; NO completeness proofs; UNSAT checked
     before hasSolution()).
  3. **PT-ICM restarts** — parallel-tempering (+ICM pair moves where nontrivial) restarts;
     harvest cold-replica snapshots.
  4. **Greedy+kick** — random-restart steepest descent with random k-flip kicks.
- **Enumerated truth** at n ≤ 20: full 2^n scan. **Mode set** := single-flip local minima
  m of E with Boltzmann mass π(m) ≥ 1e-4 · π(m*) at the chain temperature (m* = argmax).
  **Coverage** of a library := π-mass-weighted fraction of modes m whose basin (steepest
  descent) contains ≥ 1 atom. Report unweighted mode counts too.
- **Downstream chain metric:** darting-chain ESS/sec AND ESS/step on the energy trace
  (magnetization secondary), 4 independent chains, half-split R̂; a cell is reportable
  only if split-R̂ ≤ 1.05, else flagged unconverged. Baselines: (i) plain single-flip MH,
  (ii) PT-ICM chain at the same wall-clock.

### Tier S — scale, n ≈ 50–200, CLASSICAL ARMS ONLY (K2 decidable)
- n ∈ {50, 100, 200}, same generator family, ladder temperatures.
- **Pre-screen (mandatory):** each (instance, T) must show measured classical
  metastability — 4 plain single-flip chains from dispersed starts with split-R̂ > 1.2
  on the energy trace at the screening budget. If baseline chains already mix, the target
  is INVALID for K2 and is discarded before racing (registered here to avoid the E02c
  trap: classical R̂ ≤ 1.03 targets make K2 fail trivially and tell us nothing).
- Arms: **solver-mined PORTAL** vs **PT-ICM baseline chain** at matched total wall-clock
  (builder budget + chain time counted against PORTAL).
- **Simulated-quench arm capped at ~n ≤ 28** (statevector; NO MPS truncation — it
  corrupts the shell physics being measured). Therefore **K1 at scale is undecidable
  without quantum hardware**; the results file must say so explicitly, not fudge it.

## Darting kernel (exactness-valid; registered)

Tjelmeland–Hegstad mode-jump proposal on binary states: with probability w_port pick
atom a_j uniform from A, propose y by flipping each bit of a_j independently with rate
ρ (ρ = 1/n registered); forward and reverse densities are the computable mixtures
  q(z) = (1/m) Σ_j ρ^{d(z,a_j)} (1−ρ)^{n−d(z,a_j)},
so kernels return (y, logq_fwd, logq_rev) and the MH ratio is exact. Alternative allowed
by registration: fixed symmetric portal matching on A with logq ≡ 0. Local component:
single-flip (symmetric, logq = 0). w_port = 0.1 registered; a w_port sweep is diagnostic
only, not a claim surface. Nothing quantum on the exact side.

## Step order

0. This registration (before any run).
1. **A3 acceptance gate (~1 h, BEFORE the race):** on energy-matched atom pairs
   (δE ≲ T, taken from shell states at the M06 ladder temperatures, n ∈ {16, 20}),
   measure portal-jump acceptance. **Gate: mean portal acceptance must reach ≈ 0.2 at
   ladder temperatures; if it cannot, STOP — write the kill memo, skip the race.**
2. Tier T race → decide K1.
3. Tier S race (pre-screen first) → decide K2.
4. P01_RESULTS.md win or lose.

## Kill criteria (verbatim, binding)

- **K1** = classical miners match quench-miner mode coverage at matched budget (Tier T)
  → quantum arm dead.
- **K2** = darting gain < 2× over a PT-ICM baseline chain (Tier S) → whole rung dead.

Both-die is a possible outcome and still feeds QMCMC-Bench as a registered negative.
No advantage claim in any writeup until NOVELTY-CHECK-04 is complete.

## Scope note (amendment A1)

All extraction-pricing claims attached to this experiment are scoped to targets whose
π-evaluation is cheap relative to shot latency (~1e-2 s). This experiment's targets
satisfy that by construction.

## Infra

Long compute in the cloud container venv (not device_bash — background jobs die at
teardown). pysat: EncType.adder forced, negative PB bounds normalized, CryptoMiniSat for
XOR. ExactShellSampler: check UNSAT before reading hasSolution(). Seeds fixed in scripts;
CSV outputs synced back to this directory.
