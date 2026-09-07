# D01 — Dequantizing the quench (2026-08-25)

Question: is the quench's E01 win (Ising, low T) reproducible classically, exactly, and cheaply? IDEATION-01 bet on Creutz's demon (energy-conserving involutive dynamics). Answer: **no — but the experiment identified what the quench actually does, and that is the useful result.**

## 1. DEMON (Creutz-demon involutive proposal): negative, with a ceiling

Implementation: `src/tunnelvision/dynamics/demon.py`. Extended state (x, d), d ~ Exp(1) refreshed each step; deterministic sweep flips site i iff T_d·d ≥ ΔE; reverse sweep is the exact inverse. Verified: involution (100–200 random trajectories, |Δd| < 1e-12), stationarity πP = π (engine gate), acceptance ≡ 1 at T_d = 1, propose() agrees with exact `proposal_matrix()` to < 0.01 (tests/test_dynamics.py, 12/12 pass).

Exact spectral gaps, n=8 all-to-all spin glass with fields, seed 1:

| kernel | T=1.0 | T=0.3 |
|---|---|---|
| single-flip | 9.0e-4 | 7.6e-10 |
| add-delete-swap | 4.2e-3 | 1.2e-5 |
| uniform | 1.1e-2 | 7.3e-3 |
| **quench** | 7.7e-2 | 8.5e-2 |
| demon L=8, T_d=1 | 7.5e-3 | 8.3e-9 |
| demon L=4, T_d=16 | 1.1e-2 | 1.9e-4 |

At T_d = 1 the demon gap is exactly single-flip × L: it is a local microcanonical walk. A hot demon (T_d = 8–16, with the engine's Hastings correction keeping it exact) buys ~10⁵× at T=0.3 but is still 400× below the quench, and costs L evaluations per step.

Why it cannot work, in one line: rejection-free requires the demon prior to be Exp(1), which fixes the energy budget at ≈ T; any prior that allows "debt" (climbing) reintroduces the Metropolis penalty exp(−ΔE(1−1/T_d)). Deterministic local dynamics is the wrong dequantization.

## 2. The mechanism: the quench is a global isoenergetic proposal

Oracle kernel: propose uniformly from S_ε(x) = {y ≠ x : |E(y) − E(x)| ≤ ε} (ε in raw coupling units; Hastings-corrected by |S_ε|). Exact gaps, all-to-all spin glass with fields:

| instance | quench | shell ε=2 | ε=3 | ε=4 | ε=6 | uniform |
|---|---|---|---|---|---|---|
| n8 s1 T=0.3 | 0.085 | 0.11 | **0.15** | 0.11 | 0.053 | 0.0073 |
| n8 s1 T=0.1 | 0.090 | 0.14 | **0.19** | 0.12 | 0.060 | 0.0064 |
| n8 s3 T=0.1 | 0.063 | **0.24** | 0.17 | 0.091 | 0.059 | 0.0039 |
| n10 s1 T=0.3 | 0.034 | 0.077 | **0.089** | 0.070 | 0.029 | 0.0016 |
| n10 s2 T=0.1 | 0.073 | **0.15** | 0.13 | 0.077 | 0.048 | 0.0010 |
| n10 s3 T=0.1 | 0.045 | 0 (disconnected) | **0.19** | 0.16 | 0.12 | 0.0016 |

The shell oracle at ε ≈ 2–4 **matches or beats the quench on every instance**, is T-independent like the quench, and is 10–100× better than uniform. ε ≲ 1 disconnects the state space (gap 0). So: the quench's advantage over uniform is that it lands on states of similar *target* energy, far away; its advantage over local kernels is that it lands far away. Not "energy-conserving dynamics" (the demon has that) — **global shell sampling**.

## 3. Shell-MTM: a classical, exact shell approximation — partial

`src/tunnelvision/dynamics/mtm.py`: multiple-try Metropolis with symmetric random-mask candidates and symmetric weight λ = (π(x)π(y))^{−a}·exp(−(ΔE/ε)²), i.e. selection ∝ π(y)^{1−a}·exp(−(ΔE/ε)²). Exact (Liu–Liang–Wong 2000); 2K−1 target evaluations per step. Exactness verified against enumeration (test_dynamics.py).

Gated sampled tier (bench2.py): n=10, 4 chains from random inits, 50k steps, split-R̂ on energy, ESS per *target evaluation* (the honest classical cost), |ΔM| = magnetization error vs enumeration. Quench proposals drawn from its exact proposal matrix (identical kernel; its wall-clock is therefore NOT a QPU number).

T = 0.3:

| arm | seed | R̂ | τ_E | evals/step | ESS_E / eval | |ΔM| |
|---|---|---|---|---|---|---|
| single-flip | 1 / 2 | 2.96 / 1.00 | 19 / 115 | 1 | (5.2e-2) / 8.7e-3 | 0.067 / 0.000 |
| add-delete-swap | 1 / 2 | 2.78 / 2.89 | 45 / 1619 | 1 | (2.2e-2) / (6.2e-4) | 0.067 / 0.066 |
| uniform | 1 / 2 | 1.01 / 1.02 | 611 / 567 | 1 | 1.6e-3 / 1.8e-3 | 0.043 / 0.002 |
| demon L=4 T_d=8 | 1 / 2 | 2.45 / 1.00 | 25 / 73 | 5 | (8.0e-3) / 2.7e-3 | 0.182 / 0.001 |
| **quench** | 1 / 2 | 1.00 / 1.00 | 33 / 20 | 1 | **3.0e-2 / 5.0e-2** | 0.003 / 0.000 |
| mtm K=8 greedy (a=0) | 1 / 2 | 1.00 / 1.00 | 20 / 18 | 15 | 3.3e-3 / 3.7e-3 | 0.001 / 0.000 |
| mtm K=8 shell ε=3 a=0.5 | 1 / 2 | 1.00 / 1.00 | 23 / 20 | 15 | 2.9e-3 / 3.3e-3 | 0.005 / 0.000 |
| mtm K=4 shell ε=3 a=0.5 | 1 | 1.00 | 33 | 7 | 4.3e-3 | 0.009 |

T = 0.1: same ordering; quench 3.0e-2 / 6.3e-2 per eval, best MTM 5.9e-3 (greedy, seed 2), uniform 2.4e-3–2.7e-3; single-flip and ADS fail the R̂ gate (2.5–6.4) with |ΔM| up to 0.25. Parenthesised numbers are from chains that failed R̂ and are not ESS.

Reading: (i) every local kernel, and the demon, fails the R̂ gate at low T — the numbers that look best per eval are the stuck ones, exactly the failure mode the protocol exists to catch; (ii) MTM arms converge and match the quench *per step* (τ_E ≈ 20 vs 20–33), which confirms the mechanism, but at 15 evaluations per step they are 7–15× below the quench *per evaluation* and only 2× above uniform; (iii) the shell weight adds nothing over greedy MTM at K=8 — random masks of weight ~n/2 rarely land on the shell, so the selection has little to choose from. The quench concentrates amplitude on the shell for free; a classical sampler has to *find* it.

## 4. Verdict and what's next

- Creutz demon: closed, negative, with a clean reason. Don't retry variants.
- Mechanism: **established** in the exact tier. This is the sharpest statement so far of what Layden's quench does, and it is a strong paper section: the quantum proposal ≈ uniform sampling of the target's energy shell at Hamming distance, width ≈ coupling scale.
- Open classical question, now well-posed: *sample the energy shell at distance with O(n)–O(n²) work.* Candidates: (a) MTM with shell-biased candidate generation (masks built from low-|ΔE| sites, still symmetric via mixture), (b) tempered-transition style up/down annealing with the shell as the turning point, (c) Wang–Landau-style flat-energy walks as proposal generators with computable Hastings terms. Any of these that reaches ~quench per-eval at n≥20 is a real, deployable classical algorithm — and would close the last claimed quantum advantage in this family.
- Per wall-clock nothing changes: the quench's per-eval edge (≈10×) is dwarfed by the 10⁴× shot cost measured in E02c.

Note: seed 3 MTM arms were still running when the logs were snapshotted (seed 3 local kernels: R̂ 8–37, |ΔM| 0.24 — same story).

Artifacts: `src/tunnelvision/dynamics/{demon,mtm}.py`, `tests/test_dynamics.py`, `experiments/D01_demon/{bench,bench2}.py`, logs `results_T0.3.log`, `results_T0.1.log`.
