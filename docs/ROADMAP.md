# Roadmap

Detailed enough to pick up a work package cold. Read `docs/ARCHITECTURE.md`
first (5 min) — especially the exact/heuristic wall — then claim a package.

**Legend** — Difficulty: 🟢 mechanical · 🟡 requires care · 🔴 easy to get subtly wrong.
Priority: P0 blocks everything downstream · P1 on the critical path · P2 parallelizable/deferrable.
⚠️ marks the tricky parts: read those notes before writing code, they are the
places a plausible-looking implementation silently breaks the science.

Dependency spine: `WP1 → WP2 → WP3 → WP4 → E01 → WP5 → E02 → WP6/E03 → WP7`.
WP2/WP3 can start against WP1's interfaces before WP1 is finished; WP5 can
start any time after WP1 (it's pure statistics, no quantum).

---

## Rung 1 — Foundations + Layden reproduction (~2 weeks)

### WP1 — Engine + diagnostics (P0, 🟡)

The exactness boundary. Pure numpy/scipy. Everything else is judged by this code.

- [x] `engine.run()`: MH loop over binary vectors. Accept prob
      `min(1, exp(Δlogp + log_q_rev − log_q_fwd))`. Store states as uint8,
      preallocate arrays, one `np.random.Generator` seeded per run.
- [x] `engine.transition_matrix()`: full 2^n × 2^n matrix from a kernel's
      `proposal_matrix()`. Row = current state; off-diagonals
      `Q[x,y]·A[x,y]`; diagonal = 1 − Σ(off-diagonals).
      ⚠️ **Rejection mass belongs on the diagonal.** Forgetting the
      self-loop term is the classic bug — rows won't sum to 1 and the
      spectral gap comes out wrong but not obviously wrong. Assert row sums
      to 1 within 1e-12 in the function itself, not just in tests.
- [x] `diagnostics.spectral_gap()`: δ = 1 − |λ₂|.
      ⚠️ The MH matrix is not symmetric — either use `scipy.linalg.eig`
      (dense, fine at n≤14) or similarity-transform to a symmetric matrix
      via π^{1/2} (reversible kernels only). Verify `π @ P ≈ π` first and
      raise loudly if it fails: a stationarity failure here means a kernel
      is lying about its log-q, and you want that caught at the diagnostic,
      not in a paper draft.
- [x] `diagnostics.integrated_autocorrelation_time()` / `ess()`: standard
      windowed IAT estimator (Sokal's adaptive window, c≈5).
      ⚠️ Naive full-length autocorrelation sums are noise-dominated;
      don't skip the windowing.
- [x] `diagnostics.pip_error()`: max-abs error of estimated inclusion
      probabilities vs. exact.

### WP2 — Classical kernels (P0, uniform/single-flip 🟢, AddDeleteSwap 🔴)

- [x] `UniformFlip`, `SingleFlip`: symmetric, return `(y, 0.0, 0.0)`, plus
      trivial `proposal_matrix()` implementations.
- [x] `AddDeleteSwap`: the field-standard baseline and the one we must not
      get wrong — beating a buggy baseline is worse than no result.
      ⚠️ **Asymmetric at boundaries.** From the empty model, "delete" is
      unavailable; from the full model, "add" is unavailable; swap counts
      depend on |γ|·(p−|γ|). The forward/reverse proposal probabilities
      differ whenever the move changes |γ|. Write out q(y|x) and q(x|y)
      explicitly per move type; unit-test at the empty model, full model,
      and |γ|=1 — those three states catch every accounting mistake.
- [x] `proposal_matrix()` for AddDeleteSwap (needed for exact-tier E02a).

### WP3 — Targets (P0, Ising 🟢, spike-and-slab 🟡)

- [x] `IsingTarget` + `random_spin_glass()` (all-to-all and 2D, J~N(0,1),
      seeded), `enumerate_exact()` via bit-tricks over 2^n states —
      vectorize, don't loop in Python (n=14 is 16k states, fine; n=20 is 1M,
      still fine vectorized).
- [x] `SpikeSlabTarget.log_prob()`: closed-form g-prior marginal
      log-posterior.
      ⚠️ **Numerics, not statistics, is the hazard.** Use
      `slogdet`/Cholesky, never `det`; center-and-scale X and y once at
      construction; handle the empty model (γ=0) as its own closed-form
      case; cache `X'X` and `X'y` at construction so per-state cost is
      O(|γ|³) not O(np²). Validate against a 3-predictor case worked out
      by hand or in R (BAS/BMS packages give reference numbers).
- [x] `posterior_inclusion_probs_exact()` by enumeration (p ≤ 20).

### WP4 — Exactness tests + quench kernel (tests P0 🟡, quench P0 🔴)

- [x] Implement all five invariants specced in `tests/test_exactness.py`.
      ⚠️ Test 4 (deliberately broken kernel stays exact) is the one that
      catches engine bugs — a kernel that proposes from the *wrong*
      distribution but reports *honest* log-q must still pass. If it
      doesn't, the engine is broken, full stop.
- [x] `QuenchKernel` statevector path, following
      `docs/papers/layden2023_qemcmc.md` to the letter: H(γ) with
      Frobenius-normalized α; γ~U[0.25,0.6], t~U[2,20] fresh per proposal;
      second-order symmetric Trotter, Δt=0.8. Exact `expm` path for E01
      (Nature Fig. 2 is not Trotterized).
      ⚠️ **Three separate symmetry traps.** (1) First-order Trotter breaks
      proposal symmetry — the accept ratio silently stops being valid. Only
      the symmetric second-order splitting is allowed. (2) `proposal_matrix()`
      must average |⟨y|U|x⟩|² over the *same* (γ,t) distribution the sampler
      draws from — use a fixed quadrature or a large fixed sample, seeded.
      (3) Do the matrix symmetry check `Q ≈ Q.T` as a test; if it fails,
      the Trotter ordering is wrong.
- [x] Aer shots path (same circuit, sampled) — needed for n>14 and all
      noise work later. Keep statevector and shots paths behind one class
      so E01–E03 code doesn't care which is running.

### E01 — Layden reproduction (P0, 🟡) — **the Rung 1 gate**

- [x] `run.py` + `config.yaml`: n ∈ {5,…,10},
      T ∈ {0.1, 0.3, 1.0}, ≥20 seeded instances per (n,T); kernels:
      uniform, single-flip, quench. Exact spectral gap for every instance.
- [x] Fit ⟨δ⟩ ∝ 2^{−kn}; target: quench k ≈ 0.29 vs uniform k ≈ 1.0 at
      T=0.1, advantage shrinking as T grows.
      ⚠️ Average the gap over instances in log space (geometric mean) —
      arithmetic means are dominated by easy instances and flatten k.
      ⚠️ Fit n=5–10, not just 8–10 — the short lever arm overestimates k.
- [x] `results/E01/summary.md` with the k-exponent table and the gap-vs-n
      plot. **Gate: qualitative match to Nature Fig. 2 (k_quench ≈ 0.32 vs
      k_uniform ≈ 1.02 at T=0.1; ratio 3.2). Rung 1 passed.**

---

## Rung 2 — TunnelVision demo (~2 weeks)

### WP5 — Surrogate (P1, 🟡) — parallelizable with WP4

- [x] `ising_surrogate_from_data()`: fields h_j from marginal correlations
      |x_j'y| (scaled), antiferromagnetic couplings J_jk from collinearity
      (X'X off-diagonals), sparsity field from prior log-odds.
      ⚠️ **Scale is the whole game.** The quench dynamics care about the
      surrogate's energy scale relative to H_mix (the α normalization
      handles part of this, but a surrogate 100× too hot or cold proposes
      garbage). Sanity check before any chain runs: the surrogate's exact
      ground state (enumerate at p=10) should be a high-posterior model,
      and surrogate-energy vs. exact-log-posterior over all 1024 models
      should correlate strongly (plot it — this one scatter plot is the
      cheapest possible de-risking of the entire project).
- [x] `learned_surrogate()`: ridge-fit (h, J) on exact log-posterior values
      at random γ samples. 🟢 once the analytic one exists.
- [x] Remember (cursor rule): surrogate quality may only ever affect
      SPEED. If changing the surrogate changes posterior estimates beyond
      Monte Carlo error, something crossed the wall — treat as P0 bug.

### E02 — the headline demo (P1, 🟡)

- [x] `data/loaders.py`: diabetes (sklearn, p=10, standardized) and
      `correlated_synthetic()` (equicorrelated or AR(1) design, tunable ρ,
      sparse true support, fixed SNR).
- [x] E02a: exact tier at p=10 — spectral gap for all five kernels
      (uniform, single-flip, ADS, quench-analytic, quench-learned), PIP
      accuracy vs. enumeration, ESS/step and ESS/wall-clock-sec.
      Full diabetes figure is in `results/E02/summary.md`.
      ⚠️ Report per-step AND per-second: the quench kernel pays a large
      constant factor in simulation; hiding that is the quantum-papers
      failure mode we built this repo to avoid.
- [x] E02b: ρ-sweep, ρ ∈ {0, 0.3, 0.5, 0.7, 0.9}, p=10 exact.
      Story curve in `results/E02/rho_sweep/summary.md`.
      ⚠️ At high ρ, chains stick badly — use R̂ across ≥4 independent
      chains and long burn-in for the sampled tier; a stuck chain that
      looks converged is the subtlest way this experiment lies to you.
- [x] Ablations (P2, parallelizable): Aer depolarizing noise sweep and
      surrogate-quality sweep, exact tier. Write-up in
      `results/E02/ablations/summary.md`. λ ≤ 0.1 barely moves the
      already-losing gap ratio; per-gate p = 0.005 already maps to
      λ̂ ≈ 0.65. Wrecking the learned surrogate kills the gap and
      leaves PIP error at Monte Carlo (wall holds).
- [x] E02c: sampled tier, p ∈ {20, 27}, ρ ∈ {0.5, 0.9}. Live
      Trotter quench at p=20 is 2–3 s/propose and unconverged
      (R̂ 1.45–1.74, PIP error 0.24–0.50). ADS mixes in ~1 s.
      p=27 quench is a construction refusal (`problem_energies`
      enumerates 2^p). Write-up in `results/E02/sampled/summary.md`.
- [x] `results/E02/summary.md`. **Gate: negative at p=10 exact
      tier, and the sampled tier does not turn it.** Quench never
      beats ADS; learned/ADS gap ratio grows 0.33× → 0.42× with ρ
      but does not cross 1. At p=20 the quench pays seconds per
      proposal and does not mix. Written up in
      `results/E02/rho_sweep/summary.md` and
      `results/E02/sampled/summary.md`.

---

## Rung 3 — Maxwell's Daemon + hardware (open-ended)

### E03 — noise as tempering (P1 after E02, 🔴)

- [x] `NoiseLadderKernel`: K quench kernels at fixed depolarizing
      levels λ ∈ {0, 1e-3, 5e-3, 2e-2, 1e-1}, random-scan composition.
      ⚠️ **Fixed levels only.** A mixture of fixed symmetric kernels chosen
      independently of the current state is symmetric; choosing the rung
      *based on chain state or history* breaks that and needs either
      explicit log-q accounting or diminishing-adaptation theory. Start
      rigid; earn flexibility later.
- [x] The physics measurement: effective proposal temperature vs. noise
      strength (fit proposal energy-change distributions per rung). This
      is the figure that names the paper. T_eff rises with λ
      (diabetes 1.86 → 2.28; ρ=0.9  1.70 → 2.04).
- [x] Compare: single best rung vs. ladder vs. classical parallel
      tempering, exact tier at p=10 + hardest ρ=0.9 instance.
      Rung-vs-ladder is in `results/E03/summary.md` (ladder matches
      its best rung, loses to ADS). Parallel tempering and the
      Aer-realistic hot rung (λ̂ = 0.65) are in
      `results/E03/pt/summary.md`. PT raises ESS/step; charged per
      target-eval it is 1.44× ADS on diabetes and 0.54× at ρ=0.9.
      The hot rung shrinks the gap. Nothing beats ADS.
- [x] `results/E03/summary.md` — outcome at p=10 exact tier: **hurts /
      no help**. Noise heats the proposal; the ladder does not beat
      its coldest rung; nothing beats ADS. The hotter Aer-realistic
      λ̂ = 0.65 rung and classical PT are in `results/E03/pt/summary.md`
      and do not change the headline.

### WP6 — Hardware (P2 until E02 gates, 🔴)

- [ ] `HardwareQuenchKernel` via qiskit-ibm-runtime; batched proposal
      pools keyed by state, cached (states revisit constantly at p=10 —
      cache hit rate will be high; log it).
      ⚠️ Credentials via env only — the .gitignore and cursor rules
      already enforce this; don't route around them in notebooks.
- [ ] **The symmetric-q bias audit** (see ARCHITECTURE §2): amplitude
      damping is non-unital ⇒ hardware proposals are only approximately
      symmetric ⇒ our "exactness" claim needs an empirical bound. Run E02a
      on hardware, compare sampled posterior to enumerated truth, report
      the total-variation deviation. ⚠️ This audit is mandatory before ANY
      hardware claim leaves the repo. It is the project's single biggest
      correctness risk.
- [ ] Physical noise rungs for E03 phase 2: DD on/off, twirling levels,
      inserted idle time.

### WP7 — Scale-up + adaptive (P2, 🟡)

- [ ] Coarse-graining per PRR 7, 013231 (pull full text first — notes in
      `docs/papers/ferguson2025_coarse_graining.md` flag what to verify,
      especially the subregion-choice symmetry argument) → p ≈ 50–100.
- [ ] `AdaptiveMixture` (N3) under diminishing adaptation
      (Roberts & Rosenthal 2007) — only if E02/E03 show there's value to
      allocate adaptively.

---

## Standing items (anyone, anytime)

- [x] **Citation-graph sweep** of arXiv:2203.12497 / Nature 619, 282
      (OpenAlex W4384009058, 83 citers / 75 unique titles). No citer
      does Bayesian variable selection. Must-cites: Orfi & Sels
      (mixing barriers), Ferguson (coarse-graining), Christmann
      (quantum-inspired proposals). Notes in
      `docs/papers/citation_sweep.md`. (P1 before drafting.)
- [ ] Keep `docs/papers/` notes updated when anyone reads a full PDF.
- [ ] `uv.lock` committed and current; every result records commit + config
      + seed (cursor rule 10).

## Suggested team split

Two people, minimal collisions: **A** takes WP1→WP2→E01 (statistics/
engine track), **B** takes WP3(spike-slab)→WP5→E02 prep (Bayesian track);
whoever finishes first takes WP4's quench kernel (the 🔴 one — pair on the
symmetry tests). Solo: follow the spine in order, skipping nothing.
