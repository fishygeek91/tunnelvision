# NOVELTY-CHECK-01 — Prior-art review for N1 and N2, and the chosen demo

*QuantumAlgorithms project · August 14, 2026 · companion to DEEPDIVE-01*

## Verdict summary

| Idea | Verdict | Confidence |
|---|---|---|
| N1 — qe-MCMC as Bayesian posterior sampler (spike-and-slab) | **Novel as an application; adjacent theory exists** | Medium-high |
| N2 — Noise-as-tempering (scheduled hardware noise as proposal-temperature ladder) | **Novel in gate-based qe-MCMC; adjacent idea exists in annealers** | Medium |

Neither idea was found already done. Both have neighbors that must be cited and differentiated. Full-text citation sweep of everything citing Layden 2203.12497 is still owed before any paper claim (see Residual risk).

## N1 prior art — what exists and why we're different

**Closest neighbors found:**

1. *A Quantum Parallel MCMC* (Holbrook, JCGS 2023) and *Quantum Speedups for Multiproposal MCMC* (Bayesian Analysis, 2025). These target Bayesian inference — but via **Grover-style multiproposal search**, a fault-tolerant-oriented primitive with oracle calls. Not Layden-style quench proposals, not runnable meaningfully on today's hardware.
2. *Quantum Dynamical Hamiltonian Monte Carlo* (arXiv:2403.01775) — continuous parameter spaces, HMC analog. Different regime (continuous vs. our discrete model space).
3. *Quantum-enhanced MCMC for Combinatorial Optimization* (arXiv:2602.06171) — the most active follow-on to Layden; extends to Max Independent Set with warm-starts + parallel tempering on IBM hardware (117 vars). Optimization, **not posterior sampling**; they want the optimum, we want the whole distribution.
4. *Irreversible qe-MCMC* (arXiv:2606.23350) — spectral-gap improvement via broken detailed balance; still Ising benchmarks only.
5. RBM training with D-Wave samplers (Frontiers in Physics 2021 etc.) — uses annealer *as the sampler* (biased, needs effective-temperature estimation). We keep the sampler exact and use quantum only for proposals — the differentiator.

**What nobody did:** use the Layden quench proposal inside an exact Metropolis–Hastings chain over a *statistical model space* (spike-and-slab inclusion vectors), with the target being a real Bayesian posterior rather than an Ising Boltzmann distribution. Every published qe-MCMC target found is a spin glass or a graph-optimization energy.

**The technical trick that makes N1 work (and is itself a contribution):** the exact spike-and-slab marginal posterior over inclusion vectors γ involves log-determinants — not a 2-local Ising energy, so it can't drive the quench Hamiltonian directly. But it doesn't have to. Build a **2-local Ising surrogate** from the data (fields from marginal correlations |x_j'y|, couplings from predictor collinearity X'X) and use *that* to shape the quantum proposal; the accept/reject step always evaluates the *exact* posterior. Metropolis–Hastings guarantees exactness under any proposal — a surrogate-driven quantum proposal is still an exact sampler. Surrogate-proposal / exact-target separation is standard in classical MCMC but unexplored in qe-MCMC.

## N2 prior art — what exists and why we're different

**Closest neighbor:** *High-quality Thermal Gibbs Sampling with Quantum Annealing Hardware* (arXiv:2109.01690, PR Applied 2022) — D-Wave line of work treating annealer noise/freeze-out as an *effective temperature*, estimating it, and sampling from the device's native distribution. Crucial differences: (a) annealer, not gate-based; (b) they sample **from** the noisy device and must estimate its bias — no exactness; (c) noise level is a nuisance parameter to be measured, not a **scheduled resource** in a ladder.

In the gate-based qe-MCMC literature: Layden et al. study noise *robustness* (speedup degrades gracefully); arXiv:2602.06171 explicitly *mitigates* noise (1/√υ extra samples); the IQEMC paper doesn't touch it. **No one found who deliberately schedules noise strength (via dynamical decoupling on/off, twirling levels, or idle-time insertion) to create a proposal-temperature ladder with Metropolis exactness intact.** That's the N2 claim, narrowed and defensible.

**Honest caveat:** the physics intuition (noise ≈ heating) is folklore; the novelty is the *construction* (hardware-native proposal-tempering ladder + exactness) and its empirical characterization. The first task of any N2 write-up is measuring whether noise-heated proposals actually help mixing or just wash out the quantum tunneling structure that creates the speedup. That's a real scientific question with a publishable answer either way.

## Residual risk before claiming novelty in writing

- Sweep all papers citing arXiv:2203.12497 (Semantic Scholar / Inspire, ~100+ citations by now) — grep for "Bayesian", "posterior", "variable selection", "noise", "temperature". (Search-engine pass done 2026-08-14 found nothing; a citation-graph pass is stronger.)
- Check QC-for-statistics workshops (Q4Stat, quantum ML venues) for unpublished preprints.

## The chosen practical demonstration

**Demo: Bayesian variable selection on a real dataset with a quantum proposal engine — exact answers, measurably faster mixing.**

- **Problem:** spike-and-slab variable selection for linear regression. Model space = 2^p inclusion vectors. This is a real tool used daily in genomics/econometrics; multimodality (correlated predictors → competing models) is exactly what defeats classical single-flip Metropolis.
- **Dataset:** start with the classic diabetes dataset (p = 10 → 10 qubits, ground truth enumerable: 1,024 models, so we can compute the *exact* posterior and the *exact* spectral gap of every kernel — airtight evaluation). Scale-up target: a correlated genomics subset with p = 20–27 (simulator ceiling) and p up to ~100 via the coarse-graining scheme on hardware.
- **Comparison arms:** (1) uniform-flip Metropolis, (2) single-flip Metropolis, (3) add-delete-swap (the field-standard kernel), (4) quantum quench proposal (simulated, then IBM Heron), (5) N2 variant: noise-ladder tempering vs. standard parallel tempering.
- **Metrics:** spectral gap (exact at p=10), effective sample size per chain step and per wall-clock second, posterior inclusion probability error vs. exact enumeration.
- **Success criterion:** quantum proposal beats add-delete-swap on spectral gap at p=10 on correlated designs, and the advantage grows with predictor correlation strength. Honest failure is publishable too ("where quantum proposals do and don't help statisticians").
- **Why this is the right demo:** real data, exact ground truth available, the baseline is the *actual tool practitioners use* (not a strawman uniform sampler — the mistake most quantum-advantage demos make), runs end-to-end on a laptop simulator before any hardware spend, and the deliverable doubles as the `qemcmc` package from Rung 1.

## Sources

- Layden et al., Nature 619, 282 (2023) / arXiv:2203.12497
- Holbrook, *A Quantum Parallel MCMC*, JCGS 32(4) (2023)
- *Quantum Speedups for Multiproposal MCMC*, Bayesian Analysis (2025)
- arXiv:2403.01775 — Quantum Dynamical HMC
- arXiv:2602.06171 — qe-MCMC for Combinatorial Optimization (IBM, 117 vars)
- arXiv:2606.23350 — Irreversible qe-MCMC
- arXiv:2109.01690 / PR Applied 17, 044046 (2022) — annealer thermal Gibbs sampling
- Frontiers Phys. 9:589626 (2021) — RBM training with D-Wave
- PRR 7, 013231 (2025) — coarse-grained qe-MCMC
