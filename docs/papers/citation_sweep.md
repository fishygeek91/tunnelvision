# Citation sweep — arXiv:2203.12497 / Nature 619, 282 (2023)

Standing item (P1 before drafting). Grep the citers of Layden et al.
for Bayesian / posterior / variable selection / noise / temperature
so we do not claim novelty that is already in the graph.

Swept 2026-08-14 against OpenAlex work
[W4384009058](https://openalex.org/W4384009058)
(doi:10.1038/s41586-023-06095-4). OpenAlex reports **83 citing works**
(75 unique titles after dropping referee reports and arXiv/journal
duplicates). Semantic Scholar's ~100+ count is the same neighborhood;
the extra entries are mostly the same papers under a second id.

Method: pull title + abstract for every citer, flag
`bayesian|posterior|variable selection|spike-and-slab|inclusion|tempering|replica exchange|noise|temperature`.
Full-text PDFs were not re-read except where we already have notes
(`ferguson2025_coarse_graining.md`). A later pass can deepen the
high-overlap rows below.

## Verdict

**No citer applies qe-MCMC to Bayesian variable selection, spike-and-slab,
or posterior inclusion probabilities.** That application, and the
exact/heuristic wall we use to evaluate it, are not in this graph.

What *is* in the graph, and must be cited rather than rediscovered:

- Mixing can fail even when the quench is symmetric (Orfi & Sels).
- Noise-robustness is claimed; noise-as-a-tempering-resource is not.
- Scale-up past the qubit count is Ferguson et al. (already WP7).
- Closest Bayesian work is multiproposal MCMC for phylogenetics
  and a cosmology posterior — different algorithm, different target.

A negative E02/E03 result is compatible with the theory papers.
It is not scooped by them.

## High overlap (must cite / must not overclaim)

### Orfi & Sels — mixing barriers and speedup bounds

- Alev Orfi & Dries Sels, *Barriers to efficient mixing of
  quantum-enhanced Markov chains*, Phys. Rev. A **110**, 052434 (2024).
  doi:10.1103/physreva.110.052434. arXiv:2408.07881.
- Alev Orfi & Dries Sels, *Bounding the speedup of the
  quantum-enhanced Markov-chain Monte Carlo algorithm*,
  Phys. Rev. A **110**, 052414 (2024). doi:10.1103/physreva.110.052414.
  arXiv:2403.03087.

The spectral gap is bounded by the inverse participation ratio of
classical states in the quench eigenbasis. An ergodic quench
collapses to uniform proposals (no advantage). A perturbative
transverse field is classically simulable (no advantage). Optimal
scaling, when it exists, sits in a fine-tuned non-ergodic window.
The second paper gives a no-go for unital proposals on unstructured
low-temperature sampling.

Relevance: E01 reproduced Layden's *Ising* gap scaling. E02/E03
ask a different target (spike-and-slab) and lose to add-delete-swap.
Orfi & Sels explain why a quench that is "too hot" (our E03 T_eff
result) or too delocalized should not beat a local classical kernel.
Cite before any sentence that sounds like "qe-MCMC should mix
faster on multimodal posteriors."

### Ferguson & Wallden — coarse-graining (WP7)

- Stuart Ferguson & Petros Wallden, *Quantum-enhanced Markov
  chain Monte Carlo for systems larger than a quantum computer*,
  Phys. Rev. Research **7**, 013231 (2025).
  doi:10.1103/physrevresearch.7.013231.

Already in `ferguson2025_coarse_graining.md`. This is the honest
scale-up path after E02c: p=27 quench cannot even construct a 2^p
energy diagonal. Subregion moves + exact global accept/reject.
Verify their symmetry argument from the full PDF before implementing.

### Christmann et al. — quantum-inspired proposals

- J. Christmann et al., *From quantum-enhanced to quantum-inspired
  Monte Carlo*, Phys. Rev. A **111**, 042615 (2025).
  doi:10.1103/physreva.111.042615. arXiv:2411.17821.

Tensor-network (and other approximate) simulators as the proposal
engine, still graded by classical MH. Closest published cousin of
"the surrogate may be wrong; accept/reject is exact" — except they
approximate the *quench*, not a 2-local fit to a non-Ising posterior.
Cite when writing the wall. Do not claim we invented
quantum-inspired MCMC.

### QAOA-enhanced MCMC

- *Markov-chain Monte Carlo method enhanced by a quantum alternating
  operator ansatz*, Phys. Rev. Research **6**, 033105 (2024).
  doi:10.1103/physrevresearch.6.033105.

Same MH envelope, different proposal unitary (QAOA instead of a
randomized quench). Ising / combinatorial. Not Bayesian variable
selection.

### Quantum-annealing proposals

- *Quantum annealing enhanced Markov-Chain Monte Carlo*,
  Sci. Rep. **15** (2025). doi:10.1038/s41598-025-07293-y.

QA as the proposal subroutine. Same envelope, different physics.
Does not use circuit-model depolarizing noise as a temperature
ladder.

### Cosmology posterior

- *Quantum Markov Chain Monte Carlo for Cosmological Functions*,
  IEEE QAI 2025. doi:10.1109/qai63978.2025.00044.

A quantum circuit proposes a *shift in continuous parameter space*
to sample a cosmological posterior. Bayesian, but not discrete
inclusion vectors and not a Layden quench on an Ising surrogate.

### Multiproposal MCMC (Bayesian Analysis)

- *Quantum Speedups for Multiproposal MCMC*, Bayesian Anal. (2025).
  doi:10.1214/25-ba1546. arXiv:2312.01402.

Genuine Bayesian venue. The algorithm is Grover-style selection
among P classical proposals (Tjelmeland), applied to Ising models
on bacterial evolutionary networks (ancestral trait reconstruction).
Not qe-MCMC, not variable selection. Cite as "quantum MCMC has
reached Bayesian Analysis; it has not reached spike-and-slab."

### Annealer + tempering (spin glasses)

- *Accelerating equilibrium spin-glass simulations using quantum
  annealers via generative deep learning*, SciPost Phys. **15**, 018
  (2023). doi:10.21468/scipostphys.15.1.018.

Uses a D-Wave annealer to propose, and discusses tempering in the
classical spin-glass sense. Not "hardware noise is a temperature."
E03's Maxwell's Daemon claim (noise heats the *proposal*) is still
distinct — and our exact-tier measurement said it does not help.

## Noise / temperature hits that are not our claim

Many citers mention noise (error mitigation, depolarizing after
gates, shot noise in VQE/QAOA) or temperature (Boltzmann / Gibbs
sampling of an Ising model). None treat a *ladder of fixed
depolarizing strengths as a tempering schedule on a Bayesian
target*. That is the E03 question; the graph does not contain a
prior measurement. Our answer at p=10 was negative
(`results/E03/summary.md`, `results/E03/pt/summary.md`).

## Incidental

The remaining ~60 unique titles are reviews, FinTech/insurance
surveys, photonic Ising machines, QML, ecology/omics think-pieces,
and unrelated Markov-chain applications that cite Layden as
"quantum computers can sample." They do not constrain our claims.
The unique-title list from this sweep is in the OpenAlex page for
W4384009058; re-pull before a submission in case the count moves.

## What we can still say, and what we cannot

Safe:

- First application of a Layden quench, driven by a 2-local
  surrogate, to exact spike-and-slab variable selection.
- Exact/heuristic wall: surrogate quality may only affect speed.
- Negative result vs add-delete-swap at p=10 (exact gap) and
  p=20 (sampled, unconverged quench, four orders of ESS/sec).
- Noise heats the proposal (T_eff); a fixed ladder does not beat
  its coldest rung or ADS.

Not safe without a hedge:

- "qe-MCMC is unexplored in Bayesian statistics" — the
  multiproposal paper in *Bayesian Analysis* and the cosmology
  posterior paper exist. Say *variable selection*.
- "We introduce quantum-inspired MCMC" — Christmann et al. 2025.
- "Coarse-graining to p~50 is new" — Ferguson et al. 2025 is WP7.
- Any mixing-advantage claim that ignores Orfi & Sels.
