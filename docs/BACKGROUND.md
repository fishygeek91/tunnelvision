# Background & bibliography

Research context lives in the parent project folder (QuantumAlgorithms/):
DEEPDIVE-01_quantum-enhanced-mcmc.md (why this bet) and NOVELTY-CHECK-01.md
(prior art, novelty verdicts, demo rationale). Summary of the essentials:

## Must-read, in order

1. Layden et al., *Quantum-enhanced MCMC*, Nature 619, 282 (2023); arXiv:2203.12497 — the algorithm. Read the SI for the symmetry argument and noise analysis.
2. PRR 7, 013231 (2025) — coarse-graining: n-qubit proposals for N≫n systems.
3. arXiv:2602.06171 — qe-MCMC + warm starts + parallel tempering, MIS on IBM hardware (117 vars). Closest active competitor; they do optimization, we do posteriors.
4. arXiv:2606.23350 — irreversible qe-MCMC (spectral-gap gains from breaking detailed balance). Compatible with our engine if log-q accounting generalized.
5. Holbrook, JCGS 32(4) (2023) + Bayesian Analysis (2025) — quantum multiproposal MCMC via Grover: the fault-tolerant cousin. Cite and differentiate.
6. arXiv:2109.01690 / PR Applied 17, 044046 — D-Wave effective-temperature Gibbs sampling: nearest neighbor to Maxwell's Daemon; they measure noise, we schedule it, and they lack exactness.
7. George & McCulloch (1993), *Variable selection via Gibbs sampling*, JASA — spike-and-slab canon; add-delete-swap baseline conventions.

## Framing sentences worth keeping

- Noise degrades the speedup, never the answer.
- The surrogate shapes the dream; the posterior grades it.
- Baselines are the real tools statisticians use, not strawmen.
