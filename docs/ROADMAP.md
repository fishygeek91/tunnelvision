# Roadmap

## Rung 1 — Foundations + Layden reproduction (target: ~2 weeks)

- [ ] `engine.py`: run() + transition_matrix(); `diagnostics.py`: spectral gap, IAT, ESS, PIP error
- [ ] Classical kernels (uniform, single-flip, add-delete-swap with boundary log-q)
- [ ] `targets/ising.py` + instance generators; `targets/spike_slab.py` exact log-posterior + enumeration
- [ ] Exactness invariant tests (tests/test_exactness.py) — all green before anything else
- [ ] `kernels/quantum.py` simulator quench (statevector n≤14, Aer shots beyond)
- [ ] **E01**: reproduce Layden spectral-gap scaling on n=8–12 spin glasses → results/E01/summary.md

Gate to Rung 2: E01 speedup curve qualitatively matches Nature 619, 282 Fig. 3.

## Rung 2 — TunnelVision demo (target: ~2 weeks)

- [ ] `surrogate.py` analytic version; sanity: surrogate ground state ≈ high-posterior models
- [ ] **E02a**: diabetes p=10 — exact spectral gap of all kernels; headline plot
- [ ] **E02b**: synthetic rho-sweep — advantage vs. predictor correlation
- [ ] Learned surrogate comparison; noisy-simulator ablation (does Aer noise kill the gap gain?)

Gate to Rung 3: quench beats add-delete-swap on spectral gap somewhere honest
(if not: write the negative result up carefully — still publishable).

## Rung 3 — Maxwell's Daemon + hardware (open-ended)

- [ ] **E03a**: noise-ladder vs. classical parallel tempering, Aer noise models, exact tier
- [ ] Characterize effective proposal temperature vs. noise strength (the physics measurement)
- [ ] Hardware runs (IBM open plan): E02a on 10 qubits; symmetric-q bias audit (ARCHITECTURE §2)
- [ ] Coarse-graining (PRR 7, 013231) for p ≈ 50–100
- [ ] AdaptiveMixture (N3), if E02/E03 justify it

## Paper checkpoints

- Before claiming novelty in writing: citation-graph sweep of arXiv:2203.12497
  (owed — see NOVELTY-CHECK-01 in the parent project folder)
- Candidate titles: "TunnelVision: exact Bayesian variable selection with quantum
  tunneling proposals" / "Maxwell's Daemon: hardware noise as a tempering resource"
