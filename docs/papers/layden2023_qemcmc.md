# Layden et al. — Quantum-enhanced MCMC (Nature 619, 282 (2023); arXiv:2203.12497)

**The algorithm this repo implements.** Implementation-critical facts, extracted 2026-08-14:

## Proposal construction

- Hamiltonian: `H(γ) = (1-γ)·α·H_prob + γ·H_mix`, with `H_mix = Σ_j X_j` and
  `H_prob` the classical Ising energy encoded as Z-terms.
- `α = ||H_mix||_F / ||H_prob||_F` — normalizes the two terms to comparable scale.
  (NOTE: in our scaffold `kernels/quantum.py` initially used "alpha" for the mixing
  parameter — the paper calls the mixing parameter γ and reserves α for this
  normalization. Follow the paper's convention when implementing.)
- Parameter randomization per proposal (no tuning): `γ ~ U[0.25, 0.6]`, `t ~ U[2, 20]`.
- Propose: prepare |s⟩, apply U = e^{-iHt}, measure → s'.

## Symmetry

U must satisfy |⟨s'|U|s⟩| = |⟨s|U|s'⟩| for all pairs ⇒ Q(s'|s)/Q(s|s') = 1
⇒ log-q terms are exactly zero in MH. Holds for real symmetric H evolved
directly; Trotterization must preserve it (see below).

## Trotterization

Second-order product formula, timestep Δt = 0.8:
`V ≈ e^{-iH1Δt} (e^{-iH2Δt} e^{-iH1Δt})^{r-1}` — needs only r−1 two-qubit gate
layers rather than 2r. Second-order symmetric ordering preserves proposal symmetry.

## Hardware & instances (their experiment)

ibmq_mumbai (27q transmon), n = 8–10, 1D nearest-neighbor model instances
matched to device connectivity, pulse-efficient RZZ via scaled echoed
cross-resonance.

## Result to reproduce (E01 gate)

Average spectral gap on fully-connected Ising: ⟨δ⟩ ∝ 2^{-kn}. At T = 0.1:
quantum k ≈ 0.29 vs uniform k ≈ 1.0 (≈3.4× better convergence exponent;
overall cubic-to-quartic speedup regime at T ≤ 1). Advantage shrinks at
high T where uniform proposals are already fine.

## Implementation checklist derived from this paper

- [x] γ and t drawn fresh per proposal from the ranges above
- [x] α normalization computed from Frobenius norms
- [x] Second-order (symmetric) Trotter, Δt = 0.8 default
- [x] Statevector path builds exact |⟨s'|U|s⟩|² proposal matrix (n ≤ ~14)
- [x] E01 measures k exponents at T ∈ {0.1, 0.3, 1.0} and compares to k≈0.29 vs 1.0
      (observed: k_quench = 0.316 vs k_uniform = 1.022 at T=0.1, n=5–10)
