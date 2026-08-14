# Coarse-grained qe-MCMC (Phys. Rev. Research 7, 013231 (2025))

**How TunnelVision scales past the qubit count (Rung 3, p ≈ 50–100).**
Extracted 2026-08-14 (abstract-level; pull the full PDF for Rung 3 work —
see get_papers.sh):

- Quadratic resource reduction: √n simulated qubits stand in for the n-qubit
  quench of standard qe-MCMC.
- Headline example: 6 qubits suffice to beat standard classical approaches on
  magnetization of a 36-spin system.
- Adaptable to limited-connectivity hardware; quantum speedup persists under
  coarse-graining.
- Mechanism (to verify from full text before implementing): propose quantum
  moves on a subregion, treat the frozen complement's couplings as effective
  local fields on the subregion, accept/reject globally with the exact target.
  Subregion choice must not depend on chain state in a way that breaks the
  proposal accounting — verify their scheme's symmetry argument.

Open question we want the full text for: scaling of speedup with subregion
fraction n/N (our Rung 2/3 measurement fills gaps here).
