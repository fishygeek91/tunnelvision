# M03 — SHELLHASH: the shell oracle, built classically from SAT + XOR hashing

**Registered 2026-08-31, BEFORE any results.** IDEATION-02 Idea A.

## Claim under test

D01/D02 proved the quench's edge is global isoenergetic proposing; M01/M02
showed no O(n) kernel we constructed (or the classical literature's ICM)
reaches the shell at distance. The Chakraborty–Meel–Vardi line proves an
NP oracle + random XOR (parity) constraints yields certified near-uniform
samples of any CNF-encodable set. The energy shell
S_eps(x) = {y : |E(y) − E(x)| <= eps} of an Ising/QUBO Hamiltonian is
CNF-encodable (product variables + integer pseudo-Boolean bounds).
"Solver-in-the-loop MCMC" is, per the IDEATION-02 sweep, absent from the
literature. SHELLHASH = a de-novo classical shell generator at NP-oracle
cost; the contest is WALL-CLOCK per shell sample.

## Pre-registered outcomes and kill criteria

- **Correctness gate (M03a, must pass before anything counts):** at n=8,10
  the solver-enumerated shell must EQUAL the numpy-enumerated shell of the
  integerized Hamiltonian (exact set equality), and the sampler's empirical
  distribution over the shell must be near-uniform (TV distance < 0.10 at
  2000 samples on every M01 instance; uniformity is the property that makes
  the oracle's Hastings correction |S(x)|/|S(y)| honest).
- **Deploy-lane kill (M03b):** median time per shell sample > 1 s at n=20
  (eps=2, all-to-all spin glass) kills the mass-deployment lane.
- **Context bar:** quench hardware proposal ~1e-2 s (E02c). SHELLHASH beats
  the quench's wall-clock only below that; between 1e-2 and 1 s it is a
  complexity anchor, not a deployment.
- **Either outcome publishes:** fast => the last quantum-sampling edge we
  believed in is dequantized in practice; slow => first evidence-backed
  wall-clock separation between the quench and the best known classical
  de-novo shell generator, with the NP-oracle upper bound as the anchor.

## Honesty notes (written in advance)

- The sampler is UniGen-LITE (hash-and-enumerate with a cell cap), not
  full UniGen: guarantees are cited, not inherited. Uniformity is therefore
  VALIDATED EMPIRICALLY at enumerable sizes, and the same machinery drives
  the scale runs unchanged.
- Integerization (scale S=1000) makes the solver's shell a quantized shell;
  M03a reports the symmetric difference between the integer shell and the
  real-J shell so quantization can never silently flatter the result.
- Timings include ALL retries/re-hashing — cost per DELIVERED sample.
- Device VM caps shell calls at 180 s; scale runs execute in the cloud
  container. Same code, same seeds; the CSV records the host.
