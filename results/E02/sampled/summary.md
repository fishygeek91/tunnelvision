# E02c — sampled tier (p=20 and p=27)

ESS, wall-clock, and split-R̂ on equicorrelated spike-and-slab
posteriors above the exact-tier cutoff. Accept/reject uses the
exact g-prior; the surrogate only shapes proposals. There is no
spectral gap here — p > 14.

## Provenance

- git commit: `30c1dbc5fea959417469eb915f2a81116d59776a`
- started (UTC): 2026-08-14T21:26:30.304438+00:00
- finished (UTC): 2026-08-14T22:30:24.045608+00:00
- package versions: numpy=2.4.6, scipy=1.17.1, scikit-learn=1.9.0, qiskit=2.5.2, qiskit-aer=0.17.2, tunnelvision=0.0.1
- design: equicorrelated n=120 k_true=3 snr=2.0
- p grid: 20, 27
- ρ grid: 0.5, 0.9
- quench: trotter statevector
- chains: 4 × classical 6000/1500, quench 160/40

## Pilot (quench wall-clock)

A live Trotter proposal at p=20 allocates a 2^20 statevector.
p=27 cannot construct the kernel: the problem Hamiltonian's
diagonal is enumerated over the full basis.

| p | ρ | kernel | s/propose | est. wall (s) | ran | note |
| --- | --- | --- | --- | --- | --- | --- |
| 20 | 0.5 | quench-learned | 1.970 | 1261 | yes |  |
| 20 | 0.9 | quench-learned | 2.815 | 1802 | yes |  |
| 27 | 0.5 | quench-learned | — | — | no | problem_energies enumerates 2^27 basis energies; all_binary_states refuses n > 24 |
| 27 | 0.9 | quench-learned | — | — | no | problem_energies enumerates 2^27 basis energies; all_binary_states refuses n > 24 |

## Surrogate diagnosis (sampled Spearman)

| p | ρ | surrogate | Spearman(−E, log p) | n |
| --- | --- | --- | --- | --- |
| 20 | 0.5 | analytic | 0.371 | 4000 |
| 20 | 0.5 | learned | 0.998 | 4000 |
| 20 | 0.9 | analytic | 0.145 | 4000 |
| 20 | 0.9 | learned | 0.998 | 4000 |
| 27 | 0.5 | analytic | 0.140 | 4000 |
| 27 | 0.5 | learned | 0.998 | 4000 |
| 27 | 0.9 | analytic | -0.107 | 4000 |
| 27 | 0.9 | learned | 0.999 | 4000 |

## p=20 scoreboard

PIP error is against enumerated truth (p=20 is the last
enumerable size). ESS/sec is live-kernel wall-clock, not
tabulated-Q sampling — this is the column that prices the
simulation.

| ρ | kernel | accept | PIP err | ESS/step (|γ|) | ESS/sec (|γ|) | R̂ (|γ|) | wall (s) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.5 | add-delete-swap | 0.288 | 0.033 | 0.031 | 381.29 | 1.001 | 1.5 |
| 0.5 | single-flip | 0.207 | 0.031 | 0.036 | 837.23 | 1.011 | 0.8 |
| 0.5 | quench-learned | 0.200 | 0.500 | 0.080 | 0.02 | 1.448 | 1903.3 |
| 0.9 | add-delete-swap | 0.299 | 0.020 | 0.035 | 426.68 | 1.004 | 1.5 |
| 0.9 | single-flip | 0.215 | 0.031 | 0.033 | 776.79 | 1.008 | 0.8 |
| 0.9 | quench-learned | 0.197 | 0.235 | 0.092 | 0.02 | 1.740 | 1839.5 |

## p=27 scoreboard

No enumerated PIPs at p=27. Cross-kernel PIP agreement and
R̂ are the honesty checks. Quench is expected to be absent.

| ρ | kernel | accept | ESS/step (|γ|) | ESS/sec (|γ|) | R̂ (|γ|) | max R̂ (PIP) | wall (s) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.5 | add-delete-swap | 0.370 | 0.022 | 272.36 | 1.014 | 1.019 | 1.5 |
| 0.5 | single-flip | 0.239 | 0.024 | 543.98 | 1.003 | 1.029 | 0.8 |
| 0.9 | add-delete-swap | 0.372 | 0.027 | 331.44 | 1.013 | 1.026 | 1.5 |
| 0.9 | single-flip | 0.248 | 0.024 | 556.01 | 1.005 | 1.019 | 0.8 |

### Cross-kernel PIP agreement (max-abs)

ρ = 0.5

| left | right | max |ΔPIP| |
| --- | --- | --- |
| add-delete-swap | single-flip | 0.063 |

ρ = 0.9

| left | right | max |ΔPIP| |
| --- | --- | --- |
| add-delete-swap | single-flip | 0.071 |


![scoreboard](scoreboard.png)

## Read

The sampled tier does **not** turn the E02 story. Quench remains
behind add-delete-swap once honesty checks are applied.

- p=20 quench is simulable (2.0–2.8 s/propose, ~30 min for 4×160
  steps) and **unconverged**. Split-R̂ is 1.45 at ρ=0.5 and 1.74
  at ρ=0.9; pooled PIP error is 0.50 and 0.24 against enumerated
  truth. The raw ESS/step ratio vs ADS (2.5–2.6×) is not a result
  — those chains have not mixed. ADS and single-flip finish in
  ~1 s with R̂ ≤ 1.01 and PIP error 0.02–0.03.
- ESS/sec is the column that prices the simulation: quench 0.02
  vs ADS 380–430. Even a trusted 2.5× ESS/step would still lose
  by four orders of magnitude on wall-clock. That is the
  quantum-papers failure mode this repo exists to avoid.
- The learned surrogate is not the problem (sampled Spearman
  0.998 at every cell). Analytic collapses with ρ and p, as on
  the exact tier. Landscape quality did not leak into accept/reject:
  ADS PIP error stays at Monte Carlo.
- p=27 quench is a construction refusal, not a missing
  measurement. `problem_energies` enumerates 2^p; `all_binary_states`
  refuses n > 24. Classical kernels mix (R̂ ≤ 1.03) and agree on
  PIPs to 0.06–0.07.
- Single-flip is competitive with ADS on ESS/step and faster per
  second. That is a classical finding; it does not rescue the quench.

## Notes

- Add-delete-swap is the baseline. Beating uniform proves nothing.
- Quench chain length is wall-clock limited (pilot: 3.3 s/propose
  at first pricing; 160 steps keep four chains inside a session).
  Longer quench chains would cost hours per cell and still pay
  seconds per proposal.
- Analytic quench was not re-run: it already lost at p=10, and
  the budget went to learned.
- The remaining scale-up path is WP7 coarse-graining (Ferguson
  et al., Phys. Rev. Research 7, 013231), not a denser 2^p
  statevector.

