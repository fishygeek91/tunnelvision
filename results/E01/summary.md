# E01 — Layden reproduction

Exact-tier spectral gaps on fully-connected spin glasses with
random fields (Layden et al., Nature 619, 282 (2023), Fig. 2).
⟨δ⟩ is the **geometric** mean over instances; k is the least-squares
fit of ⟨δ⟩ ∝ 2^{−kn}.

## Provenance

- git commit: `ca6320c7e379dd619482d45b95166177249d16e3-dirty`
- started (UTC): 2026-08-14T17:29:52.457344+00:00
- finished (UTC): 2026-08-14T17:34:08.265065+00:00
- package versions: numpy=2.4.6, scipy=1.17.1, qiskit=2.5.2, qiskit-aer=0.17.2, tunnelvision=0.0.1
- instances per (n, T): 20
- topology: all-to-all, random_fields=True
- quench evolution: exact, grid 6×8

## k exponents (geometric-mean fit)

| T | quench | single-flip | uniform |
| --- | --- | --- | --- |
| 0.1 | 0.316 | 2.378 | 1.022 |
| 0.3 | 0.312 | 2.055 | 1.034 |
| 1 | 0.272 | 0.827 | 1.002 |

Arithmetic-mean k at T = 1 (paper Fig. 2 convention): quench=0.281, single-flip=0.642, uniform=1.009. Paper: quench 0.264(4), local 0.94(4), uniform 0.948(7).

## Geometric-mean gaps

### T = 0.1

| n | quench | single-flip | uniform |
| --- | --- | --- | --- |
| 5 | 0.08327 | 5.142e-11 | 0.03441 |
| 6 | 0.07293 | 2.875e-12 | 0.01724 |
| 7 | 0.06355 | 3.339e-14 | 0.00847 |
| 8 | 0.06142 | 1.513e-12 | 0.004027 |
| 9 | 0.03989 | 1.926e-14 | 0.002069 |
| 10 | 0.02602 | 4.709e-15 | 0.001002 |

### T = 0.3

| n | quench | single-flip | uniform |
| --- | --- | --- | --- |
| 5 | 0.08835 | 1.279e-05 | 0.03903 |
| 6 | 0.07494 | 1.354e-06 | 0.02013 |
| 7 | 0.06585 | 1.701e-06 | 0.009813 |
| 8 | 0.06217 | 3.986e-06 | 0.004564 |
| 9 | 0.03939 | 3.289e-08 | 0.002224 |
| 10 | 0.02897 | 4.698e-09 | 0.001132 |

### T = 1

| n | quench | single-flip | uniform |
| --- | --- | --- | --- |
| 5 | 0.08991 | 0.009682 | 0.05481 |
| 6 | 0.07691 | 0.003965 | 0.03159 |
| 7 | 0.07222 | 0.003471 | 0.01595 |
| 8 | 0.06691 | 0.00399 | 0.008102 |
| 9 | 0.04266 | 0.0008592 | 0.003361 |
| 10 | 0.03482 | 0.0004259 | 0.001864 |

## Gate

**PASS**

At T=0.1: k_quench=0.316, k_uniform=1.022, k-ratio=3.24 (paper ≈ 3.4), gap-ratio at largest n=26.0×.
At T=1: k_quench=0.272, k_uniform=1.002, k-ratio=3.69, gap-ratio=18.7×.
Gate: qualitative match — proceed to Rung 2.

Paper reference (Fig. 2 / SM, arithmetic means over 500 instances,
n = 3–10): at low T, quench k ≈ 0.26–0.29 vs uniform k ≈ 1.0;
advantage shrinks as T grows because uniform proposals already mix.

## Notes

- Fit range is n = 5–10. n = 8–10 alone overestimates quench k (~0.62);
  the paper's lever arm is n = 3–10. Twenty instances (vs 500) is enough
  for the exponent, not for tight error bars.
- Single-flip gaps at T ≤ 0.3 underflow on the hard instances; the
  geometric-mean k there is not a physical exponent. At T = 1 the
  local kernel is well-defined (k ≈ 0.83 geo / 0.64 arith).
- E01 uses exact `e^{-iHt}` (Nature Fig. 2). The Trotter / Aer path is
  tested in `tests/test_quench.py` and is what E03 / hardware will run.

![gap vs n](gap_vs_n.png)

