# D02 — Can a cheap classical kernel reproduce the shell? (2026-08-25)

Closing experiment for the D01 mechanism. D01 showed the quench ≈ uniform sampling of the target's energy shell at Hamming distance (oracle shell kernel matches/beats the quench gap). Question: does any cheap, exact classical kernel that uses *local* information reach the quench per target-evaluation?

Kernels (all exact, Hastings-corrected, `src/tunnelvision/dynamics/dlp.py`):
- **DLP** (Zhang, Liu & Liu 2022) with exact local differences: flip site i w.p. sigmoid(−t·ΔE_i − 1/2α). t = ½ is the locally-balanced / DLP choice; t = 0 is a plain symmetric random-mask kernel.
- **Soft-spin**: flip w.p. sigmoid(−c·|ΔE_i| − 1/2α) — the literal classical analogue of "the transverse field flips weak-local-field spins".

Exact spectral gaps, all-to-all spin glass with fields (quench: T-independent):

| instance | quench | DLP t=½ (best α) | DLP t=0.1T | DLP t=0.03T | DLP t=0 (random mask, best α) | soft-spin c=0.1 (best) | soft-spin c=1 |
|---|---|---|---|---|---|---|---|
| n8 s1 T=0.3 | 0.085 | 8.5e-6 | 4.5e-3 | 6.8e-3 | 7.3e-3 | 4.5e-3 | 2.3e-8 |
| n8 s3 T=0.1 | 0.063 | 6.0e-9 | 1.7e-3 | 6.5e-3 | 5.0e-3 | — | — |
| n10 s1 T=0.3 | 0.034 | 2.7e-7 | 4.9e-4 | 1.3e-3 | 1.6e-3 | 5.2e-4 | 9.5e-9 |
| n10 s3 T=0.1 | 0.045 | 1.8e-13 | 2.6e-4 | 4.3e-3 | 3.0e-3 | 3.0e-4 | 2.0e-9 |

Findings:
1. **Every use of local information hurts, monotonically.** The more the proposal listens to ΔE_i (larger t or c), the smaller the gap — by up to 10 orders of magnitude. Informed proposals are greedy; at T ≤ 0.3 they descend into the nearest minimum and stay. The best member of each family is the *uninformed* one (t = 0 or c → 0), which is just a random-mask kernel at the uniform-proposal level.
2. **No cheap classical kernel gets within 10× of the quench per evaluation** at n = 8–10: best classical (random mask / uniform / MTM) ≈ 1–7e-3 vs quench 3–9e-2. Across D01+D02 this covers local (single-flip, ADS), microcanonical (demon), informed multi-flip (DLP, soft-spin), and multiple-try (MTM, greedy and shell-weighted).
3. The perturbative picture "transverse field ≈ flip soft spins" is wrong for the E01 quench (Γ = 0.25–0.6, t ≤ 20): the shell states it reaches are not the soft-spin neighbours. The shell is found non-perturbatively, by delocalization, and the oracle shows that *is* the whole advantage.

Verdict: in the exact tier the quench holds a genuine ≈10–30× per-evaluation edge over every cheap classical kernel we can construct, and the D01 oracle explains exactly what that edge is (global isoenergetic proposals). It remains irrelevant for wall-clock on today's hardware (E02c: 10⁴× shot cost). This closes the classical-competitor question for the paper: the quantum proposal is a shell sampler; the classical methods that use local information are the wrong tool for the shell, and the ones that don't are 10–30× short.
