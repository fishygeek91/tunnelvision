"""Exactness invariants — the tests that guard the project's core claim.

These must pass for EVERY kernel, including deliberately broken ones.
If a new kernel fails these, the kernel is fine and the engine is broken,
or the kernel is lying about its proposal probabilities. Either way: stop.

Planned invariants (implement with engine in Rung 1):

1. test_stationarity_exact:
   For each kernel with a proposal_matrix, build the full MH transition
   matrix on a small target (n=6..10) and assert pi @ P == pi to 1e-12.

2. test_detailed_balance_exact:
   pi_x * P[x, y] == pi_y * P[y, x] for all pairs (reversible kernels only;
   skip for deliberately irreversible kernels, check stationarity instead).

3. test_sampled_distribution_matches_enumeration:
   Long chain on n=8 target; TV distance to exact distribution < tol.
   Run per kernel, including a NOISY quench kernel — this is the test
   that demonstrates "noise cannot bias the answer."

4. test_broken_kernel_still_exact:
   A pathological kernel (e.g. proposes from the wrong distribution but
   reports honest log-q) must still yield the exact stationary law.

5. test_asymmetric_kernel_logq:
   AddDeleteSwap at boundary states (empty/full model): forward/reverse
   log-q asymmetry handled; empirical stationarity holds.
"""
