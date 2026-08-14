"""TunnelVision: exact Bayesian variable selection with quantum tunneling proposals.

Design invariant (the whole point of the project):
    Correctness lives in the Metropolis engine. Kernels only propose.
    A kernel may be noisy, biased, or broken — the chain remains exact
    as long as the kernel reports honest proposal probabilities (or is
    symmetric, in which case they cancel).
"""

__version__ = "0.0.1"

from tunnelvision.engine import MetropolisEngine  # noqa: F401
