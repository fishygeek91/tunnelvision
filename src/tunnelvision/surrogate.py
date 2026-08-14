"""2-local Ising surrogate of the spike-and-slab posterior.

The quantum quench needs a 2-local Hamiltonian; the exact posterior isn't
one. Build a surrogate from the data (never from chain history):

    fields    h_j  ~ marginal evidence for variable j (e.g. |x_j' y| scaled)
    couplings J_jk ~ -collinearity penalty from X'X (correlated predictors
                     compete for inclusion -> antiferromagnetic coupling)

plus a sparsity field from the prior inclusion probability. Candidate
refinement: fit (h, J) by ridge regression of exact log-posterior values
on a random sample of gammas ("learned surrogate") — compare both in E02.

EXACTNESS NOTE: the surrogate only shapes the proposal distribution.
Any surrogate — even a terrible one — leaves the sampler exact, because
accept/reject uses targets/spike_slab.py. Quality of surrogate affects
SPEED only. This asymmetry is the project's central design trick.
"""

from __future__ import annotations

import numpy as np


def ising_surrogate_from_data(
    X: np.ndarray, y: np.ndarray, prior_inclusion: float = 0.5
) -> tuple[np.ndarray, np.ndarray]:
    """Returns (h, J) for the proposal Hamiltonian. Rung 1 (analytic version)."""
    raise NotImplementedError


def learned_surrogate(
    target, n_samples: int = 2000, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Ridge-fit (h, J) to exact log-posterior evaluations. Rung 3 refinement."""
    raise NotImplementedError
