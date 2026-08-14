"""Maxwell's Daemon — scheduled noise as a proposal-temperature ladder (N2).

Idea: run the same quench circuit at K deliberately different noise levels
(dynamical decoupling on/off, Pauli twirling levels, inserted idle time).
Each level is one rung of a ladder of effectively 'hotter' proposal
distributions. Compose rungs like parallel tempering — but the replicas
differ in PROPOSAL heat, not target temperature, so every rung samples the
same exact target and no replica-exchange bias correction is needed.

Open scientific question this module exists to answer (experiment E03):
does noise-heating help mixing, or does it destroy the tunneling structure
that creates the quantum advantage? Either answer is a result.

Caution on exactness: noise levels must be FIXED per rung (a fixed unital
channel keeps the kernel symmetric). Adapting noise level based on chain
history requires diminishing-adaptation conditions — see AdaptiveMixture.
"""

from __future__ import annotations

from tunnelvision.kernels.base import Kernel


class NoiseLadderKernel(Kernel):
    """Cycles (round-robin or random-scan) over quench kernels at K noise levels."""

    name = "maxwells-daemon"

    def __init__(self, rungs: list[Kernel], scan: str = "random") -> None:
        raise NotImplementedError  # Rung 3 / E03


class AdaptiveMixture(Kernel):
    """N3: online-adapted mixture of quantum and classical kernels.

    Weights adapt from acceptance-rate feedback under diminishing
    adaptation (Roberts & Rosenthal 2007) to preserve ergodicity.
    """

    name = "adaptive-mixture"

    def __init__(self, kernels: list[Kernel]) -> None:
        raise NotImplementedError  # Rung 3
