"""Classical baseline kernels.

These are the bar to beat — especially AddDeleteSwap, the kernel
statisticians actually use for variable selection. Never benchmark
against a strawman.
"""

from __future__ import annotations

import numpy as np

from tunnelvision.kernels.base import Kernel


class UniformFlip(Kernel):
    """Propose a uniformly random state. Symmetric. The weakest baseline."""

    name = "uniform"

    def propose(self, x, rng):
        raise NotImplementedError  # Rung 1


class SingleFlip(Kernel):
    """Flip one uniformly chosen bit. Symmetric. The classic local kernel."""

    name = "single-flip"

    def propose(self, x, rng):
        raise NotImplementedError  # Rung 1


class AddDeleteSwap(Kernel):
    """Field-standard variable-selection kernel.

    With equal probability: add a variable, delete a variable, or swap an
    included variable with an excluded one. NOT symmetric in general —
    log-q terms must be computed at boundary states (empty/full models).
    """

    name = "add-delete-swap"

    def propose(self, x, rng):
        raise NotImplementedError  # Rung 1
