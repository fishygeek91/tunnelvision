"""Auxiliary-variable dynamics — deterministic, energy-conserving proposals.

Unlike ``kernels/``, modules here DO read the target energy: they need ΔE
to move an auxiliary variable (the demon). Exactness does not come from
ignorance of the target; it comes from an involution on the extended
space plus an honest Hastings term that depends only on the auxiliary
variable's change. The engine still owns accept/reject.
"""

from tunnelvision.dynamics.demon import DemonKernel
from tunnelvision.dynamics.icm import HoudayerICM

__all__ = ["DemonKernel", "HoudayerICM"]
