"""Dataset loaders.

diabetes: sklearn's diabetes dataset, p=10, standardized. 2^10 = 1024
models — exact posterior and exact spectral gaps enumerable. E02 primary.

correlated_synthetic: linear-model data with tunable predictor correlation
rho and sparse true support — the knob for "does quantum advantage grow
with multimodality?" (it should: correlation -> competing models -> modes).
"""

from __future__ import annotations

import numpy as np


def load_diabetes_standardized() -> tuple[np.ndarray, np.ndarray]:
    raise NotImplementedError  # Rung 1


def correlated_synthetic(
    n: int = 100, p: int = 20, rho: float = 0.7, k_true: int = 4, snr: float = 2.0, seed: int = 0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (X, y, true_support)."""
    raise NotImplementedError  # Rung 1
