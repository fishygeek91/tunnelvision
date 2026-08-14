"""Independent-chain helper used by E02b and the sampled tier."""

from __future__ import annotations

import numpy as np
import pytest

from experiments.E02_tunnelvision_demo.chains import (
    rho_chain_seed,
    run_replicated_chains,
    sampled_chain_seed,
)
from experiments.E02_tunnelvision_demo.run_sampled import _quench_skip_reason
from tunnelvision.engine import MetropolisEngine
from tunnelvision.kernels.classical import AddDeleteSwap, UniformFlip
from tunnelvision.targets.ising import random_spin_glass


def test_rho_chain_seed_is_stable() -> None:
    """E02b numbers were drawn from this map; do not reshuffle it."""
    assert rho_chain_seed(0, 0.5, 1) == 500_001
    assert rho_chain_seed(0, 0.9, 0) == 900_000


def test_sampled_chain_seed_blocks_do_not_collide() -> None:
    seeds = {
        sampled_chain_seed(0, p=p, rho=rho, kernel_id=k, chain_index=c)
        for p in (20, 27)
        for rho in (0.5, 0.9)
        for k in (0, 1, 2)
        for c in range(4)
    }
    assert len(seeds) == 2 * 2 * 3 * 4


def test_replicated_chains_are_deterministic() -> None:
    target = random_spin_glass(5, topology="all-to-all", seed=0, temperature=1.0)
    engine = MetropolisEngine(target, UniformFlip())
    kwargs = {
        "n_steps": 80,
        "burn_in": 20,
        "seeds": [0, 1],
        "x0": np.zeros(5, dtype=np.uint8),
    }
    first = run_replicated_chains(engine, **kwargs)
    second = run_replicated_chains(engine, **kwargs)
    assert first["rhat_size"] == second["rhat_size"]
    assert first["ess_size"] == second["ess_size"]
    assert first["pip_estimate"] == second["pip_estimate"]


def test_replicated_chains_score_ads_on_ising() -> None:
    target = random_spin_glass(6, topology="all-to-all", seed=1, temperature=1.0)
    engine = MetropolisEngine(target, AddDeleteSwap())
    scored = run_replicated_chains(
        engine,
        n_steps=120,
        burn_in=20,
        seeds=[0, 1, 2, 3],
        x0=np.zeros(6, dtype=np.uint8),
    )
    assert scored["n_chains"] == 4
    assert scored["n_kept"] == 100
    assert 0.0 < scored["ess_size_per_step"] <= 1.0
    assert np.isfinite(scored["rhat_size"])
    assert scored["rhat_size"] >= 1.0
    assert len(scored["pip_estimate"]) == 6
    assert scored["pip_error"] is None


def test_replicated_chains_reject_one_chain() -> None:
    target = random_spin_glass(4, topology="all-to-all", seed=0, temperature=1.0)
    engine = MetropolisEngine(target, UniformFlip())
    with pytest.raises(ValueError, match="at least 2"):
        run_replicated_chains(
            engine,
            n_steps=20,
            burn_in=5,
            seeds=[0],
            x0=np.zeros(4, dtype=np.uint8),
        )


def test_quench_skip_reason_is_construction_not_taste() -> None:
    assert _quench_skip_reason(20) is None
    assert _quench_skip_reason(24) is None
    reason = _quench_skip_reason(27)
    assert reason is not None
    assert "27" in reason
