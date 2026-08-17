"""E03 — symmetric-q bias audit (Aer amplitude damping, not a QPU).

Hardware proposals are only approximately symmetric: amplitude
damping is non-unital, so q(y|x) = q(x|y) is an approximation.
This script bounds the residual by comparing sampled posteriors
to the enumerated exact target. Accept/reject is still the exact
g-prior on every arm — only the proposal symmetry is under test.

Arms: ideal quench (statevector Trotter), one HardwareQuenchKernel
+ Aer amplitude-damping arm per γ, and add-delete-swap (the
baseline whose TV should sit at Monte Carlo error).

``--quick`` drops to p=5 so the entry point is smoke-testable.
This is not a hardware claim.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from experiments.E02_tunnelvision_demo.chains import run_replicated_chains
from experiments.E02_tunnelvision_demo.run import _md_table, _quench
from experiments.E02_tunnelvision_demo.run_ablations import (
    _dataset_key,
    _load_dataset_spec,
)
from experiments.provenance import REPO_ROOT, git_commit, load_yaml, package_versions

os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / ".mplconfig"))
from tunnelvision.engine import MetropolisEngine
from tunnelvision.kernels.classical import AddDeleteSwap
from tunnelvision.kernels.quantum import (
    AerCircuitSampler,
    HardwareQuenchKernel,
    amplitude_damping_noise_model,
)
from tunnelvision.surrogate import IsingSurrogate, learned_surrogate
from tunnelvision.targets.spike_slab import SpikeSlabTarget


def _resolve_dampings(config: dict[str, Any]) -> list[float]:
    """Prefer ``dampings``; a single ``damping`` still works for --quick."""
    if "dampings" in config:
        values = [float(x) for x in config["dampings"]]
    elif "damping" in config:
        values = [float(config["damping"])]
    else:
        raise ValueError("config needs dampings: [...] or damping: <float>")
    if not values:
        raise ValueError("dampings must not be empty")
    if any(gamma < 0.0 or gamma > 1.0 for gamma in values):
        raise ValueError(f"dampings must lie in [0, 1], got {values}")
    return values


def _apply_quick(config: dict[str, Any]) -> None:
    config["dampings"] = [0.1]
    config["damping"] = 0.1
    config["pool_size"] = 4
    config["datasets"] = [
        {
            "name": "synthetic",
            "n": 60,
            "p": 5,
            "rho": 0.5,
            "k_true": 2,
            "snr": 3.0,
            "seed": 0,
            "structure": "equicorrelated",
        }
    ]
    quench = dict(config.get("quench", {}))
    quench["n_gamma"] = 3
    quench["n_t"] = 3
    config["quench"] = quench
    learned = dict(config.get("learned", {}))
    learned["n_samples"] = 200
    config["learned"] = learned
    chain = dict(config.get("chain", {}))
    chain.update({"n_chains": 2, "n_steps": 120, "burn_in": 20})
    config["chain"] = chain
    config["output_dir"] = "results/E03_quick/bias_audit"


def _random_starts(n_vars: int, seeds: list[int]) -> list[np.ndarray]:
    starts: list[np.ndarray] = []
    for seed in seeds:
        rng = np.random.Generator(np.random.PCG64(int(seed) + 17))
        starts.append(rng.integers(0, 2, size=n_vars, dtype=np.uint8))
    return starts


def _cache_stats(kernel: Any) -> dict[str, float]:
    if not isinstance(kernel, HardwareQuenchKernel):
        return {
            "cache_hits": float("nan"),
            "cache_misses": float("nan"),
            "jobs_submitted": float("nan"),
            "cache_hit_rate": float("nan"),
        }
    total = kernel.cache_hits + kernel.cache_misses
    return {
        "cache_hits": float(kernel.cache_hits),
        "cache_misses": float(kernel.cache_misses),
        "jobs_submitted": float(kernel.jobs_submitted),
        "cache_hit_rate": (kernel.cache_hits / total) if total else 0.0,
    }


def _make_ad_kernel(
    learned: IsingSurrogate,
    quench_cfg: dict[str, Any],
    damping: float,
    pool_size: int,
) -> HardwareQuenchKernel:
    sampler = AerCircuitSampler(
        noise_model=amplitude_damping_noise_model(damping),
        shots=1,
    )
    hardware = HardwareQuenchKernel(
        learned.h,
        learned.J,
        gamma_range=tuple(quench_cfg["gamma_range"]),
        t_range=tuple(quench_cfg["t_range"]),
        trotter_dt=float(quench_cfg.get("trotter_dt", 0.8)),
        n_gamma=int(quench_cfg["n_gamma"]),
        n_t=int(quench_cfg["n_t"]),
        sampler=sampler,
        pool_size=pool_size,
    )
    hardware.name = f"quench-ad-{damping:g}"
    return hardware


def _make_arms(
    learned: IsingSurrogate,
    quench_cfg: dict[str, Any],
    dampings: list[float],
    pool_size: int,
) -> list[Any]:
    ideal = _quench(learned, "quench-ideal", {**quench_cfg, "backend": "statevector"})
    ads = AddDeleteSwap()
    damped = [_make_ad_kernel(learned, quench_cfg, gamma, pool_size) for gamma in dampings]
    return [ideal, *damped, ads]


def _ad_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    damped = [row for row in rows if str(row["kernel"]).startswith("quench-ad")]
    return sorted(damped, key=lambda row: float(row.get("damping", 0.0)))


def _mixed(row: dict[str, Any]) -> bool:
    return float(row["rhat_size"]) <= 1.1


def _read_audit(rows: list[dict[str, Any]]) -> list[str]:
    lines = [
        "ADS is the exact-kernel baseline: its TV and PIP error are",
        "Monte Carlo noise, not bias. Ideal quench is the symmetric",
        "Trotter proposal. Each ``quench-ad-γ`` arm is",
        "HardwareQuenchKernel with Aer amplitude damping — the",
        "non-unital stand-in for hardware. TV that tracks ADS is the",
        "approximation holding; TV that grows with γ is the residual",
        "the live-hardware audit must quote before any QPU claim.",
        "R̂ > 1.1 on a damped arm is reported as \"does not mix\"",
        "rather than a clean bias number.",
        "",
    ]
    datasets = list(dict.fromkeys(str(row["dataset"]) for row in rows))
    for dataset in datasets:
        subset = [row for row in rows if row["dataset"] == dataset]
        ads = next(row for row in subset if row["kernel"] == "add-delete-swap")
        ideal = next(row for row in subset if row["kernel"] == "quench-ideal")
        ads_tv = float(ads["tv_distance"])
        lines.append(
            f"- **{dataset}**: ADS TV {ads_tv:.3f} (R̂ {float(ads['rhat_size']):.3f}"
            f"{'' if _mixed(ads) else ', did not mix'}). "
            f"Ideal quench TV {float(ideal['tv_distance']):.3f} "
            f"(R̂ {float(ideal['rhat_size']):.3f}"
            f"{'' if _mixed(ideal) else ', did not mix'})."
        )
        for ad in _ad_rows(subset):
            ad_tv = float(ad["tv_distance"])
            ratio = ad_tv / ads_tv if ads_tv > 0.0 else float("inf")
            mix = "mixed" if _mixed(ad) else "does not mix"
            lines.append(
                f"  - γ={float(ad['damping']):g}: TV {ad_tv:.3f} ({ratio:.2f}× ADS), "
                f"PIP {float(ad['pip_error_pooled']):.3f}, "
                f"R̂ {float(ad['rhat_size']):.3f} ({mix}), "
                f"accept {float(ad['acceptance_rate']):.3f}."
            )
    lines.append("")
    return lines


def _write_summary(
    path: Path,
    config: dict[str, Any],
    meta: dict[str, Any],
    rows: list[dict[str, Any]],
) -> None:
    lines = [
        "# E03 — symmetric-q bias audit (Aer, not a QPU)",
        "",
        "This bounds the symmetric-q approximation under a non-unital",
        "channel. It is not a hardware result. Accept/reject uses the",
        "exact g-prior on every arm.",
        "",
        "## Provenance",
        "",
        f"- git commit: `{meta['git_commit']}`",
        f"- started (UTC): {meta['started_utc']}",
        f"- finished (UTC): {meta['finished_utc']}",
        f"- package versions: {', '.join(f'{k}={v}' for k, v in meta['versions'].items())}",
        f"- damping γ: {', '.join(f'{x:g}' for x in _resolve_dampings(config))}",
        f"- pool size: {config['pool_size']}",
        f"- quench: {config['quench'].get('evolution', 'trotter')}, "
        f"grid {config['quench']['n_gamma']}×{config['quench']['n_t']}",
        f"- chain: {config['chain']['n_chains']} chains, "
        f"{config['chain']['n_steps']} steps, burn-in {config['chain']['burn_in']}",
        "",
        "## Scoreboard",
        "",
        "TV is the total-variation distance between the pooled sampled",
        "posterior and ``enumerate_exact``. PIP error is max-abs vs the",
        "enumerated inclusion probabilities. R̂ is split-R̂ on |γ|.",
        "",
    ]
    table = []
    for row in rows:
        hit = row["cache_hit_rate"]
        hit_s = "—" if hit is None or not np.isfinite(float(hit)) else f"{float(hit):.2f}"
        table.append(
            [
                str(row["dataset"]),
                str(row["kernel"]),
                f"{row['tv_distance']:.3f}",
                f"{row['pip_error_pooled']:.3f}",
                f"{row['rhat_size']:.3f}",
                f"{row['acceptance_rate']:.3f}",
                hit_s,
            ]
        )
    lines.append(
        _md_table(
            ["dataset", "kernel", "TV", "PIP err", "R̂(|γ|)", "accept", "cache hit"],
            table,
        )
    )
    lines.extend(
        [
            "",
            "## Read",
            "",
            *_read_audit(rows),
            "## Notes",
            "",
            "- This is the methodology the live-hardware audit must reuse.",
            "- Each ``quench-ad-γ`` arm returns log-q = 0 (the",
            "  approximation). The residual lives in TV / PIP error,",
            "  not in the Hastings ratio.",
            "- Physical noise rungs (DD, twirling, idle) are still open.",
            "",
        ]
    )
    if meta.get("quick"):
        lines.extend(
            [
                "This file was written by ``--quick`` (p=5 synthetic). It is not",
                "the p=10 diabetes cell and lives under `results/E03_quick/`.",
                "",
            ]
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).with_name("config_bias_audit.yaml"),
        help="YAML config (default: experiments/E03_maxwells_daemon/config_bias_audit.yaml)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="p=5 synthetic smoke run, not the p=10 diabetes cell",
    )
    args = parser.parse_args(argv)

    config = load_yaml(args.config)
    if args.quick:
        _apply_quick(config)

    output_dir = REPO_ROOT / str(config.get("output_dir", "results/E03/bias_audit"))
    output_dir.mkdir(parents=True, exist_ok=True)
    prior = float(config["prior_inclusion"])
    dampings = _resolve_dampings(config)
    pool_size = int(config["pool_size"])
    if pool_size < 1:
        raise ValueError(f"pool_size must be at least 1, got {pool_size}")
    quench_cfg = dict(config["quench"])
    learned_cfg = dict(config.get("learned", {}))
    chain_cfg = dict(config["chain"])
    n_chains = int(chain_cfg["n_chains"])
    n_steps = int(chain_cfg["n_steps"])
    burn_in = int(chain_cfg["burn_in"])
    base_seed = int(chain_cfg["seed"])

    meta: dict[str, Any] = {
        "git_commit": git_commit(),
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "versions": package_versions(),
        "config": config,
        "quick": bool(args.quick),
    }
    print(
        f"E03 bias audit: datasets={[_dataset_key(s) for s in config['datasets']]} "
        f"dampings={dampings} pool_size={pool_size}",
        flush=True,
    )

    rows: list[dict[str, Any]] = []
    for spec in config["datasets"]:
        X, y, label = _load_dataset_spec(spec)
        key = _dataset_key(spec)
        target = SpikeSlabTarget(X, y, prior_inclusion=prior)
        learned: IsingSurrogate = learned_surrogate(
            target,
            n_samples=int(learned_cfg.get("n_samples", 2000)),
            seed=int(learned_cfg.get("seed", 0)),
            ridge=float(learned_cfg.get("ridge", 1e-3)),
        )
        exact_pips = target.posterior_inclusion_probs_exact()
        pi = target.enumerate_exact()
        print(f"  dataset {key}: {label}, n_vars={target.n_vars}", flush=True)
        seeds = [base_seed + chain_index for chain_index in range(n_chains)]
        starts = _random_starts(target.n_vars, seeds)
        for kernel in _make_arms(learned, quench_cfg, dampings, pool_size):
            print(f"    scoring {kernel.name} on {key}...", flush=True)
            engine = MetropolisEngine(target, kernel)
            scored = run_replicated_chains(
                engine,
                n_steps=n_steps,
                burn_in=burn_in,
                seeds=seeds,
                x0=starts,
                exact_pips=exact_pips,
                exact_pi=pi,
            )
            damping_value = float("nan")
            if kernel.name.startswith("quench-ad-"):
                damping_value = float(kernel.name.rsplit("-", 1)[-1])
            rows.append(
                {
                    **scored,
                    **_cache_stats(kernel),
                    "dataset": key,
                    "kernel": kernel.name,
                    "damping": damping_value,
                }
            )
            print(
                f"    TV={scored['tv_distance']:.3f}  "
                f"PIP={scored['pip_error_pooled']:.3f}  "
                f"R̂={scored['rhat_size']:.3f}",
                flush=True,
            )

    meta["finished_utc"] = datetime.now(timezone.utc).isoformat()

    import pandas as pd

    pd.DataFrame(rows).to_csv(output_dir / "scoreboard.csv", index=False)
    (output_dir / "meta.json").write_text(json.dumps(meta, indent=2, default=str) + "\n")
    (output_dir / "summary.json").write_text(json.dumps(rows, indent=2, default=str) + "\n")
    _write_summary(output_dir / "summary.md", config, meta, rows)
    print(f"wrote {output_dir / 'summary.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
