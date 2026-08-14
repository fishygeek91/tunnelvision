"""E02a — exact-tier TunnelVision demo (the headline, first slice).

Question: does a quench kernel driven by a 2-local surrogate mix faster
than add-delete-swap on a real spike-and-slab posterior, while sampling
the *exact* posterior?

This script answers that at p ≤ 10, where the transition matrix is
enumerable. Spectral gap is the scoreboard. ESS/step *and* ESS/sec are
reported so a larger gap that costs 100× more wall-clock cannot pose as
a win — that is the quantum-papers failure mode this repo exists to
avoid.

E02b (the ρ-sweep) is a later PR. ``--quick`` swaps diabetes for a p=5
synthetic so the entry point is smoke-testable in seconds.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from experiments.provenance import REPO_ROOT, git_commit, load_yaml, package_versions

# Headless experiment runs must not touch $HOME/.matplotlib.
os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / ".mplconfig"))
from tunnelvision.data.loaders import correlated_synthetic, load_diabetes
from tunnelvision.diagnostics import ess, pip_error, spectral_gap
from tunnelvision.engine import MetropolisEngine
from tunnelvision.kernels.classical import AddDeleteSwap, SingleFlip, UniformFlip
from tunnelvision.kernels.quantum import QuenchKernel
from tunnelvision.surrogate import (
    IsingSurrogate,
    diagnose_surrogate,
    ising_surrogate_from_data,
    learned_surrogate,
    write_energy_logprob_scatter,
)
from tunnelvision.targets.spike_slab import SpikeSlabTarget

_CLASSICAL = {
    "uniform": UniformFlip,
    "single-flip": SingleFlip,
    "add-delete-swap": AddDeleteSwap,
}


def _apply_quick(config: dict[str, Any]) -> None:
    """Tiny synthetic so ``python -m experiments.E02_tunnelvision_demo.run --quick`` is seconds."""
    config["dataset"] = "synthetic"
    synthetic = dict(config.get("synthetic", {}))
    synthetic.update({"n": 60, "p": 5, "rho": 0.5, "k_true": 2, "snr": 3.0, "seed": 0})
    config["synthetic"] = synthetic
    quench = dict(config.get("quench", {}))
    quench["n_gamma"] = 3
    quench["n_t"] = 3
    config["quench"] = quench
    chain = dict(config.get("chain", {}))
    chain["n_steps"] = 800
    chain["burn_in"] = 100
    config["chain"] = chain
    config["output_dir"] = "results/E02_quick"


def _load_dataset(config: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, str]:
    name = str(config["dataset"]).lower()
    if name == "diabetes":
        X, y = load_diabetes()
        return X, y, None, "diabetes (sklearn, p=10)"
    if name == "synthetic":
        spec = config["synthetic"]
        X, y, support = correlated_synthetic(
            n=int(spec["n"]),
            p=int(spec["p"]),
            rho=float(spec["rho"]),
            k_true=int(spec["k_true"]),
            snr=float(spec["snr"]),
            seed=int(spec["seed"]),
            structure=str(spec.get("structure", "equicorrelated")),
        )
        label = (
            f"synthetic {spec.get('structure', 'equicorrelated')} "
            f"n={spec['n']} p={spec['p']} rho={spec['rho']} "
            f"k_true={spec['k_true']} snr={spec['snr']}"
        )
        return X, y, support, label
    raise ValueError(f"dataset must be 'diabetes' or 'synthetic', got {name!r}")


def _quench(surrogate: IsingSurrogate, name: str, cfg: dict[str, Any]) -> QuenchKernel:
    kernel = QuenchKernel(
        surrogate.h,
        surrogate.J,
        gamma_range=tuple(cfg["gamma_range"]),
        t_range=tuple(cfg["t_range"]),
        trotter_dt=float(cfg.get("trotter_dt", 0.8)),
        evolution=str(cfg.get("evolution", "exact")),
        n_gamma=int(cfg["n_gamma"]),
        n_t=int(cfg["n_t"]),
    )
    kernel.name = name
    return kernel


def _make_kernels(
    names: list[str],
    analytic: IsingSurrogate,
    learned: IsingSurrogate,
    quench_cfg: dict[str, Any],
) -> dict[str, Any]:
    kernels: dict[str, Any] = {}
    for name in names:
        if name in _CLASSICAL:
            kernels[name] = _CLASSICAL[name]()
        elif name == "quench-analytic":
            kernels[name] = _quench(analytic, name, quench_cfg)
        elif name == "quench-learned":
            kernels[name] = _quench(learned, name, quench_cfg)
        else:
            raise ValueError(f"unknown kernel {name!r}")
    return kernels


def _score_kernel(
    target: SpikeSlabTarget,
    kernel: Any,
    exact_pips: np.ndarray,
    pi: np.ndarray,
    chain_cfg: dict[str, Any],
) -> dict[str, Any]:
    engine = MetropolisEngine(target, kernel)
    gap = float(spectral_gap(engine.transition_matrix(), pi))

    n_steps = int(chain_cfg["n_steps"])
    burn_in = int(chain_cfg["burn_in"])
    if burn_in >= n_steps:
        raise ValueError(f"burn_in ({burn_in}) must be < n_steps ({n_steps})")
    x0 = np.zeros(target.n_vars, dtype=np.uint8)
    t0 = time.perf_counter()
    result = engine.run(n_steps=n_steps, x0=x0, seed=int(chain_cfg["seed"]))
    elapsed = time.perf_counter() - t0
    kept = result.states[burn_in:]
    size = kept.sum(axis=1).astype(np.float64)
    ess_size = float(ess(size))
    pip_ess = [float(ess(kept[:, j].astype(np.float64))) for j in range(target.n_vars)]
    mean_pip_ess = float(np.mean(pip_ess))
    n_kept = float(kept.shape[0])
    return {
        "kernel": kernel.name,
        "spectral_gap": gap,
        "acceptance_rate": float(result.acceptance_rate),
        "pip_error": float(pip_error(kept, exact_pips)),
        "ess_size": ess_size,
        "ess_size_per_step": ess_size / n_kept,
        "ess_size_per_sec": ess_size / elapsed if elapsed > 0.0 else float("inf"),
        "ess_pip_mean": mean_pip_ess,
        "ess_pip_per_step": mean_pip_ess / n_kept,
        "ess_pip_per_sec": mean_pip_ess / elapsed if elapsed > 0.0 else float("inf"),
        "wall_seconds": elapsed,
        "n_steps": n_steps,
        "burn_in": burn_in,
    }


def _md_table(headers: list[str], rows: list[list[str]]) -> str:
    head = "| " + " | ".join(headers) + " |"
    sep = "| " + " | ".join("---" for _ in headers) + " |"
    body = "\n".join("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join([head, sep, body])


def _diagnosis_dict(diagnosis: Any) -> dict[str, Any]:
    return {
        "spearman": diagnosis.spearman,
        "ground_state": diagnosis.ground_state.astype(int).tolist(),
        "ground_state_log_prob": diagnosis.ground_state_log_prob,
        "ground_state_rank": diagnosis.ground_state_rank,
        "n_states": diagnosis.n_states,
        "top_log_prob": diagnosis.top_log_prob,
    }


def _write_summary(
    path: Path,
    config: dict[str, Any],
    meta: dict[str, Any],
    dataset_label: str,
    diagnoses: dict[str, Any],
    rows: list[dict[str, Any]],
) -> None:
    lines = [
        "# E02a — TunnelVision exact tier",
        "",
        "Spectral gaps and mixing diagnostics on a spike-and-slab posterior.",
        "Accept/reject uses the exact g-prior; the surrogate only shapes proposals.",
        "",
        "## Provenance",
        "",
        f"- git commit: `{meta['git_commit']}`",
        f"- started (UTC): {meta['started_utc']}",
        f"- finished (UTC): {meta['finished_utc']}",
        f"- package versions: {', '.join(f'{k}={v}' for k, v in meta['versions'].items())}",
        f"- dataset: {dataset_label}",
        f"- prior inclusion: {config['prior_inclusion']}",
        f"- quench: {config['quench'].get('evolution', 'exact')}, "
        f"grid {config['quench']['n_gamma']}×{config['quench']['n_t']}",
        f"- chain: {config['chain']['n_steps']} steps, burn-in {config['chain']['burn_in']}",
        "",
        "## Surrogate diagnosis",
        "",
        "If Spearman(−E, log p) is weak, stop — the quench is exploring the wrong landscape.",
        "",
    ]
    diag_rows = []
    for kind in ("analytic", "learned"):
        d = diagnoses[kind]
        diag_rows.append(
            [
                kind,
                f"{d['spearman']:.3f}",
                str(d["ground_state_rank"]),
                f"{d['top_log_prob'] - d['ground_state_log_prob']:.2f}",
            ]
        )
    lines.append(
        _md_table(
            ["surrogate", "Spearman(−E, log p)", "ground-state rank", "Δlogp vs MAP"],
            diag_rows,
        )
    )
    lines.extend(
        [
            "",
            "![analytic scatter](scatter_analytic.png)",
            "",
            "![learned scatter](scatter_learned.png)",
            "",
            "## Exact-tier scoreboard",
            "",
            "Gap is exact (transition-matrix). ESS is sampled, on model size |γ|",
            "and on the mean inclusion-indicator ESS. Per-second numbers include",
            "the quench's simulation cost on purpose.",
            "",
        ]
    )
    score_rows = []
    for row in rows:
        score_rows.append(
            [
                row["kernel"],
                f"{row['spectral_gap']:.4g}",
                f"{row['acceptance_rate']:.3f}",
                f"{row['pip_error']:.3f}",
                f"{row['ess_size_per_step']:.3f}",
                f"{row['ess_size_per_sec']:.2f}",
                f"{row['ess_pip_per_sec']:.2f}",
            ]
        )
    lines.append(
        _md_table(
            [
                "kernel",
                "gap",
                "accept",
                "PIP err",
                "ESS/step (|γ|)",
                "ESS/sec (|γ|)",
                "ESS/sec (PIP)",
            ],
            score_rows,
        )
    )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Add-delete-swap is the baseline that matters. Beating uniform proves nothing.",
            "- A larger gap with a much smaller ESS/sec is not an advantage — report both.",
            "- E02b (ρ-sweep, sampled tier at p=20–27) is not in this run.",
            "",
        ]
    )
    if meta.get("quick"):
        lines.extend(
            [
                "This file was written by ``--quick`` (p=5 synthetic). It is not the",
                "E02 gate and lives under `results/E02_quick/` (gitignored).",
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
        default=Path(__file__).with_name("config.yaml"),
        help="YAML config (default: experiments/E02_tunnelvision_demo/config.yaml)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="p=5 synthetic smoke run, not the diabetes figure",
    )
    args = parser.parse_args(argv)

    config = load_yaml(args.config)
    if args.quick:
        _apply_quick(config)

    X, y, _support, dataset_label = _load_dataset(config)
    prior = float(config["prior_inclusion"])
    target = SpikeSlabTarget(X, y, prior_inclusion=prior)
    analytic = ising_surrogate_from_data(X, y, prior_inclusion=prior)
    learned_cfg = config.get("learned", {})
    learned = learned_surrogate(
        target,
        n_samples=int(learned_cfg.get("n_samples", 2000)),
        seed=int(learned_cfg.get("seed", 0)),
        ridge=float(learned_cfg.get("ridge", 1e-3)),
    )

    output_dir = REPO_ROOT / str(config.get("output_dir", "results/E02"))
    output_dir.mkdir(parents=True, exist_ok=True)
    diagnoses = {
        "analytic": _diagnosis_dict(diagnose_surrogate(analytic, target)),
        "learned": _diagnosis_dict(diagnose_surrogate(learned, target)),
    }
    write_energy_logprob_scatter(analytic, target, output_dir / "scatter_analytic.png")
    write_energy_logprob_scatter(learned, target, output_dir / "scatter_learned.png")

    kernels = _make_kernels(list(config["kernels"]), analytic, learned, dict(config["quench"]))
    exact_pips = target.posterior_inclusion_probs_exact()
    pi = target.enumerate_exact()

    meta = {
        "git_commit": git_commit(),
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "versions": package_versions(),
        "config": config,
        "quick": bool(args.quick),
        "dataset": dataset_label,
        "n_vars": target.n_vars,
        "n_obs": int(X.shape[0]),
        "surrogate": {
            "analytic": {**analytic.meta, **diagnoses["analytic"]},
            "learned": {**learned.meta, **diagnoses["learned"]},
        },
    }
    print(
        f"E02a: {dataset_label}, kernels={list(kernels)}, n_vars={target.n_vars}",
        flush=True,
    )

    rows = [
        _score_kernel(target, kernel, exact_pips, pi, dict(config["chain"]))
        for kernel in kernels.values()
    ]
    meta["finished_utc"] = datetime.now(timezone.utc).isoformat()

    import pandas as pd

    pd.DataFrame(rows).to_csv(output_dir / "scoreboard.csv", index=False)
    (output_dir / "meta.json").write_text(json.dumps(meta, indent=2, default=str) + "\n")
    (output_dir / "summary.json").write_text(
        json.dumps({"diagnoses": diagnoses, "rows": rows}, indent=2) + "\n"
    )
    _write_summary(output_dir / "summary.md", config, meta, dataset_label, diagnoses, rows)
    print(f"wrote {output_dir / 'summary.md'}")
    for row in rows:
        print(
            f"  {row['kernel']:20s}  gap={row['spectral_gap']:.4g}  "
            f"ESS/sec={row['ess_size_per_sec']:.2f}  PIP err={row['pip_error']:.3f}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
