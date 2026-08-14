"""E03 — rung vs ladder vs classical parallel tempering (exact tier).

Phase 1 already measured T_eff(λ) and showed the noise ladder matches
its coldest rung and loses to ADS. This script is the deferred third
arm: replica exchange on the *exact* g-prior (target temperatures),
plus the Aer-realistic hot rung (λ̂ ≈ 0.65) the ablation mapped.

PT has no 2^p spectral gap — the joint chain is not a kernel on the
original state space. The scoreboard for that arm is PIP error, ESS
per cold-chain step, ESS per target evaluation, and ESS per second.
A bigger ESS/step that costs K× evaluations is not a win.

``--quick`` drops to p=5 so the entry point is smoke-testable in seconds.
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

from experiments.E02_tunnelvision_demo.run import _md_table, _quench, _score_kernel
from experiments.E02_tunnelvision_demo.run_ablations import (
    _dataset_key,
    _load_dataset_spec,
)
from experiments.provenance import REPO_ROOT, git_commit, load_yaml, package_versions

os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / ".mplconfig"))
from tunnelvision.diagnostics import ess, pip_error
from tunnelvision.kernels.classical import AddDeleteSwap
from tunnelvision.kernels.noise_ladder import NoiseLadderKernel
from tunnelvision.kernels.quantum import DepolarizedQuenchKernel
from tunnelvision.surrogate import IsingSurrogate, learned_surrogate
from tunnelvision.targets.spike_slab import SpikeSlabTarget
from tunnelvision.tempering import ParallelTempering

_ADS_COLOR = "#1a7f37"
_PT_COLOR = "#1f4e79"
_LADDER_COLOR = "#6c3483"
_RUNG_COLOR = "#c0392b"
_HOT_COLOR = "#d35400"


def _apply_quick(config: dict[str, Any]) -> None:
    config["ladder_levels"] = [0.0, 0.1]
    config["hot_lambda"] = 0.65
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
    chain = dict(config.get("chain", {}))
    chain.update({"n_steps": 400, "burn_in": 50})
    config["chain"] = chain
    pt = dict(config.get("pt", {}))
    pt.update({"n_replicas": 3, "beta_min": 0.3})
    config["pt"] = pt
    config["output_dir"] = "results/E03_quick/pt"


def _with_eval_cost(row: dict[str, Any], n_replicas: int = 1) -> dict[str, Any]:
    """Single-kernel MH spends one target eval per step."""
    per_step = float(row["ess_size_per_step"])
    return {
        **row,
        "n_replicas": int(n_replicas),
        "ess_size_per_eval": per_step / float(n_replicas),
    }


def _score_pt(
    target: SpikeSlabTarget,
    exact_pips: np.ndarray,
    chain_cfg: dict[str, Any],
    pt_cfg: dict[str, Any],
) -> dict[str, Any]:
    n_steps = int(chain_cfg["n_steps"])
    burn_in = int(chain_cfg["burn_in"])
    if burn_in >= n_steps:
        raise ValueError(f"burn_in ({burn_in}) must be < n_steps ({n_steps})")
    n_replicas = int(pt_cfg["n_replicas"])
    if n_replicas < 1:
        raise ValueError(f"n_replicas must be at least 1, got {n_replicas}")
    sampler = ParallelTempering(
        target,
        AddDeleteSwap(),
        n_replicas=n_replicas,
        beta_min=float(pt_cfg["beta_min"]),
    )
    x0 = np.zeros(target.n_vars, dtype=np.uint8)
    t0 = time.perf_counter()
    result = sampler.run(n_steps=n_steps, x0=x0, seed=int(pt_cfg.get("seed", 0)))
    elapsed = time.perf_counter() - t0
    kept = result.states[burn_in:]
    size = kept.sum(axis=1).astype(np.float64)
    ess_size = float(ess(size))
    n_kept = float(kept.shape[0])
    swap_rates = np.asarray(result.meta["swap_accept_rates"], dtype=np.float64)
    return {
        "kernel": result.kernel_name,
        "spectral_gap": float("nan"),
        "acceptance_rate": float(result.acceptance_rate),
        "pip_error": float(pip_error(kept, exact_pips)),
        "ess_size": ess_size,
        "ess_size_per_step": ess_size / n_kept,
        "ess_size_per_eval": ess_size / (n_replicas * n_kept),
        "ess_size_per_sec": ess_size / elapsed if elapsed > 0.0 else float("inf"),
        "ess_pip_mean": float("nan"),
        "gap_seconds": float("nan"),
        "wall_seconds": elapsed,
        "n_steps": n_steps,
        "burn_in": burn_in,
        "n_replicas": n_replicas,
        "n_target_evals": int(result.meta["n_target_evals"]),
        "swap_accept_rate_mean": float(np.nanmean(swap_rates)) if swap_rates.size else float("nan"),
        "betas": result.meta["betas"],
    }


def _family_color(family: str) -> str:
    return {
        "ads": _ADS_COLOR,
        "pt": _PT_COLOR,
        "ladder": _LADDER_COLOR,
        "rung": _RUNG_COLOR,
        "hot": _HOT_COLOR,
    }.get(family, "#2c3e50")


def _write_scoreboard_figure(path: Path, rows: list[dict[str, Any]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    datasets = list(dict.fromkeys(str(row["dataset"]) for row in rows))
    fig, axes = plt.subplots(
        len(datasets), 4, figsize=(14.4, 3.4 * len(datasets)), squeeze=False
    )
    panels = [
        ("pip_error", "PIP error", False),
        ("ess_size_per_step", r"ESS / step ($|\gamma|$)", False),
        ("ess_size_per_eval", r"ESS / target-eval ($|\gamma|$)", False),
        ("ess_size_per_sec", r"ESS / sec ($|\gamma|$)", True),
    ]
    for row_i, dataset in enumerate(datasets):
        subset = [row for row in rows if row["dataset"] == dataset]
        names = [str(row["kernel"]) for row in subset]
        colors = [_family_color(str(row["family"])) for row in subset]
        x = np.arange(len(names))
        for ax, (key, ylabel, log_y) in zip(axes[row_i], panels, strict=True):
            values = np.array([float(row[key]) for row in subset], dtype=np.float64)
            ax.bar(x, values, color=colors, width=0.72)
            ax.set_xticks(x)
            ax.set_xticklabels(names, rotation=35, ha="right", fontsize=7)
            ax.set_ylabel(ylabel)
            if row_i == 0:
                ax.set_title(ylabel)
            if log_y:
                positive = values[values > 0.0]
                ax.set_yscale("log")
                if positive.size:
                    ax.set_ylim(positive.min() * 0.5, positive.max() * 2.0)
            ax.grid(True, axis="y", alpha=0.3)
            ads = next((row for row in subset if row["family"] == "ads"), None)
            if ads is not None and float(ads[key]) > 0.0:
                ax.axhline(float(ads[key]), color=_ADS_COLOR, ls="--", lw=0.8)
        axes[row_i, 0].text(
            0.0,
            1.12,
            dataset,
            transform=axes[row_i, 0].transAxes,
            fontsize=10,
            fontweight="bold",
        )
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _fmt_gap(value: float) -> str:
    if not np.isfinite(value):
        return "—"
    return f"{value:.4g}"


def _read_comparison(rows: list[dict[str, Any]]) -> list[str]:
    lines = [
        "Add-delete-swap is the classical baseline. Parallel tempering",
        "wins only if it beats ADS on ESS/eval (the fair cost) *and*",
        "beats the ladder / best rung on the same column. A larger",
        "ESS/step that spends K target evaluations is not a win.",
        "The hot rung (λ̂ ≈ 0.65) is the Aer-realistic slice.",
        "",
    ]
    datasets = list(dict.fromkeys(str(row["dataset"]) for row in rows))
    for dataset in datasets:
        subset = [row for row in rows if row["dataset"] == dataset]
        by_family = {str(row["family"]): row for row in subset}
        ads = by_family["ads"]
        pt = by_family["pt"]
        ladder = by_family["ladder"]
        best = by_family["rung"]
        hot = by_family["hot"]
        ads_eval = float(ads["ess_size_per_eval"])
        pt_eval = float(pt["ess_size_per_eval"])
        ladder_eval = float(ladder["ess_size_per_eval"])
        best_eval = float(best["ess_size_per_eval"])
        hot_eval = float(hot["ess_size_per_eval"])
        vs_ads = pt_eval / ads_eval if ads_eval > 0.0 else float("inf")
        vs_ladder = pt_eval / ladder_eval if ladder_eval > 0.0 else float("inf")
        vs_best = pt_eval / best_eval if best_eval > 0.0 else float("inf")
        hot_vs_best = hot_eval / best_eval if best_eval > 0.0 else float("inf")
        verb = "beats" if vs_ads > 1.05 else ("matches" if vs_ads > 0.95 else "loses to")
        lines.append(
            f"- **{dataset}**: PT {verb} ADS on ESS/eval ({vs_ads:.2f}×); "
            f"{vs_ladder:.2f}× ladder, {vs_best:.2f}× best rung "
            f"(`{best['kernel']}`). Hot rung is {hot_vs_best:.2f}× the "
            f"λ=0 rung on ESS/eval. PIP errors: ADS {ads['pip_error']:.3f}, "
            f"PT {pt['pip_error']:.3f}, hot {hot['pip_error']:.3f}."
        )
    lines.append("")
    return lines


def _write_summary(
    path: Path,
    config: dict[str, Any],
    meta: dict[str, Any],
    score_rows: list[dict[str, Any]],
) -> None:
    pt_cfg = dict(config["pt"])
    lines = [
        "# E03 — parallel tempering vs rung vs ladder",
        "",
        "Classical replica exchange on the exact g-prior, compared to",
        "the phase-1 noise ladder, its coldest rung, ADS, and the",
        "Aer-realistic hot rung (λ̂ ≈ 0.65). Accept/reject is exact on",
        "every arm. PT has no 2^p spectral gap — that cell is a dash.",
        "",
        "## Provenance",
        "",
        f"- git commit: `{meta['git_commit']}`",
        f"- started (UTC): {meta['started_utc']}",
        f"- finished (UTC): {meta['finished_utc']}",
        f"- package versions: {', '.join(f'{k}={v}' for k, v in meta['versions'].items())}",
        f"- ladder λ: {', '.join(str(x) for x in config['ladder_levels'])}",
        f"- hot λ: {config['hot_lambda']}",
        f"- PT: K={pt_cfg['n_replicas']}, β_min={pt_cfg['beta_min']}",
        f"- quench: {config['quench'].get('evolution', 'exact')}, "
        f"grid {config['quench']['n_gamma']}×{config['quench']['n_t']}",
        f"- chain: {config['chain']['n_steps']} steps, burn-in {config['chain']['burn_in']}",
        "",
        "## Scoreboard",
        "",
        "Gap is exact for single-kernel arms. ESS is from chains that",
        "sample the same Q the gap used (tabulated), except PT, which",
        "runs real ADS + swaps. ESS/eval charges K evaluations per PT",
        "sweep.",
        "",
    ]
    score_table = []
    for row in score_rows:
        score_table.append(
            [
                str(row["dataset"]),
                str(row["kernel"]),
                _fmt_gap(float(row["spectral_gap"])),
                f"{row['acceptance_rate']:.3f}",
                f"{row['pip_error']:.3f}",
                f"{row['ess_size_per_step']:.3f}",
                f"{row['ess_size_per_eval']:.3f}",
                f"{row['ess_size_per_sec']:.2f}",
            ]
        )
    lines.append(
        _md_table(
            [
                "dataset",
                "kernel",
                "gap",
                "accept",
                "PIP err",
                "ESS/step",
                "ESS/eval",
                "ESS/sec",
            ],
            score_table,
        )
    )
    lines.extend(
        [
            "",
            "![scoreboard](scoreboard.png)",
            "",
            "## Read",
            "",
            *_read_comparison(score_rows),
            "## Notes",
            "",
            "- Add-delete-swap is the baseline that matters.",
            "- Parallel tempering changes the *target* temperature of",
            "  K replicas and swaps them. The noise ladder changes the",
            "  *proposal* temperature of one chain. They are not the",
            "  same object.",
            "- PT's joint chain has no spectral gap on {0,1}^p. Do not",
            "  invent one.",
            "",
        ]
    )
    if meta.get("quick"):
        lines.extend(
            [
                "This file was written by ``--quick`` (p=5 synthetic). It is not",
                "the E03 figure and lives under `results/E03_quick/pt/` (gitignored).",
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
        default=Path(__file__).with_name("config_pt.yaml"),
        help="YAML config (default: experiments/E03_maxwells_daemon/config_pt.yaml)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="p=5 synthetic smoke run, not the E03 figure",
    )
    args = parser.parse_args(argv)

    config = load_yaml(args.config)
    if args.quick:
        _apply_quick(config)

    output_dir = REPO_ROOT / str(config.get("output_dir", "results/E03/pt"))
    output_dir.mkdir(parents=True, exist_ok=True)
    prior = float(config["prior_inclusion"])
    quench_cfg = dict(config["quench"])
    learned_cfg = dict(config.get("learned", {}))
    chain_cfg = dict(config["chain"])
    pt_cfg = dict(config["pt"])
    ladder_levels = [float(x) for x in config["ladder_levels"]]
    hot_lambda = float(config["hot_lambda"])
    if any(lam < 0.0 or lam > 1.0 for lam in [*ladder_levels, hot_lambda]):
        raise ValueError("noise levels must lie in [0, 1]")
    if not ladder_levels:
        raise ValueError("need at least one ladder level")
    if 0.0 not in ladder_levels:
        raise ValueError("ladder_levels must include 0.0 (the best-rung arm)")

    meta: dict[str, Any] = {
        "git_commit": git_commit(),
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "versions": package_versions(),
        "config": config,
        "quick": bool(args.quick),
    }
    print(f"E03-PT: datasets={[_dataset_key(s) for s in config['datasets']]}", flush=True)

    score_rows: list[dict[str, Any]] = []

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

        inner = _quench(learned, "quench-learned", quench_cfg)
        rungs = [DepolarizedQuenchKernel(inner, lam) for lam in ladder_levels]
        best = next(kernel for kernel, lam in zip(rungs, ladder_levels, strict=True) if lam == 0.0)
        ladder = NoiseLadderKernel(rungs)
        hot = DepolarizedQuenchKernel(inner, hot_lambda)
        ads = AddDeleteSwap()

        print(f"    scoring ADS on {key}...", flush=True)
        ads_row = _with_eval_cost(_score_kernel(target, ads, exact_pips, pi, chain_cfg))
        score_rows.append({**ads_row, "dataset": key, "family": "ads", "lambda": 0.0})

        print(f"    scoring {best.name} on {key}...", flush=True)
        best_row = _with_eval_cost(_score_kernel(target, best, exact_pips, pi, chain_cfg))
        score_rows.append({**best_row, "dataset": key, "family": "rung", "lambda": 0.0})

        print(f"    scoring {ladder.name} on {key}...", flush=True)
        ladder_row = _with_eval_cost(_score_kernel(target, ladder, exact_pips, pi, chain_cfg))
        score_rows.append(
            {**ladder_row, "dataset": key, "family": "ladder", "lambda": float("nan")}
        )

        print(f"    scoring {hot.name} on {key}...", flush=True)
        hot_row = _with_eval_cost(_score_kernel(target, hot, exact_pips, pi, chain_cfg))
        score_rows.append({**hot_row, "dataset": key, "family": "hot", "lambda": hot_lambda})

        print(f"    scoring parallel-tempering on {key}...", flush=True)
        pt_row = _score_pt(target, exact_pips, chain_cfg, pt_cfg)
        score_rows.append({**pt_row, "dataset": key, "family": "pt", "lambda": float("nan")})

    meta["finished_utc"] = datetime.now(timezone.utc).isoformat()

    import pandas as pd

    pd.DataFrame(score_rows).to_csv(output_dir / "scoreboard.csv", index=False)
    (output_dir / "meta.json").write_text(json.dumps(meta, indent=2, default=str) + "\n")
    (output_dir / "summary.json").write_text(
        json.dumps({"score": score_rows}, indent=2, default=str) + "\n"
    )
    _write_scoreboard_figure(output_dir / "scoreboard.png", score_rows)
    _write_summary(output_dir / "summary.md", config, meta, score_rows)
    print(f"wrote {output_dir / 'summary.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
