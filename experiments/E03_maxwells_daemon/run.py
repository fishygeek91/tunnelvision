"""E03 — Maxwell's Daemon, phase 1 (exact tier).

Does a ladder of *fixed* depolarizing rungs mix faster than its best
single rung, and does raising λ heat the proposal? Every rung (and
the ladder) targets the same exact g-prior — there is no replica
exchange and no target-temperature schedule.

Classical parallel tempering is the deferred third arm. This script
answers rung-vs-ladder and draws the effective-proposal-temperature
figure that names the paper.

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
from tunnelvision.bits import all_binary_states
from tunnelvision.kernels.classical import AddDeleteSwap
from tunnelvision.kernels.noise_ladder import (
    NoiseLadderKernel,
    invert_temperature,
)
from tunnelvision.kernels.quantum import DepolarizedQuenchKernel
from tunnelvision.surrogate import IsingSurrogate, learned_surrogate
from tunnelvision.targets.spike_slab import SpikeSlabTarget

_ADS_COLOR = "#1a7f37"
_LADDER_COLOR = "#6c3483"
_RUNG_COLOR = "#c0392b"


def _apply_quick(config: dict[str, Any]) -> None:
    config["noise_levels"] = [0.0, 0.1]
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
    temperature = dict(config.get("temperature", {}))
    temperature["n_samples"] = 400
    config["temperature"] = temperature
    config["output_dir"] = "results/E03_quick"


def effective_proposal_temperature(
    proposal: np.ndarray,
    energies: np.ndarray,
    pi: np.ndarray,
    rng: np.random.Generator,
    n_samples: int,
) -> dict[str, float]:
    """T_eff from the mean surrogate energy of y ~ Q(·|x), x ~ π.

    Also reports the model-free jump size ⟨|ΔE|⟩. Both are proposal
    diagnostics — they do not touch accept/reject.
    """
    if n_samples < 1:
        raise ValueError(f"n_samples must be at least 1, got {n_samples}")
    q = np.asarray(proposal, dtype=np.float64)
    e = np.asarray(energies, dtype=np.float64)
    stat = np.asarray(pi, dtype=np.float64)
    if q.ndim != 2 or q.shape[0] != q.shape[1]:
        raise ValueError("proposal must be square")
    n_states = int(q.shape[0])
    if e.shape != (n_states,) or stat.shape != (n_states,):
        raise ValueError("energies and pi must have length 2^n")
    stat = stat / stat.sum()
    starts = rng.choice(n_states, size=n_samples, p=stat)
    dests = np.empty(n_samples, dtype=np.int64)
    for i, src in enumerate(starts):
        dests[i] = int(rng.choice(n_states, p=q[src]))
    e_x = e[starts]
    e_y = e[dests]
    mean_e_y = float(e_y.mean())
    return {
        "mean_energy_x": float(e_x.mean()),
        "mean_energy_y": mean_e_y,
        "mean_abs_dE": float(np.mean(np.abs(e_y - e_x))),
        "t_eff": invert_temperature(e, mean_e_y),
        "t_pi": invert_temperature(e, float(np.dot(stat, e))),
        "n_samples": float(n_samples),
    }


def _write_teff_figure(path: Path, rows: list[dict[str, Any]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    datasets = list(dict.fromkeys(str(row["dataset"]) for row in rows))
    fig, axes = plt.subplots(1, len(datasets), figsize=(5.2 * len(datasets), 3.8), squeeze=False)
    for ax, dataset in zip(axes[0], datasets, strict=True):
        subset = [row for row in rows if row["dataset"] == dataset]
        subset = sorted(subset, key=lambda row: float(row["lambda"]))
        lambdas = [float(row["lambda"]) + 1e-6 for row in subset]
        teffs = [float(row["t_eff"]) for row in subset]
        finite = [t if np.isfinite(t) else np.nan for t in teffs]
        ax.plot(lambdas, finite, marker="o", color=_RUNG_COLOR, label=r"$T_{\mathrm{eff}}$")
        ax.axhline(float(subset[0]["t_pi"]), color="#7f7f7f", ls="--", lw=1.0, label=r"$T_\pi$")
        ax.set_xscale("symlog", linthresh=1e-3)
        ax.set_xlabel(r"depolarizing $\lambda$")
        ax.set_ylabel("effective proposal temperature")
        ax.set_title(dataset)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        ax2 = ax.twinx()
        ax2.plot(
            lambdas,
            [float(row["mean_abs_dE"]) for row in subset],
            marker="s",
            ls="--",
            color="#1f618d",
            label=r"$\langle|\Delta E|\rangle$",
        )
        ax2.set_ylabel(r"mean $|\Delta E|$")
        handles, labels = ax.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(handles + handles2, labels + labels2, fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _write_scoreboard_figure(path: Path, rows: list[dict[str, Any]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    datasets = list(dict.fromkeys(str(row["dataset"]) for row in rows))
    fig, axes = plt.subplots(
        len(datasets), 3, figsize=(11.4, 3.4 * len(datasets)), squeeze=False
    )
    panels = [
        ("spectral_gap", r"spectral gap $\delta$", False),
        ("ess_size_per_step", r"ESS / step ($|\gamma|$)", False),
        ("ess_size_per_sec", r"ESS / sec ($|\gamma|$)", True),
    ]
    for row_i, dataset in enumerate(datasets):
        subset = [row for row in rows if row["dataset"] == dataset]
        names = [str(row["kernel"]) for row in subset]
        colors = []
        for row in subset:
            if row["family"] == "ads":
                colors.append(_ADS_COLOR)
            elif row["family"] == "ladder":
                colors.append(_LADDER_COLOR)
            else:
                colors.append(_RUNG_COLOR)
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


def _read_comparison(rows: list[dict[str, Any]]) -> list[str]:
    lines = [
        "Add-delete-swap is the classical baseline. The ladder wins only",
        "if it beats *both* its best single rung and ADS on the same",
        "column. A bigger gap that costs more wall-clock is not a win.",
        "",
    ]
    datasets = list(dict.fromkeys(str(row["dataset"]) for row in rows))
    for dataset in datasets:
        subset = [row for row in rows if row["dataset"] == dataset]
        ads = next(row for row in subset if row["family"] == "ads")
        ladder = next(row for row in subset if row["family"] == "ladder")
        rungs = [row for row in subset if row["family"] == "rung"]
        best = max(rungs, key=lambda row: float(row["spectral_gap"]))
        ads_gap = float(ads["spectral_gap"])
        best_gap = float(best["spectral_gap"])
        ladder_gap = float(ladder["spectral_gap"])
        vs_best = ladder_gap / best_gap if best_gap > 0.0 else float("inf")
        vs_ads = ladder_gap / ads_gap if ads_gap > 0.0 else float("inf")
        verb = "beats" if vs_best > 1.05 else ("matches" if vs_best > 0.95 else "loses to")
        lines.append(
            f"- **{dataset}**: ladder {verb} its best rung "
            f"(`{best['kernel']}`, {vs_best:.2f}×) and is {vs_ads:.2f}× ADS."
        )
    lines.append("")
    return lines


def _fmt_temp(value: float) -> str:
    if not np.isfinite(value):
        return "∞"
    return f"{value:.3g}"


def _write_summary(
    path: Path,
    config: dict[str, Any],
    meta: dict[str, Any],
    score_rows: list[dict[str, Any]],
    temp_rows: list[dict[str, Any]],
) -> None:
    lines = [
        "# E03 — Maxwell's Daemon (phase 1)",
        "",
        "Fixed depolarizing rungs composed by random-scan, exact tier.",
        "Accept/reject uses the exact g-prior on every rung. Parallel",
        "tempering (the third arm) is deferred.",
        "",
        "## Provenance",
        "",
        f"- git commit: `{meta['git_commit']}`",
        f"- started (UTC): {meta['started_utc']}",
        f"- finished (UTC): {meta['finished_utc']}",
        f"- package versions: {', '.join(f'{k}={v}' for k, v in meta['versions'].items())}",
        f"- λ grid: {', '.join(str(x) for x in config['noise_levels'])}",
        f"- quench: {config['quench'].get('evolution', 'exact')}, "
        f"grid {config['quench']['n_gamma']}×{config['quench']['n_t']}",
        f"- chain: {config['chain']['n_steps']} steps, burn-in {config['chain']['burn_in']}",
        f"- T_eff samples: {config['temperature']['n_samples']}",
        "",
        "## Effective proposal temperature",
        "",
        "x ~ exact π, y ~ Q_λ(·|x), E is the *surrogate* energy. T_eff",
        "is the Boltzmann temperature on that landscape whose mean",
        "energy matches ⟨E(y)⟩. T_π is the same invert for ⟨E(x)⟩.",
        "If T_eff rises with λ, noise is heating the proposal.",
        "",
    ]
    temp_table = []
    for row in temp_rows:
        temp_table.append(
            [
                str(row["dataset"]),
                f"{row['lambda']:g}",
                _fmt_temp(float(row["t_eff"])),
                _fmt_temp(float(row["t_pi"])),
                f"{row['mean_abs_dE']:.3g}",
                f"{row['mean_energy_y']:.3g}",
            ]
        )
    lines.append(
        _md_table(
            ["dataset", "λ", "T_eff", "T_π", "⟨|ΔE|⟩", "⟨E(y)⟩"],
            temp_table,
        )
    )
    lines.extend(
        [
            "",
            "![teff vs lambda](teff_vs_lambda.png)",
            "",
            "## Rung vs ladder vs ADS",
            "",
            "Gap is exact. ESS chains sample the same Q the gap used.",
            "",
        ]
    )
    score_table = []
    for row in score_rows:
        score_table.append(
            [
                str(row["dataset"]),
                str(row["kernel"]),
                f"{row['spectral_gap']:.4g}",
                f"{row['acceptance_rate']:.3f}",
                f"{row['pip_error']:.3f}",
                f"{row['ess_size_per_step']:.3f}",
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
                "ESS/step (|γ|)",
                "ESS/sec (|γ|)",
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
            "- The ladder is a uniform mixture of *fixed* rungs. Choosing",
            "  the rung from chain state would need log-q or diminishing",
            "  adaptation — that is AdaptiveMixture (WP7), not this file.",
            "- Classical parallel tempering is the deferred third arm.",
            "",
        ]
    )
    if meta.get("quick"):
        lines.extend(
            [
                "This file was written by ``--quick`` (p=5 synthetic). It is not",
                "the E03 figure and lives under `results/E03_quick/` (gitignored).",
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
        help="YAML config (default: experiments/E03_maxwells_daemon/config.yaml)",
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

    output_dir = REPO_ROOT / str(config.get("output_dir", "results/E03"))
    output_dir.mkdir(parents=True, exist_ok=True)
    prior = float(config["prior_inclusion"])
    quench_cfg = dict(config["quench"])
    learned_cfg = dict(config.get("learned", {}))
    chain_cfg = dict(config["chain"])
    temp_cfg = dict(config["temperature"])
    noise_levels = [float(x) for x in config["noise_levels"]]
    if any(lam < 0.0 or lam > 1.0 for lam in noise_levels):
        raise ValueError("noise_levels must lie in [0, 1]")
    if len(noise_levels) < 1:
        raise ValueError("need at least one noise level")

    meta: dict[str, Any] = {
        "git_commit": git_commit(),
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "versions": package_versions(),
        "config": config,
        "quick": bool(args.quick),
    }
    print(f"E03: datasets={[_dataset_key(s) for s in config['datasets']]}", flush=True)

    score_rows: list[dict[str, Any]] = []
    temp_rows: list[dict[str, Any]] = []

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
        rungs = [DepolarizedQuenchKernel(inner, lam) for lam in noise_levels]
        ladder = NoiseLadderKernel(rungs)
        ads = AddDeleteSwap()

        print(f"    scoring ADS on {key}...", flush=True)
        ads_row = _score_kernel(target, ads, exact_pips, pi, chain_cfg)
        score_rows.append({**ads_row, "dataset": key, "family": "ads", "lambda": 0.0})

        for kernel, lam in zip(rungs, noise_levels, strict=True):
            print(f"    scoring {kernel.name} on {key}...", flush=True)
            row = _score_kernel(target, kernel, exact_pips, pi, chain_cfg)
            score_rows.append({**row, "dataset": key, "family": "rung", "lambda": lam})

        print(f"    scoring {ladder.name} on {key}...", flush=True)
        ladder_row = _score_kernel(target, ladder, exact_pips, pi, chain_cfg)
        score_rows.append(
            {**ladder_row, "dataset": key, "family": "ladder", "lambda": float("nan")}
        )

        energies = learned.energies(all_binary_states(target.n_vars))
        rng = np.random.Generator(np.random.PCG64(int(temp_cfg["seed"])))
        n_temp = int(temp_cfg["n_samples"])
        for kernel, lam in zip(rungs, noise_levels, strict=True):
            print(f"    T_eff for {kernel.name} on {key}...", flush=True)
            t0 = time.perf_counter()
            stats = effective_proposal_temperature(
                kernel.proposal_matrix(target.n_vars),
                energies,
                pi,
                rng,
                n_temp,
            )
            temp_rows.append(
                {
                    **stats,
                    "dataset": key,
                    "kernel": kernel.name,
                    "lambda": lam,
                    "wall_seconds": time.perf_counter() - t0,
                }
            )
            print(
                f"    T_eff={_fmt_temp(stats['t_eff'])}  "
                f"⟨|ΔE|⟩={stats['mean_abs_dE']:.3g}",
                flush=True,
            )

    meta["finished_utc"] = datetime.now(timezone.utc).isoformat()

    import pandas as pd

    pd.DataFrame(score_rows).to_csv(output_dir / "scoreboard.csv", index=False)
    pd.DataFrame(temp_rows).to_csv(output_dir / "temperature.csv", index=False)
    (output_dir / "meta.json").write_text(json.dumps(meta, indent=2, default=str) + "\n")
    (output_dir / "summary.json").write_text(
        json.dumps({"score": score_rows, "temperature": temp_rows}, indent=2, default=str)
        + "\n"
    )
    _write_teff_figure(output_dir / "teff_vs_lambda.png", temp_rows)
    _write_scoreboard_figure(output_dir / "scoreboard.png", score_rows)
    _write_summary(output_dir / "summary.md", config, meta, score_rows, temp_rows)
    print(f"wrote {output_dir / 'summary.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
