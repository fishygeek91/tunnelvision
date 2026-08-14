"""E02 ablations — where the quench/ADS gap dies.

Two sweeps, exact tier:

- Noise: wrap quench-analytic and quench-learned in a global depolarizing
  channel Q_λ = (1−λ) Q + λ/2^p 11ᵀ. λ is the roadmap ladder. ADS is
  the flat baseline. A cheap Aer density-matrix check at p=6 maps
  per-gate depolarizing onto that same λ so the p=10 curve is credible.
- Surrogate quality: corrupt the learned (h, J) at relative scale σ,
  re-pin Frobenius (α stays 1), and watch Spearman, gap, and PIP error.
  PIP error is the wall check — it must stay at Monte Carlo error at
  every σ. If it tracks the surrogate, something crossed the wall.

``--quick`` drops to p=5 and skips Aer so the entry point is seconds.
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

from experiments.E02_tunnelvision_demo.run import (
    _md_table,
    _quench,
    _score_kernel,
)
from experiments.provenance import REPO_ROOT, git_commit, load_yaml, package_versions

os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / ".mplconfig"))
from tunnelvision.data.loaders import correlated_synthetic, load_diabetes
from tunnelvision.kernels.classical import AddDeleteSwap
from tunnelvision.kernels.quantum import (
    DepolarizedQuenchKernel,
    QuenchKernel,
    aer_noisy_proposal_matrix,
    depolarize_proposal,
    fit_global_depolarize,
)
from tunnelvision.surrogate import (
    IsingSurrogate,
    corrupt_surrogate,
    diagnose_surrogate,
    ising_surrogate_from_data,
    learned_surrogate,
)
from tunnelvision.targets.spike_slab import SpikeSlabTarget

_QUENCH_COLORS = {
    "quench-analytic": "#c0392b",
    "quench-learned": "#8e1b12",
}
_ADS_COLOR = "#1a7f37"


def _apply_quick(config: dict[str, Any]) -> None:
    """Tiny synthetic so the ablation entry point is smoke-testable in seconds."""
    config["noise_levels"] = [0.0, 0.1]
    config["surrogate_sigmas"] = [0.0, 1.0]
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
    aer = dict(config.get("aer_validation", {}))
    aer["enabled"] = False
    config["aer_validation"] = aer
    config["output_dir"] = "results/E02_quick/ablations"


def _load_dataset_spec(spec: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, str]:
    name = str(spec["name"]).lower()
    if name == "diabetes":
        X, y = load_diabetes()
        return X, y, "diabetes (sklearn, p=10)"
    if name == "synthetic":
        X, y, _support = correlated_synthetic(
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
        return X, y, label
    raise ValueError(f"dataset name must be 'diabetes' or 'synthetic', got {name!r}")


def _dataset_key(spec: dict[str, Any]) -> str:
    name = str(spec["name"]).lower()
    if name == "diabetes":
        return "diabetes"
    return f"synthetic-rho-{spec['rho']}"


def _diagnosis_dict(diagnosis: Any) -> dict[str, Any]:
    return {
        "spearman": diagnosis.spearman,
        "ground_state": diagnosis.ground_state.astype(int).tolist(),
        "ground_state_log_prob": diagnosis.ground_state_log_prob,
        "ground_state_rank": diagnosis.ground_state_rank,
        "n_states": diagnosis.n_states,
        "top_log_prob": diagnosis.top_log_prob,
    }


def _mean_row_tv(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.mean(0.5 * np.abs(left - right).sum(axis=1)))


def _run_aer_validation(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Map per-gate depolarizing onto the global-λ family at small p."""
    aer_cfg = dict(config.get("aer_validation", {}))
    if not bool(aer_cfg.get("enabled", True)):
        return []

    n_vars = int(aer_cfg["p"])
    if n_vars > 7:
        raise ValueError(f"aer_validation.p must be <= 7, got {n_vars}")
    X, y, _support = correlated_synthetic(
        n=int(aer_cfg["n"]),
        p=n_vars,
        rho=float(aer_cfg["rho"]),
        k_true=int(aer_cfg["k_true"]),
        snr=float(aer_cfg["snr"]),
        seed=int(aer_cfg["seed"]),
        structure="equicorrelated",
    )
    prior = float(config["prior_inclusion"])
    surrogate = ising_surrogate_from_data(X, y, prior_inclusion=prior)
    quench_cfg = dict(config["quench"])
    kernel = QuenchKernel(
        surrogate.h,
        surrogate.J,
        gamma_range=tuple(quench_cfg["gamma_range"]),
        t_range=tuple(quench_cfg["t_range"]),
        trotter_dt=float(quench_cfg.get("trotter_dt", 0.8)),
        evolution="trotter",
        n_gamma=int(aer_cfg["n_gamma"]),
        n_t=int(aer_cfg["n_t"]),
    )
    print(
        f"  Aer validation: p={n_vars}, grid {aer_cfg['n_gamma']}×{aer_cfg['n_t']}",
        flush=True,
    )
    q_ideal = kernel.proposal_matrix(n_vars)
    rows: list[dict[str, Any]] = []
    for gate_error in aer_cfg["gate_errors"]:
        p_err = float(gate_error)
        print(f"    gate_error={p_err:g} density-matrix Q...", flush=True)
        t0 = time.perf_counter()
        q_aer = aer_noisy_proposal_matrix(
            kernel,
            p_err,
            n_gamma=int(aer_cfg["n_gamma"]),
            n_t=int(aer_cfg["n_t"]),
        )
        elapsed = time.perf_counter() - t0
        lam = fit_global_depolarize(q_ideal, q_aer)
        q_lambda = depolarize_proposal(q_ideal, lam)
        row = {
            "gate_error": p_err,
            "lambda_hat": lam,
            "tv_aer_vs_ideal": _mean_row_tv(q_aer, q_ideal),
            "tv_aer_vs_lambda": _mean_row_tv(q_aer, q_lambda),
            "wall_seconds": elapsed,
            "n_vars": n_vars,
            "n_gamma": int(aer_cfg["n_gamma"]),
            "n_t": int(aer_cfg["n_t"]),
        }
        rows.append(row)
        print(
            f"    gate_error={p_err:g}  λ̂={lam:.4g}  "
            f"TV(aer,ideal)={row['tv_aer_vs_ideal']:.4g}  "
            f"TV(aer,Q_λ)={row['tv_aer_vs_lambda']:.4g}  ({elapsed:.1f}s)",
            flush=True,
        )
    return rows


def _score_ads(
    target: SpikeSlabTarget,
    exact_pips: np.ndarray,
    pi: np.ndarray,
    chain_cfg: dict[str, Any],
) -> dict[str, Any]:
    return _score_kernel(target, AddDeleteSwap(), exact_pips, pi, chain_cfg)


def _write_noise_figure(path: Path, rows: list[dict[str, Any]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    datasets = list(dict.fromkeys(str(row["dataset"]) for row in rows))
    fig, axes = plt.subplots(1, len(datasets), figsize=(5.2 * len(datasets), 3.8), squeeze=False)
    for ax, dataset in zip(axes[0], datasets, strict=True):
        ads = next(row for row in rows if row["dataset"] == dataset and row["family"] == "ads")
        ax.axhline(float(ads["spectral_gap"]), color=_ADS_COLOR, ls="--", lw=1.0, label="ADS")
        for family, color in _QUENCH_COLORS.items():
            subset = [
                row
                for row in rows
                if row["dataset"] == dataset and row["family"] == family
            ]
            if not subset:
                continue
            subset = sorted(subset, key=lambda row: float(row["lambda"]))
            ax.plot(
                [float(row["lambda"]) + 1e-6 for row in subset],
                [float(row["spectral_gap"]) for row in subset],
                marker="o",
                color=color,
                label=family,
            )
        ax.set_xscale("symlog", linthresh=1e-3)
        ax.set_xlabel(r"depolarizing $\lambda$")
        ax.set_ylabel(r"spectral gap $\delta$")
        ax.set_title(dataset)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _write_sigma_figure(path: Path, rows: list[dict[str, Any]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    datasets = list(dict.fromkeys(str(row["dataset"]) for row in rows))
    fig, axes = plt.subplots(1, len(datasets), figsize=(5.4 * len(datasets), 3.8), squeeze=False)
    for ax, dataset in zip(axes[0], datasets, strict=True):
        subset = sorted(
            [row for row in rows if row["dataset"] == dataset],
            key=lambda row: float(row["sigma"]),
        )
        color = _QUENCH_COLORS["quench-learned"]
        ax.plot(
            [float(row["sigma"]) for row in subset],
            [float(row["spectral_gap"]) for row in subset],
            marker="o",
            color=color,
            label=r"gap $\delta$",
        )
        ax.set_xlabel(r"surrogate corruption $\sigma$")
        ax.set_ylabel(r"spectral gap $\delta$")
        ax.set_title(dataset)
        ax.grid(True, alpha=0.3)
        ax2 = ax.twinx()
        ax2.plot(
            [float(row["sigma"]) for row in subset],
            [float(row["spearman"]) for row in subset],
            marker="s",
            ls="--",
            color="#7f7f7f",
            label=r"Spearman$(-E,\log p)$",
        )
        ax2.set_ylabel("Spearman")
        ax2.set_ylim(-0.05, 1.05)
        handles, labels = ax.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(handles + handles2, labels + labels2, fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _write_aer_figure(path: Path, rows: list[dict[str, Any]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.6))
    errors = [float(row["gate_error"]) for row in rows]
    axes[0].plot(errors, [float(row["lambda_hat"]) for row in rows], marker="o", color="#6c3483")
    axes[0].set_xlabel("per-gate depolarizing p")
    axes[0].set_ylabel(r"fitted global $\hat\lambda$")
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title("gate error → global λ")
    axes[1].plot(
        errors,
        [float(row["tv_aer_vs_ideal"]) for row in rows],
        marker="o",
        color="#7f7f7f",
        label="TV(Aer, ideal)",
    )
    axes[1].plot(
        errors,
        [float(row["tv_aer_vs_lambda"]) for row in rows],
        marker="s",
        color="#6c3483",
        label=r"TV(Aer, $Q_{\hat\lambda}$)",
    )
    axes[1].set_xlabel("per-gate depolarizing p")
    axes[1].set_ylabel("mean row TV")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=8)
    axes[1].set_title("global-λ model vs raw Aer Q")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _read_noise(rows: list[dict[str, Any]]) -> list[str]:
    lines = [
        "Add-delete-swap is the flat baseline. The question is not whether",
        "quench beats it — E02a/b already said no — but where the remaining",
        "gap ratio collapses as proposals are heated toward uniform.",
        "",
    ]
    datasets = list(dict.fromkeys(str(row["dataset"]) for row in rows))
    for dataset in datasets:
        ads = next(row for row in rows if row["dataset"] == dataset and row["family"] == "ads")
        ads_gap = float(ads["spectral_gap"])
        lines.append(f"**{dataset}** (ADS gap {ads_gap:.4g}):")
        for family in ("quench-analytic", "quench-learned"):
            subset = [
                row
                for row in rows
                if row["dataset"] == dataset and row["family"] == family
            ]
            if not subset:
                continue
            subset = sorted(subset, key=lambda row: float(row["lambda"]))
            first = subset[0]
            last = subset[-1]
            r0 = float(first["spectral_gap"]) / ads_gap if ads_gap > 0.0 else float("inf")
            r1 = float(last["spectral_gap"]) / ads_gap if ads_gap > 0.0 else float("inf")
            lines.append(
                f"- `{family}` / ADS is {r0:.2f}× at λ={float(first['lambda']):g} "
                f"and {r1:.2f}× at λ={float(last['lambda']):g}."
            )
        lines.append("")
    return lines


def _read_sigma(rows: list[dict[str, Any]]) -> list[str]:
    lines = [
        "PIP error is the wall check. Destroying the surrogate may collapse",
        "the gap; it must not move the sampled posterior beyond Monte Carlo",
        "error. A PIP-error column that tracks σ is a P0 bug.",
        "",
    ]
    datasets = list(dict.fromkeys(str(row["dataset"]) for row in rows))
    for dataset in datasets:
        subset = [row for row in rows if row["dataset"] == dataset]
        pips = [float(row["pip_error"]) for row in subset]
        spearmans = [float(row["spearman"]) for row in subset]
        lines.append(
            f"- **{dataset}**: Spearman {spearmans[0]:.3f} → {spearmans[-1]:.3f}; "
            f"PIP err stays in [{min(pips):.3f}, {max(pips):.3f}]."
        )
    return lines


def _write_summary(
    path: Path,
    config: dict[str, Any],
    meta: dict[str, Any],
    noise_rows: list[dict[str, Any]],
    sigma_rows: list[dict[str, Any]],
    aer_rows: list[dict[str, Any]],
) -> None:
    lines = [
        "# E02 ablations — noise and surrogate quality",
        "",
        "Exact-tier sweeps on the same spike-and-slab posteriors as E02a/b.",
        "Accept/reject uses the exact g-prior; noise and surrogate quality",
        "may only change the proposal, never the target.",
        "",
        "## Provenance",
        "",
        f"- git commit: `{meta['git_commit']}`",
        f"- started (UTC): {meta['started_utc']}",
        f"- finished (UTC): {meta['finished_utc']}",
        f"- package versions: {', '.join(f'{k}={v}' for k, v in meta['versions'].items())}",
        f"- λ grid: {', '.join(str(x) for x in config['noise_levels'])}",
        f"- σ grid: {', '.join(str(x) for x in config['surrogate_sigmas'])}",
        f"- quench: {config['quench'].get('evolution', 'exact')}, "
        f"grid {config['quench']['n_gamma']}×{config['quench']['n_t']}",
        f"- chain: {config['chain']['n_steps']} steps, burn-in {config['chain']['burn_in']}",
        "",
        "## Ablation A — depolarizing noise",
        "",
        "Global channel `Q_λ = (1−λ) Q + λ/2^p 11ᵀ` on the cached quench",
        "proposal. Unital, so the kernel stays symmetric. ADS does not",
        "see λ — it is the flat classical baseline.",
        "",
    ]
    noise_table = []
    for row in noise_rows:
        noise_table.append(
            [
                str(row["dataset"]),
                str(row["kernel"]),
                f"{row['lambda']:g}",
                f"{row['spectral_gap']:.4g}",
                f"{row['acceptance_rate']:.3f}",
                f"{row['pip_error']:.3f}",
                f"{row['ess_size_per_step']:.3f}",
            ]
        )
    lines.append(
        _md_table(
            ["dataset", "kernel", "λ", "gap", "accept", "PIP err", "ESS/step (|γ|)"],
            noise_table,
        )
    )
    lines.extend(
        [
            "",
            "![gap vs lambda](gap_vs_lambda.png)",
            "",
            *_read_noise(noise_rows),
            "## Ablation B — surrogate quality",
            "",
            "Relative Gaussian noise on the learned (h, J), then re-pin",
            "Frobenius so α stays 1. Shape is destroyed; scale is not.",
            "",
        ]
    )
    sigma_table = []
    for row in sigma_rows:
        sigma_table.append(
            [
                str(row["dataset"]),
                f"{row['sigma']:g}",
                f"{row['spearman']:.3f}",
                str(row["ground_state_rank"]),
                f"{row['spectral_gap']:.4g}",
                f"{row['pip_error']:.3f}",
                f"{row['ess_size_per_step']:.3f}",
            ]
        )
    lines.append(
        _md_table(
            ["dataset", "σ", "Spearman", "ground rank", "gap", "PIP err", "ESS/step (|γ|)"],
            sigma_table,
        )
    )
    lines.extend(
        [
            "",
            "![gap vs sigma](gap_vs_sigma.png)",
            "",
            *_read_sigma(sigma_rows),
            "",
        ]
    )
    if aer_rows:
        lines.extend(
            [
                "## Aer validation — per-gate vs global λ",
                "",
                "Trotter circuit, density-matrix Aer, p=6, reduced (γ, t) grid.",
                "If TV(Aer, Q_λ̂) stays below TV(Aer, Q_ideal) once the gates",
                "are noisy, the p=10 analytic sweep is a fair proxy.",
                "",
            ]
        )
        aer_table = []
        for row in aer_rows:
            aer_table.append(
                [
                    f"{row['gate_error']:g}",
                    f"{row['lambda_hat']:.4g}",
                    f"{row['tv_aer_vs_ideal']:.4g}",
                    f"{row['tv_aer_vs_lambda']:.4g}",
                    f"{row['wall_seconds']:.1f}",
                ]
            )
        lines.append(
            _md_table(
                ["gate p", "λ̂", "TV(Aer, ideal)", "TV(Aer, Q_λ̂)", "wall (s)"],
                aer_table,
            )
        )
        lines.extend(["", "![aer validation](aer_validation.png)", ""])
    lines.extend(
        [
            "## Notes",
            "",
            "- Add-delete-swap is the baseline that matters. Beating uniform proves nothing.",
            "- Headline comparison is the exact gap. ESS/sec is tabulated-Q, not QPU time.",
            "- Sampled p=20–27 is still open; these ablations stay at the enumerable tier.",
            "",
        ]
    )
    if meta.get("quick"):
        lines.extend(
            [
                "This file was written by ``--quick`` (p=5 synthetic). It is not the",
                "E02 ablation figure and lives under `results/E02_quick/` (gitignored).",
                "",
            ]
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def _prepare_target(
    spec: dict[str, Any],
    prior: float,
    learned_cfg: dict[str, Any],
) -> tuple[str, SpikeSlabTarget, IsingSurrogate, IsingSurrogate, dict[str, Any], str]:
    X, y, label = _load_dataset_spec(spec)
    key = _dataset_key(spec)
    target = SpikeSlabTarget(X, y, prior_inclusion=prior)
    analytic = ising_surrogate_from_data(X, y, prior_inclusion=prior)
    learned = learned_surrogate(
        target,
        n_samples=int(learned_cfg.get("n_samples", 2000)),
        seed=int(learned_cfg.get("seed", 0)),
        ridge=float(learned_cfg.get("ridge", 1e-3)),
    )
    diagnoses = {
        "analytic": _diagnosis_dict(diagnose_surrogate(analytic, target)),
        "learned": _diagnosis_dict(diagnose_surrogate(learned, target)),
    }
    return key, target, analytic, learned, diagnoses, label


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).with_name("config_ablations.yaml"),
        help="YAML config (default: experiments/E02_tunnelvision_demo/config_ablations.yaml)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="p=5 synthetic smoke run, not the ablation figure",
    )
    args = parser.parse_args(argv)

    config = load_yaml(args.config)
    if args.quick:
        _apply_quick(config)

    output_dir = REPO_ROOT / str(config.get("output_dir", "results/E02/ablations"))
    output_dir.mkdir(parents=True, exist_ok=True)
    prior = float(config["prior_inclusion"])
    quench_cfg = dict(config["quench"])
    learned_cfg = dict(config.get("learned", {}))
    chain_cfg = dict(config["chain"])
    noise_levels = [float(x) for x in config["noise_levels"]]
    sigmas = [float(x) for x in config["surrogate_sigmas"]]
    if any(lam < 0.0 or lam > 1.0 for lam in noise_levels):
        raise ValueError("noise_levels must lie in [0, 1]")
    if any(sig < 0.0 for sig in sigmas):
        raise ValueError("surrogate_sigmas must be non-negative")

    meta: dict[str, Any] = {
        "git_commit": git_commit(),
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "versions": package_versions(),
        "config": config,
        "quick": bool(args.quick),
    }
    print(f"E02 ablations: datasets={[_dataset_key(s) for s in config['datasets']]}", flush=True)

    noise_rows: list[dict[str, Any]] = []
    sigma_rows: list[dict[str, Any]] = []
    diagnoses_by_dataset: dict[str, Any] = {}

    for spec in config["datasets"]:
        key, target, analytic, learned, diagnoses, label = _prepare_target(
            spec, prior, learned_cfg
        )
        diagnoses_by_dataset[key] = {"label": label, **diagnoses}
        exact_pips = target.posterior_inclusion_probs_exact()
        pi = target.enumerate_exact()
        print(f"  dataset {key}: {label}, n_vars={target.n_vars}", flush=True)
        for kind in ("analytic", "learned"):
            d = diagnoses[kind]
            print(
                f"    {kind} Spearman={d['spearman']:.3f} "
                f"ground-state rank={d['ground_state_rank']}/{d['n_states']}",
                flush=True,
            )

        ads = _score_ads(target, exact_pips, pi, chain_cfg)
        noise_rows.append(
            {
                **ads,
                "dataset": key,
                "family": "ads",
                "lambda": 0.0,
            }
        )

        inners = {
            "quench-analytic": _quench(analytic, "quench-analytic", quench_cfg),
            "quench-learned": _quench(learned, "quench-learned", quench_cfg),
        }
        for family, inner in inners.items():
            for lam in noise_levels:
                kernel = DepolarizedQuenchKernel(inner, lam)
                print(f"    scoring {kernel.name} on {key}...", flush=True)
                row = _score_kernel(target, kernel, exact_pips, pi, chain_cfg)
                noise_rows.append(
                    {
                        **row,
                        "dataset": key,
                        "family": family,
                        "lambda": lam,
                    }
                )

        for sigma in sigmas:
            corrupted = corrupt_surrogate(learned, sigma, seed=int(learned_cfg.get("seed", 0)))
            diagnosis = _diagnosis_dict(diagnose_surrogate(corrupted, target))
            kernel = _quench(corrupted, f"quench-learned-sigma-{sigma:g}", quench_cfg)
            print(
                f"    scoring {kernel.name} on {key} "
                f"(Spearman={diagnosis['spearman']:.3f})...",
                flush=True,
            )
            row = _score_kernel(target, kernel, exact_pips, pi, chain_cfg)
            sigma_rows.append(
                {
                    **row,
                    **diagnosis,
                    "dataset": key,
                    "sigma": sigma,
                }
            )

    print("  Aer validation...", flush=True)
    aer_rows = _run_aer_validation(config)
    meta["finished_utc"] = datetime.now(timezone.utc).isoformat()
    meta["diagnoses"] = diagnoses_by_dataset

    import pandas as pd

    pd.DataFrame(noise_rows).to_csv(output_dir / "noise_scoreboard.csv", index=False)
    pd.DataFrame(sigma_rows).to_csv(output_dir / "sigma_scoreboard.csv", index=False)
    if aer_rows:
        pd.DataFrame(aer_rows).to_csv(output_dir / "aer_validation.csv", index=False)
    (output_dir / "meta.json").write_text(json.dumps(meta, indent=2, default=str) + "\n")
    (output_dir / "summary.json").write_text(
        json.dumps(
            {"noise": noise_rows, "sigma": sigma_rows, "aer": aer_rows},
            indent=2,
            default=str,
        )
        + "\n"
    )
    _write_noise_figure(output_dir / "gap_vs_lambda.png", noise_rows)
    _write_sigma_figure(output_dir / "gap_vs_sigma.png", sigma_rows)
    if aer_rows:
        _write_aer_figure(output_dir / "aer_validation.png", aer_rows)
    _write_summary(output_dir / "summary.md", config, meta, noise_rows, sigma_rows, aer_rows)
    print(f"wrote {output_dir / 'summary.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
