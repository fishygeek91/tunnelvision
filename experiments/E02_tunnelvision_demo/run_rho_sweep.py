"""E02b — ρ-sweep, exact tier (the story curve).

Question: does the quench/ADS spectral-gap ratio grow with predictor
correlation ρ? Correlation makes predictors interchangeable and splits
the posterior into competing modes; tunneling is supposed to pay there.

Exact tier only (p=10). The sampled p∈{20,27} slice is a later session —
quench proposal matrices at 2^20 are a wall-clock decision, not a
correctness one. Four independent chains and split-R̂ are here anyway:
at high ρ a single stuck chain is the subtlest way this experiment lies.

``--quick`` drops to p=5 and two ρ values so the entry point is
smoke-testable in seconds.
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

from experiments.E02_tunnelvision_demo.chains import rho_chain_seed, run_replicated_chains
from experiments.E02_tunnelvision_demo.run import (
    _KERNEL_COLORS,
    TabulatedKernel,
    _diagnosis_dict,
    _make_kernels,
    _md_table,
)
from experiments.provenance import REPO_ROOT, git_commit, load_yaml, package_versions

os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / ".mplconfig"))
from tunnelvision.data.loaders import correlated_synthetic
from tunnelvision.diagnostics import spectral_gap
from tunnelvision.engine import MetropolisEngine
from tunnelvision.surrogate import (
    diagnose_surrogate,
    ising_surrogate_from_data,
    learned_surrogate,
)
from tunnelvision.targets.spike_slab import SpikeSlabTarget


def _apply_quick(config: dict[str, Any]) -> None:
    """Tiny synthetic so the ρ-sweep entry point is smoke-testable in seconds."""
    config["rhos"] = [0.0, 0.5]
    synthetic = dict(config.get("synthetic", {}))
    synthetic.update({"n": 60, "p": 5, "k_true": 2, "snr": 3.0, "seed": 0})
    config["synthetic"] = synthetic
    quench = dict(config.get("quench", {}))
    quench["n_gamma"] = 3
    quench["n_t"] = 3
    config["quench"] = quench
    chain = dict(config.get("chain", {}))
    chain.update({"n_steps": 400, "burn_in": 50, "n_chains": 2})
    config["chain"] = chain
    config["output_dir"] = "results/E02_quick/rho_sweep"


def _score_kernel_multi(
    target: SpikeSlabTarget,
    kernel: Any,
    exact_pips: np.ndarray,
    pi: np.ndarray,
    chain_cfg: dict[str, Any],
    rho: float,
) -> dict[str, Any]:
    engine = MetropolisEngine(target, kernel)
    print(f"    {kernel.name}: transition matrix...", flush=True)
    t_gap = time.perf_counter()
    gap = float(spectral_gap(engine.transition_matrix(), pi))
    gap_seconds = time.perf_counter() - t_gap
    tabulated = TabulatedKernel(kernel.name, kernel.proposal_matrix(target.n_vars))
    chain_engine = MetropolisEngine(target, tabulated)

    n_steps = int(chain_cfg["n_steps"])
    burn_in = int(chain_cfg["burn_in"])
    n_chains = int(chain_cfg["n_chains"])
    seeds = [rho_chain_seed(int(chain_cfg["seed"]), rho, index) for index in range(n_chains)]
    scored = run_replicated_chains(
        chain_engine,
        n_steps=n_steps,
        burn_in=burn_in,
        seeds=seeds,
        x0=np.zeros(target.n_vars, dtype=np.uint8),
        exact_pips=exact_pips,
    )
    return {
        "rho": rho,
        "kernel": kernel.name,
        "spectral_gap": gap,
        "gap_seconds": gap_seconds,
        **scored,
    }


def _write_story_figure(path: Path, rows: list[dict[str, Any]]) -> None:
    """Gap vs ρ (log y) and quench/ADS gap ratio — the E02b story curve."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rhos = sorted({float(row["rho"]) for row in rows})
    kernels = list(dict.fromkeys(str(row["kernel"]) for row in rows))
    by_kernel: dict[str, list[float]] = {name: [] for name in kernels}
    for name in kernels:
        lookup = {
            float(row["rho"]): float(row["spectral_gap"])
            for row in rows
            if row["kernel"] == name
        }
        by_kernel[name] = [lookup[rho] for rho in rhos]

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.8))
    ax = axes[0]
    for name in kernels:
        ax.semilogy(
            rhos,
            by_kernel[name],
            "o-",
            color=_KERNEL_COLORS.get(name, "#2c3e50"),
            label=name,
        )
    ax.set_xlabel(r"predictor correlation $\rho$")
    ax.set_ylabel(r"spectral gap $\delta$")
    ax.set_title("exact-tier gap vs ρ")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(frameon=False, fontsize=8)

    ads = by_kernel.get("add-delete-swap")
    ax = axes[1]
    if ads is not None:
        ads_arr = np.asarray(ads, dtype=np.float64)
        for name in ("quench-analytic", "quench-learned"):
            if name not in by_kernel:
                continue
            ratio = np.asarray(by_kernel[name], dtype=np.float64) / np.clip(ads_arr, 1e-16, None)
            ax.plot(
                rhos,
                ratio,
                "o-",
                color=_KERNEL_COLORS.get(name, "#2c3e50"),
                label=f"{name} / ADS",
            )
        ax.axhline(1.0, color=_KERNEL_COLORS["add-delete-swap"], ls="--", lw=0.8)
    ax.set_xlabel(r"predictor correlation $\rho$")
    ax.set_ylabel("gap ratio vs ADS")
    ax.set_title("does advantage grow with ρ?")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _story_read(rows: list[dict[str, Any]]) -> list[str]:
    """Does the quench/ADS gap ratio grow with ρ? Either answer is a result."""
    rhos = sorted({float(row["rho"]) for row in rows})
    ads = {
        float(row["rho"]): float(row["spectral_gap"])
        for row in rows
        if row["kernel"] == "add-delete-swap"
    }
    if not ads:
        return ["- Add-delete-swap is missing; no story-curve comparison."]

    lines: list[str] = []
    for name in ("quench-analytic", "quench-learned"):
        gaps = {
            float(row["rho"]): float(row["spectral_gap"])
            for row in rows
            if row["kernel"] == name
        }
        if not gaps:
            continue
        ratios = [gaps[rho] / ads[rho] if ads[rho] > 0.0 else float("inf") for rho in rhos]
        low, high = ratios[0], ratios[-1]
        grew = high > low * 1.25
        shrink = high < low * 0.80
        trend = "grows" if grew else ("shrinks" if shrink else "is roughly flat")
        beats = sum(1 for ratio in ratios if ratio > 1.05)
        lines.append(
            f"- `{name}` / ADS gap ratio {trend} in ρ "
            f"({ratios[0]:.2f}× at ρ={rhos[0]:g} → {ratios[-1]:.2f}× at ρ={rhos[-1]:g}); "
            f"beats ADS at {beats}/{len(rhos)} correlations."
        )
    stuck = [
        row
        for row in rows
        if float(row["rhat_size"]) > 1.1
    ]
    if stuck:
        worst = max(stuck, key=lambda row: float(row["rhat_size"]))
        lines.append(
            f"- Split-R̂ flagged stuck chains (worst: `{worst['kernel']}` at "
            f"ρ={worst['rho']:g}, R̂={worst['rhat_size']:.2f}). "
            "Treat those ESS numbers as lower bounds."
        )
    else:
        lines.append("- Split-R̂ < 1.1 on |γ| for every kernel and ρ; no stuck-chain warning.")
    return lines


def _write_summary(
    path: Path,
    config: dict[str, Any],
    meta: dict[str, Any],
    diagnoses: list[dict[str, Any]],
    rows: list[dict[str, Any]],
) -> None:
    rhos = sorted({float(row["rho"]) for row in rows})
    lines = [
        "# E02b — ρ-sweep, exact tier",
        "",
        "Spectral gap vs predictor correlation on equicorrelated spike-and-slab",
        "posteriors. Accept/reject uses the exact g-prior; the surrogate only",
        "shapes proposals. The story question: does quench/ADS advantage grow with ρ?",
        "",
        "## Provenance",
        "",
        f"- git commit: `{meta['git_commit']}`",
        f"- started (UTC): {meta['started_utc']}",
        f"- finished (UTC): {meta['finished_utc']}",
        f"- package versions: {', '.join(f'{k}={v}' for k, v in meta['versions'].items())}",
        f"- design: {config['synthetic'].get('structure', 'equicorrelated')} "
        f"n={config['synthetic']['n']} p={config['synthetic']['p']} "
        f"k_true={config['synthetic']['k_true']} snr={config['synthetic']['snr']}",
        f"- ρ grid: {', '.join(f'{rho:g}' for rho in rhos)}",
        f"- quench: {config['quench'].get('evolution', 'exact')}, "
        f"grid {config['quench']['n_gamma']}×{config['quench']['n_t']}",
        f"- chains: {config['chain']['n_chains']} × {config['chain']['n_steps']} steps, "
        f"burn-in {config['chain']['burn_in']}",
        "",
        "## Surrogate diagnosis",
        "",
    ]
    diag_rows = []
    for item in diagnoses:
        d = item["diagnosis"]
        diag_rows.append(
            [
                f"{item['rho']:g}",
                item["kind"],
                f"{d['spearman']:.3f}",
                str(d["ground_state_rank"]),
                f"{d['top_log_prob'] - d['ground_state_log_prob']:.2f}",
            ]
        )
    lines.append(
        _md_table(
            ["ρ", "surrogate", "Spearman(−E, log p)", "ground-state rank", "Δlogp vs MAP"],
            diag_rows,
        )
    )
    lines.extend(
        [
            "",
            "## Exact-tier scoreboard",
            "",
            "Gap is exact. ESS and R̂ are from independent chains that sample",
            "the same Q the gap used (tabulated, not a fresh expm per propose).",
            "ESS/sec is therefore Q-sampling cost, not unitary assembly — E02a",
            "is the run that prices the simulation. R̂ is split-R̂ on |γ|;",
            "values ≫ 1 mean at least one chain stuck.",
            "",
        ]
    )
    score_rows = []
    for row in rows:
        score_rows.append(
            [
                f"{row['rho']:g}",
                row["kernel"],
                f"{row['spectral_gap']:.4g}",
                f"{row['acceptance_rate']:.3f}",
                f"{row['pip_error']:.3f}",
                f"{row['ess_size_per_step']:.3f}",
                f"{row['ess_size_per_sec']:.2f}",
                f"{row['rhat_size']:.3f}",
            ]
        )
    lines.append(
        _md_table(
            [
                "ρ",
                "kernel",
                "gap",
                "accept",
                "PIP err",
                "ESS/step (|γ|)",
                "ESS/sec (|γ|)",
                "R̂ (|γ|)",
            ],
            score_rows,
        )
    )

    ratio_rows = []
    ads = {
        float(row["rho"]): float(row["spectral_gap"])
        for row in rows
        if row["kernel"] == "add-delete-swap"
    }
    for rho in rhos:
        cells = [f"{rho:g}"]
        for name in ("quench-analytic", "quench-learned"):
            match = next(
                (row for row in rows if row["kernel"] == name and float(row["rho"]) == rho),
                None,
            )
            if match is None or ads.get(rho, 0.0) <= 0.0:
                cells.append("—")
            else:
                cells.append(f"{float(match['spectral_gap']) / ads[rho]:.2f}×")
        ratio_rows.append(cells)
    lines.extend(
        [
            "",
            "## Gap ratio vs add-delete-swap",
            "",
            _md_table(["ρ", "quench-analytic", "quench-learned"], ratio_rows),
            "",
            "![gap vs rho](gap_vs_rho.png)",
            "",
            "## Read",
            "",
            *_story_read(rows),
            "",
            "## Notes",
            "",
            "- Add-delete-swap is the baseline. Beating uniform proves nothing.",
            "- Sampled tier at p=20–27 is not in this run.",
            "",
        ]
    )
    if meta.get("quick"):
        lines.extend(
            [
                "This file was written by ``--quick`` (p=5, two ρ values). It is not",
                "the E02b gate and lives under `results/E02_quick/` (gitignored).",
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
        default=Path(__file__).with_name("config_rho.yaml"),
        help="YAML config (default: experiments/E02_tunnelvision_demo/config_rho.yaml)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="p=5, two ρ values; not the story-curve figure",
    )
    args = parser.parse_args(argv)

    config = load_yaml(args.config)
    if args.quick:
        _apply_quick(config)

    rhos = [float(rho) for rho in config["rhos"]]
    spec = config["synthetic"]
    prior = float(config["prior_inclusion"])
    learned_cfg = config.get("learned", {})
    output_dir = REPO_ROOT / str(config.get("output_dir", "results/E02/rho_sweep"))
    output_dir.mkdir(parents=True, exist_ok=True)

    meta: dict[str, Any] = {
        "git_commit": git_commit(),
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "versions": package_versions(),
        "config": config,
        "quick": bool(args.quick),
    }
    print(
        f"E02b: rhos={rhos}, p={spec['p']}, kernels={list(config['kernels'])}",
        flush=True,
    )

    rows: list[dict[str, Any]] = []
    diagnoses: list[dict[str, Any]] = []
    for rho in rhos:
        X, y, _support = correlated_synthetic(
            n=int(spec["n"]),
            p=int(spec["p"]),
            rho=rho,
            k_true=int(spec["k_true"]),
            snr=float(spec["snr"]),
            seed=int(spec["seed"]),
            structure=str(spec.get("structure", "equicorrelated")),
        )
        target = SpikeSlabTarget(X, y, prior_inclusion=prior)
        analytic = ising_surrogate_from_data(X, y, prior_inclusion=prior)
        learned = learned_surrogate(
            target,
            n_samples=int(learned_cfg.get("n_samples", 2000)),
            seed=int(learned_cfg.get("seed", 0)),
            ridge=float(learned_cfg.get("ridge", 1e-3)),
        )
        analytic_d = _diagnosis_dict(diagnose_surrogate(analytic, target))
        learned_d = _diagnosis_dict(diagnose_surrogate(learned, target))
        diagnoses.append({"rho": rho, "kind": "analytic", "diagnosis": analytic_d})
        diagnoses.append({"rho": rho, "kind": "learned", "diagnosis": learned_d})
        print(
            f"  ρ={rho:g}: analytic Spearman={analytic_d['spearman']:.3f} "
            f"rank={analytic_d['ground_state_rank']}; "
            f"learned Spearman={learned_d['spearman']:.3f} "
            f"rank={learned_d['ground_state_rank']}",
            flush=True,
        )
        kernels = _make_kernels(list(config["kernels"]), analytic, learned, dict(config["quench"]))
        exact_pips = target.posterior_inclusion_probs_exact()
        pi = target.enumerate_exact()
        for kernel in kernels.values():
            row = _score_kernel_multi(
                target, kernel, exact_pips, pi, dict(config["chain"]), rho
            )
            rows.append(row)
            print(
                f"    {kernel.name:20s}  gap={row['spectral_gap']:.4g}  "
                f"R̂={row['rhat_size']:.3f}  ESS/sec={row['ess_size_per_sec']:.2f}",
                flush=True,
            )

    meta["finished_utc"] = datetime.now(timezone.utc).isoformat()
    import pandas as pd

    pd.DataFrame(rows).to_csv(output_dir / "scoreboard.csv", index=False)
    (output_dir / "meta.json").write_text(json.dumps(meta, indent=2, default=str) + "\n")
    (output_dir / "summary.json").write_text(
        json.dumps({"diagnoses": diagnoses, "rows": rows}, indent=2) + "\n"
    )
    _write_story_figure(output_dir / "gap_vs_rho.png", rows)
    _write_summary(output_dir / "summary.md", config, meta, diagnoses, rows)
    print(f"wrote {output_dir / 'summary.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
