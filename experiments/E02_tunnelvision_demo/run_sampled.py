"""E02c — sampled tier at p=20 and p=27.

The exact-tier story stopped at p=10: quench never beat add-delete-swap,
and the learned/ADS gap ratio grew with ρ but stayed below 1. This is
the remaining place that story could turn — larger p, sampled
diagnostics only. Spectral gap is illegal here (p > 14).

Quench proposals are the live Trotter statevector, not a tabulated Q
and not dense expm. ``problem_energies`` enumerates 2^p, so p=27 cannot
even construct the kernel (``all_binary_states`` refuses n > 24). That
refusal is a result, not a gap in the runner. A short pilot prices
p=20 before any chain runs; if the estimate exceeds the budget the
quench cell is skipped and written up.

``--quick`` drops to p=6 so the entry point is smoke-testable in seconds.
``--pilot`` prices p=20 quench and exits without running chains.
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
from scipy.stats import spearmanr

from experiments.E02_tunnelvision_demo.chains import (
    run_replicated_chains,
    sampled_chain_seed,
)
from experiments.E02_tunnelvision_demo.run import (
    _KERNEL_COLORS,
    _make_kernels,
    _md_table,
)
from experiments.provenance import REPO_ROOT, git_commit, load_yaml, package_versions

os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / ".mplconfig"))
from tunnelvision.data.loaders import correlated_synthetic
from tunnelvision.engine import MetropolisEngine
from tunnelvision.kernels.base import Kernel
from tunnelvision.surrogate import IsingSurrogate, ising_surrogate_from_data, learned_surrogate
from tunnelvision.targets.spike_slab import SpikeSlabTarget

_QUENCH = {"quench-analytic", "quench-learned"}
# Must match tunnelvision.bits._MAX_ENUMERATE_VARS. Quench construction
# calls problem_energies → all_binary_states over the full basis.
_QUENCH_ENUMERATE_CAP = 24


def _apply_quick(config: dict[str, Any]) -> None:
    """Tiny synthetic so the sampled-tier entry point is seconds, not hours."""
    config["ps"] = [6]
    config["rhos"] = [0.5]
    config["include_quench_analytic_at_p20"] = False
    synthetic = dict(config.get("synthetic", {}))
    synthetic.update({"n": 60, "k_true": 2, "snr": 3.0, "seed": 0})
    config["synthetic"] = synthetic
    quench = dict(config.get("quench", {}))
    quench["n_gamma"] = 3
    quench["n_t"] = 3
    config["quench"] = quench
    learned = dict(config.get("learned", {}))
    learned["n_samples"] = 200
    config["learned"] = learned
    diagnosis = dict(config.get("diagnosis", {}))
    diagnosis["n_samples"] = 200
    config["diagnosis"] = diagnosis
    chain = dict(config.get("chain", {}))
    chain["n_chains"] = 2
    chain["classical"] = {"n_steps": 400, "burn_in": 50}
    chain["quench"] = {"n_steps": 80, "burn_in": 20}
    config["chain"] = chain
    config["output_dir"] = "results/E02_quick/sampled"


def _is_quench(name: str) -> bool:
    return name in _QUENCH or name.startswith("quench")


def _quench_skip_reason(n_vars: int) -> str | None:
    """Dense 2^p energies are the construction cost, not the MH cost."""
    if n_vars > _QUENCH_ENUMERATE_CAP:
        return (
            f"problem_energies enumerates 2^{n_vars} basis energies; "
            f"all_binary_states refuses n > {_QUENCH_ENUMERATE_CAP}"
        )
    return None


def _kernel_names(p: int, config: dict[str, Any]) -> list[str]:
    names = [str(name) for name in config["kernels"]]
    if (
        p == 20
        and bool(config.get("include_quench_analytic_at_p20", False))
        and "quench-analytic" not in names
    ):
        names.append("quench-analytic")
    return names


def _chain_lengths(name: str, chain_cfg: dict[str, Any]) -> tuple[int, int]:
    block = chain_cfg["quench"] if _is_quench(name) else chain_cfg["classical"]
    return int(block["n_steps"]), int(block["burn_in"])


def _random_starts(n_vars: int, seeds: list[int]) -> list[np.ndarray]:
    """Independent random launches so R̂ can see chains in different modes."""
    starts: list[np.ndarray] = []
    for seed in seeds:
        rng = np.random.Generator(np.random.PCG64(int(seed) + 17))
        starts.append(rng.integers(0, 2, size=n_vars, dtype=np.uint8))
    return starts


def _sampled_diagnosis(
    surrogate: IsingSurrogate,
    target: SpikeSlabTarget,
    n_samples: int,
    seed: int,
) -> dict[str, Any]:
    """Spearman(−E, log p) on a seeded sample — full-cube diagnosis is p≤20 only."""
    if n_samples < 3:
        raise ValueError(f"diagnosis n_samples must be at least 3, got {n_samples}")
    rng = np.random.Generator(np.random.PCG64(seed))
    states = rng.integers(0, 2, size=(n_samples, target.n_vars), dtype=np.uint8)
    energy = surrogate.energies(states)
    logp = np.asarray(target.log_prob_batch(states), dtype=np.float64)
    finite = np.isfinite(energy) & np.isfinite(logp)
    if int(finite.sum()) < 3:
        raise ValueError("sampled diagnosis: fewer than 3 finite (energy, logp) pairs")
    corr = spearmanr(-energy[finite], logp[finite])
    return {
        "spearman": float(corr.statistic),
        "n_samples": int(finite.sum()),
        "enumerated": False,
    }


def _price_propose_from_target(kernel: Kernel, n_vars: int, n_proposals: int, seed: int) -> float:
    """Seconds per live proposal, after one warmup draw."""
    if n_proposals < 1:
        raise ValueError(f"n_proposals must be at least 1, got {n_proposals}")
    rng = np.random.Generator(np.random.PCG64(seed))
    x = np.zeros(n_vars, dtype=np.uint8)
    kernel.propose(x, rng)
    t0 = time.perf_counter()
    for _ in range(n_proposals):
        y, _, _ = kernel.propose(x, rng)
        x = y
    return (time.perf_counter() - t0) / float(n_proposals)


def _max_pip_disagreement(estimates: dict[str, list[float]]) -> list[list[str]]:
    names = list(estimates)
    rows: list[list[str]] = []
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            delta = np.max(
                np.abs(np.asarray(estimates[left]) - np.asarray(estimates[right]))
            )
            rows.append([left, right, f"{float(delta):.3f}"])
    return rows


def _write_scoreboard_figure(path: Path, rows: list[dict[str, Any]]) -> None:
    """ESS/step and ESS/sec by (p, ρ). A bigger ESS/step that costs 100× is not a win."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cells = sorted({(int(row["p"]), float(row["rho"])) for row in rows})
    if not cells:
        return
    n_cells = len(cells)
    fig, axes = plt.subplots(n_cells, 2, figsize=(9.6, 3.2 * n_cells), squeeze=False)
    for row_idx, (p, rho) in enumerate(cells):
        cell = [row for row in rows if int(row["p"]) == p and float(row["rho"]) == rho]
        names = [str(row["kernel"]) for row in cell]
        colors = [_KERNEL_COLORS.get(name, "#2c3e50") for name in names]
        x = np.arange(len(names))
        for col, (key, ylabel, log_y) in enumerate(
            (
                ("ess_size_per_step", r"ESS / step ($|\gamma|$)", False),
                ("ess_size_per_sec", r"ESS / sec ($|\gamma|$)", True),
            )
        ):
            ax = axes[row_idx][col]
            values = np.array([float(row[key]) for row in cell], dtype=np.float64)
            ax.bar(x, values, color=colors, width=0.72)
            ax.set_xticks(x)
            ax.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
            ax.set_ylabel(ylabel)
            ax.set_title(f"p={p}, ρ={rho:g}")
            if log_y:
                positive = values[values > 0.0]
                ax.set_yscale("log")
                if positive.size:
                    ax.set_ylim(positive.min() * 0.5, positive.max() * 2.0)
            ax.grid(True, axis="y", alpha=0.3)
            ads = next((row for row in cell if row["kernel"] == "add-delete-swap"), None)
            if ads is not None and float(ads[key]) > 0.0:
                ax.axhline(
                    float(ads[key]),
                    color=_KERNEL_COLORS["add-delete-swap"],
                    ls="--",
                    lw=0.8,
                )
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _story_read(
    rows: list[dict[str, Any]],
    skips: list[dict[str, Any]],
) -> list[str]:
    lines: list[str] = []
    if skips:
        for skip in skips:
            lines.append(
                f"- Skipped `{skip['kernel']}` at p={skip['p']}, ρ={skip['rho']:g}: "
                f"{skip['reason']}."
            )
    cells = sorted({(int(row["p"]), float(row["rho"])) for row in rows})
    for p, rho in cells:
        cell = [row for row in rows if int(row["p"]) == p and float(row["rho"]) == rho]
        ads = next((row for row in cell if row["kernel"] == "add-delete-swap"), None)
        if ads is None:
            lines.append(f"- p={p}, ρ={rho:g}: add-delete-swap is missing; no baseline.")
            continue
        ads_step = float(ads["ess_size_per_step"])
        ads_sec = float(ads["ess_size_per_sec"])
        for name in ("quench-analytic", "quench-learned", "single-flip"):
            match = next((row for row in cell if row["kernel"] == name), None)
            if match is None:
                continue
            step_ratio = (
                float(match["ess_size_per_step"]) / ads_step if ads_step > 0.0 else float("inf")
            )
            sec_ratio = (
                float(match["ess_size_per_sec"]) / ads_sec if ads_sec > 0.0 else float("inf")
            )
            step_verb = (
                "beats" if step_ratio > 1.05 else ("matches" if step_ratio > 0.95 else "loses to")
            )
            sec_verb = (
                "beats" if sec_ratio > 1.05 else ("matches" if sec_ratio > 0.95 else "loses to")
            )
            lines.append(
                f"- p={p}, ρ={rho:g}: `{name}` {step_verb} ADS on ESS/step "
                f"({step_ratio:.2f}×) and {sec_verb} ADS on ESS/sec ({sec_ratio:.2f}×)."
            )
    stuck = [row for row in rows if float(row["rhat_size"]) > 1.1]
    if stuck:
        worst = max(stuck, key=lambda row: float(row["rhat_size"]))
        lines.append(
            f"- Split-R̂ flagged stuck chains (worst: `{worst['kernel']}` at "
            f"p={worst['p']}, ρ={worst['rho']:g}, R̂={worst['rhat_size']:.2f}). "
            "Treat those ESS numbers as lower bounds."
        )
    else:
        lines.append("- Split-R̂ < 1.1 on |γ| for every completed cell; no stuck-chain warning.")
    return lines


def _write_summary(
    path: Path,
    config: dict[str, Any],
    meta: dict[str, Any],
    diagnoses: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    skips: list[dict[str, Any]],
    pilots: list[dict[str, Any]],
) -> None:
    lines = [
        "# E02c — sampled tier (p=20 and p=27)",
        "",
        "ESS, wall-clock, and split-R̂ on equicorrelated spike-and-slab",
        "posteriors above the exact-tier cutoff. Accept/reject uses the",
        "exact g-prior; the surrogate only shapes proposals. There is no",
        "spectral gap here — p > 14.",
        "",
        "## Provenance",
        "",
        f"- git commit: `{meta['git_commit']}`",
        f"- started (UTC): {meta['started_utc']}",
        f"- finished (UTC): {meta['finished_utc']}",
        f"- package versions: {', '.join(f'{k}={v}' for k, v in meta['versions'].items())}",
        f"- design: {config['synthetic'].get('structure', 'equicorrelated')} "
        f"n={config['synthetic']['n']} k_true={config['synthetic']['k_true']} "
        f"snr={config['synthetic']['snr']}",
        f"- p grid: {', '.join(str(p) for p in config['ps'])}",
        f"- ρ grid: {', '.join(f'{float(rho):g}' for rho in config['rhos'])}",
        f"- quench: {config['quench'].get('evolution', 'trotter')} "
        f"{config['quench'].get('backend', 'statevector')}",
        f"- chains: {config['chain']['n_chains']} × "
        f"classical {config['chain']['classical']['n_steps']}/"
        f"{config['chain']['classical']['burn_in']}, "
        f"quench {config['chain']['quench']['n_steps']}/"
        f"{config['chain']['quench']['burn_in']}",
        "",
        "## Pilot (quench wall-clock)",
        "",
        "A live Trotter proposal at p=20 allocates a 2^20 statevector.",
        "p=27 cannot construct the kernel: the problem Hamiltonian's",
        "diagonal is enumerated over the full basis.",
        "",
    ]
    if pilots:
        pilot_rows = [
            [
                str(item["p"]),
                f"{float(item['rho']):g}",
                item["kernel"],
                (
                    f"{float(item['seconds_per_proposal']):.3f}"
                    if item.get("seconds_per_proposal") is not None
                    else "—"
                ),
                (
                    f"{float(item['estimated_wall_seconds']):.0f}"
                    if item.get("estimated_wall_seconds") is not None
                    else "—"
                ),
                "yes" if item.get("run") else "no",
                str(item.get("reason") or ""),
            ]
            for item in pilots
        ]
        lines.append(
            _md_table(
                ["p", "ρ", "kernel", "s/propose", "est. wall (s)", "ran", "note"],
                pilot_rows,
            )
        )
    else:
        lines.append("No quench cells were priced.")
    lines.extend(["", "## Surrogate diagnosis (sampled Spearman)", ""])
    diag_rows = [
        [
            str(item["p"]),
            f"{float(item['rho']):g}",
            item["kind"],
            f"{item['diagnosis']['spearman']:.3f}",
            str(item["diagnosis"]["n_samples"]),
        ]
        for item in diagnoses
    ]
    lines.append(
        _md_table(["p", "ρ", "surrogate", "Spearman(−E, log p)", "n"], diag_rows)
    )

    p20 = [row for row in rows if int(row["p"]) == 20]
    p27 = [row for row in rows if int(row["p"]) == 27]
    if p20:
        lines.extend(
            [
                "",
                "## p=20 scoreboard",
                "",
                "PIP error is against enumerated truth (p=20 is the last",
                "enumerable size). ESS/sec is live-kernel wall-clock, not",
                "tabulated-Q sampling — this is the column that prices the",
                "simulation.",
                "",
            ]
        )
        score_rows = [
            [
                f"{row['rho']:g}",
                row["kernel"],
                f"{row['acceptance_rate']:.3f}",
                f"{row['pip_error_pooled']:.3f}" if row["pip_error_pooled"] is not None else "—",
                f"{row['ess_size_per_step']:.3f}",
                f"{row['ess_size_per_sec']:.2f}",
                f"{row['rhat_size']:.3f}",
                f"{row['wall_seconds']:.1f}",
            ]
            for row in p20
        ]
        lines.append(
            _md_table(
                [
                    "ρ",
                    "kernel",
                    "accept",
                    "PIP err",
                    "ESS/step (|γ|)",
                    "ESS/sec (|γ|)",
                    "R̂ (|γ|)",
                    "wall (s)",
                ],
                score_rows,
            )
        )
    if p27:
        lines.extend(
            [
                "",
                "## p=27 scoreboard",
                "",
                "No enumerated PIPs at p=27. Cross-kernel PIP agreement and",
                "R̂ are the honesty checks. Quench is expected to be absent.",
                "",
            ]
        )
        score_rows = [
            [
                f"{row['rho']:g}",
                row["kernel"],
                f"{row['acceptance_rate']:.3f}",
                f"{row['ess_size_per_step']:.3f}",
                f"{row['ess_size_per_sec']:.2f}",
                f"{row['rhat_size']:.3f}",
                f"{row['rhat_pip_max']:.3f}",
                f"{row['wall_seconds']:.1f}",
            ]
            for row in p27
        ]
        lines.append(
            _md_table(
                [
                    "ρ",
                    "kernel",
                    "accept",
                    "ESS/step (|γ|)",
                    "ESS/sec (|γ|)",
                    "R̂ (|γ|)",
                    "max R̂ (PIP)",
                    "wall (s)",
                ],
                score_rows,
            )
        )
        grouped: dict[float, dict[str, list[float]]] = {}
        for row in p27:
            grouped.setdefault(float(row["rho"]), {})[str(row["kernel"])] = row["pip_estimate"]
        lines.extend(["", "### Cross-kernel PIP agreement (max-abs)", ""])
        for rho, mapping in grouped.items():
            pairs = _max_pip_disagreement(mapping)
            if not pairs:
                continue
            lines.append(f"ρ = {rho:g}")
            lines.append("")
            lines.append(_md_table(["left", "right", "max |ΔPIP|"], pairs))
            lines.append("")

    lines.extend(
        [
            "",
            "![scoreboard](scoreboard.png)",
            "",
            "## Read",
            "",
            *_story_read(rows, skips),
            "",
            "## Notes",
            "",
            "- Add-delete-swap is the baseline. Beating uniform proves nothing.",
            "- Headline comparison is ESS/step *and* ESS/sec. The quench pays",
            "  a 2^p statevector per proposal; hiding that is the failure mode",
            "  this repo exists to avoid.",
            "- p=27 quench is a construction refusal, not a missing measurement.",
            "",
        ]
    )
    if meta.get("quick"):
        lines.extend(
            [
                "This file was written by ``--quick`` (p=6). It is not the",
                "E02c result and lives under `results/E02_quick/` (gitignored).",
                "",
            ]
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def _build_instance(
    p: int,
    rho: float,
    config: dict[str, Any],
) -> tuple[SpikeSlabTarget, IsingSurrogate, IsingSurrogate]:
    spec = config["synthetic"]
    X, y, _support = correlated_synthetic(
        n=int(spec["n"]),
        p=p,
        rho=rho,
        k_true=int(spec["k_true"]),
        snr=float(spec["snr"]),
        seed=int(spec["seed"]),
        structure=str(spec.get("structure", "equicorrelated")),
    )
    prior = float(config["prior_inclusion"])
    target = SpikeSlabTarget(X, y, prior_inclusion=prior)
    analytic = ising_surrogate_from_data(X, y, prior_inclusion=prior)
    learned_cfg = config.get("learned", {})
    learned = learned_surrogate(
        target,
        n_samples=int(learned_cfg.get("n_samples", 4000)),
        seed=int(learned_cfg.get("seed", 0)),
        ridge=float(learned_cfg.get("ridge", 1e-3)),
    )
    return target, analytic, learned


def _run_pilot_only(config: dict[str, Any]) -> int:
    """Price p=20 quench and report the p=27 construction refusal."""
    p = 20
    rho = float(config["rhos"][-1])
    print(f"E02c pilot: p={p}, ρ={rho:g}", flush=True)
    target, analytic, learned = _build_instance(p, rho, config)
    kernels = _make_kernels(["quench-learned"], analytic, learned, dict(config["quench"]))
    kernel = kernels["quench-learned"]
    n_prop = int(config["pilot"]["n_proposals"])
    print(f"  constructing quench-learned (2^{p} energies)...", flush=True)
    t_build = time.perf_counter()
    seconds = _price_propose_from_target(kernel, target.n_vars, n_prop, seed=0)
    build_and_price = time.perf_counter() - t_build
    n_steps, _burn = _chain_lengths("quench-learned", dict(config["chain"]))
    n_chains = int(config["chain"]["n_chains"])
    estimate = seconds * n_steps * n_chains
    print(
        f"  quench-learned: {seconds:.3f} s/propose, "
        f"est. {estimate:.0f}s for {n_chains}×{n_steps} "
        f"(pricing wall {build_and_price:.1f}s)",
        flush=True,
    )
    skip_27 = _quench_skip_reason(27)
    print(f"  p=27 quench: {skip_27}", flush=True)
    output_dir = REPO_ROOT / str(config.get("output_dir", "results/E02/sampled"))
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "p20": {
            "seconds_per_proposal": seconds,
            "estimated_wall_seconds": estimate,
            "n_steps": n_steps,
            "n_chains": n_chains,
            "n_proposals_timed": n_prop,
        },
        "p27": {"reason": skip_27},
    }
    (output_dir / "pilot.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {output_dir / 'pilot.json'}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).with_name("config_sampled.yaml"),
        help="YAML config (default: experiments/E02_tunnelvision_demo/config_sampled.yaml)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="p=6 smoke run, not the sampled-tier figure",
    )
    parser.add_argument(
        "--pilot",
        action="store_true",
        help="price p=20 quench and exit; do not run chains",
    )
    args = parser.parse_args(argv)

    config = load_yaml(args.config)
    if args.quick:
        _apply_quick(config)
    if args.pilot:
        return _run_pilot_only(config)

    ps = [int(p) for p in config["ps"]]
    rhos = [float(rho) for rho in config["rhos"]]
    output_dir = REPO_ROOT / str(config.get("output_dir", "results/E02/sampled"))
    output_dir.mkdir(parents=True, exist_ok=True)

    meta: dict[str, Any] = {
        "git_commit": git_commit(),
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "versions": package_versions(),
        "config": config,
        "quick": bool(args.quick),
    }
    print(f"E02c: ps={ps}, rhos={rhos}, kernels={list(config['kernels'])}", flush=True)

    rows: list[dict[str, Any]] = []
    skips: list[dict[str, Any]] = []
    pilots: list[dict[str, Any]] = []
    diagnoses: list[dict[str, Any]] = []
    kernel_ids = {name: index for index, name in enumerate(_kernel_names(20, config))}

    for p in ps:
        for rho in rhos:
            print(f"  instance p={p} ρ={rho:g}", flush=True)
            target, analytic, learned = _build_instance(p, rho, config)
            diag_cfg = config.get("diagnosis", {})
            for kind, surrogate in (("analytic", analytic), ("learned", learned)):
                diagnosis = _sampled_diagnosis(
                    surrogate,
                    target,
                    n_samples=int(diag_cfg.get("n_samples", 4000)),
                    seed=int(diag_cfg.get("seed", 1)),
                )
                diagnoses.append(
                    {"p": p, "rho": rho, "kind": kind, "diagnosis": diagnosis}
                )
                print(
                    f"    {kind} Spearman={diagnosis['spearman']:.3f} "
                    f"(n={diagnosis['n_samples']})",
                    flush=True,
                )

            exact_pips: np.ndarray | None = None
            if p <= 20:
                print(f"    enumerating exact PIPs (2^{p})...", flush=True)
                t_pip = time.perf_counter()
                exact_pips = target.posterior_inclusion_probs_exact()
                print(f"    exact PIPs in {time.perf_counter() - t_pip:.1f}s", flush=True)

            names = _kernel_names(p, config)
            buildable: list[str] = []
            for name in names:
                skip = _quench_skip_reason(p) if _is_quench(name) else None
                if skip is not None:
                    item = {
                        "p": p,
                        "rho": rho,
                        "kernel": name,
                        "reason": skip,
                        "run": False,
                        "seconds_per_proposal": None,
                        "estimated_wall_seconds": None,
                    }
                    skips.append(item)
                    pilots.append(item)
                    print(f"    skip {name}: {skip}", flush=True)
                    continue
                buildable.append(name)
            # Construct only kernels we will actually propose with. Quench
            # __init__ enumerates 2^p energies; doing that at p=27 OOMs.
            kernels = _make_kernels(buildable, analytic, learned, dict(config["quench"]))
            chain_cfg = dict(config["chain"])
            n_chains = int(chain_cfg["n_chains"])
            budget = float(config["pilot"]["max_quench_wall_seconds"])
            n_prop = int(config["pilot"]["n_proposals"])

            for name in buildable:
                kernel = kernels[name]
                n_steps, burn_in = _chain_lengths(name, chain_cfg)
                if _is_quench(name):
                    print(f"    pricing {name}...", flush=True)
                    seconds = _price_propose_from_target(
                        kernel, target.n_vars, n_prop, seed=p + int(round(rho * 100))
                    )
                    estimate = seconds * n_steps * n_chains
                    run_it = estimate <= budget
                    pilots.append(
                        {
                            "p": p,
                            "rho": rho,
                            "kernel": name,
                            "seconds_per_proposal": seconds,
                            "estimated_wall_seconds": estimate,
                            "run": run_it,
                            "reason": None if run_it else (
                                f"est. {estimate:.0f}s > budget {budget:.0f}s"
                            ),
                        }
                    )
                    print(
                        f"    {name}: {seconds:.3f} s/propose, est. {estimate:.0f}s",
                        flush=True,
                    )
                    if not run_it:
                        skips.append(pilots[-1])
                        print(f"    skip {name}: over budget", flush=True)
                        continue

                kernel_id = kernel_ids.setdefault(name, len(kernel_ids))
                seeds = [
                    sampled_chain_seed(
                        int(chain_cfg["seed"]),
                        p=p,
                        rho=rho,
                        kernel_id=kernel_id,
                        chain_index=index,
                    )
                    for index in range(n_chains)
                ]
                print(
                    f"    {name}: {n_chains} chains × {n_steps} steps...",
                    flush=True,
                )
                scored = run_replicated_chains(
                    MetropolisEngine(target, kernel),
                    n_steps=n_steps,
                    burn_in=burn_in,
                    seeds=seeds,
                    x0=_random_starts(target.n_vars, seeds),
                    exact_pips=exact_pips,
                )
                row = {"p": p, "rho": rho, "kernel": name, **scored}
                rows.append(row)
                pip_txt = (
                    f"  PIP err={row['pip_error_pooled']:.3f}"
                    if row["pip_error_pooled"] is not None
                    else ""
                )
                print(
                    f"    {name:20s}  R̂={row['rhat_size']:.3f}  "
                    f"ESS/step={row['ess_size_per_step']:.3f}  "
                    f"ESS/sec={row['ess_size_per_sec']:.2f}{pip_txt}",
                    flush=True,
                )

    meta["finished_utc"] = datetime.now(timezone.utc).isoformat()
    import pandas as pd

    serializable_rows = []
    for row in rows:
        dumped = dict(row)
        serializable_rows.append(dumped)
    pd.DataFrame(serializable_rows).to_csv(output_dir / "scoreboard.csv", index=False)
    (output_dir / "meta.json").write_text(json.dumps(meta, indent=2, default=str) + "\n")
    (output_dir / "summary.json").write_text(
        json.dumps(
            {"diagnoses": diagnoses, "rows": rows, "skips": skips, "pilots": pilots},
            indent=2,
        )
        + "\n"
    )
    _write_scoreboard_figure(output_dir / "scoreboard.png", rows)
    _write_summary(output_dir / "summary.md", config, meta, diagnoses, rows, skips, pilots)
    print(f"wrote {output_dir / 'summary.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
