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
from tunnelvision.bits import as_binary_vector, state_to_index
from tunnelvision.data.loaders import correlated_synthetic, load_diabetes
from tunnelvision.diagnostics import ess, pip_error, spectral_gap
from tunnelvision.engine import MetropolisEngine
from tunnelvision.kernels.base import Kernel
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


def _unpack_index(index: int, n_vars: int) -> np.ndarray:
    bits = np.empty(n_vars, dtype=np.uint8)
    for bit in range(n_vars):
        bits[bit] = (index >> bit) & 1
    return bits


class TabulatedKernel(Kernel):
    """Propose from a fixed Q so ESS chains match the exact-tier gap.

    The scoreboard kernel is the transition matrix. A quench ``propose``
    that re-runs dense expm at every step is a different, much slower
    realization of a nearby kernel — and at p=10 it turns a 4000-step
    chain into an hour. Sampling the Q already built for the gap keeps
    ESS/step honest to δ. ESS/sec is then Q-sampling + MH, not unitary
    assembly; ``gap_seconds`` is the number that prices the simulation.
    """

    def __init__(self, name: str, proposal: np.ndarray) -> None:
        q = np.asarray(proposal, dtype=np.float64)
        if q.ndim != 2 or q.shape[0] != q.shape[1]:
            raise ValueError("proposal matrix must be square")
        n_states = int(q.shape[0])
        n_vars = int(np.log2(n_states))
        if (1 << n_vars) != n_states:
            raise ValueError("proposal matrix size must be a power of two")
        np.clip(q, 0.0, None, out=q)
        row_sums = q.sum(axis=1, keepdims=True)
        if np.any(row_sums <= 0.0):
            raise ValueError("proposal matrix has a zero row")
        q = q / row_sums
        self.name = name
        self._Q = q
        self._n_states = n_states
        self._n_vars = n_vars
        self._log_Q = np.full_like(q, -np.inf)
        positive = q > 0.0
        self._log_Q[positive] = np.log(q[positive])

    def propose(self, x: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, float, float]:
        i = state_to_index(as_binary_vector(x, self._n_vars))
        j = int(rng.choice(self._n_states, p=self._Q[i]))
        return _unpack_index(j, self._n_vars), float(self._log_Q[i, j]), float(self._log_Q[j, i])

    def proposal_matrix(self, n_vars: int) -> np.ndarray:
        if n_vars != self._n_vars:
            raise ValueError(f"expected n_vars={self._n_vars}, got {n_vars}")
        return self._Q


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
    print(f"    transition matrix for {kernel.name}...", flush=True)
    t_gap = time.perf_counter()
    gap = float(spectral_gap(engine.transition_matrix(), pi))
    gap_seconds = time.perf_counter() - t_gap
    print(f"    gap={gap:.4g} ({gap_seconds:.1f}s); running tabulated-Q chain...", flush=True)

    n_steps = int(chain_cfg["n_steps"])
    burn_in = int(chain_cfg["burn_in"])
    if burn_in >= n_steps:
        raise ValueError(f"burn_in ({burn_in}) must be < n_steps ({n_steps})")
    x0 = np.zeros(target.n_vars, dtype=np.uint8)
    chain_engine = MetropolisEngine(
        target, TabulatedKernel(kernel.name, kernel.proposal_matrix(target.n_vars))
    )
    t0 = time.perf_counter()
    result = chain_engine.run(n_steps=n_steps, x0=x0, seed=int(chain_cfg["seed"]))
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
        "gap_seconds": gap_seconds,
        "wall_seconds": elapsed,
        "n_steps": n_steps,
        "burn_in": burn_in,
    }


_KERNEL_COLORS = {
    "uniform": "#7f7f7f",
    "single-flip": "#4169e1",
    "add-delete-swap": "#1a7f37",
    "quench-analytic": "#c0392b",
    "quench-learned": "#8e1b12",
}


def _md_table(headers: list[str], rows: list[list[str]]) -> str:
    head = "| " + " | ".join(headers) + " |"
    sep = "| " + " | ".join("---" for _ in headers) + " |"
    body = "\n".join("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join([head, sep, body])


def _write_scoreboard_figure(path: Path, rows: list[dict[str, Any]]) -> None:
    """Gap, ESS/step, and ESS/sec so a bigger gap cannot hide a 100× cost."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = [str(row["kernel"]) for row in rows]
    colors = [_KERNEL_COLORS.get(name, "#2c3e50") for name in names]
    panels = [
        ("spectral_gap", r"spectral gap $\delta$", False),
        ("ess_size_per_step", r"ESS / step ($|\gamma|$)", False),
        ("ess_size_per_sec", r"ESS / sec ($|\gamma|$)", True),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.8))
    x = np.arange(len(names))
    for ax, (key, ylabel, log_y) in zip(axes, panels, strict=True):
        values = np.array([float(row[key]) for row in rows], dtype=np.float64)
        ax.bar(x, values, color=colors, width=0.72)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel(ylabel)
        if log_y:
            positive = values[values > 0.0]
            ax.set_yscale("log")
            if positive.size:
                ax.set_ylim(positive.min() * 0.5, positive.max() * 2.0)
        ax.grid(True, axis="y", alpha=0.3)
        ads = next((row for row in rows if row["kernel"] == "add-delete-swap"), None)
        if ads is not None and float(ads[key]) > 0.0:
            ax.axhline(float(ads[key]), color=_KERNEL_COLORS["add-delete-swap"], ls="--", lw=0.8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _read_scoreboard(rows: list[dict[str, Any]]) -> list[str]:
    """Honest quench-vs-ADS comparison. Negative results are still a result."""
    by_name = {str(row["kernel"]): row for row in rows}
    ads = by_name.get("add-delete-swap")
    if ads is None:
        return ["- Add-delete-swap is missing from this run; no baseline comparison."]

    lines = [
        "Add-delete-swap is the baseline. A larger gap that costs more wall-clock",
        "is not an advantage — both columns have to move the same way.",
        "",
    ]
    for name in ("quench-analytic", "quench-learned"):
        row = by_name.get(name)
        if row is None:
            continue
        ads_gap = float(ads["spectral_gap"])
        ads_ess = float(ads["ess_size_per_sec"])
        gap_ratio = float(row["spectral_gap"]) / ads_gap if ads_gap > 0.0 else float("inf")
        ess_ratio = float(row["ess_size_per_sec"]) / ads_ess if ads_ess > 0.0 else float("inf")
        gap_verb = "beats" if gap_ratio > 1.05 else ("matches" if gap_ratio > 0.95 else "loses to")
        ess_verb = "beats" if ess_ratio > 1.05 else ("matches" if ess_ratio > 0.95 else "loses to")
        lines.append(
            f"- `{name}` {gap_verb} ADS on exact gap "
            f"({gap_ratio:.2f}×) and {ess_verb} ADS on ESS/sec "
            f"({ess_ratio:.2f}×)."
        )
    lines.extend(
        [
            "",
            "Diabetes p=10 is mildly correlated, not a multimodal stress test.",
            "A loss here is not a gate failure — E02b asks whether the ratio",
            "grows with ρ. PIP error after a few thousand draws is Monte Carlo",
            "(1/δ is tens to hundreds of steps), not a wall breach.",
        ]
    )
    return lines


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
            "Gap is exact (transition-matrix). ESS chains sample the same Q the",
            "gap used, so ESS/step is the mixing of that kernel. ESS/sec is",
            "Q-sampling + MH, not a fresh expm per propose; `gap_seconds` in",
            "the JSON is the unitary-assembly cost.",
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
                f"{row['gap_seconds']:.1f}",
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
                "Q build (s)",
            ],
            score_rows,
        )
    )
    lines.extend(
        [
            "",
            "![scoreboard](scoreboard.png)",
            "",
            "## Read",
            "",
            *_read_scoreboard(rows),
            "",
            "## Notes",
            "",
            "- Add-delete-swap is the baseline that matters. Beating uniform proves nothing.",
            "- Headline comparison is the exact gap. ESS/sec here is tabulated-Q, not QPU time.",
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
    for kind in ("analytic", "learned"):
        d = diagnoses[kind]
        print(
            f"  {kind} surrogate: Spearman={d['spearman']:.3f} "
            f"ground-state rank={d['ground_state_rank']}/{d['n_states']}",
            flush=True,
        )
        if d["spearman"] < 0.3:
            print(
                f"  WARNING: {kind} Spearman is weak — quench may explore the wrong landscape",
                flush=True,
            )

    rows: list[dict[str, Any]] = []
    for kernel in kernels.values():
        print(f"  scoring {kernel.name}...", flush=True)
        row = _score_kernel(target, kernel, exact_pips, pi, dict(config["chain"]))
        rows.append(row)
        print(
            f"  {row['kernel']:20s}  gap={row['spectral_gap']:.4g}  "
            f"ESS/sec={row['ess_size_per_sec']:.2f}  PIP err={row['pip_error']:.3f}",
            flush=True,
        )
    meta["finished_utc"] = datetime.now(timezone.utc).isoformat()

    import pandas as pd

    pd.DataFrame(rows).to_csv(output_dir / "scoreboard.csv", index=False)
    (output_dir / "meta.json").write_text(json.dumps(meta, indent=2, default=str) + "\n")
    (output_dir / "summary.json").write_text(
        json.dumps({"diagnoses": diagnoses, "rows": rows}, indent=2) + "\n"
    )
    _write_scoreboard_figure(output_dir / "scoreboard.png", rows)
    _write_summary(output_dir / "summary.md", config, meta, dataset_label, diagnoses, rows)
    print(f"wrote {output_dir / 'summary.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
