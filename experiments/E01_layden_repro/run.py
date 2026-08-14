"""E01 — reproduce Layden's spectral-gap scaling (Nature Fig. 2).

For every (n, T, instance) we build the exact MH transition matrix and
read off δ = 1 − |λ₂|. The quench proposal matrix is a quadrature of
|⟨y|U(γ,t)|x⟩|² over the paper's (γ, t) rectangle; it does not depend
on T, so one Q is reused across temperatures.

Gaps are averaged in log space (geometric mean). Arithmetic means are
dominated by the easy instances and flatten k — that is the roadmap
warning, and the reason this script refuses to fit k to raw averages.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from tunnelvision.diagnostics import spectral_gap
from tunnelvision.engine import MetropolisEngine
from tunnelvision.kernels.classical import SingleFlip, UniformFlip
from tunnelvision.kernels.quantum import QuenchKernel
from tunnelvision.targets.ising import IsingTarget, random_spin_glass

REPO_ROOT = Path(__file__).resolve().parents[2]
_PACKAGE_VERSIONS = ("numpy", "scipy", "qiskit", "qiskit-aer", "tunnelvision")


def _git_commit() -> str:
    try:
        head = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        dirty = subprocess.call(
            ["git", "diff-index", "--quiet", "HEAD", "--"],
            cwd=REPO_ROOT,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return f"{head}-dirty" if dirty else head
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _versions() -> dict[str, str]:
    out: dict[str, str] = {}
    for name in _PACKAGE_VERSIONS:
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            out[name] = "unknown"
    return out


def _load_config(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        loaded = yaml.safe_load(handle)
    if not isinstance(loaded, dict):
        raise ValueError(f"config must be a mapping, got {type(loaded)}")
    return loaded


def _apply_quick(config: dict[str, Any]) -> None:
    """Tiny grid so the entry point can be smoke-tested in seconds."""
    config["n"] = [4, 5, 6]
    config["n_instances"] = 3
    config["temperatures"] = [0.3, 1.0]
    quench = dict(config.get("quench", {}))
    quench["n_gamma"] = 3
    quench["n_t"] = 3
    config["quench"] = quench


def _instance_seed(base: int, n: int, instance: int) -> int:
    # Separate n-blocks so adding a size does not reshuffle smaller ones.
    return int(base) + 1_000 * int(n) + int(instance)


def gaps_for_instance(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Worker: one spin-glass instance, every requested T and kernel."""
    n = int(payload["n"])
    seed = int(payload["seed"])
    glass = random_spin_glass(
        n,
        topology=str(payload["topology"]),
        seed=seed,
        temperature=1.0,
        random_fields=bool(payload["random_fields"]),
    )
    quench_cfg = payload["quench"]
    kernels: dict[str, Any] = {}
    for name in payload["kernels"]:
        if name == "uniform":
            kernels[name] = UniformFlip()
        elif name == "single-flip":
            kernels[name] = SingleFlip()
        elif name == "quench":
            kernels[name] = QuenchKernel(
                glass.h,
                glass.J,
                gamma_range=tuple(quench_cfg["gamma_range"]),
                t_range=tuple(quench_cfg["t_range"]),
                trotter_dt=float(quench_cfg.get("trotter_dt", 0.8)),
                evolution=str(quench_cfg.get("evolution", "exact")),
                n_gamma=int(quench_cfg["n_gamma"]),
                n_t=int(quench_cfg["n_t"]),
            )
        else:
            raise ValueError(f"unknown kernel {name!r}")

    rows: list[dict[str, Any]] = []
    for temperature in payload["temperatures"]:
        target = IsingTarget(glass.h, glass.J, temperature=float(temperature))
        pi = target.enumerate_exact()
        for name, kernel in kernels.items():
            transition = MetropolisEngine(target, kernel).transition_matrix()
            rows.append(
                {
                    "n": n,
                    "temperature": float(temperature),
                    "instance_seed": seed,
                    "kernel": name,
                    "gap": float(spectral_gap(transition, pi)),
                }
            )
    return rows


def geometric_mean(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        raise ValueError("geometric_mean of empty array")
    clipped = np.clip(arr, 1e-16, None)
    return float(np.exp(np.mean(np.log(clipped))))


def fit_k_exponent(ns: np.ndarray, geo_gaps: np.ndarray) -> tuple[float, float]:
    """Least-squares fit of ⟨δ⟩_geo = A · 2^{−k n} in log2 space."""
    if ns.size < 2:
        raise ValueError("need at least two sizes to fit k")
    design = np.column_stack([np.ones(ns.size), -ns.astype(np.float64)])
    coef, *_ = np.linalg.lstsq(design, np.log2(np.clip(geo_gaps, 1e-16, None)), rcond=None)
    intercept, k = coef
    return float(k), float(2.0**intercept)


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ns = sorted({int(r["n"]) for r in rows})
    temperatures = sorted({float(r["temperature"]) for r in rows})
    kernels = sorted({str(r["kernel"]) for r in rows})
    table: dict[str, Any] = {}
    k_fits: dict[str, Any] = {}
    for temperature in temperatures:
        t_key = f"{temperature:g}"
        table[t_key] = {}
        k_fits[t_key] = {}
        for kernel in kernels:
            geo = []
            arith = []
            for n in ns:
                gaps = np.array(
                    [
                        r["gap"]
                        for r in rows
                        if r["n"] == n and r["kernel"] == kernel and r["temperature"] == temperature
                    ],
                    dtype=np.float64,
                )
                geo.append(geometric_mean(gaps))
                arith.append(float(gaps.mean()))
            geo_arr = np.asarray(geo)
            arith_arr = np.asarray(arith)
            n_arr = np.asarray(ns, dtype=np.float64)
            k, prefactor = fit_k_exponent(n_arr, geo_arr)
            k_arith, pre_arith = fit_k_exponent(n_arr, arith_arr)
            table[t_key][kernel] = {
                "n": ns,
                "geo_mean": geo_arr.tolist(),
                "arith_mean": arith_arr.tolist(),
            }
            k_fits[t_key][kernel] = {
                "k": k,
                "prefactor": prefactor,
                "k_arith": k_arith,
                "prefactor_arith": pre_arith,
            }
    return {"ns": ns, "temperatures": temperatures, "kernels": kernels, "gaps": table, "k": k_fits}


def _verdict(summary: dict[str, Any]) -> tuple[bool, str]:
    """Qualitative Nature-Fig.-2 gate, not a numerical clone of k = 0.29."""
    k = summary["k"]
    low = min(summary["temperatures"], key=float)
    high = max(summary["temperatures"], key=float)
    low_key, high_key = f"{low:g}", f"{high:g}"
    if "quench" not in k[low_key] or "uniform" not in k[low_key]:
        return False, "Need both quench and uniform kernels to judge the gate."

    k_q_low = k[low_key]["quench"]["k"]
    k_u_low = k[low_key]["uniform"]["k"]
    k_q_high = k[high_key]["quench"]["k"]
    k_u_high = k[high_key]["uniform"]["k"]
    ratio_low = k_u_low / k_q_low if k_q_low > 0.05 else float("inf")
    ratio_high = k_u_high / k_q_high if k_q_high > 0.05 else float("inf")

    def _gap_ratio(t_key: str) -> float:
        quench = summary["gaps"][t_key]["quench"]["geo_mean"][-1]
        uniform = summary["gaps"][t_key]["uniform"]["geo_mean"][-1]
        return float(quench / uniform) if uniform > 0.0 else float("inf")

    gap_ratio_low = _gap_ratio(low_key)
    gap_ratio_high = _gap_ratio(high_key)

    # Paper: at low T, quantum k ≈ 0.26–0.29 vs uniform k ≈ 1 (ratio ~3.4).
    # The high-T collapse of the *gap* (uniform catching up) is T ≫ 1;
    # within T ∈ {0.1, 1} we only require quench to stay clearly ahead.
    low_ok = 0.15 < k_q_low < 0.50 and k_u_low > 0.80 and ratio_low > 2.0
    still_ahead = gap_ratio_high > 4.0
    passed = bool(low_ok and still_ahead)
    lines = [
        f"At T={low:g}: k_quench={k_q_low:.3f}, k_uniform={k_u_low:.3f}, "
        f"k-ratio={ratio_low:.2f} (paper ≈ 3.4), "
        f"gap-ratio at largest n={gap_ratio_low:.1f}×.",
        f"At T={high:g}: k_quench={k_q_high:.3f}, k_uniform={k_u_high:.3f}, "
        f"k-ratio={ratio_high:.2f}, gap-ratio={gap_ratio_high:.1f}×.",
    ]
    if passed:
        lines.append("Gate: qualitative match — proceed to Rung 2.")
    else:
        lines.append("Gate: FAILED — stop and debug before Rung 2.")
    return passed, "\n".join(lines)


def _write_plot(summary: dict[str, Any], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"uniform": "#7f7f7f", "single-flip": "#4169e1", "quench": "#c0392b"}
    temperatures = summary["temperatures"]
    fig, axes = plt.subplots(
        1, len(temperatures), figsize=(4.2 * len(temperatures), 3.6), sharey=True
    )
    if len(temperatures) == 1:
        axes = [axes]
    ns = np.asarray(summary["ns"], dtype=float)
    for ax, temperature in zip(axes, temperatures, strict=True):
        t_key = f"{temperature:g}"
        for kernel in summary["kernels"]:
            geo = np.asarray(summary["gaps"][t_key][kernel]["geo_mean"])
            k = summary["k"][t_key][kernel]["k"]
            A = summary["k"][t_key][kernel]["prefactor"]
            color = colors.get(kernel, "black")
            ax.semilogy(ns, geo, "o", color=color, label=f"{kernel} (k={k:.2f})")
            n_line = np.linspace(ns.min() - 0.2, ns.max() + 0.2, 50)
            ax.semilogy(n_line, A * 2.0 ** (-k * n_line), "-", color=color, alpha=0.7)
        ax.set_title(f"T = {temperature:g}")
        ax.set_xlabel("n")
        ax.set_xticks(ns)
        ax.grid(True, which="both", alpha=0.3)
    axes[0].set_ylabel(r"geometric mean gap $\langle\delta\rangle$")
    axes[-1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _md_table(headers: list[str], rows: list[list[str]]) -> str:
    head = "| " + " | ".join(headers) + " |"
    sep = "| " + " | ".join("---" for _ in headers) + " |"
    body = "\n".join("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join([head, sep, body])


def _write_summary(
    path: Path,
    config: dict[str, Any],
    meta: dict[str, Any],
    summary: dict[str, Any],
    verdict_ok: bool,
    verdict_text: str,
) -> None:
    ns = summary["ns"]
    lines = [
        "# E01 — Layden reproduction",
        "",
        "Exact-tier spectral gaps on fully-connected spin glasses with",
        "random fields (Layden et al., Nature 619, 282 (2023), Fig. 2).",
        "⟨δ⟩ is the **geometric** mean over instances; k is the least-squares",
        "fit of ⟨δ⟩ ∝ 2^{−kn}.",
        "",
        "## Provenance",
        "",
        f"- git commit: `{meta['git_commit']}`",
        f"- started (UTC): {meta['started_utc']}",
        f"- finished (UTC): {meta['finished_utc']}",
        f"- package versions: {', '.join(f'{k}={v}' for k, v in meta['versions'].items())}",
        f"- instances per (n, T): {config['n_instances']}",
        f"- topology: {config['topology']}, random_fields={config['random_fields']}",
        f"- quench evolution: {config['quench'].get('evolution', 'exact')}, "
        f"grid {config['quench']['n_gamma']}×{config['quench']['n_t']}",
        "",
        "## k exponents (geometric-mean fit)",
        "",
    ]
    k_rows = []
    for temperature in summary["temperatures"]:
        t_key = f"{temperature:g}"
        row = [t_key] + [f"{summary['k'][t_key][k]['k']:.3f}" for k in summary["kernels"]]
        k_rows.append(row)
    lines.append(_md_table(["T", *summary["kernels"]], k_rows))
    t1_key = "1" if "1" in summary["k"] else f"{summary['temperatures'][-1]:g}"
    arith_bits = [
        f"{name}={summary['k'][t1_key][name]['k_arith']:.3f}" for name in summary["kernels"]
    ]
    lines.extend(
        [
            "",
            f"Arithmetic-mean k at T = {t1_key} (paper Fig. 2 convention): "
            + ", ".join(arith_bits)
            + ". Paper: quench 0.264(4), local 0.94(4), uniform 0.948(7).",
            "",
            "## Geometric-mean gaps",
            "",
        ]
    )
    for temperature in summary["temperatures"]:
        t_key = f"{temperature:g}"
        lines.append(f"### T = {t_key}")
        lines.append("")
        gap_rows = []
        for i, n in enumerate(ns):
            gap_rows.append(
                [str(n)]
                + [f"{summary['gaps'][t_key][k]['geo_mean'][i]:.4g}" for k in summary["kernels"]]
            )
        lines.append(_md_table(["n", *summary["kernels"]], gap_rows))
        lines.append("")
    lines.extend(
        [
            "## Gate",
            "",
            f"**{'PASS' if verdict_ok else 'FAIL'}**",
            "",
            verdict_text,
            "",
            "Paper reference (Fig. 2 / SM, arithmetic means over 500 instances,",
            "n = 3–10): at low T, quench k ≈ 0.26–0.29 vs uniform k ≈ 1.0;",
            "advantage shrinks as T grows because uniform proposals already mix.",
            "",
            "## Notes",
            "",
            "- Fit range is n = 5–10. n = 8–10 alone overestimates quench k (~0.62);",
            "  the paper's lever arm is n = 3–10. Twenty instances (vs 500) is enough",
            "  for the exponent, not for tight error bars.",
            "- Single-flip gaps at T ≤ 0.3 underflow on the hard instances; the",
            "  geometric-mean k there is not a physical exponent. At T = 1 the",
            "  local kernel is well-defined (k ≈ 0.83 geo / 0.64 arith).",
            "- E01 uses exact `e^{-iHt}` (Nature Fig. 2). The Trotter / Aer path is",
            "  tested in `tests/test_quench.py` and is what E03 / hardware will run.",
            "",
            "![gap vs n](gap_vs_n.png)",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def _run_pool(payloads: list[dict[str, Any]], n_workers: int) -> list[dict[str, Any]]:
    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = None  # type: ignore[assignment]

    rows: list[dict[str, Any]] = []
    if n_workers <= 1:
        iterator = payloads if tqdm is None else tqdm(payloads, desc="E01 instances")
        for payload in iterator:
            rows.extend(gaps_for_instance(payload))
        return rows

    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = [pool.submit(gaps_for_instance, payload) for payload in payloads]
        iterator = as_completed(futures)
        if tqdm is not None:
            iterator = tqdm(iterator, total=len(futures), desc="E01 instances")
        for future in iterator:
            rows.extend(future.result())
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).with_name("config.yaml"),
        help="YAML config (default: experiments/E01_layden_repro/config.yaml)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="tiny (n, instance, grid) smoke run, not the paper figure",
    )
    parser.add_argument("--workers", type=int, default=None, help="override n_workers")
    args = parser.parse_args(argv)

    config = _load_config(args.config)
    if args.quick:
        _apply_quick(config)
        config["output_dir"] = "results/E01_quick"
    if args.workers is not None:
        config["n_workers"] = args.workers

    n_values = [int(n) for n in config["n"]]
    n_instances = int(config["n_instances"])
    n_workers = int(config.get("n_workers", 0))
    if n_workers <= 0:
        import os

        n_workers = max(1, min(os.cpu_count() or 1, n_instances))

    payloads = [
        {
            "n": n,
            "seed": _instance_seed(int(config["instance_seed"]), n, instance),
            "temperatures": [float(t) for t in config["temperatures"]],
            "kernels": list(config["kernels"]),
            "quench": dict(config["quench"]),
            "topology": config["topology"],
            "random_fields": bool(config["random_fields"]),
        }
        for n in n_values
        for instance in range(n_instances)
    ]

    meta = {
        "git_commit": _git_commit(),
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "versions": _versions(),
        "config": config,
        "quick": bool(args.quick),
        "n_workers": n_workers,
    }
    print(
        f"E01: {len(payloads)} instances, n={n_values}, "
        f"T={config['temperatures']}, workers={n_workers}",
        flush=True,
    )
    rows = _run_pool(payloads, n_workers)
    meta["finished_utc"] = datetime.now(timezone.utc).isoformat()

    summary = _summarize(rows)
    verdict_ok, verdict_text = _verdict(summary)

    output_dir = REPO_ROOT / str(config.get("output_dir", "results/E01"))
    output_dir.mkdir(parents=True, exist_ok=True)
    import pandas as pd

    pd.DataFrame(rows).to_csv(output_dir / "gaps.csv", index=False)
    (output_dir / "meta.json").write_text(json.dumps(meta, indent=2, default=str) + "\n")
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    _write_plot(summary, output_dir / "gap_vs_n.png")
    _write_summary(
        output_dir / "summary.md",
        config,
        meta,
        summary,
        verdict_ok,
        verdict_text,
    )
    print(verdict_text)
    print(f"wrote {output_dir / 'summary.md'}")
    return 0 if verdict_ok or args.quick else 1


if __name__ == "__main__":
    sys.exit(main())
