"""Remake the four paper figures from recorded summary tables.

The numbers are copied from results/**/summary.md. Re-running an
experiment is not this script's job — if a table moves, update
both the summary and the literals here.
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[2] / ".mplconfig"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parent / "figures"

# results/E01/summary.md, T=0.1 geometric-mean gaps
E01_N = np.array([5, 6, 7, 8, 9, 10])
E01_QUENCH = np.array([0.08327, 0.07293, 0.06355, 0.06142, 0.03989, 0.02602])
E01_UNIFORM = np.array([0.03441, 0.01724, 0.00847, 0.004027, 0.002069, 0.001002])
E01_LOCAL = np.array([5.142e-11, 2.875e-12, 3.339e-14, 1.513e-12, 1.926e-14, 4.709e-15])

# results/E02/summary.md, learned/ADS gap ratio
E02_RHO = np.array([0.0, 0.3, 0.5, 0.7, 0.9])
E02_LEARNED_RATIO = np.array([0.33, 0.32, 0.34, 0.35, 0.42])
E02_ANALYTIC_RATIO = np.array([0.17, 0.18, 0.19, 0.22, 0.19])

# results/E03/summary.md
E03_LAM = np.array([0.0, 0.001, 0.005, 0.02, 0.1])
E03_TEFF_DIA = np.array([1.86, 1.93, 1.9, 2.04, 2.28])
E03_TEFF_RHO = np.array([1.7, 1.71, 1.72, 1.75, 2.04])
E03_TPI_DIA = 0.292
E03_TPI_RHO = 0.199

# results/E03/bias_audit/summary.md
AUDIT_GAMMA = np.array([0.0, 0.01, 0.05, 0.1])
AUDIT_TV = np.array([0.067, 0.154, 0.808, 1.000])  # ideal, then AD
AUDIT_ADS_TV = 0.071


def _style() -> None:
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.grid": True,
            "grid.alpha": 0.3,
            "grid.linestyle": "-",
        }
    )


def fig1_e01() -> None:
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    ax.semilogy(E01_N, E01_QUENCH, "o-", color="#c0392b", label="quench")
    ax.semilogy(E01_N, E01_UNIFORM, "s-", color="#7f7f7f", label="uniform")
    ax.semilogy(E01_N, E01_LOCAL, "^-", color="#4169e1", label="single-flip")
    ax.set_xlabel("n")
    ax.set_ylabel(r"geometric-mean gap $\langle\delta\rangle$")
    ax.set_title("E01, T = 0.1")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "fig1_e01_gap.png", dpi=160)
    plt.close(fig)


def fig2_rho() -> None:
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    ax.plot(E02_RHO, E02_LEARNED_RATIO, "o-", color="#8e1b12", label="learned / ADS")
    ax.plot(E02_RHO, E02_ANALYTIC_RATIO, "s--", color="#c0392b", label="analytic / ADS")
    ax.axhline(1.0, color="#1a7f37", ls=":", lw=1.0, label="parity")
    ax.set_xlabel(r"equicorrelation $\rho$")
    ax.set_ylabel("gap ratio vs add-delete-swap")
    ax.set_ylim(0.0, 1.15)
    ax.set_title("E02b, p = 10 exact tier")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "fig2_rho_ratio.png", dpi=160)
    plt.close(fig)


def fig3_teff() -> None:
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    x = E03_LAM + 1e-6
    ax.plot(x, E03_TEFF_DIA, "o-", color="#c0392b", label=r"diabetes $T_{\mathrm{eff}}$")
    ax.plot(x, E03_TEFF_RHO, "s-", color="#1f4e79", label=r"$\rho=0.9$ $T_{\mathrm{eff}}$")
    ax.axhline(E03_TPI_DIA, color="#c0392b", ls="--", lw=0.8, label=r"diabetes $T_\pi$")
    ax.axhline(E03_TPI_RHO, color="#1f4e79", ls="--", lw=0.8, label=r"$\rho=0.9$ $T_\pi$")
    ax.set_xscale("symlog", linthresh=1e-3)
    ax.set_xlabel(r"depolarizing $\lambda$")
    ax.set_ylabel("effective proposal temperature")
    ax.set_title("E03, surrogate landscape")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "fig3_teff.png", dpi=160)
    plt.close(fig)


def fig4_audit() -> None:
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    ax.plot(AUDIT_GAMMA, AUDIT_TV, "o-", color="#c0392b", label="quench (ideal, then AD)")
    ax.axhline(AUDIT_ADS_TV, color="#1a7f37", ls="--", label="add-delete-swap")
    ax.set_xlabel(r"amplitude-damping $\gamma$")
    ax.set_ylabel("TV vs enumerated posterior")
    ax.set_ylim(0.0, 1.05)
    ax.set_title("Bias audit, diabetes p = 10")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "fig4_audit_tv.png", dpi=160)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    _style()
    fig1_e01()
    fig2_rho()
    fig3_teff()
    fig4_audit()
    print(f"wrote figures in {OUT}")


if __name__ == "__main__":
    main()
