"""
Headline plots for the H4-square sample-complexity scaling experiment.

All functions take the same record types as :mod:`OmegaQMC.cs.analysis` and
return a ``matplotlib.figure.Figure``. ``matplotlib`` is imported lazily so
that ``OmegaQMC.cs.analysis`` remains importable in headless environments.
"""

from typing import Mapping, Optional, Sequence

import numpy as np


def _mpl():
    import matplotlib.pyplot as plt  # noqa: WPS433
    return plt


def plot_scaling(k_star_table: Sequence[Mapping], fit: Optional[Mapping] = None):
    """Plot 1: log K_s* vs log K_eff, one curve per (R, basis).

    Overlays the fitted regression line if ``fit`` is provided.
    """
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(5.0, 4.0))

    by_geom: dict = {}
    for row in k_star_table:
        if row["K_s_star"] is None:
            continue
        by_geom.setdefault((row["R"], row["basis"]), []).append(row)

    for (R, basis), rows in sorted(by_geom.items()):
        rows.sort(key=lambda r: r["K_eff"])
        K = np.array([r["K_eff"] for r in rows], dtype=float)
        Ks = np.array([r["K_s_star"] for r in rows], dtype=float)
        ax.loglog(K, Ks, marker="o", linestyle="-", label=f"R={R}A, {basis}")

    if fit is not None:
        K_grid = np.array([r["K_eff"] for r in k_star_table if r["K_s_star"] is not None],
                          dtype=float)
        N_grid = np.array([r["N_det"] for r in k_star_table if r["K_s_star"] is not None],
                          dtype=float)
        if len(K_grid) > 0:
            Klin = np.linspace(K_grid.min(), K_grid.max(), 50)
            mean_logN = np.mean(np.log(np.log(N_grid)))
            y_fit = fit["alpha"] * np.log(Klin) + fit["beta"] * mean_logN + fit["gamma"]
            ax.plot(Klin, np.exp(y_fit), "k--",
                    label=f"fit: K_s*~K^{fit['alpha']:.2f}")

    ax.set_xlabel(r"$K_\mathrm{eff}$")
    ax.set_ylabel(r"$K_s^\star$")
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    return fig


def plot_log_N_det(k_star_table: Sequence[Mapping], K_target: int, tol: float = 0.2):
    """Plot 2: K_s* vs N_det at approximately fixed K_eff.

    Includes cells whose K_eff is within ``(1 +/- tol)*K_target``.
    """
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(5.0, 4.0))

    rows = [r for r in k_star_table
            if r["K_s_star"] is not None
            and (1 - tol) * K_target <= r["K_eff"] <= (1 + tol) * K_target]
    if not rows:
        ax.text(0.5, 0.5, f"no cells near K_eff={K_target}",
                ha="center", va="center", transform=ax.transAxes)
        return fig

    N = np.array([r["N_det"] for r in rows], dtype=float)
    Ks = np.array([r["K_s_star"] for r in rows], dtype=float)
    ax.semilogx(N, Ks, "o")
    ax.set_xlabel(r"$N_\mathrm{det}$")
    ax.set_ylabel(r"$K_s^\star$  at  $K_\mathrm{eff}\!\approx\!{}$".format(K_target))
    fig.tight_layout()
    return fig


def plot_phase_transition(
    cells: Sequence[Mapping],
    R: float,
    basis: str,
    eta: float,
):
    """Plot 3: recovery error (L_inf, median over seeds) vs K_s, one curve.

    The horizontal line is the eta threshold; the K_s where the curve crosses
    it is K_s*.
    """
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(5.0, 4.0))

    sub = [c for c in cells
           if c["R"] == R and c["basis"] == basis and c["eta"] == eta]
    by_K_s: dict = {}
    for c in sub:
        by_K_s.setdefault(c["K_s"], []).append(c["L_inf_err"])
    ordered = sorted(by_K_s.items())
    Ks = np.array([k for k, _ in ordered], dtype=float)
    med = np.array([np.median(v) for _, v in ordered])
    lo = np.array([np.percentile(v, 25) for _, v in ordered])
    hi = np.array([np.percentile(v, 75) for _, v in ordered])

    ax.loglog(Ks, med, "o-", label="median")
    ax.fill_between(Ks, lo, hi, alpha=0.2, label="IQR")
    ax.axhline(eta, color="red", linestyle="--", label=r"$\eta$")
    ax.set_xlabel(r"$K_s$")
    ax.set_ylabel(r"$L_\infty$ error")
    ax.set_title(f"R={R}A, {basis}, eta={eta}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig
