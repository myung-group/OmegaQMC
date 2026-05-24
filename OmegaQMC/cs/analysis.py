"""
Pre-registered analysis for the H4-square compressed-sensing scaling experiment.

This module is the *contract* between the data-collection pipeline and the
publication-grade scaling claim. It must be frozen before data collection.
Any change after data is collected invalidates the pre-registration; bump
``SCHEMA_VERSION`` and commit before editing.

Protocol summary
----------------
For each cell ``(R, basis, eta, K_s, seed)`` the collection pipeline writes
one record. For each ``(R, basis)`` geometry it writes one auxiliary record.

1. Convergence gate. Drop every (R, basis) whose trained Psi_NN deviates
   from FCI by more than ``PSI_NN_ENERGY_GATE_MEH`` milli-Hartree. Excluded
   geometries are listed in the result so the count is transparent.

2. K_s*. For each (R, basis, eta), K_s* is the smallest K_s in the sweep
   at which the median over seeds of the L_infinity recovery error falls
   below eta.

3. Regression. Fit ``log K_s* = alpha log K_eff + beta log log N_det + gamma``
   on the surviving (R, basis, eta) cells via ordinary least squares. Report
   95% bootstrap confidence intervals on alpha and beta.

4. Flat-sparsity diagnostic. Fit log max|c_I^corr| vs log K_eff across
   geometries. A healthy flat-sparsity regime predicts slope near -0.5
   (largest correlation coefficient decays as 1/sqrt(K)). Slope above
   ``FLAT_SPARSITY_SLOPE_THRESHOLD`` indicates the spectrum is dominated
   by a small number of large coefficients and the flat-sparsity theorem
   does not apply.

5. Regime classification (decision rule, frozen):

   * alpha in [0.9, 1.2], beta in [0.7, 1.3]  -> "RIP"
   * alpha in [1.8, 2.2], beta in [0.7, 1.3]  -> "coherence"
   * alpha > 2.5 or beta > 2.0                -> "broken"
   * flat-sparsity diagnostic slope > -0.3    -> "broken"
   * otherwise                                -> "ambiguous"

Schema
------
``cells`` is a sequence of mappings with the fields in ``CELL_FIELDS``.
``aux`` is a sequence of mappings with the fields in ``AUX_FIELDS``.
``validate_schema`` enforces both.
"""

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Optional, Sequence

import numpy as np


SCHEMA_VERSION = "1.0.0"

CELL_FIELDS = (
    "R",            # float, Angstrom
    "basis",        # str, e.g. "cc-pVDZ"
    "eta",          # float, sparsity threshold
    "K_eff",        # int, # of FCI coefficients with |c| > eta
    "K_s",          # int, samples used
    "seed",         # int, subsample seed
    "L_inf_err",    # float, max_I |c_hat_I - c_FCI_I|
    "L_2_err",      # float
    "support_err",  # float in [0, 1], symmetric difference / |support|
    "N_det",        # int, candidate-set size
    "lambda_used",  # float, Lasso regularization that was applied
)

AUX_FIELDS = (
    "R",                     # float
    "basis",                 # str
    "N_det",                 # int
    "psi_nn_energy_error",   # float, mE_h: E_NN - E_FCI (signed)
    "max_c_corr",            # float, max_{I != 0} |c_FCI_I|
    "K_eff_for_diag",        # int, K_eff at the smallest eta used (for flat-sparsity diagnostic)
)


PSI_NN_ENERGY_GATE_MEH = 0.5

ALPHA_RIP = (0.9, 1.2)
ALPHA_COHERENCE = (1.8, 2.2)
BETA_OK = (0.7, 1.3)
ALPHA_BAD = 2.5
BETA_BAD = 2.0
FLAT_SPARSITY_SLOPE_THRESHOLD = -0.3


@dataclass
class ScalingResult:
    """Output of the end-to-end analysis."""
    alpha: float
    alpha_ci: tuple
    beta: float
    beta_ci: tuple
    gamma: float
    r_squared: float
    n_cells: int
    regime: str
    flat_sparsity_slope: float
    gated_geometries: list = field(default_factory=list)
    k_star_table: list = field(default_factory=list)


def validate_schema(
    cells: Sequence[Mapping[str, Any]],
    aux: Sequence[Mapping[str, Any]],
) -> None:
    """Raise ValueError if the input records do not match the frozen schema."""
    if not cells:
        raise ValueError("cells is empty")
    if not aux:
        raise ValueError("aux is empty")
    for i, row in enumerate(cells):
        missing = [f for f in CELL_FIELDS if f not in row]
        if missing:
            raise ValueError(f"cells[{i}] missing fields: {missing}")
    for i, row in enumerate(aux):
        missing = [f for f in AUX_FIELDS if f not in row]
        if missing:
            raise ValueError(f"aux[{i}] missing fields: {missing}")


def apply_convergence_gate(
    cells: Sequence[Mapping[str, Any]],
    aux: Sequence[Mapping[str, Any]],
) -> tuple:
    """Filter cells whose geometry failed the Psi_NN energy gate.

    Returns ``(kept_cells, gated_pairs)`` where ``gated_pairs`` is a list of
    ``(R, basis)`` tuples that were excluded.
    """
    gated = {
        (row["R"], row["basis"])
        for row in aux
        if abs(row["psi_nn_energy_error"]) > PSI_NN_ENERGY_GATE_MEH
    }
    kept = [c for c in cells if (c["R"], c["basis"]) not in gated]
    return kept, sorted(gated)


def compute_K_s_star_table(cells: Sequence[Mapping[str, Any]]) -> list:
    """For each (R, basis, eta), find smallest K_s s.t. median(L_inf_err) <= eta.

    Returns a list of dicts with keys: R, basis, eta, K_eff, N_det, K_s_star.
    ``K_s_star`` is ``None`` if no K_s in the sweep met the criterion.
    """
    groups: dict = {}
    for row in cells:
        key = (row["R"], row["basis"], row["eta"])
        groups.setdefault(key, []).append(row)

    table = []
    for (R, basis, eta), rows in groups.items():
        K_eff = rows[0]["K_eff"]
        N_det = rows[0]["N_det"]
        by_K_s: dict = {}
        for r in rows:
            by_K_s.setdefault(r["K_s"], []).append(r["L_inf_err"])
        ordered = sorted(by_K_s.items())
        K_s_star: Optional[int] = None
        for K_s, errs in ordered:
            if np.median(errs) <= eta:
                K_s_star = int(K_s)
                break
        table.append(dict(
            R=R, basis=basis, eta=eta,
            K_eff=K_eff, N_det=N_det, K_s_star=K_s_star,
        ))
    return table


def _ols(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    return coef


def fit_scaling(
    k_star_table: Sequence[Mapping[str, Any]],
    n_bootstrap: int = 2000,
    seed: int = 0,
) -> dict:
    """Fit log K_s* = alpha log K_eff + beta log log N_det + gamma.

    Cells with ``K_s_star is None`` are excluded (the sweep never met the
    target). Bootstrap is over surviving cells.
    """
    valid = [r for r in k_star_table if r["K_s_star"] is not None]
    if len(valid) < 5:
        raise ValueError(
            f"too few cells with K_s_star for regression: {len(valid)} < 5"
        )

    y = np.log(np.array([r["K_s_star"] for r in valid], dtype=float))
    x1 = np.log(np.array([r["K_eff"] for r in valid], dtype=float))
    x2 = np.log(np.log(np.array([r["N_det"] for r in valid], dtype=float)))
    X = np.column_stack([x1, x2, np.ones_like(x1)])

    alpha, beta, gamma = _ols(X, y)
    y_pred = X @ np.array([alpha, beta, gamma])
    ss_res = float(np.sum((y - y_pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    rng = np.random.default_rng(seed)
    n = len(y)
    alphas = np.empty(n_bootstrap)
    betas = np.empty(n_bootstrap)
    for b in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        try:
            coef = _ols(X[idx], y[idx])
        except np.linalg.LinAlgError:
            coef = np.full(3, np.nan)
        alphas[b] = coef[0]
        betas[b] = coef[1]
    alpha_ci = (float(np.nanpercentile(alphas, 2.5)),
                float(np.nanpercentile(alphas, 97.5)))
    beta_ci = (float(np.nanpercentile(betas, 2.5)),
               float(np.nanpercentile(betas, 97.5)))

    return dict(
        alpha=float(alpha), beta=float(beta), gamma=float(gamma),
        alpha_ci=alpha_ci, beta_ci=beta_ci,
        r_squared=r2, n_cells=n,
    )


def flat_sparsity_diagnostic(aux: Sequence[Mapping[str, Any]]) -> float:
    """Slope of log max|c_corr| vs log K_eff across geometries.

    Healthy flat-sparsity regime: slope near -0.5. Returns NaN if too few
    geometries to fit.
    """
    pts = [(r["K_eff_for_diag"], r["max_c_corr"]) for r in aux
           if r["K_eff_for_diag"] is not None and r["max_c_corr"] is not None]
    if len(pts) < 3:
        return float("nan")
    K_arr = np.array([p[0] for p in pts], dtype=float)
    c_arr = np.array([p[1] for p in pts], dtype=float)
    mask = (K_arr > 1) & (c_arr > 0) & np.isfinite(K_arr) & np.isfinite(c_arr)
    if mask.sum() < 3:
        return float("nan")
    slope, _ = np.polyfit(np.log(K_arr[mask]), np.log(c_arr[mask]), 1)
    return float(slope)


def classify_regime(
    alpha: float,
    beta: float,
    flat_slope: float,
) -> str:
    """Apply the pre-registered decision rule to the fitted exponents."""
    if not np.isfinite(alpha) or not np.isfinite(beta):
        return "broken"
    if alpha > ALPHA_BAD or beta > BETA_BAD:
        return "broken"
    if np.isfinite(flat_slope) and flat_slope > FLAT_SPARSITY_SLOPE_THRESHOLD:
        return "broken"

    in_rip = (ALPHA_RIP[0] <= alpha <= ALPHA_RIP[1]) and (BETA_OK[0] <= beta <= BETA_OK[1])
    in_coh = (ALPHA_COHERENCE[0] <= alpha <= ALPHA_COHERENCE[1]) and (BETA_OK[0] <= beta <= BETA_OK[1])
    if in_rip and not in_coh:
        return "RIP"
    if in_coh and not in_rip:
        return "coherence"
    return "ambiguous"


def run_analysis(
    cells: Sequence[Mapping[str, Any]],
    aux: Sequence[Mapping[str, Any]],
    n_bootstrap: int = 2000,
    seed: int = 0,
) -> ScalingResult:
    """End-to-end: validate, gate, K_s*, regression, classify."""
    validate_schema(cells, aux)
    kept, gated = apply_convergence_gate(cells, aux)
    k_star = compute_K_s_star_table(kept)
    fit = fit_scaling(k_star, n_bootstrap=n_bootstrap, seed=seed)
    flat = flat_sparsity_diagnostic(aux)
    regime = classify_regime(fit["alpha"], fit["beta"], flat)
    return ScalingResult(
        alpha=fit["alpha"], alpha_ci=fit["alpha_ci"],
        beta=fit["beta"], beta_ci=fit["beta_ci"],
        gamma=fit["gamma"], r_squared=fit["r_squared"],
        n_cells=fit["n_cells"], regime=regime,
        flat_sparsity_slope=flat,
        gated_geometries=list(gated),
        k_star_table=k_star,
    )
