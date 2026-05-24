"""
Sweep orchestrator for the H4-square compressed-sensing scaling experiment.

Given one (R, basis) cell — a trained Psi and an FCI reference — this module
runs the K_s and seed sweep and emits records matching the frozen
``OmegaQMC.cs.analysis`` schema. Memory is bounded by the determinant chunk
size, not by the (n_det x K_s_max) full f_I matrix, so the orchestrator
scales to candidate sets of 1e5+ on a single workstation.

Pipeline
--------
1. Evaluate natural orbitals at every walker once (shape K_s_max x N x n_orb).
2. For each (K_s, seed) draw a fixed subsample of walker indices; reused
   for every determinant chunk so the per-determinant means are mutually
   consistent within a (K_s, seed).
3. Stream over determinant chunks: build the f_I block, slice the
   pre-drawn subsample indices, accumulate the chunk mean into the
   per-(K_s, seed) running vector.
4. After all chunks: for each (K_s, eta, seed) apply the Lasso (soft-
   threshold with the reference preserved) and record the cell.

The default lambda strategy is ``lambda = lambda_coef * eta``. CV selection
remains exposed via :func:`OmegaQMC.cs.estimators.lambda_cv` for ablations.
"""

from math import comb
from typing import Mapping, Sequence, Tuple

import numpy as np

from OmegaQMC.cs.estimators import (
    evaluate_orbitals_on_walkers,
    f_I_matrix,
    normalize_and_align,
    recovery_metrics,
    soft_threshold,
)


def _subsample_indices(
    K_s_sweep: Sequence[int],
    n_seeds: int,
    K_s_max: int,
    seed_base: int,
) -> dict:
    """For each (K_s, seed) in the sweep, pre-draw walker indices."""
    out = {}
    for K_s in K_s_sweep:
        if K_s > K_s_max:
            continue
        K_s = int(K_s)
        for seed_idx in range(n_seeds):
            rng = np.random.default_rng((seed_base, K_s, seed_idx))
            out[(K_s, seed_idx)] = rng.choice(K_s_max, size=K_s, replace=False)
    return out


def precompute_means(
    mol,
    fci_ref: Mapping,
    walker_bank: np.ndarray,
    psi_vals_bank: np.ndarray,
    K_s_sweep: Sequence[int],
    n_seeds: int = 20,
    det_chunk_size: int = 200,
    seed_base: int = 0,
    walker_convention: str = "grouped",
    return_second_moments: bool = False,
) -> dict:
    """Stream f_I in determinant chunks, accumulate per-(K_s, seed) means.

    ``walker_convention`` controls how alpha/beta electrons are extracted
    from the bank. ``"grouped"`` is the default (matches synthetic tests
    and PySCF). NN-VMC walker banks dumped by ``OmegaQMC.vmc_nn`` are in
    ``"interleaved"`` convention and require that string.

    With ``return_second_moments=False`` (default) returns
    ``{(K_s, seed_idx): means_vector_of_length_n_det}``.

    With ``return_second_moments=True`` returns
    ``(means, second_moments)`` where ``second_moments[key]`` holds the
    per-coefficient sample mean of ``f_I^2``. Needed for the
    bias-corrected proj_mass estimator
    (see :func:`OmegaQMC.cs.estimators.bias_corrected_proj_mass`) when
    the candidate set is large relative to the walker count.
    """
    candidate = fci_ref["candidate_set"]
    n_det = len(candidate)
    no_coeff = fci_ref["no_coeff_ao"]
    n_alpha, n_beta = fci_ref["nelec"]
    K_s_max = walker_bank.shape[0]
    if psi_vals_bank.shape[0] != K_s_max:
        raise ValueError(
            f"walker_bank K_s_max={K_s_max} does not match "
            f"psi_vals_bank length={psi_vals_bank.shape[0]}"
        )

    orb_vals = evaluate_orbitals_on_walkers(
        mol, walker_bank, no_coeff,
        convention=walker_convention,
        n_alpha=n_alpha, n_beta=n_beta,
    )
    subsamples = _subsample_indices(K_s_sweep, n_seeds, K_s_max, seed_base)
    if not subsamples:
        raise ValueError(
            "no (K_s, seed) pair survived: every K_s in the sweep exceeds "
            f"walker bank size {K_s_max}"
        )

    means = {key: np.zeros(n_det, dtype=float) for key in subsamples}
    m2s = ({key: np.zeros(n_det, dtype=float) for key in subsamples}
           if return_second_moments else None)
    for start in range(0, n_det, det_chunk_size):
        end = min(start + det_chunk_size, n_det)
        chunk = candidate[start:end]
        f_I_chunk = f_I_matrix(orb_vals, chunk, psi_vals_bank, n_alpha, n_beta)
        if return_second_moments:
            f_I_sq_chunk = f_I_chunk ** 2
        for key, idx in subsamples.items():
            means[key][start:end] = f_I_chunk[:, idx].mean(axis=1)
            if return_second_moments:
                m2s[key][start:end] = f_I_sq_chunk[:, idx].mean(axis=1)
    if return_second_moments:
        return means, m2s
    return means


def _build_cells(
    means: Mapping,
    c_true: np.ndarray,
    R: float,
    basis: str,
    K_s_sweep: Sequence[int],
    etas: Sequence[float],
    n_seeds: int,
    N_det_full: int,
    lambda_coef: float,
) -> list:
    """Build per-(K_s, eta, seed) cell records.

    For each cell the raw sample mean is normalized via
    :func:`OmegaQMC.cs.estimators.normalize_and_align` (essential when the
    trial Psi is unnormalized in L^2, as in NN-VMC) and the resulting
    vector is soft-thresholded with ``lam = lambda_coef * eta``. The
    reference determinant (index 0) is preserved un-thresholded. The
    diagnostic ``proj_mass = sum(c_raw**2)`` is stored on each cell
    alongside the required schema fields.
    """
    ref_sign = float(np.sign(c_true[0])) if c_true[0] != 0 else 0.0
    cells = []
    for K_s in K_s_sweep:
        K_s = int(K_s)
        for eta in etas:
            eta = float(eta)
            lam = lambda_coef * eta
            K_eff = int(np.sum(np.abs(c_true) > eta))
            for seed_idx in range(n_seeds):
                key = (K_s, seed_idx)
                if key not in means:
                    continue
                mean_vec = means[key]
                c_norm, proj_mass = normalize_and_align(mean_vec, ref_sign)
                c_hat = soft_threshold(c_norm, lam)
                c_hat[0] = c_norm[0]
                metrics = recovery_metrics(c_hat, c_true, eta)
                cells.append(dict(
                    R=float(R), basis=str(basis), eta=eta,
                    K_eff=K_eff, K_s=K_s, seed=int(seed_idx),
                    L_inf_err=metrics["L_inf_err"],
                    L_2_err=metrics["L_2_err"],
                    support_err=metrics["support_err"],
                    N_det=int(N_det_full),
                    lambda_used=float(lam),
                    proj_mass=float(proj_mass),
                ))
    return cells


def _build_aux(
    fci_ref: Mapping,
    R: float,
    basis: str,
    etas: Sequence[float],
    psi_nn_energy_error: float,
    N_det_full: int,
) -> dict:
    ci_dict = fci_ref["ci_dict"]
    ref_det = fci_ref["reference_det"]
    max_c = max(
        (abs(v) for k, v in ci_dict.items() if k != ref_det),
        default=0.0,
    )
    eta_min = float(min(etas))
    K_diag = sum(1 for v in ci_dict.values() if abs(v) > eta_min)
    return dict(
        R=float(R),
        basis=str(basis),
        N_det=int(N_det_full),
        psi_nn_energy_error=float(psi_nn_energy_error),
        max_c_corr=float(max_c),
        K_eff_for_diag=int(K_diag),
    )


def run_sweep(
    mol,
    fci_ref: Mapping,
    walker_bank: np.ndarray,
    psi_vals_bank: np.ndarray,
    R: float,
    basis: str,
    K_s_sweep: Sequence[int],
    etas: Sequence[float],
    n_seeds: int = 20,
    psi_nn_energy_error: float = 0.0,
    lambda_coef: float = 0.5,
    det_chunk_size: int = 200,
    seed_base: int = 0,
    walker_convention: str = "grouped",
) -> Tuple[list, dict]:
    """End-to-end sweep for one (R, basis) cell.

    Returns ``(cells, aux)`` matching ``OmegaQMC.cs.analysis.CELL_FIELDS``
    and ``AUX_FIELDS``. Set ``walker_convention="interleaved"`` for NN-VMC
    walker banks produced by ``OmegaQMC.vmc_nn``.
    """
    candidate = fci_ref["candidate_set"]
    ci_dict = fci_ref["ci_dict"]
    c_true = np.array([ci_dict[k] for k in candidate], dtype=float)
    n_orb = int(fci_ref["n_orb"])
    n_alpha, n_beta = fci_ref["nelec"]
    N_det_full = comb(n_orb, n_alpha) * comb(n_orb, n_beta)

    means = precompute_means(
        mol, fci_ref, walker_bank, psi_vals_bank,
        K_s_sweep, n_seeds=n_seeds,
        det_chunk_size=det_chunk_size, seed_base=seed_base,
        walker_convention=walker_convention,
    )
    cells = _build_cells(
        means, c_true, R, basis,
        K_s_sweep, etas, n_seeds,
        N_det_full, lambda_coef,
    )
    aux = _build_aux(
        fci_ref, R, basis, etas, psi_nn_energy_error, N_det_full,
    )
    return cells, aux
