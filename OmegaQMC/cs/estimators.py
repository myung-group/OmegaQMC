"""
f_I(R_k) estimators and Lasso recovery for compressed-sensing CI extraction.

Key formula
-----------
For natural orbitals {eta_p} with C_I = c_I times a normalized Slater
determinant D_I^norm = N D_I^raw with N = 1 / sqrt(N_alpha! * N_beta!), the
ratio

    f_I(R) = D_I^norm(R) / Psi(R)

satisfies E_{|Psi|^2 / Z}[f_I] = c_I when Psi is normalized in L^2 and the
orbitals are orthonormal. The empirical sample mean estimates c_I with
covariance (delta_{IJ} - c_I c_J*) / K_s. The Lasso loss

    L(c) = (1/(2 K_s)) sum_{k, I} (f_I(R_k) - c_I)^2 + lambda |c|_1

separates per-I because the implicit design matrix is the identity; the
minimizer is the soft-thresholded sample mean (reference excluded).
"""

import math
from typing import Sequence, Tuple, Optional

import numpy as np
import jax
import jax.numpy as jnp


def _normalization(n_alpha: int, n_beta: int) -> float:
    return 1.0 / math.sqrt(math.factorial(n_alpha) * math.factorial(n_beta))


def _interleaved_to_grouped_indices(n_alpha: int, n_beta: int) -> np.ndarray:
    """Permutation that maps OmegaQMC interleaved walker layout (alpha at
    even positions, beta at odd) into grouped layout (all alpha first,
    then all beta). Use as ``walkers[:, perm, :]`` or ``orb[:, perm, :]``.

    Assumes ``n_alpha >= n_beta`` (the closed-shell or high-spin-alpha
    convention used by OmegaQMC's NN adapter ``r_up = elec_crds[::2]``).
    Raises ValueError otherwise to surface ambiguous spin assignments
    before they corrupt the orbital evaluation.
    """
    if n_alpha < n_beta:
        raise ValueError(
            f"interleaved->grouped permutation expects n_alpha >= n_beta; "
            f"got ({n_alpha}, {n_beta})"
        )
    n_total = n_alpha + n_beta
    a_idx = np.arange(0, n_total, 2)[:n_alpha]
    b_idx = np.arange(1, n_total, 2)[:n_beta]
    return np.concatenate([a_idx, b_idx])


def evaluate_orbitals_on_walkers(
    mol,
    walkers: np.ndarray,
    no_coeff_ao: np.ndarray,
    convention: str = "grouped",
    n_alpha: int = None,
    n_beta: int = None,
) -> np.ndarray:
    """Natural orbital values at every electron position.

    ``walkers`` shape ``(K_s, N_electrons, 3)``, units of Bohr.

    The ``convention`` argument selects the electron layout of the input
    walkers. ``"grouped"`` (the default, matching test synthetic data and
    PySCF FCI) puts alpha electrons in positions ``0..n_alpha-1`` and beta
    in ``n_alpha..n_total-1``. ``"interleaved"`` (matching OmegaQMC NN-VMC
    walkers — see ``OmegaQMC.psi.nn.adapter`` where ``r_up = elec_crds[::2]``)
    puts alpha at even positions and beta at odd. For interleaved input the
    returned orbitals are permuted into grouped order so the downstream
    estimators always see alpha-first walkers; this requires ``n_alpha``
    and ``n_beta``.

    Returns ``(K_s, N_electrons, n_orb)`` in grouped layout.
    """
    K_s, N, _ = walkers.shape
    flat = np.asarray(walkers, dtype=float).reshape(-1, 3)
    ao_vals = mol.eval_gto("GTOval_sph", flat)
    no_vals = ao_vals @ np.asarray(no_coeff_ao)
    out = no_vals.reshape(K_s, N, -1)
    if convention == "grouped":
        return out
    if convention == "interleaved":
        if n_alpha is None or n_beta is None:
            raise ValueError(
                "convention='interleaved' requires n_alpha and n_beta"
            )
        if n_alpha + n_beta != N:
            raise ValueError(
                f"n_alpha+n_beta ({n_alpha}+{n_beta}) != walker electrons ({N})"
            )
        perm = _interleaved_to_grouped_indices(n_alpha, n_beta)
        return out[:, perm, :]
    raise ValueError(f"unknown convention {convention!r}")


def evaluate_ci_wavefunction(
    orb_vals: np.ndarray,
    candidate_set: Sequence,
    coeffs: np.ndarray,
    n_alpha: int,
    n_beta: int,
) -> np.ndarray:
    """Evaluate Psi(R) = sum_I c_I D_I^norm(R) at walker points.

    ``orb_vals`` shape ``(K_s, N_total, n_orb)``. Returns ``(K_s,)``.
    """
    n_norm = _normalization(n_alpha, n_beta)
    K_s = orb_vals.shape[0]
    orb_a = orb_vals[:, :n_alpha, :]
    orb_b = orb_vals[:, n_alpha:, :]
    psi = np.zeros(K_s, dtype=float)
    for c, (occ_a, occ_b) in zip(coeffs, candidate_set):
        M_a = orb_a[:, :, list(occ_a)]
        M_b = orb_b[:, :, list(occ_b)]
        psi += float(c) * n_norm * np.linalg.det(M_a) * np.linalg.det(M_b)
    return psi


def _f_I_matrix_core(
    orb_a: jnp.ndarray,
    orb_b: jnp.ndarray,
    occ_a_arr: jnp.ndarray,
    occ_b_arr: jnp.ndarray,
    psi_vals: jnp.ndarray,
    n_norm: float,
) -> jnp.ndarray:
    def per_sample(orb_a_k, orb_b_k, psi_k):
        def per_det(occ_a, occ_b):
            M_a = orb_a_k[:, occ_a]
            M_b = orb_b_k[:, occ_b]
            return n_norm * jnp.linalg.det(M_a) * jnp.linalg.det(M_b) / psi_k
        return jax.vmap(per_det)(occ_a_arr, occ_b_arr)
    out = jax.vmap(per_sample)(orb_a, orb_b, psi_vals)  # (K_s, n_det)
    return out.T


_f_I_matrix_core_jit = jax.jit(_f_I_matrix_core)


def f_I_matrix(
    orb_vals: np.ndarray,
    candidate_set: Sequence,
    psi_vals: np.ndarray,
    n_alpha: int,
    n_beta: int,
    use_jit: bool = True,
) -> np.ndarray:
    """f_I(R_k) = D_I^norm(R_k) / Psi(R_k), shape ``(n_det, K_s)``."""
    n_norm = _normalization(n_alpha, n_beta)
    orb_a = jnp.asarray(orb_vals[:, :n_alpha, :])
    orb_b = jnp.asarray(orb_vals[:, n_alpha:, :])
    occ_a_arr = jnp.asarray([list(o_a) for o_a, _ in candidate_set],
                            dtype=jnp.int32)
    occ_b_arr = jnp.asarray([list(o_b) for _, o_b in candidate_set],
                            dtype=jnp.int32)
    psi_j = jnp.asarray(psi_vals)
    fn = _f_I_matrix_core_jit if use_jit else _f_I_matrix_core
    out = fn(orb_a, orb_b, occ_a_arr, occ_b_arr, psi_j, n_norm)
    return np.asarray(out)


def soft_threshold(x: np.ndarray, lam: float) -> np.ndarray:
    return np.sign(x) * np.maximum(np.abs(x) - lam, 0.0)


def bias_corrected_proj_mass(
    c_raw: np.ndarray,
    m2: np.ndarray,
    K_s: int,
) -> float:
    """Unbiased estimator of basis_frac/Z = E[Sum c_I^2 / Z].

    The naive Sum c_raw^2 is biased high by Sum Var(c_raw_I) = (1/K_s)
    Sum Var(f_I). Subtracting the empirical Sum (m2 - c_raw^2)/K_s
    removes that bias. Use when n_det >> K_s and the noise floor
    dominates the signal in the naive projection mass.

    Args:
        c_raw: per-coefficient sample mean of f_I, shape (n_det,)
        m2: per-coefficient sample mean of f_I^2, shape (n_det,)
        K_s: number of samples per coefficient

    Returns:
        Bias-corrected proj_mass scalar. Can be negative if signal is
        swamped by noise; the caller is expected to handle that case.
    """
    c_raw = np.asarray(c_raw)
    m2 = np.asarray(m2)
    sample_var = m2 - c_raw ** 2
    bias = float(np.sum(sample_var) / K_s)
    naive = float(np.sum(c_raw ** 2))
    return naive - bias


def normalize_and_align_bias_corrected(
    c_raw: np.ndarray,
    m2: np.ndarray,
    K_s: int,
    reference_sign: float,
    floor: float = 1e-30,
) -> tuple:
    """Like :func:`normalize_and_align` but uses bias-corrected proj_mass.

    Returns ``(c_norm, proj_mass_corrected, proj_mass_naive)`` so the
    caller can see how much bias was subtracted.
    """
    proj_mass_naive = float(np.sum(np.asarray(c_raw) ** 2))
    proj_mass = bias_corrected_proj_mass(c_raw, m2, K_s)
    if proj_mass < floor:
        # Signal swamped by noise: fall back to the naive proj_mass with a warning.
        proj_mass = proj_mass_naive
    c_norm = np.asarray(c_raw) / np.sqrt(proj_mass)
    if reference_sign != 0.0 and c_norm[0] * reference_sign < 0:
        c_norm = -c_norm
    return c_norm, proj_mass, proj_mass_naive


def normalize_and_align(
    c_raw: np.ndarray,
    reference_sign: float,
) -> tuple:
    """Normalize raw sample means and align the global sign.

    The estimator ``E[f_I] = c_I / sqrt(Z)`` where ``Z = integral |Psi|^2``,
    so an unnormalized trial Psi (the typical NN-VMC case) leaves ``c_raw``
    scaled by ``1/sqrt(Z)``. Dividing by ``|c_raw|_2`` restores the
    coefficients to the same scale as a normalized CI vector.

    The wavefunction's overall sign is unobservable, so we also flip the
    sign of the whole vector when ``c_raw[0]`` disagrees with
    ``reference_sign``. Index 0 is the reference determinant by convention.

    Returns ``(c_norm, proj_mass)``. ``proj_mass = sum(c_raw**2)`` is
    ``1/Z`` (times the basis-completeness fraction). When ``proj_mass`` is
    below ``1e-30`` the input is returned unchanged.
    """
    proj_mass = float(np.sum(np.asarray(c_raw) ** 2))
    if proj_mass < 1e-30:
        return np.asarray(c_raw).copy(), proj_mass
    c_norm = np.asarray(c_raw) / np.sqrt(proj_mass)
    if reference_sign != 0.0 and c_norm[0] * reference_sign < 0:
        c_norm = -c_norm
    return c_norm, proj_mass


def lasso_recover_auto(
    c_raw: np.ndarray,
    m2: np.ndarray,
    K_s: int,
    n_det: int,
    reference_sign: float,
    lam_mult: float = 0.5,
) -> Tuple[np.ndarray, dict]:
    """End-to-end identity-design Lasso recovery with auto-λ.

    Pipeline: normalize_and_align → universal-threshold soft-threshold
    (Donoho–Johnstone, scaled by ``lam_mult``) with reference coefficient
    preserved un-thresholded → renormalize to unit L₂ → sign re-align.

    The threshold uses per-coefficient noise estimated from the second
    moments: σ_per = √((m2 - c_raw²)/K_s) in the raw (sample-mean) scale,
    rescaled to the normalized (unit L₂) scale by /√proj_mass_naive.
    The universal threshold is λ_u = median(σ_norm) · √(2 log n_det); we
    use λ = ``lam_mult`` · λ_u (empirically ``lam_mult ≈ 0.5`` balances
    support recovery vs noise for multireference systems).

    Returns ``(c_hat, info)`` where ``info`` carries diagnostics for
    downstream reporting (proj_mass naive/bias-corrected, σ_median,
    λ used, support, max|c|).
    """
    proj_mass_naive = float(np.sum(np.asarray(c_raw) ** 2))
    proj_mass_corr = bias_corrected_proj_mass(c_raw, m2, K_s)
    if proj_mass_naive < 1e-30:
        return np.asarray(c_raw).copy(), dict(
            proj_mass_naive=proj_mass_naive,
            proj_mass_bias_corrected=proj_mass_corr,
            sigma_median=float("nan"), lam_universal=0.0, lam_used=0.0,
            lam_mult=float(lam_mult), support=0, max_abs=0.0,
            renorm_after_threshold=0.0,
        )
    c_norm, _ = normalize_and_align(c_raw, reference_sign)

    sample_var_raw = np.maximum(np.asarray(m2) - np.asarray(c_raw) ** 2, 0.0)
    sigma_per_raw = np.sqrt(sample_var_raw / max(K_s, 1))
    sigma_norm = sigma_per_raw / np.sqrt(proj_mass_naive)
    sigma_med = float(np.median(sigma_norm))
    lam_universal = sigma_med * np.sqrt(2.0 * np.log(max(int(n_det), 2)))
    lam = float(lam_mult) * lam_universal

    c_hat = soft_threshold(c_norm, lam)
    c_hat[0] = c_norm[0]
    pre_norm = float(np.linalg.norm(c_hat))
    if pre_norm > 0:
        c_hat = c_hat / pre_norm
    if reference_sign != 0.0 and c_hat[0] * reference_sign < 0:
        c_hat = -c_hat

    info = dict(
        proj_mass_naive=float(proj_mass_naive),
        proj_mass_bias_corrected=float(proj_mass_corr),
        sigma_median=sigma_med,
        lam_universal=float(lam_universal),
        lam_used=float(lam),
        lam_mult=float(lam_mult),
        support=int(np.sum(np.abs(c_hat) > 1e-12)),
        max_abs=float(np.max(np.abs(c_hat))) if c_hat.size else 0.0,
        renorm_after_threshold=pre_norm,
    )
    return c_hat, info


def estimate_ci(
    f_I: np.ndarray,
    lam: float,
    reference_idx: int = 0,
) -> np.ndarray:
    """Sample mean + soft-threshold. The reference coefficient is preserved."""
    means = f_I.mean(axis=1)
    out = soft_threshold(means, lam)
    out[reference_idx] = means[reference_idx]
    return out


def lambda_cv(
    f_I: np.ndarray,
    lambdas: np.ndarray,
    reference_idx: int = 0,
    n_folds: int = 5,
    rng_seed: int = 0,
) -> Tuple[float, np.ndarray]:
    """5-fold CV selecting lambda by held-out mean-squared residual."""
    n_det, K_s = f_I.shape
    if K_s < n_folds:
        raise ValueError(f"K_s={K_s} smaller than n_folds={n_folds}")
    rng = np.random.default_rng(rng_seed)
    perm = rng.permutation(K_s)
    fold_size = K_s // n_folds
    losses = np.zeros((len(lambdas), n_folds))
    for fold in range(n_folds):
        held = perm[fold * fold_size:(fold + 1) * fold_size]
        train = np.setdiff1d(perm, held, assume_unique=False)
        train_means = f_I[:, train].mean(axis=1)
        held_block = f_I[:, held]
        for i, lam in enumerate(lambdas):
            c_hat = soft_threshold(train_means, float(lam))
            c_hat[reference_idx] = train_means[reference_idx]
            residual = held_block - c_hat[:, None]
            losses[i, fold] = float(np.mean(residual ** 2))
    mean_loss = losses.mean(axis=1)
    best = int(np.argmin(mean_loss))
    return float(lambdas[best]), mean_loss


def recovery_metrics(
    c_hat: np.ndarray,
    c_true: np.ndarray,
    eta: float,
) -> dict:
    """L_infinity, L_2, and symmetric-support-difference error."""
    diff = np.asarray(c_hat) - np.asarray(c_true)
    L_inf = float(np.max(np.abs(diff)))
    L_2 = float(np.linalg.norm(diff))
    true_support = np.abs(c_true) > eta
    hat_support = np.asarray(c_hat) != 0
    support_err = float(np.mean(true_support != hat_support))
    return dict(L_inf_err=L_inf, L_2_err=L_2, support_err=support_err)
