"""Compressed-sensing CI vector recovery via random orbital rotations.

The canonical recovery primitive in this paper (one estimator per
coefficient ĉ_I = ⟨D_I/Ψ_NN⟩) has an identity design matrix --
mathematically a Lasso special case, not "compressed sensing" in the
Candès-Tao sense (no m ≪ N reduction in measurements).

This module implements a TRUE compressed-sensing variant: pick m ≪
N_det random orthogonal orbital rotations U_j; for each, sample-mean
a SINGLE rotated HF determinant D_HF^(U_j)/Ψ_NN; reconstruct the CI
vector via L1 minimisation under a non-trivial design matrix
A_{j,I} = ⟨D_HF^(U_j)|D_I⟩.

Sample-complexity scaling becomes canonical CS:
  m ≳ C · K_eff · log(N_det / K_eff)

Computational scaling: walker-determinant evaluations drop from
O(K_s · N_det) for the identity-design scheme to O(K_s · m) for the
random-rotation scheme. The L1 recovery step now requires a proper
Lasso solver (FISTA / coordinate descent) rather than a closed-form
soft-thresholding, but is one-shot post-processing on the
(m × K_s) measurement matrix.

Implementation: pure NumPy / scikit-learn for the L1 solver.
"""

from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np


def random_orthogonal(n: int, rng: np.random.Generator) -> np.ndarray:
    """Uniformly random n×n orthogonal matrix via QR of a Gaussian matrix."""
    G = rng.standard_normal((n, n))
    Q, R = np.linalg.qr(G)
    # Fix sign so the distribution is uniform on O(n) (Mezzadri 2007).
    d = np.sign(np.diag(R))
    d[d == 0] = 1
    Q = Q * d[None, :]
    return Q


def sample_rotations(
    n_orb: int,
    m: int,
    rng: np.random.Generator | int = 0,
) -> np.ndarray:
    """Stack of m random orthogonal n_orb×n_orb matrices.

    Returns ``(m, n_orb, n_orb)``.
    """
    if isinstance(rng, int):
        rng = np.random.default_rng(rng)
    return np.stack([random_orthogonal(n_orb, rng) for _ in range(m)])


def design_matrix(
    U_stack: np.ndarray,
    candidate_set: Sequence[Tuple[Tuple[int, ...], Tuple[int, ...]]],
    n_alpha: int,
    n_beta: int,
) -> np.ndarray:
    """Build the compressed-sensing design matrix A.

    ``A[j, I] = ⟨D_HF^(U_j) | D_I⟩``

    For each rotation U_j, the rotated HF determinant has its alpha
    electrons in the first n_alpha rotated orbitals and beta electrons
    in the first n_beta. The overlap with a determinant D_I that
    occupies orbital indices ``occ_I^α`` (alpha) and ``occ_I^β``
    (beta) is the product of two small determinants of submatrices of
    U_j.

    Args:
      U_stack: (m, n_orb, n_orb) stack of orthogonal rotations.
      candidate_set: list of (occ_alpha, occ_beta) tuples.
      n_alpha, n_beta: alpha and beta electron counts.

    Returns:
      A: (m, n_det) ndarray.
    """
    m = U_stack.shape[0]
    n_det = len(candidate_set)
    A = np.zeros((m, n_det), dtype=np.float64)
    for j in range(m):
        U = U_stack[j]
        for I_idx, (occ_a, occ_b) in enumerate(candidate_set):
            rows_a = np.asarray(occ_a, dtype=int)
            rows_b = np.asarray(occ_b, dtype=int)
            sub_a = U[rows_a, :n_alpha]
            sub_b = U[rows_b, :n_beta]
            det_a = np.linalg.det(sub_a) if rows_a.size > 0 else 1.0
            det_b = np.linalg.det(sub_b) if rows_b.size > 0 else 1.0
            A[j, I_idx] = det_a * det_b
    return A


def design_column(
    U_stack: np.ndarray,
    occ_alpha: Tuple[int, ...],
    occ_beta: Tuple[int, ...],
    n_alpha: int,
    n_beta: int,
) -> np.ndarray:
    """One column of the CS design matrix:  A[:, I] = ⟨D_HF^(U_j) | D_I⟩.

    Returns ``(m,)`` array.
    """
    m = U_stack.shape[0]
    rows_a = np.asarray(occ_alpha, dtype=int)
    rows_b = np.asarray(occ_beta, dtype=int)
    out = np.zeros(m, dtype=np.float64)
    for j in range(m):
        U = U_stack[j]
        sub_a = U[rows_a, :n_alpha]
        sub_b = U[rows_b, :n_beta]
        det_a = np.linalg.det(sub_a) if rows_a.size > 0 else 1.0
        det_b = np.linalg.det(sub_b) if rows_b.size > 0 else 1.0
        out[j] = det_a * det_b
    return out


def omp_recover(
    y: np.ndarray,
    U_stack: np.ndarray,
    candidate_pool: Sequence[Tuple[Tuple[int, ...], Tuple[int, ...]]],
    n_alpha: int,
    n_beta: int,
    initial_support: Sequence = None,
    max_support: int = None,
    rel_residual_tol: float = 1e-3,
    abs_residual_tol: float = 0.0,
    return_history: bool = False,
):
    """Orthogonal matching pursuit on the random-rotation CS measurements.

    Grows the support adaptively, starting from ``initial_support`` (default
    HF reference, ``[((0..n_alpha-1), (0..n_beta-1))]``), by repeatedly
    adding the determinant in ``candidate_pool`` whose design-matrix
    column has highest absolute correlation with the current residual.
    After each addition the coefficients on the support are re-fit by
    least squares; the residual is updated.

    Stops when either
      (a) ``len(support) == max_support``  (default min(|pool|, m)),
      (b) ``residual_norm < abs_residual_tol``,
      (c) ``residual_norm / initial_residual_norm < rel_residual_tol``,
      (d) no remaining candidate has a correlation > 1e-12.

    The OMP variant is genuinely candidate-set-free at the "I'll try
    this pool" level: the support grows from one determinant. The pool
    only delimits which determinants OMP is allowed to consider; the
    output support is typically a tiny subset.

    Args:
      y: (m,) measurement vector.
      U_stack: (m, n_orb, n_orb) the rotations used to produce y.
      candidate_pool: list of (occ_alpha, occ_beta) tuples to consider.
      n_alpha, n_beta: electron counts.
      initial_support: starting support; default is just the HF reference.
      max_support: maximum support size; default min(|pool|, m).
      rel_residual_tol: stop when residual / initial < this fraction.
      abs_residual_tol: stop when residual < this absolute value.
      return_history: also return per-iteration diagnostics.

    Returns:
      support: list of selected (occ_alpha, occ_beta) tuples.
      coeffs: (|support|,) ndarray of fitted coefficients (un-normalised).
      history: optional list of per-iteration dicts.
    """
    m = U_stack.shape[0]
    pool = list(candidate_pool)
    n_pool = len(pool)
    if max_support is None:
        max_support = min(n_pool, m)

    if initial_support is None:
        hf_a = tuple(range(n_alpha))
        hf_b = tuple(range(n_beta))
        initial_support = [(hf_a, hf_b)]
    initial_support = [tuple((tuple(I[0]), tuple(I[1])))
                       for I in initial_support]

    # Precompute design columns for the entire pool (O(|pool|·m·N^3) work,
    # one-shot). For very large pools this would benefit from chunking.
    cols = np.empty((n_pool, m), dtype=np.float64)
    pool_index = {}
    for k, I in enumerate(pool):
        cols[k] = design_column(U_stack, I[0], I[1], n_alpha, n_beta)
        pool_index[(tuple(I[0]), tuple(I[1]))] = k

    # Track support: list of pool indices
    support_idx = []
    for I in initial_support:
        if I in pool_index:
            support_idx.append(pool_index[I])

    # Initial least-squares fit on the seed support
    A_S = cols[support_idx].T   # (m, |S|)
    coeffs, *_ = np.linalg.lstsq(A_S, y, rcond=None)
    r = y - A_S @ coeffs
    r0 = float(np.linalg.norm(y))
    in_support = np.zeros(n_pool, dtype=bool)
    in_support[support_idx] = True

    history = [dict(iter=0, support_size=len(support_idx),
                    residual_norm=float(np.linalg.norm(r)))]

    for it in range(1, max_support + 1):
        # Correlations of all out-of-support columns with the residual
        corr = cols @ r                    # (n_pool,)
        corr[in_support] = 0.0
        best_k = int(np.argmax(np.abs(corr)))
        best_corr = float(abs(corr[best_k]))
        if best_corr < 1e-12:
            break

        support_idx.append(best_k)
        in_support[best_k] = True

        A_S = cols[support_idx].T          # (m, |S|)
        coeffs, *_ = np.linalg.lstsq(A_S, y, rcond=None)
        r = y - A_S @ coeffs
        res_norm = float(np.linalg.norm(r))

        history.append(dict(iter=it, support_size=len(support_idx),
                            residual_norm=res_norm,
                            best_corr=best_corr,
                            added_det=pool[best_k]))

        if res_norm < abs_residual_tol:
            break
        if r0 > 0 and res_norm / r0 < rel_residual_tol:
            break

    support = [pool[k] for k in support_idx]
    if return_history:
        return support, coeffs, history
    return support, coeffs


def lasso_recover(
    A: np.ndarray,
    y: np.ndarray,
    lam: float = 1e-3,
    max_iter: int = 5000,
    tol: float = 1e-6,
) -> np.ndarray:
    """L1-regularised recovery:  ĉ = argmin_c  ½‖y - Ac‖² + λ‖c‖₁.

    Uses scikit-learn's coordinate-descent Lasso (closed form per
    coordinate but iterated; standard for non-identity design).

    Returns the *unnormalised* solution; caller should L2-normalise
    if desired.
    """
    from sklearn.linear_model import Lasso

    clf = Lasso(alpha=lam, fit_intercept=False, max_iter=max_iter,
                tol=tol, selection="cyclic")
    clf.fit(A, y)
    return np.asarray(clf.coef_, dtype=np.float64)


def evaluate_rotated_hf_on_walkers(
    mol,
    walkers: np.ndarray,
    orbital_coeff_ao: np.ndarray,
    U_stack: np.ndarray,
    n_alpha: int,
    n_beta: int,
    walker_convention: str = "interleaved",
) -> np.ndarray:
    """Compute D_HF^(U_j)(R_k) for every walker k and every rotation j.

    Returns ``(m, K_s)`` array. Each entry is a SINGLE Slater
    determinant (alpha block × beta block) in the rotation-rotated
    basis.

    Args:
      mol: PySCF mole.
      walkers: (K_s, N_electrons, 3) walker positions.
      orbital_coeff_ao: (n_AO, n_orb) base orbital coefficients
                       (HF MOs or NN-NOs in AO basis).
      U_stack: (m, n_orb, n_orb) stack of orthogonal rotations.
      n_alpha, n_beta: electron counts.
      walker_convention: 'interleaved' (NN-VMC default) or 'grouped'.

    Implementation: compute orbital values at every electron position
    once (in the BASE orbital basis), then apply U_j as a small matrix
    multiplication to get rotated orbital values for each j.
    """
    K_s, N, _ = walkers.shape

    # Evaluate base orbitals at every electron position
    flat = np.asarray(walkers, dtype=np.float64).reshape(-1, 3)
    ao_vals = mol.eval_gto("GTOval_sph", flat)         # (K_s*N, n_AO)
    base_vals = ao_vals @ np.asarray(orbital_coeff_ao)  # (K_s*N, n_orb)
    base_vals = base_vals.reshape(K_s, N, -1)           # (K_s, N, n_orb)

    if walker_convention == "interleaved":
        # alpha = even indices, beta = odd indices
        idx_a = np.arange(0, n_alpha * 2, 2)
        idx_b = np.arange(1, n_beta * 2 + 1, 2)
        base_a = base_vals[:, idx_a, :]   # (K_s, n_alpha, n_orb)
        base_b = base_vals[:, idx_b, :]   # (K_s, n_beta, n_orb)
    elif walker_convention == "grouped":
        base_a = base_vals[:, :n_alpha, :]
        base_b = base_vals[:, n_alpha:n_alpha + n_beta, :]
    else:
        raise ValueError(f"unknown walker convention {walker_convention!r}")

    m = U_stack.shape[0]
    # For each rotation, compute the determinant of the rotated alpha
    # block and rotated beta block.
    # Rotated alpha orbitals at walker positions:
    #   base_a @ U[:, :n_alpha] -> (K_s, n_alpha, n_alpha)
    # Determinant is then computed over the last two axes.
    D = np.zeros((m, K_s), dtype=np.float64)
    import math
    n_norm = 1.0 / np.sqrt(
        float(math.factorial(n_alpha) * math.factorial(n_beta))
    )
    for j in range(m):
        U = U_stack[j]
        rot_a = base_a @ U[:, :n_alpha]   # (K_s, n_alpha, n_alpha)
        rot_b = base_b @ U[:, :n_beta]    # (K_s, n_beta, n_beta)
        det_a = np.linalg.det(rot_a)
        det_b = np.linalg.det(rot_b)
        D[j] = n_norm * det_a * det_b
    return D


def omp_decode(
    mol,
    walkers: np.ndarray,
    psi_vals: np.ndarray,
    orbital_coeff_ao: np.ndarray,
    candidate_pool: Sequence[Tuple[Tuple[int, ...], Tuple[int, ...]]],
    n_alpha: int,
    n_beta: int,
    m: int = None,
    rel_residual_tol: float = 1e-3,
    max_support: int = None,
    seed: int = 0,
    walker_convention: str = "interleaved",
    return_diagnostics: bool = False,
):
    """End-to-end OMP recovery: walker bank → support → coefficients.

    The OMP decoder discovers the determinant support adaptively from a
    candidate pool, starting from the HF reference. Unlike the Lasso CS
    variant (which requires the candidate set in advance), OMP needs
    only a "pool" of determinants it is *allowed* to consider; the
    output support is typically much smaller than the pool.

    Returns: support (list of (occ_a, occ_b) tuples), coeffs (ndarray
    on the support, L2-normalised so ||coeffs||₂ = 1 unless support is
    empty).
    """
    n_pool = len(candidate_pool)
    if m is None:
        m = min(n_pool, max(50, int(4 * n_pool ** (2.0 / 3))))
    if max_support is None:
        max_support = min(n_pool, m)

    rng = np.random.default_rng(seed)
    n_orb = int(orbital_coeff_ao.shape[1])
    U_stack = sample_rotations(n_orb, m, rng)

    D = evaluate_rotated_hf_on_walkers(
        mol, walkers, orbital_coeff_ao, U_stack,
        n_alpha, n_beta, walker_convention=walker_convention,
    )
    f = D / psi_vals[None, :]
    y = f.mean(axis=1)

    support, coeffs, history = omp_recover(
        y, U_stack, candidate_pool, n_alpha, n_beta,
        max_support=max_support, rel_residual_tol=rel_residual_tol,
        return_history=True,
    )

    nrm = float(np.linalg.norm(coeffs))
    if nrm > 0:
        coeffs = coeffs / nrm

    if return_diagnostics:
        diag = dict(
            m=m, support_size=len(support),
            initial_residual_norm=float(np.linalg.norm(y)),
            final_residual_norm=history[-1]["residual_norm"],
            iterations=len(history) - 1,
            history=history,
        )
        return support, coeffs, diag
    return support, coeffs


def compressed_sensing_decode(
    mol,
    walkers: np.ndarray,
    psi_vals: np.ndarray,
    orbital_coeff_ao: np.ndarray,
    candidate_set: Sequence[Tuple[Tuple[int, ...], Tuple[int, ...]]],
    n_alpha: int,
    n_beta: int,
    m: int = None,
    lam: float = 1e-3,
    seed: int = 0,
    walker_convention: str = "interleaved",
    return_diagnostics: bool = False,
) -> np.ndarray:
    """End-to-end CS recovery via random orbital rotations.

    Workflow:
      1. Sample m random orthogonal U_j.
      2. Compute D_HF^(U_j)(R_k) at every walker, for each j.
      3. Sample-mean y_j = ⟨D_HF^(U_j)/Ψ_NN⟩.
      4. Build design matrix A_{j,I} = ⟨D_HF^(U_j)|D_I⟩.
      5. Lasso recovery ĉ = argmin ½‖y - Ac‖² + λ‖c‖₁.
      6. L2 normalise.

    If ``m`` is None, defaults to ``min(n_det, max(50, 4·n_det^(2/3)))``
    -- a heuristic that grows sublinearly in n_det, matching the
    CS sample-complexity intuition.
    """
    n_det = len(candidate_set)
    if m is None:
        m = min(n_det, max(50, int(4 * n_det ** (2.0 / 3))))

    rng = np.random.default_rng(seed)
    n_orb = int(orbital_coeff_ao.shape[1])
    U_stack = sample_rotations(n_orb, m, rng)

    # 2-3. Measurements
    D = evaluate_rotated_hf_on_walkers(
        mol, walkers, orbital_coeff_ao, U_stack,
        n_alpha, n_beta, walker_convention=walker_convention,
    )
    f = D / psi_vals[None, :]                         # (m, K_s)
    y = f.mean(axis=1)                                # (m,)

    # 4. Design matrix
    A = design_matrix(U_stack, candidate_set, n_alpha, n_beta)

    # 5. Lasso. Auto-pick lambda if lam < 0: scale to per-coefficient
    # noise estimate. Otherwise treat lam as a fraction of max|A.T y|
    # for scale-invariance (sklearn Lasso uses this convention).
    if lam < 0:
        lam_max = float(np.max(np.abs(A.T @ y))) / max(1, m)
        lam = abs(lam) * lam_max
    c_raw = lasso_recover(A, y, lam=lam)

    # 6. L2 normalise
    nrm = np.linalg.norm(c_raw)
    if nrm == 0:
        c_hat = c_raw
    else:
        c_hat = c_raw / nrm

    if not return_diagnostics:
        return c_hat
    diag = dict(
        m=m, lam=lam, A_norm=float(np.linalg.norm(A)),
        y_norm=float(np.linalg.norm(y)),
        c_raw_norm=float(nrm),
        n_kept=int(np.sum(np.abs(c_raw) > 0)),
        n_det=n_det,
    )
    return c_hat, diag
