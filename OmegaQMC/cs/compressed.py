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
