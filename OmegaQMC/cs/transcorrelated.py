"""
Transcorrelated (TC) decode: reading a trained NN-VMC trial wavefunction
out as the biorthogonal CI vectors of a transcorrelated Hamiltonian.

Motivation
----------
The plain CS recovery primitive (:mod:`OmegaQMC.cs.estimators`) projects the
*full* trial Psi_NN = Phi_det * e^{J} onto Slater determinants. Because the
Jastrow e^{J} carries the electron-electron cusp, Psi_NN has slow Gaussian-
basis convergence and the recovered c-hat leaks diffuse mass onto high-virtual
determinants (the "basis-mismatch" limitation). Removing the cusp before
projection fixes this: with the multiplicative correlation factor J the
similarity-transformed Hamiltonian

    H~ = e^{-J} H e^{J}

has the same spectrum as H, and its *right* eigenvector is the smooth

    Phi_R = e^{-J} Psi_NN ,

while the right eigenvector of H~^dagger = e^{+J} H e^{-J} is

    Phi_L = e^{+J} Psi_NN ,

with the *exact* biorthogonal normalisation <Phi_L|Phi_R> = <Psi_NN|Psi_NN> = Z.
Both are decoded by the same sample-mean primitive with one extra factor:

    f_I^(tau)(R) = D_I(R) e^{-tau J(R)} / Psi_NN(R),
    E_{|Psi_NN|^2/Z}[f_I^(tau)] = <D_I | e^{-tau J} Psi_NN> / Z,

so tau = +1 -> c_R (right TC CI vector), tau = -1 -> c_L (left), and tau = 0
recovers exactly the Hermitian decode of :mod:`estimators` (a regression
anchor). Only J evaluated *pointwise* on walkers is needed here -- no TC
integrals. The integral-bearing stages (TC energy, TC-NEVPT2) live elsewhere.

This module provides the single-knob estimator, the two-sided decode, and the
diagnostics that gate the whole TC program: the leakage-mass collapse
M_spurious(0) -> M_spurious(1) and the biorthogonal overlap (a conditioning /
health metric for the downstream non-Hermitian problem).
"""

from typing import Optional, Sequence, Tuple

import numpy as np

from .estimators import f_I_matrix, lasso_recover_auto, normalize_and_align


def f_I_matrix_tc(
    orb_vals: np.ndarray,
    candidate_set: Sequence,
    psi_vals: np.ndarray,
    jastrow_vals: np.ndarray,
    tau: float,
    n_alpha: int,
    n_beta: int,
    use_jit: bool = True,
) -> np.ndarray:
    """TC single-knob estimator ``f_I^(tau)(R_k) = f_I(R_k) * exp(-tau J_k)``.

    ``jastrow_vals`` is ``J(R_k)`` (the *log* of the multiplicative
    correlation factor, i.e. ``jastrow_term`` in the ansatz, such that
    ``Psi_NN = exp(ln_slater + J)``), shape ``(K_s,)``. Returns the
    ``(n_det, K_s)`` matrix; ``tau = 0`` reproduces :func:`f_I_matrix`
    bit-for-bit.
    """
    f = f_I_matrix(orb_vals, candidate_set, psi_vals, n_alpha, n_beta,
                   use_jit=use_jit)
    J = np.asarray(jastrow_vals, dtype=float)
    if J.shape[0] != f.shape[1]:
        raise ValueError(
            f"jastrow_vals length {J.shape[0]} != K_s {f.shape[1]}"
        )
    if tau == 0.0:
        return f
    return f * np.exp(-float(tau) * J)[None, :]


def _means_and_m2(f: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    return f.mean(axis=1), (f ** 2).mean(axis=1)


def decode_single(
    orb_vals: np.ndarray,
    candidate_set: Sequence,
    psi_vals: np.ndarray,
    jastrow_vals: np.ndarray,
    tau: float,
    n_alpha: int,
    n_beta: int,
    reference_sign: float = 1.0,
    lam_mult: float = 0.5,
    use_lasso: bool = True,
) -> Tuple[np.ndarray, dict]:
    """One TC decode at a given ``tau``.

    Returns ``(c_hat, info)``. With ``use_lasso`` the auto-threshold Lasso
    of :func:`estimators.lasso_recover_auto` is applied (noise estimated
    from the *TC* second moments, so the threshold is correct for this
    estimator); otherwise a plain L2 normalise + sign align is used.
    """
    f = f_I_matrix_tc(orb_vals, candidate_set, psi_vals, jastrow_vals, tau,
                      n_alpha, n_beta)
    means, m2 = _means_and_m2(f)
    K_s = f.shape[1]
    n_det = len(candidate_set)
    if use_lasso:
        c_hat, info = lasso_recover_auto(
            means, m2, K_s, n_det, reference_sign, lam_mult=lam_mult,
        )
    else:
        c_hat, proj_mass = normalize_and_align(means, reference_sign)
        info = dict(proj_mass_naive=float(proj_mass), lam_used=0.0,
                    support=int(np.sum(np.abs(c_hat) > 1e-12)))
    info["tau"] = float(tau)
    info["c_raw"] = means
    info["m2"] = m2
    return c_hat, info


def decode_two_sided(
    orb_vals: np.ndarray,
    candidate_set: Sequence,
    psi_vals: np.ndarray,
    jastrow_vals: np.ndarray,
    n_alpha: int,
    n_beta: int,
    reference_sign: float = 1.0,
    lam_mult: float = 0.5,
    use_lasso: bool = True,
) -> dict:
    """Full two-sided TC decode: Hermitian (tau=0), right (tau=+1), left
    (tau=-1).

    Returns a dict with ``c_herm``, ``c_R``, ``c_L`` (each L2-unit, sign
    aligned), the per-tau ``info`` blocks, and the biorthogonal overlap
    ``rho`` between ``c_L`` and ``c_R``.
    """
    c_herm, info0 = decode_single(
        orb_vals, candidate_set, psi_vals, jastrow_vals, 0.0,
        n_alpha, n_beta, reference_sign, lam_mult, use_lasso)
    c_R, infoR = decode_single(
        orb_vals, candidate_set, psi_vals, jastrow_vals, +1.0,
        n_alpha, n_beta, reference_sign, lam_mult, use_lasso)
    c_L, infoL = decode_single(
        orb_vals, candidate_set, psi_vals, jastrow_vals, -1.0,
        n_alpha, n_beta, reference_sign, lam_mult, use_lasso)
    return dict(
        c_herm=c_herm, c_R=c_R, c_L=c_L,
        info_herm=info0, info_R=infoR, info_L=infoL,
        rho=biorthogonal_overlap(c_L, c_R),
    )


def biorthogonal_overlap(c_L: np.ndarray, c_R: np.ndarray) -> float:
    """Normalised left-right overlap ``(c_L . c_R)/(||c_L|| ||c_R||)``.

    The conditioning metric for the non-Hermitian TC problem: values near
    +/-1 are healthy; values near zero signal a near-defective projected
    operator whose biorthogonal RDMs / energies are ill-conditioned. The
    TC analogue of the CAS match-overlap figure of merit.
    """
    a = np.asarray(c_L, dtype=float)
    b = np.asarray(c_R, dtype=float)
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-30 or nb < 1e-30:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def leakage_mass(
    c_hat: np.ndarray,
    c_ref: np.ndarray,
    thresh: float = 1e-3,
) -> float:
    """Spurious squared mass: ``sum_{|c_ref_I| < thresh} |c_hat_I|^2``.

    With an L2-unit ``c_hat`` this is the fraction of recovered weight that
    falls on determinants the reference (FCI / TC-FCI) calls negligible --
    the basis-incompleteness leakage. The headline Gate-0 number is the
    drop from ``leakage_mass(c_herm, ...)`` to ``leakage_mass(c_R, ...)``.
    """
    c_hat = np.asarray(c_hat, dtype=float)
    c_ref = np.asarray(c_ref, dtype=float)
    if c_hat.shape != c_ref.shape:
        raise ValueError(
            f"c_hat shape {c_hat.shape} != c_ref shape {c_ref.shape}"
        )
    mask = np.abs(c_ref) < thresh
    return float(np.sum(c_hat[mask] ** 2))


def gate0_metrics(
    candidate: Sequence,
    c_true: np.ndarray,
    decode_out: dict,
    leak_thresh: float = 1e-3,
) -> dict:
    """Scalar Gate-0 metrics from a :func:`decode_two_sided` result.

    Returns leakage(tau=0) vs leakage(tau=+1), the relative drop, the
    biorthogonal overlap, and the c-vs-FCI overlaps for the Hermitian and
    right-TC decodes. ``c_true`` is the FCI (or TC-FCI) reference vector
    aligned to ``candidate``.
    """
    c_true = np.asarray(c_true, dtype=float)
    c_herm = np.asarray(decode_out["c_herm"], dtype=float)
    c_R = np.asarray(decode_out["c_R"], dtype=float)

    def _ov(a, b):
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        return float(np.dot(a, b) / (na * nb)) if na > 0 and nb > 0 else 0.0

    leak0 = leakage_mass(c_herm, c_true, leak_thresh)
    leak1 = leakage_mass(c_R, c_true, leak_thresh)
    drop = (1.0 - leak1 / leak0) if leak0 > 0 else float("nan")
    return dict(
        leak_herm=leak0, leak_R=leak1, leak_drop_frac=drop,
        rho=float(decode_out["rho"]),
        overlap_herm=_ov(c_herm, c_true), overlap_R=_ov(c_R, c_true),
        gate_pass=bool(leak0 > 0 and leak1 < 0.5 * leak0
                       and abs(decode_out["rho"]) > 0.5),
    )


def print_gate0_report(
    candidate: Sequence,
    c_true: np.ndarray,
    decode_out: dict,
    leak_thresh: float = 1e-3,
    top_k: int = 8,
) -> dict:
    """Print the Gate-0 decision artifact and return :func:`gate0_metrics`."""
    m = gate0_metrics(candidate, c_true, decode_out, leak_thresh)
    c_true = np.asarray(c_true, dtype=float)
    c_herm = np.asarray(decode_out["c_herm"], dtype=float)
    c_R = np.asarray(decode_out["c_R"], dtype=float)

    print("\n=== Gate-0 summary ===")
    print(f"  leakage M_spurious(tau=0, Hermitian) = {m['leak_herm']:.4f}")
    print(f"  leakage M_spurious(tau=+1, TC right) = {m['leak_R']:.4f}")
    drop_pct = m["leak_drop_frac"] * 100.0
    print(f"  leakage drop                         = {drop_pct:+.1f} %")
    print(f"  biorthogonal overlap rho             = {m['rho']:+.4f}")
    print(f"  overlap <c_herm|c_FCI>               = {m['overlap_herm']:+.4f}")
    print(f"  overlap <c_R   |c_FCI>               = {m['overlap_R']:+.4f}")

    order = np.argsort(-np.abs(c_true))[:top_k]
    print(f"\n  top-{top_k} determinants (by |c_FCI|):")
    print(f"    {'det':>26s} {'c_FCI':>10s} {'c_herm':>10s} {'c_R':>10s}")
    for i in order:
        print(f"    {str(candidate[i]):>26s} {c_true[i]:+10.4f} "
              f"{c_herm[i]:+10.4f} {c_R[i]:+10.4f}")

    verdict = "PASS" if m["gate_pass"] else "INSPECT"
    print(f"\n  GATE-0: {verdict} (leakage halved & rho well-conditioned)")
    return m


def biorthogonal_1rdm(
    c_L: np.ndarray,
    c_R: np.ndarray,
    candidate_set: Sequence,
    n_orb: int,
    nelec: Tuple[int, int],
) -> np.ndarray:
    """Biorthogonal spin-summed 1-RDM ``gamma_pq = <Phi_L|a_p^dag a_q|Phi_R>``.

    Generalises :func:`properties._compute_1rdm_sparse` to a left/right pair,
    normalised by ``c_L . c_R`` (the biorthogonal metric). For a Hermitian
    state ``c_L == c_R`` this reduces to the standard 1-RDM. ``gamma`` is
    not symmetric in general; downstream natural-orbital frames use its
    symmetric part ``(gamma + gamma.T)/2``.
    """
    from .properties import _excite_sign

    n_alpha, n_beta = int(nelec[0]), int(nelec[1])
    cL = np.asarray(c_L, dtype=float)
    cR = np.asarray(c_R, dtype=float)
    denom = float(np.dot(cL, cR))
    if abs(denom) < 1e-30:
        raise ValueError(
            "biorthogonal overlap c_L.c_R is ~0: near-defective TC pair, "
            "1-RDM is ill-conditioned"
        )
    gamma = np.zeros((n_orb, n_orb), dtype=float)

    cand = [(tuple(int(x) for x in occ_a), tuple(int(x) for x in occ_b))
            for occ_a, occ_b in candidate_set]
    idx_of = {key: i for i, key in enumerate(cand)}

    for I, (occ_a, occ_b) in enumerate(cand):
        cL_I = cL[I]
        if cL_I == 0.0:
            continue
        # Diagonal: <D_I| n_p |D_I> contributes cL_I * cR_I.
        diag = cL_I * cR[I]
        if diag != 0.0:
            for p in occ_a:
                gamma[p, p] += diag
            for p in occ_b:
                gamma[p, p] += diag
        # Alpha single excitations p (in I) -> q (in J).
        occ_a_set = set(occ_a)
        for p in occ_a:
            for q in range(n_orb):
                if q in occ_a_set:
                    continue
                new_occ_a = tuple(sorted(
                    [orb for orb in occ_a if orb != p] + [q]
                ))
                J = idx_of.get((new_occ_a, occ_b))
                if J is None:
                    continue
                cR_J = cR[J]
                if cR_J == 0.0:
                    continue
                gamma[p, q] += _excite_sign(occ_a, p, q) * cL_I * cR_J
        # Beta single excitations.
        occ_b_set = set(occ_b)
        for p in occ_b:
            for q in range(n_orb):
                if q in occ_b_set:
                    continue
                new_occ_b = tuple(sorted(
                    [orb for orb in occ_b if orb != p] + [q]
                ))
                J = idx_of.get((occ_a, new_occ_b))
                if J is None:
                    continue
                cR_J = cR[J]
                if cR_J == 0.0:
                    continue
                gamma[p, q] += _excite_sign(occ_b, p, q) * cL_I * cR_J
    return gamma / denom
