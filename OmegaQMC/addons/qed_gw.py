"""
Cavity QED extension of the G0W0 / evGW quasiparticle method on top of
a QED-HF reference.

The standard (non-cavity) GW equations follow Marie, Ammar, Loos,
"The GW Approximation: A Quantum Chemistry Perspective"
(arXiv:2311.05351, J. Chem. Theory Comput. 2024). The QED extension
re-uses the QED-dRPA spectrum built by :mod:`OmegaQMC.addons.qed_rpa`:
the non-Hermitian RPA matrix is diagonalised for both eigenvalues and
symplectic-normalised eigenvectors (Eq. 21–22 of the GW paper, with the
photon-augmented A, B blocks of arXiv:2602.09968 Eq. 6 / 34).

Working equations (paper equation numbers in parentheses):

* RPA   (Eq. 23a-b, 22a)

    A_{ia,jb} = (ε_a − ε_i) δ_ij δ_ab + ⟨ib|aj⟩
    B_{ia,jb} = ⟨ij|ab⟩                              (physicist, direct)
    X^T X − Y^T Y = 1

  augmented with the photon row/column (g_{ai} = −√(ω_cav/2) d_{ai})
  and the DSE blocks d_{pq} d_{rs} of QED-dRPA.

* Screening matrix elements (Eq. 29 of the GW paper)

    M_{pq,m} = Σ_{jb} ⟨pj|qb⟩ (X+Y)_{jb,m}

  ⟨pj|qb⟩ is the standard physicist 2-electron integral, equal to the
  chemist (pq|jb). For real spin orbitals this is the same as
  ⟨pb|qj⟩, which is what the code computes by slicing g_phys to
  (p, b∈vir, q, j∈occ) and transposing. The QED extension adds the
  DSE direct term d_{pq} d_{ai} and the photon-mediated coupling
  −√(ω_cav/2) d_{pq} contracted with (M+N)_m.

* Correlation self-energy (Eq. 33)

    Σ_c,{pp}(ω) =  Σ_{i,m} M^2_{pi,m} / (ω − ε_i + Ω_m − iη)
                + Σ_{a,m} M^2_{pa,m} / (ω − ε_a − Ω_m + iη)

  − iη on the occupied branch, + iη on the virtual branch (paper sign
  convention).

Three modes are exposed through ``mode``:

* ``'linG0W0'`` (default) — paper Eq. 37 / Fig. 3 "linG0W0@HF":

    ε^QP_p = ε^HF_p + Z_p Re Σ_c(ε^HF_p),
    Z_p^{-1} = 1 − ∂_ω Σ_c |_{ω = ε^HF_p}  (Eq. 38)

* ``'G0W0'`` — paper Fig. 3 "G0W0@HF": Newton iteration on the
  unlinearised QP equation ω = ε^HF_p + Σ_c(ω), single shot (RPA
  spectrum is *not* iterated, only ω is).

* ``'evGW'`` — paper Fig. 3 "evGW@HF": outer self-consistency loop.
  In each outer iteration, the RPA matrices A, B are rebuilt with the
  current ε^GW (the orbital energies; orbitals themselves are fixed),
  the RPA is re-diagonalised, M is recomputed, and the non-linear QP
  equation ω = ε^HF_p + Σ_c(ω; ε^GW, Ω^GW, M^GW) is solved by Newton
  for every orbital. Converges when max|Δε^GW| < tol.

Defaults to HOMO and LUMO (spatial-orbital indexing). The
result-dict's ``eps_QP[homo]`` and ``eps_QP[lumo]`` then give the
vertical IP = −ε^QP_HOMO and EA = −ε^QP_LUMO.
"""

import math
import warnings

import numpy as np
import scipy.linalg as la
from pyscf import gto

from .qed_hf import run_qed_hf
from .qed_rpa import _build_spin_orbital_quantities


def _solve_qed_rpa_eigensystem(A_big, B_big):
    """Diagonalise M = [[A,B],[−B,−A]] and return the positive-Ω modes
    with symplectic-normalised (U, V) blocks (Eq. 22a: U^T U − V^T V = 1)."""
    dim = A_big.shape[0]
    M = np.zeros((2 * dim, 2 * dim))
    M[:dim, :dim] = A_big
    M[:dim, dim:] = B_big
    M[dim:, :dim] = -B_big
    M[dim:, dim:] = -A_big

    evals, evecs = la.eig(M)
    real_evals = evals.real

    pos_idx = np.argsort(real_evals)[dim:]
    Omega = real_evals[pos_idx]

    V_full = evecs[:, pos_idx].real
    U_block = V_full[:dim, :]
    V_block = V_full[dim:, :]

    norm_sq = (U_block * U_block).sum(axis=0) - (V_block * V_block).sum(axis=0)
    if np.any(norm_sq <= 0):
        bad = int(np.sum(norm_sq <= 0))
        raise RuntimeError(
            f"QED-RPA reference appears unstable: {bad} of {dim} modes have "
            f"non-positive symplectic norm. Try a smaller coupling λ."
        )
    norm = np.sqrt(norm_sq)
    return Omega, U_block / norm, V_block / norm


def _build_static_quantities(qedhf, direct=True):
    """Quantities that don't change between evGW iterations: the AO→MO
    integrals, the bare K interaction in the photon-augmented basis,
    and the HF reference energies."""
    omega_cav = qedhf['omega']
    so = _build_spin_orbital_quantities(qedhf)
    d_so = so['d_so']
    g_phys = so['g_phys_d'] if direct else so['g_phys_a']
    nocc, nso = so['nocc'], so['nso']

    # K_{pm,ai} = (pm|ai)_chem + DSE; (pm|ai)_chem = ⟨pa|mi⟩_phys.
    # g_phys[:, nocc:, :, :nocc] has axes (p, a, m, i); transpose to (p, m, a, i).
    d_vo = d_so[nocc:, :nocc]
    K_pmai = g_phys[:, nocc:, :, :nocc].transpose(0, 2, 1, 3).copy()
    K_pmai += np.einsum('pm,ai->pmai', d_so, d_vo)
    if not direct:
        K_pmai -= np.einsum('pi,am->pmai', d_so[:, :nocc], d_so[nocc:, :])
    K_pmphot = -math.sqrt(omega_cav / 2.0) * d_so

    return {
        'so': so,
        'K_pmai': K_pmai,
        'K_pmphot': K_pmphot,
        'eps_HF': np.diag(so['F_so']),
        'omega_cav': omega_cav,
        'direct': bool(direct),
    }


def _rpa_at_eps(static, eps_so):
    """Build A_big, B_big from the given orbital energies, diagonalise,
    and return (Ω, M_full) where M_full[p, m, s] is the full screening
    matrix-element tensor used by the self-energy."""
    so = static['so']
    omega_cav = static['omega_cav']
    direct = static['direct']
    d_so = so['d_so']
    g_phys = so['g_phys_d'] if direct else so['g_phys_a']
    nocc, nso = so['nocc'], so['nso']
    nvir = nso - nocc
    nov = nvir * nocc

    eps_occ = eps_so[:nocc]
    eps_vir = eps_so[nocc:]

    # Electronic A, B (paper Eq. 23a-b, physicist non-antisymmetric).
    A_ee = np.diag((eps_vir[:, None] - eps_occ[None, :]).reshape(-1))
    A_ee = A_ee + (g_phys[:nocc, nocc:, nocc:, :nocc]
                   .transpose(2, 0, 1, 3).reshape(nov, nov))
    B_ee = (g_phys[:nocc, :nocc, nocc:, nocc:]
            .transpose(2, 0, 3, 1).reshape(nov, nov))

    # DSE blocks (QED extension).
    d_vo = d_so[nocc:, :nocc]
    if direct:
        Delta = np.einsum('ai,bj->aibj', d_vo, d_vo).reshape(nov, nov)
        Delta_prime = Delta
    else:
        d_ov = d_so[:nocc, nocc:]
        d_oo = d_so[:nocc, :nocc]
        d_vv = d_so[nocc:, nocc:]
        Delta = (np.einsum('ai,jb->aibj', d_vo, d_ov)
                 - np.einsum('ab,ij->aibj', d_vv, d_oo)).reshape(nov, nov)
        Delta_prime = (np.einsum('ai,bj->aibj', d_vo, d_vo)
                       - np.einsum('aj,ib->aibj', d_vo, d_ov)).reshape(nov, nov)

    g_vec = -math.sqrt(omega_cav / 2.0) * d_vo.reshape(nov)
    dim = nov + 1
    A_big = np.zeros((dim, dim))
    B_big = np.zeros((dim, dim))
    A_big[:nov, :nov] = A_ee + Delta
    A_big[:nov, -1] = g_vec
    A_big[-1, :nov] = g_vec
    A_big[-1, -1] = omega_cav
    B_big[:nov, :nov] = B_ee + Delta_prime
    B_big[:nov, -1] = g_vec
    B_big[-1, :nov] = g_vec

    Omega, U_block, V_block = _solve_qed_rpa_eigensystem(A_big, B_big)
    UpV = U_block + V_block
    XplusY_e_4d = UpV[:nov, :].reshape(nvir, nocc, -1)
    XplusY_p = UpV[-1, :]

    # M_{p,m,s} for the full p, m range (Eq. 29 + QED additions).
    M_full = (np.einsum('pmai,ais->pms', static['K_pmai'], XplusY_e_4d)
              + np.einsum('pm,s->pms', static['K_pmphot'], XplusY_p))
    return Omega, M_full


def _eval_sigma(omega_val, M_ms, Omega, eps_used, nocc, eta):
    """Return (Σ_pp, dΣ_pp/dω) at scalar ω for one orbital's M_{pm,s}
    (paper Eq. 33 + its analytic ω-derivative)."""
    occ_mask = np.arange(M_ms.shape[0]) < nocc
    eps_m = eps_used[:, None]
    Omega_s = Omega[None, :]

    denom_occ = omega_val - eps_m + Omega_s - 1j * eta
    denom_vir = omega_val - eps_m - Omega_s + 1j * eta
    inv_denom = np.where(occ_mask[:, None], 1.0 / denom_occ, 1.0 / denom_vir)

    M_sq = M_ms * M_ms
    sigma = np.sum(M_sq * inv_denom)
    dsigma = -np.sum(M_sq * inv_denom * inv_denom)
    return complex(sigma), complex(dsigma)


def _solve_qp_newton(eps_HF_p, M_ms, Omega, eps_used, nocc, eta,
                     omega_start, max_iter, tol):
    """Newton solve of  ω − ε^HF_p − Re Σ_c(ω) = 0  (paper Fig. 3)."""
    omega = float(omega_start)
    sigma = 0j
    for it in range(1, max_iter + 1):
        sigma, dsigma = _eval_sigma(omega, M_ms, Omega, eps_used, nocc, eta)
        f = omega - eps_HF_p - sigma.real
        fp = 1.0 - dsigma.real
        delta = -f / fp
        omega += delta
        if abs(delta) < tol:
            return omega, sigma, it
    return omega, sigma, max_iter


def run_qed_gw(qedhf, orbs=None, direct=True, mode='linG0W0', eta=1e-3,
               max_iter=50, tol=1e-7,
               inner_max=50, inner_tol=1e-9, verbose=True):
    """G0W0 / linG0W0 / evGW quasiparticle energies on QED-HF.

    Args:
        qedhf: dict returned by :func:`OmegaQMC.qed_hf.run_qed_hf`.
        orbs: list of *spatial* orbital indices (0-based) to report
            QP energies for. ``None`` → HOMO and LUMO.
        direct: True → QED-dRPA-based W (the standard GW choice — only
            direct Coulomb screening in W, so Σ_c does not contain any
            exchange-line diagrams). False → screen the antisymmetrised
            interaction ⟨pq‖rs⟩, i.e. build W from the full QED-RPA.
            WARNING: ``direct=False`` is exposed for completeness only
            and is NOT a recommended production setting — every loop in
            W then carries a bare-exchange line which, when convolved
            with G in Σ_c, double-counts the Fock exchange already
            sitting inside ε^HF_p (the LHS of the QP equation). The
            resulting quasiparticle energies are unphysical (e.g. IPs
            ≳ 40 eV on water/cc-pVDZ). A proper "GW + exchange in W"
            requires a compensating vertex (GWΓ, SOSEX, BSE-style
            schemes) and is not implemented here.
        mode: one of ``'linG0W0'``, ``'G0W0'``, ``'evGW'``. See module
            docstring for the precise definition of each.
        eta: imaginary regulariser in the Σ_c denominators (Ha).
        max_iter, tol: outer-loop limit / threshold for ``'evGW'``.
        inner_max, inner_tol: Newton-loop limit / threshold for the
            non-linear QP equation in ``'G0W0'`` and ``'evGW'``.
        verbose: print iteration progress and a summary table.

    Returns:
        Dict with ``method``, ``eps_HF`` (Koopmans), ``eps_QP``,
        renormalisation factors ``Z`` (only for ``'linG0W0'``), the
        diagonal ``Sigma_c`` at the QP point, and the final ``Omega_RPA``.
    """
    if mode not in ('linG0W0', 'G0W0', 'evGW'):
        raise ValueError(
            f"mode must be one of 'linG0W0', 'G0W0', 'evGW'; got {mode!r}")
    if not direct:
        warnings.warn(
            "direct=False uses antisymmetric ⟨pq||rs⟩ throughout. Every "
            "loop in W then contains a bare-exchange line, and Σ_c "
            "double-counts the Fock exchange already in ε^HF. QP "
            "energies will be unphysical (e.g. IP ≳ 40 eV on water/"
            "cc-pVDZ). Use direct=True for production GW; the proper "
            "way to put exchange into W requires a vertex correction "
            "(GWΓ / SOSEX / BSE) that is not implemented here.",
            RuntimeWarning,
            stacklevel=2,
        )

    static = _build_static_quantities(qedhf, direct=direct)
    eps_HF = static['eps_HF']
    nocc, nso = static['so']['nocc'], static['so']['nso']
    nocc_spatial = qedhf['nocc_spatial']

    if orbs is None:
        orbs = [nocc_spatial - 1, nocc_spatial]
    orbs = list(orbs)
    orbs_so = [2 * p for p in orbs]
    if any(p_so >= nso for p_so in orbs_so):
        raise ValueError("requested orbital index out of range")

    flavour = 'QED-dRPA' if direct else 'QED-RPA (with exchange)'
    if verbose:
        print(f"\n{mode}@{flavour}@QED-HF")
        print(f"  nocc(SO)={nocc}, nso={nso},"
              f"  ω_cav={static['omega_cav']:.6f} Ha,  η={eta:.1e}")

    if mode in ('linG0W0', 'G0W0'):
        Omega, M_full = _rpa_at_eps(static, eps_HF)
        eps_used = eps_HF
        qp_energies, Z_factors, sigma_vals = {}, {}, {}
        if verbose:
            if mode == 'linG0W0':
                print("\n  spatial orb   ε_HF (Ha)        ε_QP (Ha)       "
                      "Re Σ_c (Ha)    Z")
            else:
                print("\n  spatial orb   ε_HF (Ha)        ε_QP (Ha)       "
                      "Re Σ_c (Ha)   iter")
        for p_spatial, p_so in zip(orbs, orbs_so):
            M_ms = M_full[p_so]
            eps_p = float(eps_HF[p_so])
            if mode == 'linG0W0':
                sigma, dsigma = _eval_sigma(eps_p, M_ms, Omega, eps_used,
                                            nocc, eta)
                Z = 1.0 / (1.0 - dsigma.real)
                eps_qp = eps_p + Z * sigma.real
                Z_factors[p_spatial] = float(Z)
                if verbose:
                    print(f"  {p_spatial:3d}            {eps_p:+13.8f}   "
                          f"{eps_qp:+13.8f}   {sigma.real:+10.6f}   {Z:.4f}")
            else:  # 'G0W0' — Newton on the non-linear QP equation
                eps_qp, sigma, niter = _solve_qp_newton(
                    eps_p, M_ms, Omega, eps_used, nocc, eta,
                    omega_start=eps_p, max_iter=inner_max, tol=inner_tol)
                if verbose:
                    print(f"  {p_spatial:3d}            {eps_p:+13.8f}   "
                          f"{eps_qp:+13.8f}   {sigma.real:+10.6f}   "
                          f"{niter:>3d}")
            qp_energies[p_spatial] = float(eps_qp)
            sigma_vals[p_spatial] = complex(sigma)
        return {
            'method': mode,
            'direct': bool(direct),
            'orbs': orbs,
            'eps_HF': {p: float(eps_HF[2 * p]) for p in orbs},
            'eps_QP': qp_energies,
            'Z': Z_factors,
            'Sigma_c': sigma_vals,
            'Omega_RPA': Omega,
            'eta': float(eta),
        }

    # ---- evGW: outer self-consistency over the RPA spectrum ---------------
    eps_current = eps_HF.copy()
    if verbose:
        print("\n  evGW outer iterations:")
    Omega = None
    M_full = None
    for outer in range(1, max_iter + 1):
        Omega, M_full = _rpa_at_eps(static, eps_current)
        eps_new = np.empty_like(eps_current)
        for p_so in range(nso):
            eps_new[p_so], _, _ = _solve_qp_newton(
                eps_HF[p_so], M_full[p_so], Omega, eps_current, nocc, eta,
                omega_start=eps_current[p_so],
                max_iter=inner_max, tol=inner_tol)
        delta_max = float(np.max(np.abs(eps_new - eps_current)))
        if verbose:
            print(f"    iter {outer:3d}:  max |Δε^GW| = {delta_max:.3e}")
        eps_current = eps_new
        if delta_max < tol:
            break

    # Final reporting at the converged ε^GW.
    qp_energies = {p: float(eps_current[2 * p]) for p in orbs}
    sigma_vals = {}
    if verbose:
        print("\n  spatial orb   ε_HF (Ha)        ε_QP (Ha)       "
              "Re Σ_c (Ha)")
    for p_spatial, p_so in zip(orbs, orbs_so):
        sigma, _ = _eval_sigma(eps_current[p_so], M_full[p_so], Omega,
                               eps_current, nocc, eta)
        sigma_vals[p_spatial] = complex(sigma)
        if verbose:
            print(f"  {p_spatial:3d}            "
                  f"{float(eps_HF[p_so]):+13.8f}   "
                  f"{qp_energies[p_spatial]:+13.8f}   "
                  f"{sigma.real:+10.6f}")
    return {
        'method': mode,
        'direct': bool(direct),
        'orbs': orbs,
        'eps_HF': {p: float(eps_HF[2 * p]) for p in orbs},
        'eps_QP': qp_energies,
        'Z': {},  # no Z factor in evGW
        'Sigma_c': sigma_vals,
        'Omega_RPA': Omega,
        'eta': float(eta),
        'eps_GW_all': eps_current.copy(),
    }


if __name__ == '__main__':
    HARTREE_TO_EV = 27.211386245988

    half_angle = math.radians(104.5 / 2.0)
    rOH = 1.0
    hx = rOH * math.sin(half_angle)
    hz = -rOH * math.cos(half_angle)
    mol = gto.M(
        atom=[
            ['O', (0.0, 0.0, 0.0)],
            ['H', (+hx, 0.0, hz)],
            ['H', (-hx, 0.0, hz)],
        ],
        basis='cc-pVDZ',
        unit='Angstrom',
        symmetry=False,
        verbose=0,
    )

    omega = 0.415668
    lambda_z = 0.05
    lambda_cav = (0.0, 0.0, lambda_z)

    print(f"omega  = {omega:.6f} Ha")
    print(f"lambda = {lambda_cav}")

    qedhf = run_qed_hf(mol, omega, lambda_cav, verbose=False)
    print(f"\nE_QED_HF = {qedhf['E_qed_hf']:.10f}")

    lin = run_qed_gw(qedhf, mode='linG0W0', verbose=True)
    nlg = run_qed_gw(qedhf, mode='G0W0',    verbose=True)
    evg = run_qed_gw(qedhf, mode='evGW',    verbose=True)

    nocc_spatial = qedhf['nocc_spatial']
    homo, lumo = nocc_spatial - 1, nocc_spatial
    print("\nVertical IP and EA (eV, IP=−ε_HOMO, EA=−ε_LUMO):")
    print(f"  Koopmans (QED-HF): IP = {-lin['eps_HF'][homo]*HARTREE_TO_EV:7.3f}"
          f"   EA = {-lin['eps_HF'][lumo]*HARTREE_TO_EV:7.3f}")
    print(f"  linG0W0@QED-dRPA:  IP = {-lin['eps_QP'][homo]*HARTREE_TO_EV:7.3f}"
          f"   EA = {-lin['eps_QP'][lumo]*HARTREE_TO_EV:7.3f}")
    print(f"  G0W0@QED-dRPA:     IP = {-nlg['eps_QP'][homo]*HARTREE_TO_EV:7.3f}"
          f"   EA = {-nlg['eps_QP'][lumo]*HARTREE_TO_EV:7.3f}")
    print(f"  evGW@QED-dRPA:     IP = {-evg['eps_QP'][homo]*HARTREE_TO_EV:7.3f}"
          f"   EA = {-evg['eps_QP'][lumo]*HARTREE_TO_EV:7.3f}")
