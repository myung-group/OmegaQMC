"""Singlet-adapted (closed-shell) QED-RPA and polaritonic QED-BSE@GW.

Spin-adapted counterparts of :mod:`OmegaQMC.addons.qed_rpa`,
:mod:`OmegaQMC.addons.qed_gw` (evGW) and
:mod:`OmegaQMC.addons.qed_bse_polaritonic` for restricted (closed-shell)
QED-HF references. For a singlet absorption spectrum only the
singlet-coupled block of the spin-orbital excitation space matters; the
spin sum is carried out analytically:

* electronic kernels become ``2*(Coulomb) - (exchange)`` over *spatial*
  ov pairs (the triplet block, ``-(exchange)`` only, is also built for
  the RPA correlation energy: the spin-orbital spectrum is the singlet
  block plus three degenerate triplet copies);
* the cavity photon couples only to the singlet channel, with the
  spin-summed vertex ``g^S_ai = -sqrt(2) * sqrt(w/2) * d_ai``;
* the GW screening (QED-dRPA) is the charge channel: the spatial
  transition amplitudes enter the screening tensor as
  ``M_pq,s = sqrt(2) * [ (pq|ai) + d_pq d_ai ] (X+Y)_ai,s + K^phot``,
  and the static W keeps its spin-free density-density form.

All two-electron quantities are assembled from a 3-index DF factor
(``qedhf['eri_df']`` or an exact eigen-factorisation of the dense AO
ERI), so no nmo^4 tensor is ever formed. Matrix dimensions are the
spatial N_ov (+1 photon) instead of the spin-orbital 4*N_ov: for
naphthalene/cc-pVDZ the full non-TDA polaritonic problem drops from an
infeasible ~40 GB working set to ~3 GB.

Validation: eigenvalues/oscillator strengths/photon weights reproduce
the spin-orbital modules exactly on closed-shell references (the
spin-orbital spectrum contains the singlet spectrum; bright roots and
the RPA correlation energy match to machine precision) — see
tools/qed_ccsd_df_derivation/validate_polariton_singlet.py.
"""

import math

import numpy as np
import scipy.linalg as la

from opt_einsum import contract

from .qed_hf import run_qed_hf
from .qed_gw import _solve_qed_rpa_eigensystem, _eval_sigma, _solve_qp_newton
from .qed_ccsd_utils import _eigh_factor_ao

EV = 27.211386245988


# ---------------------------------------------------------------------------
# Spatial (closed-shell) reference quantities
# ---------------------------------------------------------------------------
def _build_spatial_quantities(qedhf):
    """Spatial-orbital Fock, cavity dipole and bare DF factor from a
    restricted QED-HF reference. The DSE is NOT folded into the DF
    factor — it enters the RPA/GW/BSE kernels explicitly through the
    d x d products, exactly as in the spin-orbital modules."""
    if 'Ca' in qedhf:
        raise NotImplementedError(
            "qed_polariton_singlet requires a restricted (closed-shell) "
            "QED-HF reference; use the spin-orbital modules for QED-UHF")
    C = np.asarray(qedhf['C'])
    F_sf = C.T @ np.asarray(qedhf['F']) @ C
    d_sf = -(C.T @ np.asarray(qedhf['dipole_x_lambda_tot']) @ C)
    if 'eri_df' in qedhf:
        B_ao = qedhf['eri_df']
    else:
        B_ao = _eigh_factor_ao(qedhf['eri_ao'])
    B_mo = contract('pi,Ppq,qj->Pij', C, B_ao, C)
    return {
        'F': F_sf,
        'd': d_sf,
        'B': np.ascontiguousarray(B_mo),
        'nocc': qedhf['nocc_spatial'],
        'nmo': qedhf['nmo_spatial'],
        'omega_cav': qedhf['omega'],
    }


def _ov_blocks(sq):
    """Chemist blocks needed by the RPA/BSE kernels (no nmo^4 tensor):
    v_ovov[i,a,j,b] = (ia|jb) and v_oovv[i,j,a,b] = (ij|ab)."""
    B = sq['B']
    no = sq['nocc']
    o = slice(None, no)
    v = slice(no, None)
    v_ovov = contract('xia,xjb->iajb', B[:, o, v], B[:, o, v])
    v_oovv = contract('xij,xab->ijab', B[:, o, o], B[:, v, v])
    return v_ovov, v_oovv


def _osc_strengths_singlet(qedhf, XpY, Omega, nocc):
    """Length-gauge oscillator strengths from singlet-normalised (X+Y):
    f = (2/3) Omega * 2 * |sum_ai mu_ai (X+Y)_ai|^2 (the factor 2 is the
    spin sum (sqrt2)^2)."""
    C = np.asarray(qedhf['C'])
    f = np.zeros_like(Omega)
    for key in ('mu_x_ao', 'mu_y_ao', 'mu_z_ao'):
        mu = C.T @ np.asarray(qedhf[key]) @ C
        mu_n = np.einsum('ai,ais->s', mu[nocc:, :nocc], XpY)
        f += (2.0 / 3.0) * Omega * 2.0 * mu_n ** 2
    return f


# ---------------------------------------------------------------------------
# Singlet-adapted QED-RPA (spectra + correlation energy)
# ---------------------------------------------------------------------------
def run_qed_rpa_singlet(qedhf, direct=False, verbose=True):
    """Singlet-adapted QED-RPA on a closed-shell QED-HF reference.

    The (spatial) singlet block carries the photon; the triplet block is
    photon-free and enters the correlation energy with multiplicity 3:

        E_c = 1/2 [ (Tr O^S + 3 Tr O^T) - (Tr O~^S + 3 Tr O~^T) ].

    Args:
        direct: False (default) -> full QED-RPA kernels (with exchange);
            True -> QED-dRPA (ring channel only; triplet block is bare).

    Returns:
        dict with 'Omega' / 'f_osc' / 'photon_weight' (singlet+photon
        branch, sorted), 'Omega_triplet', 'E_qed_rpa_corr',
        'E_qed_rpa_total'.
    """
    sq = _build_spatial_quantities(qedhf)
    omega_cav = sq['omega_cav']
    no = sq['nocc']
    nmo = sq['nmo']
    nv = nmo - no
    nov = no * nv
    eps = np.diag(sq['F'])
    d = sq['d']
    d_vo = d[no:, :no]
    v_ovov, v_oovv = _ov_blocks(sq)

    dE = np.diag((eps[no:, None] - eps[None, :no]).reshape(-1))
    coul_A = v_ovov.transpose(1, 0, 3, 2).reshape(nov, nov)   # (ia|bj)->(a,i,b,j)
    exch_A = v_oovv.transpose(2, 0, 3, 1).reshape(nov, nov)   # (ij|ab)->(a,i,b,j)
    coul_B = coul_A                                           # (ia|jb) same layout
    exch_B = v_ovov.transpose(3, 0, 1, 2).reshape(nov, nov)   # (ib|ja)->(a,i,b,j)

    dd_tt = np.einsum('ai,bj->aibj', d_vo, d_vo).reshape(nov, nov)
    if direct:
        A_S = dE + 2.0 * coul_A + 2.0 * dd_tt
        B_S = 2.0 * coul_B + 2.0 * dd_tt
        A_T = dE.copy()
        B_T = np.zeros_like(dE)
    else:
        dd_A = np.einsum('ab,ij->aibj', d[no:, no:],
                         d[:no, :no]).reshape(nov, nov)
        dd_B = np.einsum('aj,ib->aibj', d_vo,
                         d[:no, no:]).reshape(nov, nov)
        A_S = dE + 2.0 * coul_A - exch_A + 2.0 * dd_tt - dd_A
        B_S = 2.0 * coul_B - exch_B + 2.0 * dd_tt - dd_B
        A_T = dE - exch_A - dd_A
        B_T = -exch_B - dd_B

    # photon-augmented singlet blocks; spin-summed vertex sqrt(2) g
    g_vec = -math.sqrt(2.0) * math.sqrt(omega_cav / 2.0) * d_vo.reshape(nov)
    dim = nov + 1
    A_big = np.zeros((dim, dim))
    B_big = np.zeros((dim, dim))
    A_big[:nov, :nov] = A_S
    A_big[:nov, -1] = A_big[-1, :nov] = g_vec
    A_big[-1, -1] = omega_cav
    B_big[:nov, :nov] = B_S
    B_big[:nov, -1] = B_big[-1, :nov] = g_vec

    Omega_bar_S = la.eigh(A_big, eigvals_only=True)
    Omega_bar_T = la.eigh(A_T, eigvals_only=True)

    Omega_S, U, V = _solve_qed_rpa_eigensystem(A_big, B_big)
    # The triplet block is needed only for the correlation energy. Full
    # RPA/TDHF triplet instabilities are common for closed-shell pi
    # systems (acenes!) and are a property of the electronic reference,
    # not of the cavity coupling — they must not abort the (stable)
    # singlet absorption calculation.
    triplet_stable = True
    if np.allclose(B_T, 0.0):
        Omega_T = np.sort(Omega_bar_T)
    else:
        try:
            Omega_T, _, _ = _solve_qed_rpa_eigensystem(A_T, B_T)
        except RuntimeError:
            triplet_stable = False
            Omega_T = np.full_like(Omega_bar_T, np.nan)
            if verbose:
                print("  warning: triplet RPA instability of the "
                      "closed-shell reference — the RPA correlation "
                      "energy is undefined (NaN); the singlet spectrum "
                      "below is unaffected.")

    if triplet_stable:
        E_c = 0.5 * ((np.sum(Omega_S) + 3.0 * np.sum(Omega_T))
                     - (np.sum(Omega_bar_S) + 3.0 * np.sum(Omega_bar_T)))
    else:
        E_c = float('nan')
    E_qedhf = qedhf['E_qed_hf']

    order = np.argsort(Omega_S)
    Omega_S = Omega_S[order]
    UpV = (U + V)[:, order]
    XpY = UpV[:nov, :].reshape(nv, no, -1)
    ph_amp = UpV[-1, :]
    f_osc = _osc_strengths_singlet(qedhf, XpY, Omega_S, no)

    if verbose:
        flavour = 'QED-dRPA' if direct else 'QED-RPA'
        print(f"\n{flavour} (singlet-adapted, closed shell):")
        print(f"  nocc = {no}, nvir = {nv} (spatial), N_ov+1 = {dim},"
              f" w_cav = {omega_cav:.6f} Ha")
        print(f"  E_corr   = {float(E_c):.12f}")
        print(f"  E_total  = {E_qedhf + float(E_c):.12f}")
        bright = np.where(f_osc > 1e-3)[0]
        print("   Omega(eV)   f_osc     photon_wt")
        for i in bright[:10]:
            print(f"   {Omega_S[i] * EV:8.3f}   {f_osc[i]:8.4f}"
                  f"  {ph_amp[i] ** 2:8.3f}")

    return {
        'E_qed_rpa_corr': float(E_c),
        'E_qed_rpa_total': float(E_qedhf + E_c),
        'Omega': Omega_S,
        'f_osc': f_osc,
        'photon_weight': ph_amp ** 2,
        'Omega_triplet': np.sort(Omega_T),
        'Omega_tda_rwa': Omega_bar_S,
        'Omega_tda_rwa_triplet': Omega_bar_T,
        'triplet_stable': bool(triplet_stable),
        'direct': bool(direct),
    }


# ---------------------------------------------------------------------------
# Singlet-adapted QED-GW (dRPA screening, charge channel)
# ---------------------------------------------------------------------------
def _screening_at_eps(sq, eps, include_dse=True, include_photon=True):
    """Solve the singlet QED-dRPA screening problem at the given orbital
    energies and return (Omega, M_full) with the spatial screening tensor
    M[p,q,s] (the sqrt(2) spin sum folded in)."""
    omega_cav = sq['omega_cav']
    no = sq['nocc']
    nmo = sq['nmo']
    nv = nmo - no
    nov = no * nv
    B = sq['B']
    d = sq['d']
    d_vo = d[no:, :no]
    o = slice(None, no)
    v = slice(no, None)

    B_ov = B[:, o, v]
    coul = contract('xia,xjb->aibj', B_ov, B_ov).reshape(nov, nov)
    dd = np.einsum('ai,bj->aibj', d_vo, d_vo).reshape(nov, nov)
    AB_common = 2.0 * coul + (2.0 * dd if include_dse else 0.0)

    dE = np.diag((eps[no:, None] - eps[None, :no]).reshape(-1))
    if include_photon:
        g_vec = -math.sqrt(2.0 * omega_cav / 2.0) * d_vo.reshape(nov)
        dim = nov + 1
    else:
        g_vec = None
        dim = nov
    A_big = np.zeros((dim, dim))
    B_big = np.zeros((dim, dim))
    A_big[:nov, :nov] = dE + AB_common
    B_big[:nov, :nov] = AB_common
    if include_photon:
        A_big[:nov, -1] = A_big[-1, :nov] = g_vec
        A_big[-1, -1] = omega_cav
        B_big[:nov, -1] = B_big[-1, :nov] = g_vec

    Omega, U, V = _solve_qed_rpa_eigensystem(A_big, B_big)
    UpV = U + V
    XpY_e = UpV[:nov, :]                                   # (nov, s)

    # M[p,q,s] = sqrt(2) [ (pq|ai) + d_pq d_ai ] (X+Y)_ai,s + K^phot d_pq
    t_x = B_ov.reshape(-1, nov) @ np.ascontiguousarray(
        XpY_e.reshape(nv, no, -1).transpose(1, 0, 2)).reshape(nov, -1)
    M_full = math.sqrt(2.0) * contract('xpq,xs->pqs', B, t_x)
    if include_dse:
        w_d = d_vo.reshape(nov) @ XpY_e
        M_full += math.sqrt(2.0) * np.einsum('pq,s->pqs', d, w_d)
    if include_photon:
        XpY_p = UpV[-1, :]
        M_full += np.einsum(
            'pq,s->pqs', -math.sqrt(omega_cav / 2.0) * d, XpY_p)
    return Omega, M_full


def run_qed_gw_singlet(qedhf, mode='evGW', eta=1e-3, max_iter=50, tol=1e-7,
                       inner_max=50, inner_tol=1e-9, verbose=True):
    """Singlet-adapted (spatial) QED-GW quasiparticle energies with
    QED-dRPA screening. 'evGW' iterates all spatial orbitals to
    self-consistency in the spectrum; 'G0W0' does one-shot Newton solves
    for all orbitals. Returns eps_QP for all spatial orbitals."""
    sq = _build_spatial_quantities(qedhf)
    no = sq['nocc']
    nmo = sq['nmo']
    eps_HF = np.diag(sq['F']).copy()

    if verbose:
        print(f"\n{mode}@QED-dRPA@QED-HF (singlet-adapted)")
        print(f"  nocc = {no}, nmo = {nmo} (spatial), "
              f"w_cav = {sq['omega_cav']:.6f} Ha, eta = {eta:.1e}")

    eps_cur = eps_HF.copy()
    n_outer = max_iter if mode == 'evGW' else 1
    Omega = M_full = None
    for outer in range(1, n_outer + 1):
        Omega, M_full = _screening_at_eps(sq, eps_cur)
        eps_new = np.empty_like(eps_cur)
        for p in range(nmo):
            eps_new[p], _, _ = _solve_qp_newton(
                eps_HF[p], M_full[p], Omega, eps_cur, no, eta,
                omega_start=eps_cur[p], max_iter=inner_max, tol=inner_tol)
        delta = float(np.max(np.abs(eps_new - eps_cur)))
        eps_cur = eps_new
        if verbose:
            print(f"    iter {outer:3d}:  max |d eps^GW| = {delta:.3e}")
        if mode != 'evGW' or delta < tol:
            break

    if verbose:
        print(f"  HOMO: {eps_HF[no-1]:+.6f} -> {eps_cur[no-1]:+.6f} Ha,  "
              f"LUMO: {eps_HF[no]:+.6f} -> {eps_cur[no]:+.6f} Ha")
    return {
        'method': mode,
        'eps_HF': eps_HF,
        'eps_QP': eps_cur.copy(),
        'Omega_RPA': Omega,
        'eta': float(eta),
    }


# ---------------------------------------------------------------------------
# Singlet-adapted polaritonic QED-BSE@GW
# ---------------------------------------------------------------------------
def run_qed_bse_polaritonic_singlet(qedhf, gw_mode='evGW', tda=False,
                                    eta=1e-3, eps_QP=None, verbose=True):
    """Singlet-adapted polaritonic QED-BSE@GW (excitons + LP/UP).

    Mirrors :func:`qed_bse_polaritonic.run_qed_bse_polaritonic` in the
    singlet channel: electronic BSE blocks with purely electronic dRPA
    screening (photon NOT in W — it appears once, explicitly, through
    the sqrt(2)-vertex row/column), DSE kept as a genuine electronic
    interaction.

    Args:
        eps_QP: optional spatial QP energies (skips the GW step).

    Returns:
        dict with 'Omega', 'f_osc', 'photon_weight', 'omega_cav',
        'eps_QP', 'tda'.
    """
    sq = _build_spatial_quantities(qedhf)
    omega_cav = sq['omega_cav']
    no = sq['nocc']
    nmo = sq['nmo']
    nv = nmo - no
    nov = no * nv
    d = sq['d']
    d_vo = d[no:, :no]

    if eps_QP is None:
        gw = run_qed_gw_singlet(qedhf, mode=gw_mode, eta=eta,
                                verbose=verbose)
        eps_QP = gw['eps_QP']
    else:
        eps_QP = np.asarray(eps_QP, dtype=float).copy()
        if verbose:
            print("\npol-BSE (singlet): using caller-supplied QP energies"
                  " (GW step skipped).")

    # Electronic screening only (photon excluded from W), DSE kept.
    Omega_scr, M_full = _screening_at_eps(sq, eps_QP, include_dse=True,
                                          include_photon=False)
    inv_Om = 1.0 / Omega_scr
    M_oo = M_full[:no, :no, :]
    M_vv = M_full[no:, no:, :]
    M_ov = M_full[:no, no:, :]
    M_vo = M_full[no:, :no, :]

    v_ovov, v_oovv = _ov_blocks(sq)

    # W^stat_{ij,ab} = (ij|ab) + d_ij d_ab - 2 sum_s M_ij,s M_ab,s / O_s
    W_ijab = (v_oovv
              + np.einsum('ij,ab->ijab', d[:no, :no], d[no:, no:])
              - 2.0 * contract('ijs,abs,s->ijab', M_oo, M_vv, inv_Om))
    # W^stat_{ib,aj} = (ib|aj) + d_ib d_aj - 2 sum_s M_ib,s M_aj,s / O_s
    # (ib|aj) as an (i,b,a,j) tensor is v_ovov[i,b,j,a] -> transpose(0,1,3,2)
    W_ibaj = (v_ovov.transpose(0, 1, 3, 2)
              + np.einsum('ib,aj->ibaj', d[:no, no:], d_vo)
              - 2.0 * contract('ibs,ajs,s->ibaj', M_ov, M_vo, inv_Om))

    dd_tt = np.einsum('ai,bj->aibj', d_vo, d_vo)
    # A^BSE,S = dE^QP + 2[(ai|jb) + d d] - W_{ij,ab}
    A_e = (2.0 * (v_ovov.transpose(1, 0, 3, 2) + dd_tt)
           - W_ijab.transpose(2, 0, 3, 1)).reshape(nov, nov)
    A_e += np.diag((eps_QP[no:, None] - eps_QP[None, :no]).reshape(-1))
    A_e = 0.5 * (A_e + A_e.T)
    if not tda:
        # B^BSE,S = 2[(ai|bj) + d d] - W_{ib,aj}
        B_e = (2.0 * (v_ovov.transpose(1, 0, 3, 2) + dd_tt)
               - W_ibaj.transpose(2, 0, 1, 3)).reshape(nov, nov)
        B_e = 0.5 * (B_e + B_e.T)

    g_vec = -math.sqrt(2.0 * omega_cav / 2.0) * d_vo.reshape(nov)
    dim = nov + 1
    A_big = np.zeros((dim, dim))
    A_big[:nov, :nov] = A_e
    A_big[:nov, -1] = A_big[-1, :nov] = g_vec
    A_big[-1, -1] = omega_cav
    if tda:
        Omega, vecs = la.eigh(A_big)
        XpY_e = vecs[:nov, :]
        ph_amp = vecs[-1, :]
        method = 'pol-TDA-BSE (singlet)'
    else:
        B_big = np.zeros((dim, dim))
        B_big[:nov, :nov] = B_e
        B_big[:nov, -1] = B_big[-1, :nov] = g_vec
        Omega, U, V = _solve_qed_rpa_eigensystem(A_big, B_big)
        UpV = U + V
        XpY_e = UpV[:nov, :]
        ph_amp = UpV[-1, :]
        method = 'pol-BSE (singlet)'

    order = np.argsort(Omega)
    Omega = Omega[order]
    XpY_e = XpY_e[:, order]
    ph_amp = ph_amp[order]

    f_osc = _osc_strengths_singlet(
        qedhf, XpY_e.reshape(nv, no, -1), Omega, no)

    if verbose:
        print(f"\n{method}@QED-{gw_mode}@QED-HF   "
              f"(photon explicit, not in W)")
        print(f"  nocc = {no}, nmo = {nmo} (spatial), dim = {dim}, "
              f"w_cav = {omega_cav * EV:.3f} eV")
        print("   Omega(eV)   f_osc     photon_wt")
        bright = np.where(f_osc > 1e-3)[0]
        for i in bright[:12]:
            print(f"   {Omega[i] * EV:8.3f}   {f_osc[i]:8.4f}  "
                  f"{ph_amp[i] ** 2:8.3f}")

    return {
        'method': method,
        'tda': bool(tda),
        'gw_mode': gw_mode,
        'Omega': Omega,
        'f_osc': f_osc,
        'photon_weight': ph_amp ** 2,
        'omega_cav': float(omega_cav),
        'eps_QP': eps_QP,
        'nov': nov,
    }


if __name__ == '__main__':
    from pyscf import gto
    mol = gto.M(atom="O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692",
                basis='6-31g', verbose=0)
    omega = 0.4
    qedhf = run_qed_hf(mol, omega, (0.0, 0.0, 0.05), verbose=False)
    run_qed_rpa_singlet(qedhf, direct=False, verbose=True)
    run_qed_bse_polaritonic_singlet(qedhf, gw_mode='evGW', verbose=True)
