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

The GW step additionally offers an auxiliary-basis dielectric backend
(``screening='aux-cd' | 'aux-pade'``) that never diagonalises the
nov-dimensional QED-dRPA problem: the screened interaction is obtained
from (naux+1)-dimensional dielectric inversions on an imaginary-
frequency grid — O(N^4) time and O(naux*nov) memory — with the cavity
photon folded in exactly as a frequency-dependent weight on a single
extra auxiliary vector. The BSE static W has the matching backend
(``w_screening='aux'``): one exact nu = 0 Woodbury inversion replaces
the nov-dim eigensolve (identical W to machine precision). Finally,
``solver='davidson'`` makes the BSE itself matrix-free: a paired
(symplectic) Davidson driven by O(naux*nov*nmo) matvecs converges the
lowest polariton roots without ever forming the (nov+1)-dim blocks.
See the "Auxiliary-basis dielectric" section below and
tools/qed_ccsd_df_derivation/validate_gw_aux_screening.py.
"""

import math

import numpy as np
import scipy.linalg as la
from scipy.interpolate import CubicSpline

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


# ---------------------------------------------------------------------------
# Auxiliary-basis dielectric: QED-dRPA screening without the nov-dim diag
# ---------------------------------------------------------------------------
# With the direct (ring) kernel the singlet QED-dRPA problem has A - B =
# diag(dE_ai, w_cav): the photon coupling g and the kernel appear in both A
# and B and cancel. The screened-interaction matrix elements therefore reduce
# to a resolvent of M = dE^2 + dE^{1/2} K dE^{1/2} whose kernel K is low rank:
# naux Coulomb channels (2 B_P) plus ONE cavity channel (2 d_ai). Folding the
# photon row/column out exactly gives the cavity channel the frequency-
# dependent weight
#
#     w_d(nu) = [dse] + [photon] * w_cav^2 / (nu^2 - w_cav^2)
#
# (with both channels on, w_d = nu^2/(nu^2 - w_cav^2): DSE and photon
# screening cancel exactly at nu = 0). By Woodbury, evaluating
#
#     Wt_pm(nu) = sum_s M_pms^2 * 2 Omega_s / (nu^2 - Omega_s^2)
#
# (the correlation part of the screened interaction; M_pms is exactly the
# tensor built by :func:`_screening_at_eps`) needs only the (naux+1)^2
# dielectric matrix at each frequency — O(naux^2 nov + naux^3) per point,
# O(N^4) overall — instead of the O(nov^3) eigensolve. The self-energy is
# then obtained by contour deformation (CD; exact up to quadrature) or by
# Pade analytic continuation (AC; no residue evaluations, right choice for
# full-spectrum evGW). Validated against the dense sum-over-poles path in
# tools/qed_ccsd_df_derivation/validate_gw_aux_screening.py.
def _imag_freq_grid(n, scale):
    """Gauss-Legendre grid mapped to (0, inf): w' = scale*(1+x)/(1-x)."""
    x, w = np.polynomial.legendre.leggauss(n)
    freqs = scale * (1.0 + x) / (1.0 - x)
    wts = w * 2.0 * scale / (1.0 - x) ** 2
    return freqs, wts


def _aux_setup(sq, eps, include_dse=True, include_photon=True):
    """Frequency-independent ingredients of the aux-basis screening:
    the ov-space channel vectors b (naux Coulomb rows + one cavity dipole
    row), the full-MO vertex factors rho, and the bare gaps dE_ia."""
    no, nmo = sq['nocc'], sq['nmo']
    B, d = sq['B'], sq['d']
    dE = (eps[None, no:] - eps[:no, None]).reshape(-1)      # (i,a) order
    b = np.ascontiguousarray(B[:, :no, no:]).reshape(-1, dE.size)
    use_d = include_dse or include_photon
    if use_d:
        b = np.vstack([b, d[:no, no:].reshape(1, -1)])
        rho = np.concatenate([B, d[None]], axis=0)
    else:
        rho = B
    return {
        'b': b, 'rho': rho, 'dE': dE, 'd_mo': d,
        'no': no, 'nmo': nmo, 'use_d': use_d,
        'omega_cav': sq['omega_cav'],
        'include_dse': bool(include_dse),
        'include_photon': bool(include_photon),
    }


def _aux_wtilde_core(setup, nu2):
    """Effective screening matrix at (complex) squared frequency nu2:
    Wt_pm(nu) = rho_pm^T Lam_eff(nu) rho_pm + c_bare(nu) d_pm^2, with
    Lam_eff = 2 s [pi + 4 pi (1 - 4 W pi)^{-1} W pi] s via Woodbury
    (pi_XY = sum_ia b_X dE/(nu2 - dE^2) b_Y, s = sqrt(2) W)."""
    b, dE = setup['b'], setup['dE']
    x = dE / (nu2 - dE * dE)
    pi = (b * x) @ b.T
    nch = pi.shape[0]
    w = np.ones(nch, dtype=pi.dtype)
    c_bare = 0.0
    if setup['use_d']:
        w_d = 1.0 if setup['include_dse'] else 0.0
        if setup['include_photon']:
            wc = setup['omega_cav']
            c_bare = wc * wc / (nu2 - wc * wc)
            w_d = w_d + c_bare
        w[-1] = w_d
    wpi = w[:, None] * pi
    Q = np.linalg.solve(np.eye(nch, dtype=pi.dtype) - 4.0 * wpi, wpi)
    lam = 2.0 * (pi + 4.0 * pi @ Q)
    s = math.sqrt(2.0) * w
    return s[:, None] * lam * s[None, :], c_bare


def _aux_wtilde_grid(setup, freqs):
    """Wt_pm(i w'_k) for all (p, m) on the imaginary-frequency grid.
    Purely real arithmetic; O(naux^2 nov) per point."""
    rho = setup['rho']
    nmo = setup['nmo']
    R = rho.reshape(rho.shape[0], -1)
    d_sq = setup['d_mo'] ** 2
    Wt = np.empty((len(freqs), nmo, nmo))
    for k, wpr in enumerate(freqs):
        lam_eff, c_bare = _aux_wtilde_core(setup, -wpr * wpr)
        Wt[k] = ((lam_eff @ R) * R).sum(axis=0).reshape(nmo, nmo)
        if setup['include_photon']:
            Wt[k] += c_bare * d_sq
    return Wt


def _aux_wtilde_pt(setup, nu2, p, m):
    """Wt_pm at a single (complex) squared frequency — one CD residue."""
    lam_eff, c_bare = _aux_wtilde_core(setup, nu2)
    r = setup['rho'][:, p, m]
    val = r @ lam_eff @ r
    if setup['include_photon']:
        val = val + c_bare * setup['d_mo'][p, m] ** 2
    return val


def _wt_spline(freqs_all, Wt_p):
    """Spline of Wt_pm(i w') in w' for one p (all m). Wt is even in w',
    so the first derivative is clamped to zero at w' = 0."""
    return CubicSpline(freqs_all, Wt_p, axis=0,
                       bc_type=((1, np.zeros(Wt_p.shape[1])),
                                'not-a-knot'))


def _sigma_aux_cd(setup, Wt, freqs_all, quad_w, spline, p, omega, eps, eta):
    """Contour-deformation Sigma_c,pp(omega): imaginary-axis quadrature
    (with the sgn(omega - eps_m) jump split off analytically through the
    interpolated Wt_pm(i|omega - eps_m|)) plus pole residues evaluated
    with an +i*eta shift. Exact up to quadrature/interpolation error."""
    no, nmo = setup['no'], setup['nmo']
    Delta = omega - eps
    aD = np.abs(Delta)
    wmax = freqs_all[-1]
    idx = np.arange(nmo)
    WD = spline(np.minimum(aD, wmax))[idx, idx]
    tail = aD > wmax
    if tail.any():
        WD[tail] *= (wmax / aD[tail]) ** 2
    om2 = freqs_all[1:] ** 2
    kern = Delta[None, :] / (Delta[None, :] ** 2 + om2[:, None])
    sig = -(quad_w[:, None] * (Wt[1:, p, :] - WD[None, :]) * kern).sum() \
        / math.pi
    sig -= 0.5 * (WD * np.sign(Delta)).sum()
    for m in range(nmo):
        occ = m < no
        if occ and eps[m] > omega:
            fac, nu = -1.0, eps[m] - omega
        elif (not occ) and eps[m] < omega:
            fac, nu = 1.0, omega - eps[m]
        elif eps[m] == omega:
            fac, nu = (-0.5 if occ else 0.5), 0.0
        else:
            continue
        sig += fac * _aux_wtilde_pt(setup, (nu + 1j * eta) ** 2, p, m).real
    return sig


def _sigma_ac_samples(setup, Wt, freqs_all, quad_w, p, eps, mu, nu_grid):
    """Exact Sigma_c,pp(mu + i*nu) on the imaginary axis through the
    chemical potential mu (no residues there), for Pade continuation."""
    a = mu - eps
    Wp = Wt[1:, p, :]
    wq = quad_w[:, None]
    S = np.empty(len(nu_grid), dtype=complex)
    for j, nu in enumerate(nu_grid):
        den1 = a[None, :] + 1j * (nu + freqs_all[1:])[:, None]
        den2 = a[None, :] + 1j * (nu - freqs_all[1:])[:, None]
        S[j] = -(wq * Wp * (1.0 / den1 + 1.0 / den2)).sum() / (2.0 * math.pi)
    return S


def _thiele_fit(z, f):
    """Thiele continued-fraction coefficients through the points (z, f)."""
    z = np.asarray(z, dtype=complex)
    g = np.array(f, dtype=complex)
    n = len(z)
    a = np.empty(n, dtype=complex)
    a[0] = g[0]
    for i in range(1, n):
        with np.errstate(divide='ignore', invalid='ignore'):
            g[i:] = (a[i - 1] - g[i:]) / ((z[i:] - z[i - 1]) * g[i:])
        a[i] = g[i]
        if not np.isfinite(a[i]):
            return a[:i]                 # degenerate tail: truncate the fit
    return a


def _thiele_eval(a, z, x):
    """Evaluate the Thiele continued fraction fitted by :func:`_thiele_fit`."""
    r = 1.0 + 0.0j
    for i in range(len(a) - 1, 0, -1):
        r = 1.0 + a[i] * (x - z[i - 1]) / r
    return a[0] / r


def _aux_w_static_blocks(sq, eps, include_dse=True):
    """Correlation part of the static screened interaction from the aux
    dielectric at nu = 0 (photon excluded from W, per the polaritonic
    BSE convention: it appears once, explicitly, in the excitation
    space). Off-diagonal generalisation of :func:`_aux_wtilde_core`:

        W^corr_pq,rs = -2 sum_s M_pq,s M_rs,s / Omega_s
                     = rho_pq^T Lam_eff(0) rho_rs

    exactly (a single Woodbury inversion — no quadrature), so it matches
    the dense `_screening_at_eps` construction to machine precision.
    Returns the (i,j,a,b) and (i,b,a,j) blocks needed by the BSE."""
    setup = _aux_setup(sq, eps, include_dse=include_dse,
                       include_photon=False)
    lam, _ = _aux_wtilde_core(setup, 0.0)
    rho = setup['rho']
    no = sq['nocc']
    T_oo = contract('Xij,XY->Yij', rho[:, :no, :no], lam)
    Wc_ijab = contract('Yij,Yab->ijab', T_oo, rho[:, no:, no:])
    T_ov = contract('Xib,XY->Yib', rho[:, :no, no:], lam)
    Wc_ibaj = contract('Yib,Yaj->ibaj', T_ov, rho[:, no:, :no])
    return Wc_ijab, Wc_ibaj


def _qp_secant(fun, x0, step=1e-4, max_iter=50, tol=1e-9):
    """Secant solve of fun(omega) = 0 for the QP equation (the aux path
    has no cheap analytic dSigma/domega, so Newton is replaced by secant)."""
    f0 = fun(x0)
    x1 = x0 + step
    f1 = fun(x1)
    for _ in range(max_iter):
        if f1 == f0:
            break
        x2 = x1 - f1 * (x1 - x0) / (f1 - f0)
        x0, f0 = x1, f1
        x1 = x2
        f1 = fun(x1)
        if abs(x1 - x0) < tol:
            break
    return x1


def run_qed_gw_singlet(qedhf, mode='evGW', eta=1e-3, max_iter=50, tol=1e-7,
                       inner_max=50, inner_tol=1e-9, verbose=True,
                       screening='dense', n_freq=100, freq_scale=0.5,
                       n_pade=16, pade_numax=4.0):
    """Singlet-adapted (spatial) QED-GW quasiparticle energies with
    QED-dRPA screening. 'evGW' iterates all spatial orbitals to
    self-consistency in the spectrum; 'G0W0' does one-shot solves for
    all orbitals. Returns eps_QP for all spatial orbitals.

    Args:
        screening: 'dense' (default) diagonalises the (nov+1)-dim QED-dRPA
            problem and sums over its poles — O(nov^3) time, O(nov^2)
            memory. 'aux-cd' / 'aux-pade' use the auxiliary-basis
            dielectric (O(naux^2 nov) per frequency, no nov-dim matrix):
            'aux-cd' evaluates Sigma by contour deformation (exact up to
            quadrature; residue cost grows for deep states), 'aux-pade'
            by Pade analytic continuation from the imaginary axis (no
            residues — the right choice for full-spectrum evGW). Pade
            accuracy degrades with distance from the gap: ~1e-5 Ha
            within ~1 Ha of the gap centre, ~1e-3..1e-2 Ha for deep
            valence / high virtuals, worse for the core (H2O/6-31g
            benchmark) — use 'aux-cd' when those states matter.
        n_freq / freq_scale: imaginary-axis Gauss-Legendre grid (aux only).
        n_pade / pade_numax: number and upper bound (Ha) of the Pade
            sample frequencies i*nu along the shifted imaginary axis.
    """
    sq = _build_spatial_quantities(qedhf)
    no = sq['nocc']
    nmo = sq['nmo']
    eps_HF = np.diag(sq['F']).copy()
    if screening not in ('dense', 'aux-cd', 'aux-pade'):
        raise ValueError(f"unknown screening='{screening}'")

    if verbose:
        print(f"\n{mode}@QED-dRPA@QED-HF (singlet-adapted, {screening})")
        print(f"  nocc = {no}, nmo = {nmo} (spatial), "
              f"w_cav = {sq['omega_cav']:.6f} Ha, eta = {eta:.1e}")

    if screening != 'dense':
        quad_f, quad_w = _imag_freq_grid(n_freq, freq_scale)
        freqs_all = np.concatenate([[0.0], quad_f])
        nu_grid = np.logspace(math.log10(1e-2), math.log10(pade_numax),
                              n_pade)

    eps_cur = eps_HF.copy()
    n_outer = max_iter if mode == 'evGW' else 1
    Omega = M_full = None
    for outer in range(1, n_outer + 1):
        eps_new = np.empty_like(eps_cur)
        if screening == 'dense':
            Omega, M_full = _screening_at_eps(sq, eps_cur)
            for p in range(nmo):
                eps_new[p], _, _ = _solve_qp_newton(
                    eps_HF[p], M_full[p], Omega, eps_cur, no, eta,
                    omega_start=eps_cur[p], max_iter=inner_max,
                    tol=inner_tol)
        else:
            setup = _aux_setup(sq, eps_cur)
            Wt = _aux_wtilde_grid(setup, freqs_all)
            eps_scr = eps_cur.copy()      # frozen inside the QP solves
            if screening == 'aux-cd':
                for p in range(nmo):
                    spl = _wt_spline(freqs_all, Wt[:, p, :])
                    eps_new[p] = _qp_secant(
                        lambda w, p=p, spl=spl: w - eps_HF[p] - _sigma_aux_cd(
                            setup, Wt, freqs_all, quad_w, spl, p, w,
                            eps_scr, eta),
                        eps_cur[p], max_iter=inner_max, tol=inner_tol)
            else:
                mu = 0.5 * (eps_cur[no - 1] + eps_cur[no])
                for p in range(nmo):
                    S = _sigma_ac_samples(setup, Wt, freqs_all, quad_w,
                                          p, eps_scr, mu, nu_grid)
                    coef = _thiele_fit(1j * nu_grid, S)
                    eps_new[p] = _qp_secant(
                        lambda w, p=p, coef=coef: w - eps_HF[p]
                        - _thiele_eval(coef, 1j * nu_grid, w - mu).real,
                        eps_cur[p], max_iter=inner_max, tol=inner_tol)
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
        'screening': screening,
    }


# ---------------------------------------------------------------------------
# Iterative (Davidson) polaritonic BSE from the DF factors
# ---------------------------------------------------------------------------
def _bse_aux_operators(sq, eps_QP):
    """Closures applying the singlet polaritonic BSE blocks A and B to a
    packed vector [X_ia, x_photon] without forming any nov x nov matrix:
    bare Coulomb/DSE terms from the 3-index factor, the static-W
    correlation from the nu = 0 dielectric (Lambda-folded vertex
    factors), the photon through its explicit row/column. Every matvec
    is O(naux * nov * nmo). Also returns the exact diagonal of A for
    the Davidson preconditioner."""
    no, nmo = sq['nocc'], sq['nmo']
    nv = nmo - no
    d = sq['d']
    omega_cav = sq['omega_cav']

    setup = _aux_setup(sq, eps_QP, include_dse=True, include_photon=False)
    lam, _ = _aux_wtilde_core(setup, 0.0)
    rho = setup['rho']
    rho_ov = np.ascontiguousarray(rho[:, :no, no:])
    rho_vv = np.ascontiguousarray(rho[:, no:, no:])
    rt_oo = contract('XY,Yij->Xij', lam, rho[:, :no, :no])
    rt_ov = contract('XY,Yia->Xia', lam, rho_ov)

    B3 = sq['B']
    B_oo = np.ascontiguousarray(B3[:, :no, :no])
    B_ov = np.ascontiguousarray(B3[:, :no, no:])
    B_vv = np.ascontiguousarray(B3[:, no:, no:])
    d_oo, d_ov, d_vv = d[:no, :no], d[:no, no:], d[no:, no:]
    dE = eps_QP[None, no:] - eps_QP[:no, None]               # (no, nv)
    g = -math.sqrt(omega_cav) * d_ov                          # spin-summed

    def mv_A(vec):
        X = vec[:-1].reshape(no, nv)
        out = dE * X
        out += 2.0 * contract('Pia,P->ia', B_ov,
                              contract('Pjb,jb->P', B_ov, X))
        out += 2.0 * d_ov * np.sum(d_ov * X)
        out -= contract('Pij,jb,Pab->ia', B_oo, X, B_vv)
        out -= d_oo @ X @ d_vv
        out -= contract('Xij,jb,Xab->ia', rt_oo, X, rho_vv)
        out += g * vec[-1]
        return np.concatenate(
            [out.ravel(), [np.sum(g * X) + omega_cav * vec[-1]]])

    def mv_B(vec):
        Y = vec[:-1].reshape(no, nv)
        out = 2.0 * contract('Pia,P->ia', B_ov,
                             contract('Pjb,jb->P', B_ov, Y))
        out += 2.0 * d_ov * np.sum(d_ov * Y)
        out -= contract('Pib,jb,Pja->ia', B_ov, Y, B_ov)
        out -= d_ov @ (Y.T @ d_ov)
        out -= contract('Xib,jb,Xja->ia', rho_ov, Y, rt_ov)
        out += g * vec[-1]
        return np.concatenate([out.ravel(), [np.sum(g * Y)]])

    diag_e = dE + 2.0 * (contract('Pia,Pia->ia', B_ov, B_ov) + d_ov ** 2)
    diag_e -= contract('Pi,Pa->ia', np.einsum('Pii->Pi', B_oo),
                       np.einsum('Paa->Pa', B_vv))
    diag_e -= np.outer(np.diag(d_oo), np.diag(d_vv))
    diag_e -= contract('Xi,Xa->ia', np.einsum('Xii->Xi', rt_oo),
                       np.einsum('Xaa->Xa', rho_vv))
    diag = np.concatenate([diag_e.ravel(), [omega_cav]])
    return mv_A, mv_B, diag


def _orthonormalize(W, V=None, tol=1e-8):
    """Twice-iterated MGS: orthonormalize the columns of W against the
    orthonormal V and among themselves; drop numerically null columns."""
    cols = []
    for w in np.asarray(W).T:
        nrm0 = np.linalg.norm(w)
        if nrm0 == 0.0:
            continue
        w = w / nrm0
        for _ in range(2):
            if V is not None and V.shape[1]:
                w = w - V @ (V.T @ w)
            for c in cols:
                w = w - c * (c @ w)
        nrm = np.linalg.norm(w)
        if nrm > tol:
            cols.append(w / nrm)
    if not cols:
        return np.zeros((np.asarray(W).shape[0], 0))
    return np.array(cols).T


def _paired_davidson(mv_A, mv_B, diag, nroots, tda, conv_tol=1e-7,
                     max_cycle=100, max_space=None, verbose=False):
    """Davidson for the lowest nroots of the polaritonic BSE.

    TDA: standard symmetric Davidson on A. Full problem: paired
    subspace projection of [[A, B], [-B, -A]] — the small projected
    problem is solved symplectically by `_solve_qed_rpa_eigensystem`,
    so the Ritz (X, Y) keep the X^T X - Y^T Y = 1 normalisation of the
    dense path. Residuals of both the upper and lower components are
    preconditioned with the exact diagonal of A.

    A small buffer of extra roots is converged internally and discarded,
    so a (near-)degenerate multiplet straddling the nroots boundary does
    not leave the last returned root's partner untracked."""
    n = diag.size
    nreq = min(nroots, n - 1)
    nroots = min(nreq + max(2, nreq // 4), n - 1)
    nguess = min(n, nroots + min(nroots, 8))
    if max_space is None:
        max_space = min(n, max(6 * nroots, 40))

    guess_idx = list(np.argsort(diag)[:nguess])
    if n - 1 not in guess_idx:                # always seed the photon mode
        guess_idx[-1] = n - 1
    V = np.zeros((n, len(guess_idx)))
    V[guess_idx, np.arange(len(guess_idx))] = 1.0

    def _extend(Vnew):
        cols_A = [mv_A(v) for v in Vnew.T]
        cols_B = None if tda else [mv_B(v) for v in Vnew.T]
        return np.array(cols_A).T, (None if tda else np.array(cols_B).T)

    AV, BV = _extend(V)
    Y = None
    for cycle in range(1, max_cycle + 1):
        a = V.T @ AV
        a = 0.5 * (a + a.T)
        if tda:
            w, u = la.eigh(a)
            sel = np.argsort(w)[:nroots]
            Om, U = w[sel], u[:, sel]
            X = V @ U
            RX = AV @ U - X * Om
            res = np.linalg.norm(RX, axis=0)
        else:
            b = V.T @ BV
            b = 0.5 * (b + b.T)
            Om_all, Us, Vs = _solve_qed_rpa_eigensystem(a, b)
            sel = np.argsort(Om_all)[:nroots]
            Om, U, W = Om_all[sel], Us[:, sel], Vs[:, sel]
            X, Y = V @ U, V @ W
            RX = AV @ U + BV @ W - X * Om
            RY = BV @ U + AV @ W + Y * Om
            res = np.sqrt(np.linalg.norm(RX, axis=0) ** 2
                          + np.linalg.norm(RY, axis=0) ** 2)
        if verbose:
            print(f"    davidson {cycle:3d}: space = {V.shape[1]:4d}, "
                  f"max|r| = {res.max():.3e}")
        if np.all(res < conv_tol):
            return (Om[:nreq], X[:, :nreq],
                    None if tda else Y[:, :nreq], True, cycle)

        news = []
        for k in np.where(res >= conv_tol)[0]:
            den = Om[k] - diag
            den = np.where(np.abs(den) < 1e-6,
                           np.copysign(1e-6, den + 1e-300), den)
            news.append(RX[:, k] / den)
            if not tda:
                den = -Om[k] - diag
                den = np.where(np.abs(den) < 1e-6,
                               np.copysign(1e-6, den + 1e-300), den)
                news.append(RY[:, k] / den)
        news = np.array(news).T

        if V.shape[1] + news.shape[1] > max_space:   # collapse to Ritz space
            ritz = X if tda else np.hstack([X, Y])
            V = _orthonormalize(ritz)
            AV, BV = _extend(V)
        Vnew = _orthonormalize(news, V)
        if Vnew.shape[1] == 0:
            break                                    # subspace stagnated
        AVn, BVn = _extend(Vnew)
        V = np.hstack([V, Vnew])
        AV = np.hstack([AV, AVn])
        if not tda:
            BV = np.hstack([BV, BVn])
    return (Om[:nreq], X[:, :nreq], None if tda else Y[:, :nreq],
            bool(np.all(res < conv_tol)), cycle)


# ---------------------------------------------------------------------------
# Singlet-adapted polaritonic QED-BSE@GW
# ---------------------------------------------------------------------------
def run_qed_bse_polaritonic_singlet(qedhf, gw_mode='evGW', tda=False,
                                    eta=1e-3, eps_QP=None, verbose=True,
                                    gw_screening='dense',
                                    w_screening='dense',
                                    solver='dense', nroots=12,
                                    davidson_tol=1e-7):
    """Singlet-adapted polaritonic QED-BSE@GW (excitons + LP/UP).

    Mirrors :func:`qed_bse_polaritonic.run_qed_bse_polaritonic` in the
    singlet channel: electronic BSE blocks with purely electronic dRPA
    screening (photon NOT in W — it appears once, explicitly, through
    the sqrt(2)-vertex row/column), DSE kept as a genuine electronic
    interaction.

    Args:
        eps_QP: optional spatial QP energies (skips the GW step).
        gw_screening: screening backend for the GW step
            (see :func:`run_qed_gw_singlet`).
        w_screening: 'dense' builds the static W from the poles of the
            nov-dim dRPA eigenproblem; 'aux' builds the identical W (to
            machine precision — the nu = 0 dielectric is exact, no
            quadrature) from one (naux+1)-dim Woodbury inversion,
            skipping the nov-dim diagonalisation and the M_full tensor.
            Only used by solver='dense'.
        solver: 'dense' (default) assembles the (nov+1)-dim BSE blocks
            and diagonalises them — all roots, O(nov^2) memory,
            O(nov^3) time. 'davidson' never forms them: matvecs from
            the DF factors + nu = 0 dielectric (O(naux*nov*nmo) each,
            O(naux*nov) memory) drive a paired Davidson for the lowest
            `nroots` polariton roots (converged to `davidson_tol` on
            the residual norm).

    Returns:
        dict with 'Omega', 'f_osc', 'photon_weight', 'omega_cav',
        'eps_QP', 'tda' ('Omega' holds all roots for solver='dense',
        the lowest nroots for solver='davidson').
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
                                verbose=verbose, screening=gw_screening)
        eps_QP = gw['eps_QP']
    else:
        eps_QP = np.asarray(eps_QP, dtype=float).copy()
        if verbose:
            print("\npol-BSE (singlet): using caller-supplied QP energies"
                  " (GW step skipped).")

    dim = nov + 1
    if solver == 'davidson':
        mv_A, mv_B, diagA = _bse_aux_operators(sq, eps_QP)
        Omega, Xr, Yr, dav_conv, dav_cyc = _paired_davidson(
            mv_A, mv_B, diagA, nroots=nroots, tda=tda,
            conv_tol=davidson_tol, verbose=verbose)
        if verbose and not dav_conv:
            print(f"  warning: davidson not fully converged in "
                  f"{dav_cyc} cycles")
        XpY = Xr if tda else Xr + Yr
        # packed (i, a) -> the (a, i)-flattened convention of the tail
        XpY_e = (XpY[:-1].reshape(no, nv, -1)
                 .transpose(1, 0, 2).reshape(nov, -1))
        ph_amp = XpY[-1, :]
        method = ('pol-TDA-BSE (singlet, davidson)' if tda
                  else 'pol-BSE (singlet, davidson)')
    elif solver != 'dense':
        raise ValueError(f"unknown solver='{solver}'")

    # Electronic screening only (photon excluded from W), DSE kept.
    if solver == 'dense' and w_screening == 'dense':
        Omega_scr, M_full = _screening_at_eps(sq, eps_QP, include_dse=True,
                                              include_photon=False)
        inv_Om = 1.0 / Omega_scr
        Wc_ijab = -2.0 * contract('ijs,abs,s->ijab', M_full[:no, :no, :],
                                  M_full[no:, no:, :], inv_Om)
        Wc_ibaj = -2.0 * contract('ibs,ajs,s->ibaj', M_full[:no, no:, :],
                                  M_full[no:, :no, :], inv_Om)
    elif solver == 'dense' and w_screening == 'aux':
        Wc_ijab, Wc_ibaj = _aux_w_static_blocks(sq, eps_QP)
    elif solver == 'dense':
        raise ValueError(f"unknown w_screening='{w_screening}'")

    if solver == 'dense':
        v_ovov, v_oovv = _ov_blocks(sq)

        # W^stat_{ij,ab} = (ij|ab) + d_ij d_ab - 2 sum_s M_ij,s M_ab,s / O_s
        W_ijab = (v_oovv
                  + np.einsum('ij,ab->ijab', d[:no, :no], d[no:, no:])
                  + Wc_ijab)
        # W^stat_{ib,aj} = (ib|aj) + d_ib d_aj - 2 sum_s M_ib,s M_aj,s / O_s
        # (ib|aj) as an (i,b,a,j) tensor is v_ovov[i,b,j,a] -> (0,1,3,2)
        W_ibaj = (v_ovov.transpose(0, 1, 3, 2)
                  + np.einsum('ib,aj->ibaj', d[:no, no:], d_vo)
                  + Wc_ibaj)

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
        'solver': solver,
    }


if __name__ == '__main__':
    from pyscf import gto
    mol = gto.M(atom="O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692",
                basis='6-31g', verbose=0)
    omega = 0.4
    qedhf = run_qed_hf(mol, omega, (0.0, 0.0, 0.05), verbose=False)
    run_qed_rpa_singlet(qedhf, direct=False, verbose=True)
    run_qed_bse_polaritonic_singlet(qedhf, gw_mode='evGW', verbose=True)
