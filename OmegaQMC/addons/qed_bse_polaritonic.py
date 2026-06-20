"""Polaritonic QED-BSE@GW: the photon as an explicit excitation-space state.

The static QED-BSE of :mod:`OmegaQMC.addons.qed_bse` folds the cavity
photon into the screened interaction W (``include_photon=True``), where
the omega->0 limit cancels the DSE exchange and the kernel becomes
*cavity-transparent*: its purely electronic (nov-dimensional) excitation
space supports excitons, never polaritons.

This module instead places the photon **explicitly** in the excitation
manifold, exactly as the photon-augmented QED-RPA does, but with the
screened *BSE* electronic blocks in place of the bare RPA blocks:

    A_pol = [ A^BSE_e   g    ]      B_pol = [ B^BSE_e   g  ]
            [ g^dag    w_cav ]              [ g^dag     0  ]

    g_ai = -sqrt(w_cav/2) d_ai          (the QED-RPA/GW vertex).

Double counting is avoided by building the electronic blocks with
``include_photon=False`` -- the photon appears *once*, explicitly via
(g, w_cav), and never inside W. The DSE is a genuine electronic
interaction and is kept (``include_dse=True``).

Validation (FCI-benchmarked): against exact QED-FCI on STO-3G water,
tuned to a strongly bright state (~15.5 eV), the resonant Rabi splitting
matches FCI to ~5% (BSE 0.601 vs FCI 0.629 eV at lambda=0.05) and follows
the two-level formula Omega_R = sqrt(3) lambda sqrt(f) to ~1% -- so the
light-matter coupling is implemented correctly and the residual gap is
exciton (oscillator-strength) quality, not the construction.

IMPORTANT -- this is NOT the static limit of the transparent BSE of
:mod:`qed_bse` (include_photon=True). The explicit photon enters the
transition/exchange channel (g_ai g_bj ~ d_ai d_bj), whereas the
transparent BSE's photon-screening enters the *direct* channel
(d_ij d_ab); the two are near-orthogonal (verified numerically,
cos ~ 0.03), so downfolding the explicit photon does NOT reproduce the
transparent kernel. The polaritonic BSE is the QED-RPA analogue with
screened BSE blocks -- a distinct (FCI-validated) object, not the
omega->0 limit of the screening-folded BSE. The non-TDA form is the
production setting; the TDA form (photon only in A) is a quick check.

Eigenvectors split into an electronic block (length nov) and one photonic
amplitude; the photon weight is |amp|^2 and the oscillator strength is
computed from the electronic block (length gauge), so the bright bare
exciton hands its intensity to the two polariton branches at resonance.
"""

import math

import numpy as np
import scipy.linalg as la

from .qed_hf import run_qed_hf
from .qed_gw import (_build_static_quantities, _rpa_at_eps,
                     _solve_qed_rpa_eigensystem)
from .qed_bse import (_assemble_A_static, _eh_exchange_kernel, _W_static,
                      _resolve_eps_QP)

EV = 27.211386245988


def _electronic_bse_blocks(static, Omega_RPA, M_full, eps_QP, tda):
    """A^BSE_e (and B^BSE_e for the full problem), nov x nov, with the
    ELECTRONIC screening already baked into ``static`` (the caller must
    have used include_photon=False)."""
    so = static['so']
    nocc = so['nocc']
    nvir = so['nso'] - nocc
    nov = nvir * nocc
    A_BSE = _assemble_A_static(static, Omega_RPA, M_full, eps_QP)
    if tda:
        return A_BSE, None
    K_x_B = _eh_exchange_kernel(static, 'ab,ij')
    W_ibaj = _W_static(static, Omega_RPA, M_full, 'ib,aj')
    W_B_aibj = W_ibaj.transpose(2, 0, 1, 3)            # (i,b,a,j) -> (a,i,b,j)
    B_BSE = (K_x_B - W_B_aibj).reshape(nov, nov)
    B_BSE = 0.5 * (B_BSE + B_BSE.T)
    return A_BSE, B_BSE


def _osc_strength(qedhf, XpY_e_4d, Omega, nocc, nso):
    """Length-gauge oscillator strengths f = (2/3) Omega |sum_ai mu_ai v_ai|^2
    from the electronic part v of each (polariton) eigenvector."""
    C = np.asarray(qedhf['C'])
    idx = np.arange(nso)
    same = (idx[:, None] % 2) == (idx[None, :] % 2)
    f = np.zeros_like(Omega)
    for key in ('mu_x_ao', 'mu_y_ao', 'mu_z_ao'):
        mu_sf = C.T @ np.asarray(qedhf[key]) @ C
        mu_so = same * mu_sf[idx[:, None] // 2, idx[None, :] // 2]
        mu_m = np.einsum('ai,ais->s', mu_so[nocc:, :nocc], XpY_e_4d)
        f += (2.0 / 3.0) * Omega * mu_m ** 2
    return f


def run_qed_bse_polaritonic(qedhf, gw_mode='evGW', tda=False, eta=1e-3,
                            eps_QP=None, verbose=True):
    """Polaritonic QED-BSE@GW excitation energies (excitons + LP/UP).

    Args:
        qedhf: dict from :func:`OmegaQMC.addons.qed_hf.run_qed_hf`. The
            cavity frequency ``qedhf['omega']`` becomes the photon
            diagonal w_cav.
        gw_mode: 'evGW' / 'G0W0' / 'linG0W0' for the QP energies.
        tda: False (default) -> full (non-TDA) polaritonic BSE, exact
            static-limit cancellation. True -> TDA (photon only in A),
            quick check; static limit only half-cancels the DSE.
        eta: GW imaginary regulariser (Ha).
        eps_QP: optional spin-orbital QP energies (skip the GW step).
        verbose: print a summary of the bright polariton roots.

    Returns:
        dict with 'Omega' (sorted excitation energies, Ha), 'f_osc'
        (oscillator strengths), 'photon_weight' (|photonic amplitude|^2
        per root), 'omega_cav', 'eps_QP', and 'tda'.
    """
    omega_cav = qedhf['omega']
    eps_QP = _resolve_eps_QP(qedhf, gw_mode, eta, eps_QP, verbose)

    # Double-counting rule: ELECTRONIC screening only (no photon in W).
    static = _build_static_quantities(qedhf, direct=True,
                                      include_photon=False, include_dse=True)
    Omega_RPA, M_full = _rpa_at_eps(static, eps_QP)
    so = static['so']
    nocc, nso = so['nocc'], so['nso']
    nvir = nso - nocc
    nov = nvir * nocc

    A_e, B_e = _electronic_bse_blocks(static, Omega_RPA, M_full, eps_QP, tda)

    # Explicit photon vertex g_ai = -sqrt(w_cav/2) d_ai.
    d_vo = so['d_so'][nocc:, :nocc]
    g_vec = -math.sqrt(omega_cav / 2.0) * d_vo.reshape(nov)

    dim = nov + 1
    if tda:
        A_big = np.zeros((dim, dim))
        A_big[:nov, :nov] = A_e
        A_big[:nov, -1] = g_vec
        A_big[-1, :nov] = g_vec
        A_big[-1, -1] = omega_cav
        A_big = 0.5 * (A_big + A_big.T)
        Omega, vecs = la.eigh(A_big)
        XpY_e = vecs[:nov, :]                      # electronic block
        ph_amp = vecs[-1, :]                       # photon amplitude
        method = 'pol-TDA-BSE'
    else:
        A_big = np.zeros((dim, dim))
        B_big = np.zeros((dim, dim))
        A_big[:nov, :nov] = A_e
        A_big[:nov, -1] = A_big[-1, :nov] = g_vec
        A_big[-1, -1] = omega_cav
        B_big[:nov, :nov] = B_e
        B_big[:nov, -1] = B_big[-1, :nov] = g_vec   # B photon diagonal = 0
        Omega, U, V = _solve_qed_rpa_eigensystem(A_big, B_big)
        UpV = U + V
        XpY_e = UpV[:nov, :]
        ph_amp = UpV[-1, :]
        method = 'pol-BSE'

    order = np.argsort(Omega)
    Omega = Omega[order]
    XpY_e = XpY_e[:, order]
    ph_amp = ph_amp[order]
    photon_weight = ph_amp ** 2

    XpY_e_4d = XpY_e.reshape(nvir, nocc, -1)
    f_osc = _osc_strength(qedhf, XpY_e_4d, Omega, nocc, nso)

    if verbose:
        print(f"\n{method}@QED-{gw_mode}@QED-HF   "
              f"(photon explicit, include_photon=False in W)")
        print(f"  nocc(SO)={nocc}, nso={nso}, dim={dim}, "
              f"w_cav={omega_cav * EV:.3f} eV")
        print("   Omega(eV)   f_osc     photon_wt")
        bright = np.where(f_osc > 1e-3)[0]
        for i in bright[:12]:
            print(f"   {Omega[i] * EV:8.3f}   {f_osc[i]:8.4f}  "
                  f"{photon_weight[i]:8.3f}")

    return {
        'method': method,
        'tda': bool(tda),
        'gw_mode': gw_mode,
        'Omega': Omega,
        'f_osc': f_osc,
        'photon_weight': photon_weight,
        'omega_cav': float(omega_cav),
        'eps_QP': eps_QP,
        'nov': nov,
    }
