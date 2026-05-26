"""
Cavity QED Bethe-Salpeter Equation (BSE) for neutral excitation
energies on top of a QED-GW reference.

Static BSE in the spin-orbital basis (see, e.g., Onida-Reining-Rubio
Rev. Mod. Phys. 2002 Eq. 224, and Bruneval & Gonze PRB 2008):

    A^BSE_{ia,jb} = (ε^QP_a − ε^QP_i) δ_{ij}δ_{ab}
                    + ⟨aj‖ib⟩_phys                 (bare e-h exchange,
                                                    NB antisymmetric)
                    − W^stat_{ij,ab}               (screened e-h direct,
                                                    dRPA-W only — no
                                                    double counting of K_x)

    B^BSE_{ia,jb} = ⟨ab‖ij⟩_phys − W^stat_{ib,aj}

TDA-BSE keeps only the A block and gives a Hermitian eigenvalue
problem. Full BSE solves the non-Hermitian Casida-like form
``[[A, B], [−B, −A]]·[X; Y] = Ω·[X; Y]``.

W^stat is the static (ω = 0) limit of the QED-dRPA screened interaction:

    W^stat_{pq,rs} = ⟨pq|rs⟩ + d_{pq} d_{rs}
                     − 2 Σ_m M_{pq,m} M_{rs,m} / Ω_m

where (Ω_m, M_{pq,m}) are the QED-RPA eigenvalues and
matrix-elements produced by :mod:`qed_gw` (which already absorb the
photon channel through the augmented A,B blocks of arXiv:2602.09968
Eq. 6). The bare-exchange kernel includes the antisymmetric DSE
contribution Δ^x_{aj,ib} = d_{aj}d_{ib} − d_{ab}d_{ij} (paper Eq. 13).

The cavity enters in three places:

1. ε^QP_p from QED-GW (cavity-dressed quasiparticle energies),
2. v_{pq,rs} = ⟨pq|rs⟩ + d_{pq} d_{rs} (DSE direct in the bare W),
3. M_{pq,m} carries the photon-mediated piece via the (M+N)_m
   amplitudes of every QED-RPA mode.

Scope / caveats
---------------
* Spin-orbital implementation; for closed-shell molecules the
  spectrum contains both singlet and triplet roots (and triplet
  instabilities show up as small / imaginary roots).
* "Full" BSE (``tda=False``) inherits the standard BSE caveat that the
  resonant–antiresonant coupling can introduce spurious instabilities;
  TDA-BSE (default) is usually more stable.
* The static-W approximation is standard; a frequency-dependent W
  (dynamical BSE) is not implemented.
"""

import math

import numpy as np
import scipy.linalg as la
from pyscf import gto

from .qed_hf import run_qed_hf
from .qed_gw import (_build_static_quantities, _rpa_at_eps, run_qed_gw)


def _W_static(static, Omega, M_full, channel):
    """Static screened interaction in one of two 4-orbital channels.

    Args:
        static, Omega, M_full: outputs of qed_gw._build_static_quantities
            and _rpa_at_eps.
        channel: one of 'ij,ab' or 'ib,aj' selecting which orbital
            ranges go into which slot of W_{pq,rs}.

    Returns:
        ndarray of shape matching the channel:
            'ij,ab' → (nocc, nocc, nvir, nvir) indexed (i, j, a, b)
            'ib,aj' → (nocc, nvir, nvir, nocc) indexed (i, b, a, j)
    """
    so = static['so']
    nocc = so['nocc']
    g_phys_d = so['g_phys_d']
    d_so = so['d_so']
    inv_Omega = 1.0 / Omega

    if channel == 'ij,ab':
        # v^dir_{ij,ab} = ⟨ij|ab⟩ + d_{ij} d_{ab}
        v_dir = (g_phys_d[:nocc, :nocc, nocc:, nocc:].copy()
                 + np.einsum('ij,ab->ijab',
                             d_so[:nocc, :nocc], d_so[nocc:, nocc:]))
        # W_c_{ij,ab} = −2 Σ_m M_{ij,m} M_{ab,m} / Ω_m
        M_oo = M_full[:nocc, :nocc, :]
        M_vv = M_full[nocc:, nocc:, :]
        W_c = -2.0 * np.einsum('ijm,abm,m->ijab', M_oo, M_vv, inv_Omega)
        return v_dir + W_c

    if channel == 'ib,aj':
        v_dir = (g_phys_d[:nocc, nocc:, nocc:, :nocc].copy()
                 + np.einsum('ib,aj->ibaj',
                             d_so[:nocc, nocc:], d_so[nocc:, :nocc]))
        M_ov = M_full[:nocc, nocc:, :]
        M_vo = M_full[nocc:, :nocc, :]
        W_c = -2.0 * np.einsum('ibm,ajm,m->ibaj', M_ov, M_vo, inv_Omega)
        return v_dir + W_c

    raise ValueError(f"unknown channel {channel!r}")


def _bare_exchange_kernel(static, channel):
    """Antisymmetric bare e-h kernel ⟨··‖··⟩ + Δ^x in one of two
    channels:

        'aj,ib' → returns (a, i, b, j) tensor for the A block
        'ab,ij' → returns (a, i, b, j) tensor for the B block
    """
    so = static['so']
    nocc = so['nocc']
    g_phys_a = so['g_phys_a']
    d_so = so['d_so']
    d_vo = d_so[nocc:, :nocc]
    d_ov = d_so[:nocc, nocc:]
    d_vv = d_so[nocc:, nocc:]
    d_oo = d_so[:nocc, :nocc]

    if channel == 'aj,ib':
        # ⟨aj||ib⟩ in (a, j, i, b) → transpose to (a, i, b, j) via (0,2,3,1)
        K_x = g_phys_a[nocc:, :nocc, :nocc, nocc:].transpose(0, 2, 3, 1)
        # Δ^x_{aj,ib} = d_{aj} d_{ib} − d_{ab} d_{ij}, indexed (a, i, b, j)
        delta_x = (np.einsum('aj,ib->aibj', d_vo, d_ov)
                   - np.einsum('ab,ij->aibj', d_vv, d_oo))
        return K_x + delta_x

    if channel == 'ab,ij':
        # ⟨ab||ij⟩ in (a, b, i, j) → transpose to (a, i, b, j) via (0,2,1,3)
        K_x = g_phys_a[nocc:, nocc:, :nocc, :nocc].transpose(0, 2, 1, 3)
        # Δ^x_{ab,ij} = d_{ab} d_{ij} − d_{aj} d_{ib}, indexed (a, i, b, j)
        delta_x = (np.einsum('ab,ij->aibj', d_vv, d_oo)
                   - np.einsum('aj,ib->aibj', d_vo, d_ov))
        return K_x + delta_x

    raise ValueError(f"unknown channel {channel!r}")


def run_qed_bse(qedhf, gw_mode='evGW', tda=True, n_print=10,
                eta=1e-3, verbose=True):
    """BSE@QED-GW for cavity-modified neutral excitation energies.

    Args:
        qedhf: dict from :func:`OmegaQMC.qed_hf.run_qed_hf`.
        gw_mode: 'linG0W0' / 'G0W0' / 'evGW' — quasiparticle scheme.
            'evGW' (default) gives QP energies for all orbitals.
            With the cheaper modes the QED-GW is run for HOMO/LUMO
            only and other orbital energies fall back to ε^HF.
        tda: True (default) → Tamm-Dancoff approximation
            (A only, Hermitian, more stable). False → full BSE
            (non-Hermitian Casida-like problem).
        n_print: number of low-lying excitations to print in verbose mode.
        eta: imaginary regulariser used by the GW step (Ha).
        verbose: print progress.

    Returns:
        dict with 'Omega_BSE' (sorted excitation energies, Ha),
        'tda', 'gw_mode', and the underlying 'eps_QP', 'Omega_RPA'.
    """
    # 1) QP energies from QED-GW.
    if gw_mode == 'evGW':
        if verbose:
            print(f"\nQED-BSE: running underlying QED-evGW...")
        gw = run_qed_gw(qedhf, mode='evGW', eta=eta, verbose=False)
        eps_QP = np.asarray(gw['eps_GW_all'])
    else:
        static_init = _build_static_quantities(qedhf, direct=True)
        n_spat = static_init['so']['nso'] // 2
        if verbose:
            print(f"\nQED-BSE: running underlying QED-{gw_mode} "
                  f"for {n_spat} spatial orbitals...")
        gw = run_qed_gw(qedhf, mode=gw_mode, orbs=list(range(n_spat)),
                        eta=eta, verbose=False)
        eps_QP = static_init['eps_HF'].copy()
        for p in range(n_spat):
            eps_QP[2 * p] = gw['eps_QP'][p]
            eps_QP[2 * p + 1] = gw['eps_QP'][p]

    # 2) Build static W from the QED-dRPA spectrum evaluated at the
    #    same orbital energies (matches evGW philosophy).
    static = _build_static_quantities(qedhf, direct=True)
    Omega_RPA, M_full = _rpa_at_eps(static, eps_QP)

    so = static['so']
    nocc = so['nocc']
    nso = so['nso']
    nvir = nso - nocc
    nov = nvir * nocc

    # 3) Build A^BSE.
    K_x_A = _bare_exchange_kernel(static, 'aj,ib')
    W_ijab = _W_static(static, Omega_RPA, M_full, 'ij,ab')
    # − W^stat_{ij,ab} sits in A_{a,i,b,j}: transpose (i,j,a,b) → (a,i,b,j)
    W_A_aibj = W_ijab.transpose(2, 0, 3, 1)

    A_BSE_4d = K_x_A - W_A_aibj
    A_BSE = A_BSE_4d.reshape(nov, nov)
    A_diag = (eps_QP[nocc:, None] - eps_QP[None, :nocc]).reshape(-1)
    A_BSE = A_BSE + np.diag(A_diag)
    # Symmetrise (it should already be symmetric; tighten numerical noise).
    A_BSE = 0.5 * (A_BSE + A_BSE.T)

    if tda:
        Omega_BSE = la.eigvalsh(A_BSE)
        method = 'TDA-BSE'
    else:
        K_x_B = _bare_exchange_kernel(static, 'ab,ij')
        W_ibaj = _W_static(static, Omega_RPA, M_full, 'ib,aj')
        # − W^stat_{ib,aj} into B_{a,i,b,j}: (i,b,a,j) → (a,i,b,j) is (2,0,1,3)
        W_B_aibj = W_ibaj.transpose(2, 0, 1, 3)
        B_BSE_4d = K_x_B - W_B_aibj
        B_BSE = B_BSE_4d.reshape(nov, nov)
        B_BSE = 0.5 * (B_BSE + B_BSE.T)

        big = np.zeros((2 * nov, 2 * nov))
        big[:nov, :nov] = A_BSE
        big[:nov, nov:] = B_BSE
        big[nov:, :nov] = -B_BSE
        big[nov:, nov:] = -A_BSE
        evals = la.eigvals(big)
        re = evals.real
        if np.max(np.abs(evals.imag)) > 1e-6:
            if verbose:
                print(f"  warning: full-BSE complex eigenvalues "
                      f"max|Im|={np.max(np.abs(evals.imag)):.2e} — "
                      f"possible instability; consider tda=True.")
        Omega_BSE = np.sort(re[re > 1e-10])
        method = 'BSE'

    if verbose:
        print(f"\n{method}@QED-{gw_mode}@QED-HF")
        print(f"  nocc(SO)={nocc}, nso={nso}, "
              f"BSE matrix dim={nov}, n_modes_W={len(Omega_RPA)}")
        n_show = min(n_print, len(Omega_BSE))
        print(f"  Lowest {n_show} {method} excitation energies:")
        for i in range(n_show):
            print(f"    Ω_{i+1:<2d} = {Omega_BSE[i]:.6f} Ha "
                  f"= {Omega_BSE[i] * 27.211386245988:8.4f} eV")

    return {
        'method': method,
        'tda': bool(tda),
        'gw_mode': gw_mode,
        'Omega_BSE': Omega_BSE,
        'eps_QP': eps_QP,
        'Omega_RPA': Omega_RPA,
    }


if __name__ == '__main__':
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
        basis='cc-pVDZ', unit='Angstrom', symmetry=False, verbose=0,
    )

    omega = 0.415668
    for lam in [0.0, 0.05, 0.10]:
        print('=' * 60)
        print(f'λ = {lam}')
        qedhf = run_qed_hf(mol, omega, (0, 0, lam), verbose=False)
        bse = run_qed_bse(qedhf, gw_mode='evGW', tda=True, n_print=5,
                          verbose=True)
