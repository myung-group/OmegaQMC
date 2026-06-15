"""
Cavity QED Bethe-Salpeter Equation (BSE) for neutral excitation
energies on top of a QED-GW reference.

Static BSE in the spin-orbital basis (see, e.g., Onida-Reining-Rubio
Rev. Mod. Phys. 2002 Eq. 224, and Bruneval & Gonze PRB 2008):

    A^BSE_{ia,jb} = (ε^QP_a − ε^QP_i) δ_{ij}δ_{ab}
                    + ⟨aj|ib⟩_phys + d_ai d_bj     (bare e-h exchange of
                                                    v̄ = v + d⊗d,
                                                    unantisymmetrized)
                    − W^stat_{ij,ab}               (screened e-h direct;
                                                    its bare part is the
                                                    chemist (ij|ab) +
                                                    d_ij d_ab, once)

    B^BSE_{ia,jb} = ⟨ab|ij⟩_phys + d_ai d_bj − W^stat_{ib,aj}

The direct e-h attraction lives only inside W^stat (screened); the e-h
exchange of the QED interaction v̄ appears once, bare. In the W → v̄
limit the TDA-BSE therefore reduces exactly to CIS on the QED-RPA
kernel, and the spin-adapted structure is the textbook one: singlets
feel 2(ia|jb) + 2 d_ia d_jb − W, triplets −W only.

TDA-BSE keeps only the A block and gives a Hermitian eigenvalue
problem. Full BSE solves the non-Hermitian Casida-like form
``[[A, B], [−B, −A]]·[X; Y] = Ω·[X; Y]``.

W^stat is the static (ω = 0) limit of the QED-dRPA screened interaction
(chemist density pairing, matching the M_{pq,m} transition densities):

    W^stat_{pq,rs} = (pq|rs)_chem + d_{pq} d_{rs}
                     − 2 Σ_m M_{pq,m} M_{rs,m} / Ω_m

where (Ω_m, M_{pq,m}) are the QED-RPA eigenvalues and
matrix-elements produced by :mod:`qed_gw` (which already absorb the
photon channel through the augmented A,B blocks of arXiv:2602.09968
Eq. 6).

The cavity enters in three places:

1. ε^QP_p from QED-GW (cavity-dressed quasiparticle energies),
2. v̄_{pq,rs} = (pq|rs) + d_{pq} d_{rs} (DSE direct in the bare W and
   in the e-h exchange kernel),
3. M_{pq,m} carries the photon-mediated piece via the (M+N)_m
   amplitudes of every QED-RPA mode (polaritonic poles of W).

Scope / caveats
---------------
* Spin-orbital implementation; for closed-shell molecules the
  spectrum contains both singlet and triplet roots (and triplet
  instabilities show up as small / imaginary roots). In TDA mode the
  roots are spin-classified via the singlet projection
  w_S = ½ Σ_AI (X_αα + X_ββ)² and electronic oscillator strengths are
  computed from the BSE eigenvectors, so the optical gap Ω_S1, the
  lowest triplet Ω_T1, and the exciton binding energy
  E_b = E_gap(QP) − Ω_S1 are returned directly.
* The two cavity channels of the kernel can be toggled independently
  (``include_dse``, ``include_photon``) and the QP energies can be
  overridden (``eps_QP=...``), enabling a channel decomposition of
  cavity-induced exciton shifts.
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
            and _rpa_at_eps. The DSE-direct d⊗d term follows the
            ``include_dse`` flag stored in ``static`` (the photon
            channel is controlled inside M_full itself).
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
    include_dse = static.get('include_dse', True)
    inv_Omega = 1.0 / Omega

    if channel == 'ij,ab':
        # v̄_{ij,ab} = (ij|ab)_chem + d_{ij} d_{ab}. The chemist direct
        # (ij|ab) couples the densities (i j) and (a b) — the same
        # pairing as the M_{ij,m} M_{ab,m} correlation term — and equals
        # the physicist ⟨ia|jb⟩, i.e. g_phys[i,a,j,b].
        v_dir = (g_phys_d[:nocc, nocc:, :nocc, nocc:]
                 .transpose(0, 2, 1, 3).copy())
        if include_dse:
            v_dir += np.einsum('ij,ab->ijab',
                               d_so[:nocc, :nocc], d_so[nocc:, nocc:])
        # W_c_{ij,ab} = −2 Σ_m M_{ij,m} M_{ab,m} / Ω_m
        M_oo = M_full[:nocc, :nocc, :]
        M_vv = M_full[nocc:, nocc:, :]
        W_c = -2.0 * np.einsum('ijm,abm,m->ijab', M_oo, M_vv, inv_Omega)
        return v_dir + W_c

    if channel == 'ib,aj':
        # v̄_{ib,aj} = (ib|aj)_chem + d_{ib} d_{aj} = ⟨ia|bj⟩_phys + DSE,
        # i.e. g_phys[i,a,b,j] reordered to (i, b, a, j).
        v_dir = (g_phys_d[:nocc, nocc:, nocc:, :nocc]
                 .transpose(0, 2, 1, 3).copy())
        if include_dse:
            v_dir += np.einsum('ib,aj->ibaj',
                               d_so[:nocc, nocc:], d_so[nocc:, :nocc])
        M_ov = M_full[:nocc, nocc:, :]
        M_vo = M_full[nocc:, :nocc, :]
        W_c = -2.0 * np.einsum('ibm,ajm,m->ibaj', M_ov, M_vo, inv_Omega)
        return v_dir + W_c

    raise ValueError(f"unknown channel {channel!r}")


def _eh_exchange_kernel(static, channel):
    """Bare electron-hole exchange kernel of the QED interaction
    v̄ = v + d⊗d, in one of two channels:

        'aj,ib' → returns (a, i, b, j) tensor for the A block,
                   ⟨aj|ib⟩_phys + d_ai d_bj
        'ab,ij' → returns (a, i, b, j) tensor for the B block,
                   ⟨ab|ij⟩_phys + d_ai d_bj

    NB: *unantisymmetrized*. The screened direct e-h attraction enters
    the BSE once, through −W^stat (whose bare part is the chemist
    (ij|ab) + d_ij d_ab); antisymmetrizing here would double-count it.
    This kernel is exactly the e-h exchange of the QED-dRPA blocks that
    generate W, so the W→v̄ limit of the TDA-BSE recovers CIS. The DSE
    part d_ai d_bj follows the ``include_dse`` flag in ``static``.
    """
    so = static['so']
    nocc = so['nocc']
    g_phys_d = so['g_phys_d']
    d_so = so['d_so']
    include_dse = static.get('include_dse', True)
    d_vo = d_so[nocc:, :nocc]

    if channel == 'aj,ib':
        # ⟨aj|ib⟩ in (a, j, i, b) → transpose to (a, i, b, j) via (0,2,3,1)
        K_x = g_phys_d[nocc:, :nocc, :nocc, nocc:].transpose(0, 2, 3, 1)
    elif channel == 'ab,ij':
        # ⟨ab|ij⟩ in (a, b, i, j) → transpose to (a, i, b, j) via (0,2,1,3)
        K_x = g_phys_d[nocc:, nocc:, :nocc, :nocc].transpose(0, 2, 1, 3)
    else:
        raise ValueError(f"unknown channel {channel!r}")

    if not include_dse:
        return K_x.copy()
    # DSE e-h exchange d_ai d_bj (the d⊗d analogue of the direct
    # integral above; same in both channels since d is symmetric).
    return K_x + np.einsum('ai,bj->aibj', d_vo, d_vo)


def _classify_and_brightness(qedhf, X_4d, Omega, nocc):
    """Spin-classify TDA-BSE roots and compute oscillator strengths.

    Args:
        qedhf: QED-HF dict (for MO coefficients and AO dipole matrices).
        X_4d: TDA eigenvectors reshaped to (nvir, nocc, nroots),
            normalized to Σ|X|² = 1, spin-orbital (even=α, odd=β,
            closed-shell interleaved ordering).
        Omega: excitation energies (Ha).
        nocc: number of occupied spin orbitals (even).

    Returns:
        (labels, w_singlet, f_osc): labels is an array of 'S'/'T'
        characters, w_singlet the singlet projection weight per root,
        f_osc the electronic oscillator strengths (length gauge).
    """
    nvir = X_4d.shape[0]
    nso = nocc + nvir

    # Singlet projection weight: w_S = ½ Σ_AI (X_αα + X_ββ)².
    # Pure singlet → 1; triplet (any M_S) → 0.
    X_aa = X_4d[0::2, 0::2, :]
    X_bb = X_4d[1::2, 1::2, :]
    w_singlet = 0.5 * np.sum((X_aa + X_bb) ** 2, axis=(0, 1))
    labels = np.where(w_singlet > 0.5, 'S', 'T')

    # Oscillator strengths f_n = (2/3) Ω_n Σ_κ |Σ_ai μ^κ_ai X_ai,n|².
    C = np.asarray(qedhf['C'])
    idx = np.arange(nso)
    same = (idx[:, None] % 2) == (idx[None, :] % 2)
    f_osc = np.zeros_like(Omega)
    for key in ('mu_x_ao', 'mu_y_ao', 'mu_z_ao'):
        mu_sf = C.T @ np.asarray(qedhf[key]) @ C
        mu_so = same * mu_sf[idx[:, None] // 2, idx[None, :] // 2]
        mu_vo = mu_so[nocc:, :nocc]
        mu_n = np.einsum('ai,ain->n', mu_vo, X_4d)
        f_osc += (2.0 / 3.0) * Omega * mu_n ** 2
    return labels, w_singlet, f_osc


def _resolve_eps_QP(qedhf, gw_mode, eta, eps_QP, verbose):
    """Spin-orbital quasiparticle energies for the BSE diagonal/screening.

    Either uses the caller-supplied ``eps_QP`` verbatim, or runs the
    underlying QED-GW: 'evGW' corrects every orbital, the cheaper
    one-shot modes correct HOMO/LUMO (spatial) and leave the rest at
    ε^HF. Shared by the static (:func:`run_qed_bse`) and dynamical
    (:func:`OmegaQMC.addons.qed_bse_dynamical.run_qed_bse_dynamical`)
    drivers so both rest on an identical QP set.
    """
    if eps_QP is not None:
        eps_QP = np.asarray(eps_QP, dtype=float).copy()
        if verbose:
            print("\nQED-BSE: using caller-supplied QP energies "
                  "(GW step skipped).")
        return eps_QP
    if gw_mode == 'evGW':
        if verbose:
            print("\nQED-BSE: running underlying QED-evGW...")
        gw = run_qed_gw(qedhf, mode='evGW', eta=eta, verbose=False)
        return np.asarray(gw['eps_GW_all'])
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
    return eps_QP


def _assemble_A_static(static, Omega_RPA, M_full, eps_QP):
    """Static TDA-BSE A block A^(0)_{ai,bj} as an (nov, nov) matrix.

    A^(0)_{ai,bj} = (ε^QP_a − ε^QP_i) δ_ij δ_ab
                    + [⟨aj|ib⟩ + d_ai d_bj]            (bare e-h exchange)
                    − W^stat_{ij,ab}.                  (screened e-h direct)

    This is the ω → ∞ (equivalently ω = 0) limit of the dynamical A
    block; the dynamical correction is added on top of it. Returns a
    symmetrised matrix.
    """
    so = static['so']
    nocc = so['nocc']
    nvir = so['nso'] - nocc
    nov = nvir * nocc
    K_x_A = _eh_exchange_kernel(static, 'aj,ib')
    W_ijab = _W_static(static, Omega_RPA, M_full, 'ij,ab')
    # − W^stat_{ij,ab} sits in A_{a,i,b,j}: transpose (i,j,a,b) → (a,i,b,j)
    W_A_aibj = W_ijab.transpose(2, 0, 3, 1)
    A_BSE = (K_x_A - W_A_aibj).reshape(nov, nov)
    A_diag = (eps_QP[nocc:, None] - eps_QP[None, :nocc]).reshape(-1)
    A_BSE = A_BSE + np.diag(A_diag)
    # Symmetrise (it should already be symmetric; tighten numerical noise).
    return 0.5 * (A_BSE + A_BSE.T)


def run_qed_bse(qedhf, gw_mode='evGW', tda=True, n_print=10,
                eta=1e-3, include_dse=True, include_photon=True,
                eps_QP=None, verbose=True):
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
        include_dse: keep the DSE channel in the *BSE kernel* — the
            d⊗d term of the bare W, the DSE-exchange Δ^x of the bare
            e-h kernel, and the DSE blocks of the screening RPA. The
            QP energies are NOT affected (channel decomposition of the
            kernel, not of the GW step).
        include_photon: keep the bilinear electron–photon coupling in
            the *BSE kernel* screening (photon-augmented RPA poles and
            photon kernel inside W). QP energies are NOT affected.
        eps_QP: optional spin-orbital QP energies (length nso). When
            given, the GW step is skipped and these energies are used
            for the BSE diagonal and the screening — e.g. to feed λ=0
            quasiparticles into a finite-λ kernel for the channel
            decomposition of the exciton binding energy.
        verbose: print progress.

    Returns:
        dict with 'Omega_BSE' (sorted excitation energies, Ha), 'tda',
        'gw_mode', the underlying 'eps_QP', 'Omega_RPA', the
        fundamental QP gap 'E_gap', and — in TDA mode — per-root spin
        labels 'spin' ('S'/'T'), singlet weights 'w_singlet',
        oscillator strengths 'f_osc', the lowest singlet/triplet
        energies 'Omega_S1' / 'Omega_T1', and the exciton binding
        energy 'E_b' = E_gap − Omega_S1 (all Ha).
    """
    # 1) QP energies from QED-GW (or caller-supplied override).
    eps_QP = _resolve_eps_QP(qedhf, gw_mode, eta, eps_QP, verbose)

    # 2) Build static W from the QED-dRPA spectrum evaluated at the
    #    same orbital energies (matches evGW philosophy). The kernel
    #    channel flags enter here (and only here).
    static = _build_static_quantities(qedhf, direct=True,
                                      include_dse=include_dse,
                                      include_photon=include_photon)
    Omega_RPA, M_full = _rpa_at_eps(static, eps_QP)

    so = static['so']
    nocc = so['nocc']
    nso = so['nso']
    nvir = nso - nocc
    nov = nvir * nocc

    E_gap = float(eps_QP[nocc] - eps_QP[nocc - 1])

    # 3) Build A^BSE (static TDA A block).
    A_BSE = _assemble_A_static(static, Omega_RPA, M_full, eps_QP)

    spin = w_singlet = f_osc = None
    Omega_S1 = Omega_T1 = E_b = None
    if tda:
        Omega_BSE, X_vec = la.eigh(A_BSE)
        X_4d = X_vec.reshape(nvir, nocc, -1)
        spin, w_singlet, f_osc = _classify_and_brightness(
            qedhf, X_4d, Omega_BSE, nocc)
        singlets = Omega_BSE[spin == 'S']
        triplets = Omega_BSE[spin == 'T']
        Omega_S1 = float(singlets[0]) if len(singlets) else None
        Omega_T1 = float(triplets[0]) if len(triplets) else None
        if Omega_S1 is not None:
            E_b = E_gap - Omega_S1
        method = 'TDA-BSE'
    else:
        K_x_B = _eh_exchange_kernel(static, 'ab,ij')
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

    EV = 27.211386245988
    if verbose:
        print(f"\n{method}@QED-{gw_mode}@QED-HF  "
              f"(include_dse={include_dse}, include_photon={include_photon})")
        print(f"  nocc(SO)={nocc}, nso={nso}, "
              f"BSE matrix dim={nov}, n_modes_W={len(Omega_RPA)}")
        print(f"  E_gap(QP) = {E_gap:.6f} Ha = {E_gap * EV:8.4f} eV")
        n_show = min(n_print, len(Omega_BSE))
        print(f"  Lowest {n_show} {method} excitation energies:")
        for i in range(n_show):
            tag = ''
            if spin is not None:
                tag = (f"  [{spin[i]}]  w_S={w_singlet[i]:.3f}  "
                       f"f={f_osc[i]:.4f}")
            print(f"    Ω_{i+1:<2d} = {Omega_BSE[i]:.6f} Ha "
                  f"= {Omega_BSE[i] * EV:8.4f} eV{tag}")
        if E_b is not None:
            print(f"  Ω_S1 = {Omega_S1 * EV:.4f} eV,  "
                  f"Ω_T1 = {Omega_T1 * EV:.4f} eV,  "
                  f"E_b = E_gap − Ω_S1 = {E_b * EV:.4f} eV")

    return {
        'method': method,
        'tda': bool(tda),
        'gw_mode': gw_mode,
        'include_dse': bool(include_dse),
        'include_photon': bool(include_photon),
        'Omega_BSE': Omega_BSE,
        'eps_QP': eps_QP,
        'Omega_RPA': Omega_RPA,
        'E_gap': E_gap,
        'spin': spin,
        'w_singlet': w_singlet,
        'f_osc': f_osc,
        'Omega_S1': Omega_S1,
        'Omega_T1': Omega_T1,
        'E_b': E_b,
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
