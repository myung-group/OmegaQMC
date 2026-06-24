"""
Dynamical (frequency-dependent) Bethe-Salpeter equation on top of a
cavity QED-GW reference.

The static QED-BSE of :mod:`OmegaQMC.addons.qed_bse` freezes the
screened electron-hole interaction at ω = 0. As shown for the
cavity-coupled molecule, the static kernel is (nearly)
"cavity-transparent": at ω = 0 the photon-exchange and DSE
contributions to the two-particle kernel cancel, so the genuine
photon-mediated electron-hole binding (and any exciton-photon
hybridisation / Rabi splitting) lives entirely in the *frequency
dependence* of W. This module restores that dependence.

Formalism (Strinati; Rohlfing-Louie; Loos & Blase, J. Chem. Phys. 153,
114120 (2020), hereafter LB20). The dynamically screened interaction
that enters the resonant BSE A block is, in the present spin-orbital /
chemist-density convention (LB20 Eq. 29, mapped so that its Ω_m → ∞
limit reproduces the ``_W_static`` of :mod:`qed_bse`,
W^stat = v̄ − 2 Σ_m M M / Ω_m):

    W̃_{ij,ab}(Ω_S) = v̄_{ij,ab}
        + Σ_m M_{ij,m} M_{ab,m} [ 1/(Ω^S_{ib} − Ω_m + iη)
                                + 1/(Ω^S_{ja} − Ω_m + iη) ],

    Ω^S_{pq} = Ω_S − (ε^QP_q − ε^QP_p).

Only the A (resonant) block is made dynamical: the anti-resonant
B-block dynamical terms (LB20 Eq. 31) develop spurious divergences
when Ω_S − Ω_m − (ε_a − ε_b) ≈ 0, which is why every published
dynamical-BSE study restricts the *dynamical* part to the
Tamm-Dancoff approximation (dTDA). We follow that prescription: the
zeroth-order problem is the static TDA-BSE A^(0) built by
:func:`qed_bse._assemble_A_static`, and the dynamical correction acts
on the A block alone.

Two schemes are provided through ``mode``:

* ``'perturbative'`` (default) — the renormalised first-order
  correction of LB20 (Eqs. 40-42):

    Ω^dyn_S = Ω^(0)_S + Z_S Ω^(1)_S,
    Ω^(1)_S = X_S^T A^(1)(Ω^(0)_S) X_S,
    A^(1)(Ω_S) = W^stat − W̃(Ω_S)  (in the (ai),(bj) layout),
    Z_S = [ 1 − X_S^T ∂_Ω A^(1)(Ω_S)|_{Ω^(0)_S} X_S ]^{-1}.

  X_S are the static TDA eigenvectors. This recovers, for transitions
  with dominant single-excitation character, the dynamical relaxation
  from higher excitations; it cannot create new (double-excitation)
  roots.

* ``'selfconsistent'`` — solve the non-linear dynamical eigenvalue
  problem A(Ω_S) X = Ω_S X self-consistently per root (dTDA), with
  A(Ω_S) = A^(0) + Re A^(1)(Ω_S). The target root is tracked across
  iterations by maximum overlap with the static seed eigenvector. This
  resums the diagonal frequency dependence beyond first order; it still
  does not add double excitations (that needs the full non-Hermitian
  unfolding, outside the present scope).

The cavity enters through exactly the three channels of the static
QED-BSE — the QP energies ε^QP on the diagonal and in Ω^S_{pq}, the
DSE d⊗d inside v̄/M, and the polaritonic poles Ω_m of the screening —
but now the poles are resolved at their true frequency. A photon-
dominated polariton mode Ω_m near a low-lying exciton (Ω^S_{ib} ≈ Ω_m)
makes W̃ resonant, which is precisely the exciton-photon coupling the
static kernel projects out.
"""

import math

import numpy as np
import scipy.linalg as la
from pyscf import gto

from .qed_hf import run_qed_hf
from .qed_gw import _build_static_quantities, _rpa_at_eps
from .qed_bse import (_assemble_A_static, _resolve_eps_QP,
                      _classify_and_brightness)

EV = 27.211386245988


def _dynamical_A1(static, Omega_RPA, M_full, eps_QP, Omega_S, eta,
                  deriv=False):
    """First-order dynamical A-block correction A^(1)(Ω_S).

    Returns A^(1) as an (nov, nov) complex matrix in the (ai),(bj)
    layout used by the BSE A matrix, defined as

        A^(1)_{ij,ab}(Ω_S) = W^stat_{ij,ab} − W̃_{ij,ab}(Ω_S)
            = Σ_m M_{ij,m} M_{ab,m}
                  [ −2/Ω_m − D_{ib,m}(Ω_S) − D_{ja,m}(Ω_S) ],

    with the only frequency dependence in
    D_{ov,m}(Ω_S) = 1/(Ω_S + ε^QP_occ[o] − ε^QP_vir[v] − Ω_m + iη)
    (the bare v̄ cancels in W^stat − W̃). With ``deriv=True`` also
    returns ∂A^(1)/∂Ω_S, whose only non-zero piece is

        ∂A^(1)_{ij,ab}/∂Ω_S = Σ_m M_{ij,m} M_{ab,m}
                                 [ D_{ib,m}^2 + D_{ja,m}^2 ].

    In the static limit (Ω_m → ∞) D → −1/Ω_m, so A^(1) → 0 and the
    static TDA-BSE is recovered, as it must be.
    """
    so = static['so']
    nocc = so['nocc']
    nso = so['nso']
    nvir = nso - nocc
    nov = nvir * nocc

    eps_occ = eps_QP[:nocc]
    eps_vir = eps_QP[nocc:]
    inv_Omega = 1.0 / Omega_RPA

    M_oo = M_full[:nocc, :nocc, :]          # M_{ij,m}
    M_vv = M_full[nocc:, nocc:, :]          # M_{ab,m}

    # D_{ov,m} = 1/(Ω_S + ε_occ[o] − ε_vir[v] − Ω_m + iη), one tensor
    # reused for both D_{ib,m} and D_{ja,m} (same occ/vir/mode meaning).
    shift = (Omega_S
             + eps_occ[:, None, None]
             - eps_vir[None, :, None]
             - Omega_RPA[None, None, :])     # (o, v, m)
    D = 1.0 / (shift + 1j * eta)             # (o, v, m)

    # A^(1)_{ij,ab} = static W_c  − W̃_c(Ω_S)
    Wc_stat = -2.0 * np.einsum('ijm,abm,m->ijab', M_oo, M_vv, inv_Omega,
                               optimize=True)
    term_ib = np.einsum('ijm,abm,ibm->ijab', M_oo, M_vv, D, optimize=True)
    term_ja = np.einsum('ijm,abm,jam->ijab', M_oo, M_vv, D, optimize=True)
    A1_ijab = Wc_stat - term_ib - term_ja
    A1 = A1_ijab.transpose(2, 0, 3, 1).reshape(nov, nov)

    if not deriv:
        return A1

    D2 = D * D
    dterm_ib = np.einsum('ijm,abm,ibm->ijab', M_oo, M_vv, D2, optimize=True)
    dterm_ja = np.einsum('ijm,abm,jam->ijab', M_oo, M_vv, D2, optimize=True)
    dA1_ijab = dterm_ib + dterm_ja
    dA1 = dA1_ijab.transpose(2, 0, 3, 1).reshape(nov, nov)
    return A1, dA1


def _select_states(spin, n_states, states):
    """Indices (into the sorted static spectrum) of the roots to correct."""
    nroots = len(spin)
    if states == 'lowest':
        return list(range(min(n_states, nroots)))
    if states == 'bright':
        # lowest n_states singlet roots (the optically relevant ones)
        idx = [i for i in range(nroots) if spin[i] == 'S']
        return idx[:n_states]
    if isinstance(states, (list, tuple, np.ndarray)):
        return [int(i) for i in states]
    raise ValueError(f"unknown states selector {states!r}")


def run_qed_bse_dynamical(qedhf, gw_mode='evGW', mode='perturbative',
                          n_states=10, states='lowest', eta=1e-3,
                          include_dse=True, include_photon=True,
                          eps_QP=None, sc_max=50, sc_tol=1e-6,
                          n_print=10, verbose=True):
    """Dynamical-BSE neutral excitation energies on a QED-GW reference.

    Args:
        qedhf: dict from :func:`OmegaQMC.addons.qed_hf.run_qed_hf`.
        gw_mode: underlying QP scheme ('evGW' / 'G0W0' / 'linG0W0'),
            forwarded to :func:`qed_bse._resolve_eps_QP`.
        mode: 'perturbative' (renormalised first-order LB20 correction)
            or 'selfconsistent' (non-linear dTDA solve per root).
        n_states: number of low-lying roots to dynamically correct.
        states: 'lowest' (default), 'bright' (lowest singlets), or an
            explicit list of indices into the static spectrum.
        eta: imaginary regulariser in the screened-interaction
            denominators (Ha) — used by both the GW step and W̃.
        include_dse, include_photon: kernel-channel switches (passed to
            :func:`qed_gw._build_static_quantities`), for the cavity
            channel decomposition of the dynamical shift.
        eps_QP: optional spin-orbital QP energies (skips the GW step).
        sc_max, sc_tol: iteration cap / convergence threshold (Ha) for
            the self-consistent mode.
        n_print, verbose: reporting controls.

    Returns:
        dict with the static spectrum ('Omega_stat', full sorted array),
        the corrected roots ('idx', 'Omega_stat_sel', 'Omega_dyn',
        'dOmega' = Ω_dyn − Ω_stat, and 'Z' in perturbative mode),
        per-root 'spin'/'f_osc', the dynamically corrected lowest
        singlet/triplet 'Omega_S1'/'Omega_T1' and exciton binding
        'E_b' = E_gap − Ω_S1, plus 'E_gap', 'eps_QP', 'Omega_RPA'.
    """
    if mode not in ('perturbative', 'selfconsistent'):
        raise ValueError("mode must be 'perturbative' or 'selfconsistent'; "
                         f"got {mode!r}")

    # 1) QP energies and static screening — identical path to run_qed_bse.
    eps_QP = _resolve_eps_QP(qedhf, gw_mode, eta, eps_QP, verbose)
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

    # 2) Static TDA-BSE: zeroth-order spectrum and eigenvectors.
    A0 = _assemble_A_static(static, Omega_RPA, M_full, eps_QP)
    Omega_stat, X_vec = la.eigh(A0)
    X_4d = X_vec.reshape(nvir, nocc, -1)
    spin, w_singlet, f_osc = _classify_and_brightness(
        qedhf, X_4d, Omega_stat, nocc)

    idx = _select_states(spin, n_states, states)

    Omega_dyn = np.empty(len(idx))
    Z_arr = np.full(len(idx), np.nan)
    for k, s in enumerate(idx):
        Om0 = float(Omega_stat[s])
        X_s = X_vec[:, s]
        if mode == 'perturbative':
            A1, dA1 = _dynamical_A1(static, Omega_RPA, M_full, eps_QP,
                                    Om0, eta, deriv=True)
            Omega1 = float((X_s @ (A1 @ X_s)).real)
            slope = float((X_s @ (dA1 @ X_s)).real)
            Z = 1.0 / (1.0 - slope)
            Z_arr[k] = Z
            Omega_dyn[k] = Om0 + Z * Omega1
        else:  # selfconsistent (non-linear dTDA)
            Om = Om0
            seed = X_s.copy()
            for _ in range(sc_max):
                A1 = _dynamical_A1(static, Omega_RPA, M_full, eps_QP,
                                   Om, eta, deriv=False)
                A_dyn = A0 + A1.real
                A_dyn = 0.5 * (A_dyn + A_dyn.T)
                evals, evecs = la.eigh(A_dyn)
                ovlp = (evecs.T @ seed) ** 2
                pick = int(np.argmax(ovlp))
                Om_new = float(evals[pick])
                seed = evecs[:, pick]
                if abs(Om_new - Om) < sc_tol:
                    Om = Om_new
                    break
                Om = Om_new
            Omega_dyn[k] = Om

    dOmega = Omega_dyn - Omega_stat[idx]

    # 3) Dynamically corrected optical observables. Spin character is
    #    taken from the static eigenvectors (robust under the small
    #    dynamical shift); S1/T1 are the lowest corrected roots of each
    #    spin among the selected set.
    sel_spin = np.array([spin[s] for s in idx])
    s_mask = sel_spin == 'S'
    t_mask = sel_spin == 'T'
    Omega_S1 = float(np.min(Omega_dyn[s_mask])) if np.any(s_mask) else None
    Omega_T1 = float(np.min(Omega_dyn[t_mask])) if np.any(t_mask) else None
    E_b = (E_gap - Omega_S1) if Omega_S1 is not None else None

    if verbose:
        tag = ('renorm. 1st-order' if mode == 'perturbative'
               else 'self-consistent dTDA')
        print(f"\nDynamical TDA-BSE@QED-{gw_mode}@QED-HF  ({tag})")
        print(f"  nocc(SO)={nocc}, nso={nso}, dim={nov}, "
              f"n_modes_W={len(Omega_RPA)}, η={eta:.1e}")
        print(f"  E_gap(QP) = {E_gap * EV:.4f} eV")
        head = ("  root  spin   Ω_stat (eV)   Ω_dyn (eV)   ΔΩ_dyn (eV)"
                + ("     Z" if mode == 'perturbative' else ""))
        print(head)
        for k, s in enumerate(idx[:n_print]):
            z = f"  {Z_arr[k]:.4f}" if mode == 'perturbative' else ""
            print(f"  {s:4d}   [{spin[s]}]  {Omega_stat[s] * EV:11.4f}  "
                  f"{Omega_dyn[k] * EV:11.4f}  {dOmega[k] * EV:+11.4f}{z}")
        if Omega_S1 is not None:
            print(f"  Ω_S1(dyn) = {Omega_S1 * EV:.4f} eV"
                  + (f",  Ω_T1(dyn) = {Omega_T1 * EV:.4f} eV"
                     if Omega_T1 is not None else "")
                  + f",  E_b(dyn) = {E_b * EV:.4f} eV")

    return {
        'method': f'dyn-TDA-BSE({mode})',
        'mode': mode,
        'gw_mode': gw_mode,
        'include_dse': bool(include_dse),
        'include_photon': bool(include_photon),
        'idx': np.asarray(idx),
        'Omega_stat': Omega_stat,
        'Omega_stat_sel': Omega_stat[idx],
        'Omega_dyn': Omega_dyn,
        'dOmega': dOmega,
        'Z': Z_arr,
        'spin': spin,
        'w_singlet': w_singlet,
        'f_osc': f_osc,
        'Omega_S1': Omega_S1,
        'Omega_T1': Omega_T1,
        'E_b': E_b,
        'E_gap': E_gap,
        'eps_QP': eps_QP,
        'Omega_RPA': Omega_RPA,
    }


if __name__ == '__main__':
    half_angle = math.radians(104.5 / 2.0)
    rOH = 1.0
    hx = rOH * math.sin(half_angle)
    hz = -rOH * math.cos(half_angle)
    mol = gto.M(
        atom=[['O', (0.0, 0.0, 0.0)],
              ['H', (+hx, 0.0, hz)],
              ['H', (-hx, 0.0, hz)]],
        basis='cc-pVDZ', unit='Angstrom', symmetry=False, verbose=0,
    )
    omega = 0.415668
    for lam in (0.0, 0.05):
        print('=' * 64)
        print(f'lambda = {lam}')
        qedhf = run_qed_hf(mol, omega, (0.0, 0.0, lam), verbose=False)
        run_qed_bse_dynamical(qedhf, gw_mode='evGW', mode='perturbative',
                              n_states=6, verbose=True)
