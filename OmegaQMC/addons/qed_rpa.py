"""
Reference: arXiv:2602.09968 (DePrince, Yuwono, Eshuis, 2026)

Cavity QED random phase approximation (QED-RPA) on top of a QED-HF
reference. Implements both the full QED-RPA and the "direct" QED-dRPA
correlation energy, with the latter being equivalent to the ring
simplification of QED-CCD (QED-drCCD) proved in Sec. II.D of the paper.

The QED-RPA eigenvalue problem (Eq. 6) takes the block form

       ( A+Δ    B+Δ'    g    g̃  ) ( X )       ( I  0  0  0 ) ( X )
       ( B+Δ'   A+Δ     g̃    g  ) ( Y )   =   ( 0 -I  0  0 ) ( Y ) Ω
       ( g†     g̃†      ω    0  ) ( M )       ( 0  0  1  0 ) ( M )
       ( g̃†     g†      0    ω  ) ( N )       ( 0  0  0 -1 ) ( N )

with

       A_{ai,bj}  = (ε_a − ε_i) δ_ij δ_ab + ⟨ib||aj⟩
       B_{ai,bj}  = ⟨ij||ab⟩
       Δ_{ai,bj}  = d_{ai} d_{jb} − d_{ab} d_{ij}
       Δ'_{ai,bj} = d_{ai} d_{bj} − d_{aj} d_{ib}
       g_{ai}     = g̃_{ai} = −√(ω/2) d_{ai}
       d_{pq}     = −λ · ⟨φ_p|r|φ_q⟩

(ε_p is the diagonal of the QED-HF Fock matrix). Regrouping into the
composite excitation / de-excitation vectors of Eq. 34 the problem
reduces to

       ( A   B ) ( U )      ( U )
       (-B  -A ) ( V )  =   ( V ) Ω

with

       A = ((Ã, g), (g†, ω)),    B = ((B̃, g̃), (g̃†, 0)),
       Ã = A + Δ,                B̃ = B + Δ',                 (1)

a (2(N_ov+1))-dimensional non-Hermitian eigenvalue problem whose
positive solutions are the QED-RPA excitation energies.

The TDA-RWA approximation (Eq. 15) is the hermitian problem

       ( Ã   g ) ( X̄ )     ( X̄ )
       ( g†  ω ) ( M̄ )  =  ( M̄ ) Ω̄ .

The QED-RPA correlation energy (Eq. 16) is then

       E_c = ½ ( Tr Ω − Tr Ω̄ ) .

QED-dRPA is obtained by dropping the exchange contributions everywhere:
⟨pq||rs⟩ → ⟨pq|rs⟩ and Δ = Δ' = d_{ai} d_{bj}.

Origin dependence: the QED-dRPA correlation energy is *not* invariant
under rigid translations of the molecule — the direct channel keeps
only the d_{ai} d_{bj} DSE product and drops the −d_{ab} d_{ij}
counterpart that restores translational invariance in the full QED-RPA
(the QED-HF energy and the full QED-RPA correlation energy are both
origin-invariant; verified numerically to <1e-7 Ha). To reproduce
Table I of arXiv:2602.09968 to all printed digits, the molecule must
be re-centred at its nuclear centre of mass, the default orientation
of the Psi4 backend used there; with the oxygen at the coordinate
origin instead, the dRPA correlation energy carries an ~λ²-scaling
offset (2×10⁻⁴ Ha at λ = 0.2 for water/cc-pVDZ).

The implementation works in spin orbitals, so the formulas above are
applied verbatim for either a restricted QED-HF reference
(:mod:`OmegaQMC.addons.qed_hf`) or a spin-unrestricted QED-UHF
reference (:mod:`OmegaQMC.addons.qed_uhf`) for open-shell systems —
``run_qed_rpa`` detects which from the reference dict and builds the
spin-orbital tensors accordingly. Validation at λ = 0 reduces to the
standard direct RPA correlation energy because the bare photon block
contributes the same ω to both Tr Ω and Tr Ω̄, which then cancels in
the difference.
"""

import math

import numpy as np
import scipy.linalg as la
from pyscf import gto

from .qed_hf import run_qed_hf, eri_mo_transform


def _build_spin_orbital_quantities_uhf(qedhf):
    """Spin-orbital QED-RPA tensors from a spin-unrestricted QED-UHF
    reference.

    Counterpart of :func:`_build_spin_orbital_quantities` for open-shell
    systems: the α and β orbitals differ, so the spin-orbital ERIs are
    assembled from the four spatial chemist blocks (αα/αβ/ββ/βα), and the
    Fock and dipole are spin-block-diagonal. Spin orbitals are ordered
    occupied-first, ``[α-occ, β-occ, α-vir, β-vir]``, matching the
    ``[:nocc]`` / ``[nocc:]`` slicing in :func:`_assemble_AB`. Returns the
    same dict as :func:`_build_spin_orbital_quantities`.
    """
    Ca = np.asarray(qedhf['Ca'])
    Cb = np.asarray(qedhf['Cb'])
    Fa = np.asarray(qedhf['Fa'])
    Fb = np.asarray(qedhf['Fb'])
    dipole = qedhf['dipole_x_lambda_tot']    # λ·μ_AO
    nocc_a = qedhf['nocc_a']
    nocc_b = qedhf['nocc_b']
    nmo = qedhf['nmo_spatial']

    nocc = nocc_a + nocc_b
    nso = 2 * nmo

    # Bare ERI (DSE enters via Δ, Δ'), dispatching on dense/DF representation.
    def _trans(Cp, Cq, Cr, Cs):
        return eri_mo_transform(qedhf, Cp, Cq, Cr, Cs, dse=False)

    # Gstack[s1, s2] = chemist (i j | k l), electron-1 spin s1, e-2 spin s2.
    Gstack = np.zeros((2, 2, nmo, nmo, nmo, nmo))
    Gstack[0, 0] = _trans(Ca, Ca, Ca, Ca)
    Gstack[1, 1] = _trans(Cb, Cb, Cb, Cb)
    Gstack[0, 1] = _trans(Ca, Ca, Cb, Cb)
    Gstack[1, 0] = _trans(Cb, Cb, Ca, Ca)

    order = ([(0, p) for p in range(nocc_a)]
             + [(1, p) for p in range(nocc_b)]
             + [(0, p) for p in range(nocc_a, nmo)]
             + [(1, p) for p in range(nocc_b, nmo)])
    spins = np.array([s for s, _ in order], dtype=int)
    spat = np.array([p for _, p in order], dtype=int)

    idx = np.arange(nso)
    P = idx[:, None, None, None]
    Q = idx[None, :, None, None]
    R = idx[None, None, :, None]
    S = idx[None, None, None, :]
    sP, sQ, sR, sS = spins[P], spins[Q], spins[R], spins[S]
    pP, pQ, pR, pS = spat[P], spat[Q], spat[R], spat[S]

    direct_chem = Gstack[sP, sR, pP, pQ, pR, pS] * (sP == sQ) * (sR == sS)
    exchange_chem = Gstack[sP, sR, pP, pS, pR, pQ] * (sP == sS) * (sR == sQ)
    g_phys_d = direct_chem.transpose(0, 2, 1, 3)                    # ⟨pq|rs⟩
    g_phys_a = (direct_chem - exchange_chem).transpose(0, 2, 1, 3)  # ⟨pq||rs⟩

    fa = Ca.T @ Fa @ Ca
    fb = Cb.T @ Fb @ Cb
    da = -(Ca.T @ dipole @ Ca)
    db = -(Cb.T @ dipole @ Cb)
    Fstack = np.stack([fa, fb])
    Dstack = np.stack([da, db])
    P2 = idx[:, None]
    Q2 = idx[None, :]
    same = (spins[P2] == spins[Q2])
    F_so = Fstack[spins[P2], spat[P2], spat[Q2]] * same
    d_so = Dstack[spins[P2], spat[P2], spat[Q2]] * same

    return {
        'F_so': F_so,
        'd_so': d_so,
        'g_phys_a': g_phys_a,
        'g_phys_d': g_phys_d,
        'nocc': nocc,
        'nso': nso,
    }


def _build_spin_orbital_quantities(qedhf):
    """Build the spin-orbital QED-HF Fock, dipole, and ERI tensors.

    Returns a dict with:
        F_so       (nso, nso)             — block-spin Fock (diagonal = ε_p)
        d_so       (nso, nso)             — d_{pq} = −λ·⟨p|r|q⟩
        g_phys_a   (nso, nso, nso, nso)   — antisymmetrised ⟨pq||rs⟩
        g_phys_d   (nso, nso, nso, nso)   — direct ⟨pq|rs⟩ (no exchange)
        nocc, nso                          — sizes

    Dispatches on the reference type: a restricted QED-HF dict (key
    ``'C'``) or an unrestricted QED-UHF dict (key ``'Ca'``).
    """
    if 'Ca' in qedhf:
        return _build_spin_orbital_quantities_uhf(qedhf)

    C = qedhf['C']
    F_ao = qedhf['F']
    dipole = qedhf['dipole_x_lambda_tot']
    nocc_spatial = qedhf['nocc_spatial']
    nmo = qedhf['nmo_spatial']

    nso = 2 * nmo
    nocc = 2 * nocc_spatial

    # Spatial MO integrals (bare ERI; dispatches on dense/DF representation).
    F_sf = C.T @ F_ao @ C
    d_sf = -C.T @ dipole @ C
    g_mo_sf = eri_mo_transform(qedhf, C, C, C, C, dse=False)

    # Spin-block: spin index convention even=alpha, odd=beta.
    idx = np.arange(nso)
    pe, qe = idx[:, None], idx[None, :]
    same_spin = (pe % 2 == qe % 2)
    F_so = same_spin * F_sf[pe // 2, qe // 2]
    d_so = same_spin * d_sf[pe // 2, qe // 2]

    # Spin-orbital 2-electron tensors in physicist notation.
    p = idx[:, None, None, None]
    q = idx[None, :, None, None]
    r = idx[None, None, :, None]
    s = idx[None, None, None, :]
    sp, sq, sr, ss = p % 2, q % 2, r % 2, s % 2
    p2, q2, r2, s2 = p // 2, q // 2, r // 2, s // 2

    # chemist (pq|rs) split into direct and exchange spin-orbital tensors
    direct_chem = g_mo_sf[p2, q2, r2, s2] * (sp == sq) * (sr == ss)
    exchange_chem = g_mo_sf[p2, s2, r2, q2] * (sp == ss) * (sq == sr)

    g_phys_d = direct_chem.transpose(0, 2, 1, 3)                    # ⟨pq|rs⟩
    g_phys_a = (direct_chem - exchange_chem).transpose(0, 2, 1, 3)  # ⟨pq||rs⟩

    return {
        'F_so': F_so,
        'd_so': d_so,
        'g_phys_a': g_phys_a,
        'g_phys_d': g_phys_d,
        'nocc': nocc,
        'nso': nso,
    }


def _assemble_AB(F_so, d_so, g_phys, nocc, nso, direct):
    """Build the (N_ov+1)×(N_ov+1) photon-augmented A and B matrices."""
    nvir = nso - nocc
    nov = nvir * nocc
    eps = np.diag(F_so)
    eps_occ = eps[:nocc]
    eps_vir = eps[nocc:]

    # Electronic A and B in (ai, bj) layout.
    A_ee = np.diag((eps_vir[:, None] - eps_occ[None, :]).reshape(-1))
    # ⟨ib||aj⟩ (or ⟨ib|aj⟩ in the direct case) → A_{ai,bj}.
    A_ee = A_ee + (g_phys[:nocc, nocc:, nocc:, :nocc]
                   .transpose(2, 0, 1, 3)
                   .reshape(nov, nov))
    # ⟨ij||ab⟩ (or ⟨ij|ab⟩) → B_{ai,bj}.
    B_ee = (g_phys[:nocc, :nocc, nocc:, nocc:]
            .transpose(2, 0, 3, 1)
            .reshape(nov, nov))

    # DSE contributions Δ, Δ'.
    d_vo = d_so[nocc:, :nocc]   # (a, i)
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

    A_tilde = A_ee + Delta
    B_tilde = B_ee + Delta_prime

    # g_{ai} = −√(ω/2) d_{ai}. The √(ω/2) prefactor is folded in by the
    # caller, so here we just return the bare d_{ai} block and let the
    # driver scale it when assembling the photon-augmented A / B.
    return A_tilde, B_tilde, d_vo.reshape(nov)


def run_qed_rpa(qedhf, direct=True, verbose=True):
    """Cavity QED random phase approximation correlation energy.

    Args:
        qedhf: dict returned by :func:`OmegaQMC.qed_hf.run_qed_hf`.
        direct: if True compute QED-dRPA (the ring channel only, formally
            equivalent to QED-drCCD); if False keep exchange / DSE
            exchange contributions, giving full QED-RPA.
        verbose: print a summary of the eigenvalue spectra.

    Returns:
        dict with the QED-RPA correlation energy, the corresponding
        total energy on top of QED-HF, the full and TDA-RWA excitation
        spectra, and the (N_ov+1)-dimensional A and B blocks.
    """
    omega = qedhf['omega']
    so = _build_spin_orbital_quantities(qedhf)
    nocc, nso = so['nocc'], so['nso']
    nvir = nso - nocc
    nov = nvir * nocc

    g_phys = so['g_phys_d'] if direct else so['g_phys_a']
    A_tilde, B_tilde, d_ai_flat = _assemble_AB(
        so['F_so'], so['d_so'], g_phys, nocc, nso, direct,
    )

    # Photon coupling vector g_{ai} = −√(ω/2) d_{ai}; counter-rotating term
    # g̃ coincides with g (Eq. 12 of the paper).
    g_vec = -math.sqrt(omega / 2.0) * d_ai_flat

    dim = nov + 1
    A_big = np.zeros((dim, dim))
    B_big = np.zeros((dim, dim))
    A_big[:nov, :nov] = A_tilde
    A_big[:nov, -1] = g_vec
    A_big[-1, :nov] = g_vec
    A_big[-1, -1] = omega
    B_big[:nov, :nov] = B_tilde
    B_big[:nov, -1] = g_vec
    B_big[-1, :nov] = g_vec
    # B_big[-1, -1] = 0 — no double-photon coupling at the RPA level.

    # TDA-RWA (Eq. 15) — hermitian.
    Omega_bar = la.eigh(A_big, eigvals_only=True)

    # Full QED-RPA (Eq. 34) — non-Hermitian, ±Ω pairs.
    M = np.zeros((2 * dim, 2 * dim))
    M[:dim, :dim] = A_big
    M[:dim, dim:] = B_big
    M[dim:, :dim] = -B_big
    M[dim:, dim:] = -A_big

    evals = la.eig(M, right=False)
    imag_max = float(np.max(np.abs(evals.imag)))
    real_max = float(np.max(np.abs(evals.real)))
    if imag_max > 1e-6 * max(real_max, 1.0):
        if verbose:
            print(f"  warning: complex QED-RPA eigenvalues detected, "
                  f"max |Im| = {imag_max:.3e}; the reference state may be "
                  f"unstable at this coupling strength.")
    real_vals = evals.real
    Omega_full = np.sort(real_vals[real_vals > 1e-10])
    if len(Omega_full) != dim:
        # Falls back to the upper half of the sorted spectrum; useful when
        # the ±Ω pairing has been broken by an instability.
        Omega_full = np.sort(real_vals)[-dim:]

    E_c = 0.5 * (np.sum(Omega_full) - np.sum(Omega_bar))
    E_qedhf = qedhf.get('E_qed_hf', qedhf.get('E_qed_uhf'))

    if verbose:
        flavour = 'QED-dRPA' if direct else 'QED-RPA'
        print(f"\n{flavour}:")
        print(f"  nocc (SO) = {nocc}, nvir (SO) = {nvir},"
              f"  N_ov+1 = {dim},  ω = {omega:.6f} Ha")
        print(f"  Tr(Ω)      = {float(np.sum(Omega_full)):.10f}")
        print(f"  Tr(Ω̄)      = {float(np.sum(Omega_bar)):.10f}")
        print(f"  E_corr     = {float(E_c):.15f}")
        print(f"  E_QED_HF   = {E_qedhf:.15f}")
        print(f"  E_total    = {E_qedhf + float(E_c):.15f}")

    return {
        'E_qed_rpa_corr': float(E_c),
        'E_qed_rpa_total': float(E_qedhf + E_c),
        'E_qed_hf': float(E_qedhf),
        'Omega_full': Omega_full,
        'Omega_tda_rwa': Omega_bar,
        'A_big': A_big,
        'B_big': B_big,
        'direct': bool(direct),
    }


if __name__ == '__main__':
    # Water / cc-pVDZ, geometry from Table I of arXiv:2602.09968:
    # R(O-H) = 1 Å, ∠HOH = 104.5°, C₂ axis along z. The molecule is
    # re-centred at its nuclear centre of mass (the Psi4 default
    # orientation of the reference) — required to reproduce the
    # published QED-dRPA values to all digits, since dRPA is not
    # origin-invariant (see module docstring).
    import numpy as _np
    half_angle = math.radians(104.5 / 2.0)
    rOH = 1.0
    hx = rOH * math.sin(half_angle)
    hz = -rOH * math.cos(half_angle)
    coords = _np.array([[0.0, 0.0, 0.0], [+hx, 0.0, hz], [-hx, 0.0, hz]])
    masses = _np.array([15.99491461957, 1.00782503224, 1.00782503224])
    coords -= (masses @ coords) / masses.sum()
    mol = gto.M(
        atom=[['O', tuple(coords[0])],
              ['H', tuple(coords[1])],
              ['H', tuple(coords[2])]],
        basis='cc-pVDZ',
        unit='Angstrom',
        symmetry=False,
        verbose=0,
    )

    # Cavity parameters from the paper (z-polarised, resonant with the
    # lowest dipole-allowed RPA excitation of water).
    omega = 0.415668
    lambda_z = 0.05
    lambda_cav = (0.0, 0.0, lambda_z)

    print(f"omega  = {omega:.6f} Ha")
    print(f"lambda = {lambda_cav}")

    qedhf = run_qed_hf(mol, omega, lambda_cav, verbose=False, tol=1e-12)
    print(f"\nE_QED_HF = {qedhf['E_qed_hf']:.15f}")
    print(f"E_RHF    = {qedhf['E_rhf']:.15f}  (pyscf, no cavity)")

    drpa = run_qed_rpa(qedhf, direct=True, verbose=True)
    rpa = run_qed_rpa(qedhf, direct=False, verbose=True)

    print("\nReference values from Table I of arXiv:2602.09968 "
          f"(λ_z = {lambda_z}):")
    print("  E_QED_HF   = -76.016355")
    print("  E_dRPA_corr = -0.234676")
