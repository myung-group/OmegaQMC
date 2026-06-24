"""
QED ring-coupled-cluster doubles (QED-rCCD) — iterative solver for the
Riccati-like residual equations of arXiv:2602.09968 Sec. II.C.

The cluster operator retains the same photon-augmented amplitude set as
the QED-RPA / QED-rCCD framework of the paper:

    T = (1/4) Σ_{ijab}    T^{2,0}_{ai,bj}  â†_a â†_b â_j â_i
        +     Σ_{ia}      T^{1,1}_{ai}     â†_a â_i  b†
        + (1/2)           T^{0,2}          b†  b†

(no pure-singles T^{1,0}, no electron-photon-pair T^{1,2}, T^{2,1}, …).
The "ring" simplification of the QED-CCD residual equations is, in
matrix form (paper Eq. 23, 26, 27):

    0 = B̃ + Ã T^{2,0} + T^{2,0} Ã + T^{2,0} B̃ T^{2,0}
        + T^{2,0} g̃ (T^{1,1})† + T^{1,1} g̃† T^{2,0}
        + T^{1,1} g† + g (T^{1,1})†                               (T^{2,0})

    0 = g + ω T^{1,1} + Ã T^{1,1} + T^{2,0} B̃ T^{1,1}
        + T^{0,2} g + T^{2,0} g
        + T^{1,1} g† T^{1,1} + T^{0,2} T^{2,0} g                  (T^{1,1})

    0 = 2 ω T^{0,2} + (T^{1,1})† g̃ T^{0,2} + T^{0,2} g̃† T^{1,1}
        + (T^{1,1})† g + g† T^{1,1} + (T^{1,1})† B̃ T^{1,1}        (T^{0,2})

with Ã = A + Δ, B̃ = B + Δ′ and g̃ = g (Eq. 12 of the paper).
Direct ("ring-drCCD") variant: ⟨pq||rs⟩ → ⟨pq|rs⟩ and Δ = Δ′ = d_{ai}d_{bj}.

Correlation energy (Eq. 29 / 30 of the paper):

    direct=False:  E_c = (1/4) Tr(B̃ T^{2,0}) + g† T^{1,1}
    direct=True :  E_c = (1/2) Tr(B̃ T^{2,0}) + g† T^{1,1}

The paper proves QED-drCCD ≡ QED-dRPA at the level of E_c, so the
direct-variant of this module is numerically equivalent (to machine
precision when both are converged) to :func:`qed_rpa.run_qed_rpa(..., direct=True)`.

Solver
------
Jacobi/DIIS-able iteration of the orbital-energy-diagonal CC update:

    T^{2,0}_new[ai,bj] = T^{2,0}[ai,bj] − R^{2,0}[ai,bj] / Δε^{(2)}_{ai,bj}
    T^{1,1}_new[ai]    = T^{1,1}[ai]    − R^{1,1}[ai]    / Δε^{(1,1)}_{ai}
    T^{0,2}_new        = T^{0,2}        − R^{0,2}        / (2 ω)

with Δε^{(2)}_{ai,bj} = ε_a + ε_b − ε_i − ε_j and Δε^{(1,1)}_{ai} =
ε_a − ε_i + ω. A simple level-shift (`level_shift`) is available for
hard cases.
"""

import math

import numpy as np
import scipy.linalg as la
from pyscf import gto

from .qed_hf import run_qed_hf
from .qed_rpa import _assemble_AB, _build_spin_orbital_quantities


def _assemble_AB_g(qedhf, direct):
    """Build the photon-augmented blocks Ã, B̃, g vector and ω used by
    the QED-rCCD residual equations."""
    omega_cav = qedhf['omega']
    so = _build_spin_orbital_quantities(qedhf)
    F_so = so['F_so']
    d_so = so['d_so']
    nocc = so['nocc']
    nso = so['nso']
    nvir = nso - nocc
    nov = nvir * nocc

    g_phys = so['g_phys_d'] if direct else so['g_phys_a']
    A_tilde, B_tilde, d_ai_flat = _assemble_AB(
        F_so, d_so, g_phys, nocc, nso, direct)
    g_vec = -math.sqrt(omega_cav / 2.0) * d_ai_flat

    eps = np.diag(F_so)
    eps_occ = eps[:nocc]
    eps_vir = eps[nocc:]
    # Orbital-energy denominators
    de_2 = (eps_vir[:, None, None, None] - eps_occ[None, :, None, None]
            + eps_vir[None, None, :, None] - eps_occ[None, None, None, :])
    de_2 = de_2.reshape(nov, nov)   # (ai, bj)
    de_11 = (eps_vir[:, None] - eps_occ[None, :]).reshape(nov) + omega_cav

    return {
        'A_tilde': A_tilde,
        'B_tilde': B_tilde,
        'g_vec': g_vec,                # shape (nov,)
        'omega_cav': float(omega_cav),
        'de_2': de_2,                  # (nov, nov)
        'de_11': de_11,                # (nov,)
        'nov': nov,
        'nocc': nocc,
        'nso': nso,
    }


def _residuals(A, B, g, omega, T2, T11, T02):
    """Evaluate the three ring-CCD residuals (paper Eq. 23, 26, 27).

    A, B  : (nov, nov) matrices (Ã, B̃)
    g     : (nov,)
    omega : float
    T2    : (nov, nov)  ←  T^{2,0} as a matrix
    T11   : (nov,)
    T02   : scalar
    """
    # T^{2,0} residual
    R2 = (B
          + A @ T2
          + T2 @ A
          + T2 @ B @ T2
          + np.outer(T2 @ g, T11)        # T^{2,0} g (T^{1,1})† — outer
          + np.outer(T11, g @ T2)        # T^{1,1} g† T^{2,0}
          + np.outer(T11, g)             # T^{1,1} g†
          + np.outer(g, T11))            # g (T^{1,1})†

    # T^{1,1} residual
    R11 = (g
           + omega * T11
           + A @ T11
           + T2 @ B @ T11
           + T02 * g
           + T2 @ g
           + (g @ T11) * T11             # T^{1,1} g† T^{1,1}  (g†T11 is scalar)
           + T02 * (T2 @ g))

    # T^{0,2} residual
    T11_g_T02 = (T11 @ g) * T02          # scalar
    T02_g_T11 = T02 * (g @ T11)
    T11_BT11 = T11 @ B @ T11
    R02 = (2.0 * omega * T02
           + T11_g_T02 + T02_g_T11
           + 2.0 * (T11 @ g)
           + T11_BT11)

    return R2, R11, R02


def run_qed_rccsd(qedhf, direct=True, max_iter=200, tol=1e-10,
                  level_shift=0.0, verbose=True):
    """Solve the QED-ring-CCD equations (T^{2,0}, T^{1,1}, T^{0,2}).

    Args:
        qedhf: dict from :func:`OmegaQMC.qed_hf.run_qed_hf`.
        direct: True (default) → QED-drCCD (uses ⟨pq|rs⟩ and Δ = d_{ai}d_{bj}).
            Provably equivalent to QED-dRPA at convergence.
            False → keep antisymmetric integrals (QED-rCCD); the
            correlation energy differs from QED-RPA in general.
        max_iter, tol: iteration cap / energy-change threshold.
        level_shift: optional positive shift added to the
            orbital-energy denominators (helps small/zero gaps).
        verbose: print iteration energies.

    Returns:
        dict with the correlation energy, total energy on QED-HF,
        the converged amplitudes, and iteration counts.
    """
    aux = _assemble_AB_g(qedhf, direct=direct)
    A = aux['A_tilde']
    B = aux['B_tilde']
    g = aux['g_vec']
    omega = aux['omega_cav']
    nov = aux['nov']
    de_2 = aux['de_2'] + level_shift
    de_11 = aux['de_11'] + level_shift
    de_02 = 2.0 * omega + level_shift

    # Initial amplitudes
    T2 = np.zeros((nov, nov))
    T11 = np.zeros(nov)
    T02 = 0.0

    e_old = 0.0
    e_prefactor = 0.5 if direct else 0.25  # paper Eq. 29 vs 30

    if verbose:
        flavour = 'QED-drCCD' if direct else 'QED-rCCD (antisym)'
        print(f"\n{flavour} (ring CC with photon channels)")
        print(f"  nov={nov}, ω={omega:.6f} Ha, "
              f"level_shift={level_shift}, tol={tol:.1e}")
        print(f"\n  {'iter':>4}  {'E_corr':>16}  {'|dE|':>10}  "
              f"{'|R2|':>10}  {'|R11|':>10}  {'|R02|':>10}")

    for it in range(1, max_iter + 1):
        R2, R11, R02 = _residuals(A, B, g, omega, T2, T11, T02)

        # Jacobi-style amplitude update
        T2 = T2 - R2 / de_2
        T11 = T11 - R11 / de_11
        T02 = T02 - R02 / de_02

        # Energy
        e_c = e_prefactor * np.einsum('ab,ba->', B, T2) + np.dot(g, T11)
        e_c = float(e_c)

        r2_norm = float(np.linalg.norm(R2))
        r11_norm = float(np.linalg.norm(R11))
        r02_norm = float(abs(R02))

        if verbose:
            print(f"  {it:>4d}  {e_c:16.12f}  {abs(e_c - e_old):.3e}"
                  f"  {r2_norm:.3e}  {r11_norm:.3e}  {r02_norm:.3e}")

        if abs(e_c - e_old) < tol and r2_norm + r11_norm + r02_norm < tol * 1e3:
            break
        e_old = e_c
    else:
        if verbose:
            print(f"  warning: QED-rCCD did not converge in {max_iter} iters")

    return {
        'method': 'QED-drCCD' if direct else 'QED-rCCD',
        'E_qed_rccd_corr': float(e_c),
        'E_qed_rccd_total': float(qedhf['E_qed_hf'] + e_c),
        'E_qed_hf': float(qedhf['E_qed_hf']),
        'T2': T2,
        'T11': T11,
        'T02': float(T02),
        'iterations': it,
        'direct': bool(direct),
        'omega_cav': float(omega),
    }


if __name__ == '__main__':
    from .qed_rpa import run_qed_rpa

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
        print('=' * 70)
        print(f'λ = {lam}')
        qedhf = run_qed_hf(mol, omega, (0, 0, lam), verbose=False)
        rcc = run_qed_rccsd(qedhf, direct=True, verbose=True, tol=1e-11)
        rpa = run_qed_rpa(qedhf, direct=True, verbose=False)
        diff = rcc['E_qed_rccd_corr'] - rpa['E_qed_rpa_corr']
        print(f"\n  E_QED-drCCD = {rcc['E_qed_rccd_corr']:.12f}")
        print(f"  E_QED-dRPA  = {rpa['E_qed_rpa_corr']:.12f}")
        print(f"  Δ           = {diff:+.3e} "
              f"(paper proves they are equal)")
