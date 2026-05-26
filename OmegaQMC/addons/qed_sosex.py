"""
Cavity QED second-order screened exchange (QED-SOSEX) on top of
QED-dRPA / QED-drCCD.

Background
----------
The dRPA correlation energy can be written as

    E_c^dRPA = (1/2) Σ_{ai,bj} v^dir_{ai,bj} T^{drCCD}_{ai,bj}                (Eq. 30)

with v^dir = ⟨ab|ij⟩ + d_{ai} d_{bj} the bare direct interaction
(electronic Coulomb + DSE direct) and T^{drCCD} the converged
ring-CCD doubles amplitudes. The same dRPA energy can be obtained from
the QED-dRPA eigenvalue problem (paper proves E_c^drCCD = E_c^dRPA).

The well-defined "exchange correction" beyond dRPA that does NOT
suffer the double-counting that breaks GW with antisymmetric W is the
**second-order screened exchange (SOSEX)** of Grüneis, Marsman,
Harl, Schimka & Kresse (J. Chem. Phys. 131, 154115 (2009)). The
SOSEX energy contracts the dRPA amplitude with the *antisymmetric*
bare integral, NOT with the bare exchange alone:

    E_c^SOSEX = − (1/2) Σ_{ai,bj} v^ex_{ai,bj} T^{drCCD}_{ai,bj}                (electronic)

where v^ex_{ai,bj} is the exchange counterpart of v^dir, i.e. the
"swapped-index" matrix element ⟨ab|ji⟩ + d_{aj} d_{bi}. The total

    E_c^{dRPA+SOSEX} = E_c^dRPA + E_c^SOSEX

is the standard exchange-corrected dRPA correlation energy. In the
spatial closed-shell singlet formulation (Henderson & Scuseria,
J. Chem. Phys. 132, 234110 (2010)) this equals the singlet
antisymmetric ring-CCD energy. The spin-orbital antisymmetric ring-CCD
implemented in :mod:`qed_rccsd` (``direct=False``) additionally
includes triplet-channel contributions, so the two will *not* be
numerically identical — they correspond to different spin restrictions
of the same diagrammatic series.
Importantly, SOSEX adds exchange consistently inside the
amplitude contraction — no double counting with the Fock exchange
already built into ε^HF and no Σ_x^HF on the LHS of the QP equation.

QED extension
-------------
In the QED-dRPA framework the direct bare interaction acquires the
DSE block d_{ai} d_{bj} (paper Eq. 13 with exchange dropped); the
photon channel is absorbed automatically because the same QED-dRPA
amplitudes T^{drCCD} are used. The SOSEX exchange kernel that we
contract with — v^ex — is the antisymmetric counterpart of the
*augmented* (Coulomb + DSE) direct interaction:

    v^ex_{ai,bj} = ⟨ab|ji⟩  +  d_{aj} d_{bi}

so that v^dir − v^ex is precisely the antisymmetric form
⟨ab||ij⟩ + d_{ai}d_{bj} − d_{aj}d_{bi} of arXiv:2602.09968
(antisymmetric Δ block).

Output
------
* ``E_c^dRPA``           — paper Eq. 30 evaluated from T^{drCCD}.
* ``E_c^SOSEX``          — the second-order screened-exchange correction.
* ``E_c^{dRPA+SOSEX}``   — sum; equals E_c^rCCD (antisymmetric ring CCD)
                           in the electronic case, and is a *bona-fide*
                           "GW + exchange in W" correlation energy in
                           the QED case.

What this is NOT
----------------
SOSEX corrects the **correlation energy**, not the GW quasiparticle
self-energy. A full Σ_x-vertex correction to the QP equation (the
"GWΓ" approach in the strict diagrammatic sense) requires also
modifying Σ_c with a screened-exchange term Σ_SX that is paired with a
matching vertex piece — that's a substantial separate development and
is *not* implemented here. So:

* IP / EA via this module: don't use the SOSEX-corrected number as a
  better QP energy — it isn't one.
* Total correlation energy: this *is* an exchange-corrected,
  double-counting-free improvement over QED-dRPA.
"""

import math

import numpy as np
from pyscf import gto

from .qed_hf import run_qed_hf
from .qed_rccsd import run_qed_rccsd
from .qed_rpa import run_qed_rpa, _build_spin_orbital_quantities


def _build_v_dir_ex(qedhf):
    """Direct (v^dir) and exchange (v^ex) bare 2e blocks in the
    (ai, bj) channel for QED-dRPA-style screening.

    v^dir_{ai,bj} = ⟨ab|ij⟩_phys + d_{ai} d_{bj}
    v^ex_{ai,bj}  = ⟨ab|ji⟩_phys + d_{aj} d_{bi}

    Both have shape (nov, nov) with composite index (a, i) packed
    in C order matching the rest of the code.
    """
    so = _build_spin_orbital_quantities(qedhf)
    nocc = so['nocc']
    nso = so['nso']
    nvir = nso - nocc
    nov = nvir * nocc
    g_phys_d = so['g_phys_d']
    d_so = so['d_so']
    d_vo = d_so[nocc:, :nocc]
    d_ov = d_so[:nocc, nocc:]

    # v^dir[a,i,b,j] = ⟨ab|ij⟩ + d_ai d_bj
    # ⟨ab|ij⟩ is g_phys_d[a, b, i, j] → transpose (a, b, i, j) → (a, i, b, j)
    v_dir = (g_phys_d[nocc:, nocc:, :nocc, :nocc].transpose(0, 2, 1, 3).copy()
             + np.einsum('ai,bj->aibj', d_vo, d_vo))
    # v^ex[a,i,b,j] = ⟨ab|ji⟩ + d_aj d_bi
    # ⟨ab|ji⟩ is g_phys_d[a, b, j, i] → transpose (a, b, j, i) → (a, i, b, j)
    # = transpose(0, 3, 1, 2)
    v_ex = (g_phys_d[nocc:, nocc:, :nocc, :nocc].transpose(0, 3, 1, 2).copy()
            + np.einsum('aj,bi->aibj', d_vo, d_vo))

    return v_dir.reshape(nov, nov), v_ex.reshape(nov, nov)


def run_qed_sosex(qedhf, verbose=True, **rccd_kwargs):
    """Compute QED-dRPA + SOSEX correlation energy.

    Args:
        qedhf: dict from :func:`OmegaQMC.qed_hf.run_qed_hf`.
        verbose: print summary.
        **rccd_kwargs: forwarded to :func:`qed_rccsd.run_qed_rccsd`
            (e.g. ``max_iter``, ``tol``, ``level_shift``).

    Returns:
        dict with E_c^dRPA, E_c^SOSEX, E_c^{dRPA+SOSEX} and the
        underlying T^{drCCD} amplitudes.
    """
    # Solve QED-drCCD to get the amplitudes T^{2,0}, T^{1,1}, T^{0,2}.
    rccd = run_qed_rccsd(qedhf, direct=True, verbose=False, **rccd_kwargs)
    T2 = rccd['T2']
    T11 = rccd['T11']

    # Bare 2e blocks (direct / exchange) — used for both the dRPA and
    # SOSEX energy contractions.
    v_dir, v_ex = _build_v_dir_ex(qedhf)

    # Paper Eq. 30:   E_c^dRPA = (1/2) Tr(B̃ T^{2,0}) + g† T^{1,1}
    # where B̃ already contains the DSE direct piece. v_dir built above
    # is essentially the same direct block — using it gives the same
    # number (verified against run_qed_rpa).
    omega_cav = float(qedhf['omega'])
    so = _build_spin_orbital_quantities(qedhf)
    nocc = so['nocc']
    d_so = so['d_so']
    d_vo = d_so[nocc:, :nocc]
    g_vec = -math.sqrt(omega_cav / 2.0) * d_vo.reshape(-1)

    E_c_dRPA = 0.5 * np.einsum('ab,ba->', v_dir, T2) + float(np.dot(g_vec, T11))

    # SOSEX correction:  E_c^SOSEX = − (1/2) Tr(v^ex T^{drCCD})
    E_c_SOSEX = -0.5 * float(np.einsum('ab,ba->', v_ex, T2))

    E_c_total = E_c_dRPA + E_c_SOSEX
    E_total = float(qedhf['E_qed_hf']) + E_c_total

    if verbose:
        print(f"\nQED-dRPA + SOSEX correlation energy")
        print(f"  ω_cav = {omega_cav:.6f} Ha")
        print(f"  E_QED-HF              = {qedhf['E_qed_hf']:.10f}")
        print(f"  E_c^dRPA              = {E_c_dRPA:+.10f}")
        print(f"  E_c^SOSEX             = {E_c_SOSEX:+.10f}")
        print(f"  E_c^(dRPA+SOSEX)      = {E_c_total:+.10f}")
        print(f"  E_total^(dRPA+SOSEX)  = {E_total:.10f}")
        # Cross check against direct QED-dRPA:
        rpa = run_qed_rpa(qedhf, direct=True, verbose=False)
        print(f"  cross-check |E_c^dRPA − qed_rpa.E_c| = "
              f"{abs(E_c_dRPA - rpa['E_qed_rpa_corr']):.3e}")

    return {
        'method': 'QED-dRPA+SOSEX',
        'E_c_dRPA': float(E_c_dRPA),
        'E_c_SOSEX': float(E_c_SOSEX),
        'E_c_total': float(E_c_total),
        'E_total': float(E_total),
        'E_qed_hf': float(qedhf['E_qed_hf']),
        'T2': T2,
        'T11': T11,
        'T02': rccd['T02'],
        'iterations_rccd': rccd['iterations'],
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
        print('=' * 70)
        print(f'λ = {lam}')
        qedhf = run_qed_hf(mol, omega, (0, 0, lam), verbose=False)
        run_qed_sosex(qedhf, verbose=True, tol=1e-11, max_iter=500)
