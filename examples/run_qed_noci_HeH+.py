"""
QED-NOCI-6 for HeH+ in a cavity, compared with QED-FCI (OmegaQMC.addons.qed_fci).

System (Manderna, Vu, Foley, J. Chem. Phys. 161, 174105 (2024) setup):
    HeH+ / cc-pVDZ, R = 0.790 A (FCI/cc-pVDZ equilibrium), He at the origin, H on +z
    cavity hw = 26 eV (resonant with the S0 -> S1 charge-transfer excitation),
    lam = 0.1 a.u. || z (bond axis), gauge origin = nuclear center of mass

QED-NOCI-6 determinants, all converged in the cavity (notation |1s^H 1s^He 2s^H>):
    0     QED-RHF                          |0 2 0>
    1, 2  CT1 QED-BS-ROHF pair             |dn up 0>, |up dn 0>   (He 1s(a) H 1s(b) + spin inverse)
    3     ionic He2+ H-, QED-RHF + IMOM    |2 0 0>                (reference: atomic H- 1s^2)
    4, 5  CT2 QED-BS-ROHF pair             |0 up dn>, |0 dn up>   (QED-RHF MO 0 (a), MO 2 (b))
Each determinant I carries its own coherent shift z_I = <lam.d>_I / sqrt(2w); the spin pairs
(1, 2) and (4, 5) are combined into singlets. Hamiltonian dimension 6 (N + 1).

Polaritons: P0 (ground), LP and UP (lower / upper polariton = 2nd and 3rd singlet roots).
QED-FCI singlets are selected by <S^2> from the dense run_qed_fci eigenvectors.

Reference output (N = 20; QED-NOCI-6 values from the QED-NOCI paper draft, R0790 table):
    method        E_P0 / Eh    LP-P0 / eV   UP-P0 / eV   Rabi / eV
    QED-NOCI-6    -2.931512    24.589       27.590       3.001
    QED-FCI       -2.958562    24.626       27.554       2.928     (Manderna 2024: 2.922)

usage: python examples/run_qed_noci_HeH+.py   (about a minute; QED-FCI at N = 20 dominates)
"""

import numpy as np
from pyscf import gto, scf
from pyscf.fci import spin_op

from OmegaQMC.addons.qed_noci import QEDNOCIIntegrals, QEDNOCI, spin_flip
from OmegaQMC.addons.qed_noci_scf import QEDRHF, imom_rhf, run_qed_bs_rohf
from OmegaQMC.addons.qed_fci import run_qed_fci

EV = 27.211386
R = 0.790                      # Angstrom
BASIS = 'cc-pvdz'
OMEGA = 26.0 / EV              # Hartree
LAM = np.array([0., 0., 0.1])  # a.u., polarization || bond
NS = [0, 1, 2, 3, 4, 20]       # photon cutoffs for the QED-NOCI convergence table
N_FCI = 20
PAIRS = [(1, 2), (4, 5)]


def make_mol():
    return gto.M(atom=[('He', (0, 0, 0)), ('H', (0, 0, R))], basis=BASIS, unit='Angstrom',
                 charge=1, verbose=0)


def atomic_orbital(atom, charge, spin, nao, sl):
    """Lowest atomic SCF orbital of `atom` embedded in the molecular AO slice `sl`."""
    m = gto.M(atom=f'{atom} 0 0 0', basis=BASIS, charge=charge, spin=spin, verbose=0)
    mf = (scf.RHF if spin == 0 else scf.ROHF)(m).run(conv_tol=1e-12)
    c = np.zeros((nao, 1))
    c[sl] = mf.mo_coeff[:, :1]
    return c


def noci6_dets(mol):
    nao = mol.nao
    a0, a1 = mol.aoslice_by_atom()[:, 2:4]
    he, h = slice(a0[0], a0[1]), slice(a1[0], a1[1])
    none = np.zeros((nao, 0))

    rhf = QEDRHF(mol, OMEGA, LAM)
    rhf.verbose, rhf.conv_tol = 0, 1e-12
    rhf.kernel()
    assert rhf.converged
    c_hf = rhf.mo_coeff[:, :1]

    ct1 = run_qed_bs_rohf(mol, OMEGA, LAM, none,
                          atomic_orbital('He', 0, 0, nao, he), atomic_orbital('H', 0, 1, nao, h))
    ref = atomic_orbital('H', -1, 0, nao, h)
    ion = imom_rhf(QEDRHF(mol, OMEGA, LAM), ref)
    ion.verbose, ion.conv_tol, ion.max_cycle = 0, 1e-12, 300
    ion.kernel(dm0=2 * ref @ ref.T)
    c_ion = ion.mo_coeff[:, ion.mo_occ > 1]
    ct2 = run_qed_bs_rohf(mol, OMEGA, LAM, none, rhf.mo_coeff[:, 0:1], rhf.mo_coeff[:, 2:3])
    for name, mf in (('CT1', ct1), ('ionic', ion), ('CT2', ct2)):
        assert mf.converged, f'{name} SCF did not converge'
    scf_e = {'QED-RHF': rhf.e_tot, 'CT1 BS-ROHF': ct1.e_tot, 'ionic IMOM': ion.e_tot,
             'CT2 BS-ROHF': ct2.e_tot}
    dets = [(c_hf, c_hf), ct1.det(), spin_flip(ct1.det()), (c_ion, c_ion),
            ct2.det(), spin_flip(ct2.det())]
    return dets, scf_e


def fci_singlets(mol, nroots=3):
    """P0, LP, UP from OmegaQMC's dense QED-FCI (singlets by <S^2>)."""
    mf = scf.RHF(mol).run(conv_tol=1e-12)
    res = run_qed_fci(mf, OMEGA, LAM, nph_max=N_FCI)
    norb, nelec, ne = mol.nao, mol.nelec, res['ndim_elec']
    na = int(round(np.sqrt(ne))) if nelec[0] == nelec[1] else None
    out = []
    for k, e in enumerate(res['eigenvalues']):
        v = res['eigenvectors'][:, k].reshape(N_FCI + 1, na, ne // na)
        s2 = sum(np.vdot(c, spin_op.contract_ss(c, norb, nelec)) for c in v)
        if s2 < .5:
            out.append(e)
            if len(out) == nroots:
                break
    return np.array(out), res['ndim_total']


def main():
    mol = make_mol()
    print(f'HeH+ {BASIS} R = {R} A, hw = {OMEGA * EV:.1f} eV, lam = {LAM[2]} || z, origin COM')
    dets, scf_e = noci6_dets(mol)
    ints = QEDNOCIIntegrals(mol, OMEGA, LAM)
    print('determinant SCF energies (Eh): '
          + ', '.join(f'{k} {v:.8f}' for k, v in scf_e.items()))
    print('z_I: ' + ' '.join(f'{ints.z_of(d):+.4f}' for d in dets))

    print('\nQED-NOCI-6 photon convergence (singlets)')
    print(f'{"N":>3s} {"dim":>4s} {"E_P0 / Eh":>12s} {"LP-P0/eV":>9s} {"UP-P0/eV":>9s} {"Rabi/eV":>8s}'
          f' {"<b+b> P0":>9s} {"LP":>7s} {"UP":>7s}')
    for n in NS:
        q = QEDNOCI(ints, dets, nph=n)
        q.kernel(spin_pairs=PAIRS, sign=+1)
        e = q.e[:3]
        x = (e[1:] - e[0]) * EV
        print(f'{n:3d} {q.dim:4d} {e[0]:12.6f} {x[0]:9.3f} {x[1]:9.3f} {x[1] - x[0]:8.3f}'
              f' {q.nphoton[0]:9.4f} {q.nphoton[1]:7.4f} {q.nphoton[2]:7.4f}')
    e_noci = e

    ef, dim_fci = fci_singlets(mol)
    print(f'\nN = {N_FCI}        {"E_P0 / Eh":>12s} {"LP-P0/eV":>9s} {"UP-P0/eV":>9s} {"Rabi/eV":>8s}  dim')
    for name, e, dim in (('QED-NOCI-6', e_noci, 6 * (N_FCI + 1)), ('QED-FCI', ef, dim_fci)):
        x = (e[1:] - e[0]) * EV
        print(f'{name:12s} {e[0]:12.6f} {x[0]:9.3f} {x[1]:9.3f} {x[1] - x[0]:8.3f}  {dim}')
    d = (e_noci - ef)
    print(f'NOCI - FCI: P0 {d[0] * 1e3:+.2f} mEh, LP {(d[1] - d[0]) * EV:+.3f} eV, '
          f'UP {(d[2] - d[0]) * EV:+.3f} eV')


if __name__ == '__main__':
    main()
