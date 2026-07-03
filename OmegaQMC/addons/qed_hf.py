"""
Reference: https://pubs.acs.org/doi/10.1021/jacs.1c13201
QED-Hartree-Fock in the dipole gauge (PySCF backend).

Self-consistent mean-field for the Pauli-Fierz Hamiltonian:

    H = H_elec + ω a†a + √(ω/2)(a + a†)(λ·d) + (1/2)(λ·d)²

The QED-HF Fock matrix in the dipole gauge is

    F = H_core + 2J - K
        + (λ·μ_AO) ⟨λ·μ⟩ + (1/2)<λλ:rr>_AO          # 1-electron DSE
        + 2 ⟨ρ, λ·μ⟩ (λ·μ_AO) − (λ·μ_AO) ρ (λ·μ_AO)  # 2-electron DSE

where ⟨λ·μ⟩ is the electronic dipole-moment expectation value, ρ is
the spatial density matrix, μ_AO are the position integrals
<i|r|j> and <λλ:rr>_AO are the second-moment integrals. PySCF does
not provide a built-in QED-HF, so we iterate the dressed Fock build
manually, seeding the orbitals with a plain RHF.

The output dict carries everything :func:`OmegaQMC.qed_ccsd.qed_ccsd`
needs to build the spin-orbital matter Hamiltonian and run the
DIIS-accelerated QED-CCSD iteration on top of this reference.

Sign convention: pyscf's ``int1e_rr`` returns positive
<i|x_a x_b|j>, whereas psi4's ``so_quadrupole`` used by the
DePrince/White reference code returns the same integrals with the
opposite sign (the "electronic moment" convention). To match the
published QED-CCSD-21 reference value we pre-flip the sign here so
the subsequent ``oei -= quadrupole`` line corresponds to the
physically correct +(1/2)<λλ:rr> Pauli-Fierz 1-electron DSE term.

Validation: with the glycolaldehyde / STO-3G demo at the bottom of
this file (ω = 3 eV, λ = (0, 0, 0.1)) the converged QED-HF energy
agrees with the reference psi4 implementation to ≲ 10⁻¹⁰ Ha, and the
λ = 0 limit reproduces pyscf's plain RHF.
"""

import numpy as np
from pyscf import gto, scf as pyscf_scf


def build_eri_df(mol, auxbasis):
    """Density-fitting 3-index factor ``B`` of the AO two-electron integrals.

    Returns ``B`` with shape ``(naux, nao, nao)`` such that the chemist-notation
    ERI factorises as ``(pq|rs) ≈ Σ_P B[P, p, q] · B[P, r, s]``. ``B`` is the
    Cholesky-fitted 3-center integral ``L⁻¹(P|pq)`` (metric ``(P|Q) = L Lᵀ``),
    so storage is ``naux·nao²`` instead of the dense ``nao⁴``.

    ``auxbasis`` is an explicit auxiliary/fitting basis name (e.g. ``'weigend'``,
    ``'def2-universal-jkfit'``, ``'cc-pvdz-jkfit'``); it is passed straight to
    PySCF's :func:`df.incore.cholesky_eri`.
    """
    from pyscf import df, lib
    cderi = df.incore.cholesky_eri(mol, auxbasis=auxbasis)  # (naux, nao*(nao+1)/2)
    return np.ascontiguousarray(lib.unpack_tril(cderi))     # (naux, nao, nao)


def eri_mo_transform(qedhf, Cp, Cq, Cr, Cs, dse=False):
    """AO→MO chemist tensor ``(ij|kl)`` from a QED-HF reference dict.

    Works for both the dense (``qedhf['eri_ao']``) and density-fitted
    (``qedhf['eri_df']``) representations produced by :func:`run_qed_hf` /
    :func:`OmegaQMC.qed_uhf.run_qed_uhf`, dispatching on which key is present.

    The returned array is indexed ``[i, j, k, l] = (ij|kl)`` in chemist
    notation, with ``i,j`` transformed by ``Cp,Cq`` and ``k,l`` by ``Cr,Cs``.

    With ``dse=True`` the dipole self-energy contribution
    ``Σ_X λ_X² μ_X ⊗ μ_X`` is folded in. Since each DSE term is a rank-1 outer
    product ``(λ_X μ_X) ⊗ (λ_X μ_X)``, it enters the DF path as three extra
    auxiliary vectors appended to ``B`` — no dense ``nao⁴`` tensor is ever built.
    """
    if dse:
        lam = qedhf['lambda_cav']
        mu = (qedhf['mu_x_ao'], qedhf['mu_y_ao'], qedhf['mu_z_ao'])

    if 'eri_df' in qedhf:
        B = qedhf['eri_df']
        if dse:
            dse_vecs = np.stack([lam[a] * mu[a] for a in range(3)])  # (3, nao, nao)
            B = np.concatenate([B, dse_vecs], axis=0)
        L = np.einsum('pi,Ppq,qj->Pij', Cp, B, Cq, optimize=True)
        R = np.einsum('rk,Prs,sl->Pkl', Cr, B, Cs, optimize=True)
        return np.einsum('Pij,Pkl->ijkl', L, R, optimize=True)

    eri = qedhf['eri_ao']
    if dse:
        eri = eri + sum(lam[a] * lam[a]
                        * np.einsum('pq,rs->pqrs', mu[a], mu[a], optimize=True)
                        for a in range(3))
    return np.einsum('pi,qj,pqrs,rk,sl->ijkl', Cp, Cq, eri, Cr, Cs, optimize=True)


def run_qed_hf(mol, omega, lambda_cav, max_iter=500, tol=1e-10, verbose=False,
               auxbasis=None):
    """Self-consistent QED-Hartree-Fock in the dipole gauge.

    Args:
        mol: PySCF :class:`gto.Mole` object (closed shell).
        omega: cavity frequency in Hartree.
        lambda_cav: 3-vector of coupling λ = (λx, λy, λz).
        max_iter: SCF iteration cap.
        tol: energy convergence threshold (on total energy).
        verbose: print per-iteration energies.
        auxbasis: if ``None`` (default) the exact dense ``nao⁴`` ERI is used and
            the result carries ``'eri_ao'`` — reproducing the reference energies
            to machine precision. If an auxiliary-basis name is given, the whole
            pipeline runs density-fitted: SCF J/K are built from the 3-index
            factor and the result carries ``'eri_df'`` (shape ``(naux, nao, nao)``)
            instead of the dense tensor, trading ~mHa DF error for ``nao⁴`` →
            ``naux·nao²`` memory. Consumers (qed_ccsd, qed_rpa) dispatch on the
            key via :func:`eri_mo_transform`.

    Returns:
        dict with the converged orbitals, dressed Fock, AO operators
        and other quantities consumed by :func:`OmegaQMC.qed_ccsd.qed_ccsd`.
    """
    lambda_x, lambda_y, lambda_z = (float(v) for v in lambda_cav)

    # --- Plain RHF (used purely as the starting guess for the orbitals) ---
    mf_rhf = pyscf_scf.RHF(mol)
    mf_rhf.verbose = 0
    E_rhf = mf_rhf.kernel()

    # --- AO-basis integrals from PySCF ---
    S = mol.intor('int1e_ovlp')
    T = mol.intor('int1e_kin')
    V = mol.intor('int1e_nuc')
    H_core = T + V
    nao = mol.nao_nr()
    use_df = auxbasis is not None
    if use_df:
        B_df = build_eri_df(mol, auxbasis)          # (naux, nao, nao)
        eri_ao = None
    else:
        eri_ao = mol.intor('int2e').reshape(nao, nao, nao, nao)
        B_df = None
    E_nuc = mol.energy_nuc()
    nocc = mol.nelec[0]
    assert mol.nelec[0] == mol.nelec[1], "qed_hf assumes a closed-shell molecule"

    # Symmetric orthogonalisation X = S^{-1/2}
    s_eig, s_vec = np.linalg.eigh(S)
    X = s_vec @ np.diag(s_eig ** -0.5) @ s_vec.T

    # Dipole / position integrals
    mu_xyz = mol.intor('int1e_r', comp=3)  # (3, nao, nao), pure <i|r|j>
    mu_x_ao, mu_y_ao, mu_z_ao = mu_xyz[0], mu_xyz[1], mu_xyz[2]

    # Quadrupole / second-moment integrals (sign-flipped vs raw int1e_rr;
    # see module docstring for the convention mismatch with psi4).
    rr = mol.intor('int1e_rr', comp=9).reshape(3, 3, nao, nao)
    quadrupole_x_lambda2_tot = -(0.5 * (
        lambda_x * lambda_x * rr[0, 0]
        + lambda_y * lambda_y * rr[1, 1]
        + lambda_z * lambda_z * rr[2, 2]
    ) + (
        lambda_x * lambda_y * rr[0, 1]
        + lambda_x * lambda_z * rr[0, 2]
        + lambda_y * lambda_z * rr[1, 2]
    ))

    dipole_x_lambda_tot = (lambda_x * mu_x_ao
                           + lambda_y * mu_y_ao
                           + lambda_z * mu_z_ao)

    # Initial density from RHF
    C = np.asarray(mf_rhf.mo_coeff)
    Cocc = C[:, :nocc]
    D = Cocc @ Cocc.T

    E_old = 0.0
    E_new = 0.0
    F = np.zeros_like(H_core)
    oei = np.zeros_like(H_core)

    for scf_iter in range(1, max_iter + 1):
        if use_df:
            gamma = np.einsum('Prs,rs->P', B_df, D, optimize=True)
            J = np.einsum('Ppq,P->pq', B_df, gamma, optimize=True)
            Kt = np.einsum('Pqs,rs->Pqr', B_df, D, optimize=True)
            K = np.einsum('Ppr,Pqr->pq', B_df, Kt, optimize=True)
        else:
            J = np.einsum('pqrs,rs->pq', eri_ao, D, optimize=True)
            K = np.einsum('prqs,rs->pq', eri_ao, D, optimize=True)

        F = H_core + 2.0 * J - K

        # Electronic dipole expectation (×2 for closed shell)
        mu_x_exp = np.einsum('pq,pq->', 2 * D, mu_x_ao, optimize=True)
        mu_y_exp = np.einsum('pq,pq->', 2 * D, mu_y_ao, optimize=True)
        mu_z_exp = np.einsum('pq,pq->', 2 * D, mu_z_ao, optimize=True)
        mu_lambda_tot = -(lambda_x * mu_x_exp
                          + lambda_y * mu_y_exp
                          + lambda_z * mu_z_exp)

        DSE = 0.5 * mu_lambda_tot * mu_lambda_tot

        oei = dipole_x_lambda_tot * mu_lambda_tot
        oei -= quadrupole_x_lambda2_tot
        F = F + oei

        scaled_mu = np.einsum('pq,pq->', D, dipole_x_lambda_tot, optimize=True)
        F = F + 2.0 * scaled_mu * dipole_x_lambda_tot
        F = F - np.einsum('pr,qs,rs->pq',
                          dipole_x_lambda_tot, dipole_x_lambda_tot, D,
                          optimize=True)

        E_new = (np.einsum('pq,pq->', (oei + H_core + F), D, optimize=True)
                 + E_nuc + DSE)

        if verbose:
            print('QED-HF iter %3d: E = %20.12f  dE = % .3E'
                  % (scf_iter, E_new, E_new - E_old))

        if abs(E_new - E_old) < tol:
            break
        E_old = E_new

        _, Ct = np.linalg.eigh(X @ F @ X)
        C = X @ Ct
        Cocc = C[:, :nocc]
        D = Cocc @ Cocc.T
    else:
        raise RuntimeError("QED-HF did not converge in %d iterations" % max_iter)

    result = {
        'E_qed_hf': float(E_new),
        'E_rhf': float(E_rhf),
        'E_nuc': float(E_nuc),
        'C': C,
        'F': F,
        'H_core': H_core,
        'oei': oei,
        'mu_x_ao': mu_x_ao,
        'mu_y_ao': mu_y_ao,
        'mu_z_ao': mu_z_ao,
        'dipole_x_lambda_tot': dipole_x_lambda_tot,
        'nocc_spatial': nocc,
        'nmo_spatial': C.shape[1],
        'lambda_cav': (lambda_x, lambda_y, lambda_z),
        'omega': float(omega),
        'mol': mol,
    }
    if use_df:
        result['eri_df'] = B_df
    else:
        result['eri_ao'] = eri_ao
    return result


if __name__ == '__main__':
    # Glycolaldehyde / STO-3G QED-HF reference (matches the geometry used
    # by the QED-CCSD demo in qed_ccsd.py).
    mol = gto.M(
        atom="""
            C   0.00000000   0.00000000   0.00000000
            O   0.00000000   1.23456800   0.00000000
            H   0.97075033  -0.54577032   0.00000000
            C  -1.21509881  -0.80991169   0.00000000
            H  -1.15288176  -1.89931439   0.00000000
            C  -2.43440063  -0.19144555   0.00000000
            H  -3.37262777  -0.75937214   0.00000000
            O  -2.62194056   1.12501165   0.00000000
            H  -1.71446384   1.51627790   0.00000000
        """,
        basis='STO-3G',
        unit='Angstrom',
        symmetry=False,
        verbose=0,
    )

    omega_eV = 3.0
    omega = omega_eV / 27.211386245988
    lambda_cav = (0.0, 0.0, 0.1)

    print(f"omega  = {omega_eV} eV  ({omega:.6f} Ha)")
    print(f"lambda = {lambda_cav}")

    qedhf = run_qed_hf(mol, omega, lambda_cav, verbose=True)
    print(f"\nE_QED_HF = {qedhf['E_qed_hf']:.15f}")
    print(f"E_RHF    = {qedhf['E_rhf']:.15f}  (pyscf, no cavity)")
    print(f"ΔE_cavity = {qedhf['E_qed_hf'] - qedhf['E_rhf']:+.6f} Ha")
