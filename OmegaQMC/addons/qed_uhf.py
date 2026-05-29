"""
Reference: https://pubs.acs.org/doi/10.1021/jacs.1c13201
QED-Unrestricted-Hartree-Fock in the dipole gauge (PySCF backend).

This is the open-shell (spin-unrestricted) analogue of
:mod:`OmegaQMC.addons.qed_hf`, which is restricted (RHF). It is a
self-consistent mean-field for the Pauli-Fierz Hamiltonian

    H = H_elec + ω a†a + √(ω/2)(a + a†)(λ·d) + (1/2)(λ·d)²

solved with independent α and β spin orbitals. The QED-UHF Fock
matrices in the dipole gauge are, for each spin σ ∈ {α, β},

    F_σ = H_core + J[ρ_α+ρ_β] - K[ρ_σ]
          + (λ·μ_AO) ⟨λ·μ⟩ + (1/2)<λλ:rr>_AO          # 1-electron DSE
          + ⟨ρ, λ·μ⟩ (λ·μ_AO) − (λ·μ_AO) ρ_σ (λ·μ_AO)  # 2-electron DSE

where J is built from the *total* density ρ = ρ_α + ρ_β, the exchange
K is built from the same-spin density ρ_σ, ⟨λ·μ⟩ is the electronic
dipole-moment expectation value over the total density, μ_AO are the
position integrals <i|r|j> and <λλ:rr>_AO are the second-moment
integrals.

As in the dipole gauge, the dipole-expectation ("⟨λ·μ⟩") pieces of the
one-electron DSE and the two-electron DSE Coulomb term cancel exactly,
so the net cavity contribution to each spin Fock matrix is the
one-body quadrupole (1/2)<λλ:rr> plus a spin-resolved DSE exchange
−(λ·μ_AO) ρ_σ (λ·μ_AO). The implementation keeps the two pieces
explicit (rather than pre-cancelling) so the code reads as a direct
generalisation of the RHF reference in :mod:`OmegaQMC.addons.qed_hf`
and reduces to it term-by-term when ρ_α = ρ_β.

PySCF does not provide a built-in QED-UHF, so we iterate the dressed
Fock build manually, seeding the orbitals with a plain UHF.

The output dict mirrors :func:`OmegaQMC.addons.qed_hf.run_qed_hf` but
with spin-resolved orbitals (``Ca``/``Cb``), Fock matrices
(``Fa``/``Fb``) and occupations (``nocc_a``/``nocc_b``). It also
reports the spin expectation ``<S^2>`` and multiplicity, which gauge
the spin contamination of the unrestricted solution.

Sign convention: pyscf's ``int1e_rr`` returns positive
<i|x_a x_b|j>, whereas psi4's ``so_quadrupole`` used by the
DePrince/White reference code returns the same integrals with the
opposite sign (the "electronic moment" convention). To match the
published QED-HF reference value we pre-flip the sign here so the
subsequent ``oei -= quadrupole`` line corresponds to the physically
correct +(1/2)<λλ:rr> Pauli-Fierz 1-electron DSE term.

Validation: for a closed-shell molecule QED-UHF reproduces the
QED-RHF energy of :func:`OmegaQMC.addons.qed_hf.run_qed_hf` to ≲ 10⁻¹⁰
Ha (the unrestricted solution collapses to the restricted one), and
the λ = 0 limit reproduces pyscf's plain UHF.
"""

import numpy as np
from pyscf import gto, scf as pyscf_scf


def _spin_square(Ca, Cb, nocc_a, nocc_b, S):
    """<S^2> and spin multiplicity (2S+1) for a UHF determinant.

    Uses the standard expression

        <S^2> = Sz(Sz+1) + N_β − Σ_{i∈occα, j∈occβ} |⟨ψ_i^α|ψ_j^β⟩|²

    with Sz = (N_α − N_β)/2 and the orbital overlaps taken in the AO
    metric S.
    """
    Ca_occ = Ca[:, :nocc_a]
    Cb_occ = Cb[:, :nocc_b]
    s_ab = Ca_occ.T @ S @ Cb_occ          # (nocc_a, nocc_b)
    n_ab = np.einsum('ij,ij->', s_ab, s_ab, optimize=True)
    sz = 0.5 * (nocc_a - nocc_b)
    ss = sz * (sz + 1.0) + nocc_b - n_ab
    mult = np.sqrt(1.0 + 4.0 * ss)        # <S^2> = S(S+1) ⇒ 2S+1 = √(1+4<S^2>)
    return float(ss), float(mult)


def run_qed_uhf(mol, omega, lambda_cav, max_iter=500, tol=1e-10, verbose=False):
    """Self-consistent QED-Unrestricted-Hartree-Fock in the dipole gauge.

    Args:
        mol: PySCF :class:`gto.Mole` object (open or closed shell).
        omega: cavity frequency in Hartree.
        lambda_cav: 3-vector of coupling λ = (λx, λy, λz).
        max_iter: SCF iteration cap.
        tol: energy convergence threshold (on total energy).
        verbose: print per-iteration energies.

    Returns:
        dict with the converged spin-resolved orbitals, dressed Fock
        matrices, AO operators and other quantities. The keys mirror
        :func:`OmegaQMC.addons.qed_hf.run_qed_hf` but α/β resolved.
    """
    lambda_x, lambda_y, lambda_z = (float(v) for v in lambda_cav)

    # --- Plain UHF (used purely as the starting guess for the orbitals) ---
    mf_uhf = pyscf_scf.UHF(mol)
    mf_uhf.verbose = 0
    E_uhf = mf_uhf.kernel()

    # --- AO-basis integrals from PySCF ---
    S = mol.intor('int1e_ovlp')
    T = mol.intor('int1e_kin')
    V = mol.intor('int1e_nuc')
    H_core = T + V
    nao = mol.nao_nr()
    eri_ao = mol.intor('int2e').reshape(nao, nao, nao, nao)
    E_nuc = mol.energy_nuc()
    nocc_a, nocc_b = mol.nelec  # (nalpha, nbeta)

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

    # Initial densities from UHF
    Ca, Cb = (np.asarray(c) for c in mf_uhf.mo_coeff)
    Da = Ca[:, :nocc_a] @ Ca[:, :nocc_a].T
    Db = Cb[:, :nocc_b] @ Cb[:, :nocc_b].T

    E_old = 0.0
    E_new = 0.0
    Fa = np.zeros_like(H_core)
    Fb = np.zeros_like(H_core)
    oei = np.zeros_like(H_core)
    mo_energy_a = np.zeros(nao)
    mo_energy_b = np.zeros(nao)

    for scf_iter in range(1, max_iter + 1):
        Dt = Da + Db

        # Coulomb from the total density; exchange from the same-spin density
        J = np.einsum('pqrs,rs->pq', eri_ao, Dt, optimize=True)
        Ka = np.einsum('prqs,rs->pq', eri_ao, Da, optimize=True)
        Kb = np.einsum('prqs,rs->pq', eri_ao, Db, optimize=True)

        Fa = H_core + J - Ka
        Fb = H_core + J - Kb

        # Electronic dipole expectation over the total density
        mu_x_exp = np.einsum('pq,pq->', Dt, mu_x_ao, optimize=True)
        mu_y_exp = np.einsum('pq,pq->', Dt, mu_y_ao, optimize=True)
        mu_z_exp = np.einsum('pq,pq->', Dt, mu_z_ao, optimize=True)
        mu_lambda_tot = -(lambda_x * mu_x_exp
                          + lambda_y * mu_y_exp
                          + lambda_z * mu_z_exp)

        DSE = 0.5 * mu_lambda_tot * mu_lambda_tot

        # One-electron DSE (spin-independent): dipole-expectation + quadrupole
        oei = dipole_x_lambda_tot * mu_lambda_tot
        oei -= quadrupole_x_lambda2_tot
        Fa = Fa + oei
        Fb = Fb + oei

        # Two-electron DSE: Coulomb-like (total density) + exchange-like
        # (same-spin density). The Coulomb-like piece cancels the
        # dipole-expectation part of oei at self-consistency; it is kept
        # explicit so the build mirrors the RHF reference.
        scaled_mu = np.einsum('pq,pq->', Dt, dipole_x_lambda_tot, optimize=True)
        Fa = Fa + scaled_mu * dipole_x_lambda_tot
        Fb = Fb + scaled_mu * dipole_x_lambda_tot
        Fa = Fa - np.einsum('pr,qs,rs->pq',
                            dipole_x_lambda_tot, dipole_x_lambda_tot, Da,
                            optimize=True)
        Fb = Fb - np.einsum('pr,qs,rs->pq',
                            dipole_x_lambda_tot, dipole_x_lambda_tot, Db,
                            optimize=True)

        # Energy. The (1/2)Tr[D F] form would under-count the one-body
        # DSE (oei) by half, so the missing half Tr[oei Dt] is added back.
        # Reduces exactly to the RHF expression when Da = Db.
        E_new = (0.5 * (np.einsum('pq,pq->', H_core, Dt, optimize=True)
                        + np.einsum('pq,pq->', Fa, Da, optimize=True)
                        + np.einsum('pq,pq->', Fb, Db, optimize=True))
                 + 0.5 * np.einsum('pq,pq->', oei, Dt, optimize=True)
                 + E_nuc + DSE)

        if verbose:
            print('QED-UHF iter %3d: E = %20.12f  dE = % .3E'
                  % (scf_iter, E_new, E_new - E_old))

        if abs(E_new - E_old) < tol:
            break
        E_old = E_new

        mo_energy_a, Ca_t = np.linalg.eigh(X @ Fa @ X)
        mo_energy_b, Cb_t = np.linalg.eigh(X @ Fb @ X)
        Ca = X @ Ca_t
        Cb = X @ Cb_t
        Da = Ca[:, :nocc_a] @ Ca[:, :nocc_a].T
        Db = Cb[:, :nocc_b] @ Cb[:, :nocc_b].T
    else:
        raise RuntimeError("QED-UHF did not converge in %d iterations" % max_iter)

    ss, mult = _spin_square(Ca, Cb, nocc_a, nocc_b, S)

    return {
        'E_qed_uhf': float(E_new),
        'E_uhf': float(E_uhf),
        'E_nuc': float(E_nuc),
        'Ca': Ca,
        'Cb': Cb,
        'Fa': Fa,
        'Fb': Fb,
        'mo_energy_a': mo_energy_a,
        'mo_energy_b': mo_energy_b,
        'H_core': H_core,
        'oei': oei,
        'eri_ao': eri_ao,
        'mu_x_ao': mu_x_ao,
        'mu_y_ao': mu_y_ao,
        'mu_z_ao': mu_z_ao,
        'dipole_x_lambda_tot': dipole_x_lambda_tot,
        'nocc_a': nocc_a,
        'nocc_b': nocc_b,
        'nmo_spatial': Ca.shape[1],
        's_squared': ss,
        'multiplicity': mult,
        'lambda_cav': (lambda_x, lambda_y, lambda_z),
        'omega': float(omega),
        'mol': mol,
    }


if __name__ == '__main__':
    # --- Open-shell demo: triplet O2 / STO-3G in a cavity ---
    mol = gto.M(
        atom='O 0 0 0; O 0 0 1.208',
        basis='STO-3G',
        spin=2,            # triplet ground state (N_alpha - N_beta = 2)
        unit='Angstrom',
        symmetry=False,
        verbose=0,
    )

    omega_eV = 3.0
    omega = omega_eV / 27.211386245988
    lambda_cav = (0.0, 0.0, 0.1)

    print(f"omega  = {omega_eV} eV  ({omega:.6f} Ha)")
    print(f"lambda = {lambda_cav}")

    qeduhf = run_qed_uhf(mol, omega, lambda_cav, verbose=True)
    print(f"\nE_QED_UHF = {qeduhf['E_qed_uhf']:.15f}")
    print(f"E_UHF     = {qeduhf['E_uhf']:.15f}  (pyscf, no cavity)")
    print(f"ΔE_cavity = {qeduhf['E_qed_uhf'] - qeduhf['E_uhf']:+.6f} Ha")
    print(f"<S^2>     = {qeduhf['s_squared']:.6f}  (2S+1 = {qeduhf['multiplicity']:.4f})")

    # --- Closed-shell consistency check: QED-UHF must match QED-RHF ---
    from OmegaQMC.addons.qed_hf import run_qed_hf

    mol_cs = gto.M(
        atom='O 0 0 0; H 0 0 0.96; H 0 0.93 -0.24',
        basis='STO-3G',
        unit='Angstrom',
        symmetry=False,
        verbose=0,
    )
    rhf = run_qed_hf(mol_cs, omega, lambda_cav)
    uhf = run_qed_uhf(mol_cs, omega, lambda_cav)
    print("\nClosed-shell H2O / STO-3G consistency check:")
    print(f"E_QED_RHF = {rhf['E_qed_hf']:.15f}")
    print(f"E_QED_UHF = {uhf['E_qed_uhf']:.15f}")
    print(f"difference = {abs(rhf['E_qed_hf'] - uhf['E_qed_uhf']):.3E} Ha "
          f"(<S^2> = {uhf['s_squared']:.2E})")
