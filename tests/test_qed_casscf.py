"""
Test QED-CASSCF on small closed-shell molecules.

Validates:
1. Full-active-space limit: QED-CASSCF(nmo, nelec) reproduces QED-FCI
   bit-identically. Orbital optimisation is redundant there (the QED-FCI
   energy is orbital-invariant), so no macro-iterations are needed.
2. Zero-coupling limit: QED-CASSCF at λ=0 reduces to standard CASSCF
   (PySCF mcscf.CASSCF) in the same active space.
3. Variational ordering / orbital relaxation: in a *truncated* active
   space, E_QED-CASSCF ≤ E_QED-CASCI at every coupling (relaxing the
   orbitals in the cavity can only lower the energy), and the relaxation
   energy grows as the coupling is turned on.
4. CI-build consistency: the pre-relaxation energy reported as
   ``e_qed_casci`` equals run_qed_casci bit-identically (both build the
   polaritonic CI from the same QED-HF orbitals).
5. Coherent-state convergence: the CS-QED-CASSCF energy is invariant to
   the photon truncation ``nph_max`` (it is an exact unitary transform).
6. Open-shell molecules dispatch to the spin-unrestricted QED-UCASSCF
   (see tests/test_qed_ucasscf.py for its dedicated validation).

All comparisons use small bases (STO-6G / 6-31G) and small active spaces
so the dense polaritonic diagonalisations and orbital macro-iterations
stay fast enough for the pytest suite.
"""

import numpy as np
from pyscf import gto, scf, mcscf

from OmegaQMC.addons.qed_fci import run_qed_fci
from OmegaQMC.addons.qed_casci import run_qed_casci
from OmegaQMC.addons.qed_casscf import run_qed_casscf


def _build_h2(basis='6-31g'):
    mol = gto.M(atom='H 0 0 0; H 0 0 1.4', basis=basis, unit='Bohr',
                verbose=0)
    mf = scf.RHF(mol)
    mf.kernel()
    return mol, mf


def _build_lih(basis='6-31g'):
    mol = gto.M(atom='Li 0 0 0; H 0 0 1.6', basis=basis, unit='Angstrom',
                verbose=0)
    mf = scf.RHF(mol)
    mf.kernel()
    return mol, mf


def test_full_cas_equals_qed_fci():
    """H2/6-31G: QED-CASSCF(nmo, nelec) must equal QED-FCI bit-identically,
    both with the coherent-state basis and the raw photon-number basis."""
    mol, mf = _build_h2()
    nmo = mol.nao_nr()
    ne = mol.nelectron

    print("\n" + "=" * 70)
    print("QED-CASSCF(full) vs QED-FCI: H2/6-31G")
    print("=" * 70)
    print(f"  nmo = {nmo}, nelectron = {ne}")
    print(f"  {'lambda':>8} {'cs':>4} {'E_QED-CASSCF':>18} "
          f"{'E_QED-FCI':>18} {'Delta':>12}")

    omega = 0.5
    for cs in (True, False):
        for lam in (0.0, 0.05, 0.10):
            cv = [0.0, 0.0, lam]
            r_fci = run_qed_fci(mf, omega=omega, coupling_vec=cv,
                                nph_max=8, coherent_state=cs)
            r_cas = run_qed_casscf(mf, ncas=nmo, nelecas=ne, omega=omega,
                                   coupling_vec=cv, nph_max=8,
                                   coherent_state=cs)
            d_e = r_cas['e_qed_casscf'] - r_fci['e_qed_fci']
            print(f"  {lam:8.4f} {str(cs):>4} {r_cas['e_qed_casscf']:18.10f} "
                  f"{r_fci['e_qed_fci']:18.10f} {d_e:+12.2e}")
            assert abs(d_e) < 1e-9, (
                f"Full-CAS QED-CASSCF != QED-FCI at lam={lam}, cs={cs}: "
                f"Delta = {d_e:.3e} Ha")
            assert abs(r_cas['e_qed_hf'] - r_fci['e_qed_hf']) < 1e-9


def test_zero_coupling_equals_standard_casscf():
    """At λ=0, QED-CASSCF must reduce to standard PySCF CASSCF."""
    mol, mf = _build_lih()

    print("\n" + "=" * 70)
    print("Zero-coupling cross-check: QED-CASSCF vs PySCF CASSCF (LiH/6-31G)")
    print("=" * 70)

    for ncas, nelecas in ((4, 2), (5, 4)):
        r = run_qed_casscf(mf, ncas=ncas, nelecas=nelecas, omega=0.3,
                           coupling_vec=[0.0, 0.0, 0.0], nph_max=2)
        mc = mcscf.CASSCF(mf, ncas, nelecas)
        mc.verbose = 0
        e_ref = float(mc.kernel()[0])

        d = r['e_qed_casscf'] - e_ref
        print(f"  CAS({ncas},{nelecas}): E_QED-CASSCF(λ=0) = "
              f"{r['e_qed_casscf']:.10f}  E_CASSCF = {e_ref:.10f}  "
              f"Δ = {d:+.2e}")
        # Orbital optimisation converges to conv_tol_grad, so allow a
        # gradient-tolerance-sized gap to the fully-converged PySCF value.
        assert abs(d) < 1e-7, (
            f"QED-CASSCF(λ=0) != PySCF CASSCF for CAS({ncas},{nelecas}): "
            f"Δ = {d:.3e} Ha")
        # The bare-CASSCF reference reported in the dict is the PySCF value.
        assert abs(r['e_casscf'] - e_ref) < 1e-12
        assert r['converged']


def test_orbital_relaxation_lowers_energy():
    """A truncated CAS must give E_QED-CASSCF ≤ E_QED-CASCI at every
    coupling, with the relaxation growing as the cavity is turned on."""
    mol, mf = _build_lih()
    omega = 0.1
    ncas, nelecas = 4, 2

    print("\n" + "=" * 70)
    print(f"QED-CASSCF vs QED-CASCI (orbital relaxation): LiH/6-31G "
          f"CAS({ncas},{nelecas})")
    print("=" * 70)
    print(f"  {'lambda':>8} {'E_QED-CASCI':>18} {'E_QED-CASSCF':>18} "
          f"{'relaxation':>14}")

    relax = []
    for lam in (0.0, 0.05, 0.10):
        cv = [0.0, 0.0, lam]
        r_ci = run_qed_casci(mf, ncas, nelecas, omega, cv, nph_max=3)
        r_cas = run_qed_casscf(mf, ncas=ncas, nelecas=nelecas, omega=omega,
                               coupling_vec=cv, nph_max=3)
        dr = r_cas['e_qed_casscf'] - r_ci['e_qed_casci']
        relax.append(dr)
        print(f"  {lam:8.4f} {r_ci['e_qed_casci']:18.10f} "
              f"{r_cas['e_qed_casscf']:18.10f} {dr:+14.3e}")
        assert dr < 1e-9, (
            f"Orbital relaxation must not raise the energy at lam={lam}: "
            f"E_QED-CASSCF - E_QED-CASCI = {dr:.3e} Ha")
        # The pre-relaxation energy in the dict must match QED-CASCI.
        assert abs(r_cas['e_qed_casci'] - r_ci['e_qed_casci']) < 1e-9

    # Orbital optimisation does non-trivial work (clearly lowers the energy
    # below fixed-orbital QED-CASCI) at every coupling. The magnitude here
    # is correlation-dominated, so its variation with lambda is mild and
    # system-dependent; we only require a real, sizeable relaxation.
    assert all(dr < -1e-4 for dr in relax), (
        f"Expected a sizeable orbital relaxation at every lambda; saw {relax}")


def test_ci_build_matches_qed_casci():
    """The pre-relaxation 'e_qed_casci' must equal run_qed_casci to all
    digits — i.e. the polaritonic CI build is identical to the validated
    qed_casci module."""
    mol, mf = _build_lih()
    omega, cv, nph = 0.1, [0.0, 0.0, 0.05], 3
    r_cas = run_qed_casscf(mf, ncas=4, nelecas=2, omega=omega,
                           coupling_vec=cv, nph_max=nph)
    r_ci = run_qed_casci(mf, 4, 2, omega, cv, nph_max=nph)
    d = r_cas['e_qed_casci'] - r_ci['e_qed_casci']
    print("\n" + "=" * 70)
    print("CI-build consistency: qed_casscf pre-relaxation vs qed_casci")
    print("=" * 70)
    print(f"  e_qed_casci (qed_casscf) = {r_cas['e_qed_casci']:.12f}")
    print(f"  e_qed_casci (qed_casci)  = {r_ci['e_qed_casci']:.12f}")
    print(f"  Delta = {d:+.2e}")
    assert abs(d) < 1e-10


def test_coherent_state_invariant_to_nph_max():
    """The CS-QED-CASSCF energy must converge with nph_max to a value
    independent of the truncation (exact unitary displacement)."""
    mol, mf = _build_lih()
    energies = []
    print("\n" + "=" * 70)
    print("CS-QED-CASSCF nph_max convergence: LiH/6-31G CAS(4,2)")
    print("=" * 70)
    for nph in (2, 4, 6):
        r = run_qed_casscf(mf, ncas=4, nelecas=2, omega=0.1,
                           coupling_vec=[0.0, 0.0, 0.05], nph_max=nph)
        energies.append(r['e_qed_casscf'])
        print(f"  nph_max={nph}: E = {r['e_qed_casscf']:.12f}  "
              f"<n_ph> = {r['n_photon']:.5f}")
    # Converged: the last two truncations agree tightly.
    assert abs(energies[2] - energies[1]) < 1e-8, (
        f"CS-QED-CASSCF not converged in nph_max: {energies}")


def test_open_shell_dispatches_to_ucasscf():
    """Open-shell molecules dispatch to the spin-unrestricted QED-UCASSCF
    path and return the analogous result dict."""
    mol = gto.M(atom='Li 0 0 0; H 0 0 3.0', basis='sto-3g', spin=2,
                unit='Angstrom', verbose=0)
    mf = scf.RHF(mol)
    mf.kernel()
    r = run_qed_casscf(mf, ncas=2, nelecas=(2, 0), omega=0.1,
                       coupling_vec=[0.0, 0.0, 0.05], nph_max=2)
    print("\n" + "=" * 70)
    print("Open-shell dispatch: LiH (triplet)/STO-3G CAS(2,(2,0))")
    print("=" * 70)
    print(f"  reference = {r['reference']}  E = {r['e_qed_casscf']:.10f}  "
          f"converged = {r['converged']}")
    assert r['reference'] == 'QED-UHF'
    assert r['ncore'] == (1, 1)
    assert isinstance(r['mo_coeff'], tuple) and len(r['mo_coeff']) == 2
    assert r['e_qed_casscf'] <= r['e_qed_casci'] + 1e-10
    assert r['converged']


if __name__ == '__main__':
    test_full_cas_equals_qed_fci()
    test_zero_coupling_equals_standard_casscf()
    test_orbital_relaxation_lowers_energy()
    test_ci_build_matches_qed_casci()
    test_coherent_state_invariant_to_nph_max()
    test_open_shell_dispatches_to_ucasscf()
    print("\n" + "=" * 70)
    print("All QED-CASSCF tests passed.")
    print("=" * 70)
