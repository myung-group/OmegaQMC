"""
Test QED-FCI on H2 and H2O molecules.

Validates:
1. Zero coupling: QED-FCI recovers standard FCI energy.
2. Photon number convergence with nph_max.
3. Coupling strength scan.
"""

from pyscf import gto, scf, fci

from OmegaQMC.addons.qed_fci import run_qed_fci


def test_h2_qed_fci():
    """Test QED-FCI on H2/STO-6G."""
    mol = gto.M(
        atom='H 0 0 0; H 0 0 1.4',
        basis='sto-6g',
        unit='Bohr',
        verbose=0,
    )
    mf = scf.RHF(mol)
    mf.kernel()

    cisolver = fci.FCI(mf)
    e_fci, _ = cisolver.kernel()

    print("=" * 70)
    print("QED-FCI: H2/STO-6G")
    print("=" * 70)
    print(f"E_HF  = {mf.e_tot:.10f}")
    print(f"E_FCI = {e_fci:.10f}")

    # Test 1: Zero coupling → should recover FCI
    result = run_qed_fci(mf, omega=0.5, coupling_vec=[0, 0, 0], nph_max=4)
    print(f"\nλ=0:    E_QED-FCI = {result['e_qed_fci']:.10f}  "
          f"(should be {e_fci:.10f})")
    assert abs(result['e_qed_fci'] - e_fci) < 1e-10, \
        f"Zero coupling mismatch: {result['e_qed_fci']} vs {e_fci}"

    # Test 2: Photon convergence
    omega = 0.5
    lam = 0.05
    print(f"\nPhoton convergence (ω={omega}, λ={lam}, z-polarized):")
    print(f"  {'nph_max':>8} {'E_QED-FCI':>16} {'n_photon':>12}")
    for nph in [2, 4, 6, 8, 10, 15, 20]:
        result = run_qed_fci(mf, omega=omega, coupling_vec=[0, 0, lam],
                         nph_max=nph)
        print(f"  {nph:8d} {result['e_qed_fci']:16.10f} "
              f"{result['n_photon']:12.8f}")

    # Test 3: Coupling scan
    omega = 0.5
    print(f"\nCoupling scan (ω={omega}, nph_max=10, z-polarized):")
    print(f"  {'lambda':>8} {'E_QED-FCI':>16} {'ΔE':>12} {'n_ph':>12}")
    for lam in [0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]:
        result = run_qed_fci(mf, omega=omega, coupling_vec=[0, 0, lam],
                         nph_max=10)
        dE = result['e_qed_fci'] - e_fci
        print(f"  {lam:8.4f} {result['e_qed_fci']:16.10f} "
              f"{dE:12.8f} {result['n_photon']:12.8f}")


def test_h2o_qed_fci():
    """Test QED-FCI on H2O/STO-3G."""
    mol = gto.M(
        atom='''
        O  0.0000  0.0000  0.1173
        H  0.0000  0.7572 -0.4692
        H  0.0000 -0.7572 -0.4692
        ''',
        basis='sto-3g',
        unit='Angstrom',
        verbose=0,
    )
    mf = scf.RHF(mol)
    mf.kernel()

    cisolver = fci.FCI(mf)
    e_fci, _ = cisolver.kernel()

    print("\n" + "=" * 70)
    print("QED-FCI: H2O/STO-3G")
    print("=" * 70)
    print(f"E_HF  = {mf.e_tot:.10f}")
    print(f"E_FCI = {e_fci:.10f}")
    # print(f"FCI dim = {result['ndim_elec']}" if False else "")

    # Zero coupling check
    result = run_qed_fci(mf, omega=0.5, coupling_vec=[0, 0, 0], nph_max=4)
    print(f"\nλ=0:    E_QED-FCI = {result['e_qed_fci']:.10f}  "
          f"(FCI dim = {result['ndim_elec']})")
    assert abs(result['e_qed_fci'] - e_fci) < 1e-9, \
        f"Zero coupling mismatch: {result['e_qed_fci']} vs {e_fci}"

    # Coupling scan (z-polarized)
    omega = 0.5
    print(f"\nCoupling scan (ω={omega}, nph_max=10, z-polarized):")
    print(f"  {'lambda':>8} {'E_QED-FCI':>16} {'ΔE':>12} {'n_ph':>12}")
    for lam in [0.0, 0.01, 0.05, 0.1, 0.2]:
        result = run_qed_fci(mf, omega=omega, coupling_vec=[0, 0, lam],
                         nph_max=10)
        dE = result['e_qed_fci'] - e_fci
        print(f"  {lam:8.4f} {result['e_qed_fci']:16.10f} "
              f"{dE:12.8f} {result['n_photon']:12.8f}")

    # Polarization dependence (x vs y vs z)
    lam = 0.1
    print(f"\nPolarization dependence (ω={omega}, λ={lam}, nph_max=10):")
    print(f"  {'polar.':>8} {'E_QED-FCI':>16} {'ΔE':>12} {'n_ph':>12}")
    for label, cvec in [('x', [lam, 0, 0]), ('y', [0, lam, 0]),
                        ('z', [0, 0, lam])]:
        result = run_qed_fci(mf, omega=omega, coupling_vec=cvec, nph_max=10)
        dE = result['e_qed_fci'] - e_fci
        print(f"  {label:>8} {result['e_qed_fci']:16.10f} "
              f"{dE:12.8f} {result['n_photon']:12.8f}")


if __name__ == '__main__':
    test_h2_qed_fci()
    test_h2o_qed_fci()
