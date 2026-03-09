"""
Test QED-AFQMC on H2 molecule in an optical cavity.

Tests:
1. Zero coupling limit: should recover standard AFQMC energy.
2. Finite coupling: cavity lowers the ground state energy.
3. Symmetry check: <q> ≈ 0 for symmetric H2.
"""

import jax
from pyscf import gto, scf, fci

from OmegaQMC import get_afqmc_func, get_qed_afqmc_func


def test_h2_no_cavity():
    """QED-AFQMC with zero coupling should match standard AFQMC."""
    mol = gto.M(
        atom='H 0 0 0; H 0 0 1.4',
        basis='sto-6g',
        unit='Bohr',
        verbose=0,
    )
    mf = scf.RHF(mol)
    mf.kernel()

    # FCI reference
    cisolver = fci.FCI(mf)
    e_fci, _ = cisolver.kernel()

    # Standard AFQMC
    driver_std = get_afqmc_func(mf, dt=0.005, chol_cut=1e-6)
    result_std = driver_std(
        rng_key=jax.random.key(42),
        num_walkers=100,
        num_blocks=100,
        num_steps_per_block=25,
        stabilize_freq=5,
        pop_control_freq=5,
        num_eqlb_blocks=10,
    )

    # QED-AFQMC with zero coupling
    driver_qed = get_qed_afqmc_func(
        mf, omega=0.5,
        coupling_vec=[0.0, 0.0, 0.0],
        dt=0.005, chol_cut=1e-6,
    )
    result_qed = driver_qed(
        rng_key=jax.random.key(42),
        num_walkers=100,
        num_blocks=100,
        num_steps_per_block=25,
        stabilize_freq=5,
        pop_control_freq=5,
        num_eqlb_blocks=10,
    )

    print("\n" + "=" * 60)
    print("Test: H2/STO-6G, zero coupling (should match std AFQMC)")
    print("=" * 60)
    print(f"E_HF         = {mf.e_tot:.10f}")
    print(f"E_FCI        = {e_fci:.10f}")
    print(f"E_AFQMC      = {result_std['energy_mean']:.10f} "
          f"+/- {result_std['energy_err']:.10f}")
    print(f"E_QED(λ=0)   = {result_qed['energy_mean']:.10f} "
          f"+/- {result_qed['energy_err']:.10f}")
    print(f"<q>          = {result_qed['q_mean']:.6f}")
    print(f"E_photon     = {result_qed['e_photon_mean']:.6f}")


def test_h2_in_cavity():
    """QED-AFQMC for H2 in a z-polarized optical cavity."""
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

    # Cavity parameters
    omega = 0.5       # photon frequency (Hartree)
    lam = 0.05        # coupling strength

    driver = get_qed_afqmc_func(
        mf, omega=omega,
        coupling_vec=[0.0, 0.0, lam],  # z-polarized
        dt=0.005, chol_cut=1e-6,
    )
    result = driver(
        rng_key=jax.random.key(42),
        num_walkers=100,
        num_blocks=100,
        num_steps_per_block=25,
        stabilize_freq=5,
        pop_control_freq=5,
        num_eqlb_blocks=10,
    )

    print("\n" + "=" * 60)
    print(f"Test: H2/STO-6G in cavity (ω={omega}, λ={lam})")
    print("=" * 60)
    print(f"E_HF          = {mf.e_tot:.10f}")
    print(f"E_FCI (no cav)= {e_fci:.10f}")
    print(f"E_QED-AFQMC   = {result['energy_mean']:.10f} "
          f"+/- {result['energy_err']:.10f}")
    print(f"<q>           = {result['q_mean']:.6f}")
    print(f"E_photon      = {result['e_photon_mean']:.6f}")


def test_h2_coupling_scan():
    """Scan coupling strength to see the cavity effect on H2."""
    mol = gto.M(
        atom='H 0 0 0; H 0 0 1.4',
        basis='sto-6g',
        unit='Bohr',
        verbose=0,
    )
    mf = scf.RHF(mol)
    mf.kernel()

    omega = 0.5
    lambdas = [0.0, 0.01, 0.05, 0.1]

    print("\n" + "=" * 60)
    print(f"Coupling scan: H2/STO-6G, ω={omega} Ha, z-polarized")
    print("=" * 60)
    print(f"{'lambda':>8} {'E_QED-AFQMC':>16} {'err':>12} "
          f"{'<q>':>10} {'E_ph':>12}")
    print("-" * 60)

    for lam in lambdas:
        driver = get_qed_afqmc_func(
            mf, omega=omega,
            coupling_vec=[0.0, 0.0, lam],
            dt=0.005, chol_cut=1e-6,
            verbose=False,
        )
        result = driver(
            rng_key=jax.random.key(42),
            num_walkers=100,
            num_blocks=80,
            num_steps_per_block=25,
            stabilize_freq=5,
            pop_control_freq=5,
            num_eqlb_blocks=10,
        )
        print(f"{lam:8.4f} {result['energy_mean']:16.10f} "
              f"{result['energy_err']:12.10f} "
              f"{result['q_mean']:10.4f} "
              f"{result['e_photon_mean']:12.6f}")


if __name__ == '__main__':
    test_h2_no_cavity()
    test_h2_in_cavity()
    # test_h2_coupling_scan()
