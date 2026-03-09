"""
Test AFQMC on H2 molecule.

Compares AFQMC energy against Hartree-Fock and FCI (exact) results.
H2 at equilibrium bond length with STO-6G basis is a minimal test case.
"""

import jax
from pyscf import gto, scf, fci

from OmegaQMC import get_afqmc_func


def test_h2_sto6g():
    """Test AFQMC on H2/STO-6G at equilibrium geometry."""
    # Build molecule
    mol = gto.M(
        atom='H 0 0 0; H 0 0 1.4',  # bond length ~0.74 Angstrom = 1.4 Bohr
        basis='sto-6g',
        unit='Bohr',
        verbose=0,
    )

    # Run HF
    mf = scf.RHF(mol)
    mf.kernel()
    print(f"E_HF  = {mf.e_tot:.10f}")

    # Run FCI for reference
    cisolver = fci.FCI(mf)
    e_fci, _ = cisolver.kernel()
    print(f"E_FCI = {e_fci:.10f}")

    # Run AFQMC (two-step API)
    driver = get_afqmc_func(mf, dt=0.005, chol_cut=1e-6)
    result = driver(
        rng_key=jax.random.key(42),
        num_walkers=100,
        num_blocks=100,
        num_steps_per_block=25,
        stabilize_freq=5,
        pop_control_freq=5,
        num_eqlb_blocks=10,
    )

    e_afqmc = result['energy_mean']
    e_err = result['energy_err']

    print(f"\n=== Results ===")
    print(f"E_HF    = {mf.e_tot:.10f}")
    print(f"E_FCI   = {e_fci:.10f}")
    print(f"E_AFQMC = {e_afqmc:.10f} +/- {e_err:.10f}")
    print(f"E_corr (FCI)   = {e_fci - mf.e_tot:.10f}")
    print(f"E_corr (AFQMC) = {e_afqmc - mf.e_tot:.10f}")


def test_h2_631g():
    """Test AFQMC on H2/6-31G with larger basis."""
    mol = gto.M(
        atom='H 0 0 0; H 0 0 1.4',
        basis='6-31g',
        unit='Bohr',
        verbose=0,
    )

    mf = scf.RHF(mol)
    mf.kernel()
    print(f"\n--- H2 / 6-31G ---")
    print(f"E_HF  = {mf.e_tot:.10f}")

    cisolver = fci.FCI(mf)
    e_fci, _ = cisolver.kernel()
    print(f"E_FCI = {e_fci:.10f}")

    driver = get_afqmc_func(mf, dt=0.005, chol_cut=1e-6)
    result = driver(
        rng_key=jax.random.key(123),
        num_walkers=100,
        num_blocks=100,
        num_steps_per_block=25,
        stabilize_freq=5,
        pop_control_freq=5,
        num_eqlb_blocks=10,
    )

    e_afqmc = result['energy_mean']
    e_err = result['energy_err']

    print(f"\n=== Results ===")
    print(f"E_HF    = {mf.e_tot:.10f}")
    print(f"E_FCI   = {e_fci:.10f}")
    print(f"E_AFQMC = {e_afqmc:.10f} +/- {e_err:.10f}")


if __name__ == '__main__':
    test_h2_sto6g()
    # test_h2_631g()
