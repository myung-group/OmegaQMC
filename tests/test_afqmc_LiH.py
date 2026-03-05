"""
Final LiH AFQMC test with better statistics for comparison with ipie.
"""

import jax
from pyscf import gto, scf, fci

from vmc_pgcs import get_afqmc_func


def test_LiH_sto6g():
    mol = gto.M(
        atom='Li 0 0 0; H 0 0 3.015',
        basis='sto-6g',
        unit='Bohr',
        verbose=0,
    )

    mf = scf.RHF(mol)
    mf.kernel()
    print(f"E_HF  = {mf.e_tot:.10f}")

    cisolver = fci.FCI(mf)
    e_fci, _ = cisolver.kernel()
    print(f"E_FCI = {e_fci:.10f}")

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

    print(f"\n=== Final Results ===")
    print(f"E_HF    = {mf.e_tot:.10f}")
    print(f"E_FCI   = {e_fci:.10f}")
    print(f"E_AFQMC = {e_afqmc:.10f} +/- {e_err:.10f}")
    print(f"E_corr (FCI)   = {e_fci - mf.e_tot:.10f}")
    print(f"E_corr (AFQMC) = {e_afqmc - mf.e_tot:.10f}")
    print(f"ipie ref       = -7.9693 +/- 0.0005")


if __name__ == '__main__':
    test_LiH_sto6g()
