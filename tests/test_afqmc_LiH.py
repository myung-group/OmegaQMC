"""
LiH AFQMC test with better statistics for comparison with ipie.

Exercises the memory‑friendly options:

* ``walker_chunk_size`` — caps peak VHS allocation at
  ``(chunk_size, nbasis, nbasis)`` instead of ``(nwalkers, nbasis, nbasis)``.
* ``chol_h5_path`` — spills the (naux, nbasis, nbasis) MO‑basis Cholesky
  tensor to HDF5 and streams slabs of size ``chol_chunk_g`` along the
  auxiliary axis during propagation. For LiH/sto‑6g the tensor is tiny,
  so this run is just verifying the disk‑backed code path; the same
  flags are what unlock naphthalene‑class systems.
"""

import os
import tempfile

import jax
from pyscf import gto, scf, fci

from OmegaQMC import get_afqmc_func
from OmegaQMC.integrals.cholesky import DiskChol


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

    with tempfile.TemporaryDirectory() as td:
        chol_h5 = os.path.join(td, 'LiH_chol.h5')

        driver = get_afqmc_func(
            mf, dt=0.005, chol_cut=1e-6,
            chol_h5_path=chol_h5,   # spill chol to HDF5
            chol_chunk_g=16,        # stream 16 aux vectors at a time
                                    # Note, choose either 128 or 256 for a large-size molecule
        )
        assert isinstance(driver.chol, DiskChol), (
            f"chol should be disk‑backed, got {type(driver.chol).__name__}"
        )
        assert os.path.exists(chol_h5), "HDF5 file was not written"

        try:
            result = driver(
                rng_key=jax.random.key(42),
                num_walkers=100,
                num_blocks=100,
                num_steps_per_block=25,
                stabilize_freq=5,
                pop_control_freq=5,
                num_blocks_equil=10,
                walker_chunk_size=10,  # cap VHS peak memory
            )
        finally:
            driver.close()  # release the HDF5 handle

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
