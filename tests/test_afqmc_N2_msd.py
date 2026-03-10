"""Multi-determinant AFQMC test: N2 with CASSCF(14,10)/6-31G trial."""

import jax
from pyscf import gto, scf, mcscf

from OmegaQMC import get_afqmc_func
from OmegaQMC.afqmc_gto import extract_casscf_trial


def test_n2_multidet():
    mol = gto.M(
        atom='N 0 0 -1.03715; N 0 0 1.03715',
        basis='6-31g', unit='Bohr', verbose=0,
    )
    mf = scf.RHF(mol)
    mf.kernel()

    mc = mcscf.CASSCF(mf, ncas=10, nelecas=(7, 7))
    mc.kernel()

    trial = extract_casscf_trial(mc, coeff_threshold=1e-4)
    print(f"Number of determinants: {trial['ndet']}")

    driver = get_afqmc_func(mc._scf, dt=0.005, chol_cut=1e-6,
                            trial=trial)
    result = driver(
        rng_key=jax.random.key(42),
        num_walkers=100,
        num_blocks=100,
        num_steps_per_block=25,
        stabilize_freq=5,
        pop_control_freq=5,
        num_eqlb_blocks=10,
    )

    print(f"E_HF     = {mf.e_tot:.10f}")
    print(f"E_CASSCF = {mc.e_tot:.10f}")
    print(f"E_AFQMC  = {result['energy_mean']:.10f} "
          f"+/- {result['energy_err']:.10f}")

    assert result['energy_mean'] < mf.e_tot


if __name__ == '__main__':
    test_n2_multidet()
