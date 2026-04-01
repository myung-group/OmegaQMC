"""Multi-determinant VMC test: N2 with CASSCF(14,10)/6-31G."""

import jax
import jax.numpy as jnp
from pyscf import mcscf

from OmegaQMC import generate_molecular_orbitals, get_vmc_gto_func
from OmegaQMC.afqmc_gto import extract_casscf_trial
from OmegaQMC.utils import format_basis_name

rng_key = jax.random.key(888)

L = 2.0743
bset_name = "6-31G"

params_jastrow = {
    "J1_pade": {"N": jnp.array([0.0, 0.0])},
}

atoms_string = '''
N       0.00    0.00    {:.6f}
N       0.00    0.00    {:.6f}
'''.format(-L/2, L/2)

modrv = generate_molecular_orbitals(
    atoms_string, units="Bohr", basis=bset_name
)

# Run CASSCF on top of the RHF reference
mc = mcscf.CASSCF(modrv, ncas=10, nelecas=(7, 7))
mc.kernel()

trial = extract_casscf_trial(mc, coeff_threshold=1e-2)
print(f"Number of determinants: {trial['ndet']}")

chkfile_prefix = 'N2_vmc_msd_{}'.format(
    format_basis_name(bset_name)
)

vmc_run = get_vmc_gto_func(
    modrv, params_jastrow,
    cusp_scheme='Quady2025',
    gr_scheme='scheme1',
    prefix=chkfile_prefix,
    trial=trial,
)

vmc_run(
    rng_key,
    num_walkers=100,
    num_steps_per_block=100,
    num_blocks=20,
    num_blocks_equil=10,
    mc_timestep=0.001,
    compute_gradients=False,
)
