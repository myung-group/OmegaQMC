#!/usr/bin/env python3

import jax
import jax.numpy as jnp
from OmegaQMC import generate_molecular_orbitals, get_vmc_gto_func
from OmegaQMC.vmcopt_gto_linear import get_vmcopt_gto_func
# from OmegaQMC.observables.force import postproc_h5_pgcs as vmc_forces
from OmegaQMC.utils import format_basis_name
# from OmegaQMC.vmc_gto_symm import process_symmetric_diatomic_molecule

rng_key = jax.random.key(888)

L = 2.000
bset_name = "6-31G"

# optimizable Jastrow parameters:
# params_jastrow = {'J2_bspline': {'like': jnp.array([0.58797057, 0.65907418, 0.69278570, 0.65525893, 0.68239000, 0.74233749, 0.53475126, 0.22349206]),
#                                  'unlike': jnp.array([0.60584840, 0.95933236, 0.85933138, 0.80862766, 0.91734598, 0.66742384, 0.31912302, 0.16823676])}}
params_jastrow = {'J2_pade': {'like': jnp.array([0.25, 2.15950317]),
                              'unlike': jnp.array([0.5, 2.73679642])}}

atoms_string = '''
N       0.00    0.00    {:.6f}
N       0.00    0.00    {:.6f}
'''.format(-L/2, L/2)

modrv = generate_molecular_orbitals(atoms_string, units="Bohr",
                                    basis=bset_name)

chkfile_prefix = 'N2_vmc_{}'.format(format_basis_name(bset_name))

vmcopt_run = get_vmcopt_gto_func(modrv,
                             jastrow_config=None)
# {"J2": {"r_cut": 10.0}}

params_jastrow_final, E_data \
    = vmcopt_run(rng_key, params_corr_init=params_jastrow,
                 frozen_keys={'J2_pade': {'like': [0], 'unlike': [0]}},
                 num_epochs=10)

# print(E_data)
vmc_run = get_vmc_gto_func(modrv, params_jastrow_final,
                       cusp_scheme="Quady2025",
                       gr_scheme='scheme1',
                       prefix=chkfile_prefix,
                       symmop_list=None,
                       jastrow_config=None)
# {"J2": {"r_cut": 10.0}}

# fname_log="E_loc.dat",
vmc_run(rng_key,
        num_walkers=1000,
        num_steps_per_block=100,   # MC steps per block (per walker)
        num_blocks=1000,             # MC blocks
        num_blocks_equil=100,        # MC blocks for equilibration
        mc_timestep=0.001,          # Brownian time; will be auto-adjusted
        compute_gradients=False)
