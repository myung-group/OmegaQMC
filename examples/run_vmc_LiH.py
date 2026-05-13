import jax
import jax.numpy as jnp
from OmegaQMC import generate_molecular_orbitals, get_vmc_gto_func
from OmegaQMC.observables.force import postproc_h5_pgcs as vmc_forces
from OmegaQMC.utils import format_basis_name

rng_key = jax.random.key(888)

L = 3.015
bset_name = "6-31G"

# optimizable Jastrow parameters:
params_jastrow = {
    "J1_pade": {'H': jnp.array([-0.05700013,  0.20863551]),
                'Li': jnp.array([-0.18315847,  0.18684607])},
    "J2_pade": {"like": jnp.array([0.25, 1.56486474]),
                "unlike": jnp.array([0.5, 0.83352021])}
}
#     "J1_pade": jnp.array([41.0714, 7.7096]),
#     "J2_pade": {"like": jnp.array([0.25, 1.78]),
#                   "unlike": jnp.array([0.5, 0.917])}

atoms_string = '''
    Li       0.000000    0.00    0.00
    H        0.000000    0.00    {:.6f}
'''.format(L)

modrv = generate_molecular_orbitals(atoms_string, units="Bohr",
                                    basis=bset_name)

chkfile_prefix = 'LiH_vmc_{}'.format(format_basis_name(bset_name))

# symmetry_ops = ['E', 'C2z']
symmetry_ops = None

vmc_run = get_vmc_gto_func(modrv, params_jastrow,
                       gr_scheme='scheme1',
                       prefix=chkfile_prefix,
                       symmop_list=symmetry_ops)

l_grad = True
vmc_run(rng_key,
        num_walkers=100,
        num_steps_per_block=100,   # MC steps per block (per walker)
        num_blocks=20,             # MC blocks
        num_blocks_equil=10,        # MC blocks for equilibration
        mc_timestep=0.001,          # Brownian time; will be auto-adjusted
        compute_gradients=l_grad)

if l_grad:
    forces, std_forces = vmc_forces(prefix=chkfile_prefix)
