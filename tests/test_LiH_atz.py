import jax
import jax.numpy as jnp
from vmc_mlsw import generate_molecular_orbitals, get_vmc_func
from vmc_mlsw.utils import vmc_forces_with_space_warping as vmc_grad

rng_key = jax.random.key(888)

L = 3.015
bset_name = "aug-cc-pVTZ"

# No optimizable Jastrow parameters:
params_jastrow = {
    "J1_params": jnp.array([48.89, 8.93]),
    "J2_params": jnp.array([1.78,  0.917])
}

atoms_string = '''
    Li       0.000000    0.00    0.00
    H        0.000000    0.00    {:.6f}
'''.format(L)

modrv = generate_molecular_orbitals(atoms_string, units="Bohr",
                                    basis=bset_name)

chkfile_prefix = 'LiH_vmc_aVTZ'

# reflection_op_list = ['I', 'x', 'y', 'C2']
reflection_op_list = ['I', 'C2']
# 'C2' here means 180-degree rotation
#                 ... or negate both x and y coordinates

vmc_run \
    = get_vmc_func(modrv, params_jastrow,
                   cusp_scheme='Quady2025',
                   gr_scheme='scheme1',
                   prefix=chkfile_prefix,
                   symmop_list=reflection_op_list)

l_grad = True
vmc_run(rng_key,
        num_walkers=100,
        num_steps_per_block=100,   # MC steps per block (per walker)
        num_blocks=10,            # MC blocks
        num_blocks_equil=5,       # MC blocks for equilibration
        mc_timestep=0.1,    # Brownian time; will be auto-adjusted
        l_grad=l_grad)

if l_grad:
    forces = vmc_grad(prefix=chkfile_prefix)
