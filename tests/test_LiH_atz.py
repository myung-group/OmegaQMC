import jax
import jax.numpy as jnp
from vmc_mlsw import generate_molecular_orbitals, get_vmc_func
from vmc_mlsw.utils import vmc_forces_with_space_warping as vmc_forces

rng_key = jax.random.key(888)

L = 3.015
bset_name = "aug-cc-pVTZ"

# No optimizable Jastrow parameters:
params_jastrow = {
    "J1_params": {'H': jnp.array([-0.01586607,  0.06306258]),
                  'Li': jnp.array([-0.06306050, -0.05937204])},
    "J2_params": {"like": jnp.array([0.25, 1.56329176]),
                  "unlike": jnp.array([0.5, 0.84623194])}
}
#     "J1_params": {"Li": 48.89, "H": 8.93},
#     "J2_params": {"like": jnp.array([0.25, 1.78]),
#                   "unlike": jnp.array([0.5, 0.917])}

atoms_string = '''
    Li       0.000000    0.00    0.00
    H        0.000000    0.00    {:.6f}
'''.format(L)

modrv = generate_molecular_orbitals(atoms_string, units="Bohr",
                                    symmetrization_level=2,
                                    basis=bset_name)

chkfile_prefix = 'LiH_vmc_aVTZ'

# symmetry_ops = ['E', 'x', 'y', 'Rz180']
# symmetry_ops = ['E', 'Rz90', 'Rz270', 'Rz180']
symmetry_ops = ['E', 'C2z']
# 'Rz180' here means 180-degree rotation
#                 ... or negate both x and y coordinates

vmc_run = get_vmc_func(modrv, params_jastrow,
                       cusp_scheme='Quady2025',
                       gr_scheme='scheme1',
                       symmop_list=symmetry_ops,
                       prefix=chkfile_prefix)

l_grad = True
vmc_run(rng_key,
        num_walkers=100,
        num_steps_per_block=100,   # MC steps per block (per walker)
        num_blocks=10,            # MC blocks
        num_blocks_equil=5,       # MC blocks for equilibration
        mc_timestep=0.1,    # Brownian time; will be auto-adjusted
        compute_gradients=l_grad)

if l_grad:
    forces = vmc_forces(prefix=chkfile_prefix)
