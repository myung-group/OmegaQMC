import jax
import jax.numpy as jnp

from vmc_mlsw import generate_molecular_orbitals, get_vmc_func
from vmc_mlsw.utils import vmc_forces_with_space_warping as vmc_forces
from vmc_mlsw.utils import format_basis_name


rng_key = jax.random.key(888)
bset_name = '6-31G'

# No optimizable Jastrow parameters:
params_jastrow = {
    "J1_params": jnp.array([]),
    "J2_params": jnp.array([])
}

myUnits = "ang"
atoms_string = '''
O       8.707172158e-01  -1.720661566e+00  -4.814067179e-01     1
H       1.689789528e+00  -1.221338203e+00  -3.052747557e-01     1
H       1.142438797e+00  -2.587969761e+00  -7.799456175e-01     1
O      -7.398283056e-01   4.040418183e-01  -1.654300203e+00     2
H      -2.723133426e-01  -4.319081553e-01  -1.528862134e+00     2
H      -1.614078540e+00   2.476812916e-01  -1.263515900e+00     2
'''
# atoms_string = "geo_water2.xyz"

# myUnits = "ang"
# atoms_string = '''
# O       -0.001278   -1.391642    0.000000     1
# H       -0.483340   -1.721717    0.759608     1
# H       -0.483340   -1.721717   -0.759608     1
# O       -0.001278    1.520358    0.000000     2
# H        0.091108    0.560896    0.000000     2
# H        0.896014    1.852810    0.000000     2
# '''

modrv = generate_molecular_orbitals(atoms_string, units=myUnits,
                                    basis=bset_name)

data_prefix = 'water2_vmc_{}'.format(format_basis_name(bset_name))
# symmetry_ops = ['E', 'x', 'y', 'Rz180']
symmetry_ops = ['E', 'z']
# symmetry_ops = ['E']

# Load VMC functions
vmc_run = get_vmc_func(
    modrv,
    params_jastrow,
    prefix=data_prefix,
    cluster_idx=None   # or [[0,1,2], [3,4,5]]
)
# symmop_list=symmetry_ops,

l_grad = True
vmc_run(
    rng_key,
    num_walkers=100,
    num_steps_per_block=100,  # MC steps per each walker
    num_blocks=10,
    num_blocks_equil=5,
    mc_timestep=0.001,  # electrons movement distance
    compute_gradients=l_grad,
)

# Compute gradients of energy
if l_grad:
    grd, grd_err = vmc_forces(prefix=data_prefix)
