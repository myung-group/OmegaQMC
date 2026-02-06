import jax
import jax.numpy as jnp
from vmc_mlsw import generate_molecular_orbitals, get_vmc_func
from vmc_mlsw.utils import vmc_forces_with_space_warping as vmc_forces
from vmc_mlsw.utils import format_basis_name
# from vmc_mlsw.vmc_gto_symm import process_symmetric_water_molecule

rng_key = jax.random.key(777)
bset_name = "cc-pVDZ"
# other choices: 6-31G, cc-pVDZ, etc.

# No optimizable Jastrow parameters:
params_vmc_no_jastrow = {
    "J1_params": jnp.array([]),
    "J2_params": jnp.array([])
}

# myUnits = "ang"
# atoms_string = '''
# O                0.000000    0.000000    0.000000
# H                0.000000    0.000000    0.957800
# H                0.927385    0.000000   -0.239451
# '''

myUnits = "bohr"
atoms_string = '''
O                0.   0.   0.
H                0.   1.52610182  1.12172672
H                0.  -1.51745721  1.11537270
'''

# myUnits = "ang"
# atoms_string = '''
# O                0.000000    0.000000    0.117306
# H                0.000000    0.757208   -0.469224
# H                0.000000   -0.757208   -0.469224
# '''

modrv = generate_molecular_orbitals(atoms_string, units=myUnits,
                                    basis=bset_name,
                                    symmetrization_level=2,
                                    ignore_hydrogen_mass=False)

chkfile_prefix = 'H2O_vmc_{}'.format(format_basis_name(bset_name))

vmc_run = get_vmc_func(modrv, params_vmc_no_jastrow,
                       cusp_scheme='Quady2025',
                       gr_scheme='scheme1',
                       prefix=chkfile_prefix)

l_grad = True
vmc_run(rng_key,
        num_walkers=100,
        num_steps_per_block=100,  # MC steps per each walker
        num_blocks=10,
        num_blocks_equil=5,       # MC blocks for equilibration
        mc_timestep=0.002,    # Brownian time; will be auto-adjusted
        compute_gradients=l_grad)

if l_grad:
    grd = vmc_forces(prefix=chkfile_prefix)
