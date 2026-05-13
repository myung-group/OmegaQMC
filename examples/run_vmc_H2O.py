import jax
import jax.numpy as jnp
from OmegaQMC import generate_molecular_orbitals, get_vmc_gto_func
from OmegaQMC.observables.force import postproc_h5_pgcs as vmc_forces
from OmegaQMC.utils import format_basis_name
# from OmegaQMC.vmc_gto_symm import process_symmetric_diatomic_molecule

rng_key = jax.random.key(888)

bset_name = "6-31G"

# optimizable Jastrow parameters:
params_jastrow = {'J1_pade': {'H': jnp.array([-0.10898312,  0.21565190]),
                              'O': jnp.array([-0.34446705,  0.03732419])},
                  'J2_pade': {'like': jnp.array([0.25, 1.31908493]),
                              'unlike': jnp.array([0.5, 1.03946413])}}

atoms_string = '''
O        0.000000    0.000000    0.117307
H        0.000000    0.757216   -0.469229
H        0.000000   -0.757216   -0.469229
'''

modrv = generate_molecular_orbitals(atoms_string, units="Ang",
                                    basis=bset_name)

h5file_prefix = 'H2O_vmc_{}'.format(format_basis_name(bset_name))

vmc_run = get_vmc_gto_func(modrv, params_jastrow,
                           cusp_scheme='Quady2025',
                           gr_scheme='scheme1',
                           prefix=h5file_prefix,
                           symmop_list=['E', 'Rz180'])

l_grad = True
vmc_run(rng_key,
        num_walkers=10,
        num_steps_per_block=10,   # MC steps per block (per walker)
        num_blocks=20,             # MC blocks
        num_blocks_equil=10,        # MC blocks for equilibration
        mc_timestep=0.001,          # Brownian time; will be auto-adjusted
        fname_log="E_loc.dat",
        compute_gradients=l_grad)

if l_grad:
    forces, std_forces = vmc_forces(prefix=h5file_prefix)
