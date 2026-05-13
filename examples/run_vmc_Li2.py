import jax
import jax.numpy as jnp
from OmegaQMC import generate_molecular_orbitals, get_vmc_gto_func
from OmegaQMC.observables.force import postproc_h5_pgcs as vmc_forces
from OmegaQMC.utils import format_basis_name
# from OmegaQMC.vmc_gto_symm import process_symmetric_diatomic_molecule

rng_key = jax.random.key(888)

L = 5.051
bset_name = "aug-cc-pVQZ"

# optimizable Jastrow parameters:
params_jastrow = {
    "J1_pade": {"Li": jnp.array([-0.05106690, -0.00998437])},
    "J2_pade": {"like": jnp.array([0.25, 0.65678319]),
                  "unlike": jnp.array([0.5, 0.91966118])}
}
#     "J1_pade": {"Li": jnp.array([-7.0, 4.2])},
#     "J2_pade": jnp.array([0.6046799, 0.6046799])

atoms_string = '''
Li      0.00    0.00    {:.6f}      1
Li      0.00    0.00    {:.6f}      1
'''.format(-L/2, L/2)

modrv = generate_molecular_orbitals(atoms_string, units="Bohr",
                                    basis=bset_name)

chkfile_prefix = 'Li2_vmc_{}'.format(format_basis_name(bset_name))

symmetry_ops = {1: ['E']}

vmc_run = get_vmc_gto_func(modrv, params_jastrow,
                       cusp_scheme='Quady2025',
                       gr_scheme='scheme1',
                       prefix=chkfile_prefix,
                       symmop_list=symmetry_ops)

l_grad = True
vmc_run(rng_key,
        num_walkers=100,
        num_steps_per_block=100,   # MC steps per block (per walker)
        num_blocks=10,             # MC blocks
        num_blocks_equil=10,        # MC blocks for equilibration
        mc_timestep=0.001,          # Brownian time; will be auto-adjusted
        compute_gradients=l_grad)

if l_grad:
    forces, std_forces = vmc_forces(prefix=chkfile_prefix)
