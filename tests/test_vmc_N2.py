import jax
import jax.numpy as jnp
from OmegaQMC import generate_molecular_orbitals, get_vmc_func
from OmegaQMC.utils import vmc_forces_with_pgcs as vmc_forces
from OmegaQMC.utils import format_basis_name
# from OmegaQMC.vmc_gto_symm import process_symmetric_diatomic_molecule

rng_key = jax.random.key(888)

L = 2.0743
bset_name = "6-31G"

# optimizable Jastrow parameters:
params_jastrow = {
    "J1_pade": {"N": jnp.array([0.0, 0.0])},
}
#     "J1_pade": {"N": jnp.array([-7.0, 4.2])},
#     "J2_pade": jnp.array([0.6046799, 0.6046799])

atoms_string = '''
N       0.00    0.00    {:.6f}
N       0.00    0.00    {:.6f}
'''.format(-L/2, L/2)

modrv = generate_molecular_orbitals(atoms_string, units="Bohr",
                                    basis=bset_name)

chkfile_prefix = 'N2_vmc_{}'.format(format_basis_name(bset_name))

# symmetry_ops = ['E', 'C2z']
symmetry_ops = None

vmc_run = get_vmc_func(modrv, params_jastrow,
                       cusp_scheme='Quady2025',
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
