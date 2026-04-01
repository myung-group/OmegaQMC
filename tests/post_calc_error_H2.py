import jax
import jax.numpy as jnp

from OmegaQMC import generate_molecular_orbitals, get_vmc_gto_func
from OmegaQMC.utils import vmc_forces_with_pgcs as vmc_forces
from OmegaQMC.utils import compute_energy_with_error, format_basis_name

rng_key = jax.random.key(888)

params_jastrow = {
    "J1_pade": {"H": jnp.array([-0.05574627,  0.08272289])},
    "J2_pade": {"like": jnp.array([0.25, 0.6046799]),
                "unlike": jnp.array([0.5, 0.38077791])}
}

# Set H2 molecule
L = 1.4010
bset_name = "6-31G"
atoms_string = '''
H       0.0000    0.0000    {:.6f}      1
H       0.0000    0.0000    {:.6f}      1
'''.format(-L/2, L/2)

modrv = generate_molecular_orbitals(atoms_string, units="Bohr",
                                    basis=bset_name)

chkfile_prefix = 'H2_vmc_{}'.format(format_basis_name(bset_name))

# Set parameters
symmetry_ops = ['E', 'x', 'y', 'C2z']

l_grad = True
# Load VMC functions
vmc_run = get_vmc_gto_func(
    modrv,
    params_corr=params_jastrow,
    prefix=chkfile_prefix,
    symmop_list=symmetry_ops,
)

# Calculate energy and error using {chkfile}
e_mean, e_err = compute_energy_with_error(chkfile_prefix)
print(f'Total energy | error [Ha]: {e_mean:.6f} | {e_err:.6f}')

# Compute gradients of energy
if l_grad:
    forces, std_forces = vmc_forces(prefix=chkfile_prefix)
