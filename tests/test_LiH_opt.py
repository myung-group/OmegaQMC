import jax
import jax.numpy as jnp
from OmegaQMC import generate_molecular_orbitals
from OmegaQMC.vmcopt_gto_linear import get_vmcopt_gto_func

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

# reflection_op_list = ['E', 'x', 'y', 'Rz180']

vmcopt_run = get_vmcopt_gto_func(modrv)
params_jastrow_final, E_data \
    = vmcopt_run(rng_key, params_corr_init=params_jastrow,
                 frozen_keys={"J2_pade": {"like": [0], "unlike": [0]}})
print(E_data)
