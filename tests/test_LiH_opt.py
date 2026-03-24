import jax
import jax.numpy as jnp
from OmegaQMC import generate_molecular_orbitals
from OmegaQMC.vmcopt_gto_linear import get_vmcopt_func

rng_key = jax.random.key(888)

L = 3.015
bset_name = "6-31G"

# optimizable Jastrow parameters:
params_jastrow = {
    "J1_pade": {'H': jnp.array([-0.01586607,  0.06306258]),
                'Li': jnp.array([-0.06306050, -0.05937204])},
    "J2_pade": {"like": jnp.array([0.25, 1.56329176]),
                "unlike": jnp.array([0.5, 0.84623194])}
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

vmcopt_run = get_vmcopt_func(modrv)
params_jastrow_final, E_data \
    = vmcopt_run(rng_key, params_corr_init=params_jastrow,
                 frozen_keys={"J2_pade": {"like": [0], "unlike": [0]}})
print(params_jastrow_final)
print(E_data)
