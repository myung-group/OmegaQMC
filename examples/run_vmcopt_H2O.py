import jax
import jax.numpy as jnp
from OmegaQMC import generate_molecular_orbitals
from OmegaQMC.vmcopt_gto_irsgd import get_vmcopt_gto_func

rng_key = jax.random.key(888)

bset_name = "6-31G"

# optimizable Jastrow parameters:
params_jastrow = {
    "J1_pade": {"H": jnp.array([-0.14351574,  0.40882649]),
                "O": jnp.array([-0.03921187, 13.46851739])},
    "J2_pade": {"like": jnp.array([0.25, 1.71181446]),
                "unlike": jnp.array([0.5, 2.66981181])}
}
#     "J1_pade": jnp.array([4.2, 4.2]),
#     "J2_pade": {"like": jnp.array([0.25, 0.6047]),
#                   "unlike": jnp.array([0.5, 0.6047])}

myUnits = "ang"
atoms_string = '''
O                0.000000    0.000000    0.117306
H                0.000000    0.757208   -0.469224
H                0.000000   -0.757208   -0.469224
'''

modrv = generate_molecular_orbitals(atoms_string, units=myUnits,
                                    basis=bset_name)

vmcopt_run = get_vmcopt_gto_func(modrv)

l_grad = True
params_jastrow_final, E_data \
    = vmcopt_run(rng_key, params_corr_init=params_jastrow,
                 frozen_keys={"J2_pade": {"like": [0], "unlike": [0]}})
print(params_jastrow_final)
print(E_data)
