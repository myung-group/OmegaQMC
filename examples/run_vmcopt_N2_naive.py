import jax
import jax.numpy as jnp
from OmegaQMC import generate_molecular_orbitals
from OmegaQMC.vmcopt_gto_naive import get_vmcopt_gto_func

rng_key = jax.random.key(888)

L = 2.0743
bset_name = "6-31G"

# optimizable Jastrow parameters:
params_jastrow = {
    "J1_pade": {"N": jnp.array([0., 0.])}
}
#     "J1_pade": jnp.array([4.2, 4.2]),
#     "J2_pade": jnp.array([0.6046799, 0.6046799])

atoms_string = '''
N       0.00    0.00    {:.6f}
N       0.00    0.00    {:.6f}
'''.format(-L/2, L/2)

modrv = generate_molecular_orbitals(atoms_string, units="Bohr",
                                    basis=bset_name)

vmcopt_run = get_vmcopt_gto_func(modrv)

l_grad = True
params_jastrow_final, E_data \
    = vmcopt_run(rng_key, params_corr_init=params_jastrow)
print(params_jastrow_final)
print(E_data)
