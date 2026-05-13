import jax
import jax.numpy as jnp
from OmegaQMC import generate_molecular_orbitals
from OmegaQMC.vmcopt_gto_irsgd import get_vmcopt_gto_func

rng_key = jax.random.key(888)

L = 5.051
bset_name = "6-31G"

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

vmcopt_run = get_vmcopt_gto_func(modrv)

l_grad = True
params_jastrow_final, E_data \
    = vmcopt_run(rng_key, params_corr_init=params_jastrow,
                 num_blocks=20,
                 frozen_keys={"J2_pade": {"like": [0], "unlike": [0]}})
print(E_data)
