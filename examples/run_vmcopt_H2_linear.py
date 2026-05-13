"""Test linear method VMC optimizer on H2."""

import jax
import jax.numpy as jnp
from OmegaQMC import generate_molecular_orbitals
from OmegaQMC.vmcopt_gto_linear import get_vmcopt_gto_func

rng_key = jax.random.key(888)

L = 1.4010
bset_name = "6-31G"

params_jastrow = {
    "J1_pade": {
        "H": jnp.array([-0.05574627, 0.08272289])
    },
    "J2_pade": {
        "like": jnp.array([0.25, 0.6046799]),
        "unlike": jnp.array([0.5, 0.38077791]),
    },
}

atoms_string = """
H       0.0000    0.0000    {:.6f}      1
H       0.0000    0.0000    {:.6f}      1
""".format(-L / 2, L / 2)

modrv = generate_molecular_orbitals(
    atoms_string, units="Bohr", basis=bset_name
)

vmcopt_run = get_vmcopt_gto_func(modrv)

params_final, E_data = vmcopt_run(
    rng_key,
    params_corr_init=params_jastrow,
    frozen_keys={
        "J2_pade": {"like": [0], "unlike": [0]}
    },
    num_walkers=500,
    num_steps_per_block=100,
    deriv_batch_size=50
)

print(f"\nFinal params: {params_final}")
print(f"Energy data: {E_data}")
