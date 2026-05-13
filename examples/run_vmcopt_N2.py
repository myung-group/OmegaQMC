import jax
import jax.numpy as jnp
from OmegaQMC import generate_molecular_orbitals
from OmegaQMC.vmcopt_gto_irsgd import get_vmcopt_gto_func
# from OmegaQMC.vmc_gto_symm import process_symmetric_diatomic_molecule

rng_key = jax.random.key(888)

L = 2.0743
bset_name = "6-31G"

# optimizable Jastrow parameters:
params_jastrow = {
    "J1_pade": {"N": jnp.array([-7.0, 4.2])},
    "J2_pade": jnp.array([0.6046799, 0.6046799])
}

atoms_string = '''
N       0.00    0.00    {:.6f}
N       0.00    0.00    {:.6f}
'''.format(-L/2, L/2)

modrv = generate_molecular_orbitals(atoms_string, units="Bohr",
                                    basis=bset_name)

vmcopt_run = get_vmcopt_gto_func(modrv)

params_jastrow_final, E_data \
    = vmcopt_run(rng_key, params_corr_init=params_jastrow,
                 frozen_keys={"J2_pade": {"like": [0], "unlike": [0]}})

print(E_data)
