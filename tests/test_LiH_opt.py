import jax
import jax.numpy as jnp
from vmc_mlsw import generate_molecular_orbitals
from vmc_mlsw.vmcopt_gto import get_vmcopt_func

rng_key = jax.random.key(888)

L = 3.015
bset_name = "6-31G"

# optimizable Jastrow parameters:
params_jastrow = {
    "J1_params": jnp.array([41.0714, 7.7096]),
    "J2_params": jnp.array([1.5846,  0.9614])
}

atoms_string = '''
    Li       0.000000    0.00    0.00
    H        0.000000    0.00    {:.6f}
'''.format(L)

modrv = generate_molecular_orbitals(atoms_string, units="Bohr",
                                    basis=bset_name)

# reflection_op_list = ['E', 'x', 'y', 'Rz180']

vmcopt_run = get_vmcopt_func(modrv, params_jastrow)
params_jastrow_final, E_data = vmcopt_run(rng_key)
print(params_jastrow_final)
print(E_data)
