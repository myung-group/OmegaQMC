import jax
import jax.numpy as jnp
from pyscf import gto, scf
from vmc_mlsw.vmcopt_gto_fast import get_vmcopt_func

rng_key = jax.random.key(888)

L = 3.015
bset_name = "6-31G"

# optimizable Jastrow parameters:
params_jastrow = {
    "J1_params": jnp.array([41.0714, 7.7096]),
    "J2_params": jnp.array([1.5846,  0.9614])
}

# LiH molecule
mol = gto.M(atom='''
Li       0.000000    0.00    0.00
H        0.000000    0.00    {:.6f}
'''.format(L),
            basis=bset_name,
            unit='Bohr')

mol.build()
mf = scf.RHF(mol)
mf.kernel()

# reflection_op_list = ['I', 'x', 'y', 'C2']

vmcopt_run = get_vmcopt_func(mf, params_jastrow)
params_jastrow_final, E_data = vmcopt_run(rng_key)
print(params_jastrow_final)
print(E_data)
