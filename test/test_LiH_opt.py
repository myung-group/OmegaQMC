import os
os.environ["JAX_ENABLE_X64"] = "1"
import jax
import jax.numpy as jnp
from pyscf import gto, scf, cc
from vmc_mlsw.vmcopt_gto_fast import get_vmcopt_func

rng_key = jax.random.key(888)

# No optimizable Jastrow parameters:
params_jastrow = {
    "J1_params" : jnp.array([41.0, 7.683]), #4.2, 4.2]),  # H_1, H_2
    "J2_params" : jnp.array([1.582136, 0.912]) #0.6046799, 0.6046799])
}

# LiH molecule
mol = gto.M(atom='''
Li       0.000000    0.00    0.00
H        0.000000    0.00    3.00
''',
            basis='6-31g',
            #basis='aug-cc-pvtz',
            unit='Bohr')

mol.build()
mf = scf.RHF(mol)
mf.kernel()


chkfile_grd = 'LiH_vmc_631gd_grd.hdf5'


cgto_coeff_631g = {
    1: {'q0': 0.973382957446313,
        'coeff': jnp.array([1.0, 1.043187883484018, -0.02914875129226644, 0.7835559367804041, -2.950812002592256, 5.435069523362307, -5.084911845017996, 1.9426518433141349])},
    3: {'q0': 1.0021612523221026,
        'coeff': [1.0, 2.612767320229715, -0.3799239999976849, 3.7530608784072617, -12.779688830699529, 19.645905035281846, -13.863384959084412, 3.694589755364547]},
    8: {'q0': 0.9782267667962238,
        'coeff': jnp.array([1.0, 12.455780474652094, -2.36689972805331, 30.34950214299913, -125.50398084213195, 252.1603002813416, -239.18053266486535, 86.62256481397924])}
}



reflection_op_list = ['I', 'x', 'y', 'xy']
l_cusp = True
vmcopt_run = get_vmcopt_func(mf,
                     params_jastrow,
                     cgto_coeff=cgto_coeff_631g)

vmcopt_run(rng_key,
        nwalkers=1000,
        num_steps=1000, # MC steps per each walker
        num_epochs=100,
        num_equilibration=1000,
        step_size=0.10, # electrons movement distance
        lr=0.02,
        optimizer='sgd')

