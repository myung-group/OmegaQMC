import os
os.environ["JAX_ENABLE_X64"] = "1"
import jax
import jax.numpy as jnp
from pyscf import gto, scf
from vmc_mlsw import get_vmc_func
from vmc_mlsw.vmc_utils import (
    compute_torque_with_error,
    compute_energy_with_error
)



# No optimizable Jastrow parameters:
params_vmc_no_jastrow = {
      "J1_params": jnp.array([]), # J1 params
      "J2_params": jnp.array([]) # J2 params
}
print("jastrow:", params_vmc_no_jastrow)

# Calculate y, z coordinates of H2O from bond length and angle
# rng_key = jax.random.key(888)
rng_key = 888

"""
    O   0.000000   0.000000   0.000000
    H   0.000000   0.632456   0.489897
    H   0.000000  -0.632456   0.489897
    O   0.000000   0.000000   1.500000
    H   0.000000   0.632456   1.989897
    H   0.000000  -0.632456   1.989897
    O   1.000000   0.000000   0.000000
    H   1.000000   0.632456   0.489897
    H   1.000000  -0.632456   0.489897
    O   2.000000   0.000000   1.500000
    H   2.000000   0.632456   1.989897
    H   2.000000  -0.632456   1.989897
"""
# 2 (H2O) molecule
mol = gto.M(
            atom=f'''
O       8.707172158e-01  -1.720661566e+00  -4.814067179e-01
H       1.689789528e+00  -1.221338203e+00  -3.052747557e-01
H       1.142438797e+00  -2.587969761e+00  -7.799456175e-01
    ''',
            basis={
            'H':  '6-31g',
            'O':  '6-31g'
            },
            unit='Ang'
)

mol.build()
mf = scf.RHF(mol)
mf.kernel()
mf_grad = mf.nuc_grad_method()
grad = mf_grad.kernel()

chkfile = 'water2_vmc.hdf5'
cgto_coeff = {
    1: {
        'q0': 0.99999989,
        'coeff': jnp.array ([1., 1.04318788, -0.02914877,  0.78355609, -2.95081258,  5.43507057,
 -5.08491278,  1.94265217])
    },
    8: {
        'q0': 0.97846901,
        'coeff': jnp.array ([1., 12.45593483,   -2.38348622,   30.46159035, -125.8241975,
  252.619023,   -239.50247682,   86.70949994])
    }
}

import pprint
pprint.pprint(cgto_coeff)

num_mol = 1
num_atoms_mol = 3
cluster_idx = [
    list(range(i*num_atoms_mol, (i*num_atoms_mol + num_atoms_mol)))            
        for i in range(num_mol)
]
# cluster_idx = [[0, 1, 2], [3, 4, 5]]
# or
# cluster_idx = None

l_cusp = True
reflection_op_list = ['I', 'x', 'y', 'xy']
save_all = False
if l_cusp:
    vmc_run, vmc_grad =\
                get_vmc_func(mf,
                     params_vmc_no_jastrow,
                     scheme='scheme1',
                     chkfile=chkfile,
                     cgto_coeff=cgto_coeff,
                     reflection_op_list=reflection_op_list,
                     cluster_idx=cluster_idx)
else:
    vmc_run, vmc_grad =\
                get_vmc_func(mf,
                     params_vmc_no_jastrow,
                     scheme='scheme1',
                     chkfile_grd=chkfile_grd,
                     cgto_coeff=None,
                     reflection_op_list=reflection_op_list,
                     cluster_idx=cluster_idx)

l_grad = True

vmc_run(rng_key,
            nwalkers=20,
            num_mc_steps=500, # MC steps per each walker
            max_mc_iter=20,
            mc_step_size=0.10, # electrons movement distance
            tolerance_enr_std_per_elec=0.005, #
            fname_log='vmc_water2_enr.log',
            l_grad=l_grad,
            batch_size=50,
            restart=False)

# e_mean, e_err = compute_energy_with_error(chkfile)
if l_grad:
    grd, grd_err = vmc_grad (fname_log='vmc_water2_grad.log')
    # torque, dtau = compute_torque_with_error(mol, grd, grd_err)

