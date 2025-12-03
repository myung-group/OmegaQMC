import os
os.environ["JAX_ENABLE_X64"] = "1"
import jax
import jax.numpy as jnp
from pyscf import gto, scf
from vmc_mlsw import get_vmc_func
#import sys
#import re
#import json
#from importlib import resources
#import math



# No optimizable Jastrow parameters:
params_vmc_no_jastrow = {
      "J1_params": jnp.array([]), # J1 params
      "J2_params": jnp.array([]) # J2 params
}
print("jastrow:", params_vmc_no_jastrow)

# Calculate y, z coordinates of H2O from bond length and angle
rng_key = jax.random.key(888)

# 2 (H2O) molecule
mol = gto.M(
            atom=f'''
O       8.707172158e-01  -1.720661566e+00  -4.814067179e-01
H       1.689789528e+00  -1.221338203e+00  -3.052747557e-01
H       1.142438797e+00  -2.587969761e+00  -7.799456175e-01
O      -7.398283056e-01   4.040418183e-01  -1.654300203e+00
H      -2.723133426e-01  -4.319081553e-01  -1.528862134e+00
H      -1.614078540e+00   2.476812916e-01  -1.263515900e+00
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

chkfile_grd = 'water2_vmc_grd.hdf5'
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


l_cusp = True
reflection_op_list = ['I', 'x', 'y', 'xy']
if l_cusp:
    vmc_run, vmc_grad =\
                get_vmc_func(mf,
                     params_vmc_no_jastrow,
                     scheme='scheme1',
                     chkfile_grd=chkfile_grd,
                     cgto_coeff=cgto_coeff,
                     reflection_op_list=reflection_op_list)
else:
    vmc_run, vmc_grad =\
                get_vmc_func(mf,
                     params_vmc_no_jastrow,
                     scheme='scheme1',
                     chkfile_grd=chkfile_grd,
                     cgto_coeff=None,
                     reflection_op_list=reflection_op_list)
l_grad = True

vmc_run(rng_key,
            nwalkers=50,
            num_mc_steps=5000, # MC steps per each walker
            max_mc_iter=20,
            mc_step_size=0.10, # electrons movement distance
            tolerance_enr_std_per_elec=0.020, #
            fname_log='vmc_water2_enr.log',
            l_grad=l_grad)

if l_grad:
    grd = vmc_grad (fname_log='vmc_water2_grad.log')

