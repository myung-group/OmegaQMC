import jax.numpy as jnp
import jax
from functools import partial

from pyscf import gto, scf
from vmc_mlsw import get_vmc_func 
#from vmc_mlsw.vmc_gto_symm import process_symmetric_water_molecule

rng_key = jax.random.key(777)
# No optimizable Jastrow parameters:
params_vmc_no_jastrow = jnp.array([])


mol = gto.M(
              atom='''
O                0.   0.   0.
H                0.   1.52610182  1.12172672
H                0.  -1.51745721  1.11537270
''',
              basis='6-31g',
              # basis='cc-pvdz',
              unit='Ang'
          )
mol.build()
mf = scf.RHF(mol)
mf.kernel()
mf_grad = mf.nuc_grad_method()
grad = mf_grad.kernel()

nuc_crds = mol.atom_coords(unit='Bohr')
print('nuc_crds(Bohr)\n', nuc_crds)

chkfile_grd = 'H2O_vmc_631gd_grd.hdf5'

cgto_coeff = {
    1: {
        'q0': 0.99999989,
        'coeff': jnp.array ([1., 1.04318788, -0.02914877,  0.78355609, -2.95081258,  5.43507057,
 -5.08491278,  1.94265217
    ])
    },
    3: {
        'q0': 1.00001336,
        'coeff': jnp.array ([1., 2.61276736,  -0.37992217,   3.7529964,  -12.77929097,
  19.6449144,  -13.86228449,   3.6941363])
    },
    8: {
        'q0': 0.97846901,
        'coeff': jnp.array ([1., 12.45593483,   -2.38348622,   30.46159035, -125.8241975,
  252.619023,   -239.50247682,   86.70949994])
    }
}


l_cusp = True
if l_cusp:
    vmc_run, vmc_grad =\
                get_vmc_func(mf,
                     params_vmc_no_jastrow,
                     scheme='scheme1',
                     chkfile_grd=chkfile_grd,
                     cgto_coeff=cgto_coeff)
else:
    vmc_run, vmc_grad =\
                get_vmc_func(mf,
                     params_vmc_no_jastrow,
                     scheme='scheme1',
                     chkfile_grd=chkfile_grd,
                     cgto_coeff=None)

l_grad = False
vmc_run(rng_key,
        nwalkers=1000, 
        num_mc_steps=1000, # MC steps per each walker
        max_mc_iter=500,
        mc_step_size=0.10, # electrons movement distance
        tolerance_enr_std=0.01, # 
        fname_log='vmc_H2O_enr.log',
        l_grad=l_grad)

if l_grad:
    grd = vmc_grad ()
