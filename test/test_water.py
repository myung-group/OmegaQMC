import os
os.environ["JAX_ENABLE_X64"] = "1"
import jax.numpy as jnp
import jax
from pyscf import gto, scf
from vmc_mlsw import get_vmc_func

rng_key = jax.random.key(777)
# No optimizable Jastrow parameters:
params_vmc_no_jastrow = {
    "J1_params" : jnp.array([]),
    "J2_params" : jnp.array([])
}

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

cgto_coeff_631g = {
    1: {'q0': 0.973382957446313,
            'coeff': jnp.array([1.0, 1.043187883484018, -0.02914875129226644, 0.7835559367804041, -2.950812002592256, 5.435069523362307, -5.084911845017996, 1.9426518433141349])},
    8: {'q0': 0.9782267667962238,
            'coeff': jnp.array([1.0, 12.455780474652094, -2.36689972805331, 30.34950214299913, -125.50398084213195, 252.1603002813416, -239.18053266486535, 86.62256481397924])}
}

import pprint
pprint.pprint(cgto_coeff_631g)


l_cusp = True
cgto_coeff = None
if l_cusp:
    cgto_coeff = cgto_coeff_631g

vmc_run, vmc_grad = get_vmc_func(mf,
                     params_vmc_no_jastrow,
                     scheme='scheme1',
                     chkfile_grd=chkfile_grd,
                     cgto_coeff=cgto_coeff,
                     reflection_op_list=['I', 'x', 'y', 'xy'])

l_grad = True
vmc_run(rng_key,
        nwalkers=1000,
        num_mc_steps=1000, # MC steps per each walker
        max_mc_iter=500,
        mc_step_size=0.10, # electrons movement distance
        tolerance_enr_std_per_elec=0.01, #
        fname_log='vmc_H2O_enr.log',
        l_grad=l_grad)

if l_grad:
    grd = vmc_grad (fname_log='vmc_H2O_grd.log')
