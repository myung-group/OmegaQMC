import jax.numpy as jnp
import jax
from functools import partial

from pyscf import gto, scf
from vmc_mlsw import get_vmc_func 
from vmc_mlsw.vmc_gto_symm import process_symmetric_water_molecule

    
mol = gto.M(
              atom='''
O                0.   0.   0.
H                0.   1.52610182  1.12172672
H                0.  -1.51745721  1.11537270
''',
              basis='6-31g*',
              # basis='cc-pvdz',
              unit='Bohr'
          )
mol.build()
mf = scf.RHF(mol)
mf.kernel()
mf_grad = mf.nuc_grad_method()
grad = mf_grad.kernel()

nuc_crds = mol.atom_coords(unit='Bohr')
rng_key = jax.random.key(777)
# No optimizable Jastrow parameters:
params_vmc_no_jastrow = jnp.array([])

chkfile_mc =  'H2O_vmc0_631gd_mc.hdf5'
chkfile_elc = 'H2O_vmc0_631gd_elc.hdf5'
chkfile_enr = 'H2O_vmc0_631gd_enr.hdf5'
chkfile_grd = 'H2O_vmc0_631gd_grd.hdf5'

cgto_coeff = {
        1: jnp.array ([1, 1.0431879, -0.02914878, 0.78355617,
        -2.95081286, 5.43507108, -5.08491324, 1.94265234]),
        8: jnp.array ([1, 12.45593615, -2.38348643, 30.46159315,
        -125.8242091, 252.61904634, -239.5024989, 86.70950789])
    }
    
vmc_run, vmc_energy, vmc_gradient_prep, vmc_grad =\
        get_vmc_func (mf, 
                      params_vmc_no_jastrow,
                      chkfile_mc=chkfile_mc,
                      chkfile_enr=chkfile_enr,
                      chkfile_grd=chkfile_grd,
                      chkfile_elc=chkfile_elc,
                      cgto_coeff=None)

vmc_run (rng_key, 
             num_steps=1000000,
             num_equilibration=50000,
             step_size=0.05)
    
vmc_energy () 


process_symmetric_water_molecule(
        chkfile_mc,
        chkfile_elc,
        sigma=0.5,
        reflection_ops=['x','y', 'xy']
    )

vmc_gradient_prep ()

grd = vmc_grad (scheme='scheme1',
          mark_std=3.0)

with jnp.printoptions (precision=5, suppress=True):
    print('grd_tot\n', grd)
