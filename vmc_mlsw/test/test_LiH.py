import jax
import jax.numpy as jnp
from pyscf import gto, scf
from vmc_mlsw import (vmc_run, 
                      vmc_energy, 
                      vmc_gradient_prep,
                      vmc_gradient_with_space_warping)
# from vmc_mlsw import JASTROW_EE_L_CUT, JASTROW_EE_M_POWER

rng_key = jax.random.key(7)
params_vmc_no_jastrow = jnp.array([])

mol = gto.M(
          atom='''
H         0.000000    0.000000   1.75
Li       -0.000000    0.000000  -1.75
''',
          basis='sto-3g',
          # basis='cc-pvdz',
          unit='Bohr'
      )
mol.build()
mf = scf.RHF(mol)
mf.kernel()
g = mf.nuc_grad_method()
grad = g.kernel()

nuc_crds = jnp.array(mol.atom_coords(unit='Bohr'))
print('nuc_crds(Bohr)\n', nuc_crds)
nuc_crds_A = jnp.array(mol.atom_coords(unit='Ang'))
print('nuc_crds(Ang)\n', nuc_crds_A)
chkfile = 'LiH_vmc.hdf5'
# (1) Sample electrons
vmc_run(mf,
         rng_key,
         nuc_crds,
         params_vmc_no_jastrow,
         num_steps=500000,
         num_equilibration=50000,
         step_size=0.15,
         chkfile=chkfile)

# (2) Estimate VMC energy
vmc_energy(mf,
            params_vmc_no_jastrow,
            chkfile)


# (3) Calculate the gradients acting on electrons and nuclei 
# based on sampled electrons
vmc_gradient_prep(mf,
                   params_vmc_no_jastrow,
                   chkfile)

# (4) Calculate the total VMC gradients
grd = vmc_gradient_with_space_warping (mf,
                                       chkfile,
                                       scheme='scheme1')
print ('\nScheme1:grd\n', grd, '\n')

grd = vmc_gradient_with_space_warping (mf,
                                       chkfile,
                                       scheme='scheme2')
print ('\nScheme2:grd\n', grd, '\n')
