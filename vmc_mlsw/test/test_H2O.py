import jax
import jax.numpy as jnp
from pyscf import gto, scf
from vmc_mlsw import (vmc_run, 
                      vmc_energy, 
                      vmc_gradient_prep,
                      vmc_gradient_with_space_warping)


rng_key = jax.random.key(777)

# No optimizable Jastrow parameters:
params_vmc_no_jastrow = jnp.array([])

H2O_mol = gto.M(
              atom='''
O        0.000000    0.000000    0.117307
H       -0.000000    0.757216   -0.469229
H       -0.000000   -0.757216   -0.469229
''',
              basis='sto-3g',
              # basis='cc-pvdz',
              unit='Ang'
          )
H2O_mol.build()
H2O_mf = scf.RHF(H2O_mol)
H2O_mf.kernel()
H2O_grad = H2O_mf.nuc_grad_method()
H2O_grad = H2O_grad.kernel()

H2O_nuc_crds = jnp.array(H2O_mol.atom_coords(unit='Bohr'))
print('H2O_nuc_crds(Bohr)\n', H2O_nuc_crds)

chkfile = 'H2O_vmc.hdf5'
# (1) Sample electrons
vmc_run(H2O_mf,
        rng_key,
        H2O_nuc_crds,
        params_vmc_no_jastrow,
        num_steps=1000000,
        num_equilibration=50000,
        step_size=0.05,
        chkfile=chkfile
        )

# (2) Estimate VMC energy
vmc_energy(H2O_mf,
           params_vmc_no_jastrow,
           chkfile)

# (3) Calculate the gradients acting on electrons and nuclei 
# based on sampled electrons
vmc_gradient_prep(H2O_mf,
                  params_vmc_no_jastrow,
                  chkfile)


# (4) Calculate the total VMC gradients
grd = vmc_gradient_with_space_warping (H2O_mf,
                                       chkfile,
                                       scheme='scheme1')
print ('\nScheme1:grd\n', grd, '\n')

grd = vmc_gradient_with_space_warping (H2O_mf,
                                       chkfile,
                                       scheme='scheme2')
print ('\nScheme2:grd\n', grd, '\n')
