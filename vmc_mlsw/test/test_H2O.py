import jax
import jax.numpy as jnp
from pyscf import gto, scf
from vmc_mlsw import vqmc_run, vqmc_energy, vqmc_gradient
from vmc_mlsw import JASTROW_EE_L_CUT, JASTROW_EE_M_POWER


rng_key = jax.random.key(777)

# Jastrow hyperparameters (passed to functions but not used if Jastrow is off)
L_cut = JASTROW_EE_L_CUT
M_power = JASTROW_EE_M_POWER
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
print('H2O_nuc_crds', H2O_nuc_crds)

H2O_stacked_samples = vqmc_run(H2O_mf,
                               rng_key,
                               H2O_nuc_crds,
                               params_vmc_no_jastrow,
                               num_steps=1000000,
                               num_equilibration=20000,
                               step_size=0.05
                               )
# num_steps: number of steps for each point on the curve

H2O_enr_samples, H2O_enr_nn = vqmc_energy(H2O_mf,
                                          H2O_nuc_crds,
                                          params_vmc_no_jastrow,
                                          H2O_stacked_samples)

enr_mean = H2O_enr_samples.mean() + H2O_enr_nn
enr_std_err = H2O_enr_samples.std()/jnp.sqrt(H2O_enr_samples.shape[0])
print('enr_mean', enr_mean, enr_std_err)

grad_total = vqmc_gradient(H2O_mf,
                           H2O_nuc_crds,
                           params_vmc_no_jastrow,
                           H2O_stacked_samples,
                           H2O_enr_samples)
print('grad_total', grad_total)
