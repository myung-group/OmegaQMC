import jax
import jax.numpy as jnp
from pyscf import gto, scf
from vmc_mlsw import vqmc_run, vqmc_energy, vqmc_gradient
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
LiH_stacked_samples = vqmc_run(mf,
                               rng_key,
                               nuc_crds,
                               params_vmc_no_jastrow,
                               num_steps=500000,
                               num_equilibration=20000,
                               step_size=0.1
                               )


LiH_enr_samples, LiH_enr_nn = vqmc_energy(mf,
                                          nuc_crds,
                                          params_vmc_no_jastrow,
                                          LiH_stacked_samples)
LiH_enr_mean = LiH_enr_samples.mean() + LiH_enr_nn
LiH_enr_std_err = LiH_enr_samples.std()/jnp.sqrt(LiH_enr_samples.shape[0])
print('LiH_enr_mean', LiH_enr_mean, LiH_enr_std_err)


LiH_grad_total = vqmc_gradient(mf,
                               nuc_crds,
                               params_vmc_no_jastrow,
                               LiH_stacked_samples,
                               LiH_enr_samples,
                               l_scheme1=False)
print('grad_total\n', LiH_grad_total)
