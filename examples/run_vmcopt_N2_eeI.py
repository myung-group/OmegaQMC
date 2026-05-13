#!/usr/bin/env python3

import jax
import jax.numpy as jnp
from OmegaQMC import generate_molecular_orbitals
from OmegaQMC.vmcopt_gto_linear import get_vmcopt_gto_func
# from OmegaQMC.observables.force import postproc_h5_pgcs as vmc_forces
# from OmegaQMC.utils import format_basis_name
# from OmegaQMC.vmc_gto_symm import process_symmetric_diatomic_molecule

rng_key = jax.random.key(888)

L = 2.074
bset_name = "6-31G"

# optimizable Jastrow parameters:
# params_jastrow = {'J2_bspline': {'like': jnp.zeros(10),
#                                  'unlike': jnp.zeros(10)},
#                   'J3_eeI': {'uuN': jnp.zeros(26), 'udN': jnp.zeros(26)}}
# params_jastrow = {'J2_bspline': {'like': jnp.array([-7.23483833e-08, -1.23245844e-07,  3.89987177e-07,  2.20083132e-07, -2.92316094e-07, -1.11257828e-07, -1.49153011e-08, -1.16709145e-09, 5.37775064e-09,  3.19039717e-11], dtype=jnp.float64),
#                                  'unlike': jnp.array([-1.67708470e-07, -3.83806878e-07,  7.50054041e-07,  1.74929037e-07, -2.32653804e-07, -1.23222023e-07, -2.07377262e-08, -1.11764205e-09, 3.30474919e-09,  4.55059152e-10], dtype=jnp.float64)},
#                   'J3_eeI': {'unlike+N': jnp.array([ 5.14730809e-06,  1.08275888e-05, -6.34031459e-06, -7.66038085e-06, -2.89695634e-05, -3.38816880e-05,  1.68975306e-05,  2.88322516e-06, -9.22077112e-06,  3.70484311e-05,  3.67997812e-05, -1.37618045e-07, -1.09097405e-05,  1.28312887e-05, -3.78521096e-05, -9.11105722e-05, 1.84217791e-05,  2.24490960e-05,  1.92845581e-05,  3.08428694e-05, 2.88986069e-05,  3.10370124e-05,  5.39486144e-05,  9.72186064e-06, 5.22712428e-06,  9.98907014e-06], dtype=jnp.float64),
#                              'like+N': jnp.array([-1.28089715e-05,  1.30860579e-05,  6.32974607e-06, -7.56752173e-06, -2.63471478e-05, -2.15918326e-05, -6.68700418e-06, -4.65530181e-06, -1.52940857e-05,  8.14414577e-06,  1.87727381e-05, -1.35889880e-06, 5.06918382e-07, -1.20385504e-06, -2.12514012e-05, -6.91453533e-05, 1.58722821e-05,  2.56214612e-05,  2.18963229e-05,  3.04154053e-05, 2.48431512e-05,  3.58959984e-05,  5.36591756e-05,  7.59629825e-06, 8.65906166e-06,  1.40976303e-05], dtype=jnp.float64)}}
params_jastrow = {'J2_bspline': {'like': jnp.array([0.01853326,  0.01535234,  0.05149525, -0.00698700, -0.01500012, 0.01953730,  0.03714619,  0.01987518, -0.00972901, -0.08348470]),
                                 'unlike': jnp.array([-0.01357849, -0.00935871,  0.03553590,  0.02637899,  0.00136680, 0.00882927, -0.00875332, -0.01581401,  0.05128419, -0.01286095])},
                  'J3_eeI': {'like+N': jnp.zeros(26), 'unlike+N': jnp.zeros(26)}}
# params_jastrow = {'J2_pade': {'like': jnp.array([0.25, 1.0]),
#                               'unlike': jnp.array([0.5, 1.0])}}

atoms_string = '''
N       0.00    0.00    {:.6f}
N       0.00    0.00    {:.6f}
'''.format(-L/2, L/2)

modrv = generate_molecular_orbitals(atoms_string, units="Bohr",
                                    basis=bset_name)

vmcopt_run = get_vmcopt_gto_func(modrv)

params_jastrow_final, E_data \
    = vmcopt_run(rng_key, params_corr_init=params_jastrow,
                 num_epochs=20, num_opt_samples=16000,
                 frozen_keys={"J2_bspline": {"like": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                                             "unlike": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]}},
                 verbose=2)
# frozen_keys={"J2_pade": {"like": [0], "unlike": [0]}}

print(E_data)
