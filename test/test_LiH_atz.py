import os
os.environ["JAX_ENABLE_X64"] = "1"
import jax
import jax.numpy as jnp
from pyscf import gto, scf, cc
from vmc_mlsw import get_vmc_func


# Set H2 molecule
mol = gto.M(atom='''
Li       0.000000    0.00    0.00
H        0.000000    0.00    3.40
''',
            basis={
                'H': 'aug-cc-pVTZ',
                'Li': 'aug-cc-pVTZ'
                },
            # basis='cc-pvdz',
            unit='Bohr')

mol.build()
mf = scf.RHF(mol)
mf.kernel()
mf_grad = mf.nuc_grad_method()
grad = mf_grad.kernel()

postmf = cc.CCSD(mf).run()
cc_grad = postmf.nuc_grad_method()
cc_grad.kernel()

# Optimized Jastrow parameters
params_jastrow = {
    "J1_params" : jnp.array([48.89, 8.93]), #4.2, 4.2]),  # H_1, H_2
    "J2_params" : jnp.array([1.78,  0.917]) #0.6046799, 0.6046799])
}

# H: aug-cc-pVTZ | Li: aug-cc-pVTZ
cgto_coeff_atz = {
    1: {
        'q0': 0.9999996890322799,
        'coeff': jnp.array([1.0, 1.455012273289097, -0.04709376066167056, 0.7751276822426114, -2.9297756459229167, 5.221146863667661, -4.720082018322578, 1.7514609658945985])
        },
    3: {'q0': 0.9999898424789888,
        'coeff': jnp.array([1.0, 2.620420496220936, -0.054245241883422224, 1.0342557478664784, -4.66108593884602, 10.03952703918778, -10.09862161386026, 3.814926108647762])
        },
}

chkfile = 'LiH_vmc_atz.hdf5'
reflection_op_list = ['I', 'x', 'y', 'xy']
rng_key = 888
l_cusp = True
cgto_coeff = None
if l_cusp:
    cgto_coeff = cgto_coeff_atz

vmc_run, vmc_grad =\
                get_vmc_func(mf,
                     params_jastrow,
                     scheme='scheme1',
                     chkfile=chkfile,
                     cgto_coeff=cgto_coeff,
                     reflection_op_list=reflection_op_list)

l_grad = True
vmc_run(rng_key,
        nwalkers=1000,
        num_mc_steps=1000, # MC steps per each walker
        max_mc_iter=500,
        mc_step_size=0.10, # electrons movement distance
        tolerance_enr_std_per_elec=0.015, #
        fname_log='vmc_LiH_enr_atz_l34.log',
        l_grad=l_grad)

if l_grad:
    grd, grd_err = vmc_grad (fname_log='vmc_LiH_grd_atz_l34.log')
