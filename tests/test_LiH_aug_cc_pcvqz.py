import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

import json
import pprint
from pyscf import gto, scf
from importlib import resources

from vmc_mlsw import get_vmc_func
from vmc_mlsw.basis import extract_basis_block


# Load aug-cc-pCVQZ basis set
with resources.open_text('vmc_mlsw.basis', 'aug-cc-pCVQZ.gbs') as f:
    basis_data = f.read()

Li_basis = gto.basis.parse(
        extract_basis_block(basis_data, 'Li')
)
# Set H2 molecule
mol = gto.M(
    atom='''
Li       0.000000    0.00    0.00
H        0.000000    0.00    3.40
    ''',
    basis={
    'H':  'aug-cc-pVQZ',
    'Li': Li_basis
        },
    unit='Bohr'
)
mol.build()
mf = scf.RHF(mol)
mf.kernel()
mf_grad = mf.nuc_grad_method()
grad = mf_grad.kernel()


# No Jastrow parameters
params_jastrow = {
    "J1_params" : jnp.array([]),
    "J2_params" : jnp.array([]) 
}

# Load cusp coefficients for the aug-cc-pCVQZ basis set
with resources.open_text('vmc_mlsw.basis', 'cusp_coeff_aug_cc_pcvqz.json') as f:
    coeff_data = json.load(f)

cgto_coeff = {
    int(Z): {
        'q0': v['q0'],
        'coeff': jnp.array(v['coeff'])
    } for Z, v in coeff_data.items()
}
# if no cusp correction: cgto_coeff = None
pprint.pprint(cgto_coeff)

# Set filenames for saving results and checkpoints
chkfile = 'LiH_vmc_pcvqz.hdf5'

# Set parameters
reflection_op_list = ['I', 'x', 'y', 'xy']
rng_key = 888
l_cusp = True
if not l_cusp:
    cgto_coeff = None
l_grad = True

# Load VMC functions 
vmc_run, vmc_grad = get_vmc_func(
    mf=mf,
    params_vmc=params_jastrow,
    scheme='scheme1',
    chkfile=chkfile,
    cgto_coeff=cgto_coeff,
    reflection_op_list=reflection_op_list,
    cluster_idx=None
)

# Run VMC
vmc_run(
    rng_key=rng_key,
    nwalkers=1000,
    num_mc_steps=1000, # MC steps per each walker
    max_mc_iter=500,
    mc_step_size=0.10, # electrons movement distance
    tolerance_enr_std_per_elec=0.01, # VMC termination criteria
    fname_log='vmc_LiH_enr.log',
    l_grad=l_grad,
    batch_size=500,    # Batch size for gradient calc. (memory vs. speed trade-off)
    restart=False,     # Restart or initial run
)                      # File {chkfile} is required for restarting.

# Compute gradients of energy
if l_grad:
    grd, grd_err = vmc_grad(
        fname_log='vmc_LiH_grd.log',
        compute_error=True,
        walker_based_batch_size=10  # Walker-based Batch size for error calc. 
    )                                

