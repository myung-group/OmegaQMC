import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

import json
import pprint
from pyscf import gto, scf
from importlib import resources

from vmc_mlsw import get_vmc_func




# Set 2(H2O) molecule
mol = gto.M(
        atom=f'''
O       8.707172158e-01  -1.720661566e+00  -4.814067179e-01
H       1.689789528e+00  -1.221338203e+00  -3.052747557e-01
H       1.142438797e+00  -2.587969761e+00  -7.799456175e-01
O      -7.398283056e-01   4.040418183e-01  -1.654300203e+00
H      -2.723133426e-01  -4.319081553e-01  -1.528862134e+00
H      -1.614078540e+00   2.476812916e-01  -1.263515900e+00
    ''',
        basis={
        'H':  '6-31g',
        'O':  '6-31g'
        },
        unit='Ang'
)
mol.build()
mf = scf.RHF(mol)
mf.kernel()
mf_grad = mf.nuc_grad_method()
grad = mf_grad.kernel()

# No optimizable Jastrow parameters:
params_jastrow = {
    "J1_params" : jnp.array([]),
    "J2_params" : jnp.array([])
}

# Load cusp coefficients for the 6-31G basis set
with resources.open_text('vmc_mlsw.basis', 'cusp_coeff_631g.json') as f:
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
chkfile = 'water2_vmc.hdf5'

# Set parameters
reflection_op_list = ['I', 'x', 'y', 'Rz180']
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
    symmop_list=reflection_op_list,
    cluster_idx=None   # or [[0,1,2], [3,4,5]]
)

# Run VMC
vmc_run(
    rng_key=rng_key,
    nwalkers=50,
    num_mc_steps=5000, # MC steps per each walker
    max_mc_iter=500,
    mc_step_size=0.10, # electrons movement distance
    tolerance_enr_std_per_elec=0.02, # VMC termination criteria
    fname_log='vmc_water2_enr.log',
    l_grad=l_grad,
    batch_size=10,     # Batch size for gradient calc. (memory vs. speed trade-off)
    restart=False,     # Restart or initial run
)                      # File {chkfile} is required for restarting.

# Compute gradients of energy
if l_grad:
    grd, grd_err = vmc_grad(
        fname_log='vmc_water2_grd.log',
        compute_error=True,
        walker_based_batch_size=10  # Walker-based Batch size for error calc. 
    )   

