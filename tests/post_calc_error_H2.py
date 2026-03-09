import jax.numpy as jnp

import json
import pprint
from pyscf import gto, scf
from importlib import resources

from OmegaQMC.import get_vmc_func
from OmegaQMC.vmc_utils import (
    compute_energy_with_error
)


# Set H2 molecule
mol = gto.M(atom='''
H       0.000000    0.00    0.00
H       0.000000    0.00    1.40
''',
            basis='6-31g',
            unit='Bohr')

mol.build()
mf = scf.RHF(mol)
mf.kernel()
mf_grad = mf.nuc_grad_method()
grad = mf_grad.kernel()

# Optimized Jastrow parameters
params_jastrow = {
    "J1_params": {"H": jnp.array([-0.15486924,  0.08576524])},
    "J2_params": {"like": jnp.array([0.25, 0.6046799]),
                  "unlike": jnp.array([0.5, 0.46045336])}
}

# Load cusp coefficients for the 6-31G basis set
with resources.open_text('OmegaQMC.basis', 'cusp_coeff_631g.json') as f:
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
chkfile = 'H2_vmc_631gd.hdf5'

# Set parameters
reflection_op_list = ['E', 'x', 'y', 'Rz180']
rng_key = 888
l_cusp = True
l_grad = True

# Load VMC functions
vmc_run, vmc_grad = get_vmc_func(
    mf=mf,
    params_vmc=params_jastrow,
    scheme='scheme1',
    chkfile=chkfile,
    cgto_coeff=cgto_coeff,
    symmop_list=reflection_op_list,
    cluster_idx=None
)

# Calculate energy and error using {chkfile}
e_mean, e_err = compute_energy_with_error(chkfile)
print(f'Total energy | error [Ha]: {e_mean:.6f} | {e_err:.6f}')

# Compute gradients of energy
if l_grad:
    grd, grd_err = vmc_grad(
        fname_log='vmc_H2_grd.log',
        compute_error=True,
        walker_based_batch_size=10  # Walker-based Batch size for error calc.
    )
    # Compute torque and error
    # torque, dtau = compute_torque_with_error(mol, grd, grd_err)
