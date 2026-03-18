import jax
import jax.numpy as jnp

import json
import pprint
from pyscf import gto, scf
from importlib import resources

from OmegaQMC import get_vmc_func


# Set (H2O) molecule
mol = gto.M(
            atom=f'''
    O   0.000000   0.000000   0.000000
    H   0.000000   0.632456   0.489897
    H   0.000000  -0.632456   0.489897
    O   0.000000   0.000000   1.500000
    H   0.000000   0.632456   1.989897
    H   0.000000  -0.632456   1.989897
    O   1.000000   0.000000   0.000000
    H   1.000000   0.632456   0.489897
    H   1.000000  -0.632456   0.489897
    O   2.000000   0.000000   1.500000
    H   2.000000   0.632456   1.989897
    H   2.000000  -0.632456   1.989897
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
      "J2_pade": jnp.array([]) # J2 params
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
chkfile = 'water_cluster_vmc.hdf5'

# Set cluster indices
"""
if H2O,
cluster_idx = [0, 1, 2] or None

elif 2(H2O),
cluster_idx = [[0, 1, 2], [3, 4, 5]] or None

elif 4(H2O),
cluster_idx = [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]]
"""
num_mol = 4        # Number of molecules in the system
num_atoms_mol = 3  # Number of atoms in each molecule
cluster_idx = [
    list(range(i*num_atoms_mol, (i*num_atoms_mol + num_atoms_mol)))            
        for i in range(num_mol)
]

# Set parameters
reflection_op_list = ['E', 'x', 'y', 'Rz180']
rng_key = 888
l_grad = True
l_cusp = True
if not l_cusp:
    cgto_coeff = None

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

# Run VMC
vmc_run(
    rng_key=rng_key,
    nwalkers=20,
    num_mc_steps=1000, # MC steps per each walker
    max_mc_iter=500,
    mc_step_size=0.10, # electrons movement distance
    tolerance_enr_std_per_elec=0.01, # VMC termination criteria
    fname_log='vmc_Nwater_enr.log',
    l_grad=l_grad,
    batch_size=50,    # Batch size for gradient calc. (memory vs. speed trade-off)
    restart=False,     # Restart or initial run
)                      # File {chkfile} is required for restarting.

# Compute gradients of energy
if l_grad:
    grd, grd_err = vmc_grad(
        fname_log='vmc_Nwater_grd.log',
        compute_error=True,
        walker_based_batch_size=10  # Walker-based Batch size for error calc. 
    )                                


