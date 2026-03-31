from OmegaQMC import generate_molecular_orbitals, get_vmc_func
from OmegaQMC.utils import vmc_forces_with_pgcs as vmc_forces


# Set LiH molecule
atoms_string = '''
Li       0.000000    0.00    0.00
H        0.000000    0.00    3.40
    '''

modrv = generate_molecular_orbitals(atoms_string, units="Bohr",
                                    basis={'H':  'aug-cc-pVQZ',
                                           'Li': 'aug-cc-pCVQZ'})


# No Jastrow parameters
params_jastrow = dict()
#     "J1_pade": {"H": jnp.array([-0.05574627,  0.08272289]),
#                 "Li": jnp.array([-0.05574627,  0.08272289])},
#     "J2_pade": {"like": jnp.array([0.25, 0.6046799]),
#                 "unlike": jnp.array([0.5, 0.38077791])}

# Set filenames for saving results and checkpoints
chkfile_prefix = 'LiH_vmc_pcvqz'

# Set parameters
symmetry_ops = ['E', 'x', 'y', 'Rz180']
rng_key = 888

l_grad = True
# Load VMC functions
vmc_run = get_vmc_func(
    modrv,
    params_corr=params_jastrow,
    prefix=chkfile_prefix,
    symmop_list=symmetry_ops,
    cluster_idx=None
)

# Run VMC
vmc_run(
    rng_key=rng_key,
    num_walkers=100,
    num_steps_per_block=100,    # MC steps per block (per walker)
    num_blocks=10,              # MC blocks
    num_blocks_equil=5,         # MC blocks for equilibration
    mc_timestep=0.1,            # Brownian time; will be auto-adjusted
    fname_log='vmc_LiH_enr.log',
    compute_gradients=l_grad
)                      # File {chkfile} is required for restarting.

# Compute gradients of energy
if l_grad:
    forces, std_forces = vmc_forces(prefix=chkfile_prefix)
