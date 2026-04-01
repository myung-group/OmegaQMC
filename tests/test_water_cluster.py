import jax
import jax.numpy as jnp

from OmegaQMC import generate_molecular_orbitals, get_vmc_func
from OmegaQMC.utils import vmc_forces_with_pgcs as vmc_forces


# Set (H2O) molecule
modrv = generate_molecular_orbitals('''
O   0.000000   0.000000   0.000000      1
H   0.000000   0.632456   0.489897      1
H   0.000000  -0.632456   0.489897      1
O   0.000000   0.000000   1.500000      2
H   0.000000   0.632456   1.989897      2
H   0.000000  -0.632456   1.989897      2
O   1.000000   0.000000   0.000000      3
H   1.000000   0.632456   0.489897      3
H   1.000000  -0.632456   0.489897      3
O   2.000000   0.000000   1.500000      4
H   2.000000   0.632456   1.989897      4
H   2.000000  -0.632456   1.989897      4
''',
                                    basis={'H':  '6-31G', 'O':  '6-31G'},
                                    unit='Ang')


# No optimizable Jastrow parameters:
params_jastrow = {"J2_pade": jnp.array([])}

# Set filenames for saving results and checkpoints
chkfile_prefix = 'water_cluster_vmc'

# Set parameters
symmetry_ops = ['E', 'x', 'y', 'C2z']
rng_key = jax.random.key(888)
l_grad = True

# Load VMC functions
vmc_run = get_vmc_func(
    modrv,
    params_corr=params_jastrow,
    prefix=chkfile_prefix,
    symmop_list=symmetry_ops,
)

# Run VMC
vmc_run(
    rng_key,
    num_walkers=100,
    num_steps_per_block=100,    # MC steps per block (per walker)
    num_blocks=10,              # MC blocks
    num_blocks_equil=5,         # MC blocks for equilibration
    mc_timestep=0.002,          # electrons movement distance
    l_grad=l_grad,
    )                      # File {chkfile} is required for restarting.

# Compute gradients of energy
if l_grad:
    forces, std_forces = vmc_forces(prefix=chkfile_prefix)
