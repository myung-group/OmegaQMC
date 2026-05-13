#!/usr/bin/env python3

import jax
import jax.numpy as jnp
from OmegaQMC import generate_molecular_orbitals
from OmegaQMC.vmcopt_gto_linear import get_vmcopt_gto_func

rng_key = jax.random.key(888)

atoms_string = """
O       -0.86897        -1.36054         0.06241        1
H        0.09097        -1.21926         0.02587        1
H       -1.05834        -2.02048        -0.60584        1
O        1.61042        -0.07444        -0.05150        2
H        2.21236         0.04539         0.68462        2
H        1.01150         0.68905        -0.02288        2
O       -0.73705         1.43616         0.05642        3
H       -1.17407         1.95157        -0.62279        3
H       -1.11729         0.54544        -0.00337        3
"""

bset_name = "aug-cc-pVTZ"
params_jastrow = {
    "J1_pade": {
        "H": jnp.array([-0.4526821 ,  1.79131828]),
        "O": jnp.array([-0.45701215,  0.32069568])
    },
    "J2_pade": {
        "like": jnp.array([0.25, 1.65713946]),
        "unlike": jnp.array([0.5, 1.31446277]),
    },
}

modrv = generate_molecular_orbitals(
    atoms_string, units="Ang", basis=bset_name
)

vmcopt_run = get_vmcopt_gto_func(modrv)

params_final, E_data = vmcopt_run(
    rng_key,
    params_corr_init=params_jastrow,
    num_epochs=5,
    num_walkers=80,
    num_opt_samples=400,
    frozen_keys={ "J2_pade": {"like": [0], "unlike": [0]} },
    verbose=2,
)
