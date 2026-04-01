"""Test NN VMC optimizer on H2 with PsiFormer ansatz."""

import jax
import jax.numpy as jnp
from OmegaQMC.psi.nn.wf import MoleculeInfo
from OmegaQMC.vmcopt_nn import get_vmcopt_nn_func

rng_key = jax.random.key(42)

L = 1.4010

mol = MoleculeInfo(
    charges=jnp.array([1.0, 1.0]),
    coords=jnp.array([
        [0.0, 0.0, -L / 2],
        [0.0, 0.0, L / 2],
    ]),
    n_up=1,
    n_down=1,
)

init_key, opt_key = jax.random.split(rng_key)

vmcopt_run = get_vmcopt_nn_func(
    mol, 'psiformer', init_key,
)

params_final, E_data = vmcopt_run(
    opt_key,
    num_epochs=3,
    num_walkers=100,
    num_steps_per_block=50,
    num_blocks=2,
    num_blocks_equil=2,
    batch_size=50,
    lr=1e-3,
    verbose=1,
)

print(f"\nFinal params keys: {list(params_final.keys())}")
print(f"Energy data: {E_data}")
