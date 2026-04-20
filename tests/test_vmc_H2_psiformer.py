"""Integration test: NN VMC driver on H2.

Runs a short VMC simulation with the PsiFormer ansatz
on H2 and checks that the energy is finite and in a
reasonable range.
"""

import pytest
import jax
import jax.numpy as jnp

from OmegaQMC.utils import Mole_custom
from OmegaQMC.vmc_nn import get_vmc_nn_func


@pytest.fixture
def h2_driver():
    """Build a PsiFormer VMC driver for H2."""
    L = 1.4010
    mol = Mole_custom.from_arrays(
        charges=[1, 1],
        coords=[
            [0.0, 0.0, -L / 2],
            [0.0, 0.0, L / 2],
        ],
        n_up=1,
        n_down=1,
    )
    init_key = jax.random.key(99)
    return get_vmc_nn_func(mol, 'psiformer', init_key)


def test_vmc_nn_h2_energy(h2_driver):
    """Energy is finite and in [-2, 0] Ha for H2."""
    run_key = jax.random.key(77)
    result = h2_driver(
        run_key,
        num_walkers=50,
        num_steps_per_block=20,
        num_blocks=5,
        num_blocks_equil=2,
        mc_timestep=0.1,
        verbose=0,
    )
    E = result['E_mean']
    assert jnp.isfinite(E), (
        f"Energy not finite: {E}"
    )
    assert -2.0 < E < 0.0, (
        f"Energy {E} outside [-2, 0] Ha for H2"
    )


def test_vmc_nn_h2_serr(h2_driver):
    """Statistical error is finite and positive."""
    run_key = jax.random.key(77)
    result = h2_driver(
        run_key,
        num_walkers=50,
        num_steps_per_block=20,
        num_blocks=5,
        num_blocks_equil=2,
        mc_timestep=0.1,
        verbose=0,
    )
    serr = result['E_serr']
    assert jnp.isfinite(serr), (
        f"Standard error not finite: {serr}"
    )
    assert serr > 0, (
        f"Standard error should be positive: {serr}"
    )


def test_vmc_nn_h2_blocks(h2_driver):
    """Correct number of block energies returned."""
    run_key = jax.random.key(77)
    n_blocks = 5
    result = h2_driver(
        run_key,
        num_walkers=50,
        num_steps_per_block=20,
        num_blocks=n_blocks,
        num_blocks_equil=2,
        mc_timestep=0.1,
        verbose=0,
    )
    assert len(result['E_blocks']) == n_blocks


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
