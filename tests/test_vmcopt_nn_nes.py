"""Tests for the NES-VMC penalty optimiser scaffolding.

Validates the overlap estimator on synthetic walker banks. The
full training loop is exercised by the example
``run_h2_nesvmc.py`` rather than as a unit test (it takes minutes
of GPU compute).
"""

import os
import tempfile

import h5py
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from OmegaQMC.utils import Mole_custom
from OmegaQMC.vmcopt_nn_nes import (
    _VMCOptDriverNN_NES,
    get_vmcopt_nn_nes_func,
)
from OmegaQMC.vmcopt_nn_iradam import _VMCOptDriverNN_IRAdam
from OmegaQMC.psi.nn.checkpoint import save_nn_checkpoint


@pytest.fixture(scope="module")
def h2_setup(tmp_path_factory):
    """Build a small H2 in STO-3G and produce a fresh checkpoint that
    can serve as the 'ground-state' input to the NES driver."""
    mol = Mole_custom()
    mol.build(atom=[("H", [0, 0, 0]), ("H", [0, 0, 1.4])],
              basis="sto-3g", spin=0, charge=0, unit="Bohr", verbose=0)

    key = jax.random.key(0)
    driver = _VMCOptDriverNN_IRAdam(mol, "psiformer", key)

    # Persist init_params as a "ground state" checkpoint for NES to load
    tmp_dir = tmp_path_factory.mktemp("nes_gs")
    chk_path = str(tmp_dir / "gs.chk.h5")
    save_nn_checkpoint(
        chk_path, driver.init_params,
        epoch=0, config_name="psiformer",
        mol_info=mol, energy=0.0,
    )
    return dict(mol=mol, key=key, chk_path=chk_path,
                init_params=driver.init_params)


@pytest.mark.slow
def test_nes_driver_instantiates(h2_setup):
    """get_vmcopt_nn_nes_func runs without error."""
    nes = get_vmcopt_nn_nes_func(
        h2_setup["mol"], "psiformer", h2_setup["key"],
        ground_state_checkpoint=h2_setup["chk_path"],
        lambda_penalty=1.0,
    )
    assert hasattr(nes, "loss_fn")
    assert hasattr(nes, "overlap_estimator")
    assert hasattr(nes, "ground_state_params")
    assert nes.lambda_penalty == 1.0


@pytest.mark.slow
def test_overlap_estimator_unity_when_same_params(h2_setup):
    """When excited params == ground params, Psi_0/Psi_1 = 1 at every
    walker so the overlap estimator returns ~1 (not zero)."""
    nes = get_vmcopt_nn_nes_func(
        h2_setup["mol"], "psiformer", h2_setup["key"],
        ground_state_checkpoint=h2_setup["chk_path"],
        lambda_penalty=1.0,
    )
    rng = np.random.default_rng(0)
    walkers = jnp.asarray(rng.normal(size=(20, 2, 3)))
    # Pass the ground-state params as the excited params: they're identical
    val = nes.evaluate_overlap(walkers, nes.ground_state_params)
    assert abs(val - 1.0) < 1e-6, (
        f"overlap should be 1 for identical params, got {val}"
    )


@pytest.mark.slow
def test_overlap_estimator_is_finite_for_different_params(h2_setup):
    """For random initialisations of excited vs ground, the overlap is
    finite (no NaN/Inf) and generally non-zero."""
    nes = get_vmcopt_nn_nes_func(
        h2_setup["mol"], "psiformer", h2_setup["key"],
        ground_state_checkpoint=h2_setup["chk_path"],
        lambda_penalty=1.0,
    )
    # Use the freshly-initialised driver init_params as "excited"; they
    # differ from ground_state_params if the rng key differs. To force
    # difference, re-init with a different key.
    diff_driver = _VMCOptDriverNN_IRAdam(
        h2_setup["mol"], "psiformer", jax.random.key(42),
    )
    rng = np.random.default_rng(1)
    walkers = jnp.asarray(rng.normal(size=(20, 2, 3)))
    val = nes.evaluate_overlap(walkers, diff_driver.init_params)
    assert np.isfinite(val)


@pytest.mark.slow
def test_loss_fn_callable(h2_setup):
    """Forward-pass on the loss function returns a finite scalar."""
    nes = get_vmcopt_nn_nes_func(
        h2_setup["mol"], "psiformer", h2_setup["key"],
        ground_state_checkpoint=h2_setup["chk_path"],
        lambda_penalty=1.0,
    )
    rng = np.random.default_rng(2)
    walkers = jnp.asarray(rng.normal(size=(8, 2, 3)))
    loss = float(nes.loss_fn(nes.init_params, walkers))
    assert np.isfinite(loss)


@pytest.mark.slow
def test_loss_fn_gradient_exists(h2_setup):
    """jax.grad of the loss function returns a finite pytree of grads."""
    nes = get_vmcopt_nn_nes_func(
        h2_setup["mol"], "psiformer", h2_setup["key"],
        ground_state_checkpoint=h2_setup["chk_path"],
        lambda_penalty=1.0,
    )
    rng = np.random.default_rng(3)
    walkers = jnp.asarray(rng.normal(size=(8, 2, 3)))
    grads = jax.grad(nes.loss_fn)(nes.init_params, walkers)
    leaves = jax.tree.leaves(grads)
    assert all(np.all(np.isfinite(np.asarray(g))) for g in leaves), (
        "gradient contains non-finite values"
    )
