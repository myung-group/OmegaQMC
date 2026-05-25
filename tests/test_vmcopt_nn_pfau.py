"""Tests for the Pfau-NES K=2 driver."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from OmegaQMC.utils import Mole_custom
from OmegaQMC.vmcopt_nn_pfau import (
    _VMCOptDriverNN_Pfau_K2,
    get_vmcopt_nn_pfau_k2_func,
)


@pytest.fixture(scope="module")
def h2_pfau_setup():
    mol = Mole_custom()
    mol.build(atom=[("H", [0, 0, 0]), ("H", [0, 0, 1.4])],
              basis="sto-3g", spin=0, charge=0, unit="Bohr", verbose=0)
    key = jax.random.key(0)
    return dict(mol=mol, key=key)


@pytest.mark.slow
def test_pfau_k2_driver_instantiates(h2_pfau_setup):
    """The K=2 driver constructs with two independent parameter sets."""
    driver = get_vmcopt_nn_pfau_k2_func(
        h2_pfau_setup["mol"], "psiformer", h2_pfau_setup["key"],
    )
    assert driver.params_1 is not None
    assert driver.params_2 is not None
    # The two parameter sets must differ (different init keys)
    p1_leaves = jax.tree.leaves(driver.params_1)
    p2_leaves = jax.tree.leaves(driver.params_2)
    assert any(not np.allclose(np.asarray(a), np.asarray(b))
               for a, b in zip(p1_leaves, p2_leaves))


@pytest.mark.slow
def test_joint_psi_zero_at_collapse(h2_pfau_setup):
    """If the two parameter sets coincide, det(M) = 0 exactly."""
    driver = get_vmcopt_nn_pfau_k2_func(
        h2_pfau_setup["mol"], "psiformer", h2_pfau_setup["key"],
    )
    rng = np.random.default_rng(0)
    x_a = jnp.asarray(rng.normal(size=(2, 3)))
    x_b = jnp.asarray(rng.normal(size=(2, 3)))
    # Use same params for both states
    val = driver.joint_psi(x_a, x_b, driver.params_1, driver.params_1)
    assert abs(float(val)) < 1e-8, (
        f"determinantal Psi should vanish when both states are identical; "
        f"got {float(val)}"
    )


@pytest.mark.slow
def test_joint_psi_nonzero_with_distinct_states(h2_pfau_setup):
    """With distinct parameter sets, det(M) is generically nonzero."""
    driver = get_vmcopt_nn_pfau_k2_func(
        h2_pfau_setup["mol"], "psiformer", h2_pfau_setup["key"],
    )
    rng = np.random.default_rng(1)
    x_a = jnp.asarray(rng.normal(size=(2, 3)))
    x_b = jnp.asarray(rng.normal(size=(2, 3)))
    val = float(driver.joint_psi(x_a, x_b, driver.params_1, driver.params_2))
    assert abs(val) > 1e-30
    assert np.isfinite(val)


@pytest.mark.slow
def test_loss_fn_finite_and_differentiable(h2_pfau_setup):
    """The trace loss is finite and has finite gradients in both params."""
    driver = get_vmcopt_nn_pfau_k2_func(
        h2_pfau_setup["mol"], "psiformer", h2_pfau_setup["key"],
    )
    rng = np.random.default_rng(2)
    walkers = jnp.asarray(rng.normal(size=(4, 2, 2, 3)))
    loss = float(driver.loss_fn(driver.params_1, driver.params_2, walkers))
    assert np.isfinite(loss)
    grads = jax.grad(driver.loss_fn, argnums=(0, 1))(
        driver.params_1, driver.params_2, walkers,
    )
    g1_leaves = jax.tree.leaves(grads[0])
    g2_leaves = jax.tree.leaves(grads[1])
    assert all(np.all(np.isfinite(np.asarray(g))) for g in g1_leaves)
    assert all(np.all(np.isfinite(np.asarray(g))) for g in g2_leaves)


@pytest.mark.slow
def test_initialize_joint_walkers_shape(h2_pfau_setup):
    driver = get_vmcopt_nn_pfau_k2_func(
        h2_pfau_setup["mol"], "psiformer", h2_pfau_setup["key"],
    )
    walkers = driver.initialize_joint_walkers(jax.random.key(7), 32)
    assert walkers.shape == (32, 2, driver.nelec, 3)
    assert np.all(np.isfinite(np.asarray(walkers)))


@pytest.mark.slow
def test_sr_loop_runs_one_iter(h2_pfau_setup):
    """The SR-based __call__ completes one iteration on tiny H2 STO-3G
    without raising and produces a valid parameter pytree of the same
    structure as the initial params."""
    driver = get_vmcopt_nn_pfau_k2_func(
        h2_pfau_setup["mol"], "psiformer", h2_pfau_setup["key"],
    )
    init_p1_leaves = jax.tree.leaves(driver.params_1)
    init_p2_leaves = jax.tree.leaves(driver.params_2)
    rng = jax.random.key(42)
    (p1_out, p2_out), info = driver(
        rng,
        num_iters=1,
        num_walkers=8,
        num_steps_per_block=4,
        num_blocks_equil=1,
        num_steps_decorr=1,
        cg_maxiter=4,
        jac_batch_size=4,
        prefix="/tmp/pfau_sr_smoke",
        verbose=0,
    )
    out_p1_leaves = jax.tree.leaves(p1_out)
    out_p2_leaves = jax.tree.leaves(p2_out)
    assert len(out_p1_leaves) == len(init_p1_leaves)
    assert len(out_p2_leaves) == len(init_p2_leaves)
    for a, b in zip(out_p1_leaves, init_p1_leaves):
        assert np.asarray(a).shape == np.asarray(b).shape
    assert "trace_E" in info
    assert np.isfinite(info["trace_E"]["mean"])
