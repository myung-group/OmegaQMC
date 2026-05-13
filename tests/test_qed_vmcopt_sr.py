"""Smoke tests for QED-NN-VMC SR optimizer.

These verify the optimizer runs end-to-end on H2 and produces sensible
parameter updates. Convergence to QED-FCI accuracy is the responsibility
of the Phase 2e benchmark suite — here we only check:

  * Optimizer constructs without error.
  * SR step modifies parameters (delta_p is nonzero).
  * Energy decreases (mostly) over iterations from random init.
  * Alpha trains (changes from initial value when alpha_train=True).
  * Alpha frozen (does not change when alpha_train=False).
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from OmegaQMC.utils import Mole_custom
from OmegaQMC.qed_vmcopt_nn_sr import get_qed_vmcopt_nn_sr_func


def _build_h2():
    L = 1.4010
    return Mole_custom.from_arrays(
        charges=[1, 1],
        coords=[
            [0.0, 0.0, -L / 2],
            [0.0, 0.0,  L / 2],
        ],
        n_up=1, n_down=1,
    )


def test_optimizer_constructs_with_alpha_train():
    """alpha_train=True should expose 'alpha' in params."""
    mol = _build_h2()
    opt = get_qed_vmcopt_nn_sr_func(
        mol, "psiformer", jax.random.key(99),
        omega=0.5, coupling_vec=jnp.array([0., 0., 0.05]),
        alpha_init=0.1, alpha_train=True, nph_max=4,
    )
    assert "alpha" in opt.init_params
    assert float(opt.init_params["alpha"]) == pytest.approx(0.1)


def test_optimizer_constructs_alpha_frozen():
    """alpha_train=False should NOT expose alpha (frozen via closure)."""
    mol = _build_h2()
    opt = get_qed_vmcopt_nn_sr_func(
        mol, "psiformer", jax.random.key(99),
        omega=0.5, coupling_vec=jnp.array([0., 0., 0.05]),
        alpha_init=0.0, alpha_train=False, nph_max=4,
    )
    assert "alpha" not in opt.init_params


def test_short_sr_run_lambda_zero():
    """At λ=0, short SR run on H2 should: (i) start finite, (ii) decrease
    energy mostly, (iii) leave α at 0 since no coupling drives it."""
    mol = _build_h2()
    opt = get_qed_vmcopt_nn_sr_func(
        mol, "psiformer", jax.random.key(99),
        omega=0.5, coupling_vec=jnp.array([0., 0., 0.0]),
        alpha_init=0.0, alpha_train=True, nph_max=4,
    )
    params, hist = opt(
        rng_key=jax.random.key(123),
        num_iters=8,
        num_walkers=64,
        num_steps_per_block=8,
        num_blocks_equil=2,
        num_steps_decorr=3,
        mc_timestep=0.1,
        lr=0.05,
        damping=1e-3,
        cg_maxiter=50,
        max_param_change=0.5,
        jac_batch_size=32,
        verbose=0,
    )
    assert len(hist["energies"]) == 8
    for e in hist["energies"]:
        assert jnp.isfinite(e), f"non-finite energy in history: {e}"
    # Param change max should be > 0 (we did make updates)
    assert max(hist["param_change_max"]) > 1e-6
    # Energy should generally decrease — but on a 64-walker x 8-iter run
    # with random NN init it's noisy. Check that the last 4 mean isn't
    # higher than the first 4 mean by more than 5x stderr.
    first_half = np.mean(hist["energies"][:4])
    last_half = np.mean(hist["energies"][-4:])
    avg_serr = np.mean(hist["energy_serrs"])
    assert last_half < first_half + 5.0 * avg_serr, (
        f"Energy went up: first half {first_half:.4f}, "
        f"last half {last_half:.4f}, avg serr {avg_serr:.4f}"
    )


def test_alpha_trains_at_finite_lambda():
    """At λ=0.1 with α_init=1.0, α should evolve under SR updates.

    Why α_init must be large enough: the discrete-photon walker
    distribution is Poisson(α²). With α=1.0, |α|² = 1.0, so a healthy
    fraction of walkers populate n≥1 sectors and the α-column of the
    centered Jacobian has nontrivial spread → f_α is nonzero. With
    α_init very small (e.g. 0.05), |α|² ≈ 0.0025 and most walkers land
    at n=0 → centered O_α collapses → f_α = 0 → α stuck. This is a
    practical reminder that the factorized ansatz needs an initial
    coherent-state shift large enough to populate the relevant Fock
    sectors before SR can refine α further.
    """
    mol = _build_h2()
    opt = get_qed_vmcopt_nn_sr_func(
        mol, "psiformer", jax.random.key(99),
        omega=0.5, coupling_vec=jnp.array([0., 0., 0.1]),
        alpha_init=1.0, alpha_train=True, nph_max=8,
    )
    params, hist = opt(
        rng_key=jax.random.key(321),
        num_iters=6,
        num_walkers=128,
        num_steps_per_block=8,
        num_blocks_equil=2,
        num_steps_decorr=3,
        mc_timestep=0.1,
        lr=0.05,
        damping=1e-3,
        cg_maxiter=50,
        max_param_change=0.5,
        jac_batch_size=32,
        alpha_lr_scale=2.0,  # α can move faster than NN params
        verbose=0,
    )
    # α history should have entries (alpha_train=True)
    assert len(hist["alpha_history"]) == 6
    # α must move from its initial value of 1.0
    alpha_changes = np.diff(hist["alpha_history"])
    assert np.max(np.abs(alpha_changes)) > 1e-6, (
        f"α did not move during training. History: {hist['alpha_history']}"
    )


def test_alpha_at_zero_is_stationary_with_vacuum_walkers():
    """Document the α=0 stationary-point behaviour: with α_init=0, walkers
    sampled from |⟨n|0⟩|² = δ_{n,0} cannot probe ∂E/∂α via discrete
    photon variation. SR sees zero force on α; α stays at 0 throughout.
    This is correct behaviour for the factorized ansatz, NOT a bug."""
    mol = _build_h2()
    opt = get_qed_vmcopt_nn_sr_func(
        mol, "psiformer", jax.random.key(99),
        omega=0.5, coupling_vec=jnp.array([0., 0., 0.1]),
        alpha_init=0.0, alpha_train=True, nph_max=4,
    )
    _, hist = opt(
        rng_key=jax.random.key(456),
        num_iters=4,
        num_walkers=32,
        num_steps_per_block=4,
        num_blocks_equil=1,
        num_steps_decorr=2,
        mc_timestep=0.1,
        lr=0.05,
        damping=1e-3,
        cg_maxiter=20,
        jac_batch_size=16,
        verbose=0,
    )
    # All α entries should be ~0 (stationary)
    for a in hist["alpha_history"]:
        np.testing.assert_allclose(a, 0.0, atol=1e-10)


def test_alpha_frozen_in_alpha_train_false():
    """When alpha_train=False, alpha must not appear in any history entry."""
    mol = _build_h2()
    opt = get_qed_vmcopt_nn_sr_func(
        mol, "psiformer", jax.random.key(99),
        omega=0.5, coupling_vec=jnp.array([0., 0., 0.1]),
        alpha_init=0.3, alpha_train=False, nph_max=6,
    )
    _, hist = opt(
        rng_key=jax.random.key(321),
        num_iters=4,
        num_walkers=32,
        num_steps_per_block=4,
        num_blocks_equil=1,
        num_steps_decorr=2,
        mc_timestep=0.1,
        lr=0.05,
        damping=1e-3,
        cg_maxiter=20,
        jac_batch_size=16,
        verbose=0,
    )
    # No alpha key in params -> alpha_history stays empty
    assert hist["alpha_history"] == []


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
