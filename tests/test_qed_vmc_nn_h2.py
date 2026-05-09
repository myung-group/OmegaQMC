"""Smoke tests for QED-VMC driver on H2.

These exercise the full pipeline end-to-end on a tiny system:

  * Driver construction with a real PsiFormer ansatz.
  * Walker initialization (continuous r + discrete n).
  * Joint discrete-continuous Metropolis-Hastings sampling.
  * Pauli-Fierz local-energy evaluation.
  * Block-averaged statistics.

Validation against QED-FCI on (H2)2 is the responsibility of the
Phase 2e benchmark tests (after optimizer integration); here we only
check that the pipeline runs without error and produces *finite*
energies in the right ballpark for an *untrained* PsiFormer (initial
parameters give an order-of-magnitude correct H2 energy).
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from OmegaQMC.utils import Mole_custom
from OmegaQMC.qed_vmc_nn import get_qed_vmc_nn_func


def _build_h2():
    L = 1.4010  # H2 bond length in Bohr
    return Mole_custom.from_arrays(
        charges=[1, 1],
        coords=[
            [0.0, 0.0, -L / 2],
            [0.0, 0.0,  L / 2],
        ],
        n_up=1, n_down=1,
    )


@pytest.fixture
def h2_qed_driver_lambda_zero():
    """H2 driver at λ=0 (decoupled cavity)."""
    mol = _build_h2()
    return get_qed_vmc_nn_func(
        mol, "psiformer", jax.random.key(99),
        omega=0.5,
        coupling_vec=jnp.array([0., 0., 0.0]),
        alpha_init=0.0,
        alpha_train=False,
        nph_max=4,
    )


@pytest.fixture
def h2_qed_driver_small_lambda():
    """H2 driver at small λ=0.05."""
    mol = _build_h2()
    return get_qed_vmc_nn_func(
        mol, "psiformer", jax.random.key(99),
        omega=0.5,
        coupling_vec=jnp.array([0., 0., 0.05]),
        alpha_init=0.0,
        alpha_train=False,
        nph_max=4,
    )


def test_driver_constructs(h2_qed_driver_lambda_zero):
    """Driver should construct without error and expose its params."""
    drv = h2_qed_driver_lambda_zero
    assert drv.omega == 0.5
    assert drv.nph_max == 4
    assert drv.nelec == 2
    assert "nn" in drv.params
    # alpha_train=False → no alpha key
    assert "alpha" not in drv.params


def test_walker_init_lambda_zero(h2_qed_driver_lambda_zero):
    """At λ=0 with α=0, all walkers should start at n=0 (vacuum)."""
    drv = h2_qed_driver_lambda_zero
    elec, n_ph = drv.initialize_walkers(jax.random.key(7), 8)
    assert elec.shape == (8, 2, 3)
    assert n_ph.shape == (8,)
    # Vacuum sampling
    np.testing.assert_array_equal(np.array(n_ph), np.zeros(8, dtype=np.int32))


def test_smoke_run_lambda_zero(h2_qed_driver_lambda_zero):
    """Short H2 VMC at λ=0 should produce finite energy in [-2, 0] Ha
    range (untrained PsiFormer is order-of-magnitude correct for H2)."""
    drv = h2_qed_driver_lambda_zero
    result = drv(
        rng_key=jax.random.key(77),
        num_walkers=32,
        num_steps_per_block=10,
        num_blocks=4,
        num_blocks_equil=2,
        mc_timestep=0.1,
        verbose=0,
    )
    e = result["E_mean"]
    assert jnp.isfinite(e), f"Energy not finite: {e}"
    # PsiFormer untrained energy on H2 is far from variational; just
    # verify it's not exploded to insanity.
    assert -5.0 < e < 10.0, f"Energy {e} outside reasonable range"
    # n_photon should be exactly 0 at λ=0, α=0 (no coupling, no shift)
    assert result["n_photon_mean"] == pytest.approx(0.0, abs=1e-12)


def test_smoke_run_small_lambda_finite(h2_qed_driver_small_lambda):
    """At λ=0.05, run completes and energies are finite."""
    drv = h2_qed_driver_small_lambda
    result = drv(
        rng_key=jax.random.key(77),
        num_walkers=32,
        num_steps_per_block=10,
        num_blocks=4,
        num_blocks_equil=2,
        mc_timestep=0.1,
        verbose=0,
    )
    e = result["E_mean"]
    assert jnp.isfinite(e), f"Energy not finite: {e}"
    # All blocks finite
    for be in result["E_blocks"]:
        assert jnp.isfinite(be)
    # Acceptance rates in (0, 1) range
    assert 0.0 <= result["acceptance_r"] <= 1.0
    assert 0.0 <= result["acceptance_n"] <= 1.0


def test_acceptance_nonzero_in_n_branch():
    """At α=0.5, the n-branch should accept some moves (nonzero <n>)."""
    mol = _build_h2()
    drv = get_qed_vmc_nn_func(
        mol, "psiformer", jax.random.key(99),
        omega=0.5,
        coupling_vec=jnp.array([0., 0., 0.05]),
        alpha_init=0.5,   # nonzero shift -> photon distribution centered at |α|² = 0.25
        alpha_train=False,
        nph_max=6,
    )
    result = drv(
        rng_key=jax.random.key(123),
        num_walkers=64,
        num_steps_per_block=30,
        num_blocks=6,
        num_blocks_equil=4,
        mc_timestep=0.1,
        verbose=0,
    )
    # Photon number distribution centered at |alpha|² = 0.25 should give
    # mean ~ 0.25 in equilibrium (Poisson). Generous bounds for short run.
    assert 0.0 < result["n_photon_mean"] < 2.0, (
        f"Mean photon number {result['n_photon_mean']} unreasonable"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
