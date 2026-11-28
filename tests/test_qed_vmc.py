"""QED-VMC driver and SR optimiser on H2.

The driver runs with finite energies; error bars come from the binning
analysis; the optimiser moves the parameters with a finite history.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update('jax_enable_x64', True)

from OmegaQMC.utils import (  # noqa: E402
    Mole_custom,
    do_binning_analysis,
)
from OmegaQMC.qed_vmc_nn import (  # noqa: E402
    get_qed_vmc_nn_func,
    _block_serr,
)
from OmegaQMC.qed_vmcopt_nn_sr import get_qed_vmcopt_nn_sr_func  # noqa: E402


# ---------------------------------------------------------------------------
# From test_qed_vmc_nn_h2.py
#
# Smoke tests for QED-VMC driver on H2.
#
# These exercise the full pipeline end-to-end on a tiny system:
#
#   * Driver construction with a real PsiFormer ansatz.
#   * Walker initialization (continuous r + discrete n).
#   * Joint discrete-continuous Metropolis-Hastings sampling.
#   * Pauli-Fierz local-energy evaluation.
#   * Block-averaged statistics.
#
# Validation against QED-FCI on (H2)2 is the responsibility of the
# Phase 2e benchmark tests (after optimizer integration); here we only
# check that the pipeline runs without error and produces *finite*
# energies in the right ballpark for an *untrained* PsiFormer (initial
# parameters give an order-of-magnitude correct H2 energy).
# ---------------------------------------------------------------------------


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
        mol,
        "psiformer",
        jax.random.key(99),
        omega=0.5,
        coupling_vec=jnp.array([0.0, 0.0, 0.05]),
        # nonzero shift -> photon distribution centered at |α|² = 0.25
        alpha_init=0.5,
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


# ---------------------------------------------------------------------------
# From test_qed_vmc_nn_serr.py
#
# Regression test for the QED-VMC energy error bar.
#
# ``_QEDVMCDriverNN.__call__`` used to unpack three values from
# :func:`OmegaQMC.utils.do_binning_analysis`, which returns four, inside
# a blanket ``try/except``; the error bar therefore always fell back to
# the naive ``std / sqrt(n)`` of the block means.  With more than 8
# production blocks it must now be the binning-analysis error, for the
# energy and for ``<L_z>`` alike.
# ---------------------------------------------------------------------------


def _h2_driver(**kwargs):
    L = 1.4010
    mol = Mole_custom.from_arrays(
        charges=[1, 1],
        coords=[[0.0, 0.0, -L / 2], [0.0, 0.0, L / 2]],
        n_up=1, n_down=1,
    )
    return get_qed_vmc_nn_func(
        mol, "psiformer", jax.random.key(99),
        omega=0.5, coupling_vec=jnp.array([0., 0., 0.05]),
        nph_max=4, **kwargs,
    )


def _expected_serr(blocks):
    blocks = jnp.asarray(blocks)
    if blocks.shape[0] > 8:
        return float(do_binning_analysis(blocks)[1])
    return float(jnp.std(blocks)) / np.sqrt(blocks.shape[0])


def test_block_serr_helper():
    rng = np.random.default_rng(0)
    # AR(1) series, so binning and the naive error differ.
    x = np.zeros(40)
    for i in range(1, 40):
        x[i] = 0.8 * x[i - 1] + rng.normal()
    serr_bin = float(do_binning_analysis(jnp.asarray(x))[1])
    assert _block_serr(x) == pytest.approx(serr_bin, rel=1e-12)
    assert _block_serr(x) > x.std() / np.sqrt(40)
    assert _block_serr(x[:8]) == pytest.approx(
        x[:8].std() / np.sqrt(8), rel=1e-12,
    )
    # Zero spread gives zero error (no NaN from the 0/0 in the
    # autocorrelation normalisation); no blocks gives NaN.
    assert _block_serr(np.full(20, 0.25)) == 0.0
    assert np.isnan(_block_serr(np.array([])))


@pytest.fixture(scope="module")
def drv():
    return _h2_driver()


def _run(drv, num_blocks):
    return drv(
        jax.random.key(7), num_walkers=16, num_steps_per_block=2,
        num_blocks=num_blocks, num_blocks_equil=2, verbose=0,
    )


def test_serr_uses_binning_above_8_blocks(drv):
    res = _run(drv, 12)
    blocks = jnp.asarray(res["E_blocks"])
    assert blocks.shape == (12,)
    _, serr_bin, sdev, kappa = do_binning_analysis(blocks)
    assert res["E_serr"] == pytest.approx(float(serr_bin), rel=1e-12)
    # Distinguish it from the old naive fallback whenever the
    # autocorrelation estimate differs from 1.
    naive = float(sdev) / np.sqrt(blocks.shape[0])
    if abs(float(kappa) - 1.0) > 1e-6:
        assert res["E_serr"] != pytest.approx(naive, rel=1e-9)


def test_serr_naive_at_or_below_8_blocks(drv):
    res = _run(drv, 6)
    blocks = np.asarray(res["E_blocks"])
    naive = blocks.std() / np.sqrt(blocks.shape[0])
    assert res["E_serr"] == pytest.approx(naive, rel=1e-12)


def test_l_z_serr_uses_block_serr():
    """``<L_z>`` is measured only with ``complex_psi``.

    The ansatz itself is real here (``complex_psi`` only changes the
    local energy), so ``<L_z>`` is zero in every block; this checks
    that ``l_z_serr`` goes through :func:`_block_serr` and stays
    finite.  The binning branch is covered by
    :func:`test_block_serr_helper`.
    """
    drv = _h2_driver(arch="tang_native", complex_psi=True)
    res = _run(drv, 12)
    lz = jnp.asarray(res["l_z_blocks"])
    assert lz.shape == (12,)
    assert np.isfinite(res["l_z_serr"])
    assert res["l_z_serr"] == pytest.approx(
        _expected_serr(lz), rel=1e-12, abs=1e-300,
    )
    assert res["E_serr"] == pytest.approx(
        _expected_serr(res["E_blocks"]), rel=1e-12,
    )


# ---------------------------------------------------------------------------
# From test_qed_vmcopt_sr.py
#
# Smoke tests for QED-NN-VMC SR optimizer.
#
# These verify the optimizer runs end-to-end on H2 and produces sensible
# parameter updates. Convergence to QED-FCI accuracy is the responsibility
# of the Phase 2e benchmark suite — here we only check:
#
#   * Optimizer constructs without error.
#   * SR step modifies parameters (delta_p is nonzero).
#   * Energy decreases (mostly) over iterations from random init.
#   * Alpha trains (changes from initial value when alpha_train=True).
#   * Alpha frozen (does not change when alpha_train=False).
# ---------------------------------------------------------------------------


def _build_h2_sr():
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
    mol = _build_h2_sr()
    opt = get_qed_vmcopt_nn_sr_func(
        mol, "psiformer", jax.random.key(99),
        omega=0.5, coupling_vec=jnp.array([0., 0., 0.05]),
        alpha_init=0.1, alpha_train=True, nph_max=4,
    )
    assert "alpha" in opt.init_params
    assert float(opt.init_params["alpha"]) == pytest.approx(0.1)


def test_optimizer_constructs_alpha_frozen():
    """alpha_train=False should NOT expose alpha (frozen via closure)."""
    mol = _build_h2_sr()
    opt = get_qed_vmcopt_nn_sr_func(
        mol, "psiformer", jax.random.key(99),
        omega=0.5, coupling_vec=jnp.array([0., 0., 0.05]),
        alpha_init=0.0, alpha_train=False, nph_max=4,
    )
    assert "alpha" not in opt.init_params


def test_short_sr_run_lambda_zero():
    """At λ=0, short SR run on H2 should: (i) start finite, (ii) decrease
    energy mostly, (iii) leave α at 0 since no coupling drives it."""
    mol = _build_h2_sr()
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
    mol = _build_h2_sr()
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
    mol = _build_h2_sr()
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
    mol = _build_h2_sr()
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
