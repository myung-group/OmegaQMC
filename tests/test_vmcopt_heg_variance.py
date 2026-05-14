"""Tests for the Umrigar-style mixed energy-variance objective.

Verifies:
  * With ``var_weight = 0.0`` the gradient is bit-identical to the
    pure-energy gradient (regression guard).
  * With ``var_weight > 0`` the gradient differs in a controlled way
    consistent with the analytic formula
    ``∇L = 2 ⟨[(E_L − ⟨E⟩) + β · ((E_L − ⟨E⟩)² − Var)] · O⟩``.
  * The mixed-objective training loop runs without NaNs at β > 0
    and produces a finite energy history.
"""

import numpy as np
import jax
import jax.numpy as jnp
import pytest

from OmegaQMC.psi.nn.heg_wf import HEGConfig
from OmegaQMC.vmcopt_nn_heg import get_vmcopt_nn_heg_func
from OmegaQMC.afqmc_pw_heg import build_3deg_system


def _cfg():
    sys = build_3deg_system(
        rs=2.0, N_elec=14, N_pw=7, polarization='unpolarized',
    )
    return HEGConfig(n_up=7, n_down=7, L=sys['L'],
                     n_det=1, use_jastrow=True)


def test_zero_var_weight_matches_pure_energy_gradient():
    cfg = _cfg()
    opt_pure = get_vmcopt_nn_heg_func(
        cfg, jax.random.key(0), lr=1e-3,
        ewald_n_real=2, ewald_n_recip=4,
    )
    opt_zero = get_vmcopt_nn_heg_func(
        cfg, jax.random.key(0), lr=1e-3,
        var_weight=0.0,
        ewald_n_real=2, ewald_n_recip=4,
    )
    walkers = opt_pure.initialize_walkers(jax.random.key(1), 16)
    g_pure, e_pure, var_pure = opt_pure._vmc_grad(
        walkers, opt_pure.params,
    )
    g_zero, e_zero, var_zero = opt_zero._vmc_grad(
        walkers, opt_zero.params,
    )
    assert float(e_pure) == float(e_zero)
    assert float(var_pure) == float(var_zero)
    # All gradient leaves bit-identical.
    for g_p, g_z in zip(
        jax.tree_util.tree_leaves(g_pure),
        jax.tree_util.tree_leaves(g_zero),
    ):
        np.testing.assert_array_equal(np.asarray(g_p), np.asarray(g_z))


def test_var_weight_shifts_gradient_in_expected_direction():
    """With β > 0, the gradient decomposes as
    ``g_total = g_energy + β · g_variance`` where
    ``g_variance = 2 ⟨(δE² − Var) · O⟩``.  We verify this by
    comparing the β=0 and β>0 gradients."""
    cfg = _cfg()
    beta = 0.05
    opt_pure = get_vmcopt_nn_heg_func(
        cfg, jax.random.key(0), lr=1e-3,
        ewald_n_real=2, ewald_n_recip=4,
    )
    opt_mix = get_vmcopt_nn_heg_func(
        cfg, jax.random.key(0), lr=1e-3,
        var_weight=beta,
        ewald_n_real=2, ewald_n_recip=4,
    )
    walkers = opt_pure.initialize_walkers(jax.random.key(1), 16)
    g_pure, _, _ = opt_pure._vmc_grad(walkers, opt_pure.params)
    g_mix, _, _ = opt_mix._vmc_grad(walkers, opt_mix.params)

    # The difference (g_mix − g_pure) / β must equal the
    # "pure-variance" gradient component (same analytic formula,
    # β = 1).
    opt_var_only = get_vmcopt_nn_heg_func(
        cfg, jax.random.key(0), lr=1e-3,
        var_weight=1.0,
        ewald_n_real=2, ewald_n_recip=4,
    )
    g_var_only, _, _ = opt_var_only._vmc_grad(
        walkers, opt_var_only.params,
    )
    # g_var_only = g_pure + 1.0 · g_variance → g_variance = g_var − g_pure.
    # Expected: (g_mix − g_pure) / β = g_variance = g_var − g_pure.
    for gm, gp, gv in zip(
        jax.tree_util.tree_leaves(g_mix),
        jax.tree_util.tree_leaves(g_pure),
        jax.tree_util.tree_leaves(g_var_only),
    ):
        diff = np.asarray(gm) - np.asarray(gp)
        expected = beta * (np.asarray(gv) - np.asarray(gp))
        np.testing.assert_allclose(diff, expected, rtol=1e-6, atol=1e-10)


@pytest.mark.slow
def test_mixed_training_runs_and_finishes():
    cfg = _cfg()
    opt = get_vmcopt_nn_heg_func(
        cfg, jax.random.key(0), lr=3e-3,
        var_weight=0.05,
        ewald_n_real=3, ewald_n_recip=6,
    )
    result = opt(
        jax.random.key(1),
        num_iters=30,
        num_walkers=64,
        mcmc_decorr_steps=5,
        num_equil_steps=100,
        mc_timestep=0.1,
        verbose=0,
    )
    assert np.all(np.isfinite(result['E_per_elec_history']))
    assert result['E_final_ha'] is not None
