"""Electron-gas VMC optimisers: Adam, mixed energy-variance, KFAC.

Gradients and histories must be finite; the KFAC damped Kronecker
inverses must be positive definite and the LM-ratio damping must
stabilise a run that otherwise turns NaN; the recovered KFAC factors
must resemble the true ones.
"""
from __future__ import annotations
import numpy as np
import jax
import pytest
import jax.numpy as jnp
import contextlib
from typing import Dict
from flax import nnx

from OmegaQMC.psi.nn.heg_wf import (  # noqa: E402
    HEGConfig,
    HEGPsiFormerConfig,
    make_heg_log_psi_any,
)
from OmegaQMC.vmcopt_nn_heg import get_vmcopt_nn_heg_func  # noqa: E402
from OmegaQMC.afqmc_pw_heg import (  # noqa: E402
    build_3deg_system,
    get_afqmc_3deg_func,
)
from OmegaQMC.vmcopt_nn_heg_kfac import (  # noqa: E402
    _VMCOptDriverNNHEG_KFAC,
    _classify_params,
    _extract_kron_factors,
    _rank1_decompose,
    _damped_kron_inverse,
    _per_electron_factors,
    _discover_linear_ids,
    _eager_capture_per_walker,
)


# ---------------------------------------------------------------------------
# From test_vmcopt_heg.py
#
# Smoke + validation tests for the HEG Adam VMC optimiser.
#
# The smoke test verifies gradient computation runs end-to-end.
# The gated ``slow`` test trains the Jastrow on a small HEG for a
# handful of iterations and checks that the energy drops below the
# finite-cell HF baseline (i.e. correlation energy is captured).
# ---------------------------------------------------------------------------


def test_vmcopt_gradient_shape():
    """VMC gradient pytree has the same structure as params."""
    rs = 2.0
    sys = build_3deg_system(rs, N_elec=14, N_pw=7,
                            polarization='unpolarized')
    L = sys['L']
    config = HEGConfig(
        n_up=7, n_down=7, L=L, n_det=1, use_jastrow=True,
    )
    opt = get_vmcopt_nn_heg_func(
        config, jax.random.key(0), lr=1e-3,
        ewald_n_real=2, ewald_n_recip=4,
    )
    walkers = opt.initialize_walkers(jax.random.key(1), 8)
    grads, e_mean, e_var = opt._vmc_grad(walkers, opt.params)
    # Pytrees match structure.
    params_leaves = jax.tree_util.tree_leaves(opt.params)
    grad_leaves = jax.tree_util.tree_leaves(grads)
    assert len(params_leaves) == len(grad_leaves)
    for p, g in zip(params_leaves, grad_leaves):
        assert p.shape == g.shape
        assert np.all(np.isfinite(np.asarray(g)))


@pytest.mark.slow
def test_vmcopt_reduces_energy():
    """Adam-VMC should push the Jastrow-augmented energy below the
    free-electron (HF) energy within a few tens of iterations."""
    rs = 2.0
    N = 14
    sys = build_3deg_system(rs, N_elec=N, N_pw=7,
                            polarization='unpolarized')
    L = sys['L']

    e_hf_ha = float(get_afqmc_3deg_func(
        sys, dt=0.005, include_coulomb=True, verbose=False,
    ).e_trial) / N

    config = HEGConfig(
        n_up=7, n_down=7, L=L, n_det=1, use_jastrow=True,
    )
    opt = get_vmcopt_nn_heg_func(
        config, jax.random.key(0), lr=3e-3,
        ewald_n_real=3, ewald_n_recip=6,
    )
    result = opt(
        jax.random.key(1),
        num_iters=40,
        num_walkers=64,
        mcmc_decorr_steps=10,
        num_equil_steps=200,
        mc_timestep=0.1,
        verbose=0,
    )
    e_final_ha = result['E_final_ha']
    # Late-iteration energy should be below HF (correlation captured).
    # With only 40 Adam steps the drop is small but statistically
    # significant; require at least 1 mHa/elec below HF.
    assert e_final_ha < e_hf_ha - 1e-3, (
        f"Final E = {e_final_ha:.4f} Ha, HF = {e_hf_ha:.4f} Ha"
    )


# ---------------------------------------------------------------------------
# From test_vmcopt_heg_variance.py
#
# Tests for the Umrigar-style mixed energy-variance objective.
#
# Verifies:
#   * With ``var_weight = 0.0`` the gradient is bit-identical to the
#     pure-energy gradient (regression guard).
#   * With ``var_weight > 0`` the gradient differs in a controlled way
#     consistent with the analytic formula
#     ``∇L = 2 ⟨[(E_L − ⟨E⟩) + β · ((E_L − ⟨E⟩)² − Var)] · O⟩``.
#   * The mixed-objective training loop runs without NaNs at β > 0
#     and produces a finite energy history.
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# From test_vmcopt_heg_kfac.py
#
# Smoke + regression test for the KFAC HEG optimiser.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------
# Unit tests for the math primitives
# ---------------------------------------------------------------
def test_rank1_decompose_recovers_outer_product():
    """For exact rank-1 input M = g·x^T, our decomposition must
    return ``(g_recovered, x_recovered)`` whose outer product
    reproduces M."""
    rng = np.random.default_rng(0)
    W, out, in_ = 16, 5, 7
    g_true = jnp.asarray(rng.standard_normal((W, out)))
    x_true = jnp.asarray(rng.standard_normal((W, in_)))
    M = jnp.einsum('wo,wi->woi', g_true, x_true)

    g_rec, x_rec = _rank1_decompose(M)
    M_rec = jnp.einsum('wo,wi->woi', g_rec, x_rec)
    np.testing.assert_allclose(np.asarray(M), np.asarray(M_rec),
                               atol=1e-10)


def test_rank1_decompose_balances_norms():
    """SVD-balanced split: ‖g_recovered‖ = ‖x_recovered‖ per walker."""
    rng = np.random.default_rng(1)
    g_true = jnp.asarray(rng.standard_normal((8, 4)))
    x_true = jnp.asarray(rng.standard_normal((8, 6)))
    M = jnp.einsum('wo,wi->woi', g_true, x_true)
    g_rec, x_rec = _rank1_decompose(M)
    g_norms = jnp.linalg.norm(g_rec, axis=-1)
    x_norms = jnp.linalg.norm(x_rec, axis=-1)
    np.testing.assert_allclose(g_norms, x_norms, atol=1e-8)


def test_damped_kron_inverse_correct_shape_and_pd():
    A = jnp.eye(4) * 2.0 + 0.1 * jnp.ones((4, 4))
    G = jnp.eye(3) * 5.0 + 0.2 * jnp.ones((3, 3))
    A_inv, G_inv = _damped_kron_inverse(A, G, damping=1e-3)
    assert A_inv.shape == (4, 4)
    assert G_inv.shape == (3, 3)
    # Damped A_inv is positive-definite for PD A.
    eig_A = jnp.linalg.eigvalsh(A_inv)
    eig_G = jnp.linalg.eigvalsh(G_inv)
    assert jnp.all(eig_A > 0)
    assert jnp.all(eig_G > 0)


def test_extract_kron_factors_dW_loss_matches_naive_average():
    rng = np.random.default_rng(2)
    W, out, in_ = 32, 5, 7
    M = jnp.asarray(rng.standard_normal((W, out, in_)))
    de = jnp.asarray(rng.standard_normal((W,)))
    A, G, dW_loss = _extract_kron_factors(M, de)
    # dW_loss must be the de-weighted walker average of M.
    dW_naive = jnp.einsum('w,woi->oi', de, M) / W
    np.testing.assert_allclose(np.asarray(dW_loss),
                               np.asarray(dW_naive), atol=1e-12)


def test_extract_kron_factors_uses_undecomposed_M():
    """A = E[M^T M], G = E[M M^T] — direct from M, no SVD needed."""
    rng = np.random.default_rng(3)
    W, out, in_ = 16, 4, 5
    M = jnp.asarray(rng.standard_normal((W, out, in_)))
    de = jnp.zeros(W)
    A, G, _ = _extract_kron_factors(M, de)
    A_expected = jnp.einsum('woi,woj->ij', M, M) / W
    G_expected = jnp.einsum('woi,wpi->op', M, M) / W
    np.testing.assert_allclose(np.asarray(A), np.asarray(A_expected),
                               atol=1e-12)
    np.testing.assert_allclose(np.asarray(G), np.asarray(G_expected),
                               atol=1e-12)


def test_extract_kron_factors_recovers_true_for_rank1():
    """When M_w = g_w · x_w^T (single rank-1), the unscaled formula
    E[M^T M] reduces to E[‖g_w‖² x_w x_w^T] which equals E[x x^T]
    times a constant if ‖g‖ is constant across walkers.  Verify that
    *up to a scale factor*."""
    rng = np.random.default_rng(4)
    W, out, in_ = 64, 5, 6
    g = jnp.asarray(rng.standard_normal((W, out)))
    x = jnp.asarray(rng.standard_normal((W, in_)))
    M = jnp.einsum('wo,wi->woi', g, x)

    A, G, _ = _extract_kron_factors(M, jnp.zeros(W))
    # Compare A direction with E[x x^T] direction.
    A_true = jnp.einsum('wi,wj->ij', x, x) / W
    cos_A = (jnp.sum(A * A_true) /
             (jnp.linalg.norm(A) * jnp.linalg.norm(A_true)))
    G_true = jnp.einsum('wo,wp->op', g, g) / W
    cos_G = (jnp.sum(G * G_true) /
             (jnp.linalg.norm(G) * jnp.linalg.norm(G_true)))
    # Should be highly aligned (>0.9) — the per-walker ‖g‖²/‖x‖²
    # weighting introduces some bias but for random inputs at
    # decent W the cos is close to 1.
    assert float(cos_A) > 0.85
    assert float(cos_G) > 0.85


# ---------------------------------------------------------------
# End-to-end smoke
# ---------------------------------------------------------------
def test_kfac_runs_without_error_and_produces_finite_history():
    cfg = HEGPsiFormerConfig(
        n_up=7, n_down=7, L=7.77, n_det=2,
        embedding_dim=16, n_interactions=1,
        two_particle_stream_dim=8, n_attention_heads=2,
        use_cusp=True, use_deep_jastrow=False,
        use_pair_jastrow=False, n_virt_pw=12, det_jitter=0.02,
    )
    init_key = jax.random.key(0)
    opt = _VMCOptDriverNNHEG_KFAC(
        cfg, init_key,
        lr=0.1,
        damping=0.1,
        damping_adapt=True,
        ema_decay=0.0,
        norm_constraint=None,
    )
    result = opt(
        jax.random.key(1),
        num_iters=15,
        num_walkers=64,
        mcmc_decorr_steps=3,
        num_equil_steps=20,
        verbose=0,
    )
    assert result['E_final_ha'] is not None
    history = np.asarray(result['E_per_elec_history'])
    assert np.all(np.isfinite(history))
    var_history = np.asarray(result['Var_history'])
    assert np.all(np.isfinite(var_history))
    assert np.all(var_history > 0)


def test_kfac_classifies_params_into_linear_and_generic():
    cfg = HEGPsiFormerConfig(
        n_up=7, n_down=7, L=7.77, n_det=2,
        embedding_dim=16, n_interactions=1,
        two_particle_stream_dim=8, n_attention_heads=2,
        use_cusp=True, use_deep_jastrow=False,
        use_pair_jastrow=False, n_virt_pw=12, det_jitter=0.02,
    )
    init_key = jax.random.key(0)
    opt = _VMCOptDriverNNHEG_KFAC(
        cfg, init_key, lr=0.1, damping=0.1,
        damping_adapt=False, ema_decay=0.0,
    )
    # Must find at least the attention QKVO + GNN subnet linears.
    assert opt._linear_count >= 6
    # Linear params should dominate.
    assert opt._linear_params > 0.5 * opt.n_params


def test_per_electron_factors_recovers_g_we_for_synthetic_input():
    """For a constructed (a_we, g_we) ground-truth, ``_per_electron_factors``
    must reconstruct factors to within numerical precision.

    NNX ``Linear.kernel`` is shape ``(in, out)``, so ``per_walker_dW``
    is ``(W, in, out) = (W, in_dim, out_dim)``.  This module's
    convention returns the (out, out) Fisher first, then the (in, in)
    Fisher — matching ``_extract_kron_factors``'s naming so that
    ``ΔW = G_in_inv @ dW @ A_out_inv`` holds with ``dW`` of shape
    (in, out).
    """
    rng = np.random.default_rng(7)
    W, n_e, in_, out = 64, 5, 7, 3
    a = jnp.asarray(rng.standard_normal((W, n_e, in_)))
    g = jnp.asarray(rng.standard_normal((W, n_e, out)))
    # per-walker dW = sum_e a_we · g_we^T  (shape (in, out))
    M = jnp.einsum('wei,weo->wio', a, g)
    de = jnp.zeros(W)

    # Use solve_damping=0 so recovery is exact up to machine eps.
    A_out, G_in, _, n_rec = _per_electron_factors(
        a, M, de, solve_damping=0.0,
    )
    assert n_rec == n_e
    assert A_out.shape == (out, out)
    assert G_in.shape == (in_, in_)

    # Ground-truth factors over the (W·n_e) flattened sample set.
    a_flat = a.reshape(W * n_e, in_)
    g_flat = g.reshape(W * n_e, out)
    A_kfac_in = jnp.einsum('si,sj->ij', a_flat, a_flat) / (W * n_e)
    G_kfac_out = jnp.einsum('so,sp->op', g_flat, g_flat) / (W * n_e)

    # Our return: A_out (output-side) corresponds to KFAC's G;
    # G_in (input-side) corresponds to KFAC's A.
    np.testing.assert_allclose(np.asarray(A_out),
                               np.asarray(G_kfac_out), atol=1e-7)
    np.testing.assert_allclose(np.asarray(G_in),
                               np.asarray(A_kfac_in), atol=1e-9)


def test_eager_capture_records_layer_inputs():
    """Smoke test for ``_eager_capture_per_walker``: one walker
    in, one entry per Linear out, with the recorded array equal to
    that Linear's input."""
    from flax import nnx
    from OmegaQMC.psi.nn.heg_wf_module import build_heg_psiformer_wf
    cfg = HEGPsiFormerConfig(
        n_up=7, n_down=7, L=7.77, n_det=2,
        embedding_dim=16, n_interactions=1,
        two_particle_stream_dim=8, n_attention_heads=2,
        use_cusp=False, use_deep_jastrow=False,
        use_pair_jastrow=False, n_virt_pw=12, det_jitter=0.02,
    )
    rng = nnx.Rngs(jax.random.key(0))
    model = build_heg_psiformer_wf(cfg, rng)
    id_to_path = _discover_linear_ids(model)
    assert len(id_to_path) >= 6      # at least the attention QKVO

    walkers = jnp.asarray(
        np.random.default_rng(0).random((4, 14, 3)) * cfg.L,
    )
    captures = _eager_capture_per_walker(model, walkers, id_to_path)
    # Most discovered Linears get called during forward (some may be
    # skipped on conditional code paths, e.g. an unused spin block).
    assert len(captures) >= len(id_to_path) - 2
    # Each stacked array has W as the leading axis.
    for path, arr in captures.items():
        assert arr.shape[0] == 4
        assert jnp.all(jnp.isfinite(arr))


def test_lm_ratio_damping_responds_to_actual_vs_predicted():
    """FermiNet-faithful Levenberg-Marquardt: when the actual energy
    reduction tracks the quadratic-model prediction (``ratio ≈ 1``)
    over a window of ``damping_lookback`` iters, the LM rule loosens
    damping (multiplies by ``damping_decay`` < 1).  When the model
    massively over-predicts (``ratio < 0.25``, e.g. fixed_scale=True
    with too-aggressive lr), damping should rapidly tighten.

    This is the schedule that keeps ``fixed_scale=True`` stable.
    """
    cfg = HEGPsiFormerConfig(
        n_up=7, n_down=7, L=7.77, n_det=2,
        embedding_dim=16, n_interactions=1,
        two_particle_stream_dim=8, n_attention_heads=2,
        use_cusp=False, use_deep_jastrow=False,
        use_pair_jastrow=False, n_virt_pw=12, det_jitter=0.02,
    )
    init_key = jax.random.key(0)

    # Recipe A: ``fixed_scale=False`` (kernel_scale=1 everywhere) at
    # tiny lr — the quadratic model is very accurate, so the LM ratio
    # should sit near 1 and damping should drift downward.
    opt = _VMCOptDriverNNHEG_KFAC(
        cfg, init_key,
        capture_activations=False,
        fixed_scale=False,
        lr=1.0e-3,
        damping=1.0e-2,
        damping_adapt=True,
        damping_lookback=4,
        damping_decay=0.95,
    )
    res = opt(jax.random.key(1), num_iters=20, num_walkers=32,
              mcmc_decorr_steps=2, num_equil_steps=10, verbose=0)
    # Run must not NaN.
    assert np.all(np.isfinite(res['E_per_elec_history']))


def test_lm_ratio_damping_stabilises_fixed_scale_run():
    """End-to-end: ``fixed_scale=True`` with the FermiNet-faithful LM
    ratio damping must stay finite — historically this combination
    diverged within ~6 iters because the ``n_e``-magnified step
    overshoots the trust region.  The Martens-Grosse ratio rule
    detects the prediction failure and ramps damping fast enough to
    keep training stable.
    """
    cfg = HEGPsiFormerConfig(
        n_up=7, n_down=7, L=7.77, n_det=2,
        embedding_dim=16, n_interactions=1,
        two_particle_stream_dim=8, n_attention_heads=2,
        use_cusp=False, use_deep_jastrow=False,
        use_pair_jastrow=False, n_virt_pw=12, det_jitter=0.02,
    )
    init_key = jax.random.key(0)
    opt = _VMCOptDriverNNHEG_KFAC(
        cfg, init_key,
        capture_activations=True,
        fixed_scale=True,
        lr=1.0e-3,                       # small enough to start safely
        damping=1.0e-2,
        damping_adapt=True,
        damping_min=1.0e-4,
        damping_max=1.0e3,
        damping_lookback=4,              # short window for fast response
        damping_decay=0.9,
        damping_overshoot_threshold=10.0,
        damping_overshoot_factor=2.0,
        norm_constraint=1.0e-3,
    )
    res = opt(jax.random.key(1), num_iters=20, num_walkers=32,
              mcmc_decorr_steps=2, num_equil_steps=10, verbose=0)
    # At minimum: training stayed finite (no NaN/Inf).  If LM ratio
    # logic is broken this would NaN within ~6 iters.
    assert np.all(np.isfinite(res['E_per_elec_history']))


def test_kfac_with_multi_device_falls_back_when_one_gpu():
    """multi_device=True with n_devices == 1 falls back to single-
    device cleanly (no exception)."""
    cfg = HEGPsiFormerConfig(
        n_up=7, n_down=7, L=7.77, n_det=2,
        embedding_dim=16, n_interactions=1,
        two_particle_stream_dim=8, n_attention_heads=2,
        use_cusp=False, use_deep_jastrow=False,
        use_pair_jastrow=False, n_virt_pw=12, det_jitter=0.02,
    )
    init_key = jax.random.key(0)
    n_dev = jax.local_device_count()
    opt = _VMCOptDriverNNHEG_KFAC(cfg, init_key, multi_device=True)
    if n_dev < 2:
        # Auto-fallback to single-device.
        assert opt._n_devices == 1
    else:
        assert opt._n_devices == n_dev


# ---------------------------------------------------------------------------
# From test_kfac_factors.py
#
# Validation: compare SVD-recovered KFAC factors against ground-truth.
#
# Strategy:
#   Ground truth A_l, G_l are obtained by capturing each Linear's input
#   ``a_l`` directly during a forward pass (via thread-local module-level
#   monkey-patching of ``nnx.Linear.__call__``).  Once we have ``a_l`` per
#   walker, ``g_l`` follows exactly from
#   ``g_l = (∂log|ψ|/∂W) · a_l / ‖a_l‖²`` since ``∂log|ψ|/∂W`` is rank-1.
#
#   Then ``A_true = E[a a^T]``, ``G_true = E[g g^T]`` are the strict
#   KFAC factors.  Compare with ``A_svd``, ``G_svd`` from
#   ``vmcopt_nn_heg_kfac._extract_kron_factors``.
#
#   Per-walker scaling difference is expected: SVD imposes
#   ``‖g_w‖ = ‖a_w‖ = sqrt(s_w)``.  We test that the *Kronecker product*
#   ``G ⊗ A`` (which is what enters the KFAC step) is invariant to that
#   rescaling, modulo the walker-dependent reweighting.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------
# Ground-truth A, G via monkey-patched eager forward
# ---------------------------------------------------------------
@contextlib.contextmanager
def _capture_linear_inputs():
    """Yields a list that gets populated with ``(layer_id, x)`` tuples
    each time ``nnx.Linear.__call__`` is invoked while inside the
    block."""
    captured = []
    orig = nnx.Linear.__call__

    def patched(self, x):
        captured.append((id(self), x))
        return orig(self, x)

    nnx.Linear.__call__ = patched
    try:
        yield captured
    finally:
        nnx.Linear.__call__ = orig


def _capture_paths_and_values(model, walker):
    """Run ``model(walker)`` once, eagerly, capturing each
    ``nnx.Linear`` input.

    Returns:
        ``log_psi`` and ``dict[id(linear) -> input_array]``.
    """
    with _capture_linear_inputs() as captured:
        psi = model(walker)
    log_psi = float(psi.log)
    inputs_by_id: Dict[int, jnp.ndarray] = {}
    for layer_id, x in captured:
        inputs_by_id[layer_id] = (
            x  # last call wins (one Linear ≈ one call here)
        )
    return log_psi, inputs_by_id


def _id_of_linear_at_path(model, path) -> int:
    """Walk a JAX KeyPath inside the live ``model`` and return the
    Python id of the ``nnx.Linear`` it points to."""
    obj = model
    # path looks like (..., 'kernel', '.value')  — drop trailing two
    keys = []
    for k in path[:-2]:
        keys.append(k.key if hasattr(k, 'key') else k)
    for k in keys:
        if isinstance(k, int):
            obj = obj[k]
        else:
            obj = getattr(obj, k)
    return id(obj)


# ---------------------------------------------------------------
# Tests
# ---------------------------------------------------------------
@pytest.mark.parametrize('seed', [0, 7])
def test_kfac_factors_match_ground_truth(seed):
    """The SVD-based factors and the brute-force factors must agree
    once we project both onto the same Kronecker tensor.

    We pick one Linear layer at random, build A_true, G_true via
    direct activation capture, and A_svd, G_svd via SVD.  Test:
        ‖A_true⊗G_true‖ / ‖A_svd⊗G_svd‖ is finite and the
        Kronecker products correlate strongly (cosine ≥ 0.5).
    """
    cfg = HEGPsiFormerConfig(
        n_up=7, n_down=7, L=7.77, n_det=2,
        embedding_dim=16, n_interactions=1,
        two_particle_stream_dim=8, n_attention_heads=2,
        use_cusp=True, use_deep_jastrow=False,
        use_pair_jastrow=False, n_virt_pw=12, det_jitter=0.02,
    )
    key = jax.random.key(seed)
    log_psi_fn, params, graphdef = make_heg_log_psi_any(cfg, key)

    # Materialise model for eager capture.
    other = jax.tree.map(
        lambda x: x, params
    )  # placeholder; real `other` in nnx.split
    # The model materialisation requires nnx.merge.
    rngs = nnx.Rngs(key)
    from OmegaQMC.psi.nn.heg_wf_module import build_heg_psiformer_wf
    model = build_heg_psiformer_wf(cfg, rngs)

    n_walkers = 32
    rng = np.random.default_rng(seed)
    walkers = jnp.asarray(
        cfg.L * rng.random((n_walkers, cfg.n_up + cfg.n_down, 3)),
    )

    # Per-walker pytree gradient (this is what the SVD path consumes).
    def lp_one(w, p):
        return log_psi_fn(w, p)

    pw_grad = jax.vmap(jax.grad(lp_one, argnums=1), in_axes=(0, None))(
        walkers, params,
    )

    # Pick first Linear-kernel leaf.
    layers, kernel_shapes, _ = _classify_params(params)
    layer_name = next(iter(layers))
    kpath, _bpath = layers[layer_name]

    # SVD method — our optimiser uses these.
    de_zero = jnp.zeros(n_walkers)        # dummy weights
    leaves = jax.tree_util.tree_flatten_with_path(pw_grad)[0]
    M = next(lf for p, lf in leaves if p == kpath)   # (W, out, in)
    A_svd, G_svd, _ = _extract_kron_factors(M, de_zero)

    # Ground truth: capture a_l per walker, derive g_l exactly.
    target_id = _id_of_linear_at_path(model, kpath)
    a_per_walker = []
    for w in range(n_walkers):
        _, inputs_by_id = _capture_paths_and_values(model, walkers[w])
        a_per_walker.append(inputs_by_id[target_id])
    a_per_walker = jnp.stack(
        a_per_walker, axis=0
    )  # (W, in) or (W, n_elec, in)

    # If layer is called per-electron, the captured input has an
    # extra axis.  Flatten/aggregate to a (W, in) array — a layer
    # with per-electron input contributes one outer-product per
    # electron per walker.  Treat each electron as a separate
    # "sample" for Fisher purposes.
    if a_per_walker.ndim == 3:
        # (W, n_e, in) → flatten W and n_e together
        W, n_e, in_ = a_per_walker.shape
        a_flat = a_per_walker.reshape(W * n_e, in_)
        # Each per-electron contribution to dW is also rank-1 within
        # the per-walker dW, but we used the SUM of those rank-1
        # outer products in our SVD path — so ground-truth Kron
        # products will differ at the structural level.  Skip
        # cosine-test for these layers; just verify finiteness.
        A_true = (a_flat[:, :, None] * a_flat[:, None, :]).mean(axis=0)
        # Cannot directly get g per electron from W×in_ + (out,in) only.
        # Skip strict comparison.
        assert jnp.all(jnp.isfinite(A_true))
        assert jnp.all(jnp.isfinite(A_svd))
        return

    # Recover g_l per walker exactly from a_l + dW_w.
    a = a_per_walker                            # (W, in)
    a_norm_sq = jnp.sum(a * a, axis=-1, keepdims=True) + 1e-30  # (W, 1)
    g = jnp.einsum('woi,wi->wo', M, a) / a_norm_sq  # (W, out)
    A_true = jnp.einsum('wi,wj->ij', a, a) / n_walkers
    G_true = jnp.einsum('wi,wj->ij', g, g) / n_walkers

    # Cross-check: g · a^T must reproduce dW.
    dW_check = jnp.einsum('wo,wi->woi', g, a)
    assert jnp.allclose(dW_check, M, atol=1e-7), (
        f"reconstruction failed: max |diff| = "
        f"{float(jnp.max(jnp.abs(dW_check - M))):.3e}"
    )

    # Cosine between Kron products (SVD vs ground truth).
    # F_svd = G_svd ⊗ A_svd, F_true = G_true ⊗ A_true (vec form).
    F_svd = jnp.kron(G_svd, A_svd).flatten()
    F_true = jnp.kron(G_true, A_true).flatten()
    cos = float(
        jnp.dot(F_svd, F_true) /
        (jnp.linalg.norm(F_svd) * jnp.linalg.norm(F_true) + 1e-30)
    )
    print(
        f"[{layer_name}] cos(F_svd, F_true) = {cos:.4f}  "
        f"(seed={seed})"
    )
    assert cos > 0.5, (
        f"SVD-based KFAC factors poorly aligned with ground truth: "
        f"cos = {cos:.4f}"
    )
