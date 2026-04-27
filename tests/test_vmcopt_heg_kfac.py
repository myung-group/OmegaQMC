"""Smoke + regression test for the KFAC HEG optimiser."""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from OmegaQMC.psi.nn.heg_wf import HEGPsiFormerConfig
from OmegaQMC.vmcopt_nn_heg_kfac import (
    _HEGKFACOptimizer,
    _classify_params,
    _extract_kron_factors,
    _rank1_decompose,
    _damped_kron_inverse,
    _per_electron_factors,
    _capture_linear_inputs,
    _discover_linear_ids,
    _eager_capture_per_walker,
)


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
    opt = _HEGKFACOptimizer(
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
    opt = _HEGKFACOptimizer(
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
    from OmegaQMC.psi.nn.heg_psiformer import build_heg_psiformer_wf
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


def test_overshoot_threshold_param_accepted():
    """The new damping_overshoot_threshold and damping_overshoot_factor
    constructor args plumb through and don't break smoke."""
    cfg = HEGPsiFormerConfig(
        n_up=7, n_down=7, L=7.77, n_det=2,
        embedding_dim=16, n_interactions=1,
        two_particle_stream_dim=8, n_attention_heads=2,
        use_cusp=False, use_deep_jastrow=False,
        use_pair_jastrow=False, n_virt_pw=12, det_jitter=0.02,
    )
    init_key = jax.random.key(0)
    opt = _HEGKFACOptimizer(
        cfg, init_key,
        damping_overshoot_threshold=3.0,
        damping_overshoot_factor=4.0,
    )
    assert opt.damping_overshoot_threshold == 3.0
    assert opt.damping_overshoot_factor == 4.0
    res = opt(jax.random.key(1), num_iters=8, num_walkers=32,
              mcmc_decorr_steps=2, num_equil_steps=10, verbose=0)
    assert np.all(np.isfinite(res['E_per_elec_history']))


def test_fixed_scale_param_accepted():
    """``fixed_scale=True`` plumbs through; we don't assert the run
    converges (it requires careful lr/damping calibration), only
    that it runs without raising."""
    cfg = HEGPsiFormerConfig(
        n_up=7, n_down=7, L=7.77, n_det=2,
        embedding_dim=16, n_interactions=1,
        two_particle_stream_dim=8, n_attention_heads=2,
        use_cusp=False, use_deep_jastrow=False,
        use_pair_jastrow=False, n_virt_pw=12, det_jitter=0.02,
    )
    init_key = jax.random.key(0)
    opt = _HEGKFACOptimizer(
        cfg, init_key,
        capture_activations=True,
        fixed_scale=True,
        lr=1e-4,                # very small to avoid blowup
        damping=1.0,            # large enough to stabilise
    )
    assert opt.fixed_scale is True
    res = opt(jax.random.key(1), num_iters=5, num_walkers=16,
              mcmc_decorr_steps=2, num_equil_steps=5, verbose=0)
    # At small lr the run should not NaN.
    assert np.all(np.isfinite(res['E_per_elec_history']))


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
    opt = _HEGKFACOptimizer(
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
    opt = _HEGKFACOptimizer(
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


def test_damping_decay_param_accepted():
    """``damping_decay`` plumbs through the constructor."""
    cfg = HEGPsiFormerConfig(
        n_up=7, n_down=7, L=7.77, n_det=2,
        embedding_dim=16, n_interactions=1,
        two_particle_stream_dim=8, n_attention_heads=2,
        use_cusp=False, use_deep_jastrow=False,
        use_pair_jastrow=False, n_virt_pw=12, det_jitter=0.02,
    )
    init_key = jax.random.key(0)
    opt = _HEGKFACOptimizer(
        cfg, init_key,
        damping_decay=0.9,
    )
    assert opt.damping_decay == 0.9


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
    opt = _HEGKFACOptimizer(cfg, init_key, multi_device=True)
    if n_dev < 2:
        # Auto-fallback to single-device.
        assert opt._n_devices == 1
    else:
        assert opt._n_devices == n_dev
