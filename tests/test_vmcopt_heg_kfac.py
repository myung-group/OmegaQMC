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


def test_kfac_with_multi_device_raises():
    cfg = HEGPsiFormerConfig(
        n_up=7, n_down=7, L=7.77, n_det=2,
        embedding_dim=16, n_interactions=1,
        two_particle_stream_dim=8, n_attention_heads=2,
        use_cusp=False, use_deep_jastrow=False,
        use_pair_jastrow=False, n_virt_pw=12, det_jitter=0.02,
    )
    init_key = jax.random.key(0)
    with pytest.raises(NotImplementedError):
        _HEGKFACOptimizer(cfg, init_key, multi_device=True)
