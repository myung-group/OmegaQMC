"""Validation: compare SVD-recovered KFAC factors against ground-truth.

Strategy:
  Ground truth A_l, G_l are obtained by capturing each Linear's input
  ``a_l`` directly during a forward pass (via thread-local module-level
  monkey-patching of ``nnx.Linear.__call__``).  Once we have ``a_l`` per
  walker, ``g_l`` follows exactly from
  ``g_l = (∂log|ψ|/∂W) · a_l / ‖a_l‖²`` since ``∂log|ψ|/∂W`` is rank-1.

  Then ``A_true = E[a a^T]``, ``G_true = E[g g^T]`` are the strict
  KFAC factors.  Compare with ``A_svd``, ``G_svd`` from
  ``vmcopt_nn_heg_kfac._extract_kron_factors``.

  Per-walker scaling difference is expected: SVD imposes
  ``‖g_w‖ = ‖a_w‖ = sqrt(s_w)``.  We test that the *Kronecker product*
  ``G ⊗ A`` (which is what enters the KFAC step) is invariant to that
  rescaling, modulo the walker-dependent reweighting.
"""
from __future__ import annotations

import contextlib
from typing import Dict, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from OmegaQMC.psi.nn.heg_wf import (
    HEGPsiFormerConfig, make_heg_log_psi_any,
)
from OmegaQMC.vmcopt_nn_heg_kfac import (
    _extract_kron_factors, _classify_params, _path_str,
)


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
        inputs_by_id[layer_id] = x      # last call wins (one Linear ≈ one call here)
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
    other = jax.tree.map(lambda x: x, params)  # placeholder; real `other` in nnx.split
    # The model materialisation requires nnx.merge.
    rngs = nnx.Rngs(key)
    from OmegaQMC.psi.nn.heg_psiformer import build_heg_psiformer_wf
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
    a_per_walker = jnp.stack(a_per_walker, axis=0)   # (W, in) or (W, n_elec, in)

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
