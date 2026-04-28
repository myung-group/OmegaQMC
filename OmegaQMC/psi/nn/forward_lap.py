"""Forward-Laplacian primitives for the NN log-ψ stack.

Each primitive consumes and returns a :class:`VGL` triple
``(value, grad, lap)`` where, given a flat input ``x`` of
shape ``(D,)``:

* ``value`` carries the layer's tensor ``S``.
* ``grad`` carries ``∂value/∂x`` as shape ``(D, *S)``.
* ``lap`` carries the input-trace
  ``Σ_α ∂²value/∂x_α²`` as shape ``S``.

Composition rule (unary elementwise ``f``):

    value' = f(value)
    grad'  = f'(value) ⊙ grad
    lap'   = f''(value) ⊙ Σ_α grad_α²  +  f'(value) ⊙ lap

Slices landed so far:

* slice 1 — convention probe: ``vgl_input``, ``vgl_linear``,
  ``vgl_tanh``, ``vgl_safe_norm``.
* slice 2 — activation registry + elementwise binaries:
  ``vgl_unary`` factory plus ``vgl_silu``, ``vgl_softplus``,
  ``vgl_sigmoid``, ``vgl_ssp``, ``vgl_exp``, ``vgl_log``;
  ``vgl_add``, ``vgl_sub``, ``vgl_mul``.
* slice 3 — non-linear reductions on the trailing axis:
  ``vgl_logsumexp``, ``vgl_softmax``.
* slice 4 — linear reductions + power unaries +
  layer norm: ``vgl_sum``, ``vgl_mean``, ``vgl_sqrt``,
  ``vgl_pow``, ``vgl_layernorm``.
* slice 5 — shape ops + constant ops + edge features:
  ``vgl_log1p``, ``vgl_unsqueeze``, ``vgl_concat``,
  ``vgl_offset``, ``vgl_scale``;
  ``vgl_difference_edge_feature``,
  ``vgl_distance_power_edge_feature``,
  ``vgl_gaussian_edge_feature``.
* slice 6 — Slater identity: ``slogdet_vgl`` (per-det
  ``log|det|`` triple with leading batch axis) and
  ``slogdet_multidet_vgl`` (signed-sum log-aggregation
  via the log-Laplacian identity).
* slice 7 — graph construction: ``vgl_reshape`` (linear
  shape op), ``vgl_constant`` (constant wrapper),
  ``vgl_pairwise_diffs`` and ``vgl_pairwise_self_distance``
  (mirrors of :mod:`OmegaQMC.psi.nn.physics`).
* slice 8 — multi-head attention:
  ``vgl_einsum_bilinear`` (Leibniz rule for an arbitrary
  einsum pattern) and ``vgl_multi_head_attention`` (q/k/v
  linears + scaled dot product + softmax over keys + value
  mixer + output projection).
* slice 9 — exponential envelopes:
  ``vgl_swapaxes`` (negative-axis pair swap),
  ``vgl_einsum_const_lhs`` (linear einsum with a constant
  left-hand tensor) and
  ``vgl_exponential_envelopes_one_spin`` (twin of
  :meth:`OmegaQMC.psi.nn.env.ExponentialEnvelopes._call_one_spin`
  covering the isotropic and Mahalanobis paths, with or
  without per-orbital exponents).
* slice 10 — cusp + backflow:
  ``vgl_sum_axes`` / ``vgl_sum_all`` (multi-axis linear
  reductions), ``vgl_min_along`` (min selection via
  argmin gather), ``vgl_backflow_cutoff`` (polynomial
  ``R²(6-8R+3R²)`` with where-clamp at ``R=1``),
  ``vgl_psiformer_cusp`` and ``vgl_deepqmc_cusp`` (scalar
  cusp twins of :mod:`OmegaQMC.psi.nn.cusp`), and
  ``vgl_backflow_op`` (twin of
  :class:`OmegaQMC.psi.nn.wf.BackflowOp` with the default
  multiplicative / additive activations).
* slice 11 — PsiFormer GNN building blocks:
  ``vgl_mlp`` (MLP block as composition of linears +
  named activations), ``vgl_residual`` (post-block
  residual with optional ``1/sqrt(2)`` normalization),
  and ``vgl_node_attention_update`` (twin of
  :class:`OmegaQMC.psi.nn.gnn.update_features.NodeAttentionElectronUpdateFeature`
  composing :func:`vgl_multi_head_attention` + residual +
  MLP + residual).
* slice 12 — PsiFormer ElectronGNN outer loop:
  ``vgl_ne_diff_vectors`` (ne edge twin matching
  :func:`OmegaQMC.psi.nn.gnn.graph._compute_edges`),
  ``vgl_electron_embedding_positional`` (twin of
  :class:`ElectronEmbedding` in positional mode),
  ``vgl_electron_gnn_layer_psiformer`` (single layer:
  node-attention update + concatenate-rule subnet +
  electron residual), and
  ``vgl_electron_gnn_psiformer`` (positional embedding +
  N layers).  Restricted to the PsiFormer config:
  ``edge_types=null``, ``deep_features=False``,
  single ``node_attention`` update feature, ``concatenate``
  rule.  Other configurations defer to a later slice.

* slice 13 — top-level ``log|ψ|`` builder for PsiFormer:
  ``log_psi_vgl_psiformer`` composes the slice-12 GNN
  with per-spin backflow MLPs, the slice-9 envelopes, the
  slice-10 cusp + backflow op, and the slice-6 Slater
  multi-det aggregation into a scalar
  ``log|ψ|(elec_flat, R) → VGL`` matching the
  PsiFormer-config :class:`NeuralNetworkWaveFunction`.

Remaining primitives (driver wiring of ``log_psi_vgl``
into ``_VMCDriverNN`` / ``_VMCOptDriverNN_SR`` and
config-coverage extensions) follow the same convention
and land in subsequent slices.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp


class VGL(NamedTuple):
    """(value, grad, lap) triple for forward-Laplacian.

    Shape contract — let ``D`` be the input-coord dim and
    ``S`` the layer's tensor shape:

    * ``value``: ``S``
    * ``grad``:  ``(D, *S)``
    * ``lap``:   ``S``
    """

    value: jnp.ndarray
    grad: jnp.ndarray
    lap: jnp.ndarray


def vgl_input(x: jnp.ndarray) -> VGL:
    """Identity at the entry: grad = I_D, lap = 0.

    ``x`` must be 1-D of length ``D``.
    """
    if x.ndim != 1:
        raise ValueError(
            f"vgl_input expects a 1-D array; got shape "
            f"{x.shape}"
        )
    d = x.shape[0]
    return VGL(
        value=x,
        grad=jnp.eye(d, dtype=x.dtype),
        lap=jnp.zeros((d,), dtype=x.dtype),
    )


def vgl_reshape(vin: VGL, shape) -> VGL:
    """Reshape ``value`` (and ``lap``) to ``shape``.

    The leading ``D`` axis of ``grad`` is preserved; the
    remaining axes reshape to match the new value shape.
    Use case: turning the flat input from :func:`vgl_input`
    into a coords tensor of shape ``(n_e, 3)``.
    """
    shape = tuple(shape)
    return VGL(
        value=vin.value.reshape(shape),
        grad=vin.grad.reshape(
            (vin.grad.shape[0],) + shape,
        ),
        lap=vin.lap.reshape(shape),
    )


def vgl_constant(arr, D: int, dtype=None) -> VGL:
    """Wrap a constant array as a VGL with zero grad/lap.

    Used for nuclear coordinates and any other tensor that
    is not a function of the input ``x``.  ``D`` is the
    input-coordinate dimension shared with the variable
    branch.
    """
    arr = jnp.asarray(arr)
    if dtype is not None:
        arr = arr.astype(dtype)
    return VGL(
        value=arr,
        grad=jnp.zeros((D,) + arr.shape, dtype=arr.dtype),
        lap=jnp.zeros(arr.shape, dtype=arr.dtype),
    )


def vgl_linear(vin: VGL, w: jnp.ndarray, b: jnp.ndarray) -> VGL:
    """Affine map ``y = vin.value @ w + b`` on the trailing axis.

    Args:
        vin: VGL triple with feature axis trailing.
            ``vin.value`` shape ``(..., n_in)``.
        w: weight matrix, shape ``(n_in, n_out)``.
        b: bias, shape ``(n_out,)`` (or broadcastable).

    The Jacobian ``w`` is constant in ``x``, so the chain
    rule reduces to a contraction on the trailing axis for
    both ``grad`` and ``lap``.
    """
    value = vin.value @ w + b
    grad = vin.grad @ w
    lap = vin.lap @ w
    return VGL(value=value, grad=grad, lap=lap)


def vgl_unary(
    vin: VGL,
    value: jnp.ndarray,
    fp: jnp.ndarray,
    fpp: jnp.ndarray,
) -> VGL:
    """Elementwise unary chain rule given pre-computed derivs.

    Caller supplies the new ``value = f(vin.value)`` plus
    ``fp = f'(vin.value)`` and ``fpp = f''(vin.value)`` —
    each evaluated elementwise on ``vin.value``.

    Applies::

        grad' = f'(value) ⊙ grad
        lap'  = f''(value) ⊙ Σ_α grad_α²  +  f'(value) ⊙ lap
    """
    grad = fp[None, ...] * vin.grad
    grad_sq = jnp.sum(vin.grad * vin.grad, axis=0)
    lap = fpp * grad_sq + fp * vin.lap
    return VGL(value=value, grad=grad, lap=lap)


def vgl_tanh(vin: VGL) -> VGL:
    """Elementwise ``tanh``.

    ``f' = 1 - tanh²``,  ``f'' = -2·tanh·f'``.
    """
    y = jnp.tanh(vin.value)
    fp = 1.0 - y * y
    fpp = -2.0 * y * fp
    return vgl_unary(vin, y, fp, fpp)


def vgl_sigmoid(vin: VGL) -> VGL:
    """Elementwise sigmoid.

    ``f' = s(1-s)``, ``f'' = s(1-s)(1-2s)``.
    """
    s = jax.nn.sigmoid(vin.value)
    fp = s * (1.0 - s)
    fpp = fp * (1.0 - 2.0 * s)
    return vgl_unary(vin, s, fp, fpp)


def vgl_softplus(vin: VGL) -> VGL:
    """Elementwise ``log(1 + exp(x))``.

    ``f' = sigmoid(x)``, ``f'' = sigmoid(x)·(1-sigmoid(x))``.
    """
    s = jax.nn.sigmoid(vin.value)
    y = jax.nn.softplus(vin.value)
    fp = s
    fpp = s * (1.0 - s)
    return vgl_unary(vin, y, fp, fpp)


def vgl_ssp(vin: VGL) -> VGL:
    """Shifted softplus ``softplus(x) + log(0.5)``.

    Same derivatives as ``softplus``.
    """
    s = jax.nn.sigmoid(vin.value)
    y = jax.nn.softplus(vin.value) + jnp.log(0.5)
    fp = s
    fpp = s * (1.0 - s)
    return vgl_unary(vin, y, fp, fpp)


def vgl_silu(vin: VGL) -> VGL:
    """Elementwise SiLU/swish ``x · sigmoid(x)``.

    With ``s = sigmoid(x)``, ``s' = s(1-s)``, ``s'' =
    s'(1-2s)``::

        f(x)   = x s
        f'(x)  = s + x s'
        f''(x) = 2 s' + x s''
    """
    x = vin.value
    s = jax.nn.sigmoid(x)
    sp = s * (1.0 - s)
    spp = sp * (1.0 - 2.0 * s)
    y = x * s
    fp = s + x * sp
    fpp = 2.0 * sp + x * spp
    return vgl_unary(vin, y, fp, fpp)


def vgl_exp(vin: VGL) -> VGL:
    """Elementwise ``exp``."""
    y = jnp.exp(vin.value)
    return vgl_unary(vin, y, y, y)


def vgl_log(vin: VGL) -> VGL:
    """Elementwise ``log``.

    Caller is responsible for ensuring ``vin.value > 0``.
    """
    inv = 1.0 / vin.value
    return vgl_unary(vin, jnp.log(vin.value), inv, -inv * inv)


# ---------- Binary elementwise ops ----------

def vgl_add(va: VGL, vb: VGL) -> VGL:
    """Elementwise sum.  ``value`` shapes must match (or
    broadcast) on all three components.
    """
    return VGL(
        value=va.value + vb.value,
        grad=va.grad + vb.grad,
        lap=va.lap + vb.lap,
    )


def vgl_sub(va: VGL, vb: VGL) -> VGL:
    """Elementwise difference."""
    return VGL(
        value=va.value - vb.value,
        grad=va.grad - vb.grad,
        lap=va.lap - vb.lap,
    )


def vgl_mul(va: VGL, vb: VGL) -> VGL:
    """Elementwise product (Leibniz rule).

    For ``y = a·b``::

        ∇y  = b·∇a  +  a·∇b
        Δy  = b·Δa  +  2·(∇a · ∇b)_α  +  a·Δb

    where ``(∇a · ∇b)_α`` is the per-element dot over the
    leading input-coord axis.
    """
    value = va.value * vb.value
    grad = (
        va.grad * vb.value[None, ...]
        + va.value[None, ...] * vb.grad
    )
    cross = jnp.sum(va.grad * vb.grad, axis=0)
    lap = va.lap * vb.value + 2.0 * cross + va.value * vb.lap
    return VGL(value=value, grad=grad, lap=lap)


# ---------- Reductions along the trailing axis ----------

def vgl_logsumexp(vin: VGL) -> VGL:
    """``log Σ_k exp(z_k)`` along axis ``-1``.

    Stable via the standard ``z − max(z)`` shift, so input
    magnitudes up to float64 range are safe.

    Math (with ``s = softmax(z)`` and ``L = logsumexp(z)``):

        ∂L/∂x_α  = Σ_k s_k · ∂z_k/∂x_α
        Σ_α ∂²L/∂x_α²
              = Σ_k s_k · Δz_k
              + Σ_α [Σ_k s_k (∂z_k/∂x_α)²  −  (∂L/∂x_α)²]
    """
    z = vin.value
    z_grad = vin.grad
    z_lap = vin.lap

    m = jnp.max(z, axis=-1, keepdims=True)
    e = jnp.exp(z - m)
    z_sum = jnp.sum(e, axis=-1)
    value = jnp.log(z_sum) + jnp.squeeze(m, axis=-1)

    s = e / z_sum[..., None]

    # ∂L/∂x_α  — shape (D, ...)
    grad = jnp.einsum('...k,a...k->a...', s, z_grad)

    weighted_lap = jnp.einsum('...k,...k->...', s, z_lap)
    weighted_grad_sq = jnp.sum(
        s[None, ...] * (z_grad * z_grad), axis=-1,
    )
    lap = weighted_lap + jnp.sum(
        weighted_grad_sq - grad * grad, axis=0,
    )
    return VGL(value=value, grad=grad, lap=lap)


def vgl_softmax(vin: VGL) -> VGL:
    """``softmax`` along axis ``-1``.

    With ``s_k = exp(z_k − L)`` where ``L = logsumexp(z)``,
    and ``Δ_α z_k = ∂z_k/∂x_α − ∂L/∂x_α``,

        ∂s_k/∂x_α  = s_k · Δ_α z_k
        Σ_α ∂²s_k/∂x_α²
                  = s_k · Σ_α (Δ_α z_k)²
                    + s_k · (Δz_k − ΔL)

    Same ``max(z)`` shift as :func:`vgl_logsumexp` for
    numerical stability.
    """
    z = vin.value
    z_grad = vin.grad
    z_lap = vin.lap

    m = jnp.max(z, axis=-1, keepdims=True)
    e = jnp.exp(z - m)
    z_sum = jnp.sum(e, axis=-1, keepdims=True)
    s = e / z_sum

    # ∂L/∂x_α — shape (D, ...)
    dlse = jnp.einsum('...k,a...k->a...', s, z_grad)
    # ΔL — shape (...)
    dlse_grad_sq = jnp.einsum(
        '...k,a...k->a...', s, z_grad * z_grad,
    )
    weighted_lap = jnp.einsum('...k,...k->...', s, z_lap)
    Δlse = weighted_lap + jnp.sum(
        dlse_grad_sq - dlse * dlse, axis=0,
    )

    # ∂s_k/∂x_α  — shape (D, ..., n)
    dz_minus_dlse = z_grad - dlse[..., None]
    grad = s[None, ...] * dz_minus_dlse

    # Σ_α (∂z_k/∂x_α − ∂L/∂x_α)²  — shape (..., n)
    diff_grad_sq = jnp.sum(dz_minus_dlse * dz_minus_dlse, axis=0)
    lap = s * (diff_grad_sq + z_lap - Δlse[..., None])

    return VGL(value=s, grad=grad, lap=lap)


# ---------- Linear reductions on the trailing axis ----------

def vgl_sum(vin: VGL, keepdims: bool = False) -> VGL:
    """Sum along axis ``-1`` (linear; same op on each slot).

    With ``keepdims=True`` the trailing axis is retained as
    size 1 — useful for broadcasting back against the
    pre-reduction tensor.
    """
    return VGL(
        value=jnp.sum(vin.value, axis=-1, keepdims=keepdims),
        grad=jnp.sum(vin.grad, axis=-1, keepdims=keepdims),
        lap=jnp.sum(vin.lap, axis=-1, keepdims=keepdims),
    )


def vgl_mean(vin: VGL, keepdims: bool = False) -> VGL:
    """Mean along axis ``-1`` (linear)."""
    return VGL(
        value=jnp.mean(vin.value, axis=-1, keepdims=keepdims),
        grad=jnp.mean(vin.grad, axis=-1, keepdims=keepdims),
        lap=jnp.mean(vin.lap, axis=-1, keepdims=keepdims),
    )


# ---------- Power-family unaries ----------

def vgl_sqrt(vin: VGL) -> VGL:
    """Elementwise ``sqrt``.  Caller must ensure ``vin > 0``.

    ``f' = 1/(2√v)``, ``f'' = -1/(4 v^{3/2})``.
    """
    y = jnp.sqrt(vin.value)
    fp = 0.5 / y
    fpp = -0.5 * fp / vin.value
    return vgl_unary(vin, y, fp, fpp)


def vgl_pow(vin: VGL, p: float) -> VGL:
    """Elementwise ``v^p``.

    For non-integer ``p`` the caller is responsible for
    ensuring ``vin > 0``.

    ``f'(v) = p · v^{p-1} = p · v^p / v``,
    ``f''(v) = p(p-1) · v^{p-2}``.
    """
    y = jnp.power(vin.value, p)
    inv = 1.0 / vin.value
    fp = p * y * inv
    fpp = (p - 1.0) * fp * inv
    return vgl_unary(vin, y, fp, fpp)


# ---------- LayerNorm (no bias, no scale) ----------

def vgl_layernorm(vin: VGL, eps: float = 1e-6) -> VGL:
    """Layer normalization along axis ``-1`` (no scale, no bias).

    Matches :class:`flax.nnx.LayerNorm` with
    ``use_bias=False, use_scale=False`` and the configured
    ``epsilon`` (default 1e-6, the flax default).

    Computed as ``y = (v − μ) / √(var + eps)`` via composition
    of ``vgl_mean``, ``vgl_sub``, ``vgl_mul``, ``vgl_sqrt``,
    ``vgl_pow(p=-1)`` so that the chain rule is shared with
    the rest of the primitive set.
    """
    mean = vgl_mean(vin, keepdims=True)
    centered = vgl_sub(vin, mean)
    var = vgl_mean(vgl_mul(centered, centered), keepdims=True)
    var_eps = VGL(
        value=var.value + eps, grad=var.grad, lap=var.lap,
    )
    inv_sigma = vgl_pow(vgl_sqrt(var_eps), -1.0)
    return vgl_mul(centered, inv_sigma)


def vgl_log1p(vin: VGL) -> VGL:
    """Elementwise ``log(1 + v)``.  Caller must ensure ``v > -1``.

    ``f' = 1/(1+v)``, ``f'' = -1/(1+v)²``.
    """
    inv = 1.0 / (1.0 + vin.value)
    return vgl_unary(vin, jnp.log1p(vin.value), inv, -inv * inv)


# ---------- Shape ops (no derivative content) ----------

def vgl_unsqueeze(vin: VGL, axis: int = -1) -> VGL:
    """Insert a singleton axis on the value-side.

    Only negative axes are supported so that the same axis
    spec applies to ``value`` / ``lap`` (shape ``S``) and
    ``grad`` (shape ``(D, *S)``) without offset bookkeeping.
    """
    if axis >= 0:
        raise ValueError(
            "vgl_unsqueeze: only negative axes are supported"
        )
    return VGL(
        value=jnp.expand_dims(vin.value, axis),
        grad=jnp.expand_dims(vin.grad, axis),
        lap=jnp.expand_dims(vin.lap, axis),
    )


def vgl_concat(vgls, axis: int = -1) -> VGL:
    """Concatenate a list of VGLs along a value-side axis.

    Only negative axes are supported (same rationale as
    :func:`vgl_unsqueeze`).  All inputs must share leading
    ``D`` and matching shapes on every non-concatenation
    axis.
    """
    if axis >= 0:
        raise ValueError(
            "vgl_concat: only negative axes are supported"
        )
    return VGL(
        value=jnp.concatenate(
            [v.value for v in vgls], axis=axis,
        ),
        grad=jnp.concatenate(
            [v.grad for v in vgls], axis=axis,
        ),
        lap=jnp.concatenate(
            [v.lap for v in vgls], axis=axis,
        ),
    )


# ---------- Constant ops (zero derivative content) ----------

def vgl_offset(vin: VGL, c) -> VGL:
    """Add a constant ``c`` (zero gradient/Laplacian).

    ``c`` may broadcast to a strictly larger shape than
    ``vin.value`` — the returned VGL then carries the
    broadcasted shape on all three components, matching the
    strict shape contract.
    """
    new_value = vin.value + c
    target = new_value.shape
    return VGL(
        value=new_value,
        grad=jnp.broadcast_to(
            vin.grad, (vin.grad.shape[0],) + target,
        ),
        lap=jnp.broadcast_to(vin.lap, target),
    )


def vgl_scale(vin: VGL, c) -> VGL:
    """Multiply by a constant ``c`` (zero gradient/Laplacian).

    Like :func:`vgl_offset`, ``c`` may broadcast to a larger
    shape; the result carries that broadcasted shape.
    """
    new_value = vin.value * c
    target = new_value.shape
    return VGL(
        value=new_value,
        grad=jnp.broadcast_to(
            vin.grad * c, (vin.grad.shape[0],) + target,
        ),
        lap=jnp.broadcast_to(vin.lap * c, target),
    )


def vgl_safe_norm(vin: VGL) -> VGL:
    """``r = sqrt(eps + Σ_K v_K²)`` over the trailing axis.

    Matches :func:`OmegaQMC.psi.nn.utils.norm` with
    ``safe=True``.  Output drops the last axis: if
    ``vin.value`` has shape ``(..., n)`` then the returned
    value has shape ``(...,)``.

    Math (with ``d_hat = v / r``):

        ∂r/∂x_α   = Σ_K d_hat_K · ∂v_K/∂x_α
        ∂²r/∂x_α² = (1/r) [Σ_K (∂v_K/∂x_α)²
                          − (Σ_K d_hat_K ∂v_K/∂x_α)²]
                    + Σ_K d_hat_K · ∂²v_K/∂x_α²

    summed over ``α`` for the input-trace.
    """
    eps = jnp.finfo(vin.value.dtype).eps
    sq = jnp.sum(vin.value * vin.value, axis=-1)
    r = jnp.sqrt(eps + sq)
    d_hat = vin.value / r[..., None]

    # ∇r per input coord: shape (D, ...)
    grad = jnp.einsum('...K,a...K->a...', d_hat, vin.grad)

    # Direct part: Σ_K d_hat_K · ∇²v_K, shape (...)
    lap_direct = jnp.einsum('...K,...K->...', d_hat, vin.lap)

    # Curvature part: per-α gradient projected on d_hat,
    # shape (D, ...); subtract ‖∇v‖² along last axis.
    proj = jnp.einsum('...K,a...K->a...', d_hat, vin.grad)
    grad_sq = jnp.sum(vin.grad * vin.grad, axis=-1)
    lap_curv = jnp.sum(grad_sq - proj * proj, axis=0) / r

    lap = lap_direct + lap_curv
    return VGL(value=r, grad=grad, lap=lap)


# ---------- Edge-feature VGL twins ----------
#
# Mirror the production callables in
# ``OmegaQMC.psi.nn.gnn.edge_features`` exactly.  Each twin
# consumes a VGL of pairwise difference vectors ``d`` of
# shape ``(..., 3)`` and emits the corresponding edge
# features as a VGL whose ``.value`` matches the production
# class's output bit-for-bit.

def _log_rescale_factor(r_vgl: VGL) -> VGL:
    """Build ``log(1+r)/r`` as a VGL on the same input.

    Used by the ``log_rescale`` branch of every edge-feature
    twin.  Returns a VGL of shape matching ``r_vgl``.
    """
    return vgl_mul(vgl_log1p(r_vgl), vgl_pow(r_vgl, -1.0))


def vgl_difference_edge_feature(
    d_vgl: VGL, log_rescale: bool = False,
) -> VGL:
    """VGL twin of ``DifferenceEdgeFeature``.

    ``d_vgl.value`` has shape ``(..., 3)``.  Without
    ``log_rescale`` this is the identity; with it the result
    is ``d * (log(1+r)/r)[..., None]``.
    """
    if not log_rescale:
        return d_vgl
    r = vgl_safe_norm(d_vgl)
    factor_un = vgl_unsqueeze(_log_rescale_factor(r), axis=-1)
    return vgl_mul(d_vgl, factor_un)


def vgl_distance_power_edge_feature(
    d_vgl: VGL,
    powers,
    eps: float = 0.0,
    log_rescale: bool = False,
) -> VGL:
    """VGL twin of ``DistancePowerEdgeFeature``.

    ``powers`` is an iterable of exponents (mirrored from
    the production class).  For ``p > 0`` the component is
    ``r^p``; for ``p < 0`` it is ``1 / (r^|p| + eps)``.
    """
    r = vgl_safe_norm(d_vgl)
    components = []
    for p in list(powers):
        p_f = float(p)
        if p_f > 0.0:
            comp = vgl_pow(r, p_f)
        else:
            base = vgl_pow(r, -p_f)
            shifted = VGL(
                value=base.value + eps,
                grad=base.grad,
                lap=base.lap,
            )
            comp = vgl_pow(shifted, -1.0)
        components.append(vgl_unsqueeze(comp, axis=-1))
    pw = vgl_concat(components, axis=-1)
    if log_rescale:
        factor_un = vgl_unsqueeze(
            _log_rescale_factor(r), axis=-1,
        )
        pw = vgl_mul(pw, factor_un)
    return pw


def vgl_gaussian_edge_feature(
    d_vgl: VGL, mus, sigmas,
) -> VGL:
    """VGL twin of ``GaussianEdgeFeature``.

    ``mus`` and ``sigmas`` are 1-D arrays of length
    ``n_gaussian``.  Computes
    ``exp(-((r[..., None] - mus) / sigmas)²)``.
    """
    r = vgl_safe_norm(d_vgl)
    r_un = vgl_unsqueeze(r, axis=-1)
    diff = vgl_offset(r_un, -mus)
    sq = vgl_mul(diff, diff)
    scaled = vgl_scale(sq, -1.0 / (sigmas * sigmas))
    return vgl_exp(scaled)


def vgl_apply_edge_feature(d_vgl: VGL, spec: dict) -> VGL:
    """Dispatch to a VGL edge-feature twin based on a spec.

    Recognised ``spec['type']`` values:

    * ``'distance_power'`` — keys ``powers``, optional
      ``eps`` (default ``0.0``) and ``log_rescale``
      (default ``False``).  Maps to
      :func:`vgl_distance_power_edge_feature`.
    * ``'difference'`` — optional ``log_rescale``
      (default ``False``).  Maps to
      :func:`vgl_difference_edge_feature`.
    * ``'gaussian'`` — keys ``mus``, ``sigmas``.  Maps to
      :func:`vgl_gaussian_edge_feature`.
    * ``'combined'`` — key ``features`` (list of nested
      specs).  Applies each in turn and concatenates along
      the last axis.

    Used to thread the production ``edge_features`` dict
    through the VGL pipeline without requiring callers to
    name the primitive directly.
    """
    t = spec['type']
    if t == 'distance_power':
        return vgl_distance_power_edge_feature(
            d_vgl, spec['powers'],
            eps=spec.get('eps', 0.0),
            log_rescale=spec.get('log_rescale', False),
        )
    if t == 'difference':
        return vgl_difference_edge_feature(
            d_vgl,
            log_rescale=spec.get('log_rescale', False),
        )
    if t == 'gaussian':
        return vgl_gaussian_edge_feature(
            d_vgl, spec['mus'], spec['sigmas'],
        )
    if t == 'combined':
        feats = [
            vgl_apply_edge_feature(d_vgl, sub)
            for sub in spec['features']
        ]
        return vgl_concat(feats, axis=-1)
    raise ValueError(
        f"unknown edge_feature type: {t!r}",
    )


# ---------- Edge construction (sender, receiver) ----------
#
# Layout matches the production ``_compute_edges``:
#
#     diffs[i_send, j_recv, :] = pos_recv[j_recv]
#                                - pos_sender[i_send]
#
# All three presets (psiformer, ferminet, deeperwin) set
# ``self_interaction=True``; the off-diagonal-only branch
# (``mask_self=True``) is implemented for completeness but
# not yet exercised by adapter-level configs.

def _vgl_offdiag_pairs(diffs_vgl: VGL) -> VGL:
    """Filter the diagonal (sender == receiver) from a square
    pairwise VGL, returning shape
    ``(n - 1, n, *trailing)``."""
    n = diffs_vgl.value.shape[0]
    assert diffs_vgl.value.shape[1] == n
    send_idx = (
        jnp.arange(n)[None, :]
        <= jnp.arange(n - 1)[:, None]
    ) + jnp.arange(n - 1)[:, None]
    recv_idx = jnp.broadcast_to(
        jnp.arange(n)[None], (n - 1, n),
    )
    return VGL(
        value=diffs_vgl.value[send_idx, recv_idx],
        grad=diffs_vgl.grad[:, send_idx, recv_idx],
        lap=diffs_vgl.lap[send_idx, recv_idx],
    )


def vgl_pair_diffs_send_recv(
    sender_vgl: VGL, receiver_vgl: VGL,
) -> VGL:
    """Produce ``recv[j] - send[i]`` as a 3-D pairwise VGL.

    Result has shape ``(n_send, n_recv, 3)`` matching the
    production ``_compute_edges``.
    """
    s_un = vgl_unsqueeze(sender_vgl, axis=-2)
    r_un = vgl_unsqueeze(receiver_vgl, axis=-3)
    return vgl_sub(r_un, s_un)


def vgl_build_same_edges(
    r_vgl: VGL,
    *,
    n_up: int,
    n_down: int,
    edge_feature_spec: dict,
    self_interaction: bool,
) -> tuple:
    """Build ``(uu, dd)`` same-spin edge VGLs.

    Mirrors the production
    :class:`SameGraphEdges` constituent arrays.  Each block
    is the receiver-major pairwise diff between same-spin
    coordinates, transformed by ``edge_feature_spec``.

    Returns:
        ``(uu_vgl, dd_vgl)`` — value shapes ``(n_up, n_up,
        F)`` and ``(n_down, n_down, F)`` if
        ``self_interaction`` else ``(n_up - 1, n_up, F)``
        and ``(n_down - 1, n_down, F)``.
    """
    r_up = _vgl_slice0(r_vgl, slice(None, n_up))
    r_dn = _vgl_slice0(r_vgl, slice(n_up, None))

    def _block(r_block):
        diffs = vgl_pair_diffs_send_recv(r_block, r_block)
        if not self_interaction:
            diffs = _vgl_offdiag_pairs(diffs)
        return vgl_apply_edge_feature(
            diffs, edge_feature_spec,
        )

    return _block(r_up), _block(r_dn)


def vgl_build_anti_edges(
    r_vgl: VGL,
    *,
    n_up: int,
    n_down: int,
    edge_feature_spec: dict,
) -> tuple:
    """Build ``(du, ud)`` anti-spin edge VGLs.

    Mirrors the production :class:`AntiGraphEdges`
    constituent arrays.  ``du`` carries diffs from
    spin-down senders to spin-up receivers (shape
    ``(n_down, n_up, F)``); ``ud`` is the transpose
    (``(n_up, n_down, F)``).
    """
    r_up = _vgl_slice0(r_vgl, slice(None, n_up))
    r_dn = _vgl_slice0(r_vgl, slice(n_up, None))
    du_diff = vgl_pair_diffs_send_recv(r_dn, r_up)
    ud_diff = vgl_pair_diffs_send_recv(r_up, r_dn)
    return (
        vgl_apply_edge_feature(du_diff, edge_feature_spec),
        vgl_apply_edge_feature(ud_diff, edge_feature_spec),
    )


def vgl_same_edges_single_array(
    uu_vgl: VGL, dd_vgl: VGL,
) -> VGL:
    """Concatenate ``uu`` and ``dd`` blocks into the
    flat ``single_array`` layout of
    :class:`SameGraphEdges`.

    Output value shape: ``(n_uu + n_dd, F)`` where
    ``n_uu = uu.shape[0] * uu.shape[1]`` and
    ``n_dd = dd.shape[0] * dd.shape[1]``.
    """
    F = uu_vgl.value.shape[-1]
    uu_flat = vgl_reshape(uu_vgl, (-1, F))
    dd_flat = vgl_reshape(dd_vgl, (-1, F))
    return vgl_concat([uu_flat, dd_flat], axis=-2)


def vgl_anti_edges_single_array(
    du_vgl: VGL, ud_vgl: VGL,
) -> VGL:
    """Concatenate ``du`` and ``ud`` blocks into the
    flat ``single_array`` layout of
    :class:`AntiGraphEdges`."""
    F = du_vgl.value.shape[-1]
    du_flat = vgl_reshape(du_vgl, (-1, F))
    ud_flat = vgl_reshape(ud_vgl, (-1, F))
    return vgl_concat([du_flat, ud_flat], axis=-2)


def vgl_build_up_edges(
    r_vgl: VGL,
    *,
    n_up: int,
    n_down: int,
    edge_feature_spec: dict,
) -> VGL:
    """Build ``'up'`` edges as a single VGL.

    Mirrors the production :class:`UpGraphEdges` constructed
    by :func:`MolecularGraphEdgeBuilder`: senders are the
    spin-up electrons, receivers are all electrons.

    Returns:
        Edge VGL of value shape ``(n_up, n_e, F)``.
    """
    r_up = _vgl_slice0(r_vgl, slice(None, n_up))
    diffs = vgl_pair_diffs_send_recv(r_up, r_vgl)
    return vgl_apply_edge_feature(diffs, edge_feature_spec)


def vgl_build_down_edges(
    r_vgl: VGL,
    *,
    n_up: int,
    n_down: int,
    edge_feature_spec: dict,
) -> VGL:
    """Build ``'down'`` edges as a single VGL.

    Mirrors the production :class:`DownGraphEdges`: senders
    are the spin-down electrons, receivers are all electrons.

    Returns:
        Edge VGL of value shape ``(n_down, n_e, F)``.
    """
    r_dn = _vgl_slice0(r_vgl, slice(n_up, None))
    diffs = vgl_pair_diffs_send_recv(r_dn, r_vgl)
    return vgl_apply_edge_feature(diffs, edge_feature_spec)


def _vgl_sum_along0(vin: VGL, normalize: bool) -> VGL:
    """Reduce ``vin`` along value-axis 0 (sender axis).

    ``normalize=True`` divides by the sender count, matching
    ``jnp.mean``.  Empty senders short-circuit to ``jnp.sum``
    semantics (no division).
    """
    n = vin.value.shape[0]
    neg_axis = -vin.value.ndim
    pooled = vgl_sum_axes(
        vin, axes=(neg_axis,), keepdims=False,
    )
    if normalize and n > 0:
        pooled = vgl_scale(pooled, 1.0 / n)
    return pooled


def vgl_sum_senders_simple(
    edges_vgl: VGL, normalize: bool,
) -> VGL:
    """``sum_senders`` for :class:`SimpleGraphEdges` /
    :class:`UpGraphEdges` / :class:`DownGraphEdges`.

    Reduces the sender axis (value axis 0) of an edge VGL
    of shape ``(n_send, n_recv, F)`` to ``(n_recv, F)``.
    """
    return _vgl_sum_along0(edges_vgl, normalize)


def vgl_sum_senders_same(
    uu_vgl: VGL, dd_vgl: VGL, normalize: bool,
) -> VGL:
    """``sum_senders`` for :class:`SameGraphEdges`.

    Per-block reduction (uu over its senders, dd over its
    senders), each optionally normalized by its own sender
    count, then concatenated along the receiver axis.
    """
    up = _vgl_sum_along0(uu_vgl, normalize)
    dn = _vgl_sum_along0(dd_vgl, normalize)
    return vgl_concat([up, dn], axis=-2)


def vgl_sum_senders_anti(
    du_vgl: VGL, ud_vgl: VGL, normalize: bool,
) -> VGL:
    """``sum_senders`` for :class:`AntiGraphEdges`.

    ``du`` reduces over its (down-spin) senders and lands on
    spin-up receivers; ``ud`` reduces over its (up-spin)
    senders and lands on spin-down receivers.  Concatenated
    along the receiver axis to give shape ``(n_e, F)``.
    """
    up = _vgl_sum_along0(du_vgl, normalize)
    dn = _vgl_sum_along0(ud_vgl, normalize)
    return vgl_concat([up, dn], axis=-2)


def vgl_convolution_update(
    edges_vgl: dict,
    h_vgl: VGL,
    *,
    edge_types,
    n_up: int,
    n_down: int,
    normalize: bool,
    w_specs: dict,
    h_specs: dict,
) -> VGL:
    """VGL twin of
    :class:`OmegaQMC.psi.nn.gnn.update_features.ConvolutionElectronUpdateFeature`.

    For each requested edge type, applies the per-type
    edge-feature MLP ``w_nets[et]`` to the edge VGL,
    transforms the full per-electron embedding by the
    per-type node MLP ``h_nets[et]``, multiplies the
    sender-indexed transformed nodes into the transformed
    edges, then reduces over senders (per the production
    ``GraphEdges.convolve`` semantics for that class).
    Concatenates across edge types along the feature axis.

    Args:
        edges_vgl: dict with the same layout as
            :func:`vgl_edge_sum_update`.
        h_vgl: full per-electron embedding VGL
            ``(n_e, node_dim)``.
        edge_types: edge types to convolve over (subset of
            ``{'same','anti','up','down','ee'}``).
        n_up, n_down: spin-block sizes.
        normalize: per-type sum-vs-mean toggle (matches the
            ``EdgeSumElectronUpdateFeature`` semantics: 'ee'
            divides the *combined* same+anti by ``n_e``,
            other types divide *per block* by the sender
            count).
        w_specs, h_specs: dict keyed by edge type, each
            value a dict ``{'layers': [(W,b),...],
            'activation': name, 'last_linear': bool}``
            forwarded to :func:`vgl_mlp`.

    Returns:
        VGL of value shape
        ``(n_e, len(edge_types) * tp_dim)``.
    """
    n_e = n_up + n_down

    def _w(et, edge_part):
        ws = w_specs[et]
        return vgl_mlp(
            edge_part, ws['layers'],
            activation=ws.get('activation', 'tanh'),
            last_linear=ws.get('last_linear', False),
        )

    def _h(et):
        hs = h_specs[et]
        return vgl_mlp(
            h_vgl, hs['layers'],
            activation=hs.get('activation', 'tanh'),
            last_linear=hs.get('last_linear', False),
        )

    def _conv_simple(et, sender_slice):
        we = _w(et, edges_vgl[et])
        hx = _h(et)
        sender = _vgl_slice0(hx, sender_slice)
        sender_un = vgl_unsqueeze(sender, axis=-2)
        prod = vgl_mul(we, sender_un)
        return vgl_sum_senders_simple(prod, normalize)

    def _conv_same(et, normalize_):
        uu, dd = edges_vgl[et]
        uu_we = _w(et, uu)
        dd_we = _w(et, dd)
        hx = _h(et)
        send_up = vgl_unsqueeze(
            _vgl_slice0(hx, slice(None, n_up)), axis=-2,
        )
        send_dn = vgl_unsqueeze(
            _vgl_slice0(hx, slice(n_up, None)), axis=-2,
        )
        uu_prod = vgl_mul(uu_we, send_up)
        dd_prod = vgl_mul(dd_we, send_dn)
        return vgl_sum_senders_same(
            uu_prod, dd_prod, normalize_,
        )

    def _conv_anti(et, normalize_):
        du, ud = edges_vgl[et]
        du_we = _w(et, du)
        ud_we = _w(et, ud)
        hx = _h(et)
        send_up = vgl_unsqueeze(
            _vgl_slice0(hx, slice(None, n_up)), axis=-2,
        )
        send_dn = vgl_unsqueeze(
            _vgl_slice0(hx, slice(n_up, None)), axis=-2,
        )
        du_prod = vgl_mul(du_we, send_dn)
        ud_prod = vgl_mul(ud_we, send_up)
        return vgl_sum_senders_anti(
            du_prod, ud_prod, normalize_,
        )

    pieces = []
    for et in edge_types:
        if et == 'ee':
            same_conv = _conv_same('same', False)
            anti_conv = _conv_anti('anti', False)
            total = vgl_add(same_conv, anti_conv)
            if normalize:
                total = vgl_scale(total, 1.0 / n_e)
            pieces.append(total)
        elif et == 'same':
            pieces.append(_conv_same(et, normalize))
        elif et == 'anti':
            pieces.append(_conv_anti(et, normalize))
        elif et == 'up':
            pieces.append(
                _conv_simple(et, slice(None, n_up)),
            )
        elif et == 'down':
            pieces.append(
                _conv_simple(et, slice(n_up, None)),
            )
        else:
            raise NotImplementedError(
                f"convolution edge type not supported: "
                f"{et!r}",
            )
    if len(pieces) == 1:
        return pieces[0]
    return vgl_concat(pieces, axis=-1)


def vgl_edge_sum_update(
    edges_vgl: dict,
    *,
    edge_types,
    n_up: int,
    n_down: int,
    normalize: bool,
) -> VGL:
    """VGL twin of
    :class:`OmegaQMC.psi.nn.gnn.update_features.EdgeSumElectronUpdateFeature`.

    For each requested edge type, reduce the corresponding
    edge VGL along the sender axis (per the production
    ``sum_senders`` semantics for that class) and concatenate
    the results along the feature axis.  Output value shape
    ``(n_e, len(edge_types) * F)``.

    The ``edges_vgl`` dict carries one entry per edge type
    available to this layer.  Layout per key:

    * ``'same'``: ``(uu_vgl, dd_vgl)`` tuple with shapes
      ``(n_uu, n_up, F)`` / ``(n_dd, n_down, F)``.
    * ``'anti'``: ``(du_vgl, ud_vgl)`` tuple with shapes
      ``(n_down, n_up, F)`` / ``(n_up, n_down, F)``.
    * ``'up'`` / ``'down'`` / any other simple type: a single
      VGL of shape ``(n_send, n_e, F)``.

    The ``'ee'`` alias matches the production special case
    ``(same.sum_senders(False) + anti.sum_senders(False)) /
    n_e`` (or ``/ 1`` if ``normalize=False``); it requires
    both ``'same'`` and ``'anti'`` to be present in
    ``edges_vgl``.
    """
    n_e = n_up + n_down
    pieces = []
    for et in edge_types:
        if et == 'ee':
            uu, dd = edges_vgl['same']
            du, ud = edges_vgl['anti']
            same_sum = vgl_sum_senders_same(
                uu, dd, normalize=False,
            )
            anti_sum = vgl_sum_senders_anti(
                du, ud, normalize=False,
            )
            total = vgl_add(same_sum, anti_sum)
            if normalize:
                total = vgl_scale(total, 1.0 / n_e)
            pieces.append(total)
        elif et == 'same':
            uu, dd = edges_vgl[et]
            pieces.append(
                vgl_sum_senders_same(uu, dd, normalize),
            )
        elif et == 'anti':
            du, ud = edges_vgl[et]
            pieces.append(
                vgl_sum_senders_anti(du, ud, normalize),
            )
        else:
            pieces.append(
                vgl_sum_senders_simple(
                    edges_vgl[et], normalize,
                ),
            )
    if len(pieces) == 1:
        return pieces[0]
    return vgl_concat(pieces, axis=-1)


# ---------- Deep-features edge update (shared subnet_g) ----------

def _vgl_slice_axis_m2(vin: VGL, start, stop) -> VGL:
    """Slice ``vin`` along value-side axis -2.

    The leading ``D`` axis of ``grad`` is preserved; ``start``
    / ``stop`` index into the value-side axis -2 (which is
    ``grad`` axis -2 as well, since ``grad`` shares the
    trailing value shape).
    """
    return VGL(
        value=vin.value[..., start:stop, :],
        grad=vin.grad[..., start:stop, :],
        lap=vin.lap[..., start:stop, :],
    )


def _vgl_edge_residual_one(
    old_vgl: VGL, new_vgl: VGL, *, normalize: bool,
) -> VGL:
    """Per-leaf residual mirroring :class:`ResidualConnection`.

    Returns ``new_vgl`` when shapes differ (matches the
    production projection-free branch), otherwise the
    optionally-normalised sum.
    """
    if old_vgl.value.shape != new_vgl.value.shape:
        return new_vgl
    summed = vgl_add(old_vgl, new_vgl)
    if normalize:
        return vgl_scale(summed, 1.0 / jnp.sqrt(2.0))
    return summed


def _vgl_apply_tp_residual(old, new, *, normalize: bool):
    """Apply ``ResidualConnection`` over the dict-of-edges
    layout used by :func:`vgl_update_edges_shared`.

    Mirrors :meth:`ResidualConnection.__call__` operating
    leaf-by-leaf on the production
    :class:`SameGraphEdges` / :class:`AntiGraphEdges` /
    :class:`SimpleGraphEdges` pytrees: the ``same`` / ``anti``
    leaves are ``(uu, dd)`` / ``(du, ud)`` pairs of VGLs;
    other types are a single VGL leaf.
    """
    if isinstance(old, VGL):
        return _vgl_edge_residual_one(
            old, new, normalize=normalize,
        )
    return tuple(
        _vgl_edge_residual_one(o, n, normalize=normalize)
        for o, n in zip(old, new)
    )


def vgl_update_edges_shared(
    edges_vgl: dict,
    *,
    edge_types_order,
    subnet_layers,
    subnet_activation='tanh',
    subnet_last_linear: bool = False,
    tp_residual_normalize=None,
) -> dict:
    """VGL twin of :meth:`ElectronGNNLayer._update_edges` with
    ``deep_features='shared'``.

    Mirrors the production op-for-op: stack each edge type's
    ``single_array`` representation, concatenate along axis
    ``-2``, apply a single shared ``subnet_g`` MLP on the
    feature axis, split the result back into per-type chunks,
    reshape into the original structured layout, then
    optionally apply the residual connection.

    Restricted to a homogeneous edge-type list (all keys
    must produce ``single_array`` tensors of matching ``ndim``
    on every axis except ``-2``); the production
    :meth:`_update_edges` carries the same restriction
    implicitly through ``jnp.concatenate(..., axis=-2)``.

    Args:
        edges_vgl: dict layout matching
            :func:`vgl_edge_sum_update`.  ``'same'`` / ``'anti'``
            entries are ``(uu, dd)`` / ``(du, ud)`` tuples;
            other types are single VGLs.
        edge_types_order: ordered list of dict keys to process.
            Must match the iteration order used by
            :meth:`_update_edges` (i.e. ``list(edges.keys())``
            of the production graph).
        subnet_layers, subnet_activation,
            subnet_last_linear: forwarded to :func:`vgl_mlp`
            (the per-layer ``subnet_g``).
        tp_residual_normalize: ``None`` to skip the residual,
            else a bool forwarded to
            :class:`ResidualConnection`.

    Returns:
        Dict with the same structure as ``edges_vgl``,
        carrying the updated edge VGLs.
    """
    if len(edge_types_order) == 0:
        return {}

    arrs = []
    metas = []
    for k in edge_types_order:
        e = edges_vgl[k]
        if k == 'same':
            uu, dd = e
            F = uu.value.shape[-1]
            uu_flat = vgl_reshape(uu, (-1, F))
            dd_flat = vgl_reshape(dd, (-1, F))
            arrs.append(
                vgl_concat([uu_flat, dd_flat], axis=-2),
            )
            metas.append({
                'kind': 'same',
                'uu_shape': uu.value.shape[:-1],
                'dd_shape': dd.value.shape[:-1],
            })
        elif k == 'anti':
            du, ud = e
            F = du.value.shape[-1]
            du_flat = vgl_reshape(du, (-1, F))
            ud_flat = vgl_reshape(ud, (-1, F))
            arrs.append(
                vgl_concat([du_flat, ud_flat], axis=-2),
            )
            metas.append({
                'kind': 'anti',
                'du_shape': du.value.shape[:-1],
                'ud_shape': ud.value.shape[:-1],
            })
        else:
            arrs.append(e)
            metas.append({
                'kind': 'simple',
                'shape': e.value.shape,
            })

    ref_ndim = arrs[0].value.ndim
    for a in arrs[1:]:
        if a.value.ndim != ref_ndim:
            raise NotImplementedError(
                "vgl_update_edges_shared: heterogeneous "
                "single_array ndim across edge types is "
                "not supported (matches production "
                "concatenate(axis=-2) restriction)",
            )

    sizes = [a.value.shape[-2] for a in arrs]
    if len(arrs) == 1:
        combined = arrs[0]
    else:
        combined = vgl_concat(arrs, axis=-2)

    transformed = vgl_mlp(
        combined, subnet_layers,
        activation=subnet_activation,
        last_linear=subnet_last_linear,
    )

    F_new = transformed.value.shape[-1]
    parts = []
    start = 0
    for sz in sizes:
        parts.append(
            _vgl_slice_axis_m2(transformed, start, start + sz),
        )
        start += sz

    new_edges = {}
    for k, part, meta in zip(edge_types_order, parts, metas):
        kind = meta['kind']
        if kind == 'same':
            uu_shape = meta['uu_shape']
            n_uu_flat = uu_shape[-2] * uu_shape[-1]
            uu_part = _vgl_slice_axis_m2(part, 0, n_uu_flat)
            dd_part = _vgl_slice_axis_m2(
                part, n_uu_flat, part.value.shape[-2],
            )
            uu_new = vgl_reshape(
                uu_part, uu_shape + (F_new,),
            )
            dd_new = vgl_reshape(
                dd_part, meta['dd_shape'] + (F_new,),
            )
            new_edges[k] = (uu_new, dd_new)
        elif kind == 'anti':
            du_shape = meta['du_shape']
            n_du_flat = du_shape[-2] * du_shape[-1]
            du_part = _vgl_slice_axis_m2(part, 0, n_du_flat)
            ud_part = _vgl_slice_axis_m2(
                part, n_du_flat, part.value.shape[-2],
            )
            du_new = vgl_reshape(
                du_part, du_shape + (F_new,),
            )
            ud_new = vgl_reshape(
                ud_part, meta['ud_shape'] + (F_new,),
            )
            new_edges[k] = (du_new, ud_new)
        else:
            new_edges[k] = part

    if tp_residual_normalize is not None:
        new_edges = {
            k: _vgl_apply_tp_residual(
                edges_vgl[k], new_edges[k],
                normalize=tp_residual_normalize,
            )
            for k in edge_types_order
        }
    return new_edges


# ---------- Slater-determinant VGL ----------
#
# Differentiating ``log|det S(x)|`` analytically via Jacobi's
# formula avoids unrolling the determinant graph.  For an
# (n, n) matrix function ``S(x)`` and ``A = S⁻¹``,
#
#     ∂_α  log|det S| = trace(A · ∂_α S)
#     ∂²_α log|det S| = trace(A · ∂²_α S) − trace((A · ∂_α S)²)
#
# The input-trace ``Σ_α ∂²_α log|det S|`` is what we ship in
# ``lap``.  Both formulae extend to a leading batch dim
# ``(n_det, n, n)`` directly via ellipsis.

def slogdet_vgl(orb_vgl: VGL):
    """Per-determinant ``log|det S|`` VGL twin.

    Args:
        orb_vgl: VGL of the Slater orbital matrix.  Value
            shape ``(..., n, n)``; ``grad`` shape ``(D,
            ..., n, n)``; ``lap`` shape ``(..., n, n)``.

    Returns:
        ``(sign, log_vgl)`` — ``sign`` of shape ``(...,)``
        carries the determinant phase (zero gradient by
        construction); ``log_vgl`` is a VGL of
        ``log|det S|`` with value shape ``(...,)``.

    Math (with ``Y_α = A · ∂_α S``):

        value  = log|det S|
        grad_α = trace(A · ∂_α S)
        Σ_α ∂²_α value
               = trace(A · ΔS) − Σ_α trace(Y_α · Y_α)

    where ``ΔS = Σ_α ∂²_α S`` is the input-trace of the
    second derivatives carried in ``orb_vgl.lap``.
    """
    S = orb_vgl.value
    dS = orb_vgl.grad
    LS = orb_vgl.lap

    sign, logdet = jnp.linalg.slogdet(S)
    A = jnp.linalg.inv(S)

    # ∂_α log|det S| = Σ_{i,j} A[i,j] · ∂_α S[j,i]
    grad = jnp.einsum('...ij,a...ji->a...', A, dS)

    # Σ_α ∂²_α log|det S| = trace(A · ΔS) − Σ_α trace(Y_α²)
    direct = jnp.einsum('...ij,...ji->...', A, LS)
    Y = jnp.einsum('...ik,a...kj->a...ij', A, dS)
    cross = jnp.einsum('a...ij,a...ji->...', Y, Y)
    lap = direct - cross

    return sign, VGL(value=logdet, grad=grad, lap=lap)


def slogdet_multidet_vgl(
    log_vgl: VGL, signs: jnp.ndarray, coeffs: jnp.ndarray,
) -> VGL:
    """Signed-sum log-aggregation via log-Laplacian identity.

    Combines per-determinant ``log|det|`` triples into a
    single ``log|ψ|`` triple, where

        ψ = Σ_d coeffs[d] · signs[d] · |det_d|

    is the linear configuration sum used by an
    :class:`nnx.Linear` ``conf_coeff`` (signed weights).

    Stable max-shift on ``log|coeffs| + log|det_d|`` keeps
    the sum well-conditioned for any spread of magnitudes.

    Args:
        log_vgl: VGL of per-det ``log|det|``; value shape
            ``(n_det,)``.
        signs:   per-det determinant phase, shape
            ``(n_det,)`` — treated as constant in ``x``.
        coeffs:  configuration coefficients, shape
            ``(n_det,)`` — also constant in ``x``.

    Returns:
        VGL of ``log|ψ|``, value shape ``()``.

    Math (let ``z_d = log|det_d|``, ``w_d = c_d s_d e^{z_d −
    shift} / Σ_e c_e s_e e^{z_e − shift}``, signed weights
    summing to ``±1``):

        ∇ log|ψ|     = Σ_d w_d · ∇z_d
        Σ_α ∇²_α log|ψ|
                     = Σ_d w_d · (Δz_d + Σ_α (∇z_d)_α²)
                       − Σ_α (∇log|ψ|)_α²

    """
    z = log_vgl.value
    z_grad = log_vgl.grad
    z_lap = log_vgl.lap

    log_abs_c = jnp.log(jnp.abs(coeffs))
    phase = jnp.sign(coeffs) * signs
    log_contrib = log_abs_c + z
    shift = jnp.max(log_contrib)
    shifted = phase * jnp.exp(log_contrib - shift)
    sum_val = jnp.sum(shifted)
    log_val = jnp.log(jnp.abs(sum_val)) + shift

    # signed weights summing to ±1
    w = shifted / sum_val

    grad = jnp.einsum('d,ad->a', w, z_grad)
    grad_sq = jnp.sum(z_grad * z_grad, axis=0)
    second = jnp.sum(w * (z_lap + grad_sq))
    lap = second - jnp.sum(grad * grad)

    return VGL(value=log_val, grad=grad, lap=lap)


# ---------- Graph-construction VGL twins ----------
#
# Build the pairwise difference and self-distance tensors
# from the electron coordinates as VGLs.  Mirrors
# :mod:`OmegaQMC.psi.nn.physics`, exactly bit-for-bit on
# ``.value``.

def vgl_pairwise_diffs(c1_vgl: VGL, c2_vgl: VGL) -> VGL:
    """VGL twin of :func:`OmegaQMC.psi.nn.physics.pairwise_diffs`.

    Inputs:
        c1_vgl: value shape ``(..., n1, 3)``.
        c2_vgl: value shape ``(..., n2, 3)``.

    Returns a VGL of shape ``(..., n1, n2, 4)`` where the
    last channel is ``Σ_x (d_x)²``.

    Either argument may be a constant (wrap with
    :func:`vgl_constant`).  The broadcasting unsqueezes are
    safe under the negative-axis convention that
    ``vgl_unsqueeze`` enforces.
    """
    c1_un = vgl_unsqueeze(c1_vgl, axis=-2)   # (..., n1, 1, 3)
    c2_un = vgl_unsqueeze(c2_vgl, axis=-3)   # (..., 1, n2, 3)
    diffs = vgl_sub(c1_un, c2_un)            # (..., n1, n2, 3)
    sq = vgl_sum(
        vgl_mul(diffs, diffs), keepdims=True,
    )                                         # (..., n1, n2, 1)
    return vgl_concat([diffs, sq], axis=-1)


def vgl_pairwise_self_distance(
    coords_vgl: VGL, full: bool = False,
) -> VGL:
    """VGL twin of
    :func:`OmegaQMC.psi.nn.physics.pairwise_self_distance`.

    With ``full=False`` returns the strict upper-triangular
    flat distances of length ``n*(n-1)/2``.  With
    ``full=True`` returns the symmetric ``(n, n)`` matrix
    with **exact zero** on the diagonal — bit-matched to the
    reference even though the diagonal entries are unused
    by every downstream consumer.

    The diagonal-zeroing in the ``full=True`` branch is a
    multiplicative mask, which propagates cleanly through
    ``grad`` and ``lap`` (any large but finite intermediate
    value at ``r → 0`` is multiplied by exactly zero).
    """
    n = coords_vgl.value.shape[-2]
    c1 = vgl_unsqueeze(coords_vgl, axis=-2)   # (..., n, 1, 3)
    c2 = vgl_unsqueeze(coords_vgl, axis=-3)   # (..., 1, n, 3)
    diffs = vgl_sub(c1, c2)                    # (..., n, n, 3)
    r = vgl_safe_norm(diffs)                   # (..., n, n)
    if not full:
        i, j = jnp.triu_indices(n, k=1)
        return VGL(
            value=r.value[..., i, j],
            grad=r.grad[..., i, j],
            lap=r.lap[..., i, j],
        )
    mask = 1.0 - jnp.eye(n, dtype=r.value.dtype)
    return vgl_scale(r, mask)


# ---------- Bilinear einsum (Leibniz rule) ----------

def _free_einsum_label(pattern: str) -> str:
    """Pick a single-letter einsum label that does not occur
    in ``pattern``.  Used to inject the leading ``D``-axis
    label for the chain-rule contractions in
    :func:`vgl_einsum_bilinear`."""
    used = set(pattern)
    for c in 'abcdefghijklmnopqrstuvwxyz':
        if c not in used:
            return c
    raise ValueError(
        f"no free single-letter einsum label available "
        f"in pattern '{pattern}'"
    )


def vgl_einsum_bilinear(
    pattern: str, va: VGL, vb: VGL,
) -> VGL:
    """VGL of an arbitrary bilinear einsum ``Y = einsum(p, A, B)``.

    Applies the chain rule to a bilinear einsum, which is
    the dominant pattern in dot-product attention
    (``ihd,jhd->hij`` for scores and ``hij,jhd->ihd`` for
    the value mixer).

    For ``Y(x) = einsum(p, A(x), B(x))``:

        ∇Y      = einsum(p, ∇A, B) + einsum(p, A, ∇B)
        Σ_α ∂²_α Y
                = einsum(p, ΔA, B) + einsum(p, A, ΔB)
                  + 2·Σ_α einsum(p, ∂_α A, ∂_α B)

    The final cross term is implemented as a single einsum
    that contracts the leading ``D`` axis between the two
    grad tensors.

    Args:
        pattern: einsum string ``"<a_subs>,<b_subs>-><o_subs>"``.
        va, vb: VGLs whose ``.value`` shapes match the
            corresponding subscripts.

    Returns:
        VGL with output shape determined by ``<o_subs>``.
    """
    a_subs, rest = pattern.split(',')
    b_subs, o_subs = rest.split('->')
    x = _free_einsum_label(pattern)

    value = jnp.einsum(pattern, va.value, vb.value)

    grad = (
        jnp.einsum(
            f"{x}{a_subs},{b_subs}->{x}{o_subs}",
            va.grad, vb.value,
        )
        + jnp.einsum(
            f"{a_subs},{x}{b_subs}->{x}{o_subs}",
            va.value, vb.grad,
        )
    )

    direct = (
        jnp.einsum(pattern, va.lap, vb.value)
        + jnp.einsum(pattern, va.value, vb.lap)
    )
    cross = jnp.einsum(
        f"{x}{a_subs},{x}{b_subs}->{o_subs}",
        va.grad, vb.grad,
    )
    lap = direct + 2.0 * cross
    return VGL(value=value, grad=grad, lap=lap)


# ---------- Multi-head dot-product self-attention ----------

def vgl_multi_head_attention(
    q_vgl: VGL, k_vgl: VGL, v_vgl: VGL,
    *, wq: jnp.ndarray, wk: jnp.ndarray,
    wv: jnp.ndarray, wo: jnp.ndarray,
    num_heads: int,
) -> VGL:
    """VGL twin of
    :class:`OmegaQMC.psi.nn.gnn.update_features._MultiHeadAttention`.

    Mirrors the production ``__call__`` operation-for-
    operation, with ``use_bias=False`` linears matching the
    fixed configuration in the production class.  No mask
    support — PsiFormer-style self-attention does not use
    one (``NodeAttentionElectronUpdateFeature`` calls
    ``attention(h, h, h)``).

    Args:
        q_vgl, k_vgl, v_vgl: VGLs of the per-token inputs,
            value shape ``(n_q, emb_dim)`` for ``q`` and
            ``(n_k, emb_dim)`` for ``k`` / ``v``.
        wq, wk, wv, wo: kernel matrices of the four linear
            layers, each of shape ``(emb_dim, emb_dim)``.
        num_heads: must divide ``emb_dim``.

    Returns:
        VGL of the post-attention output with value shape
        ``(n_q, emb_dim)``.
    """
    n_q = q_vgl.value.shape[0]
    n_k = k_vgl.value.shape[0]
    emb_dim = wq.shape[0]
    head_dim = emb_dim // num_heads
    if head_dim * num_heads != emb_dim:
        raise ValueError(
            f"emb_dim ({emb_dim}) is not divisible by "
            f"num_heads ({num_heads})"
        )
    no_bias = jnp.zeros(emb_dim, dtype=q_vgl.value.dtype)

    q = vgl_reshape(
        vgl_linear(q_vgl, wq, no_bias),
        (n_q, num_heads, head_dim),
    )
    k = vgl_reshape(
        vgl_linear(k_vgl, wk, no_bias),
        (n_k, num_heads, head_dim),
    )
    v = vgl_reshape(
        vgl_linear(v_vgl, wv, no_bias),
        (n_k, num_heads, head_dim),
    )

    scale = 1.0 / jnp.sqrt(
        jnp.array(head_dim, dtype=q.value.dtype),
    )
    scores = vgl_scale(
        vgl_einsum_bilinear('ihd,jhd->hij', q, k),
        scale,
    )
    attn = vgl_softmax(scores)
    mixed = vgl_einsum_bilinear('hij,jhd->ihd', attn, v)
    flat = vgl_reshape(mixed, (n_q, num_heads * head_dim))
    return vgl_linear(flat, wo, no_bias)


# ---------- Shape op + constant-side einsum ----------

def vgl_swapaxes(vin: VGL, a: int, b: int) -> VGL:
    """Swap two value-side axes (negative-axis convention).

    The leading ``D`` axis of ``grad`` is preserved; ``a``
    and ``b`` refer to axes on the value-side shape ``S`` and
    apply identically to all three components.
    """
    if a >= 0 or b >= 0:
        raise ValueError(
            "vgl_swapaxes: only negative axes are supported"
        )
    return VGL(
        value=jnp.swapaxes(vin.value, a, b),
        grad=jnp.swapaxes(vin.grad, a, b),
        lap=jnp.swapaxes(vin.lap, a, b),
    )


def vgl_einsum_const_lhs(
    pattern: str, c: jnp.ndarray, vin: VGL,
) -> VGL:
    """``Y = einsum(p, c, V(x))`` with ``c`` constant in ``x``.

    Linear in ``V``; the same einsum applies to ``value``,
    ``grad`` (with a leading ``D`` axis label injected on the
    variable side), and ``lap``.

    Args:
        pattern: einsum string ``"<c_subs>,<v_subs>-><o_subs>"``.
        c: constant tensor matching ``<c_subs>``.
        vin: VGL whose ``.value`` matches ``<v_subs>``.
    """
    c_subs, rest = pattern.split(',')
    v_subs, o_subs = rest.split('->')
    x = _free_einsum_label(pattern)
    return VGL(
        value=jnp.einsum(pattern, c, vin.value),
        grad=jnp.einsum(
            f"{c_subs},{x}{v_subs}->{x}{o_subs}",
            c, vin.grad,
        ),
        lap=jnp.einsum(pattern, c, vin.lap),
    )


# ---------- Exponential envelopes ----------

def vgl_exponential_envelopes_one_spin(
    diffs_vgl: VGL,
    *,
    center_idx: jnp.ndarray,
    zeta: jnp.ndarray,
    pi: jnp.ndarray,
    isotropic: bool,
    per_orbital_exponent: bool,
    softplus_zeta: bool,
    n_det: int,
) -> VGL:
    """VGL twin of
    :meth:`OmegaQMC.psi.nn.env.ExponentialEnvelopes._call_one_spin`.

    Mirrors the production op-for-op:

        d = diffs[..., center_idx, :-1]    # (n_e, n_env, 3)
        # isotropic:
        r        = norm(d, safe=True)      # (n_e, n_env)
        exponent = softplus(zeta) · r      # or |zeta| · r
        # Mahalanobis:
        exponent = norm(
            einsum('...ers,ies->i...er', zeta, d),
            safe=True,
        )
        orbs = (pi · exp(-exponent)).sum(axis=-1)
        return unflatten(orbs, -1, (n_det, -1)).swapaxes(-2, -3)

    The ``softplus_zeta=False`` branch in the production
    class wraps the product in :func:`jnp.abs`; since the
    safe-norm output ``r`` is strictly positive, this is
    equivalent to ``|zeta| · r`` and is implemented as such
    here so that the chain rule has no spurious sign flip.

    Args:
        diffs_vgl: VGL of ``pairwise_diffs(r, R)`` — value
            shape ``(n_e, n_atoms, 4)``.
        center_idx: 1-D index array of length ``n_env``
            mapping envelope slot to nuclear-atom index.
        zeta, pi: production parameter arrays (constants in
            ``x``).  ``pi`` has shape ``(n_orb, n_env)``;
            ``zeta`` has shape ``(n_env,)`` /
            ``(n_orb, n_env)`` for isotropic and
            ``(n_env, 3, 3)`` / ``(n_orb, n_env, 3, 3)`` for
            Mahalanobis, with the leading orb axis present
            iff ``per_orbital_exponent=True``.
        isotropic, per_orbital_exponent, softplus_zeta:
            production flags.
        n_det: number of determinants.

    Returns:
        VGL of the per-spin orbital block, value shape
        ``(n_det, n_e, n_orb_per_det)``.
    """
    d_vgl = VGL(
        value=diffs_vgl.value[..., center_idx, :-1],
        grad=diffs_vgl.grad[..., center_idx, :-1],
        lap=diffs_vgl.lap[..., center_idx, :-1],
    )

    if isotropic:
        r = vgl_safe_norm(d_vgl)            # (n_e, n_env)
        if per_orbital_exponent:
            r = vgl_unsqueeze(r, axis=-2)   # (n_e, 1, n_env)
        if softplus_zeta:
            zeta_eff = jax.nn.softplus(zeta)
        else:
            zeta_eff = jnp.abs(zeta)
        exponent = vgl_scale(r, zeta_eff)
    else:
        if per_orbital_exponent:
            ein = 'oers,ies->ioer'
        else:
            ein = 'ers,ies->ier'
        transformed = vgl_einsum_const_lhs(ein, zeta, d_vgl)
        exponent = vgl_safe_norm(transformed)

    if not per_orbital_exponent:
        exponent = vgl_unsqueeze(exponent, axis=-2)

    neg_exp = vgl_scale(exponent, -1.0)
    exp_vgl = vgl_exp(neg_exp)
    weighted = vgl_scale(exp_vgl, pi)
    orbs = vgl_sum(weighted)                # (n_e, n_orb)

    n_e = orbs.value.shape[0]
    orbs_unflat = vgl_reshape(orbs, (n_e, n_det, -1))
    return vgl_swapaxes(orbs_unflat, -2, -3)


# ---------- Multi-axis linear reductions ----------

def vgl_sum_axes(vin: VGL, axes, keepdims: bool = False) -> VGL:
    """Sum over multiple value-side axes (negative axes only).

    Linear in the input; the same op applies identically to
    ``value``, ``grad`` (with leading ``D`` preserved), and
    ``lap``.
    """
    axes = tuple(axes)
    for ax in axes:
        if ax >= 0:
            raise ValueError(
                "vgl_sum_axes: only negative axes are supported"
            )
    return VGL(
        value=jnp.sum(vin.value, axis=axes, keepdims=keepdims),
        grad=jnp.sum(vin.grad, axis=axes, keepdims=keepdims),
        lap=jnp.sum(vin.lap, axis=axes, keepdims=keepdims),
    )


def vgl_sum_all(vin: VGL) -> VGL:
    """Sum over all value-side axes (full reduction to scalar).

    Used by the cusp scalar reductions.
    """
    grad_axes = tuple(range(1, vin.grad.ndim))
    return VGL(
        value=jnp.sum(vin.value),
        grad=jnp.sum(vin.grad, axis=grad_axes),
        lap=jnp.sum(vin.lap),
    )


# ---------- Min-along-axis (argmin gather) ----------

def vgl_min_along(vin: VGL, axis: int = -1) -> VGL:
    """Minimum along a value-side axis (negative axes only).

    Implemented as a gather at the per-slot ``argmin``, so
    the chain rule reduces to selecting the corresponding
    grad / lap entry.  Non-smooth at degenerate minima; for
    Metropolis-sampled configurations the active argmin is
    locally constant in ``x`` and the formula matches
    reverse-mode AD.
    """
    if axis >= 0:
        raise ValueError(
            "vgl_min_along: only negative axes are supported"
        )
    idx = jnp.argmin(vin.value, axis=axis, keepdims=True)
    value = jnp.take_along_axis(
        vin.value, idx, axis=axis,
    ).squeeze(axis)
    lap = jnp.take_along_axis(
        vin.lap, idx, axis=axis,
    ).squeeze(axis)
    idx_g = jnp.broadcast_to(
        idx, (vin.grad.shape[0],) + idx.shape,
    )
    grad = jnp.take_along_axis(
        vin.grad, idx_g, axis=axis,
    ).squeeze(axis)
    return VGL(value=value, grad=grad, lap=lap)


# ---------- Backflow polynomial cutoff ----------

def vgl_backflow_cutoff(R_vgl: VGL) -> VGL:
    """``where(R<1, R²(6-8R+3R²), 1)`` — :class:`BackflowOp` cutoff.

    The polynomial is C² at ``R=1`` (``f(1)=1``,
    ``f'(1)=0``, ``f''(1)=0``) so the where-clamp matches
    the constant-1 branch in value, gradient, and Laplacian
    at the boundary.

    Math::

        f(R)   = 6R² − 8R³ + 3R⁴
        f'(R)  = 12R(1 − R)²
        f''(R) = 12(1 − R)(1 − 3R)
    """
    R = R_vgl.value
    R_grad = R_vgl.grad
    R_lap = R_vgl.lap

    one_minus_R = 1.0 - R
    f_in = R * R * (6.0 - 8.0 * R + 3.0 * R * R)
    fp_in = 12.0 * R * one_minus_R * one_minus_R
    fpp_in = 12.0 * one_minus_R * (1.0 - 3.0 * R)

    mask = R < 1.0
    value = jnp.where(mask, f_in, jnp.ones_like(R))
    grad = jnp.where(
        mask[None, ...],
        fp_in[None, ...] * R_grad,
        jnp.zeros_like(R_grad),
    )
    R_grad_sq = jnp.sum(R_grad * R_grad, axis=0)
    lap = jnp.where(
        mask,
        fpp_in * R_grad_sq + fp_in * R_lap,
        jnp.zeros_like(R_lap),
    )
    return VGL(value=value, grad=grad, lap=lap)


# ---------- Cusp scalar twins ----------
#
# Both cusp callables accept (scale, alpha, dist) where
# ``scale`` and ``alpha`` are constant in ``x`` and ``dist``
# is a VGL of pairwise distances.  Each returns a scalar VGL
# matching :meth:`DeepQMCCusp.__call__` /
# :meth:`PsiformerCusp.__call__`.

def vgl_psiformer_cusp(
    scale, alpha, dist_vgl: VGL,
) -> VGL:
    """``-Σ scale · α² / (α + dist)`` — :class:`PsiformerCusp` twin."""
    denom = vgl_offset(dist_vgl, alpha)
    inv = vgl_pow(denom, -1.0)
    weighted = vgl_scale(inv, scale * (alpha * alpha))
    return vgl_scale(vgl_sum_all(weighted), -1.0)


def vgl_deepqmc_cusp(
    scale, alpha, dist_vgl: VGL,
) -> VGL:
    """``-Σ scale / (α(1 + α·dist))`` — :class:`DeepQMCCusp` twin."""
    inner = vgl_offset(vgl_scale(dist_vgl, alpha), 1.0)
    inv = vgl_pow(inner, -1.0)
    weighted = vgl_scale(inv, scale / alpha)
    return vgl_scale(vgl_sum_all(weighted), -1.0)


# ---------- Jastrow twin ----------

def vgl_jastrow(
    embeddings: VGL,
    layers,
    *,
    activation,
    last_linear: bool,
    sum_first: bool,
) -> VGL:
    """Scalar VGL twin of :class:`OmegaQMC.psi.nn.omni.Jastrow`.

    ``sum_first=True``  : reduce electrons first, then MLP.
    ``sum_first=False`` : MLP per electron, then reduce.

    The MLP produces a final trailing axis of size 1 which
    is squeezed out, matching ``Jastrow.__call__``'s
    ``squeeze(axis=-1)``.
    """
    if sum_first:
        pooled = vgl_sum_axes(embeddings, axes=(-2,))
        out = vgl_mlp(
            pooled, layers,
            activation=activation,
            last_linear=last_linear,
        )
    else:
        per_e = vgl_mlp(
            embeddings, layers,
            activation=activation,
            last_linear=last_linear,
        )
        out = vgl_sum_axes(per_e, axes=(-2,))
    # ``out.value`` shape is ``(1,)`` — squeeze to scalar.
    return VGL(
        value=out.value[..., 0],
        grad=out.grad[..., 0],
        lap=out.lap[..., 0],
    )


# ---------- BackflowOp twin ----------

def _vgl_default_mult_act(vin: VGL) -> VGL:
    """``1 + 2 tanh(x / 4)`` — default :class:`BackflowOp` mult act."""
    return vgl_offset(
        vgl_scale(vgl_tanh(vgl_scale(vin, 0.25)), 2.0),
        1.0,
    )


def _vgl_default_add_act(vin: VGL) -> VGL:
    """``0.1 tanh(x / 4)`` — default :class:`BackflowOp` add act."""
    return vgl_scale(vgl_tanh(vgl_scale(vin, 0.25)), 0.1)


def vgl_backflow_op(
    xs_vgl: VGL,
    fs_mult_vgl,
    fs_add_vgl,
    dists_nuc_vgl: VGL,
    *,
    with_envelope: bool,
) -> VGL:
    """VGL twin of :meth:`OmegaQMC.psi.nn.wf.BackflowOp.__call__`.

    Mirrors the production op-for-op with the default
    multiplicative (``1 + 2 tanh(x/4)``) and additive
    (``0.1 tanh(x/4)``) activations.

    Args:
        xs_vgl: orbital matrix VGL, value shape
            ``(n_det, n_e, n_orb)``.
        fs_mult_vgl: multiplicative correction VGL of the
            same shape, or ``None`` to skip the
            multiplicative branch.
        fs_add_vgl: additive correction VGL of the same
            shape, or ``None`` to skip the additive branch.
        dists_nuc_vgl: electron-nucleus distance VGL,
            value shape ``(n_e, n_atoms)``.
        with_envelope: if ``True``, scale the additive term
            by ``sqrt(Σ_{n_det, n_orb} xs²)`` (the
            production envelope).
    """
    if with_envelope:
        envel = vgl_sqrt(
            vgl_sum_axes(
                vgl_mul(xs_vgl, xs_vgl),
                axes=(-1, -3),
                keepdims=True,
            ),
        )
    else:
        envel = None

    out = xs_vgl
    if fs_mult_vgl is not None:
        out = vgl_mul(out, _vgl_default_mult_act(fs_mult_vgl))
    if fs_add_vgl is not None:
        R_vgl = vgl_scale(
            vgl_min_along(dists_nuc_vgl, axis=-1),
            1.0 / 0.5,
        )
        cutoff = vgl_backflow_cutoff(R_vgl)
        cutoff = vgl_unsqueeze(
            vgl_unsqueeze(cutoff, axis=-1),
            axis=-3,
        )                                # (1, n_e, 1)
        add = vgl_mul(cutoff, _vgl_default_add_act(fs_add_vgl))
        if envel is not None:
            add = vgl_mul(envel, add)
        out = vgl_add(out, add)
    return out


# ---------- MLP block ----------

_VGL_ACTIVATIONS = {
    'tanh': vgl_tanh,
    'silu': vgl_silu,
    'ssp': vgl_ssp,
    'sigmoid': vgl_sigmoid,
    'softplus': vgl_softplus,
    None: None,
}


def vgl_mlp(
    vin: VGL,
    layers,
    *,
    activation='tanh',
    last_linear: bool = False,
) -> VGL:
    """Composition of ``vgl_linear`` + activation per hidden layer.

    Mirrors :class:`OmegaQMC.psi.nn.layers.MLP.__call__`:
    activation is applied after every linear except the
    final one when ``last_linear=True``.

    Args:
        vin: input VGL with feature axis trailing.
        layers: iterable of ``(w, b)`` pairs.  ``b`` may be
            ``None`` for a bias-less layer; in that case a
            zero bias of the appropriate shape is supplied.
        activation: activation name (key into the registry
            ``_VGL_ACTIVATIONS``) or a callable taking and
            returning a VGL.
        last_linear: if ``True``, no activation on the final
            layer (matches the production flag).
    """
    if isinstance(activation, str) or activation is None:
        act_fn = _VGL_ACTIVATIONS[activation]
    else:
        act_fn = activation

    pairs = list(layers)
    n = len(pairs)
    out = vin
    for i, (w, b) in enumerate(pairs):
        if b is None:
            b = jnp.zeros(
                (w.shape[-1],), dtype=out.value.dtype,
            )
        out = vgl_linear(out, w, b)
        is_last = (i == n - 1)
        if not is_last or not last_linear:
            if act_fn is not None:
                out = act_fn(out)
    return out


def vgl_residual(
    inp: VGL, update: VGL, *, normalize: bool,
) -> VGL:
    """Residual connection — ``(inp + update) / sqrt(2)`` or sum.

    Mirrors :class:`OmegaQMC.psi.nn.layers.ResidualConnection`
    on a single tensor; if shapes match the connection adds,
    optionally normalised, otherwise it returns the update
    unchanged (used by the production class for
    dim-changing layers).
    """
    if inp.value.shape != update.value.shape:
        return update
    summed = vgl_add(inp, update)
    if normalize:
        return vgl_scale(summed, 1.0 / jnp.sqrt(2.0))
    return summed


def vgl_node_attention_update(
    h_vgl: VGL,
    *,
    wq, wk, wv, wo,
    num_heads: int,
    mlp_layers,
    mlp_activation='tanh',
    mlp_last_linear: bool = False,
    attn_residual_normalize=None,
    mlp_residual_normalize=None,
) -> VGL:
    """VGL twin of
    :meth:`OmegaQMC.psi.nn.gnn.update_features.NodeAttentionElectronUpdateFeature.__call__`.

    Mirrors the production op-for-op:

        att = MultiHeadAttention(h, h, h)
        if attn_residual: att = residual(h, att)
        out = MLP(att)
        if mlp_residual: out = residual(att, out)
        return out

    Args:
        h_vgl: input embedding VGL, value shape
            ``(n_elec, emb_dim)``.
        wq, wk, wv, wo: attention kernel matrices.
        num_heads: must divide ``emb_dim``.
        mlp_layers: ``[(w, b), ...]`` for the post-attention
            MLP (see :func:`vgl_mlp`).
        mlp_activation, mlp_last_linear: forwarded to
            :func:`vgl_mlp`.
        attn_residual_normalize, mlp_residual_normalize:
            ``None`` to skip the residual (matching a
            ``None`` value of the production attribute), or
            a bool to apply :func:`vgl_residual` with that
            ``normalize`` flag.
    """
    att = vgl_multi_head_attention(
        h_vgl, h_vgl, h_vgl,
        wq=wq, wk=wk, wv=wv, wo=wo,
        num_heads=num_heads,
    )
    if attn_residual_normalize is not None:
        att = vgl_residual(
            h_vgl, att, normalize=attn_residual_normalize,
        )
    out = vgl_mlp(
        att, mlp_layers,
        activation=mlp_activation,
        last_linear=mlp_last_linear,
    )
    if mlp_residual_normalize is not None:
        out = vgl_residual(
            att, out, normalize=mlp_residual_normalize,
        )
    return out


# ---------- PsiFormer ElectronGNN outer loop ----------

def vgl_ne_diff_vectors(
    r_vgl: VGL, R: jnp.ndarray,
) -> VGL:
    """VGL twin of the ne-edge ``_compute_edges`` call.

    Mirrors
    :func:`OmegaQMC.psi.nn.gnn.graph._compute_edges` for an
    nucleus-electron edge with ``filter_diagonal=False``:

        diffs = r[None, :, :] - R[:, None, :]    # (n_nuc, n_e, 3)

    where ``r = r_vgl.value`` is the variable electron
    coordinate tensor of shape ``(n_e, 3)`` and ``R`` is the
    constant nucleus tensor of shape ``(n_nuc, 3)``.
    """
    if r_vgl.value.ndim != 2 or r_vgl.value.shape[-1] != 3:
        raise ValueError(
            f"vgl_ne_diff_vectors expects r of shape "
            f"(n_e, 3); got {r_vgl.value.shape}"
        )
    R = jnp.asarray(R)
    if R.ndim != 2 or R.shape[-1] != 3:
        raise ValueError(
            f"vgl_ne_diff_vectors expects R of shape "
            f"(n_nuc, 3); got {R.shape}"
        )
    D = r_vgl.grad.shape[0]
    R_vgl = vgl_constant(R, D=D, dtype=r_vgl.value.dtype)
    r_un = vgl_unsqueeze(r_vgl, axis=-3)   # (1, n_e, 3)
    R_un = vgl_unsqueeze(R_vgl, axis=-2)   # (n_nuc, 1, 3)
    return vgl_sub(r_un, R_un)              # (n_nuc, n_e, 3)


def vgl_electron_embedding_positional(
    r_vgl: VGL,
    R: jnp.ndarray,
    *,
    n_up: int,
    n_down: int,
    ne_powers,
    ne_log_rescale: bool,
    use_spin: bool,
    proj_W=None,
) -> VGL:
    """VGL twin of :class:`ElectronEmbedding` in positional mode.

    Builds ne edges via :func:`vgl_ne_diff_vectors`, applies
    :func:`vgl_distance_power_edge_feature` and
    :func:`vgl_difference_edge_feature` in the same order as
    :func:`OmegaQMC.psi.nn.build._make_ne_embedding`, swaps
    the ``(n_nuc, n_e)`` axes, flattens to per-electron
    feature vectors, optionally appends a spin indicator, and
    optionally projects to ``embedding_dim`` via a single
    bias-less linear.

    Args:
        r_vgl: electron-coord VGL of shape ``(n_e, 3)``.
        R: nucleus coordinates ``(n_nuc, 3)``.
        n_up, n_down: spin-block sizes (``n_e = n_up +
            n_down``).
        ne_powers: passed through to the distance-power
            feature.
        ne_log_rescale: passed through to both edge features.
        use_spin: append the ``±1`` spin indicator column.
        proj_W: ``None`` to skip projection, or the bias-less
            kernel of the production ``nnx.Linear`` (shape
            ``(in_dim, embedding_dim)``).

    Returns:
        Per-electron embedding VGL ``(n_e, out_dim)`` where
        ``out_dim`` is either ``embedding_dim`` (if projected)
        or ``ne_feat_dim * n_nuc + (1 if use_spin else 0)``.
    """
    n_e = n_up + n_down
    if r_vgl.value.shape[0] != n_e:
        raise ValueError(
            f"vgl_electron_embedding_positional: r_vgl has "
            f"{r_vgl.value.shape[0]} electrons, expected "
            f"{n_e} (= n_up + n_down)"
        )
    d_vgl = vgl_ne_diff_vectors(r_vgl, R)
    feat_dist = vgl_distance_power_edge_feature(
        d_vgl, ne_powers, log_rescale=ne_log_rescale,
    )
    feat_diff = vgl_difference_edge_feature(
        d_vgl, log_rescale=ne_log_rescale,
    )
    feats = vgl_concat(
        [feat_dist, feat_diff], axis=-1,
    )                                          # (n_nuc, n_e, F)
    feats = vgl_swapaxes(feats, -3, -2)        # (n_e, n_nuc, F)
    feats = vgl_reshape(feats, (n_e, -1))      # (n_e, n_nuc*F)
    if use_spin:
        D = r_vgl.grad.shape[0]
        spins = jnp.concatenate([
            jnp.ones(n_up, dtype=feats.value.dtype),
            -jnp.ones(n_down, dtype=feats.value.dtype),
        ])[:, None]
        spin_const = vgl_constant(
            spins, D=D, dtype=feats.value.dtype,
        )
        feats = vgl_concat([feats, spin_const], axis=-1)
    if proj_W is not None:
        b = jnp.zeros(
            (proj_W.shape[1],), dtype=feats.value.dtype,
        )
        feats = vgl_linear(feats, proj_W, b)
    return feats


def vgl_node_sum_update(
    h_vgl: VGL,
    *,
    n_up: int,
    n_down: int,
    node_types,
    normalize: bool,
) -> VGL:
    """VGL twin of
    :class:`OmegaQMC.psi.nn.gnn.update_features.NodeSumElectronUpdateFeature`.

    For each requested node type (``'up'`` and/or
    ``'down'``), compute the sum (or mean if
    ``normalize=True``) of the electron embeddings in that
    spin block and tile the result across all electrons.
    The per-type results are concatenated along the last
    axis; total output feature width is
    ``len(node_types) * emb_dim``.
    """
    n_e = n_up + n_down
    pieces = []
    for nt in node_types:
        if nt == 'up':
            sub = _vgl_slice0(h_vgl, slice(None, n_up))
            denom = n_up if normalize else 1
        elif nt == 'down':
            sub = _vgl_slice0(h_vgl, slice(n_up, None))
            denom = n_down if normalize else 1
        else:
            raise ValueError(
                f"unsupported node type: {nt!r}",
            )
        pooled = vgl_sum_axes(
            sub, axes=(-2,), keepdims=True,
        )
        if normalize:
            pooled = vgl_scale(pooled, 1.0 / denom)
        pieces.append(VGL(
            value=jnp.broadcast_to(
                pooled.value,
                (n_e,) + pooled.value.shape[1:],
            ),
            grad=jnp.broadcast_to(
                pooled.grad,
                pooled.grad.shape[:1]
                + (n_e,)
                + pooled.grad.shape[2:],
            ),
            lap=jnp.broadcast_to(
                pooled.lap,
                (n_e,) + pooled.lap.shape[1:],
            ),
        ))
    return vgl_concat(pieces, axis=-1)


def vgl_electron_gnn_layer_psiformer(
    h_vgl: VGL,
    *,
    n_up: int,
    n_down: int,
    update_specs,
    subnet_layers,
    subnet_activation='tanh',
    subnet_last_linear: bool = False,
    electron_residual_normalize=None,
    edges_vgl=None,
    deep_features=False,
    last_layer: bool = False,
    edge_types_order=None,
    subnet_g_layers=None,
    subnet_g_activation='tanh',
    subnet_g_last_linear: bool = False,
    tp_residual_normalize=None,
):
    """PsiFormer-shape :class:`ElectronGNNLayer` VGL.

    Composes one or more update features per
    ``update_specs`` (each a dict with a ``'type'`` key
    and the corresponding kwargs), concatenates them along
    the embedding axis, and applies the ``subnet`` MLP plus
    optional residual.  When ``deep_features='shared'`` and
    ``last_layer=False``, also runs the deep-feature edge
    update (:func:`vgl_update_edges_shared`) after the node
    update; the updated edges are returned as the second
    tuple element.  Otherwise ``edges_vgl`` is returned
    unchanged.

    Supported update types:

    * ``'node_attention'`` — kwargs forwarded to
      :func:`vgl_node_attention_update`.
    * ``'residual'``       — pass-through (no kwargs).
    * ``'node_sum'``       — kwargs ``node_types`` (subset
      of ``{'up','down'}``) and ``normalize``;
      forwarded to :func:`vgl_node_sum_update`.
    * ``'edge_sum'``       — kwargs ``edge_types`` and
      ``normalize``; forwarded to
      :func:`vgl_edge_sum_update` and consumes the
      ``edges_vgl`` dict supplied by the outer GNN.
    * ``'convolution'``    — kwargs ``edge_types``,
      ``normalize``, ``w_specs``, ``h_specs``; forwarded
      to :func:`vgl_convolution_update` and consumes the
      ``edges_vgl`` dict.

    Args:
        h_vgl: input embedding VGL ``(n_e, emb_dim)``.
        n_up, n_down: spin-block sizes (used by
            ``node_sum``).
        update_specs: list of update-feature dicts.
        subnet_layers, subnet_activation,
            subnet_last_linear: forwarded to :func:`vgl_mlp`
            (production: ``MLP(uf_total_dim, emb_dim)``).
        electron_residual_normalize: ``None`` to skip the
            residual, or a bool to apply :func:`vgl_residual`.
        edges_vgl: optional dict of edge VGLs keyed by edge
            type (see :func:`vgl_edge_sum_update`); required
            when any ``'edge_sum'`` update is present.
    """
    feats = []
    for spec in update_specs:
        t = spec['type']
        if t == 'node_attention':
            f = vgl_node_attention_update(
                h_vgl,
                wq=spec['attn_wq'],
                wk=spec['attn_wk'],
                wv=spec['attn_wv'],
                wo=spec['attn_wo'],
                num_heads=spec['attn_num_heads'],
                mlp_layers=spec['attn_mlp_layers'],
                mlp_activation=spec.get(
                    'attn_mlp_activation', 'tanh',
                ),
                mlp_last_linear=spec.get(
                    'attn_mlp_last_linear', False,
                ),
                attn_residual_normalize=spec.get(
                    'attn_residual_normalize',
                ),
                mlp_residual_normalize=spec.get(
                    'attn_mlp_residual_normalize',
                ),
            )
        elif t == 'residual':
            f = h_vgl
        elif t == 'node_sum':
            f = vgl_node_sum_update(
                h_vgl,
                n_up=n_up, n_down=n_down,
                node_types=spec['node_types'],
                normalize=spec['normalize'],
            )
        elif t == 'edge_sum':
            if edges_vgl is None:
                raise ValueError(
                    "edge_sum update requires edges_vgl",
                )
            f = vgl_edge_sum_update(
                edges_vgl,
                edge_types=spec['edge_types'],
                n_up=n_up, n_down=n_down,
                normalize=spec['normalize'],
            )
        elif t == 'convolution':
            if edges_vgl is None:
                raise ValueError(
                    "convolution update requires edges_vgl",
                )
            f = vgl_convolution_update(
                edges_vgl, h_vgl,
                edge_types=spec['edge_types'],
                n_up=n_up, n_down=n_down,
                normalize=spec['normalize'],
                w_specs=spec['w_specs'],
                h_specs=spec['h_specs'],
            )
        else:
            raise NotImplementedError(
                f"update feature type not supported by "
                f"VGL path: {t!r}",
            )
        feats.append(f)
    combined = (
        feats[0] if len(feats) == 1
        else vgl_concat(feats, axis=-1)
    )
    updated = vgl_mlp(
        combined, subnet_layers,
        activation=subnet_activation,
        last_linear=subnet_last_linear,
    )
    if electron_residual_normalize is not None:
        updated = vgl_residual(
            h_vgl, updated,
            normalize=electron_residual_normalize,
        )

    if deep_features == 'shared' and not last_layer:
        if edges_vgl is None or edge_types_order is None:
            raise ValueError(
                "deep_features='shared' requires edges_vgl "
                "and edge_types_order",
            )
        if subnet_g_layers is None:
            raise ValueError(
                "deep_features='shared' requires "
                "subnet_g_layers",
            )
        new_edges = vgl_update_edges_shared(
            edges_vgl,
            edge_types_order=edge_types_order,
            subnet_layers=subnet_g_layers,
            subnet_activation=subnet_g_activation,
            subnet_last_linear=subnet_g_last_linear,
            tp_residual_normalize=tp_residual_normalize,
        )
        return updated, new_edges
    return updated


def vgl_electron_gnn_psiformer(
    r_vgl: VGL,
    R: jnp.ndarray,
    *,
    n_up: int,
    n_down: int,
    embedding_kwargs,
    layer_specs,
    edge_features=None,
    self_interaction: bool = True,
    deep_features=False,
) -> VGL:
    """VGL twin of :class:`ElectronGNN` in PsiFormer mode.

    Composes :func:`vgl_electron_embedding_positional` with a
    sequence of :func:`vgl_electron_gnn_layer_psiformer`
    layers, returning the final per-electron embedding VGL.

    When ``edge_features`` is not ``None``, the corresponding
    edge VGLs are constructed once (matching the production
    ``deep_features=False`` mode in which edges stay fixed
    across layers) and threaded through each layer for
    consumption by ``edge_sum`` updates.

    Args:
        r_vgl: electron-coord VGL ``(n_e, 3)``.
        R: nucleus coordinates ``(n_nuc, 3)``.
        n_up, n_down: spin-block sizes.
        embedding_kwargs: forwarded to
            :func:`vgl_electron_embedding_positional` (must
            include ``ne_powers``, ``ne_log_rescale``,
            ``use_spin``, optionally ``proj_W``).
        layer_specs: iterable of dicts, each forwarded as
            ``**spec`` to
            :func:`vgl_electron_gnn_layer_psiformer`.
        edge_features: ``None`` or dict mapping edge type
            (``'same'``, ``'anti'``, ``'up'``, ``'down'``)
            to an edge-feature spec dict accepted by
            :func:`vgl_apply_edge_feature`.
        self_interaction: forwarded to
            :func:`vgl_build_same_edges` for the ``'same'``
            edge type.
    """
    h = vgl_electron_embedding_positional(
        r_vgl, R, n_up=n_up, n_down=n_down,
        **embedding_kwargs,
    )

    edges_vgl = None
    if edge_features:
        edges_vgl = {}
        for et, spec in edge_features.items():
            if et == 'same':
                edges_vgl[et] = vgl_build_same_edges(
                    r_vgl,
                    n_up=n_up, n_down=n_down,
                    edge_feature_spec=spec,
                    self_interaction=self_interaction,
                )
            elif et == 'anti':
                edges_vgl[et] = vgl_build_anti_edges(
                    r_vgl,
                    n_up=n_up, n_down=n_down,
                    edge_feature_spec=spec,
                )
            elif et == 'up':
                edges_vgl[et] = vgl_build_up_edges(
                    r_vgl,
                    n_up=n_up, n_down=n_down,
                    edge_feature_spec=spec,
                )
            elif et == 'down':
                edges_vgl[et] = vgl_build_down_edges(
                    r_vgl,
                    n_up=n_up, n_down=n_down,
                    edge_feature_spec=spec,
                )
            else:
                raise NotImplementedError(
                    f"edge type not supported by VGL "
                    f"path: {et!r}",
                )

    edge_types_order = (
        list(edge_features.keys()) if edge_features else None
    )
    n_layers = len(layer_specs)
    for i, spec in enumerate(layer_specs):
        last = (i == n_layers - 1)
        out = vgl_electron_gnn_layer_psiformer(
            h, n_up=n_up, n_down=n_down,
            edges_vgl=edges_vgl,
            deep_features=deep_features,
            last_layer=last,
            edge_types_order=edge_types_order,
            **spec,
        )
        if isinstance(out, VGL):
            h = out
        else:
            h, edges_vgl = out
    return h


# ---------- Top-level log|ψ| builder (PsiFormer) ----------

def _vgl_slice0(vin: VGL, idx) -> VGL:
    """Slice ``vin`` along value-axis 0 (and grad axis 1)."""
    return VGL(
        value=vin.value[idx],
        grad=vin.grad[:, idx],
        lap=vin.lap[idx],
    )


def log_psi_vgl_psiformer(
    elec_flat: jnp.ndarray,
    R: jnp.ndarray,
    *,
    n_up: int,
    n_down: int,
    n_det: int,
    embedding_kwargs,
    layer_specs,
    bf_up_layers,
    bf_down_layers,
    envelope,
    cusp,
    jastrow=None,
    edge_features=None,
    self_interaction: bool = True,
    deep_features=False,
) -> VGL:
    """VGL twin of
    :meth:`OmegaQMC.psi.nn.wf.NeuralNetworkWaveFunction.__call__`
    restricted to the PsiFormer config.

    Assumed config:

    * ``full_determinant=True``,
    * ``backflow_transform='mult'``,
    * ``conf_coeff=SumPool(1)`` (coefficients = ones),
    * ``omni`` = GNN + per-spin multi-head backflow MLPs,
      no Jastrow,
    * envelope = isotropic, ``per_orbital_exponent=True``,
      ``spin_restricted=False``, ``softplus_zeta=False``,
    * ``cusp_electrons=PsiformerCusp`` (or ``None``);
    * ``cusp_nuclei=None``.

    Args:
        elec_flat: 1-D electron coordinates of shape
            ``(3 * (n_up + n_down),)``.
        R: nucleus coordinates ``(n_nuc, 3)`` — constant in
            ``elec_flat``.
        n_up, n_down, n_det: spin-block and determinant
            counts.
        embedding_kwargs, layer_specs: forwarded to
            :func:`vgl_electron_gnn_psiformer`.
        bf_up_layers, bf_down_layers: ``[(w, b), ...]`` for
            the per-spin backflow MLPs (single ``n_bf=1``
            head; activation is ``None`` and
            ``last_linear=True`` per the production
            ``bf_mlp_*`` config).
        envelope: dict with keys ``center_idx``, ``zetas_up``,
            ``zetas_down``, ``pi_up``, ``pi_down``,
            ``isotropic``, ``per_orbital_exponent``,
            ``softplus_zeta``.  Constants extracted from the
            production :class:`ExponentialEnvelopes`.
        cusp: ``None`` or dict with keys ``same_scale``,
            ``anti_scale``, ``same_alpha``, ``anti_alpha``,
            and an optional ``type`` (``'psiformer'`` or
            ``'deepqmc'``; defaults to ``'psiformer'``).
        jastrow: ``None`` or dict with keys ``layers``,
            ``activation``, ``last_linear``, ``sum_first``
            (forwarded to :func:`vgl_jastrow`).  When
            non-``None``, the Jastrow scalar is added to
            ``log|ψ|`` after the cusp.

    Returns:
        VGL of ``log|ψ|`` (scalar value and lap, ``(D,)`` grad).
    """
    n_e = n_up + n_down
    D = elec_flat.shape[0]
    assert D == 3 * n_e

    r_vgl = vgl_reshape(vgl_input(elec_flat), (n_e, 3))
    R_vgl = vgl_constant(R, D=D, dtype=elec_flat.dtype)

    # 1. Geometry — diffs and distances
    diffs_nuc = vgl_pairwise_diffs(r_vgl, R_vgl)
    # diff_vec: (n_e, n_nuc, 3)
    diff_vec = VGL(
        value=diffs_nuc.value[..., :3],
        grad=diffs_nuc.grad[..., :3],
        lap=diffs_nuc.lap[..., :3],
    )
    dists_nuc = vgl_safe_norm(diff_vec)        # (n_e, n_nuc)
    dists_elec_full = vgl_pairwise_self_distance(
        r_vgl, full=True,
    )                                           # (n_e, n_e)

    # 2. GNN
    h = vgl_electron_gnn_psiformer(
        r_vgl, R, n_up=n_up, n_down=n_down,
        embedding_kwargs=embedding_kwargs,
        layer_specs=layer_specs,
        edge_features=edge_features,
        self_interaction=self_interaction,
        deep_features=deep_features,
    )                                           # (n_e, emb_dim)

    # 3. Backflow MLPs (multi_head with n_bf=1 — squeeze)
    h_up = _vgl_slice0(h, slice(None, n_up))
    h_dn = _vgl_slice0(h, slice(n_up, None))
    bf_up = vgl_mlp(
        h_up, bf_up_layers,
        activation=None, last_linear=True,
    )                                           # (n_up, n_e*n_det)
    bf_dn = vgl_mlp(
        h_dn, bf_down_layers,
        activation=None, last_linear=True,
    )                                           # (n_down, n_e*n_det)
    bf_up = vgl_reshape(bf_up, (n_up, n_det, n_e))
    bf_dn = vgl_reshape(bf_dn, (n_down, n_det, n_e))
    # (n_det, n_up, n_e) and (n_det, n_down, n_e)
    bf_up = vgl_swapaxes(bf_up, -2, -3)
    bf_dn = vgl_swapaxes(bf_dn, -2, -3)

    # 4. Envelope (per spin)
    diffs_up = _vgl_slice0(diffs_nuc, slice(None, n_up))
    diffs_dn = _vgl_slice0(diffs_nuc, slice(n_up, None))
    orb_up_env = vgl_exponential_envelopes_one_spin(
        diffs_up,
        center_idx=envelope['center_idx'],
        zeta=envelope['zetas_up'],
        pi=envelope['pi_up'],
        isotropic=envelope['isotropic'],
        per_orbital_exponent=envelope['per_orbital_exponent'],
        softplus_zeta=envelope['softplus_zeta'],
        n_det=n_det,
    )                                           # (n_det, n_up, n_e)
    orb_dn_env = vgl_exponential_envelopes_one_spin(
        diffs_dn,
        center_idx=envelope['center_idx'],
        zeta=envelope['zetas_down'],
        pi=envelope['pi_down'],
        isotropic=envelope['isotropic'],
        per_orbital_exponent=envelope['per_orbital_exponent'],
        softplus_zeta=envelope['softplus_zeta'],
        n_det=n_det,
    )                                           # (n_det, n_down, n_e)

    # 5. BackflowOp (mult only) — xs = xs_env · mult_act(fs)
    #    BackflowOp.with_envelope=True, but the envel array
    #    is unused on the mult branch, so this reduces to a
    #    plain element-wise product.
    mult_up = _vgl_default_mult_act(bf_up)
    mult_dn = _vgl_default_mult_act(bf_dn)
    orb_up = vgl_mul(orb_up_env, mult_up)
    orb_dn = vgl_mul(orb_dn_env, mult_dn)

    # 6. Slater multi-det
    orb_full = vgl_concat(
        [orb_up, orb_dn], axis=-2,
    )                                           # (n_det, n_e, n_e)
    sign_per_det, log_abs = slogdet_vgl(orb_full)
    coeffs = jnp.ones((n_det,), dtype=elec_flat.dtype)
    log_psi_main = slogdet_multidet_vgl(
        log_abs, sign_per_det, coeffs,
    )                                           # scalar VGL

    # 7. Electron cusp (PsiformerCusp)
    if cusp is None:
        if jastrow is not None:
            jas = vgl_jastrow(
                h, jastrow['layers'],
                activation=jastrow['activation'],
                last_linear=jastrow['last_linear'],
                sum_first=jastrow['sum_first'],
            )
            return vgl_add(log_psi_main, jas)
        return log_psi_main

    same_up = VGL(
        value=dists_elec_full.value[:n_up, :n_up],
        grad=dists_elec_full.grad[:, :n_up, :n_up],
        lap=dists_elec_full.lap[:n_up, :n_up],
    )
    same_dn = VGL(
        value=dists_elec_full.value[n_up:, n_up:],
        grad=dists_elec_full.grad[:, n_up:, n_up:],
        lap=dists_elec_full.lap[n_up:, n_up:],
    )
    iu_up, ju_up = jnp.triu_indices(n_up, k=1)
    iu_dn, ju_dn = jnp.triu_indices(n_down, k=1)
    same_up_flat = VGL(
        value=same_up.value[iu_up, ju_up],
        grad=same_up.grad[:, iu_up, ju_up],
        lap=same_up.lap[iu_up, ju_up],
    )
    same_dn_flat = VGL(
        value=same_dn.value[iu_dn, ju_dn],
        grad=same_dn.grad[:, iu_dn, ju_dn],
        lap=same_dn.lap[iu_dn, ju_dn],
    )
    same_dists = vgl_concat(
        [same_up_flat, same_dn_flat], axis=-1,
    )
    anti_block = VGL(
        value=dists_elec_full.value[:n_up, n_up:],
        grad=dists_elec_full.grad[:, :n_up, n_up:],
        lap=dists_elec_full.lap[:n_up, n_up:],
    )
    anti_dists = vgl_reshape(
        anti_block, (n_up * n_down,),
    )
    cusp_total = vgl_constant(
        jnp.array(0.0, dtype=elec_flat.dtype), D=D,
    )
    cusp_type = cusp.get('type', 'psiformer')
    if cusp_type == 'psiformer':
        cusp_fn = vgl_psiformer_cusp
    elif cusp_type == 'deepqmc':
        cusp_fn = vgl_deepqmc_cusp
    else:
        raise ValueError(
            f"unknown cusp type: {cusp_type!r}",
        )
    if same_dists.value.size > 0:
        cusp_total = vgl_add(
            cusp_total,
            cusp_fn(
                cusp['same_scale'],
                cusp['same_alpha'],
                same_dists,
            ),
        )
    if anti_dists.value.size > 0:
        cusp_total = vgl_add(
            cusp_total,
            cusp_fn(
                cusp['anti_scale'],
                cusp['anti_alpha'],
                anti_dists,
            ),
        )
    out = vgl_add(log_psi_main, cusp_total)
    if jastrow is not None:
        jas = vgl_jastrow(
            h, jastrow['layers'],
            activation=jastrow['activation'],
            last_linear=jastrow['last_linear'],
            sum_first=jastrow['sum_first'],
        )
        out = vgl_add(out, jas)
    return out
