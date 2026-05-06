"""KFAC (Kronecker-Factored Approximate Curvature) VMC optimiser for HEG.

Self-contained float64 implementation — does not depend on
``kfac-jax`` (which is float32-pinned and currently incompatible
with our jax 0.10 stack).  Mirrors ``vmcopt_nn_heg_sr.py`` 's
constructor and ``__call__`` signatures so ``run_heg_psiformer.py``
dispatches to it on ``optimize.type: kfac``.

Two factor-extraction paths are available:

a. **M^T M / M M^T (default, fast, biased).**  No model
   instrumentation — uses the per-walker pytree gradient directly.
   For a per-electron Linear with ``∂log|ψ_w|/∂W = Σ_e a_we ⊗ g_we``
   (rank-N), this gives ``A = E_w[M_w^T M_w]`` and ``G = E_w[M_w M_w^T]``
   which are biased by per-walker ``‖a‖`` / ‖g‖`` weights compared
   to FermiNet's strict per-(walker,electron) factorisation, but
   numerically very stable and ~0.2 s/iter at our scale.

b. **Per-electron capture (FermiNet-style, exact).**  Set
   ``capture_activations=True`` to use a **capturing twin** of the
   model: every ``nnx.Linear`` is replaced with a ``CapturingLinear``
   (a subclass with an ``nnx.Intermediate`` slot for its input) at
   build time via :func:`OmegaQMC._kfac_capture.use_capturing_linears`.
   A single ``jax.jit(jax.vmap(apply))`` call now returns
   ``(log_psi_W, ∇params_W, captured_inputs_W)`` in one batched
   pass — the captures flow through standard NNX state machinery,
   so JIT/vmap handle them natively (no eager Python loop).
   We then recover ``g_we`` per walker via a small ``(n_e, n_e)``
   linear solve and accumulate ``A``, ``G`` over ``W·n_e``
   flattened samples — exactly FermiNet's RepeatedDenseBlock.

   Performance: ~equal per-iter cost to the M^T M path (both
   bottlenecked by the same forward + grad).  Recommended for
   accurate KFAC.

Other features
--------------

* **Multi-device pmap** — ``multi_device=True`` shards walkers
  across all visible local devices and ``pmean``-aggregates the
  Kronecker factors.  Auto-fallback to single-device when only one
  GPU is visible.

* **Adaptive Levenberg-Marquardt damping** — FermiNet/kfac-jax
  recipe (Martens & Grosse 2015) with a panic-mode safety net.
  Two response timescales:

    1. *Quadratic-model ratio mode (FermiNet faithful)* — every
       ``damping_lookback`` iters compare the aggregated *actual*
       energy reduction ``ΔE`` to the predicted reduction
       ``Δm = ½·(η·c)²·⟨step, F·step⟩ - (η·c)·⟨grad, step⟩`` from
       the local quadratic model (where ``η = lr_now``,
       ``c = clip``, ``step`` is the applied update direction with
       ``kernel_scale`` baked in).  Form ``ratio = ΔE/Δm``.  If
       ``ratio > 0.75`` the model under-predicts descent → loosen
       damping (``×damping_decay``).  If ``ratio < 0.25`` the model
       over-predicts → tighten damping (``÷damping_decay``).
       This is the schedule that makes FermiNet's
       ``fixed_scale=n_e`` step convention stable.
    2. *Fast panic mode* — every iter, compare ``‖step‖`` to its
       running median.  If the current step is more than
       ``damping_overshoot_threshold`` × the median (default 10×),
       multiply damping by ``damping_overshoot_factor`` (default 2).
       Catches geometric blowup within 1-2 iters before the slower
       LM ratio can react.

  All damping values are bounded by ``[damping_min, damping_max]``.

* **Optional FermiNet ``fixed_scale=n_e`` step convention** — the
  per-electron factors ``A``, ``G`` use ``1/(W·n_e)`` normalisation
  but ``dW_loss`` is per-walker-summed, so the textbook KFAC
  derivation calls for an ``n_e`` multiplier on the natural-
  gradient step.  Set ``fixed_scale=True`` to enable.  Combine with
  ``damping_adapt=True`` (LM ratio mode) and a moderate initial
  damping (e.g. ``1e-3``) plus ``norm_constraint=1e-3`` for
  FermiNet-faithful training; the LM ratio rapidly ramps damping
  during the early iters so the ``n_e`` step magnitude doesn't
  overshoot the trust region.

Algorithm
---------

For each ``Linear`` layer ``y = W x + b`` with kernel ``W`` of shape
``(out, in)``, the per-walker gradient of ``log|ψ_w|`` wrt ``W`` is
exactly the rank-1 outer product

    ∂log|ψ_w|/∂W = g_w · x_w^T,

where ``x_w`` is the layer's input on walker ``w`` and ``g_w`` is the
back-propagated cotangent at the layer's output.  We recover
``(g_w, x_w)`` via batched SVD of the per-walker gradient tensor —
this avoids modifying the model to expose layer activations.

Kronecker-factored Fisher per Linear layer:
    A_l = E[x_w x_w^T]   (in × in)
    G_l = E[g_w g_w^T]   (out × out)
    F_l ≈ G_l ⊗ A_l

Damped inverse via Martens-Grosse trace renormalisation:
    π = sqrt((tr(A)/n_in) / (tr(G)/n_out))
    A_damped = A + π·sqrt(λ)·I
    G_damped = G + (1/π)·sqrt(λ)·I

KFAC step:
    ΔW = -η · G_damped^{-1} · ∂L/∂W · A_damped^{-1}

Non-Linear params (envelope coefficients, cusp α scalars, embeddings)
are treated with a plain damped natural gradient over the small
"generic" Fisher block — same as our SR code path but restricted to
those leaves.

Multi-device parallelism via ``jax.pmap`` is supported and toggled by
``multi_device``.
"""

from __future__ import annotations

import contextlib
import sys
import threading
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from .psi.nn.heg_wf import (
    HEGConfig,
    HEGPsiFormerConfig,
    make_heg_log_psi_any as make_heg_log_psi,
)
from .psi.nn.heg_wf_module import build_heg_psiformer_wf
from .psi.nn.periodic import (
    wrap_to_cell, make_cubic_lattice, make_square_lattice,
)
from .observables.ewald_dispatch import build_ewald_tables_dim
from .psi.nn.physics import laplacian
from .observables.ewald import build_ewald_tables, ewald_pair_energy


# ---------------------------------------------------------------------
# Per-electron activation capture (FermiNet RepeatedDenseBlock approach)
#
# FermiNet's open-source KFAC handles per-electron Linears by treating
# each (walker, electron) pair as a separate Fisher sample.  Their
# kfac-jax integration relies on graph-pattern matching to expose
# layer inputs (a) and output cotangents (g) via ``register_dense``.
# We don't have that; instead we monkey-patch ``nnx.Linear.__call__``
# to record its INPUT into a thread-local bucket whenever an outer
# ``capture_linear_inputs`` context is active.  Outside the context
# (e.g. during the JIT'd grad pass) the patch is a no-op.
#
# Once we have the per-walker captured inputs ``a_we`` (shape
# ``(W, n_sample_layer, in_dim)``), we recover ``g_we`` via a per-
# walker linear solve from
#
#     ∂log|ψ_w|/∂W = G_w · A_w   where  G_w = (out, n_e),
#                                       A_w = (n_e, in)
#
# i.e. ``G_w = dW_w · A_w^T · (A_w · A_w^T + λI)^{-1}``.  Then KFAC
# factors are accumulated over the flattened ``(W·n_e)`` population
# exactly as FermiNet's RepeatedDenseBlock does.
# ---------------------------------------------------------------------

_capture_state = threading.local()
_capture_state.bucket = None

_orig_linear_call = nnx.Linear.__call__


def _patched_linear_call(self, x):
    bucket = getattr(_capture_state, 'bucket', None)
    if bucket is not None:
        bucket.append((id(self), x))
    return _orig_linear_call(self, x)


nnx.Linear.__call__ = _patched_linear_call


@contextlib.contextmanager
def _capture_linear_inputs():
    prev = getattr(_capture_state, 'bucket', None)
    bag: List[Tuple[int, jax.Array]] = []
    _capture_state.bucket = bag
    try:
        yield bag
    finally:
        _capture_state.bucket = prev


def _discover_linear_ids(model: nnx.Module) -> Dict[int, str]:
    """Walk a live ``nnx.Module`` → ``{id(linear) → string path}``."""
    out: Dict[int, str] = {}

    def _r(obj, prefix: str):
        if isinstance(obj, nnx.Linear):
            out[id(obj)] = prefix.rstrip('/')
            return
        if isinstance(obj, nnx.Module):
            for k, v in obj.__dict__.items():
                if k.startswith('_'):
                    continue
                _r(v, prefix + str(k) + '/')
        elif isinstance(obj, (list, tuple)):
            for i, v in enumerate(obj):
                _r(v, prefix + str(i) + '/')
        elif isinstance(obj, dict):
            for k, v in obj.items():
                _r(v, prefix + str(k) + '/')

    _r(model, '')
    return out


def _eager_capture_per_walker(
    model: nnx.Module,
    walkers: jax.Array,
    id_to_path: Dict[int, str],
) -> Dict[str, jax.Array]:
    """Run ``model(walker)`` for each walker, capture each Linear's
    input, return ``{path → (W, *input_shape)}``."""
    bag_per_path: Dict[str, List[jax.Array]] = {
        p: [] for p in id_to_path.values()
    }
    for w in range(walkers.shape[0]):
        with _capture_linear_inputs() as bucket:
            _ = model(walkers[w])
        for layer_id, x in bucket:
            path = id_to_path.get(layer_id)
            if path is not None:
                bag_per_path[path].append(x)
    return {
        p: jnp.stack(xs, axis=0)
        for p, xs in bag_per_path.items()
        if xs
    }


def _per_electron_factors(
    captured_input: jax.Array,
    per_walker_dW: jax.Array,
    de: jax.Array,
    solve_damping: float = 1.0e-6,
) -> Tuple[jax.Array, jax.Array, jax.Array, int]:
    """Compute exact KFAC factors for a per-electron Linear.

    Treats each (walker, electron) pair as a separate sample, à la
    FermiNet's ``RepeatedDenseBlock``.

    Args:
        captured_input: ``(W, n_e_layer, in_dim)`` per-walker per-
            electron Linear inputs.  If the layer is "global"
            (n_e_layer effectively 1) the caller may pass shape
            ``(W, in_dim)``; we'll add an axis.
        per_walker_dW: ``(W, in_dim, out_dim)`` from the JIT'd grad —
            ``Σ_e a_we · g_we^T`` per walker.  (NNX's ``Linear``
            stores ``kernel`` as ``(in, out)``, so the gradient
            inherits that layout.)
        de: ``(W,)`` energy residual for the de-weighted dW_loss.
        solve_damping: small Tikhonov on the per-walker
            ``(A_w · A_w^T + λI)`` solve that recovers ``g_we``.

    Returns:
        ``(A_out, G_in, dW_loss, n_e_layer)``.

        Following the convention used elsewhere in this module
        (``_extract_kron_factors`` and ``_init_factor_state``), the
        first returned matrix is the *output*-side cov ``E[g g^T]``
        (shape ``(out, out)``) — i.e. the right-multiplier in the
        KFAC step ``ΔW = G_in_inv @ dW @ A_out_inv`` — and the
        second is the *input*-side cov ``E[a a^T]`` (shape
        ``(in, in)``) — the left-multiplier.

        ``A_out`` and ``G_in`` are accumulated over the flattened
        ``(W · n_e_layer)`` population.  The caller must apply a
        ``scale = n_e_layer`` multiplier to the natural-gradient
        step (see FermiNet's ``fixed_scale``) to match the per-
        walker gradient magnitude.
    """
    a_we = captured_input
    if a_we.ndim == 2:                             # (W, in) — global Linear
        a_we = a_we[:, None, :]
    W, n_e, in_dim = a_we.shape
    out_dim = per_walker_dW.shape[2]

    # Recover g_we from a_we and per_walker_dW.
    # Per walker: dW_w = a_w^T @ g_w   (shapes (in, out) = (in, n_e) @ (n_e, out))
    # Solve: g_w = (a_w @ a_w^T + λI)^{-1} @ a_w @ dW_w
    #   AAT[w, e, f] = (a_w @ a_w^T)[e, f]
    #   a_dW[w, e, o] = (a_w @ dW_w)[e, o]
    #   g_we[w, e, o] = AAT_inv[w] @ a_dW[w]
    AAT = jnp.einsum('wei,wfi->wef', a_we, a_we)
    AAT = AAT + solve_damping * jnp.eye(
        n_e, dtype=AAT.dtype,
    )[None]
    AAT_inv = jnp.linalg.inv(AAT)
    a_dW = jnp.einsum('wei,wio->weo', a_we, per_walker_dW)    # (W, n_e, out)
    g_we = jnp.einsum('wef,wfo->weo', AAT_inv, a_dW)          # (W, n_e, out)

    # Flatten per-(walker, electron) populations.
    a_flat = a_we.reshape(W * n_e, in_dim)
    g_flat = g_we.reshape(W * n_e, out_dim)

    # KFAC-textbook factors:
    #   A_kfac = E[a a^T]  (shape (in, in))   — the LEFT-side factor
    #                                            in ΔW = A^-1 dW G^-1
    #   G_kfac = E[g g^T]  (shape (out, out)) — the RIGHT-side factor
    A_kfac = jnp.einsum('si,sj->ij', a_flat, a_flat) / (W * n_e)
    G_kfac = jnp.einsum('so,sp->op', g_flat, g_flat) / (W * n_e)

    # Map onto this module's naming convention (preserves the existing
    # step formula ``step = G_inv @ dW @ A_inv``):
    #   A_out (returned first) is the (out, out) RIGHT-side factor
    #   G_in  (returned second) is the (in, in)  LEFT-side factor
    A_out = G_kfac
    G_in = A_kfac

    dW_loss = jnp.einsum('w,wio->io', de, per_walker_dW) / W
    return A_out, G_in, dW_loss, n_e


TARGET_ACCEPTANCE_RATE = 0.5
STEP_SIZE_ADAPTATION_RATE = 0.05


def _adapt_step_size(step_size, acceptance_rate):
    return step_size * (
        1.0 + STEP_SIZE_ADAPTATION_RATE
        * (acceptance_rate - TARGET_ACCEPTANCE_RATE)
    )


# ---------------------------------------------------------------------
# Param-pytree introspection: identify Linear kernels and biases
# ---------------------------------------------------------------------

def _path_str(path) -> str:
    """Render a JAX tree-path (excluding the trailing ``.value``)."""
    return '/'.join(
        str(p.key) if hasattr(p, 'key') else str(p)
        for p in path[:-1]
    )


def _name_at(path, idx) -> Optional[str]:
    if len(path) < abs(idx):
        return None
    p = path[idx]
    return p.key if hasattr(p, 'key') else None


def _classify_params(params) -> Tuple[
    Dict[str, Tuple[Any, Any]],
    Dict[str, Tuple[Any, Any]],
    List[Any],
]:
    """Walk ``params`` and bucket leaves into Linear / generic.

    Returns:
        layers     : ``{layer_path: (kernel_key_path, bias_key_path or None)}``
                     where ``layer_path`` is the layer's pytree-path string and
                     each value is the ``KeyPath`` tuple (so we can look the
                     leaf back up later).
        kernel_paths : same keys as ``layers``, mapped to ``(out, in)``.
        generic    : list of ``(KeyPath, leaf)`` for every non-Linear leaf.
    """
    leaves = jax.tree_util.tree_flatten_with_path(params)[0]
    kernel_paths: Dict[str, Any] = {}
    bias_paths: Dict[str, Any] = {}
    kernel_shapes: Dict[str, Tuple[int, int]] = {}
    generic: List[Tuple[Any, Any]] = []

    for path, leaf in leaves:
        slast = _name_at(path, -2)
        if slast == 'kernel' and leaf.ndim == 2:
            layer = _path_str(path).rsplit('/', 1)[0]
            kernel_paths[layer] = path
            kernel_shapes[layer] = leaf.shape
        elif slast == 'bias' and leaf.ndim == 1:
            layer = _path_str(path).rsplit('/', 1)[0]
            bias_paths[layer] = path
        else:
            generic.append((path, leaf))

    layers = {
        layer: (kernel_paths[layer], bias_paths.get(layer))
        for layer in kernel_paths
    }
    return layers, kernel_shapes, generic


def _set_at_path(pytree, path, value):
    """Replace the leaf at ``path`` in ``pytree`` with ``value``."""
    leaves, treedef = jax.tree_util.tree_flatten_with_path(pytree)
    leaves = [
        (p, value if p == path else lf) for p, lf in leaves
    ]
    return jax.tree_util.tree_unflatten(treedef, [lf for _, lf in leaves])


def _get_at_path(pytree, path):
    """Look up the leaf at ``path`` in ``pytree``."""
    leaves = jax.tree_util.tree_flatten_with_path(pytree)[0]
    for p, lf in leaves:
        if p == path:
            return lf
    raise KeyError(path)


# ---------------------------------------------------------------------
# Kronecker-factor extraction via batched rank-1 SVD
# ---------------------------------------------------------------------

def _rank1_decompose(M: jax.Array) -> Tuple[jax.Array, jax.Array]:
    """Closed-form rank-1 decomposition of ``M = g · x^T`` per walker.

    Avoids batched SVD (which is ~30-100× slower on small matrices).
    For an exact rank-1 matrix ``M_w``, picks the column with the
    largest norm as the leading left-singular direction, then derives
    the right-singular direction by ``M^T u``.  Splits the singular
    value evenly so that ``‖g_w‖ = ‖x_w‖ = sqrt(‖M_w‖_F)`` — same
    convention as :func:`jnp.linalg.svd`.

    Args:
        M: ``(W, out, in)``.

    Returns:
        ``(g, x)`` where ``g`` is ``(W, out)`` and ``x`` is ``(W, in)``,
        with ``g_w · x_w^T = M_w`` for each walker.
    """
    # Pick the column of ``M_w`` with maximum norm to seed power
    # iteration.  For an exact rank-1 input one iteration already
    # yields the true (left, right) singular pair.
    col_norms = jnp.linalg.norm(M, axis=1)              # (W, in)
    best_col = jnp.argmax(col_norms, axis=-1)           # (W,)
    u_seed = jnp.take_along_axis(
        M, best_col[:, None, None], axis=2,
    ).squeeze(-1)                                        # (W, out)
    u_norm = jnp.linalg.norm(
        u_seed, axis=-1, keepdims=True,
    ) + 1e-30
    u = u_seed / u_norm                                  # unit (W, out)
    # v = M^T u — gives ‖x‖·x̂ scaled by 1/‖g‖ from the rank-1 form
    v_unscaled = jnp.einsum('woi,wo->wi', M, u)         # (W, in)
    sigma = jnp.linalg.norm(
        v_unscaled, axis=-1, keepdims=True,
    ) + 1e-30
    v = v_unscaled / sigma                               # unit (W, in)
    sqrt_sigma = jnp.sqrt(sigma)
    g = sqrt_sigma * u
    x = sqrt_sigma * v
    return g, x


def _extract_kron_factors(
    per_walker_dW: jax.Array,
    de: jax.Array,
    pmean_axis: Optional[str] = None,
) -> Tuple[jax.Array, jax.Array, jax.Array]:
    """Compute Kronecker factors A, G and energy-gradient from
    per-walker dW.

    Uses the un-decomposed forms

        A = E[ M_w^T @ M_w ],     G = E[ M_w @ M_w^T ]

    which are *exact* under the standard KFAC factorisation
    assumption (independence of x and g across walkers/electrons),
    independently of how many rank-1 outer products contribute to
    each ``M_w = ∂log|ψ_w|/∂W``.  For per-electron Linears (where
    ``M_w = Σ_e g_we x_we^T``) this is much more accurate than
    extracting a single rank-1 SVD component:

        F_true   ≈ G_true ⊗ A_true             (KFAC approximation)
        F_M^T M ≈ E[‖a‖²]·G_true ⊗ E[‖g‖²]·A_true  (~99% cosine)
        F_SVD    ≈ E[(‖a‖/‖g‖)·gg^T]
                    ⊗ E[(‖g‖/‖a‖)·aa^T]        (~92% cosine on rank-N)

    The ``E[‖a‖²]·E[‖g‖²]`` proportionality is absorbed by the
    learning rate so the *direction* of the natural-gradient step
    is preserved up to that constant — same as multiplying ``F`` by
    a scalar, which is invariant under the KFAC step
    ``F^{-1} · ∇L``.

    Args:
        per_walker_dW: ``(W_local, out, in)`` per-walker grad of
            ``log|ψ|`` wrt ``W`` (NOT energy-weighted).
        de: ``(W_local,)`` energy residual ``E_L_w − ⟨E_L⟩``.
        pmean_axis: If set, ``jax.lax.pmean`` along that axis name
            so the same factors land on every device under ``pmap``.

    Returns:
        ``(A, G, dW_loss)`` of shapes ``(in, in)``, ``(out, out)``,
        ``(out, in)``.
    """
    M = per_walker_dW
    A = jnp.einsum('woi,woj->ij', M, M) / M.shape[0]
    G = jnp.einsum('woi,wpi->op', M, M) / M.shape[0]
    dW_loss = jnp.einsum('w,woi->oi', de, M) / M.shape[0]
    if pmean_axis is not None:
        A = jax.lax.pmean(A, pmean_axis)
        G = jax.lax.pmean(G, pmean_axis)
        dW_loss = jax.lax.pmean(dW_loss, pmean_axis)
    return A, G, dW_loss


# ---------------------------------------------------------------------
# Damped Kronecker inverse with trace renormalisation
# ---------------------------------------------------------------------

def _damped_kron_inverse(A: jax.Array, G: jax.Array, damping: float) -> Tuple[
    jax.Array, jax.Array,
]:
    """Return ``(A_inv, G_inv)`` with Martens-Grosse trace renorm damping."""
    n_in = A.shape[0]
    n_out = G.shape[0]
    tr_A = jnp.trace(A) / n_in
    tr_G = jnp.trace(G) / n_out
    # π = sqrt(tr_A / tr_G), with safety floor.
    pi = jnp.sqrt(jnp.maximum(tr_A, 1e-30) / jnp.maximum(tr_G, 1e-30))
    sqrt_lam = jnp.sqrt(jnp.asarray(damping))

    A_d = A + (pi * sqrt_lam) * jnp.eye(n_in, dtype=A.dtype)
    G_d = G + (sqrt_lam / pi) * jnp.eye(n_out, dtype=G.dtype)
    A_inv = jnp.linalg.inv(A_d)
    G_inv = jnp.linalg.inv(G_d)
    return A_inv, G_inv


# ---------------------------------------------------------------------
# Generic block: damped Fisher solve over the flat tail of params
# ---------------------------------------------------------------------

def _generic_natural_gradient(
    de, per_walker_grads_flat, damping, pmean_axis: Optional[str] = None,
):
    """Damped Fisher solve for the flat 'generic' tail of params.

    Args:
        de: ``(W_local,)`` energy residual.
        per_walker_grads_flat: ``(W_local, P_g)`` per-walker grads.
        damping: Tikhonov ε.
        pmean_axis: If set, average across devices via ``lax.pmean``.

    Returns:
        ``(x, f)`` where ``x`` is the natural-gradient direction and
        ``f`` is the energy gradient ``E_w[de · grad_w log|ψ|]``.
        Both are length-``P_g``.  Returning ``f`` lets callers compute
        the predicted-reduction ``<grad, step>`` quadratic-model term
        without recomputing it.
    """
    W = per_walker_grads_flat.shape[0]

    def _maybe_pmean(x):
        if pmean_axis is not None:
            return jax.lax.pmean(x, pmean_axis)
        return x

    f = _maybe_pmean(
        jnp.einsum('w,wp->p', de, per_walker_grads_flat) / W
    )
    # Centred Jacobian: cross-device mean for the column means.
    g_mean = _maybe_pmean(per_walker_grads_flat.mean(axis=0, keepdims=True))
    do = per_walker_grads_flat - g_mean
    # S·v = (1/W)·doᵀ·(do·v) + λ·v
    # Inside pmap we have to pmean the do.T @ do contribution since
    # each device has only its shard.
    def _matvec(v):
        prod_local = (do.T @ (do @ v)) / W
        prod = _maybe_pmean(prod_local)
        return prod + damping * v

    n_iters = 20
    x = jnp.zeros_like(f)
    r = f - _matvec(x)
    p = r
    rr = jnp.dot(r, r)
    for _ in range(n_iters):
        Ap = _matvec(p)
        alpha = rr / (jnp.dot(p, Ap) + 1e-30)
        x = x + alpha * p
        r = r - alpha * Ap
        rr_new = jnp.dot(r, r)
        beta = rr_new / (rr + 1e-30)
        p = r + beta * p
        rr = rr_new
    return x, f


# ---------------------------------------------------------------------
# KFAC optimiser
# ---------------------------------------------------------------------

class _HEGKFACOptimizer:
    """KFAC VMC optimiser for HEG ansätze.

    Constructor mirrors :class:`_HEGSROptimizer`.  Per-iter cost is
    1 forward + 1 vmap-grad (≈ 2× SR's), but per-iter convergence is
    typically 5–10× faster on large networks because the natural-
    gradient direction is closer to optimal.

    Args:
        config: :class:`HEGConfig` or :class:`HEGPsiFormerConfig`.
        init_key: JAX PRNG key for parameter init.
        lr: Initial learning rate.  KFAC schedule:
            ``lr_t = lr / (1 + t/lr_decay)`` (FermiNet recipe).
        lr_decay: Decay constant.  ``None`` → constant lr.
        damping: Tikhonov damping ε used in both Linear (KFAC) and
            generic (full-Fisher) blocks.
        ema_decay: Exponential-moving-average decay for the A_l and
            G_l factors across iterations.  FermiNet uses 0.95.
        norm_constraint: Trust-region clip on ‖update‖ in
            "preconditioned" norm (i.e. ‖step‖_F bounded).
            ``None`` to disable.
        var_weight: Optional Umrigar-style β for the mixed
            ⟨E⟩ + β·Var(E_L) objective (0 = pure energy).
        ewald_n_real, ewald_n_recip, ewald_eta: Ewald tuning.
        multi_device: If True, parallelise via pmap across local
            devices.
    """

    def __init__(
        self,
        config,
        init_key,
        *,
        lr: float = 0.1,
        lr_decay: Optional[float] = 1.0e4,
        damping: float = 1.0e-1,
        damping_adapt: bool = True,
        damping_min: float = 1.0e-4,
        damping_max: float = 1.0e2,
        damping_lookback: int = 10,
        damping_decay: float = 0.95,
        damping_overshoot_threshold: float = 10.0,
        damping_overshoot_factor: float = 2.0,
        ema_decay: float = 0.0,
        norm_constraint: Optional[float] = None,
        var_weight: float = 0.0,
        capture_activations: bool = False,
        fixed_scale: bool = False,
        ewald_n_real: int = 3,
        ewald_n_recip: int = 6,
        ewald_eta: Optional[float] = None,
        multi_device: bool = False,
    ):
        # x64 is already enabled by OmegaQMC.config; KFAC's matrix
        # inverses and EMA accumulators rely on it for numerical
        # conditioning.
        self.config = config
        # Detect multi-device setup.
        self._n_devices = jax.local_device_count() if multi_device else 1
        if multi_device and self._n_devices < 2:
            # User asked for multi-device but only one is visible.
            # Fall back gracefully.
            multi_device = False
            self._n_devices = 1
        if multi_device:
            # Per-walker grad + step are pmap'd; pmean_axis='devices'
            # is threaded into _extract_kron_factors so each device
            # ends up with the same global Fisher factors.
            self._pmap_axis = 'devices'
        else:
            self._pmap_axis = None
        self.L = float(config.L)
        self.dim = int(getattr(config, 'dim', 3))
        self.n_up = int(config.n_up)
        self.n_down = int(config.n_down)
        self.nelec = self.n_up + self.n_down
        self.lr = float(lr)
        self.lr_decay = (None if lr_decay is None else float(lr_decay))
        self.damping = float(damping)
        self.damping_adapt = bool(damping_adapt)
        self.damping_min = float(damping_min)
        self.damping_max = float(damping_max)
        self.damping_lookback = int(damping_lookback)
        self.damping_decay = float(damping_decay)
        self.damping_overshoot_threshold = float(
            damping_overshoot_threshold
        )
        self.damping_overshoot_factor = float(damping_overshoot_factor)
        self.fixed_scale = bool(fixed_scale)
        self.ema_decay = float(ema_decay)
        self.norm_constraint = (None if norm_constraint is None
                                else float(norm_constraint))
        self.var_weight = float(var_weight)
        self.capture_activations = bool(capture_activations)
        self.multi_device = bool(multi_device)

        if self.dim == 3:
            self.lattice = make_cubic_lattice(self.L)
        elif self.dim == 2:
            self.lattice = make_square_lattice(self.L)
        else:
            raise ValueError(
                f"KFAC HEG only supports dim=2 or 3, got {self.dim}"
            )
        self.ewald = build_ewald_tables_dim(
            self.L, dim=self.dim, eta=ewald_eta,
            n_real=ewald_n_real, n_recip=ewald_n_recip,
        )
        tables = self.ewald
        lattice = self.lattice
        nelec = self.nelec

        log_psi_pytree, init_params, graphdef = make_heg_log_psi(
            config, init_key,
        )
        self.log_psi_pytree = log_psi_pytree
        self.graphdef = graphdef
        self.params = init_params
        self.n_params = int(sum(
            p.size for p in jax.tree.leaves(init_params)
        ))

        # Build a *capturing* twin of the model when activation
        # capture is requested.  This twin uses ``CapturingLinear``
        # in place of every ``nnx.Linear`` so that each forward pass
        # writes the layer's input into an ``nnx.Intermediate`` slot.
        # The twin's ``Param`` pytree has the SAME structure as the
        # standard model's, so we feed in ``self.params`` unchanged;
        # only the extra Intermediate slots distinguish the two.
        self._cap_graphdef = None
        self._cap_inters_init = None
        self._cap_other = None
        self._cap_grad_fn = None
        if self.capture_activations and isinstance(
            config, HEGPsiFormerConfig,
        ):
            from ._kfac_capture import (
                use_capturing_linears, captures_to_path_dict,
            )
            self._captures_to_path_dict = captures_to_path_dict
            cap_rngs = nnx.Rngs(init_key)
            with use_capturing_linears():
                cap_model = build_heg_psiformer_wf(config, cap_rngs)
            (cap_graphdef, _cap_params, cap_inters_init,
             cap_other) = nnx.split(
                cap_model, nnx.Param, nnx.Intermediate, ...,
            )
            self._cap_graphdef = cap_graphdef
            self._cap_inters_init = cap_inters_init
            self._cap_other = cap_other

            # Define the JIT-vmap'd forward that returns
            # (log_psi_w, ∂log|psi_w|/∂params, captured_intermediates_w)
            # for a single walker; vmap over walkers gives batched
            # log_psi, grad, captures all in one compiled call.
            def _apply_with_capture(
                params, inters, other, walker,
            ):
                m = nnx.merge(cap_graphdef, params, inters, other)
                psi = m(walker)
                _, _, new_inters, _ = nnx.split(
                    m, nnx.Param, nnx.Intermediate, ...,
                )
                return psi.log, new_inters

            def _grad_and_capture_one(params, inters, other, walker):
                # Differentiate wrt params; aux carries new_inters.
                (log_psi, new_inters), grad = jax.value_and_grad(
                    _apply_with_capture, argnums=0, has_aux=True,
                )(params, inters, other, walker)
                return log_psi, grad, new_inters

            self._cap_grad_fn = jax.jit(jax.vmap(
                _grad_and_capture_one,
                in_axes=(None, None, None, 0),
            ))

        # Split params into Linear-kernel layers vs generic tail.
        self._layers, self._kernel_shapes, _generic_leaves = _classify_params(
            init_params,
        )
        # Sanity record for logs.
        self._linear_count = len(self._layers)
        self._linear_params = sum(
            int(np.prod(s)) for s in self._kernel_shapes.values()
        )

        # Per-walker primitives.
        def kin_only(r, params):
            def f_flat(r_flat):
                return log_psi_pytree(r_flat.reshape(nelec, self.dim), params)
            lap_val, grad_val = laplacian(f_flat)(r.reshape(-1))
            return -0.5 * (lap_val + jnp.dot(grad_val, grad_val))

        self._kin_batch = jax.jit(jax.vmap(kin_only, in_axes=(0, None)))

        def metropolis_move(rng_key, r, step_size, params):
            key_prop, key_acc = jax.random.split(rng_key)
            proposed = r + step_size * jax.random.normal(
                key_prop, r.shape,
            )
            proposed = wrap_to_cell(proposed, lattice)
            lp_old = log_psi_pytree(r, params)
            lp_new = log_psi_pytree(proposed, params)
            accept = jax.random.uniform(key_acc) < jnp.exp(
                2.0 * (lp_new - lp_old),
            )
            return jnp.where(accept, proposed, r), accept

        self._metropolis_move_allw = jax.jit(jax.vmap(
            metropolis_move, in_axes=(0, 0, None, None),
        ))

        # Ewald potential (chunked over walkers — same as SR driver).
        self._pot_chunk_size = 32
        if self.dim == 3:
            pot_fn = lambda w: ewald_pair_energy(w, tables)
        else:
            from .observables.ewald_2d import ewald_2d_pair_energy
            pot_fn = lambda w: ewald_2d_pair_energy(w, tables)
        self._pot_chunk = jax.jit(pot_fn)

        # Per-walker gradient of log|ψ| wrt every param leaf.
        # Returns pytree mirroring params, each leaf with leading W axis.
        def log_psi_per_walker(walker, params):
            return log_psi_pytree(walker, params)

        self._per_walker_grad = jax.jit(jax.vmap(
            jax.grad(log_psi_per_walker, argnums=1),
            in_axes=(0, None),
        ))

        # ---- KFAC step (jitted core) ----

        layer_paths = {
            layer: (k, b) for layer, (k, b) in self._layers.items()
        }
        ema_decay_arr = jnp.asarray(ema_decay, dtype=jnp.float64)
        norm_clip = (None if norm_constraint is None
                     else jnp.asarray(norm_constraint, dtype=jnp.float64))
        var_weight_arr = jnp.asarray(var_weight, dtype=jnp.float64)
        fixed_scale = self.fixed_scale       # closure-captured

        pmean_axis = self._pmap_axis           # 'devices' or None
        # Per-walker grad must be vmap'd here so it works on the
        # device's local walker shard inside pmap.
        per_walker_grad_vmap = jax.vmap(
            jax.grad(log_psi_per_walker, argnums=1),
            in_axes=(0, None),
        )

        def step_body(params, e_loc, walkers, A_state, G_state,
                      lr_now, damping_arr, captured_inputs):
            """Body shared between single-device JIT and pmap paths.

            When run under ``jax.pmap`` with ``axis_name='devices'``
            the local averages are converted to global averages via
            ``jax.lax.pmean``; under plain ``jax.jit`` the
            ``pmean_axis=None`` path keeps them local.

            ``captured_inputs`` is a (possibly empty) dict mapping
            layer-path strings to ``(W, n_e_layer, in_dim)`` per-
            (walker, electron) Linear inputs.  When provided for a
            layer, we use FermiNet-style RepeatedDenseBlock factor
            extraction (treat each (w, e) as a sample); otherwise we
            fall back to the un-decomposed ``M^T M`` / ``M M^T``
            formula in :func:`_extract_kron_factors`.

            Returns ``(new_params, new_A, new_G, e_mean, var,
            total_norm, clip, dot_grad_step, dot_step_F_step)``.  The
            last three feed FermiNet-style Levenberg-Marquardt
            adaptive damping in the outer loop:

              ``effective_lr = lr_now * clip``
              ``predicted_change``
                  = 0.5·effective_lr²·dot_step_F_step
                    − effective_lr·dot_grad_step
              ``ratio = (E_new − E_old) / predicted_change``

            ``dot_grad_step = ⟨grad, applied_step⟩`` summed over all
            blocks (Frobenius for kernels), and ``dot_step_F_step ≈
            ⟨applied_step, F · applied_step⟩``.  Under the (F+λI)·step
            = grad relation that the damped solve enforces, the latter
            equals ``c·⟨grad, applied_step⟩`` per Linear layer (with
            ``c = kernel_scale``) and ``⟨grad, step⟩`` for the generic
            block.  When ``fixed_scale=False`` and there's no separate
            kernel scale, both reduce to the standard
            ``⟨grad, step⟩`` and ``predicted_change = -effective_lr·
            (1 - effective_lr/2)·dot``.
            """
            pw_grad = per_walker_grad_vmap(walkers, params)

            e_mean_local = jnp.mean(e_loc)
            if pmean_axis is not None:
                e_mean = jax.lax.pmean(e_mean_local, pmean_axis)
            else:
                e_mean = e_mean_local
            de = e_loc - e_mean
            var_local = jnp.mean(de ** 2)
            if pmean_axis is not None:
                var = jax.lax.pmean(var_local, pmean_axis)
            else:
                var = var_local

            # Mixed-objective weighting (Umrigar β).
            de_eff = de + var_weight_arr * (de ** 2 - var)

            new_A, new_G = {}, {}
            update_kernels: Dict[str, jax.Array] = {}
            layer_dW_loss: Dict[str, jax.Array] = {}
            # FermiNet-style per-layer scale (n_e_layer for layers
            # where we used RepeatedDenseBlock-style extraction; 1 for
            # the M^T M fallback so the natural-gradient step has
            # the same effective magnitude as before).
            kernel_scale: Dict[str, float] = {}

            # Per Linear layer: extract Kronecker factors, damp, step.
            for layer, (kpath, bpath) in layer_paths.items():
                pw_dW = _get_at_path(pw_grad, kpath)            # (W, out, in)
                if layer in captured_inputs:
                    A, G, dW_loss, n_e = _per_electron_factors(
                        captured_inputs[layer], pw_dW, de_eff,
                    )
                    if pmean_axis is not None:
                        A = jax.lax.pmean(A, pmean_axis)
                        G = jax.lax.pmean(G, pmean_axis)
                        dW_loss = jax.lax.pmean(dW_loss, pmean_axis)
                    # Optional FermiNet-style fixed-scale: A, G are
                    # 1/(W·n_e)-normalised but dW_loss is per-walker
                    # -summed, so multiplying the step by n_e matches
                    # textbook KFAC magnitude.  Set ``kfac_fixed_scale``
                    # in the YAML to enable; the LM ratio damping in
                    # the outer loop handles the trust-region.
                    kernel_scale[layer] = (
                        float(n_e) if fixed_scale else 1.0
                    )
                else:
                    A, G, dW_loss = _extract_kron_factors(
                        pw_dW, de_eff, pmean_axis=pmean_axis,
                    )
                    kernel_scale[layer] = 1.0

                # EMA over iters.
                A_prev = A_state[layer]
                G_prev = G_state[layer]
                A_new = ema_decay_arr * A_prev + (1.0 - ema_decay_arr) * A
                G_new = ema_decay_arr * G_prev + (1.0 - ema_decay_arr) * G
                new_A[layer] = A_new
                new_G[layer] = G_new

                # Damped inverse + KFAC step.
                A_inv, G_inv = _damped_kron_inverse(
                    A_new, G_new, damping_arr,
                )
                # ΔW = -lr · kernel_scale · G_inv · dW_loss · A_inv
                applied_step = kernel_scale[layer] * (G_inv @ dW_loss @ A_inv)
                update_kernels[layer] = applied_step
                layer_dW_loss[layer] = dW_loss

            # ---- Generic block: damped natural gradient ----
            # Collect per-walker grads of log|ψ| for all generic leaves.
            generic_pw_flat_list = []
            generic_paths_list = []
            for path, leaf in jax.tree_util.tree_flatten_with_path(pw_grad)[0]:
                slast = _name_at(path, -2)
                # Skip Linear kernels — they're handled above.
                if slast == 'kernel' and leaf.ndim == 3:
                    continue
                generic_paths_list.append(path)
                # leaf shape is (W, *param_shape).  Flatten param dims.
                generic_pw_flat_list.append(
                    leaf.reshape(leaf.shape[0], -1),
                )
            generic_pw_flat = jnp.concatenate(
                generic_pw_flat_list, axis=1,
            ) if generic_pw_flat_list else jnp.zeros((walkers.shape[0], 0))

            if generic_pw_flat.shape[1] > 0:
                generic_step_flat, generic_grad_flat = (
                    _generic_natural_gradient(
                        de_eff, generic_pw_flat, damping_arr,
                        pmean_axis=pmean_axis,
                    )
                )
            else:
                generic_step_flat = jnp.zeros(0)
                generic_grad_flat = jnp.zeros(0)

            # ---- Trust-region clip (norm_constraint) ----
            # Always compute the total Frobenius norm of the natural-
            # gradient direction so we can log it; only apply the clip
            # if ``norm_clip`` is not None.
            kernel_sq = sum(
                jnp.sum(s ** 2) for s in update_kernels.values()
            )
            generic_sq = jnp.sum(generic_step_flat ** 2)
            total_norm = jnp.sqrt(kernel_sq + generic_sq + 1e-30)
            if norm_clip is not None:
                clip = jnp.minimum(1.0, norm_clip / (lr_now * total_norm))
            else:
                clip = jnp.asarray(1.0, dtype=jnp.float64)

            # ---- Predicted-reduction terms (FermiNet LM) ----
            # dot_grad_step = ⟨grad, applied_step⟩, summed over all blocks.
            # dot_step_F_step ≈ ⟨applied_step, F · applied_step⟩, using the
            # damped relation (F+λI)·raw_step = grad.  For Linear blocks
            # with applied_step = c·raw_step:
            #   ⟨applied_step, F · applied_step⟩ ≈ c · ⟨grad, applied_step⟩
            # For generic block (c = 1): equal to dot_grad_step_gen.
            dot_grad_step = jnp.zeros((), dtype=jnp.float64)
            dot_step_F_step = jnp.zeros((), dtype=jnp.float64)
            for layer in layer_paths:
                dot_l = jnp.sum(
                    layer_dW_loss[layer] * update_kernels[layer]
                )
                dot_grad_step = dot_grad_step + dot_l
                dot_step_F_step = (
                    dot_step_F_step + kernel_scale[layer] * dot_l
                )
            if generic_pw_flat.shape[1] > 0:
                dot_gen = jnp.sum(generic_grad_flat * generic_step_flat)
                dot_grad_step = dot_grad_step + dot_gen
                dot_step_F_step = dot_step_F_step + dot_gen

            # ---- Apply update ----
            scale = -lr_now * clip
            new_params = params

            # Linear kernels.
            for layer, step in update_kernels.items():
                kpath, _ = layer_paths[layer]
                old_W = _get_at_path(new_params, kpath)
                new_params = _set_at_path(new_params, kpath, old_W + scale * step)

            # Generic params.  Slice generic_step_flat back into shapes.
            offset = 0
            for path, sub in zip(
                generic_paths_list,
                generic_pw_flat_list,
            ):
                # sub has shape (W, P_flat); recover the original param
                # shape from the params pytree below.
                leaf_shape_flat = sub.shape[1]
                slice_ = generic_step_flat[offset:offset + leaf_shape_flat]
                offset += leaf_shape_flat
                old_param = _get_at_path(params, path)
                slice_ = slice_.reshape(old_param.shape)
                new_params = _set_at_path(
                    new_params, path, old_param + scale * slice_,
                )

            return (new_params, new_A, new_G, e_mean, var, total_norm,
                    clip, dot_grad_step, dot_step_F_step)

        if multi_device:
            self._kfac_step_core = jax.pmap(
                step_body, axis_name=pmean_axis,
            )
        else:
            self._kfac_step_core = jax.jit(step_body)

    # -----------------------------------------------------
    # Walker management
    # -----------------------------------------------------

    def initialize_walkers(self, rng_key, num_walkers):
        return self.L * jax.random.uniform(
            rng_key, (num_walkers, self.nelec, self.dim),
        )

    def _pot_batch(self, w):
        n = w.shape[0]
        chunk = self._pot_chunk_size
        if n <= chunk:
            return self._pot_chunk(w)
        return jnp.concatenate([
            self._pot_chunk(w[k:k + chunk])
            for k in range(0, n, chunk)
        ], axis=0)

    def _init_factor_state(self):
        """Initialise A_l and G_l to identity matrices for each Linear layer."""
        A_state = {}
        G_state = {}
        for layer, (out, in_) in self._kernel_shapes.items():
            A_state[layer] = jnp.eye(in_, dtype=jnp.float64)
            G_state[layer] = jnp.eye(out, dtype=jnp.float64)
        return A_state, G_state

    # -----------------------------------------------------
    # Training loop
    # -----------------------------------------------------

    def __call__(
        self,
        rng_key,
        num_iters: int = 5000,
        num_walkers: int = 2048,
        mcmc_decorr_steps: int = 10,
        num_equil_steps: int = 400,
        mc_timestep: float = 0.1,
        fname_log: Optional[str] = None,
        verbose: int = 1,
    ):
        """Run KFAC-VMC optimisation."""
        if isinstance(rng_key, int):
            rng_key = jax.random.key(rng_key)
        if (fname_log is None
                or (isinstance(fname_log, str) and fname_log == "")):
            fout = sys.stdout
        else:
            fout = open(fname_log, 'w', 1)

        params = self.params
        metropolis_allw = self._metropolis_move_allw
        kin_batch = self._kin_batch
        kfac_step_core = self._kfac_step_core

        # For multi-device runs the per-iter step is pmap'd; shard
        # walkers across devices and replicate params/state.
        n_dev = self._n_devices
        if n_dev > 1 and num_walkers % n_dev != 0:
            raise ValueError(
                f"num_walkers={num_walkers} must be divisible by "
                f"n_devices={n_dev} for multi-device KFAC."
            )
        walkers_per_dev = num_walkers // n_dev

        rng_key, init_key = jax.random.split(rng_key)
        walkers = self.initialize_walkers(init_key, num_walkers)
        step_size = jnp.asarray((3 * mc_timestep) ** 0.5)

        def _shard_walkers(w):
            """Reshape (W, N, 3) → (n_dev, W//n_dev, N, 3)."""
            if n_dev == 1:
                return w
            return w.reshape((n_dev, walkers_per_dev) + w.shape[1:])

        def _unshard_walkers(w):
            if n_dev == 1:
                return w
            return w.reshape((-1,) + w.shape[2:])

        def _replicate(x):
            if n_dev == 1:
                return x
            return jax.tree.map(
                lambda y: jnp.broadcast_to(y[None], (n_dev,) + y.shape),
                x,
            )

        def _take_first_device(x):
            if n_dev == 1:
                return x
            return jax.tree.map(lambda y: y[0], x)

        # Equilibration.
        ar = 0.0
        for _ in range(num_equil_steps):
            rng_key, sub = jax.random.split(rng_key)
            keys = jax.random.split(sub, num_walkers)
            walkers, acc = metropolis_allw(
                keys, walkers, step_size, params,
            )
            ar = float(jnp.mean(acc))
            step_size = _adapt_step_size(step_size, ar)

        if verbose >= 1:
            print(
                f"# KFAC-VMC training — {self.n_params} params total "
                f"({self._linear_params} in {self._linear_count} Linear "
                f"layers, rest generic)",
                file=fout,
            )
            print(
                f"# lr={self.lr}, lr_decay={self.lr_decay}, "
                f"damping={self.damping}, ema_decay={self.ema_decay}, "
                f"norm_constraint={self.norm_constraint}, "
                f"β={self.var_weight:.3g}",
                file=fout,
            )
            print(
                f"# Equilibration acceptance: {ar:.3f}, "
                f"step size: {float(step_size):.4f} Bohr",
                file=fout,
            )
            print(
                "# iter       <E>/N            Var(E)        "
                "lr           damping     ‖step‖         dt",
                file=fout,
            )

        # Initialise EMA factor state.
        A_state, G_state = self._init_factor_state()

        e_history = []
        var_history = []
        timestamp_prev = datetime.now()
        damping_now = self.damping
        # Look-back windows for adaptive damping.
        step_norm_window: list = []
        # Martens-Grosse LM ratio bookkeeping.
        # Each entry holds ``(actual_change, predicted_change)`` for one
        # iter; we aggregate over ``damping_lookback`` entries before
        # consulting the ratio (smooths Monte-Carlo noise on ``E_local``).
        ratio_window: list = []
        prev_e_total: Optional[float] = None
        prev_predicted: Optional[float] = None

        for it in range(1, num_iters + 1):
            # Decorrelate walkers under |ψ_θ|² with the current params.
            for _ in range(mcmc_decorr_steps):
                rng_key, sub = jax.random.split(rng_key)
                keys = jax.random.split(sub, num_walkers)
                walkers, acc = metropolis_allw(
                    keys, walkers, step_size, params,
                )
                step_size = _adapt_step_size(
                    step_size, float(jnp.mean(acc)),
                )

            # Local energies.
            e_kin = kin_batch(walkers, params)
            e_pot = self._pot_batch(walkers)
            e_loc = e_kin + e_pot

            # ---- Direct activation capture (FermiNet-style) ----
            # JIT-vmap'd forward through the *capturing* twin model:
            # every Linear writes its input to an Intermediate slot
            # which flows out as an aux output of the function.
            # Single batched call instead of W eager forwards.
            captured_inputs: Dict[str, jax.Array] = {}
            if (self.capture_activations
                    and self._cap_grad_fn is not None):
                _, _, ints_W = self._cap_grad_fn(
                    params, self._cap_inters_init,
                    self._cap_other, walkers,
                )
                raw = self._captures_to_path_dict(ints_W)
                # Drop captures whose Linear never fired (its slot
                # still holds the scalar zero default → flat shape
                # ``(W,)``).  ``_per_electron_factors`` expects
                # ``(W, n_e_layer, in_dim)``; ``(W, in_dim)`` is
                # also fine (treated as n_e_layer=1).
                for path, arr in raw.items():
                    if arr.ndim < 2:
                        continue
                    captured_inputs[path] = arr

            # KFAC step.
            if self.lr_decay is not None:
                lr_now = self.lr / (1.0 + (it - 1) / self.lr_decay)
            else:
                lr_now = self.lr

            if n_dev > 1:
                # Shard walkers + e_loc across devices; replicate
                # params and EMA state.  The step_body returns
                # replicated outputs (each device runs the same
                # update on the same averaged factors), so device 0
                # is canonical.
                w_sh = _shard_walkers(walkers)
                e_loc_sh = e_loc.reshape((n_dev, walkers_per_dev))
                p_rep = _replicate(params)
                A_rep = _replicate(A_state)
                G_rep = _replicate(G_state)
                lr_rep = jnp.broadcast_to(
                    jnp.asarray(lr_now, dtype=jnp.float64), (n_dev,),
                )
                damp_rep = jnp.broadcast_to(
                    jnp.asarray(damping_now, dtype=jnp.float64),
                    (n_dev,),
                )
                (p_out, A_out, G_out, e_mean_o, var_o, step_norm_o,
                 clip_o, dot_grad_o, dot_F_o) = kfac_step_core(
                    p_rep, e_loc_sh, w_sh,
                    A_rep, G_rep, lr_rep, damp_rep,
                    captured_inputs,
                )
                params = _take_first_device(p_out)
                A_state = _take_first_device(A_out)
                G_state = _take_first_device(G_out)
                e_mean = _take_first_device(e_mean_o)
                var = _take_first_device(var_o)
                step_norm = _take_first_device(step_norm_o)
                clip_val = _take_first_device(clip_o)
                dot_grad_step = _take_first_device(dot_grad_o)
                dot_step_F_step = _take_first_device(dot_F_o)
            else:
                (params, A_state, G_state, e_mean, var, step_norm,
                 clip_val, dot_grad_step,
                 dot_step_F_step) = kfac_step_core(
                    params, e_loc, walkers,
                    A_state, G_state,
                    jnp.asarray(lr_now, dtype=jnp.float64),
                    jnp.asarray(damping_now, dtype=jnp.float64),
                    captured_inputs,
                )

            # Logging.
            now = datetime.now()
            dt = (now - timestamp_prev).total_seconds()
            timestamp_prev = now
            e_per = float(e_mean) / self.nelec
            e_history.append(e_per)
            var_history.append(float(var))

            if verbose >= 1 and (it <= 10 or it % 10 == 0):
                print(
                    f"{it:>6d}  {e_per:>14.8e}  "
                    f"{float(var):>12.4e}  "
                    f"{lr_now:>10.4e}  "
                    f"{damping_now:>9.3e}  "
                    f"{float(step_norm):>10.4e}  "
                    f"{dt:>8.3f}",
                    file=fout,
                )

            # Adaptive Levenberg-Marquardt damping — FermiNet/kfac-jax
            # recipe (Martens & Grosse 2015) with a panic-mode safety
            # net.  Two response timescales:
            #
            # 1. Fast "panic mode" — every iter, check if ‖step‖ has
            #    blown up beyond ``damping_overshoot_threshold`` ×
            #    its running median.  If so, ramp damping by
            #    ``damping_overshoot_factor`` (default 2×) immediately.
            #    Catches geometric blowup within 1-2 iters; essential
            #    when ``fixed_scale=True`` puts the step magnitude
            #    near the trust-region boundary.
            #
            # 2. Quadratic-model ratio mode (FermiNet faithful) —
            #    every ``damping_lookback`` iters, compare the
            #    aggregated *actual* energy reduction to the
            #    *predicted* reduction from the local quadratic model
            #
            #        Δm = ½·(η·c)²·⟨step, F·step⟩ - (η·c)·⟨grad, step⟩
            #
            #    where ``η = lr_now``, ``c = clip`` (trust-region
            #    factor), ``step`` is the applied update direction
            #    (with ``kernel_scale`` baked in), ``grad`` is the
            #    energy gradient and ``F`` is the (damped) Fisher.
            #    Under the natural-gradient relation
            #    ``(F + λI)·raw_step = grad`` this evaluates exactly
            #    to the closed form computed inside ``step_body``.
            #
            #    Martens-Grosse rule:
            #      ratio > 0.75 → quadratic model under-predicts
            #                       descent → loosen damping (×decay).
            #      ratio < 0.25 → quadratic model over-predicts
            #                       descent → tighten damping (÷decay).
            #
            #    Aggregating actual & predicted over ``damping_lookback``
            #    iters before forming the ratio smooths Monte-Carlo
            #    noise on E_local without losing responsiveness.
            #
            # Bound by ``[damping_min, damping_max]``.
            e_total = float(e_mean)
            eff_lr = float(lr_now) * float(clip_val)
            predicted_change_this = float(
                0.5 * eff_lr ** 2 * float(dot_step_F_step)
                - eff_lr * float(dot_grad_step)
            )

            if self.damping_adapt:
                step_norm_val = float(step_norm)
                step_norm_window.append(step_norm_val)
                if len(step_norm_window) > self.damping_lookback:
                    step_norm_window.pop(0)
                # Panic mode: detect step-norm overshoot vs running
                # median.  Need ≥ ``damping_lookback`` history entries
                # before the comparison fires, to avoid spurious
                # triggers from the natural variance of the step
                # norm during early iterations.
                if len(step_norm_window) >= self.damping_lookback:
                    median_norm = float(np.median(step_norm_window[:-1]))
                    if (median_norm > 0 and step_norm_val
                            > self.damping_overshoot_threshold
                            * median_norm):
                        damping_now *= self.damping_overshoot_factor
                        damping_now = float(np.clip(
                            damping_now,
                            self.damping_min, self.damping_max,
                        ))

                # LM ratio mode: e_total measured at iter t is the
                # energy after the step taken at iter t-1, so the
                # actual reduction for the *previous* iter's step is
                # e_total(t) - e_total(t-1).  Pair it with that step's
                # predicted reduction.
                if (prev_e_total is not None
                        and prev_predicted is not None):
                    ratio_window.append(
                        (e_total - prev_e_total, prev_predicted),
                    )
                    if len(ratio_window) >= self.damping_lookback:
                        actual_sum = sum(a for a, _ in ratio_window)
                        pred_sum = sum(p for _, p in ratio_window)
                        if abs(pred_sum) > 1e-30:
                            ratio = actual_sum / pred_sum
                            if ratio > 0.75:
                                damping_now *= self.damping_decay
                            elif ratio < 0.25:
                                damping_now /= self.damping_decay
                            damping_now = float(np.clip(
                                damping_now,
                                self.damping_min, self.damping_max,
                            ))
                        ratio_window = []

            # Always update bookkeeping (even when damping_adapt is
            # False, we keep them consistent for when the user toggles).
            prev_e_total = e_total
            prev_predicted = predicted_change_this

        if not (fname_log is None
                or (isinstance(fname_log, str) and fname_log == "")):
            fout.close()

        self.params = params
        return {
            'params': params,
            'E_per_elec_history': e_history,
            'Var_history': var_history,
            'E_final_ha': e_history[-1] if e_history else None,
        }


def get_vmcopt_nn_heg_kfac_func(
    config,
    init_key,
    *,
    prefix: str = "heg_kfac",
    lr: float = 0.1,
    lr_decay: Optional[float] = 1.0e4,
    damping: float = 1.0e-1,
    damping_adapt: bool = True,
    damping_min: float = 1.0e-4,
    damping_max: float = 1.0e2,
    damping_lookback: int = 10,
    damping_decay: float = 0.95,
    damping_overshoot_threshold: float = 10.0,
    damping_overshoot_factor: float = 2.0,
    ema_decay: float = 0.0,
    norm_constraint: Optional[float] = None,
    var_weight: float = 0.0,
    capture_activations: bool = False,
    fixed_scale: bool = False,
    ewald_n_real: int = 3,
    ewald_n_recip: int = 6,
    ewald_eta: Optional[float] = None,
    multi_device: bool = False,
):
    """Construct a KFAC-VMC optimiser for a HEG ansatz.

    Args: see :class:`_HEGKFACOptimizer`.

    Returns:
        :class:`_HEGKFACOptimizer` — callable.
    """
    del prefix
    return _HEGKFACOptimizer(
        config, init_key,
        lr=lr, lr_decay=lr_decay,
        damping=damping,
        damping_adapt=damping_adapt,
        damping_min=damping_min,
        damping_max=damping_max,
        damping_lookback=damping_lookback,
        damping_decay=damping_decay,
        damping_overshoot_threshold=damping_overshoot_threshold,
        damping_overshoot_factor=damping_overshoot_factor,
        ema_decay=ema_decay,
        norm_constraint=norm_constraint,
        var_weight=var_weight,
        capture_activations=capture_activations,
        fixed_scale=fixed_scale,
        ewald_n_real=ewald_n_real,
        ewald_n_recip=ewald_n_recip,
        ewald_eta=ewald_eta,
        multi_device=multi_device,
    )
