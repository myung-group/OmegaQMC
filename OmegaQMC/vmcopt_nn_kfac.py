"""KFAC (Kronecker-Factored Approximate Curvature) VMC optimiser
for molecular NN wavefunctions.

Molecular counterpart of :mod:`OmegaQMC.vmcopt_nn_heg_kfac`: the
KFAC core (Kronecker-factor extraction, Martens-Grosse damped
inverse, adaptive Levenberg-Marquardt damping, trust-region norm
constraint) is the same, while the Hamiltonian, sampler and
checkpointing follow :mod:`OmegaQMC.vmcopt_nn_sr`.

Why KFAC rather than SR for large walker counts
-----------------------------------------------
Stochastic reconfiguration must hold the full log-derivative
Jacobian ``O`` of shape ``(num_walkers, n_params)`` for the whole
CG solve, so its walker budget is inversely proportional to the
parameter count — see ``_autotune_sr_walkers`` in
:mod:`~OmegaQMC.vmcopt_nn_sr`, which literally computes
``nw = budget / (4 * n_params)``.

KFAC's persistent state is instead a pair of per-layer Kronecker
factors of shape ``(in, in)`` and ``(out, out)``, whose size does
not depend on the walker count at all.

Like SR, this optimiser never differentiates the local energy: it
needs only ``d log|psi| / d theta``, so it composes with the
analytic forward-Laplacian path in
:mod:`OmegaQMC.psi.nn.forward_lap` rather than fighting it.

Deliberate deviations from the HEG implementation
-------------------------------------------------
1. **Walker-chunked factor accumulation.**  The HEG driver
   evaluates the per-walker gradient for the whole batch inside
   its jitted step, so its peak is ``num_walkers * n_params`` —
   the very thing that caps SR.  Here the accumulation is a
   ``lax.scan`` over chunks of ``factor_chunk_size`` walkers, so
   peak memory is ``factor_chunk_size * n_params`` and the walker
   count is decoupled from the parameter count.  (For the
   PsiFormer used in testing, 2.1M parameters, the unchunked
   gradient for 1000 walkers is ~17 GB and does not fit.)
2. **Trust region in the Fisher norm.**  The HEG driver clips the
   *Euclidean* norm of the step.  That is not scale-meaningful
   for the ``M^T M`` factors, whose ``E[|a|^2] E[|g|^2]`` constant
   the learning rate is supposed to absorb; with a Euclidean bound
   the step is shrunk by that same constant every iteration and
   the optimiser stalls.  This module bounds
   ``dtheta^T F dtheta`` instead, as kfac-jax / FermiNet /
   DeepQMC do, which makes ``norm_constraint`` comparable with
   the value in DeepQMC's ``conf/task/opt/kfac.yaml``.
3. **Dtype-preserving parameter update.**  NNX initialises the
   network in float32 while ``OmegaQMC.config`` enables
   ``jax_enable_x64``, so writing the float64 KFAC step straight
   back into the parameters silently promotes the whole network
   to float64.  The factors, inverses and step stay in float64
   (conditioning), but each leaf is cast back to its own dtype on
   application.

Not implemented here (unlike the HEG driver): multi-device
``pmap``.  The generic helpers below still carry the
``pmean_axis`` plumbing, so adding it is mechanical, but it is
left out until it can be tested on more than one device.  Note
also that the HEG driver's ``pmap`` path passes ``captured_inputs``
into a ``pmap``ed step without sharding it, so its multi-device
and ``capture_activations`` options do not currently compose.

Two factor-extraction paths are available, matching the HEG driver:

a. **M^T M / M M^T (default, fast, biased).**  Uses the per-walker
   pytree gradient directly.  For a per-electron Linear with
   ``d log|psi_w| / dW = sum_e a_we (x) g_we`` this gives factors
   biased by per-walker ``||a||`` / ``||g||`` weights relative to
   FermiNet's strict per-(walker, electron) factorisation, but it
   is numerically stable and needs no model instrumentation.

b. **Per-electron capture (FermiNet-style, exact).**  Set
   ``capture_activations=True`` to build a *capturing twin* of the
   model in which every ``nnx.Linear`` is a ``CapturingLinear``
   (see :func:`OmegaQMC.psi.nn.kfac_capture.use_capturing_linears`).
   A single jitted vmap returns ``(log_psi, grad, captures)``; the
   per-electron output gradients ``g_we`` are then recovered from
   the captured inputs by a small ``(n_e, n_e)`` solve and the
   factors are accumulated over the flattened ``(W * n_e)``
   population, exactly as FermiNet's ``RepeatedDenseBlock`` does.

"""

from __future__ import annotations

import os
import sys
from datetime import datetime
from functools import partial
from typing import Any, Dict, List, Optional, Tuple

import h5py
import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from .constants import MIN_DIST_THRESHOLD
from .psi.nn.adapter import make_nn_log_psi
from .psi.nn.build import build_nn_wf
from .psi.nn.checkpoint import (
    load_nn_checkpoint,
    save_nn_checkpoint,
)
from .psi.nn.config import load_nn_config
from .psi.nn.types import PhysicalConfiguration


# Maximum walkers per forward-Laplacian kinetic-energy evaluation;
# see vmcopt_nn_iradam for the detailed rationale.
_KE_WALKER_CHUNK = 256

# Iterations between checkpoint writes.
_CHK_EVERY_KFAC = 500

# Metropolis step-size adaptation (matches the other NN drivers).
_TARGET_ACCEPTANCE_RATE = 0.5
_STEP_SIZE_ADAPTATION_RATE = 0.05


def _adapt_step_size(step_size, acceptance_rate):
    """Adapt the Metropolis step size toward the target rate."""
    return step_size * (
        1.0 + _STEP_SIZE_ADAPTATION_RATE
        * (acceptance_rate - _TARGET_ACCEPTANCE_RATE)
    )


# ---------------------------------------------------------------------
# Param-pytree introspection: identify Linear kernels vs generic leaves
# ---------------------------------------------------------------------

def _path_str(path) -> str:
    """Render a JAX tree-path, excluding the trailing ``.value``."""
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
    Dict[str, Tuple[int, int]],
    List[Any],
]:
    """Bucket parameter leaves into Linear kernels vs generic.

    Returns:
        layers: ``{layer_path: (kernel_key_path, bias_key_path)}``.
        kernel_shapes: ``{layer_path: kernel.shape}``.  NNX stores a
            ``Linear`` kernel as ``(in, out)``, so this is the
            ``(in, out)`` pair.
        generic: ``[(KeyPath, leaf), ...]`` for every other leaf.
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
    return jax.tree_util.tree_unflatten(
        treedef, [lf for _, lf in leaves],
    )


def _get_at_path(pytree, path):
    """Look up the leaf at ``path`` in ``pytree``."""
    leaves = jax.tree_util.tree_flatten_with_path(pytree)[0]
    for p, lf in leaves:
        if p == path:
            return lf
    raise KeyError(path)


# ---------------------------------------------------------------------
# Kronecker-factor accumulation
#
# All factor routines below return *sums* over the walker population
# rather than means, so a caller can accumulate them across walker
# chunks and normalise once at the end.  ``M`` is the per-walker
# gradient of ``log|psi|`` wrt a Linear kernel and therefore carries
# NNX's ``(in, out)`` kernel layout, i.e. shape ``(W, in, out)``.
#
# With that layout the two factors are
#
#     A_out[o, p] = sum_w sum_i M[w, i, o] M[w, i, p]   -> (out, out)
#     G_in [i, j] = sum_w sum_o M[w, i, o] M[w, j, o]   -> (in,  in)
#
# and the KFAC step for that layer is
#
#     dW = G_in_inv @ dW_loss @ A_out_inv               -> (in,  out)
#
# (The HEG module names these ``A``/``G`` with the opposite reading
# of which side is "input"; the maths is identical, only the labels
# here are made to match the shapes.)
# ---------------------------------------------------------------------

def _kron_factor_sums(
    per_walker_dW: jax.Array,
    de: jax.Array,
) -> Tuple[jax.Array, jax.Array, jax.Array, int]:
    """Un-decomposed ``M^T M`` / ``M M^T`` factor sums.

    Exact under the standard KFAC factorisation assumption
    (independence of activations and output gradients across the
    sample population), independently of how many rank-1 outer
    products contribute to each ``M_w``.  For per-electron Linears
    it is considerably more faithful than extracting a single rank-1
    SVD component, at the cost of an ``E[||a||^2] E[||g||^2]``
    proportionality constant that the learning rate absorbs (a
    scalar multiple of the Fisher leaves ``F^{-1} grad`` unchanged
    in direction).

    Args:
        per_walker_dW: ``(W, in, out)``.
        de: ``(W,)`` energy residual ``E_L - <E_L>``.

    Returns:
        ``(A_out_sum, G_in_sum, dW_loss_sum, n_samples)``.
    """
    M = per_walker_dW
    A_out = jnp.einsum('wio,wip->op', M, M)
    G_in = jnp.einsum('wio,wjo->ij', M, M)
    dW_loss = jnp.einsum('w,wio->io', de, M)
    return A_out, G_in, dW_loss, M.shape[0]


def _per_electron_factor_sums(
    captured_input: jax.Array,
    per_walker_dW: jax.Array,
    de: jax.Array,
    solve_damping: float = 1.0e-6,
) -> Tuple[jax.Array, jax.Array, jax.Array, int, int]:
    """Exact per-electron KFAC factor sums (FermiNet-style).

    Treats each ``(walker, electron)`` pair as its own Fisher
    sample, as FermiNet's ``RepeatedDenseBlock`` does.  The
    per-electron output gradients are recovered from the captured
    inputs and the per-walker kernel gradient by

        ``dW_w = a_w^T g_w``  ->
        ``g_w = (a_w a_w^T + lambda I)^{-1} a_w dW_w``

    Args:
        captured_input: ``(W, n_e, in)`` per-walker per-electron
            Linear inputs.  ``(W, in)`` is accepted for a global
            Linear (treated as ``n_e = 1``); higher-rank captures
            (e.g. edge features of shape ``(W, n_e, n_e, in)``) are
            flattened into a single sample axis.
        per_walker_dW: ``(W, in, out)``.
        de: ``(W,)`` energy residual.
        solve_damping: Tikhonov term on the ``(n_e, n_e)`` solve.

    Returns:
        ``(A_out_sum, G_in_sum, dW_loss_sum, n_samples, n_e)`` where
        ``n_samples = W * n_e`` is the population the factor sums run
        over, while ``dW_loss_sum`` remains a per-*walker* sum.
    """
    a_we = captured_input
    if a_we.ndim == 2:                      # (W, in) — global Linear
        a_we = a_we[:, None, :]
    elif a_we.ndim > 3:
        W_ = a_we.shape[0]
        in_dim_ = a_we.shape[-1]
        a_we = a_we.reshape(W_, -1, in_dim_)
    W, n_e, in_dim = a_we.shape

    AAT = jnp.einsum('wei,wfi->wef', a_we, a_we)
    AAT = AAT + solve_damping * jnp.eye(
        n_e, dtype=AAT.dtype,
    )[None]
    a_dW = jnp.einsum('wei,wio->weo', a_we, per_walker_dW)
    g_we = jnp.linalg.solve(AAT, a_dW)                 # (W, n_e, out)

    out_dim = per_walker_dW.shape[2]
    a_flat = a_we.reshape(W * n_e, in_dim)
    g_flat = g_we.reshape(W * n_e, out_dim)

    # E[a a^T] is the (in, in) factor; E[g g^T] the (out, out) one.
    G_in = jnp.einsum('si,sj->ij', a_flat, a_flat)
    A_out = jnp.einsum('so,sp->op', g_flat, g_flat)
    dW_loss = jnp.einsum('w,wio->io', de, per_walker_dW)
    return A_out, G_in, dW_loss, W * n_e, n_e


def _damped_kron_inverse(
    A_out: jax.Array, G_in: jax.Array, damping,
) -> Tuple[jax.Array, jax.Array]:
    """Factored-Tikhonov damped inverses (Martens & Grosse).

    The damping is split between the two factors with the trace
    ratio ``pi = sqrt(tr(A)/tr(G))`` so that the implied damping on
    the Kronecker product ``G (x) A`` is isotropic.
    """
    n_out = A_out.shape[0]
    n_in = G_in.shape[0]
    tr_A = jnp.trace(A_out) / n_out
    tr_G = jnp.trace(G_in) / n_in
    pi = jnp.sqrt(
        jnp.maximum(tr_A, 1e-30) / jnp.maximum(tr_G, 1e-30)
    )
    sqrt_lam = jnp.sqrt(jnp.asarray(damping))

    A_d = A_out + (pi * sqrt_lam) * jnp.eye(
        n_out, dtype=A_out.dtype,
    )
    G_d = G_in + (sqrt_lam / pi) * jnp.eye(
        n_in, dtype=G_in.dtype,
    )
    return jnp.linalg.inv(A_d), jnp.linalg.inv(G_d)


def _generic_natural_gradient(
    de, per_walker_grads_flat, damping, n_iters: int = 20,
    pmean_axis: Optional[str] = None,
):
    """Damped Fisher solve over the non-Linear tail of the params.

    Biases, envelope exponents and similar leaves have no Kronecker
    structure, so they get a plain conjugate-gradient natural
    gradient on the centred covariance.  This block is small — the
    Jacobian it materialises is ``(W, P_generic)``, not
    ``(W, n_params)``.

    Args:
        de: ``(W,)`` energy residual.
        per_walker_grads_flat: ``(W, P_generic)``.
        damping: Tikhonov epsilon.
        n_iters: CG iterations.
        pmean_axis: If set, average across devices via ``lax.pmean``.

    Returns:
        ``(x, f)`` — the natural-gradient direction and the raw
        energy gradient, both length ``P_generic``.  Returning ``f``
        lets the caller form the quadratic-model term
        ``<grad, step>`` without recomputing it.
    """
    W = per_walker_grads_flat.shape[0]

    def _maybe_pmean(x):
        if pmean_axis is not None:
            return jax.lax.pmean(x, pmean_axis)
        return x

    f = _maybe_pmean(
        jnp.einsum('w,wp->p', de, per_walker_grads_flat) / W
    )
    g_mean = _maybe_pmean(
        per_walker_grads_flat.mean(axis=0, keepdims=True)
    )
    do = per_walker_grads_flat - g_mean

    def _matvec(v):
        prod_local = (do.T @ (do @ v)) / W
        return _maybe_pmean(prod_local) + damping * v

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
# Driver
# ---------------------------------------------------------------------

class _VMCOptDriverNN_KFAC:
    """KFAC natural-gradient VMC optimiser for molecular NN trials.

    Each iteration decorrelates the walkers under ``|psi_theta|^2``,
    evaluates the local energy with the forward-Laplacian,
    accumulates the Kronecker factors in walker chunks, and applies
    the damped natural-gradient step
    ``dW = -lr * G_in_inv @ dW_loss @ A_out_inv`` per Linear layer
    (with a CG solve over the remaining leaves).

    Args:
        mol_info: :class:`~OmegaQMC.utils.Mole_custom` instance.
        config: :class:`~OmegaQMC.psi.nn.config.NNAnsatzConfig` or a
            string (built-in name or YAML path).
        init_key: JAX PRNG key for parameter initialisation.
        lr: Initial learning rate.  Schedule is FermiNet's
            ``lr_t = lr / (1 + t / lr_decay)``.
        lr_decay: Decay constant; ``None`` for a constant rate.
        damping: Initial Tikhonov damping, used by both the Linear
            (KFAC) and generic (full-Fisher) blocks.
        damping_adapt: Enable adaptive Levenberg-Marquardt damping.
        damping_min, damping_max: Bounds on the adapted damping.
        damping_lookback: Iterations aggregated before consulting
            the quadratic-model ratio, and the window used by the
            step-norm overshoot check.
        damping_decay: Multiplicative factor for the LM update.
        damping_overshoot_threshold, damping_overshoot_factor:
            Fast "panic mode" — if the step norm exceeds this
            multiple of its running median, damping is raised by
            this factor immediately.
        ema_decay: Exponential moving average applied to the
            Kronecker factors across iterations (FermiNet uses 0.95;
            0 disables).
        norm_constraint: Trust-region bound on ``lr * ||step||``;
            ``None`` disables the clip.
        var_weight: Umrigar-style beta for a mixed
            ``<E> + beta Var(E_L)`` objective (0 = pure energy).
        capture_activations: Use the exact per-electron factor path
            (builds a capturing twin of the model).
        fixed_scale: With ``capture_activations``, multiply each
            layer's step by ``n_e`` to match textbook KFAC
            magnitude (FermiNet's ``fixed_scale``).
    """

    def __init__(
        self,
        mol_info,
        config,
        init_key,
        *,
        lr: float = 0.05,
        lr_decay: Optional[float] = 1.0e4,
        damping: float = 1.0e-3,
        damping_adapt: bool = True,
        damping_min: float = 1.0e-6,
        damping_max: float = 1.0e2,
        damping_lookback: int = 10,
        damping_decay: float = 0.95,
        damping_overshoot_threshold: float = 10.0,
        damping_overshoot_factor: float = 2.0,
        ema_decay: float = 0.95,
        norm_constraint: Optional[float] = 1.0e-3,
        var_weight: float = 0.0,
        capture_activations: bool = False,
        fixed_scale: bool = False,
    ):
        nuc_crds = jnp.asarray(
            mol_info.coords, dtype=jnp.float64,
        )
        charges = jnp.asarray(
            mol_info.charges, dtype=jnp.float64,
        )
        nelec = mol_info.n_up + mol_info.n_down

        self.mol_info = mol_info
        self.nuc_crds = nuc_crds
        self.charges = charges
        self.nelec = nelec
        self.config_name = (
            config if isinstance(config, str)
            else getattr(config, 'name', 'custom')
        )

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
        self.damping_overshoot_factor = float(
            damping_overshoot_factor
        )
        self.ema_decay = float(ema_decay)
        self.norm_constraint = (
            None if norm_constraint is None else float(norm_constraint)
        )
        self.var_weight = float(var_weight)
        self.capture_activations = bool(capture_activations)
        self.fixed_scale = bool(fixed_scale)

        log_psi, init_params, graphdef, lap_grad = make_nn_log_psi(
            config, mol_info, init_key,
        )
        self.log_psi = log_psi
        self.graphdef = graphdef
        self.lap_grad = lap_grad
        self.init_params = init_params
        self.n_params = int(sum(
            p.size for p in jax.tree.leaves(init_params)
        ))

        # --- Nuclear repulsion ---
        n_nuc = len(charges)
        enr_nn = 0.0
        for a in range(n_nuc):
            for b in range(a + 1, n_nuc):
                rab = jnp.linalg.norm(nuc_crds[a] - nuc_crds[b])
                enr_nn = enr_nn + (charges[a] * charges[b] / rab)
        enr_nn = jnp.asarray(enr_nn, dtype=jnp.float64)
        self.enr_nn = enr_nn

        i_e, j_e = jnp.triu_indices(nelec, k=1)

        @jax.jit
        def energy_ee(elec_crds):
            diffs = elec_crds[i_e] - elec_crds[j_e]
            dists = jnp.linalg.norm(diffs, axis=-1)
            return jnp.sum(1.0 / dists)

        @jax.jit
        def energy_en(elec_crds):
            diffs = (
                elec_crds[:, None, :] - nuc_crds[None, :, :]
            )
            dists = jnp.linalg.norm(diffs, axis=-1)
            return -jnp.sum(charges[None, :] / dists)

        @jax.jit
        def energy_ke(elec_crds, params):
            lap_val, grad_val = lap_grad(
                elec_crds, nuc_crds, params,
            )
            return -0.5 * (
                lap_val + jnp.dot(grad_val, grad_val)
            )

        @jax.jit
        def total_local_energy(elec_crds, params):
            return (
                energy_ee(elec_crds) + energy_en(elec_crds)
                + energy_ke(elec_crds, params) + enr_nn
            )

        def batched_local_energy(walkers, params):
            n = walkers.shape[0]
            chunk = min(_KE_WALKER_CHUNK, n)
            return jax.lax.map(
                lambda w: total_local_energy(w, params),
                walkers,
                batch_size=chunk,
            )

        self.total_local_energy = total_local_energy
        self.compute_batch_energy = jax.jit(batched_local_energy)

        # --- Metropolis ---
        @jax.jit
        def metropolis_move(rng_key, elec_crds, step_size, params):
            key_prop, key_accept = jax.random.split(rng_key)
            proposed = elec_crds + step_size * jax.random.normal(
                key_prop, elec_crds.shape,
            )
            diffs_ee = proposed[i_e] - proposed[j_e]
            dists_ee = jnp.linalg.norm(diffs_ee, axis=-1)
            diffs_en = (
                proposed[:, None, :] - nuc_crds[None, :, :]
            )
            dists_en = jnp.linalg.norm(diffs_en, axis=-1)
            valid = (
                (dists_en.min() > MIN_DIST_THRESHOLD)
                & (dists_ee.min() > MIN_DIST_THRESHOLD)
            )
            lp_old = log_psi(elec_crds, nuc_crds, params)
            lp_new = log_psi(proposed, nuc_crds, params)
            accept = (
                jax.random.uniform(key_accept)
                < jnp.exp(2 * (lp_new - lp_old))
            ) & valid
            return (
                jnp.where(accept, proposed, elec_crds), accept,
            )

        self._metropolis_move_allw = jax.jit(jax.vmap(
            metropolis_move, in_axes=(0, 0, None, None),
        ))

        @partial(jax.jit, static_argnums=(4,))
        def decorr_scan(rng_key, walkers, step_size, params,
                        num_steps):
            def step(carry, _):
                rk, w, s, p = carry
                rk, rk1 = jax.random.split(rk)
                keys = jax.random.split(rk1, w.shape[0])
                nw, acc = jax.vmap(
                    metropolis_move, in_axes=(0, 0, None, None),
                )(keys, w, s, p)
                ar = acc.mean()
                return (rk, nw, _adapt_step_size(s, ar), p), ar

            carry = (rng_key, walkers, step_size, params)
            carry, ars = jax.lax.scan(
                step, carry, jnp.arange(num_steps),
            )
            return carry, ars

        self.decorr_scan = decorr_scan

        # --- Parameter classification ---
        (self._layers, self._kernel_shapes,
         _generic_leaves) = _classify_params(init_params)
        self._linear_count = len(self._layers)
        self._linear_params = sum(
            int(np.prod(s)) for s in self._kernel_shapes.values()
        )

        # Ordered list of non-kernel leaf paths, so the chunked
        # accumulation and the update step agree on the layout of
        # the flattened generic block.
        self._generic_paths = []
        self._generic_sizes = []
        for path, leaf in jax.tree_util.tree_flatten_with_path(
            init_params,
        )[0]:
            if _name_at(path, -2) == 'kernel' and leaf.ndim == 2:
                continue
            self._generic_paths.append(path)
            self._generic_sizes.append(int(leaf.size))

        # --- Capturing twin (exact per-electron factors) ---
        self._cap_grad_fn = None
        self._captures_to_path_dict = None
        if self.capture_activations:
            from .psi.nn.kfac_capture import (
                captures_to_path_dict, use_capturing_linears,
            )
            self._captures_to_path_dict = captures_to_path_dict
            cfg = (
                load_nn_config(config)
                if isinstance(config, str) else config
            )
            cap_rngs = nnx.Rngs(init_key)
            with use_capturing_linears():
                cap_model = build_nn_wf(cfg, mol_info, cap_rngs)
            (cap_graphdef, _cap_params, cap_inters_init,
             cap_other) = nnx.split(
                cap_model, nnx.Param, nnx.Intermediate, ...,
            )
            self._cap_inters_init = cap_inters_init
            self._cap_other = cap_other


            def _apply_with_capture(params, inters, other, elec):
                # Same interleaved -> grouped reordering the adapter
                # applies, so the twin sees identical inputs.
                r_grouped = jnp.concatenate(
                    [elec[::2], elec[1::2]], axis=0,
                )
                phys_conf = PhysicalConfiguration(
                    R=nuc_crds, r=r_grouped,
                    mol_idx=jnp.array(0),
                )
                m = nnx.merge(cap_graphdef, params, inters, other)
                out = m(phys_conf)
                _, _, new_inters, _ = nnx.split(
                    m, nnx.Param, nnx.Intermediate, ...,
                )
                return out.log, new_inters

            def _capture_one(params, inters, other, elec):
                (_lp, new_inters) = _apply_with_capture(
                    params, inters, other, elec,
                )
                return new_inters

            self._cap_grad_fn = jax.jit(jax.vmap(
                _capture_one, in_axes=(None, None, None, 0),
            ))

        # --- Per-walker gradient of log|psi| wrt the param pytree ---
        def _log_psi_of_params(params, elec):
            return log_psi(elec, nuc_crds, params)

        per_walker_grad = jax.vmap(
            jax.grad(_log_psi_of_params, argnums=0),
            in_axes=(None, 0),
        )

        layer_paths = dict(self._layers)
        generic_paths = list(self._generic_paths)

        # --- Chunked factor accumulation ---
        # A ``lax.scan`` over walker chunks, so the whole accumulation
        # is a single dispatch and XLA never unrolls it: peak memory
        # stays at ``(chunk, n_params)`` for ``pw_grad`` while the
        # carried factors are only ``(in, in)`` / ``(out, out)`` per
        # layer and do not grow with the walker count.  Doing this
        # chunking in Python instead costs ~4 dispatches per layer per
        # chunk (>1300 tiny device ops per iteration here), which
        # dominated the step time.
        def factor_chunk(params, w_chunk, de_chunk, captured_chunk):
            pw_grad = per_walker_grad(params, w_chunk)
            A_sums: Dict[str, jax.Array] = {}
            G_sums: Dict[str, jax.Array] = {}
            dW_sums: Dict[str, jax.Array] = {}
            counts: Dict[str, jax.Array] = {}
            for layer, (kpath, _bpath) in layer_paths.items():
                M = _get_at_path(pw_grad, kpath)      # (w, in, out)
                if layer in captured_chunk:
                    A_s, G_s, dW_s, n_s, _n_e = (
                        _per_electron_factor_sums(
                            captured_chunk[layer], M, de_chunk,
                        )
                    )
                else:
                    A_s, G_s, dW_s, n_s = _kron_factor_sums(
                        M, de_chunk,
                    )
                A_sums[layer] = A_s
                G_sums[layer] = G_s
                dW_sums[layer] = dW_s
                counts[layer] = jnp.asarray(
                    n_s, dtype=jnp.float64,
                )
            generic_flat = (
                jnp.concatenate(
                    [
                        _get_at_path(pw_grad, p).reshape(
                            w_chunk.shape[0], -1,
                        )
                        for p in generic_paths
                    ],
                    axis=1,
                )
                if generic_paths
                else jnp.zeros((w_chunk.shape[0], 0))
            )
            return A_sums, G_sums, dW_sums, counts, generic_flat

        self._factor_chunk = jax.jit(factor_chunk)

        capture_fn = None
        if self.capture_activations:
            cap_inters_init = self._cap_inters_init
            cap_other = self._cap_other
            captures_to_path = self._captures_to_path_dict

            def capture_fn(params, w_chunk):        # noqa: F811
                ints = jax.vmap(
                    _capture_one, in_axes=(None, None, None, 0),
                )(params, cap_inters_init, cap_other, w_chunk)
                return {
                    k: v
                    for k, v in captures_to_path(ints).items()
                    if v.ndim >= 2 and k in layer_paths
                }

        def accumulate_factors(params, walkers_r, de_r):
            """Scan the factor sums over walker chunks.

            Args:
                walkers_r: ``(n_chunks, chunk, nelec, 3)``.
                de_r: ``(n_chunks, chunk)``.

            Returns:
                ``(A_sum, G_sum, dW_sum, count, generic_flat)`` with
                ``generic_flat`` of shape ``(n_chunks * chunk, P_g)``.
            """
            def body(carry, xs):
                A_c, G_c, dW_c, cnt_c = carry
                w_c, de_c = xs
                cap_c = (
                    capture_fn(params, w_c)
                    if capture_fn is not None else {}
                )
                A_s, G_s, dW_s, n_s, gen_s = factor_chunk(
                    params, w_c, de_c, cap_c,
                )
                return (
                    {k: A_c[k] + A_s[k] for k in A_c},
                    {k: G_c[k] + G_s[k] for k in G_c},
                    {k: dW_c[k] + dW_s[k] for k in dW_c},
                    {k: cnt_c[k] + n_s[k] for k in cnt_c},
                ), gen_s

            zero = {
                layer: (
                    jnp.zeros((s[1], s[1]), dtype=jnp.float64),
                    jnp.zeros((s[0], s[0]), dtype=jnp.float64),
                    jnp.zeros(s, dtype=jnp.float64),
                    jnp.zeros((), dtype=jnp.float64),
                )
                for layer, s in self._kernel_shapes.items()
            }
            init = (
                {k: v[0] for k, v in zero.items()},
                {k: v[1] for k, v in zero.items()},
                {k: v[2] for k, v in zero.items()},
                {k: v[3] for k, v in zero.items()},
            )
            carry, gen = jax.lax.scan(
                body, init, (walkers_r, de_r),
            )
            A_c, G_c, dW_c, cnt_c = carry
            return (
                A_c, G_c, dW_c, cnt_c,
                gen.reshape(-1, gen.shape[-1]),
            )

        self._accumulate_factors = jax.jit(accumulate_factors)

        ema_decay_arr = jnp.asarray(ema_decay, dtype=jnp.float64)
        norm_clip = (
            None if norm_constraint is None
            else jnp.asarray(norm_constraint, dtype=jnp.float64)
        )
        generic_sizes = list(self._generic_sizes)

        # --- KFAC solve + parameter update ---
        def kfac_apply(params, A_hat, G_hat, dW_hat, generic_flat,
                       de_eff, A_state, G_state, lr_now,
                       damping_arr, kernel_scale):
            new_A: Dict[str, jax.Array] = {}
            new_G: Dict[str, jax.Array] = {}
            update_kernels: Dict[str, jax.Array] = {}
            layer_dW_loss: Dict[str, jax.Array] = {}

            for layer in layer_paths:
                A_new = (
                    ema_decay_arr * A_state[layer]
                    + (1.0 - ema_decay_arr) * A_hat[layer]
                )
                G_new = (
                    ema_decay_arr * G_state[layer]
                    + (1.0 - ema_decay_arr) * G_hat[layer]
                )
                new_A[layer] = A_new
                new_G[layer] = G_new
                A_inv, G_inv = _damped_kron_inverse(
                    A_new, G_new, damping_arr,
                )
                dW_loss = dW_hat[layer]
                # (in, in) @ (in, out) @ (out, out) -> (in, out)
                step = G_inv @ dW_loss @ A_inv
                update_kernels[layer] = kernel_scale[layer] * step
                layer_dW_loss[layer] = dW_loss

            if generic_flat.shape[1] > 0:
                generic_step, generic_grad = (
                    _generic_natural_gradient(
                        de_eff, generic_flat, damping_arr,
                    )
                )
            else:
                generic_step = jnp.zeros(0)
                generic_grad = jnp.zeros(0)

            # Quadratic-model terms for LM damping and the trust
            # region.  Under the damped relation
            # ``(F + lambda I) raw_step = grad`` the Fisher quadratic
            # form reduces to ``kernel_scale * <grad, step>``.
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
            if generic_flat.shape[1] > 0:
                dot_gen = jnp.sum(generic_grad * generic_step)
                dot_grad_step = dot_grad_step + dot_gen
                dot_step_F_step = dot_step_F_step + dot_gen

            kernel_sq = sum(
                jnp.sum(s ** 2) for s in update_kernels.values()
            )
            generic_sq = jnp.sum(generic_step ** 2)
            total_norm = jnp.sqrt(kernel_sq + generic_sq + 1e-30)

            # Trust region in the *Fisher* norm, as kfac-jax /
            # FermiNet / DeepQMC do: bound
            # ``dtheta^T F dtheta = lr^2 c^2 <step, F step>`` by
            # ``norm_constraint``.  Clipping the Euclidean norm
            # instead (as the HEG driver does) is not scale-
            # meaningful here, because the M^T M factors carry an
            # ``E[|a|^2] E[|g|^2]`` constant that the learning rate
            # would otherwise absorb; with a Euclidean bound that
            # constant shrinks every step by the same factor and the
            # optimiser stalls.
            if norm_clip is not None:
                fisher_sq = jnp.maximum(
                    lr_now ** 2 * dot_step_F_step, 1e-30,
                )
                clip = jnp.minimum(
                    1.0, jnp.sqrt(norm_clip / fisher_sq),
                )
            else:
                clip = jnp.asarray(1.0, dtype=jnp.float64)

            # Each leaf keeps the dtype it was initialised with.  NNX
            # builds the network in float32 while OmegaQMC.config
            # turns on jax_enable_x64, so the KFAC factors, inverses
            # and step are float64 (where conditioning matters) but
            # writing a float64 result back into the params would
            # silently promote the whole network.  On a GeForce card,
            # where FP64 runs at a small fraction of FP32, that
            # measured as an ~11x slowdown of the per-walker gradient
            # — the dominant cost of an iteration.
            scale = -lr_now * clip
            new_params = params
            for layer, step in update_kernels.items():
                kpath, _ = layer_paths[layer]
                old_W = _get_at_path(new_params, kpath)
                new_params = _set_at_path(
                    new_params, kpath,
                    (old_W + scale * step).astype(old_W.dtype),
                )
            offset = 0
            for path, size in zip(generic_paths, generic_sizes):
                slice_ = generic_step[offset:offset + size]
                offset += size
                old_p = _get_at_path(params, path)
                upd = old_p + scale * slice_.reshape(old_p.shape)
                new_params = _set_at_path(
                    new_params, path, upd.astype(old_p.dtype),
                )

            return (new_params, new_A, new_G, total_norm, clip,
                    dot_grad_step, dot_step_F_step)

        self._kfac_apply = jax.jit(kfac_apply)

    # -----------------------------------------------------
    # Walker management
    # -----------------------------------------------------

    def initialize_walkers(self, rng_key, num_walkers):
        """Place electrons near nuclei.

        Args:
            rng_key: JAX PRNG key.
            num_walkers: Number of walkers.

        Returns:
            Array ``(num_walkers, nelec, 3)``.
        """
        idx_cnt = []
        for ia, iz in enumerate(self.charges):
            idx_cnt.extend([ia] * int(iz))
        total = self.mol_info.n_up + self.mol_info.n_down
        while len(idx_cnt) < total:
            idx_cnt.append(0)
        idx_cnt = jnp.array(idx_cnt[:total])
        centers = self.nuc_crds[idx_cnt]
        return (
            centers[None, :, :]
            + 0.05 * jax.random.normal(
                rng_key, (num_walkers, self.nelec, 3),
            )
        )

    def _init_factor_state(self):
        """Identity-initialise the EMA Kronecker factors.

        ``kernel_shapes`` holds NNX's ``(in, out)`` kernel shape, so
        the output-side factor is ``(out, out)`` and the input-side
        one ``(in, in)``.
        """
        A_state = {}
        G_state = {}
        for layer, (in_, out_) in self._kernel_shapes.items():
            A_state[layer] = jnp.eye(out_, dtype=jnp.float64)
            G_state[layer] = jnp.eye(in_, dtype=jnp.float64)
        return A_state, G_state

    def _capture_for(self, params, walkers):
        """Per-layer captured Linear inputs for a walker chunk."""
        if self._cap_grad_fn is None:
            return {}
        ints_W = self._cap_grad_fn(
            params, self._cap_inters_init, self._cap_other, walkers,
        )
        out = {}
        for path, arr in self._captures_to_path_dict(ints_W).items():
            # Layers that never fired still hold the scalar default.
            if arr.ndim < 2:
                continue
            out[path] = arr
        return out

    # -----------------------------------------------------
    # Training loop
    # -----------------------------------------------------

    def __call__(
        self,
        rng_key,
        num_iters: int = 5000,
        num_walkers: int = 1000,
        factor_chunk_size: int = 256,
        num_steps_decorr: int = 10,
        num_blocks_equil: int = 5,
        num_steps_per_block: int = 200,
        mc_timestep: float = 0.1,
        fname_log: Optional[str] = None,
        verbose: int = 1,
        prefix: str = 'nnopt',
    ):
        """Run KFAC natural-gradient VMC optimisation.

        After every ``_CHK_EVERY_KFAC`` iterations (and at the last
        one) the parameters are written to ``{prefix}.chk.h5``, with
        the previous file preserved as ``{prefix}.{iter}.h5``.

        Args:
            rng_key: JAX PRNG key (int or key array).
            num_iters: KFAC iterations (one parameter update each).
            num_walkers: Number of MC walkers.
            factor_chunk_size: Walkers per Kronecker-factor
                accumulation chunk.  Peak gradient memory is
                ``factor_chunk_size * n_params``; the factors
                themselves do not depend on ``num_walkers``, so this
                is what decouples the walker count from the
                parameter count.
            num_steps_decorr: MCMC decorrelation steps per iteration.
            num_blocks_equil: Equilibration blocks (initial only).
            num_steps_per_block: MC steps per equilibration block.
            mc_timestep: Initial MC timestep.
            fname_log: Path for the plain-text per-iteration log;
                ``None`` or ``""`` writes to stdout.
            verbose: Verbosity (0 = silent).
            prefix: Filename prefix for the HDF5 checkpoint.

        Returns:
            Tuple ``(params_final, energy_data)`` where
            *energy_data* has keys ``'energy'``, ``'E_history'`` and
            ``'Var_history'``.
        """
        if isinstance(rng_key, int):
            rng_key = jax.random.key(rng_key)
        if (fname_log is None
                or (isinstance(fname_log, str) and fname_log == "")):
            fout = sys.stdout
        else:
            fout = open(fname_log, 'w', 1)

        params = self.init_params
        start_iter = 0

        # --- Checkpoint resume ---
        chkpt_path = f"{prefix}.chk.h5"
        if os.path.exists(chkpt_path):
            template_leaves = jax.tree.leaves(params)
            n_model = len(template_leaves)
            try:
                with h5py.File(chkpt_path, 'r') as f:
                    n_chk = int(f['params'].attrs['num_leaves'])
                    if n_chk != n_model:
                        print(f"Error: checkpoint '{chkpt_path}'"
                              f" has {n_chk} parameter leaves"
                              f" but current model has {n_model}."
                              " Incompatible architecture —"
                              " stopping.", file=fout)
                        return None, {}
                    for i, leaf in enumerate(template_leaves):
                        chk_shape = f['params'][str(i)].shape
                        if chk_shape != leaf.shape:
                            print(f"Error: parameter leaf {i} shape"
                                  f" mismatch: checkpoint {chk_shape}"
                                  f" vs model {leaf.shape}."
                                  " Incompatible architecture —"
                                  " stopping.", file=fout)
                            return None, {}
            except (KeyError, OSError) as exc:
                print(f"Error reading checkpoint '{chkpt_path}':"
                      f" {exc} — stopping.", file=fout)
                return None, {}
            params, meta = load_nn_checkpoint(chkpt_path, params)
            start_iter = int(meta.get('epoch', -1)) + 1
            if verbose >= 1:
                print(f"Resuming from '{chkpt_path}' (iteration"
                      f" {start_iter - 1} completed, continuing from"
                      f" iteration {start_iter})", file=fout)

        # The factor accumulation is a scan over equal-sized chunks,
        # so round the request down to the largest divisor of
        # ``num_walkers`` rather than leaving a ragged tail.
        chunk = max(1, min(int(factor_chunk_size), num_walkers))
        while num_walkers % chunk != 0:
            chunk -= 1
        n_chunks = num_walkers // chunk
        if chunk != factor_chunk_size and verbose >= 1:
            print(f"# factor_chunk_size {factor_chunk_size} ->"
                  f" {chunk} (must divide num_walkers"
                  f" {num_walkers})", file=fout)

        # --- Initialise and equilibrate walkers ---
        rng_key, init_key = jax.random.split(rng_key)
        walkers = self.initialize_walkers(init_key, num_walkers)
        step_size = jnp.asarray((3 * mc_timestep) ** 0.5)

        ars = None
        for _ in range(num_blocks_equil):
            (rng_key, walkers, step_size, _), ars = self.decorr_scan(
                rng_key, walkers, step_size, params,
                num_steps_per_block,
            )

        if verbose >= 1:
            print(
                f"# KFAC-VMC — {self.n_params} params total"
                f" ({self._linear_params} in {self._linear_count}"
                f" Linear layers, rest generic)", file=fout,
            )
            print(
                f"# lr={self.lr}, lr_decay={self.lr_decay},"
                f" damping={self.damping}, ema_decay={self.ema_decay},"
                f" norm_constraint={self.norm_constraint},"
                f" beta={self.var_weight:g}", file=fout,
            )
            print(
                f"# num_walkers={num_walkers},"
                f" factor_chunk_size={chunk},"
                f" capture_activations={self.capture_activations}",
                file=fout,
            )
            if ars is not None:
                print(
                    f"# Equilibration acceptance:"
                    f" {float(ars[-1]):.3f}, step size:"
                    f" {float(step_size):.4f} bohr", file=fout,
                )
            print(
                "#  iter              <E>          Var(E)"
                "          lr      damping       |step|        dt",
                file=fout,
            )

        A_state, G_state = self._init_factor_state()

        # Per-layer step scale.  Only the exact per-electron path with
        # ``fixed_scale`` uses anything other than 1: there the factor
        # sums run over ``W * n_e`` samples while ``dW_loss`` is a
        # per-walker mean, so the step needs an ``n_e`` multiplier to
        # match textbook KFAC magnitude (FermiNet's ``fixed_scale``).
        # ``n_e`` is fixed by the architecture, so probe it once.
        kernel_scale: Dict[str, float] = {
            layer: 1.0 for layer in self._kernel_shapes
        }
        if self.fixed_scale and self.capture_activations:
            for layer, arr in self._capture_for(
                params, walkers[:1],
            ).items():
                if layer in kernel_scale:
                    kernel_scale[layer] = float(
                        1 if arr.ndim == 2
                        else np.prod(arr.shape[1:-1])
                    )

        e_history: List[float] = []
        var_history: List[float] = []
        timestamp_prev = datetime.now()
        damping_now = self.damping
        step_norm_window: List[float] = []
        ratio_window: List[Tuple[float, float]] = []
        prev_e_total: Optional[float] = None
        prev_predicted: Optional[float] = None
        e_mean = None

        for it in range(start_iter, start_iter + num_iters):
            # (a) Decorrelate under the current |psi|^2.
            (rng_key, walkers, step_size, _), _ars = self.decorr_scan(
                rng_key, walkers, step_size, params,
                num_steps_decorr,
            )

            # (b) Local energies (forward-Laplacian, chunked).
            e_loc = self.compute_batch_energy(walkers, params)
            e_mean = jnp.mean(e_loc)
            de = e_loc - e_mean
            var = jnp.mean(de ** 2)
            de_eff = de + self.var_weight * (de ** 2 - var)

            # (c) Accumulate Kronecker factors, scanned over walker
            # chunks inside a single jitted call.
            A_acc, G_acc, dW_acc, cnt_acc, generic_flat = (
                self._accumulate_factors(
                    params,
                    walkers.reshape(
                        (n_chunks, chunk) + walkers.shape[1:],
                    ),
                    de_eff.reshape(n_chunks, chunk),
                )
            )

            # Normalise: factor sums by their sample population,
            # dW_loss always by the walker count.
            A_hat = {k: A_acc[k] / cnt_acc[k] for k in A_acc}
            G_hat = {k: G_acc[k] / cnt_acc[k] for k in G_acc}
            dW_hat = {
                k: dW_acc[k] / float(num_walkers) for k in dW_acc
            }

            # (d) Solve and apply.  The schedule is keyed on the
            # absolute iteration so a resumed run continues decaying
            # rather than restarting at the initial rate.
            if self.lr_decay is not None:
                lr_now = self.lr / (1.0 + it / self.lr_decay)
            else:
                lr_now = self.lr

            (params, A_state, G_state, step_norm, clip_val,
             dot_grad_step, dot_step_F_step) = self._kfac_apply(
                params, A_hat, G_hat, dW_hat, generic_flat, de_eff,
                A_state, G_state,
                jnp.asarray(lr_now, dtype=jnp.float64),
                jnp.asarray(damping_now, dtype=jnp.float64),
                kernel_scale,
            )

            # (e) Logging.
            now = datetime.now()
            dt = (now - timestamp_prev).total_seconds()
            timestamp_prev = now
            e_total = float(e_mean)
            e_history.append(e_total)
            var_history.append(float(var))
            if verbose >= 1 and (it - start_iter < 10 or it % 10 == 0):
                print(
                    f"{it:>7d}  {e_total:>15.8f}  {float(var):>13.5e}"
                    f"  {lr_now:>10.4e}  {damping_now:>10.4e}"
                    f"  {float(step_norm):>11.4e}  {dt:>8.3f}",
                    file=fout,
                )

            # (f) Adaptive Levenberg-Marquardt damping.
            eff_lr = float(lr_now) * float(clip_val)
            predicted_change = float(
                0.5 * eff_lr ** 2 * float(dot_step_F_step)
                - eff_lr * float(dot_grad_step)
            )
            if self.damping_adapt:
                step_norm_val = float(step_norm)
                step_norm_window.append(step_norm_val)
                if len(step_norm_window) > self.damping_lookback:
                    step_norm_window.pop(0)
                # Panic mode: a step norm far above its running
                # median means the trust region has been breached;
                # tighten immediately rather than waiting for the
                # ratio window to fill.
                if len(step_norm_window) >= self.damping_lookback:
                    median_norm = float(
                        np.median(step_norm_window[:-1])
                    )
                    if (median_norm > 0 and step_norm_val
                            > self.damping_overshoot_threshold
                            * median_norm):
                        damping_now = float(np.clip(
                            damping_now
                            * self.damping_overshoot_factor,
                            self.damping_min, self.damping_max,
                        ))
                # Martens-Grosse ratio: the energy measured at
                # iteration t reflects the step taken at t-1, so pair
                # this iteration's change with the previous
                # iteration's prediction.
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
            prev_e_total = e_total
            prev_predicted = predicted_change

            # (g) Checkpoint.
            is_last = (it == start_iter + num_iters - 1)
            if is_last or ((it + 1) % _CHK_EVERY_KFAC == 0):
                if os.path.exists(chkpt_path):
                    os.rename(chkpt_path, f"{prefix}.{it}.h5")
                save_nn_checkpoint(
                    chkpt_path, params, it, self.config_name,
                    self.mol_info, energy=e_total,
                )

        # --- Final energy estimate ---
        for _ in range(num_blocks_equil):
            (rng_key, walkers, step_size, _), _ = self.decorr_scan(
                rng_key, walkers, step_size, params,
                num_steps_per_block,
            )
        e_loc = self.compute_batch_energy(walkers, params)
        final_e = float(jnp.mean(e_loc))
        final_std = float(jnp.std(e_loc))
        final_err = final_std / max(1, e_loc.size) ** 0.5

        if verbose >= 1:
            print(
                f"Final energy: {final_e:.8f} +/- {final_err:.8f}"
                f"  (sigma(E_L) = {final_std:.4e})", file=fout,
            )
        if fout is not sys.stdout:
            fout.close()

        self.params = params
        return params, {
            'energy': {'mean': final_e, 'stderr': final_err},
            'E_history': e_history,
            'Var_history': var_history,
        }


def get_vmcopt_nn_func(mol_info, config, init_key, **kwargs):
    """Create a KFAC natural-gradient VMC optimiser for NN trials.

    Builds the NN trial wavefunction from *config*, compiles the
    Metropolis kernel, the forward-Laplacian local energy and the
    KFAC step, and returns a callable driver.

    Args:
        mol_info: :class:`~OmegaQMC.utils.Mole_custom` instance.
        config: :class:`~OmegaQMC.psi.nn.config.NNAnsatzConfig` or a
            string (built-in name or YAML path).
        init_key: JAX PRNG key for parameter initialisation.
        **kwargs: Forwarded to :class:`_VMCOptDriverNN_KFAC` (``lr``,
            ``damping``, ``ema_decay``, ``norm_constraint``,
            ``capture_activations``, ...).

    Returns:
        :class:`_VMCOptDriverNN_KFAC` instance.  Call it with
        ``driver(rng_key, ...)`` to run the optimisation.
    """
    return _VMCOptDriverNN_KFAC(
        mol_info, config, init_key, **kwargs,
    )
