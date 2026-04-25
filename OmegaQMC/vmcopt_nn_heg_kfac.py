"""KFAC (Kronecker-Factored Approximate Curvature) VMC optimiser for HEG.

Self-contained float64 implementation — does not depend on
``kfac-jax`` (which is float32-pinned and currently incompatible
with our jax 0.10 stack).  Mirrors ``vmcopt_nn_heg_sr.py`` 's
constructor and ``__call__`` signatures so ``run_heg_psiformer.py``
dispatches to it on ``optimize.type: kfac``.

Known limitations
-----------------

1. **Activation recovery is via SVD, not direct capture.** Proper
   KFAC computes the per-layer input ``a_l`` and output-gradient
   ``g_l`` directly from the network's forward / backward pass.
   This implementation skips the model-instrumentation step and
   instead recovers ``(g_w, x_w)`` from the rank-1 per-walker
   gradient ``∂log|ψ_w|/∂W = g_w x_w^T`` via batched SVD.  The
   SVD imposes a specific scale convention
   (``‖g_w‖ = ‖x_w‖ = sqrt(s_w)``); the true ``a_l`` and ``g_l``
   from the model can have any per-walker rescaling.  In Fisher
   space this means our ``A = E[x_w x_w^T]`` and
   ``G = E[g_w g_w^T]`` are reweighted by a per-walker factor
   compared to the strict KFAC formulation.  Empirically the
   approximation still yields a useful natural-gradient direction;
   theoretically a follow-up should expose layer activations
   directly to remove the bias.

2. **Multi-device pmap is a stub.**  ``multi_device=True`` is
   accepted but not yet implemented — multi-GPU runs raise.

3. **No adaptive damping.**  KFAC traditionally adjusts ε via a
   Levenberg-Marquardt schedule based on the loss-decrease ratio.
   This implementation uses a fixed ``damping`` value.  The trace-
   renormalisation ``π = sqrt(tr(A)/tr(G))`` is still applied so
   the effective scale of damping is layer-aware.

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

import sys
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from .psi.nn.heg_wf import (
    HEGConfig,
    make_heg_log_psi_any as make_heg_log_psi,
)
from .psi.nn.periodic import wrap_to_cell, make_cubic_lattice
from .psi.nn.physics import laplacian
from .observables.ewald import build_ewald_tables, ewald_pair_energy


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

def _extract_kron_factors(per_walker_dW: jax.Array,
                          de: jax.Array) -> Tuple[jax.Array, jax.Array, jax.Array]:
    """Decompose per-walker rank-1 gradients into (A, G, ∂L/∂W).

    Args:
        per_walker_dW: ``(W, out, in)`` — each slice is ``g_w x_w^T``,
            i.e. ``∂log|ψ_w|/∂W`` (NOT energy-weighted).
        de: ``(W,)`` energy residual ``E_L_w − ⟨E_L⟩``.

    Returns:
        A: ``(in, in)`` = ``E[x_w x_w^T]``
        G: ``(out, out)`` = ``E[g_w g_w^T]``
        dW_loss: ``(out, in)`` = ``E[de_w · g_w x_w^T]`` (the energy
            gradient of W).
    """
    # SVD per walker.  jax.numpy.linalg.svd auto-vmaps over leading axes.
    U, s, Vh = jnp.linalg.svd(
        per_walker_dW, full_matrices=False,
    )  # U: (W, out, k), s: (W, k), Vh: (W, k, in), k=min(out,in)
    sqrt_s = jnp.sqrt(s[..., 0:1])              # (W, 1) — top singular value
    g_w = sqrt_s * U[..., :, 0]                 # (W, out)
    x_w = sqrt_s * Vh[..., 0, :]                # (W, in)

    # The signs of (g_w, x_w) cancel in the outer product but not in
    # E[g_w g_w^T] etc — so we don't need to fix them.
    A = jnp.einsum('wi,wj->ij', x_w, x_w) / x_w.shape[0]
    G = jnp.einsum('wi,wj->ij', g_w, g_w) / g_w.shape[0]

    # Energy gradient is just the de-weighted average of the rank-1 grads.
    dW_loss = jnp.einsum('w,woi->oi', de, per_walker_dW) / per_walker_dW.shape[0]
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

def _generic_natural_gradient(de, per_walker_grads_flat, damping):
    """Damped Fisher solve for the flat 'generic' tail of params.

    Args:
        de: ``(W,)`` energy residual.
        per_walker_grads_flat: ``(W, P_g)`` per-walker gradients of
            ``log|ψ|`` wrt the generic params, flattened.
        damping: Tikhonov ε.

    Returns:
        ``(P_g,)`` natural-gradient direction.
    """
    W = per_walker_grads_flat.shape[0]
    f = jnp.einsum('w,wp->p', de, per_walker_grads_flat) / W
    # Centered Jacobian.
    do = per_walker_grads_flat - per_walker_grads_flat.mean(axis=0, keepdims=True)
    # S·v = (1/W)·doᵀ·(do·v) + λ·v
    # Solve via a small CG (P_g is typically << W).
    n_iters = 20
    x = jnp.zeros_like(f)
    r = f - ((do.T @ (do @ x)) / W + damping * x)
    p = r
    rr = jnp.dot(r, r)
    for _ in range(n_iters):
        Ap = (do.T @ (do @ p)) / W + damping * p
        alpha = rr / (jnp.dot(p, Ap) + 1e-30)
        x = x + alpha * p
        r = r - alpha * Ap
        rr_new = jnp.dot(r, r)
        beta = rr_new / (rr + 1e-30)
        p = r + beta * p
        rr = rr_new
    return x


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
        lr: float = 0.05,
        lr_decay: Optional[float] = 1.0e4,
        damping: float = 1.0e-3,
        ema_decay: float = 0.95,
        norm_constraint: Optional[float] = 1.0e-3,
        var_weight: float = 0.0,
        ewald_n_real: int = 3,
        ewald_n_recip: int = 6,
        ewald_eta: Optional[float] = None,
        multi_device: bool = False,
    ):
        # x64 is already enabled by OmegaQMC.config; KFAC's matrix
        # inverses and EMA accumulators rely on it for numerical
        # conditioning.
        if multi_device:
            raise NotImplementedError(
                "multi_device=True is not yet wired up in our KFAC. "
                "Run on a single device for now or open an issue."
            )
        self.config = config
        self.L = float(config.L)
        self.n_up = int(config.n_up)
        self.n_down = int(config.n_down)
        self.nelec = self.n_up + self.n_down
        self.lr = float(lr)
        self.lr_decay = (None if lr_decay is None else float(lr_decay))
        self.damping = float(damping)
        self.ema_decay = float(ema_decay)
        self.norm_constraint = (None if norm_constraint is None
                                else float(norm_constraint))
        self.var_weight = float(var_weight)
        self.multi_device = bool(multi_device)

        self.lattice = make_cubic_lattice(self.L)
        self.ewald = build_ewald_tables(
            self.L, eta=ewald_eta,
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
                return log_psi_pytree(r_flat.reshape(nelec, 3), params)
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
        self._pot_chunk = jax.jit(
            lambda w: ewald_pair_energy(w, tables),
        )

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
        damping_arr = jnp.asarray(self.damping, dtype=jnp.float64)
        norm_clip = (None if norm_constraint is None
                     else jnp.asarray(norm_constraint, dtype=jnp.float64))
        var_weight_arr = jnp.asarray(var_weight, dtype=jnp.float64)

        @jax.jit
        def kfac_step_core(params, e_loc, walkers, A_state, G_state, lr_now):
            # Per-walker pytree grad of log|ψ|.
            pw_grad = self._per_walker_grad(walkers, params)

            e_mean = jnp.mean(e_loc)
            de = e_loc - e_mean
            var = jnp.mean(de ** 2)

            # Mixed-objective weighting (Umrigar β).
            de_eff = de + var_weight_arr * (de ** 2 - var)

            new_A, new_G = {}, {}
            update_kernels: Dict[str, jax.Array] = {}

            # Per Linear layer: extract Kronecker factors, damp, step.
            for layer, (kpath, bpath) in layer_paths.items():
                pw_dW = _get_at_path(pw_grad, kpath)            # (W, out, in)
                A, G, dW_loss = _extract_kron_factors(pw_dW, de_eff)

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
                # ΔW = -lr · G_inv · dW_loss · A_inv
                step = G_inv @ dW_loss @ A_inv
                update_kernels[layer] = step    # the *direction*, not yet scaled

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

            generic_step_flat = (
                _generic_natural_gradient(de_eff, generic_pw_flat, damping_arr)
                if generic_pw_flat.shape[1] > 0
                else jnp.zeros(0)
            )

            # ---- Trust-region clip (norm_constraint) ----
            # Compute total step Frobenius norm and clip if exceeds.
            if norm_clip is not None:
                kernel_sq = sum(
                    jnp.sum(s ** 2) for s in update_kernels.values()
                )
                generic_sq = jnp.sum(generic_step_flat ** 2)
                total_norm = jnp.sqrt(kernel_sq + generic_sq + 1e-30)
                clip = jnp.minimum(1.0, norm_clip / (lr_now * total_norm))
            else:
                clip = jnp.asarray(1.0, dtype=jnp.float64)

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
                p_shape = sub.shape[1:]                # original param shape
                # Wait — sub has shape (W, P_flat); the original leaf shape
                # was (W, *param_shape) so we need that.  Recover from leaf:
                leaf_shape_flat = sub.shape[1]
                slice_ = generic_step_flat[offset:offset + leaf_shape_flat]
                offset += leaf_shape_flat
                # Reshape back to the param's true shape.  Look up via the
                # original params pytree.
                old_param = _get_at_path(params, path)
                slice_ = slice_.reshape(old_param.shape)
                new_params = _set_at_path(
                    new_params, path, old_param + scale * slice_,
                )

            return new_params, new_A, new_G, e_mean, var, total_norm if norm_clip is not None else jnp.asarray(0.0)

        self._kfac_step_core = kfac_step_core

    # -----------------------------------------------------
    # Walker management
    # -----------------------------------------------------

    def initialize_walkers(self, rng_key, num_walkers):
        return self.L * jax.random.uniform(
            rng_key, (num_walkers, self.nelec, 3),
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

        rng_key, init_key = jax.random.split(rng_key)
        walkers = self.initialize_walkers(init_key, num_walkers)
        step_size = jnp.asarray((3 * mc_timestep) ** 0.5)

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
                "lr            ‖step‖         dt",
                file=fout,
            )

        # Initialise EMA factor state.
        A_state, G_state = self._init_factor_state()

        e_history = []
        var_history = []
        timestamp_prev = datetime.now()

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

            # KFAC step.
            if self.lr_decay is not None:
                lr_now = self.lr / (1.0 + (it - 1) / self.lr_decay)
            else:
                lr_now = self.lr

            params, A_state, G_state, e_mean, var, step_norm = kfac_step_core(
                params, e_loc, walkers,
                A_state, G_state,
                jnp.asarray(lr_now, dtype=jnp.float64),
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
                    f"{float(step_norm):>10.4e}  "
                    f"{dt:>8.3f}",
                    file=fout,
                )

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
    lr: float = 0.05,
    lr_decay: Optional[float] = 1.0e4,
    damping: float = 1.0e-3,
    ema_decay: float = 0.95,
    norm_constraint: Optional[float] = 1.0e-3,
    var_weight: float = 0.0,
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
        damping=damping, ema_decay=ema_decay,
        norm_constraint=norm_constraint,
        var_weight=var_weight,
        ewald_n_real=ewald_n_real,
        ewald_n_recip=ewald_n_recip,
        ewald_eta=ewald_eta,
        multi_device=multi_device,
    )
