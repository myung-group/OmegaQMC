"""KFAC (Kronecker-Factored Approximate Curvature) VMC optimiser
for molecular NN wavefunctions.

The optimiser reproduces ``kfac_jax.Optimizer`` as DeepMind's FermiNet
configures it (ferminet ``train.py``, kfac-jax 0.0.8): exact
per-(walker, row) Kronecker factors for every Linear layer with the
bias in the block, kfac-jax's ``fixed_scale``, Fisher factor, bias-
corrected moving average, pi-adjusted per-block damping and norm
constraint, and uncentred diagonal blocks for the remaining leaves
(the envelope).  See :meth:`_VMCOptDriverNN_KFAC._setup_kfac`.  With
the ``ferminet_deepmind`` ansatz the steps agree with DeepMind's
ferminet + kfac-jax to ~1e-7 relative
(``tests/test_vmcopt_nn_kfac.py``).  The Hamiltonian, sampler and
checkpointing follow :mod:`OmegaQMC.vmcopt_nn_sr`.

Every Linear's input rows and output cotangents come exactly from
:mod:`OmegaQMC.psi.nn.kfac_tap` (gradients with respect to zero
perturbations of the Linear outputs), so the method needs no
model-specific code: any ansatz built from ``nnx.Linear`` works.
OmegaQMC's earlier KFAC (``kfac_style='omegaqmc'``: identity-
initialised factors, biases and envelope in a centred CG block, no
Fisher factor, with a hand-written PsiFormer backward pass, walker-
level ``M^T M`` factors or a capturing twin) trained 7-11 mHa above
DeepMind's FermiNet on N2 and CO and was no faster, and has been
removed; the HEG driver :mod:`OmegaQMC.vmcopt_nn_heg_kfac` keeps its
own implementation.

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

Implementation notes
--------------------
1. **Walker-chunked factor accumulation.**  The factor sums are a
   ``lax.scan`` over chunks of ``factor_chunk_size`` walkers, so peak
   memory is set by the chunk, not by ``num_walkers``.  Each chunk's
   products are formed in the network's dtype at JAX's default matmul
   precision (as kfac-jax does) and summed over chunks in float64.
2. **Dtype-preserving parameter update.**  NNX initialises the
   network in float32 while ``OmegaQMC.config`` enables
   ``jax_enable_x64``, so writing the float64 KFAC step straight
   back into the parameters would silently promote the whole network
   to float64.  The factors, inverses and step stay in float64
   (conditioning), but each leaf is cast back to its own dtype on
   application.
3. **Exact resume.**  The factor state, damping, walkers, step size
   and PRNG key are checkpointed next to the parameters, so a resumed
   run continues with the iterates of an uninterrupted one.

Not implemented here: multi-device ``pmap``.
"""

from __future__ import annotations

import os
import sys
from datetime import datetime
from functools import partial
from typing import Dict, List, Optional, Tuple

import h5py
import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from .constants import MIN_DIST_THRESHOLD
from .spin_layout import to_grouped
from .psi.nn.adapter import make_nn_log_psi
from .psi.nn.build import build_nn_wf
from .psi.nn.checkpoint import (
    load_nn_checkpoint,
    load_state_tree,
    save_nn_checkpoint,
    save_state_tree,
)
from .psi.nn.config import load_nn_config
from .psi.nn.types import PhysicalConfiguration


# Maximum walkers per forward-Laplacian kinetic-energy evaluation;
# see vmcopt_nn_iradam for the detailed rationale.
_KE_WALKER_CHUNK = 256

# Iterations between checkpoint writes.
_CHK_EVERY_KFAC = 500

# Non-finite guard (see ``_VMCOptDriverNN_KFAC.__call__``): give up after
# this many consecutive rejected steps, or when more than this fraction
# of the walkers has a non-finite local energy at once; record at most
# _NONFINITE_MAX_DUMPS events in ``{prefix}.nonfinite.h5``.
_NONFINITE_MAX_CONSECUTIVE = 20
_NONFINITE_MAX_FRACTION = 0.5
_NONFINITE_MAX_DUMPS = 50


@jax.jit
def _tree_all_finite(tree):
    """True when every leaf of *tree* is finite (one device reduction)."""
    leaves = [jnp.all(jnp.isfinite(x)) for x in jax.tree.leaves(tree)]
    if not leaves:
        return jnp.asarray(True)
    return jnp.all(jnp.stack(leaves))


def _dump_nonfinite(path, it, kind, walkers, e_loc, log_psi_vals,
                    grad_finite=None):
    """Append one non-finite event to the HDF5 file *path*."""
    with h5py.File(path, 'a') as f:
        n = len(f.keys())
        if n >= _NONFINITE_MAX_DUMPS:
            return
        g = f.create_group(f"{n:03d}_iter{it}_{kind}")
        g.attrs['iteration'] = int(it)
        g.attrs['kind'] = kind
        g.create_dataset('walkers', data=np.asarray(walkers))
        g.create_dataset('e_loc', data=np.asarray(e_loc))
        g.create_dataset('log_psi', data=np.asarray(log_psi_vals))
        if grad_finite is not None:
            g.create_dataset('grad_r_finite', data=np.asarray(grad_finite))

# Metropolis step-size adaptation (matches the other NN drivers).
_TARGET_ACCEPTANCE_RATE = 0.5
_STEP_SIZE_ADAPTATION_RATE = 0.05


def _adapt_step_size(step_size, acceptance_rate):
    """Adapt the Metropolis step size toward the target rate."""
    return step_size * (
        1.0 + _STEP_SIZE_ADAPTATION_RATE
        * (acceptance_rate - _TARGET_ACCEPTANCE_RATE)
    )


@partial(jax.jit, static_argnames=('clip_median', 'center_at_clip'))
def clip_local_energies(e_loc, clip_scale, clip_median=False,
                        center_at_clip=True):
    """Clip local energies for the energy gradient, as FermiNet does.

    Follows ``ferminet.loss.clip_local_values``.  With centre ``c``
    (the batch mean of *e_loc*, or its median if *clip_median*) and
    spread ``D = mean |E_L - c|`` (the mean absolute deviation, which
    is less sensitive to outliers than the standard deviation), each
    local energy is clipped to ``[c - clip_scale D, c + clip_scale D]``.
    The energy residual that weights ``d log|psi| / d theta`` in the
    gradient is then ``E_clip - <E_clip>`` with *center_at_clip*
    (FermiNet's default, and the PsiFormer paper's choice: the
    residuals sum to zero in every batch), or ``E_clip - <E_L>``
    without it (Pfau et al. 2020).

    FermiNet's defaults are ``clip_scale = 5``, ``clip_median=False``,
    ``center_at_clip=True``.  The PsiFormer paper (von Glehn et al.
    2023, Eq. A4) uses the median instead of the mean.  Only the
    gradient sees the clipped values; reported energies stay
    unclipped.

    Args:
        e_loc: ``(W,)`` local energies of one batch.
        clip_scale: Half-width of the window in units of ``D``
            (positive; the caller skips clipping for ``<= 0``).
        clip_median: Centre the window (and measure ``D``) at the
            median rather than the mean.
        center_at_clip: Subtract the mean of the clipped energies,
            rather than of the unclipped ones.

    Returns:
        ``(e_clip, de)``: the clipped local energies and the
        residuals used in the gradient.
    """
    e_mean = jnp.mean(e_loc)
    center = jnp.median(e_loc) if clip_median else e_mean
    spread = jnp.mean(jnp.abs(e_loc - center))
    e_clip = jnp.clip(e_loc, center - clip_scale * spread,
                      center + clip_scale * spread)
    de = e_clip - (jnp.mean(e_clip) if center_at_clip else e_mean)
    return e_clip, de


# ---------------------------------------------------------------------
# Kronecker blocks
# ---------------------------------------------------------------------

def _pi_adjusted_inverse(S_in, S_out, damping):
    """Inverses of kfac-jax's pi-adjusted damped Kronecker factors.

    ``kfac_jax.utils.pi_adjusted_kronecker_factors`` followed by an
    inversion: with ``n = trace / dim`` of each factor and
    ``c = sqrt(n_in n_out)``, the damped factors are
    ``c (S / n + sqrt(damping) / c I)``, whose Kronecker product
    approximates ``S_in (x) S_out + damping I``.  If either factor is
    zero both become ``sqrt(damping) I``.
    """
    d_in = S_in.shape[0]
    d_out = S_out.shape[0]
    n_in = jnp.trace(S_in) / d_in
    n_out = jnp.trace(S_out) / d_out
    zero = (n_in <= 0) | (n_out <= 0)
    n_in = jnp.where(zero, 1.0, n_in)
    n_out = jnp.where(zero, 1.0, n_out)
    c = jnp.sqrt(n_in * n_out)
    dk = jnp.sqrt(damping) / c
    eye_in = jnp.eye(d_in, dtype=S_in.dtype)
    eye_out = jnp.eye(d_out, dtype=S_out.dtype)
    F_in = jnp.where(zero, jnp.sqrt(damping) * eye_in,
                     (S_in / n_in + dk * eye_in) * c)
    F_out = jnp.where(zero, jnp.sqrt(damping) * eye_out,
                      (S_out / n_out + dk * eye_out) * c)
    return jnp.linalg.inv(F_in), jnp.linalg.inv(F_out)


# ---------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------

class _VMCOptDriverNN_KFAC:
    """KFAC natural-gradient VMC optimiser for molecular NN trials.

    Each iteration decorrelates the walkers under ``|psi_theta|^2``,
    evaluates the local energy with the forward-Laplacian,
    accumulates the Kronecker factors in walker chunks, and applies
    kfac-jax's damped natural-gradient step (Kronecker blocks for the
    Linear layers, diagonal blocks for the remaining leaves; see
    :meth:`_setup_kfac`).

    Args:
        mol_info: :class:`~OmegaQMC.utils.Mole_custom` instance.
        config: :class:`~OmegaQMC.psi.nn.config.NNAnsatzConfig` or a
            string (built-in name or YAML path).
        init_key: JAX PRNG key for parameter initialisation.
        lr: Initial learning rate.  Schedule is FermiNet's
            ``lr_t = lr / (1 + t / lr_decay)``.
        lr_decay: Decay constant; ``None`` for a constant rate.
        damping: Initial damping, added to every curvature block
            (kfac-jax's ``damping``).
        damping_adapt: Enable adaptive Levenberg-Marquardt damping.
            Off by default, matching DeepQMC's constant damping
            schedule (``conf/task/opt/kfac.yaml``): the ratio test
            compares the measured energy change with the quadratic
            model's prediction, and once sigma(E_L) ~ 2 Ha over 1000
            walkers the per-iteration noise (~0.07 Ha) exceeds the
            real improvement, so the ratio is mostly noise and damping
            random-walks upward.  On an H2O PsiFormer it inflated
            1e-3 -> 1.05 over 1200 iterations while ``|step|``
            collapsed 25x and the energy stopped improving.  The
            Fisher-norm trust region (``norm_constraint``) is what
            bounds the step.
        damping_min, damping_max: Bounds on the adapted damping.
        damping_lookback: Iterations aggregated before consulting
            the quadratic-model ratio, and the window used by the
            step-norm overshoot check.
        damping_decay: Multiplicative factor for the LM update.
        damping_overshoot_threshold, damping_overshoot_factor:
            Fast "panic mode" — if the step norm exceeds this
            multiple of its running median, damping is raised by
            this factor immediately.
        ema_decay: Decay of kfac-jax's bias-corrected moving average
            of the curvature across iterations (FermiNet uses 0.95;
            0 uses the current iteration's estimate alone).
        norm_constraint: kfac-jax's norm constraint: the step is
            scaled by ``min(1, sqrt(norm_constraint / (lr^2 <F^-1 g,
            g>)))``, a trust region in the Fisher norm; ``None``
            disables it.
        var_weight: Umrigar-style beta for a mixed
            ``<E> + beta Var(E_L)`` objective (0 = pure energy).
            With clipping on, the variance term is built from the
            clipped residuals.
        clip_local_energy: Clip the local energies entering the
            gradient to this many mean absolute deviations around
            the batch centre (FermiNet's ``optim.clip_local_energy``;
            FermiNet's configs use 5.0).  ``0`` (default) turns
            clipping off, as in earlier versions of this driver.
            The logged and returned energies are never clipped.  See
            :func:`clip_local_energies`.
        clip_median: Centre the clipping window at the median rather
            than the mean (FermiNet's ``optim.clip_median``, default
            off; the PsiFormer paper's choice).
        center_at_clip: Subtract the mean of the clipped energies in
            the gradient rather than the mean of the unclipped ones
            (FermiNet's ``optim.center_at_clip``, default on).
        kfac_style: Only ``'kfac_jax'`` (the default) is accepted;
            the argument is kept so that scripts written for the
            version with two styles still run.  ``'omegaqmc'``
            raises ``ValueError``.
        nn_dtype: Precision the network (log|psi|, its gradients
            and the forward Laplacian) is evaluated in.
            ``'float32'`` (default) matches DeepQMC and runs 20-40x
            faster on a GeForce GPU than the float64 that
            ``jax_enable_x64`` otherwise implies; potentials,
            energies and accumulators stay float64.  ``None``
            evaluates in the input precision (float64).  See
            :func:`~OmegaQMC.psi.nn.adapter.make_nn_log_psi`.
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
        damping_adapt: bool = False,
        damping_min: float = 1.0e-6,
        damping_max: float = 1.0e2,
        damping_lookback: int = 10,
        damping_decay: float = 0.95,
        damping_overshoot_threshold: float = 10.0,
        damping_overshoot_factor: float = 2.0,
        ema_decay: float = 0.95,
        norm_constraint: Optional[float] = 1.0e-3,
        var_weight: float = 0.0,
        clip_local_energy: float = 0.0,
        clip_median: bool = False,
        center_at_clip: bool = True,
        nn_dtype: Optional[str] = 'float32',
        kfac_style: str = 'kfac_jax',
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
        self.clip_local_energy = float(clip_local_energy)
        self.clip_median = bool(clip_median)
        self.center_at_clip = bool(center_at_clip)
        if kfac_style != 'kfac_jax':
            raise ValueError(
                "kfac_style must be 'kfac_jax' (OmegaQMC's former"
                f" 'omegaqmc' KFAC has been removed), got {kfac_style!r}"
            )
        self.kfac_style = kfac_style

        log_psi, init_params, graphdef, lap_grad = make_nn_log_psi(
            config, mol_info, init_key, compute_dtype=nn_dtype,
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
        # ``metropolis_move_lp`` takes and returns ``log|psi|`` of the
        # current configuration, so a move costs one network
        # evaluation instead of two (DeepQMC keeps it in the sampler
        # state the same way).  The cache is only valid for the
        # parameters it was computed with, so ``decorr_scan``
        # re-evaluates it once on entry.
        def metropolis_move_lp(rng_key, elec_crds, lp_old, step_size,
                               params):
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
            lp_new = log_psi(proposed, nuc_crds, params)
            accept = (
                jax.random.uniform(key_accept)
                < jnp.exp(2 * (lp_new - lp_old))
            ) & valid
            return (
                jnp.where(accept, proposed, elec_crds),
                jnp.where(accept, lp_new, lp_old),
                accept,
            )

        @jax.jit
        def metropolis_move(rng_key, elec_crds, step_size, params):
            lp_old = log_psi(elec_crds, nuc_crds, params)
            new_crds, _, accept = metropolis_move_lp(
                rng_key, elec_crds, lp_old, step_size, params,
            )
            return new_crds, accept

        self._metropolis_move_allw = jax.jit(jax.vmap(
            metropolis_move, in_axes=(0, 0, None, None),
        ))

        @partial(jax.jit, static_argnums=(4,))
        def decorr_scan(rng_key, walkers, step_size, params,
                        num_steps):
            def step(carry, _):
                rk, w, lp, s, p = carry
                rk, rk1 = jax.random.split(rk)
                keys = jax.random.split(rk1, w.shape[0])
                nw, nlp, acc = jax.vmap(
                    metropolis_move_lp,
                    in_axes=(0, 0, 0, None, None),
                )(keys, w, lp, s, p)
                ar = acc.mean()
                return (rk, nw, nlp, _adapt_step_size(s, ar), p), ar

            lp0 = jax.vmap(
                log_psi, in_axes=(0, None, None),
            )(walkers, nuc_crds, params)
            carry = (rng_key, walkers, lp0, step_size, params)
            (rk, w, _, s, p), ars = jax.lax.scan(
                step, carry, jnp.arange(num_steps),
            )
            return (rk, w, s, p), ars

        self.decorr_scan = decorr_scan

        self._setup_kfac(config, mol_info, init_key, nn_dtype)

    # -----------------------------------------------------
    # KFAC (kfac-jax as configured by DeepMind's FermiNet)
    # -----------------------------------------------------

    def _setup_kfac(self, config, mol_info, init_key, nn_dtype):
        """Compile the factor accumulation and step of kfac-jax.

        The per-row Linear inputs and output cotangents come from
        :class:`~OmegaQMC.psi.nn.kfac_tap.LinearTaps`, for any ansatz
        built from ``nnx.Linear``.

        Reproduces ``kfac_jax.Optimizer`` as DeepMind's FermiNet
        configures it (``ferminet/train.py``: ``estimation_mode=
        'fisher_exact'``, ``curvature_ema``, ``inverse_update_period=1``,
        fixed damping, ``norm_constraint``, no momentum), with the
        blocks its graph matcher assigns to FermiNet
        (``ferminet/curvature_tags_and_blocks.py``):

        * Every Linear (``RepeatedDenseBlock``): factors over the
          ``W * n_rep`` rows the layer is applied to (``n_rep`` =
          electrons, electron pairs or same-spin electrons per
          walker), ``A = E[x~ x~^T]`` with ``x~ = [x, 1]`` when the
          layer has a bias, ``G = 2 E[g g^T]``, curvature
          ``n_rep A (x) G`` (``fixed_scale = n_rep``).  The 2 is the
          Fisher of the normal predictive distribution of variance 1/2
          that FermiNet registers on ``log|psi|``.
        * Every other leaf (the isotropic envelope's ``pi`` and
          exponents, which kfac-jax matches as scale-only
          ``ScaleAndShiftDiagonal`` blocks): the diagonal
          ``2 E_w[(d log|psi_w| / d theta)^2]``, uncentred.
        * Factors are a bias-corrected moving average
          (``WeightedMovingAverage``: weight ``w <- ema w + 1``, value
          ``<- value + (new - value) / w``), so the first step uses
          the first estimate alone, not an identity prior.
        * The damping is added to each whole block, ``n_rep A (x) G +
          damping I``, through pi-adjusted factored damping, and to the
          diagonal blocks.
        * The step is ``-lr c F^{-1} grad`` with ``grad = E[(E_L -
          <E_L>) d log|psi|]`` and ``c = min(1, sqrt(norm_constraint /
          (lr^2 <F^{-1} grad, grad>)))``.
        """
        from .psi.nn.adapter import cast_floats
        from .psi.nn.kfac_tap import LinearTaps, build_tapped_model

        cfg = (load_nn_config(config) if isinstance(config, str)
               else config)
        twin = build_tapped_model(
            lambda: build_nn_wf(cfg, mol_info, nnx.Rngs(init_key)),
        )
        t_graphdef, t_params, t_other = nnx.split(twin, nnx.Param, ...)
        if (jax.tree_util.tree_structure(t_params)
                != jax.tree_util.tree_structure(self.init_params)):
            raise RuntimeError(
                "tapped model has a different parameter layout"
            )
        cdt = None if nn_dtype is None else jnp.dtype(nn_dtype)
        t_other_c = t_other if cdt is None else cast_floats(t_other, cdt)
        nuc_crds = self.nuc_crds
        n_up, n_down = mol_info.n_up, mol_info.n_down

        def apply(params, elec):
            if cdt is not None:
                elec = elec.astype(cdt)
                nuc = nuc_crds.astype(cdt)
                params = cast_floats(params, cdt)
            else:
                nuc = nuc_crds
            phys_conf = PhysicalConfiguration(
                R=nuc, r=to_grouped(elec, n_up, n_down),
                mol_idx=jnp.array(0),
            )
            return nnx.merge(t_graphdef, params, t_other_c)(phys_conf).log

        taps = LinearTaps(
            apply, self.init_params,
            jnp.zeros((self.nelec, 3), dtype=jnp.float64),
        )
        self._taps = taps
        layers = taps.layers
        other_idx = list(taps.other_idx)
        treedef = jax.tree_util.tree_structure(self.init_params)
        leaf_shapes = [l.shape for l in jax.tree.leaves(self.init_params)]
        other_sizes = [int(np.prod(leaf_shapes[i])) for i in other_idx]
        n_other = int(sum(other_sizes))
        f64 = jnp.float64
        # Linear layers: (in [+1], out) shape of the joint kernel+bias
        kshape = {
            name: (info['in_dim'] + (info['bias'] is not None),
                   info['out_dim'])
            for name, info in layers.items()
        }
        n_rep = {name: float(info['n_rep']) for name, info in layers.items()}
        self._linear_count = len(layers)
        self._linear_params = int(sum(a * b for a, b in kshape.values()))

        per_walker = jax.vmap(taps.per_walker, in_axes=(None, 0))

        # Each chunk's products are formed in the network's dtype
        # (float32 by default) at JAX's default matmul precision, as
        # the network itself and kfac-jax are (TF32 on an Ampere GPU;
        # ``jax_default_matmul_precision='highest'`` makes them full
        # float32), and summed over chunks in float64.  Casting the
        # rows to float64 first doubles the cost of the accumulation
        # on a GPU.
        hi = None

        def accumulate(params, walkers_r, de_r):
            def body(carry, xs):
                S_in, S_out, grad, d_sum, g_other = carry
                w_c, de_c = xs
                x_c, g_c, go_c = per_walker(params, w_c)
                S_in, S_out, grad = dict(S_in), dict(S_out), dict(grad)
                for name, info in layers.items():
                    x = x_c[name]
                    if info['bias'] is not None:
                        x = jnp.concatenate(
                            [x, jnp.ones(x.shape[:-1] + (1,), x.dtype)],
                            axis=-1,
                        )
                    g = g_c[name]
                    S_in[name] = S_in[name] + jnp.einsum(
                        'wri,wrj->ij', x, x, precision=hi).astype(f64)
                    S_out[name] = S_out[name] + jnp.einsum(
                        'wro,wrp->op', g, g, precision=hi).astype(f64)
                    grad[name] = grad[name] + jnp.einsum(
                        'wri,wro->io',
                        x * de_c.astype(x.dtype)[:, None, None], g,
                        precision=hi).astype(f64)
                if n_other:
                    J = jnp.concatenate(
                        [l.reshape(w_c.shape[0], -1) for l in go_c],
                        axis=1)
                    d_sum = d_sum + jnp.sum(J * J, axis=0).astype(f64)
                    g_other = g_other + jnp.matmul(
                        de_c.astype(J.dtype), J, precision=hi
                    ).astype(f64)
                return (S_in, S_out, grad, d_sum, g_other), None

            init = (
                {k: jnp.zeros((s[0], s[0]), f64) for k, s in kshape.items()},
                {k: jnp.zeros((s[1], s[1]), f64) for k, s in kshape.items()},
                {k: jnp.zeros(s, f64) for k, s in kshape.items()},
                jnp.zeros(n_other, f64),
                jnp.zeros(n_other, f64),
            )
            carry, _ = jax.lax.scan(body, init, (walkers_r, de_r))
            return carry

        self._accumulate_factors = jax.jit(accumulate)

        shape_groups: Dict[Tuple[int, int], List[str]] = {}
        for name in layers:
            shape_groups.setdefault(kshape[name], []).append(name)
        batched_inverse = jax.vmap(_pi_adjusted_inverse, in_axes=(0, 0, 0))
        ema = jnp.asarray(self.ema_decay, f64)
        norm_clip = (None if self.norm_constraint is None
                     else jnp.asarray(self.norm_constraint, f64))

        def apply_step(params, sums, state, lr_now, damping, num_walkers):
            S_in_s, S_out_s, grad_s, d_sum, g_other = sums
            B = num_walkers
            weight = ema * state['weight'] + 1.0
            eta = 1.0 / weight
            S_in, S_out, grad = {}, {}, {}
            for name in layers:
                S_in[name] = state['S_in'][name] + eta * (
                    S_in_s[name] / (B * n_rep[name]) - state['S_in'][name])
                S_out[name] = state['S_out'][name] + eta * (
                    2.0 * S_out_s[name] / (B * n_rep[name])
                    - state['S_out'][name])
                grad[name] = grad_s[name] / B
            D = state['D'] + eta * (2.0 * d_sum / B - state['D'])
            grad_o = g_other / B

            step = {}
            for names in shape_groups.values():
                inv_in, inv_out = batched_inverse(
                    jnp.stack([S_in[n] for n in names]),
                    jnp.stack([S_out[n] for n in names]),
                    jnp.stack([damping / n_rep[n] for n in names]),
                )
                st = inv_in @ jnp.stack([grad[n] for n in names]) @ inv_out
                for i, n in enumerate(names):
                    step[n] = st[i] / n_rep[n]
            step_o = grad_o / (D + damping)

            dot = sum(jnp.sum(step[n] * grad[n]) for n in layers)
            dot = dot + jnp.sum(step_o * grad_o)
            if norm_clip is not None:
                clip = jnp.minimum(1.0, jnp.sqrt(
                    norm_clip / jnp.maximum(lr_now ** 2 * dot, 1e-300)))
            else:
                clip = jnp.asarray(1.0, f64)
            total_norm = jnp.sqrt(
                sum(jnp.sum(step[n] ** 2) for n in layers)
                + jnp.sum(step_o ** 2) + 1e-30)

            scale = -lr_now * clip
            leaves = list(treedef.flatten_up_to(params))

            def upd(i, delta):
                old = leaves[i]
                leaves[i] = (old + scale * delta.reshape(old.shape)
                             ).astype(old.dtype)

            for name, info in layers.items():
                upd(info['kernel'], step[name][:info['in_dim']])
                if info['bias'] is not None:
                    upd(info['bias'], step[name][info['in_dim']])
            off = 0
            for i, size in zip(other_idx, other_sizes):
                upd(i, step_o[off:off + size])
                off += size
            new_params = treedef.unflatten(leaves)
            new_state = dict(S_in=S_in, S_out=S_out, D=D, weight=weight)
            return new_params, new_state, total_norm, clip, dot, dot

        self._kfac_apply = jax.jit(apply_step)

        def init_state():
            return dict(
                S_in={k: jnp.zeros((s[0], s[0]), f64)
                      for k, s in kshape.items()},
                S_out={k: jnp.zeros((s[1], s[1]), f64)
                       for k, s in kshape.items()},
                D=jnp.zeros(n_other, f64),
                weight=jnp.zeros((), f64),
            )

        self._init_factor_state = init_state

    def kfac_jax_step(self, params, walkers, de, state, it,
                      damping=None, chunk=None):
        """One KFAC update on given walkers, outside the driver loop.

        Args:
            params: Current parameters.
            walkers: ``(W, nelec, 3)``.
            de: ``(W,)`` energy residuals (clipped as configured).
            state: Factor state (``None`` starts afresh).
            it: Iteration number, for the learning-rate schedule.
            damping: Overrides ``self.damping``.
            chunk: Walkers per accumulation chunk (default all).

        Returns:
            ``(new_params, new_state, info)`` with ``info`` the step
            norm, the norm-constraint factor and ``<step, grad>``.
        """
        if state is None:
            state = self._init_factor_state()
        W = walkers.shape[0]
        chunk = W if chunk is None else chunk
        lr_now = (self.lr if self.lr_decay is None
                  else self.lr / (1.0 + it / self.lr_decay))
        sums = self._accumulate_factors(
            params,
            walkers.reshape((W // chunk, chunk) + walkers.shape[1:]),
            jnp.asarray(de).reshape(W // chunk, chunk),
        )
        new_params, new_state, norm, clip, dot, _ = self._kfac_apply(
            params, sums, state, jnp.asarray(lr_now, jnp.float64),
            jnp.asarray(self.damping if damping is None else damping,
                        jnp.float64),
            float(W),
        )
        return new_params, new_state, dict(step_norm=norm, clip=clip,
                                           dot_grad_step=dot)

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

    # -----------------------------------------------------
    # Run state for exact resume
    # -----------------------------------------------------

    def _save_run_state(self, path, it, opt_state, damping, walkers,
                        step_size, rng_key):
        """Write what a resumed run needs besides the parameters.

        The Kronecker-factor state, the current damping, the walkers,
        the Metropolis step size and the PRNG key go to *path*
        (``{prefix}.state.h5``), replaced atomically.  Unlike the
        parameters, the state is not kept in the per-iteration
        backups.
        """
        tmp = path + '.tmp'
        if os.path.exists(tmp):
            os.remove(tmp)
        attrs = dict(iteration=int(it))
        save_state_tree(tmp, 'kfac', opt_state,
                        dict(attrs, damping=float(damping)))
        save_state_tree(tmp, 'mcmc', dict(
            walkers=walkers, step_size=step_size,
            rng=jax.random.key_data(rng_key),
        ), attrs)
        os.replace(tmp, path)

    def _load_run_state(self, path, it, rng_key, num_walkers):
        """Read :meth:`_save_run_state`'s file for iteration *it*.

        Returns ``(opt, mcmc, message)``.  ``opt`` (``{'state',
        'damping'}``) is ``None`` when the file is missing, belongs
        to another iteration or KFAC style, or does not match the
        model; the factors then start afresh.  ``mcmc`` (``{'walkers',
        'step_size', 'rng'}``) is ``None`` under the same conditions
        or when ``num_walkers`` changed; the walkers are then
        initialised and equilibrated as in a fresh run.
        """
        if not os.path.exists(path):
            return None, None, (f"no '{path}': KFAC factors start"
                                " afresh, walkers re-equilibrated")
        opt, oattrs = load_state_tree(path, 'kfac',
                                      self._init_factor_state())
        msgs = []
        if opt is None:
            msgs.append(f"KFAC factors start afresh ({oattrs})")
        elif int(oattrs.get('iteration', -2)) != it:
            msgs.append(
                "KFAC factors start afresh (state file is from"
                f" iteration {oattrs.get('iteration')})")
            opt = None
        else:
            opt = dict(state=opt, damping=float(oattrs['damping']))
            msgs.append("KFAC factors and damping restored")
        mcmc_tmpl = dict(
            walkers=jnp.zeros((num_walkers, self.nelec, 3), jnp.float64),
            step_size=jnp.zeros((), jnp.float64),
            rng=jax.random.key_data(rng_key),
        )
        mcmc, mattrs = load_state_tree(path, 'mcmc', mcmc_tmpl)
        if mcmc is None:
            msgs.append(f"walkers re-equilibrated ({mattrs})")
        elif int(mattrs.get('iteration', -2)) != it:
            msgs.append("walkers re-equilibrated (state file is from"
                        f" iteration {mattrs.get('iteration')})")
            mcmc = None
        else:
            msgs.append("walkers, step size and PRNG key restored")
        return opt, mcmc, f"{path}: " + "; ".join(msgs)

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
        pretrain_iters: int = 0,
        pretrain_basis: str = 'sto-6g',
        pretrain_lr: float = 3.0e-4,
        pretrain_mcmc_width: float = 0.02,
        pretrain_hf_fraction: float = 0.0,
        pretrain_hf_mcmc_width: Optional[float] = None,
        pretrain_hf_burn_in: int = 0,
        pretrain_reinit_walkers: Optional[bool] = None,
    ):
        """Run KFAC natural-gradient VMC optimisation.

        After every ``_CHK_EVERY_KFAC`` iterations (and at the last
        one) the parameters are written to ``{prefix}.chk.h5``, with
        the previous file preserved as ``{prefix}.{iter}.h5``, and the
        rest of the run state (Kronecker factors and their moving-
        average weight, damping, walkers, Metropolis step size, PRNG
        key) to ``{prefix}.state.h5``, which only ever holds the
        latest.  If ``{prefix}.chk.h5`` exists the run resumes from
        it; with a matching state file (same iteration, KFAC style,
        model and ``num_walkers``) it continues exactly where it
        stopped: no re-equilibration, and the same iterates as an
        uninterrupted run.  Without one (e.g. a checkpoint from an
        older version) the factors start afresh and new walkers are
        equilibrated; the log says which.  The adaptive-damping
        windows (``damping_adapt``) are not saved.

        With ``pretrain_iters > 0`` and no checkpoint to resume, the
        initial parameters are first pretrained against UHF orbitals
        (:func:`OmegaQMC.pretrain_nn.pretrain_hf`, FermiNet's
        pretraining) on ``num_walkers`` walkers, the result is saved
        to ``{prefix}.pretrain.h5``, and the pretrained walkers,
        instead of fresh ones, are equilibrated before the first
        iteration.  FermiNet's defaults are ``pretrain_iters=1000``
        and the defaults of the other ``pretrain_*`` arguments; its
        burn-in after pretraining is 100 x 10 Metropolis moves of
        width 0.02 bohr, i.e. ``num_blocks_equil=100``,
        ``num_steps_per_block=10`` and
        ``mc_timestep=0.02**2 / 3`` here (whose step size then
        adapts).  A resumed run never pretrains.

        ``pretrain_hf_fraction > 0`` switches on the pretraining of
        Pfau et al., Phys. Rev. Research 2, 033429 (2020): that share
        of the pretraining walkers samples the Hartree-Fock density
        instead of ``|psi|^2`` (Pfau: 0.5; see
        :func:`~OmegaQMC.pretrain_nn.pretrain_hf`).  Those walkers are
        not distributed like ``|psi|^2``, and Pfau et al. reinitialised
        the walkers after pretraining, so by default
        (``pretrain_reinit_walkers=None``) the walkers are then
        discarded and fresh ones (:meth:`initialize_walkers`, as
        without pretraining) are equilibrated instead; with
        ``pretrain_hf_fraction=0`` the default keeps continuing from
        the pretrained walkers, as before.  ``True``/``False`` forces
        either choice.  At the default Metropolis step (0.02 bohr) the
        HF chain barely moves in 1000 iterations; give it its own step
        and a burn-in with ``pretrain_hf_mcmc_width`` and
        ``pretrain_hf_burn_in``.

        Non-finite guard.  A single walker with a non-finite local
        energy (or a non-finite network gradient) would otherwise turn
        the KFAC step, and every parameter after it, into NaN.  Each
        iteration therefore (i) replaces walkers whose local energy is
        not finite by copies of randomly chosen finite walkers before
        the step, and (ii) rejects a step whose parameters or
        Kronecker factors come out non-finite, keeping the previous
        ones.  Both are logged, and the offending walkers are recorded
        in ``{prefix}.nonfinite.h5`` for diagnosis.  When every local
        energy and the step are finite (the normal case) the
        iteration, including its random numbers, is unchanged.  The
        run stops with ``FloatingPointError`` after
        ``_NONFINITE_MAX_CONSECUTIVE`` rejected steps in a row or if
        more than ``_NONFINITE_MAX_FRACTION`` of the walkers are
        non-finite at once.

        Args:
            rng_key: JAX PRNG key (int or key array).
            num_iters: KFAC iterations (one parameter update each).
                ``0`` performs no update: with a checkpoint at
                ``{prefix}.chk.h5`` this loads it and only runs the
                final estimate, so ``energy['sigma']`` measures the
                saved wavefunction.
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
            pretrain_iters: HF pretraining iterations before a fresh
                start (0 = none).
            pretrain_basis: Basis of the UHF target.
            pretrain_lr: Adam learning rate of the pretraining.
            pretrain_mcmc_width: Metropolis step (bohr) of the one
                move per pretraining iteration.
            pretrain_hf_fraction: Share (0 to 1) of the pretraining
                walkers that sample the HF density (Pfau et al. 2020:
                0.5; default 0, FermiNet 4cfdf1e).
            pretrain_hf_mcmc_width: Metropolis step (bohr) of the HF
                chain; ``None`` (default) uses
                ``pretrain_mcmc_width``.
            pretrain_hf_burn_in: HF-chain moves before the first
                pretraining step (default 0).
            pretrain_reinit_walkers: After pretraining, equilibrate
                fresh walkers instead of the pretrained ones.
                ``None`` (default): ``True`` if
                ``pretrain_hf_fraction > 0``, else ``False``.

        Returns:
            Tuple ``(params_final, energy_data)`` where
            *energy_data* has keys ``'energy'``, ``'E_history'`` and
            ``'Var_history'``.
        """
        if isinstance(rng_key, int):
            rng_key = jax.random.key(rng_key)
        elif not jnp.issubdtype(rng_key.dtype, jax.dtypes.prng_key):
            # a raw uint32 key: wrap it so its data can be checkpointed
            rng_key = jax.random.wrap_key_data(rng_key)
        if (fname_log is None
                or (isinstance(fname_log, str) and fname_log == "")):
            fout = sys.stdout
        else:
            fout = open(fname_log, 'w', 1)

        params = self.init_params
        start_iter = 0
        resumed_opt = resumed_mcmc = None

        # --- Checkpoint resume ---
        chkpt_path = f"{prefix}.chk.h5"
        state_path = f"{prefix}.state.h5"
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
            resumed_opt, resumed_mcmc, why = self._load_run_state(
                state_path, start_iter - 1, rng_key, num_walkers,
            )
            if verbose >= 1:
                print("# " + why, file=fout)

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

        # --- Optional HF pretraining (fresh start only, as FermiNet) ---
        pretrained_walkers = None
        if start_iter == 0 and pretrain_iters > 0 and num_iters > 0:
            from .pretrain_nn import pretrain_hf
            rng_key, pt_key = jax.random.split(rng_key)
            params, pretrained_walkers, pt_info = pretrain_hf(
                self.log_psi, self.mol_info, params, pt_key,
                iterations=int(pretrain_iters),
                num_walkers=num_walkers,
                basis=pretrain_basis,
                lr=pretrain_lr,
                mcmc_width=pretrain_mcmc_width,
                hf_fraction=pretrain_hf_fraction,
                hf_mcmc_width=pretrain_hf_mcmc_width,
                hf_burn_in=pretrain_hf_burn_in,
                fout=fout,
                verbose=verbose,
            )
            save_nn_checkpoint(
                f"{prefix}.pretrain.h5", params, -1,
                self.config_name, self.mol_info,
                energy=float(pt_info['e_hf']),
            )
            reinit = (pretrain_hf_fraction > 0
                      if pretrain_reinit_walkers is None
                      else bool(pretrain_reinit_walkers))
            if verbose >= 1:
                acc_hf = (f", HF-density acceptance"
                          f" {pt_info['acceptance_hf']:.3f}"
                          f" ({pt_info['n_hf']} walkers)"
                          if pt_info['n_hf'] else "")
                print(f"# HF pretraining done: final loss"
                      f" {pt_info['loss'][-1]:.6e}, mean acceptance"
                      f" {pt_info['acceptance']:.3f}{acc_hf}; saved"
                      f" '{prefix}.pretrain.h5'", file=fout)
                if reinit:
                    print("# pretrained walkers discarded; equilibrating"
                          " fresh walkers", file=fout)
            if reinit:
                pretrained_walkers = None

        # --- Initialise and equilibrate walkers ---
        # (a resumed run continues from its saved walkers, step size
        # and PRNG key instead)
        ars = None
        if resumed_mcmc is not None:
            walkers = resumed_mcmc['walkers']
            step_size = resumed_mcmc['step_size']
            rng_key = jax.random.wrap_key_data(resumed_mcmc['rng'])
        else:
            rng_key, init_key = jax.random.split(rng_key)
            if pretrained_walkers is not None:
                walkers = pretrained_walkers
            else:
                walkers = self.initialize_walkers(init_key, num_walkers)
            step_size = jnp.asarray((3 * mc_timestep) ** 0.5)
            for _ in range(num_blocks_equil):
                (rng_key, walkers, step_size, _), ars = self.decorr_scan(
                    rng_key, walkers, step_size, params,
                    num_steps_per_block,
                )

        if verbose >= 1:
            print(
                f"# KFAC-VMC — {self.n_params} params total"
                f" ({self._linear_params} in {self._linear_count}"
                f" Linear layers, rest in diagonal blocks)", file=fout,
            )
            print(
                f"# lr={self.lr}, lr_decay={self.lr_decay},"
                f" damping={self.damping}, ema_decay={self.ema_decay},"
                f" norm_constraint={self.norm_constraint},"
                f" beta={self.var_weight:g}", file=fout,
            )
            print(
                f"# clip_local_energy={self.clip_local_energy:g}"
                + (f" (window centred at the"
                   f" {'median' if self.clip_median else 'mean'},"
                   f" center_at_clip={self.center_at_clip})"
                   if self.clip_local_energy > 0 else " (off)"),
                file=fout,
            )
            print(
                f"# num_walkers={num_walkers},"
                f" factor_chunk_size={chunk},"
                f" kfac_style={self.kfac_style}",
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
                "          lr      damping       |step|          Δt_block",
                file=fout,
            )

        kf_state = self._init_factor_state()
        damping_now = self.damping
        if resumed_opt is not None:
            kf_state = resumed_opt['state']
            damping_now = float(resumed_opt['damping'])

        def write_checkpoint(it, e_total):
            if os.path.exists(chkpt_path):
                os.rename(chkpt_path, f"{prefix}.{it}.h5")
            save_nn_checkpoint(
                chkpt_path, params, it, self.config_name,
                self.mol_info, energy=e_total,
            )
            self._save_run_state(state_path, it, kf_state, damping_now,
                                 walkers, step_size, rng_key)

        e_history: List[float] = []
        var_history: List[float] = []
        n_clipped_history: List[int] = []
        timestamp_prev = datetime.now()
        step_norm_window: List[float] = []
        ratio_window: List[Tuple[float, float]] = []
        prev_e_total: Optional[float] = None
        prev_predicted: Optional[float] = None
        e_mean = None
        nonfinite_path = f"{prefix}.nonfinite.h5"
        n_rejected_in_row = 0
        batch_log_psi = jax.jit(jax.vmap(
            self.log_psi, in_axes=(0, None, None),
        ))
        batch_grad_r_finite = jax.jit(jax.vmap(
            lambda w, p: jnp.all(jnp.isfinite(
                jax.grad(self.log_psi)(w, self.nuc_crds, p)
            )),
            in_axes=(0, None),
        ))

        for it in range(start_iter, start_iter + num_iters):
            # (a) Decorrelate under the current |psi|^2.
            (rng_key, walkers, step_size, _), _ars = self.decorr_scan(
                rng_key, walkers, step_size, params,
                num_steps_decorr,
            )

            # (b) Local energies (forward-Laplacian, chunked).
            e_loc = self.compute_batch_energy(walkers, params)
            finite = jnp.isfinite(e_loc)
            n_bad = int(num_walkers - int(jnp.sum(finite)))
            if n_bad:
                _dump_nonfinite(
                    nonfinite_path, it, 'e_loc',
                    walkers[~finite], e_loc[~finite],
                    batch_log_psi(walkers[~finite], self.nuc_crds,
                                  params),
                )
                if n_bad > _NONFINITE_MAX_FRACTION * num_walkers:
                    raise FloatingPointError(
                        f"iteration {it}: {n_bad} of {num_walkers}"
                        " walkers have a non-finite local energy"
                    )
                good_idx = jnp.nonzero(finite)[0]
                bad_idx = jnp.nonzero(~finite)[0]
                rng_key, k_rep = jax.random.split(rng_key)
                pick = jax.random.choice(k_rep, good_idx, (n_bad,))
                walkers = walkers.at[bad_idx].set(walkers[pick])
                e_loc = e_loc.at[bad_idx].set(e_loc[pick])
                print(f"# iteration {it}: {n_bad} walker(s) with a"
                      " non-finite local energy replaced by copies of"
                      f" finite ones (recorded in {nonfinite_path})",
                      file=fout)
            # <E> and Var(E_L) are logged unclipped; only the
            # gradient residuals de are clipped.
            e_mean = jnp.mean(e_loc)
            de = e_loc - e_mean
            var = jnp.mean(de ** 2)
            if self.clip_local_energy > 0:
                e_clip, de = clip_local_energies(
                    e_loc, self.clip_local_energy,
                    clip_median=self.clip_median,
                    center_at_clip=self.center_at_clip,
                )
                n_clipped_history.append(
                    int(jnp.sum(e_clip != e_loc))
                )
            de_eff = de + self.var_weight * (de ** 2 - jnp.mean(de ** 2))

            # (c)+(d) Accumulate the Kronecker factors, scanned over
            # walker chunks inside a single jitted call, then solve
            # and apply.  The schedule is keyed on the absolute
            # iteration so a resumed run continues decaying rather
            # than restarting at the initial rate.
            if self.lr_decay is not None:
                lr_now = self.lr / (1.0 + it / self.lr_decay)
            else:
                lr_now = self.lr

            sums = self._accumulate_factors(
                params,
                walkers.reshape((n_chunks, chunk) + walkers.shape[1:]),
                de_eff.reshape(n_chunks, chunk),
            )
            (params_new, kf_new, step_norm, clip_val,
             dot_grad_step, dot_step_F_step) = self._kfac_apply(
                params, sums, kf_state,
                jnp.asarray(lr_now, dtype=jnp.float64),
                jnp.asarray(damping_now, dtype=jnp.float64),
                float(num_walkers),
            )
            step_ok = bool(_tree_all_finite((params_new, kf_new)))
            if step_ok:
                params, kf_state = params_new, kf_new
                n_rejected_in_row = 0
            else:
                # Keep the previous parameters and factors.  The
                # local energies were finite, so record which walkers
                # have a non-finite log|psi| or electron gradient.
                n_rejected_in_row += 1
                _dump_nonfinite(
                    nonfinite_path, it, 'step', walkers, e_loc,
                    batch_log_psi(walkers, self.nuc_crds, params),
                    grad_finite=batch_grad_r_finite(walkers, params),
                )
                print(f"# iteration {it}: non-finite KFAC step"
                      " rejected, parameters kept"
                      f" ({n_rejected_in_row} in a row; recorded in"
                      f" {nonfinite_path})", file=fout)
                if n_rejected_in_row >= _NONFINITE_MAX_CONSECUTIVE:
                    raise FloatingPointError(
                        f"iteration {it}: {n_rejected_in_row}"
                        " consecutive non-finite KFAC steps"
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
                    f"  {float(step_norm):>11.4e}  {dt:>16.6f}",
                    file=fout,
                )

            # (f) Adaptive Levenberg-Marquardt damping (a rejected
            # step carries no information about the curvature model).
            if not step_ok:
                prev_e_total = e_total
                prev_predicted = None
                ratio_window = []
                is_last = (it == start_iter + num_iters - 1)
                if is_last or ((it + 1) % _CHK_EVERY_KFAC == 0):
                    write_checkpoint(it, e_total)
                continue
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
                write_checkpoint(it, e_total)

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
            # ``sigma`` is taken after the last update on freshly
            # equilibrated walkers, so it is the right spread to size a
            # production run with -- unlike Var_history[-1], which is
            # measured on the pre-update distribution.
            'energy': {'mean': final_e, 'stderr': final_err,
                       'sigma': final_std},
            'E_history': e_history,
            'Var_history': var_history,
            # walkers whose local energy was clipped, per iteration
            # (empty with clipping off)
            'n_clipped_history': n_clipped_history,
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
            ``clip_local_energy``, ...).

    Returns:
        :class:`_VMCOptDriverNN_KFAC` instance.  Call it with
        ``driver(rng_key, ...)`` to run the optimisation.
    """
    return _VMCOptDriverNN_KFAC(
        mol_info, config, init_key, **kwargs,
    )
