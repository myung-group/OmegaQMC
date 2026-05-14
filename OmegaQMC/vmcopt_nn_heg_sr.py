"""Stochastic-Reconfiguration VMC optimiser for the HEG trial WF.

Natural-gradient optimisation via Stochastic Reconfiguration (SR),
the standard VMC optimiser for deep neural-network wavefunctions
(Sorella 2001; used by FermiNet, PsiFormer, DeepErwin, etc.).  For
the 3D HEG Adam optimisation typically fails to escape the
initial Hartree-Fock plateau within any reasonable iter budget
because the VMC loss has a highly anisotropic Fisher metric —
Adam's diagonal rescaling is blind to this, while SR works in the
metric-aware natural-gradient direction.

The algorithm per iteration:

  1. Decorrelate walkers via Metropolis MCMC.
  2. Compute per-walker local energies ``E_L`` and Jacobian rows
     ``O_n = ∇_θ log|ψ(r_n)|`` (the ``O`` matrix).
  3. Force vector
     ``f_θ = 2 · ⟨(E_L − ⟨E_L⟩) · O_θ⟩_walkers``
     — the standard VMC energy gradient (a linear functional of
     ``O`` and ``E_L``).
  4. Solve ``(S + εI)·δθ = f`` for ``δθ`` via conjugate gradient,
     where ``S = cov(O) = ⟨(O − ⟨O⟩) · (O − ⟨O⟩)ᵀ⟩`` is the Fisher
     matrix and ε is a damping parameter.  ``S`` itself is never
     materialised — only ``S·v`` matvecs, which are O(N_walkers ·
     N_params).
  5. Update ``θ ← θ − η · δθ``.

Typical gain over Adam: convergence in 500-2000 iters to
90-99% of the trial's capacity, vs 10⁴-10⁵ iters for Adam (which
often plateaus earlier because it's direction-blind, not
step-size-blind).  Wall-clock cost per iter is 3-5× Adam due to
Jacobian computation + CG, but the iter-budget reduction
dominates.

Matches the interface of :mod:`vmcopt_nn_heg` — drop-in
replacement by swapping the import.
"""

from __future__ import annotations

import sys
from datetime import datetime
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np
from jax.flatten_util import ravel_pytree

from .psi.nn.heg_wf import (
    HEGConfig,
    make_heg_log_psi_any as make_heg_log_psi,
)
from .psi.nn.periodic import wrap_to_cell, make_cubic_lattice
from .psi.nn.physics import laplacian
from .observables.ewald import build_ewald_tables, ewald_pair_energy
from .observables.ewald_dispatch import (
    build_ewald_tables_dim, ewald_pair_energy_dim,
)


TARGET_ACCEPTANCE_RATE = 0.5
STEP_SIZE_ADAPTATION_RATE = 0.05


def _adapt_step_size(step_size, acceptance_rate):
    return step_size * (
        1.0 + STEP_SIZE_ADAPTATION_RATE
        * (acceptance_rate - TARGET_ACCEPTANCE_RATE)
    )


# ---------------------------------------------------------------------
# Conjugate gradient for S·δθ = f (S positive-definite with damping)
# ---------------------------------------------------------------------

def _cg_solve(matvec, rhs, n_iters, tol=1e-6):
    """Conjugate-gradient solver for a symmetric-positive system.

    Args:
        matvec: Callable ``v → A·v`` where ``A`` is
            symmetric-positive-definite (SPD).
        rhs: Right-hand side ``f`` of shape ``(P,)``.
        n_iters: Maximum CG iterations.
        tol: Relative residual tolerance for early termination.

    Returns:
        ``x`` with ``A·x ≈ rhs`` to the requested tolerance.

    Notes:
        Uses ``jax.lax.fori_loop`` to avoid Python-side iteration
        overhead inside JIT.  Does not early-exit inside the loop
        (would require ``lax.while_loop``); instead runs all
        ``n_iters`` iterations.  For typical SR at P~10⁵ params
        20-50 iters suffice.
    """
    def body(i, state):
        x, r, p, rr = state
        Ap = matvec(p)
        alpha = rr / (jnp.dot(p, Ap) + 1e-30)
        x_new = x + alpha * p
        r_new = r - alpha * Ap
        rr_new = jnp.dot(r_new, r_new)
        beta = rr_new / (rr + 1e-30)
        p_new = r_new + beta * p
        return (x_new, r_new, p_new, rr_new)

    x0 = jnp.zeros_like(rhs)
    r0 = rhs  # residual at x=0
    p0 = rhs  # initial conj. direction
    rr0 = jnp.dot(r0, r0)
    x, _, _, _ = jax.lax.fori_loop(
        0, n_iters, body, (x0, r0, p0, rr0),
    )
    return x


# ---------------------------------------------------------------------
# SR driver
# ---------------------------------------------------------------------

class _VMCOptDriverNNHEG_SR:
    """Natural-gradient (SR) VMC optimiser for HEG ansätze.

    Args:
        config: :class:`HEGConfig` or :class:`HEGPsiFormerConfig`.
        init_key: JAX PRNG key for parameter init.
        lr: SR learning rate ``η`` (applied to the natural-gradient
            step).  Typical: 0.05-0.1 (larger than Adam's because SR
            rescales by the Fisher metric internally).
        damping: SR damping ``ε`` — added to the diagonal of ``S``
            before CG.  Typical: 1e-3 to 1e-4.  Larger ε → step
            closer to plain gradient descent (more stable, slower);
            smaller ε → closer to pure natural gradient (faster at
            the risk of numerical instability).
        n_cg: CG iterations per SR step.  20-50 is standard.
        var_weight: Optional Umrigar-style ``β`` for the mixed
            ``⟨E⟩ + β · Var(E_L)`` objective (0 = pure energy).
        ewald_n_real, ewald_n_recip, ewald_eta: Ewald tuning.
    """

    def __init__(
        self,
        config,
        init_key,
        *,
        lr: float = 0.05,
        damping: float = 1e-3,
        n_cg: int = 30,
        var_weight: float = 0.0,
        ewald_n_real: int = 3,
        ewald_n_recip: int = 6,
        ewald_eta: Optional[float] = None,
        ofname_chkpt: Optional[str] = None,
        lr_schedule: str = 'auto',
        lr_decay_T: Optional[float] = None,
        lr_min: float = 0.0,
        lr_T_max: Optional[int] = None,
        lr_n_restarts: int = 0,
        spring_mu: float = 0.0,
        spring_norm_clip: Optional[float] = None,
        damping_adapt: bool = False,
        damping_min: float = 1.0e-5,
        damping_max: float = 1.0e-1,
        damping_factor: float = 2.0,
        damping_lookback: int = 50,
        sampler: str = 'metropolis',
        mala_grad_clip: Optional[float] = 1.0,
        save_every: int = 0,
    ):
        self.config = config
        self.L = float(config.L)
        self.n_up = int(config.n_up)
        self.n_down = int(config.n_down)
        self.nelec = self.n_up + self.n_down
        self.dim = int(getattr(config, 'dim', 3))
        self.lr = float(lr)
        self.damping = float(damping)
        self.n_cg = int(n_cg)
        self.var_weight = float(var_weight)
        self.ofname_chkpt = ofname_chkpt
        # ---- Learning-rate scheduler ----
        # Schedule names:
        #   'fixed'   : constant lr (vanilla SR).
        #   'inverse' : Smith-2024-style lr_t = lr / (1 + t / T).
        #               Set ``lr_decay_T``.
        #   'cosine'  : Cosine annealing from lr (max) to lr_min over
        #               lr_T_max iters.  ``lr_n_restarts``=0 → simple
        #               cosine; >0 → SGDR-style equal-length cycles.
        # 'auto' (default) → 'inverse' iff lr_decay_T set, else 'fixed'.
        # All schedules: at iter > T_max stay at the floor (lr_min for
        # cosine; lr/(1+T_max/T) for inverse).
        if lr_schedule == 'auto':
            lr_schedule = 'inverse' if lr_decay_T is not None else 'fixed'
        if lr_schedule not in ('fixed', 'inverse', 'cosine'):
            raise ValueError(
                f"lr_schedule must be 'fixed'/'inverse'/'cosine', "
                f"got {lr_schedule!r}"
            )
        self.lr_schedule = lr_schedule
        self.lr_decay_T = (
            None if lr_decay_T is None else float(lr_decay_T)
        )
        self.lr_min = float(lr_min)
        self.lr_T_max = (
            None if lr_T_max is None else int(lr_T_max)
        )
        self.lr_n_restarts = int(lr_n_restarts)
        if self.lr_schedule == 'cosine' and self.lr_T_max is None:
            raise ValueError(
                "lr_schedule='cosine' requires lr_T_max (set it to the "
                "total number of optimization iters, or to the desired "
                "cycle length when used with lr_n_restarts>0)"
            )
        # SPRING (Goldshlager 2024) momentum coefficient.  0 → vanilla
        # SR.  Smith uses 0.9.  Formula:
        #     dθ_t = (S + λI)^{-1} (g + λμ dθ_{t-1})
        self.spring_mu = float(spring_mu)
        # F-norm clip on dθ (KFAC-style).  Smith uses C = η_0 (the
        # initial lr).  None → take initial lr; 0 → disabled; float →
        # explicit value.
        if spring_norm_clip is None:
            self.spring_norm_clip = float(self.lr)
        elif float(spring_norm_clip) <= 0.0:
            self.spring_norm_clip = 0.0  # disabled
        else:
            self.spring_norm_clip = float(spring_norm_clip)
        # Adaptive damping (Levenberg-Marquardt-style).  Off by
        # default — Smith holds λ fixed at 0.001.
        self.damping_adapt = bool(damping_adapt)
        self.damping_min = float(damping_min)
        self.damping_max = float(damping_max)
        self.damping_factor = float(damping_factor)
        self.damping_lookback = int(damping_lookback)
        # MCMC sampler: 'metropolis' (Gaussian random walk, default) or
        # 'mala' (Metropolis-adjusted Langevin, Smith 2024).
        self.sampler = str(sampler).lower()
        if self.sampler not in ('metropolis', 'mala'):
            raise ValueError(
                f"sampler must be 'metropolis' or 'mala', got {sampler!r}"
            )
        self.mala_grad_clip = (
            None if mala_grad_clip is None else float(mala_grad_clip)
        )
        # Periodic checkpoint save during training (in addition to the
        # one written at end-of-training).  0 → save only at end (old
        # behaviour).  Recommended: 500 — limits worst-case data loss
        # on SIGTERM/SIGINT to ~500 iters.
        self.save_every = int(save_every)

        if self.dim == 3:
            self.lattice = make_cubic_lattice(self.L)
        else:
            from .psi.nn.periodic import make_square_lattice
            self.lattice = make_square_lattice(self.L)
        self.ewald = build_ewald_tables_dim(
            self.L, dim=self.dim, eta=ewald_eta,
            n_real=ewald_n_real, n_recip=ewald_n_recip,
        )

        log_psi_pytree, init_params, graphdef = make_heg_log_psi(
            config, init_key,
        )
        self.log_psi_pytree = log_psi_pytree
        self.graphdef = graphdef

        # Flatten the parameter pytree once — SR works on the flat
        # representation so the Jacobian is a plain ``(N, P)`` matrix.
        p_flat, unravel = ravel_pytree(init_params)
        self.params_flat = p_flat
        self.unravel = unravel
        self.n_params = int(p_flat.shape[0])

        def log_psi_flat(r, p_flat):
            return log_psi_pytree(r, unravel(p_flat))

        self.log_psi_flat = log_psi_flat

        tables = self.ewald
        lattice = self.lattice
        nelec = self.nelec
        dim = self.dim

        # Per-walker local energy (kin + pot).
        def kin_only(r, p_flat):
            def f_flat(r_flat):
                return log_psi_flat(
                    r_flat.reshape(nelec, dim), p_flat,
                )
            lap_val, grad_val = laplacian(f_flat)(r.reshape(-1))
            return -0.5 * (lap_val + jnp.dot(grad_val, grad_val))

        # MCMC step (same as Adam path).
        def metropolis_move(rng_key, r, step_size, p_flat):
            key_prop, key_acc = jax.random.split(rng_key)
            proposed = r + step_size * jax.random.normal(
                key_prop, r.shape,
            )
            proposed = wrap_to_cell(proposed, lattice)
            lp_old = log_psi_flat(r, p_flat)
            lp_new = log_psi_flat(proposed, p_flat)
            accept = jax.random.uniform(key_acc) < jnp.exp(
                2.0 * (lp_new - lp_old),
            )
            new = jnp.where(accept, proposed, r)
            return new, accept

        # Multi-GPU: each device handles n_walkers/n_devices walkers.
        # Outer pmap distributes across devices (axis_name='dev'),
        # inner vmap parallelises over the device-local walker batch.
        # in_axes=(0, 0, None, None): rng_key + walkers are sharded
        # along device axis; step_size + p_flat are replicated.
        self._metropolis_move_pmap = jax.pmap(
            jax.vmap(metropolis_move, in_axes=(0, 0, None, None)),
            in_axes=(0, 0, None, None),
            axis_name='dev',
        )

        # ---- MALA (Metropolis-adjusted Langevin) sampler ------------
        # Smith 2024 Eq. 22:
        #   R̃ = R + τ ∇ log|Ψ|² + sqrt(2τ) ξ
        # where ξ ~ N(0, I) and the ∇ log|Ψ|² = 2 ∇ log|Ψ|.  Use the
        # log-psi gradient already available, multiplied by 2.  The
        # Metropolis-Hastings ratio includes both the wavefunction
        # ratio and the Langevin proposal asymmetry.
        log_psi_grad_r = jax.grad(log_psi_flat, argnums=0)
        mala_clip = self.mala_grad_clip

        def mala_move(rng_key, r, step_size, p_flat):
            # Convention: step_size in MALA = sqrt(2 τ).  τ = ½ step².
            tau = 0.5 * step_size * step_size

            key_prop, key_acc = jax.random.split(rng_key)
            grad_r = log_psi_grad_r(r, p_flat)
            drift_r = 2.0 * grad_r          # ∇ log|Ψ|² = 2 ∇ log|Ψ|
            if mala_clip is not None:
                gnorm = jnp.sqrt(jnp.sum(drift_r ** 2) + 1e-12)
                scale = jnp.minimum(1.0, mala_clip / gnorm)
                drift_r = drift_r * scale

            noise = jax.random.normal(key_prop, r.shape)
            proposed = r + tau * drift_r + step_size * noise
            proposed = wrap_to_cell(proposed, lattice)

            # Reverse drift for the M-H ratio.
            grad_p = log_psi_grad_r(proposed, p_flat)
            drift_p = 2.0 * grad_p
            if mala_clip is not None:
                gnorm_p = jnp.sqrt(jnp.sum(drift_p ** 2) + 1e-12)
                scale_p = jnp.minimum(1.0, mala_clip / gnorm_p)
                drift_p = drift_p * scale_p

            lp_old = log_psi_flat(r, p_flat)
            lp_new = log_psi_flat(proposed, p_flat)

            # log q(R | R̃) − log q(R̃ | R) (Gaussian, mean = R + τ drift).
            d_fwd = proposed - r - tau * drift_r
            d_bwd = r - proposed - tau * drift_p
            two_tau = 2.0 * tau
            log_q_ratio = (
                jnp.sum(d_fwd ** 2) - jnp.sum(d_bwd ** 2)
            ) / (2.0 * two_tau)

            log_alpha = 2.0 * (lp_new - lp_old) + log_q_ratio
            accept = jax.random.uniform(key_acc) < jnp.exp(log_alpha)
            new = jnp.where(accept, proposed, r)
            return new, accept

        self._mala_move_pmap = jax.pmap(
            jax.vmap(mala_move, in_axes=(0, 0, None, None)),
            in_axes=(0, 0, None, None),
            axis_name='dev',
        )

        log_psi_grad_flat = jax.grad(log_psi_flat, argnums=1)

        # Ewald potential: chunked over walkers to avoid XLA's
        # fusion bomb at (n_real=3, n_recip=6).  See vmcopt_nn_heg.
        if dim == 3:
            pot_fn = lambda w: ewald_pair_energy(w, tables)
        else:
            from .observables.ewald_2d import ewald_2d_pair_energy
            pot_fn = lambda w: ewald_2d_pair_energy(w, tables)

        # Multi-GPU device set-up.  pmap distributes across leading
        # axis = device axis.
        self.devices = jax.devices()
        self.n_devices = len(self.devices)

        # Per-device kinetic+potential (no cross-device communication
        # needed; each device computes its own walkers' E_loc).
        def kin_pot_local(walkers, p_flat):
            e_kin = jax.vmap(kin_only, in_axes=(0, None))(walkers, p_flat)
            e_pot = pot_fn(walkers)
            return e_kin + e_pot

        self._eloc_pmap = jax.pmap(
            kin_pot_local,
            in_axes=(0, None),
            axis_name='dev',
        )

        # ---- SR step (pmap'd, all-gather o across devices) ----
        var_weight_arr = jnp.asarray(
            self.var_weight, dtype=jnp.float64,
        )

        # Closure-capture the global walker count (statically known
        # at __call__ time; we re-create the pmap'd function in
        # __call__ so this is a Python int, suitable for jnp.eye).
        n_devices_static = self.n_devices

        def make_sr_step(num_walkers):
            n_w_local = num_walkers // n_devices_static

            def sr_step(
                walkers, p_flat, e_loc,
                prev_dtheta, damping, lambda_mu, c_clip,
            ):
                """One SR step, per-device.  Cross-device collectives
                via psum / all_gather under axis_name='dev'.

                Inputs (per device):
                  walkers:    (n_w_local, nelec, dim)
                  p_flat:     (n_p,)              replicated
                  e_loc:      (n_w_local,)
                  prev_dtheta:(n_p,)              replicated
                  damping, lambda_mu, c_clip:     scalars, replicated

                Returns (per device, all replicated outputs):
                  dtheta:    (n_p,)
                  e_mean:    scalar
                  var:       scalar
                  g_norm:    scalar
                  scale:     scalar
                """
                # Per-walker o = ∇_θ log|Ψ| (local walkers).
                o_local = jax.vmap(
                    log_psi_grad_flat, in_axes=(0, None),
                )(walkers, p_flat)                          # (n_w_local, n_p)

                # Global mean energy (across all devices).
                e_sum_local = jnp.sum(e_loc)
                e_sum = jax.lax.psum(e_sum_local, axis_name='dev')
                e_mean = e_sum / num_walkers
                de_local = e_loc - e_mean
                var_sum = jax.lax.psum(
                    jnp.sum(de_local ** 2), axis_name='dev',
                )
                var = var_sum / num_walkers

                # Global mean of o (across all devices).
                o_sum_local = jnp.sum(o_local, axis=0)      # (n_p,)
                o_sum = jax.lax.psum(o_sum_local, axis_name='dev')
                o_mean = o_sum / num_walkers
                do_local = o_local - o_mean[None, :]        # (n_w_local, n_p)

                # Force g (gradient).
                de_mix_local = de_local + var_weight_arr * (
                    de_local ** 2 - var
                )
                g_partial = de_mix_local @ do_local         # (n_p,)
                g_full = jax.lax.psum(g_partial, axis_name='dev')
                g = 2.0 * g_full / num_walkers

                # SPRING RHS.
                rhs = g + lambda_mu * prev_dtheta

                # ===== SMW dual-form SR =====
                # Need full do (n_w, n_p) on every device for the
                # K = do @ do.T solve.  all_gather along device axis
                # tiles the local do shards into the global matrix.
                # Memory: n_w × n_p × 8 bytes (e.g. 1024 × 53k × 8 =
                # 437 MB at production, fits easily in 80 GB A100).
                do = jax.lax.all_gather(
                    do_local, axis_name='dev', tiled=True,
                )                                           # (n_w, n_p)

                inv_λn = 1.0 / (damping * num_walkers)
                K = (do @ do.T) * inv_λn                    # (n_w, n_w)
                I_plus_K = K + jnp.eye(num_walkers, dtype=K.dtype)
                o_rhs = (do @ rhs) * inv_λn                 # (n_w,)
                u = jnp.linalg.solve(I_plus_K, o_rhs)       # (n_w,)
                dtheta = (rhs - do.T @ u) / damping         # (n_p,)

                # F-norm clip: S dθ via (1/n) do^T (do dθ).
                s_dtheta = (do.T @ (do @ dtheta)) / num_walkers
                f_norm_sq = jnp.maximum(
                    jnp.dot(dtheta, s_dtheta), 1e-20,
                )
                f_norm = jnp.sqrt(f_norm_sq)
                raw_scale = c_clip / (f_norm + 1e-20)
                scale = jnp.where(
                    c_clip > 0.0,
                    jnp.minimum(1.0, raw_scale),
                    jnp.ones_like(raw_scale),
                )
                dtheta = scale * dtheta

                g_norm = jnp.sqrt(jnp.sum(g ** 2))
                return dtheta, e_mean, var, g_norm, scale

            return jax.pmap(
                sr_step,
                in_axes=(0, None, 0, None, None, None, None),
                axis_name='dev',
            )

        self._make_sr_step = make_sr_step

    # -----------------------------------------------------
    # Learning-rate scheduler
    # -----------------------------------------------------

    def _compute_lr(self, it: int) -> float:
        """Return effective lr at iteration ``it`` (1-indexed).

        Dispatches on ``self.lr_schedule``:
          * 'fixed'   → ``self.lr``
          * 'inverse' → ``self.lr / (1 + (it-1) / lr_decay_T)``
          * 'cosine'  → cosine anneal (with optional warm restarts)
        """
        import math
        lr_max = self.lr
        if self.lr_schedule == 'fixed':
            return lr_max
        if self.lr_schedule == 'inverse':
            T = self.lr_decay_T
            return lr_max / (1.0 + (it - 1) / T)
        # cosine (with optional restarts)
        T_max = float(self.lr_T_max)
        n_cycles = max(1, self.lr_n_restarts + 1)
        T_cycle = T_max / n_cycles
        t = float(it - 1)
        if t >= T_max:
            return self.lr_min
        cycle_idx = int(t // T_cycle)
        t_in_cycle = t - cycle_idx * T_cycle
        phase = t_in_cycle / T_cycle      # in [0, 1)
        return self.lr_min + 0.5 * (lr_max - self.lr_min) * (
            1.0 + math.cos(math.pi * phase)
        )

    # -----------------------------------------------------
    # Checkpoint helper (re-used by periodic save + end-of-training)
    # -----------------------------------------------------

    def _save_checkpoint(self, params_flat, epoch, energy=None,
                         note: str = ""):
        """Persist params + meta to ``self.ofname_chkpt``.

        Failures are logged, never raised — long training runs must
        not crash on a transient I/O hiccup.
        """
        if not self.ofname_chkpt:
            return
        from .nn_checkpoint import save_nn_checkpoint

        class _HEGMolInfoStub:
            def __init__(self, n_up, n_down, dim):
                self.n_up = int(n_up)
                self.n_down = int(n_down)
                self.charges = np.zeros((0,), dtype=np.float64)
                self.coords = np.zeros((0, dim), dtype=np.float64)

        stub = _HEGMolInfoStub(self.n_up, self.n_down, self.dim)
        try:
            save_nn_checkpoint(
                self.ofname_chkpt,
                self.unravel(params_flat),
                epoch=int(epoch),
                config_name='HEG_PsiFormer',
                mol_info=stub,
                energy=energy,
            )
            return True
        except Exception as e:
            print(
                f"[warn] checkpoint save failed "
                f"({self.ofname_chkpt}{', ' + note if note else ''}): {e}",
            )
            return False

    # -----------------------------------------------------
    # Walker management
    # -----------------------------------------------------

    def initialize_walkers(self, rng_key, num_walkers):
        """Sample walker positions for the SR optimiser.

        Dispatch order:
        1. Explicit ``config.walker_init`` (``'uniform'`` or
           ``'crystal_perturbed'``) wins.  Smith 2024 uses
           ``crystal_perturbed`` for both fluid and crystal phases.
        2. Otherwise fall back to envelope-based default: crystal_gaussian
           → crystal_perturbed; everything else → uniform.
        """
        walker_init = str(getattr(
            self.config, 'walker_init', 'auto',
        )).lower()
        envelope_type = getattr(self.config, 'envelope_type', 'plane_wave')

        if walker_init == 'auto':
            walker_init = (
                'crystal_perturbed'
                if (envelope_type == 'crystal_gaussian' and self.dim == 2)
                else 'uniform'
            )

        if walker_init == 'crystal_perturbed' and self.dim == 2:
            from .psi.nn.env_localized_2d import crystal_init_walkers_2d
            return crystal_init_walkers_2d(
                rng_key, num_walkers,
                n_up=self.n_up, n_down=self.n_down, L=self.L,
                sigma_init=float(getattr(
                    self.config, 'crystal_sigma_init', 0.25,
                )),
                spin_pattern=str(getattr(
                    self.config, 'crystal_spin_pattern', 'neel',
                )),
            )
        return self.L * jax.random.uniform(
            rng_key, (num_walkers, self.nelec, self.dim),
        )

    # -----------------------------------------------------
    # Training loop
    # -----------------------------------------------------

    # NOTE: _pot_batch removed in pmap port — kinetic+potential
    # are computed together by the pmap'd ``_eloc_pmap``.

    def __call__(
        self,
        rng_key,
        num_iters: int = 500,
        num_walkers: int = 256,
        mcmc_decorr_steps: int = 20,
        num_equil_steps: int = 400,
        mc_timestep: float = 0.1,
        fname_log: Optional[str] = None,
        verbose: int = 1,
    ):
        """Run SR-VMC optimisation.

        Args:
            rng_key: JAX PRNG key (int or array).
            num_iters: SR parameter updates.
            num_walkers: MCMC walkers per iteration.
            mcmc_decorr_steps: MCMC steps between SR updates.
            num_equil_steps: Initial equilibration.
            mc_timestep: Initial MC timestep.
            fname_log: Log file path (None = stdout).
            verbose: Verbosity level.

        Returns:
            Dict with ``'params'``, ``'params_flat'``,
            ``'E_per_elec_history'``, ``'Var_history'``,
            ``'E_final_ha'``.
        """
        if isinstance(rng_key, int):
            rng_key = jax.random.key(rng_key)
        if (fname_log is None
                or (isinstance(fname_log, str) and fname_log == "")):
            fout = sys.stdout
        else:
            fout = open(fname_log, 'w', 1)

        params_flat = self.params_flat
        # Pick sampler.  MALA's drift contribution is large at small
        # step size, so we start at smaller (3·mc_timestep)^½.
        if self.sampler == 'mala':
            mcmc_move_pmap = self._mala_move_pmap
        else:
            mcmc_move_pmap = self._metropolis_move_pmap
        eloc_pmap = self._eloc_pmap

        # ---- pmap walker shape -----------------------------------
        # Reshape sharded data to (n_dev, n_w_local, ...) so pmap
        # distributes the leading axis across devices.  Replicated
        # data (params, scalars) keep their bare shape; pmap is told
        # via in_axes=None in each pmap'd function.
        n_dev = self.n_devices
        if num_walkers % n_dev != 0:
            raise ValueError(
                f"num_walkers ({num_walkers}) must be divisible by "
                f"n_devices ({n_dev}) for pmap walker sharding."
            )
        n_w_local = num_walkers // n_dev

        rng_key, init_key = jax.random.split(rng_key)
        walkers = self.initialize_walkers(init_key, num_walkers)
        walkers = walkers.reshape(
            n_dev, n_w_local, self.nelec, self.dim,
        )

        # Build the SR step pmap with statically-known num_walkers.
        sr_update_pmap = self._make_sr_step(num_walkers)

        # Step size is a scalar; pmap'd MCMC takes it via in_axes=None.
        step_size = jnp.asarray(
            (3 * mc_timestep) ** 0.5, dtype=jnp.float64,
        )

        # Pre-jitted single MCMC step.  pmap distributes walkers,
        # vmap parallelises within each device.  acc reduction is a
        # pmean across devices so the same step_size update is seen
        # everywhere.
        @jax.jit
        def _mcmc_step(rng_key, walkers, step_size, params_flat):
            # rng_key shape (n_dev,) — one key per device.
            # Each device splits its key into n_w_local sub-keys.
            keys_per_dev = jax.vmap(
                lambda k: jax.random.split(k, n_w_local),
            )(rng_key)                                # (n_dev, n_w_local, 2)
            walkers, acc = mcmc_move_pmap(
                keys_per_dev, walkers, step_size, params_flat,
            )                                         # walkers: (n_dev,n_w_local,...)
            ar = jnp.mean(acc).astype(jnp.float64)    # global mean across all
            new_step = _adapt_step_size(step_size, ar)
            return walkers, new_step, ar

        # Equilibration.  Python loop over jitted steps; per-iter
        # sync via float() at the end blocks just enough to keep
        # async dispatch from over-staging.
        ar_jax = jnp.zeros((), dtype=jnp.float64)
        for i in range(num_equil_steps):
            rng_key, sub = jax.random.split(rng_key)
            keys_dev = jax.random.split(sub, n_dev)
            walkers, step_size, ar_jax = _mcmc_step(
                keys_dev, walkers, step_size, params_flat,
            )
            if i % 20 == 19:
                step_size.block_until_ready()
        ar = float(ar_jax)

        if verbose >= 1:
            extras = []
            if self.lr_schedule == 'inverse':
                extras.append(f"lr_decay_T={self.lr_decay_T:g}")
            elif self.lr_schedule == 'cosine':
                extras.append(
                    f"cosine[lr_min={self.lr_min:g}, "
                    f"T_max={self.lr_T_max}, "
                    f"n_restarts={self.lr_n_restarts}]"
                )
            if self.spring_mu > 0.0:
                extras.append(f"spring_mu={self.spring_mu}")
            if self.spring_norm_clip > 0.0:
                extras.append(
                    f"spring_norm_clip={self.spring_norm_clip:g}"
                )
            if self.damping_adapt:
                extras.append(
                    f"damping_adapt=[{self.damping_min:g},"
                    f"{self.damping_max:g}]"
                )
            if self.sampler != 'metropolis':
                extras.append(f"sampler={self.sampler}")
            extras_str = (' ' + ' '.join(extras)) if extras else ''
            print(
                f"# SR-VMC training — {self.n_params} params, "
                f"lr={self.lr}, damping={self.damping}, "
                f"n_cg={self.n_cg}, β={self.var_weight:.3g}"
                f"{extras_str}",
                file=fout,
            )
            print(
                f"# Equilibration acceptance: {ar:.3f}, "
                f"step size: {float(step_size):.4f} Bohr",
                file=fout,
            )
            obj_tag = (
                f"L/N (β={self.var_weight:.3g})"
                if self.var_weight > 0
                else ""
            )
            header = (
                f"# iter       <E>/N            Var(E)          "
                f"|f|             dt"
            )
            print(header, file=fout)

        e_history = []
        var_history = []
        timestamp_prev = datetime.now()

        # ---- Graceful shutdown on SIGTERM/SIGINT ----
        # When the user (or the OS) sends SIGTERM (e.g. ``kill <pid>``)
        # or SIGINT (Ctrl-C), set a flag that the training loop polls
        # at the end of each iter.  When set, we save the latest
        # checkpoint and break cleanly.  Important for long runs that
        # the user wants to terminate early without losing progress.
        import signal
        _stop_requested = {'flag': False, 'signo': None}

        def _stop_handler(signo, frame):
            _stop_requested['flag'] = True
            _stop_requested['signo'] = signo
            print(
                f"\n[signal] received {signal.Signals(signo).name}; "
                f"will save checkpoint and exit at end of iter.",
            )

        prev_sigterm = signal.signal(signal.SIGTERM, _stop_handler)
        prev_sigint = signal.signal(signal.SIGINT, _stop_handler)

        # SPRING momentum buffer (zero-init).  Replicated across devices
        # by virtue of being a single-host array passed through pmap
        # with in_axes=None.
        prev_dtheta = jnp.zeros_like(params_flat)
        spring_mu = self.spring_mu
        c_clip = self.spring_norm_clip   # 0.0 → disabled inside core
        damping_now = float(self.damping)
        damping_adapt = self.damping_adapt
        damping_factor = self.damping_factor
        damping_min = self.damping_min
        damping_max = self.damping_max
        damping_lookback = self.damping_lookback
        clip_scale = 1.0

        for it in range(1, num_iters + 1):
            # Decorrelate walkers (pmap'd, distributes across devices).
            for _ in range(mcmc_decorr_steps):
                rng_key, sub = jax.random.split(rng_key)
                keys_dev = jax.random.split(sub, n_dev)
                walkers, step_size, _ = _mcmc_step(
                    keys_dev, walkers, step_size, params_flat,
                )

            # Local energies (per-device kin + pot, no cross-device
            # communication needed).
            e_loc = eloc_pmap(walkers, params_flat)
            # e_loc shape (n_dev, n_w_local) — leave in pmap shape;
            # the SR step's pmap wants this shape via in_axes=0.

            # SPRING coupling: λμ.  When μ=0 the RHS is just g (vanilla SR).
            lambda_mu = damping_now * spring_mu

            # SR update — pmap'd; outputs are per-device but
            # all-reduced internally so every device returns the same
            # values.  Take [0] to get the host-side scalar/array.
            dtheta_pdev, e_mean_pdev, var_pdev, g_norm_pdev, clip_scale_pdev = (
                sr_update_pmap(
                    walkers, params_flat, e_loc,
                    prev_dtheta,
                    jnp.asarray(damping_now, dtype=jnp.float64),
                    jnp.asarray(lambda_mu, dtype=jnp.float64),
                    jnp.asarray(c_clip, dtype=jnp.float64),
                )
            )
            # Outputs from pmap have leading device axis; replicated
            # values are identical across [0..n_dev], so we just take
            # device 0.
            dtheta = dtheta_pdev[0]
            e_mean = e_mean_pdev[0]
            var = var_pdev[0]
            g_norm = g_norm_pdev[0]
            clip_scale = float(clip_scale_pdev[0])
            if spring_mu > 0.0:
                prev_dtheta = dtheta

            # Learning-rate scheduler (fixed / inverse / cosine).
            lr_now = self._compute_lr(it)

            params_flat = params_flat - lr_now * dtheta

            # Levenberg-Marquardt-style adaptive damping.  Compare
            # the most recent ``damping_lookback`` energies' mean to
            # the prior window of the same width: if energy increased,
            # SR is over-confident — increase damping.  If it
            # decreased substantially, decrease damping.  Window-based
            # rather than per-step so noise doesn't whipsaw λ.
            if damping_adapt and it > 2 * damping_lookback:
                e_recent = float(np.mean(
                    e_history[-damping_lookback:]
                ))
                e_prev = float(np.mean(
                    e_history[
                        -2 * damping_lookback:-damping_lookback
                    ]
                ))
                # Compare on a per-electron basis (e_history already is).
                e_diff = e_recent - e_prev
                noise_floor = max(1e-6, 0.5 * np.sqrt(
                    float(np.mean(var_history[-damping_lookback:]))
                    / num_walkers,
                ) / self.nelec)
                if e_diff > noise_floor:
                    damping_now = min(
                        damping_max, damping_now * damping_factor,
                    )
                elif e_diff < -noise_floor:
                    damping_now = max(
                        damping_min, damping_now / damping_factor,
                    )

            # Per-iter logging.
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
                    f"{float(g_norm):>12.4e}  "
                    f"{dt:>8.3f}",
                    file=fout,
                )

            # BF diagnostics (every 100 iters when coord BF is active).
            # Tracks displacement magnitude and min quasiparticle pair
            # separation — direct evidence for coord-collapse pathology.
            # min_pair_sep_min → 0 = electrons colliding in coord space →
            # singular Slater determinant → biased E_L estimator.
            bf_diag_fn = getattr(self.log_psi_pytree, 'bf_diagnostics', None)
            if (verbose >= 1 and bf_diag_fn is not None
                    and (it == 1 or it % 100 == 0)):
                stats = bf_diag_fn(walkers, self.unravel(params_flat))
                print(
                    f"  [bf] iter {it:>5d}  "
                    f"disp(mean/walker)={float(stats['mean_disp_avg']):.3f}  "
                    f"disp(max/walker)={float(stats['max_disp_avg']):.3f}  "
                    f"disp_max_overall={float(stats['max_disp_max']):.3f}  "
                    f"min_qp_sep(avg/walker)={float(stats['min_pair_sep_avg']):.4f}  "
                    f"min_qp_sep(global)={float(stats['min_pair_sep_min']):.4f}",
                    file=fout,
                )

            # Periodic checkpoint save (in-loop) — caps the worst-case
            # data loss to ~self.save_every iters when SIGKILLed.
            if (self.save_every > 0
                    and self.ofname_chkpt
                    and it % self.save_every == 0):
                ok = self._save_checkpoint(
                    params_flat, epoch=it, energy=e_per,
                    note=f"periodic@iter{it}",
                )
                if ok and verbose >= 1:
                    print(
                        f"  [chkpt] saved at iter {it}, "
                        f"E/N={e_per:+.6f} Ha → {self.ofname_chkpt}",
                        file=fout,
                    )

            # Honour SIGTERM / SIGINT — save and break loop.
            if _stop_requested['flag']:
                if self.ofname_chkpt:
                    ok = self._save_checkpoint(
                        params_flat, epoch=it, energy=e_per,
                        note=f"signal@iter{it}",
                    )
                    if ok and verbose >= 1:
                        print(
                            f"  [chkpt] saved on signal at iter {it}, "
                            f"E/N={e_per:+.6f} Ha → {self.ofname_chkpt}",
                            file=fout,
                        )
                break

        # Restore previous signal handlers so a subsequent eval
        # phase doesn't silently inherit our flags.
        signal.signal(signal.SIGTERM, prev_sigterm)
        signal.signal(signal.SIGINT, prev_sigint)

        if not (fname_log is None
                or (isinstance(fname_log, str) and fname_log == "")):
            fout.close()

        self.params_flat = params_flat
        params_pytree = self.unravel(params_flat)

        # End-of-training checkpoint save (also dumped periodically
        # during training, so this is a final-state snapshot).
        self._save_checkpoint(
            params_flat,
            epoch=len(e_history),
            energy=(e_history[-1] if e_history else None),
            note='final',
        )

        return {
            'params': params_pytree,
            'params_flat': params_flat,
            'E_per_elec_history': e_history,
            'Var_history': var_history,
            'E_final_ha': e_history[-1] if e_history else None,
        }


def get_vmcopt_nn_heg_sr_func(
    config,
    init_key,
    *,
    prefix: str = "heg_sr",
    lr: float = 0.05,
    damping: float = 1e-3,
    n_cg: int = 30,
    var_weight: float = 0.0,
    ewald_n_real: int = 3,
    ewald_n_recip: int = 6,
    ewald_eta: Optional[float] = None,
    lr_schedule: str = 'auto',
    lr_decay_T: Optional[float] = None,
    lr_min: float = 0.0,
    lr_T_max: Optional[int] = None,
    lr_n_restarts: int = 0,
    spring_mu: float = 0.0,
    spring_norm_clip: Optional[float] = None,
    damping_adapt: bool = False,
    damping_min: float = 1.0e-5,
    damping_max: float = 1.0e-1,
    damping_factor: float = 2.0,
    damping_lookback: int = 50,
    sampler: str = 'metropolis',
    mala_grad_clip: Optional[float] = 1.0,
    save_every: int = 0,
):
    """Construct an SR-VMC optimiser for a HEG ansatz.

    Args: see :class:`_VMCOptDriverNNHEG_SR`.

    Returns:
        :class:`_VMCOptDriverNNHEG_SR` — callable.
    """
    if prefix.endswith(".chk.h5"):
        prefix = prefix[: -len(".chk.h5")]
    ofname_chkpt = prefix + ".chk.h5" if prefix else None
    return _VMCOptDriverNNHEG_SR(
        config, init_key,
        lr=lr,
        damping=damping,
        n_cg=n_cg,
        var_weight=var_weight,
        ewald_n_real=ewald_n_real,
        ewald_n_recip=ewald_n_recip,
        ewald_eta=ewald_eta,
        ofname_chkpt=ofname_chkpt,
        lr_schedule=lr_schedule,
        lr_decay_T=lr_decay_T,
        lr_min=lr_min,
        lr_T_max=lr_T_max,
        lr_n_restarts=lr_n_restarts,
        spring_mu=spring_mu,
        spring_norm_clip=spring_norm_clip,
        damping_adapt=damping_adapt,
        damping_min=damping_min,
        damping_max=damping_max,
        damping_factor=damping_factor,
        damping_lookback=damping_lookback,
        sampler=sampler,
        mala_grad_clip=mala_grad_clip,
        save_every=save_every,
    )
