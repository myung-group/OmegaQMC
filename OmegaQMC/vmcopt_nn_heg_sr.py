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

class _HEGSROptimizer:
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

        self._metropolis_move_allw = jax.jit(jax.vmap(
            metropolis_move, in_axes=(0, 0, None, None),
        ))

        # Per-walker quantities for SR.
        self._kin_batch = jax.jit(jax.vmap(kin_only, in_axes=(0, None)))
        log_psi_grad_flat = jax.grad(log_psi_flat, argnums=1)
        self._o_batch = jax.jit(jax.vmap(
            log_psi_grad_flat, in_axes=(0, None),
        ))

        # Ewald potential: chunked over walkers to avoid XLA's
        # fusion bomb at (n_real=3, n_recip=6).  See vmcopt_nn_heg.
        self._pot_chunk_size = 32
        if dim == 3:
            self._pot_chunk = jax.jit(
                lambda w: ewald_pair_energy(w, tables),
            )
        else:
            from .observables.ewald_2d import ewald_2d_pair_energy
            self._pot_chunk = jax.jit(
                lambda w: ewald_2d_pair_energy(w, tables),
            )

        # ---- SR step (jitted end-to-end, takes walkers & flat params)
        damping_arr = jnp.asarray(self.damping, dtype=jnp.float64)
        var_weight_arr = jnp.asarray(
            self.var_weight, dtype=jnp.float64,
        )
        n_cg_static = self.n_cg

        @jax.jit
        def sr_update_core(walkers, p_flat, e_loc):
            """Perform one SR step, given pre-computed E_L.

            Jacobian ``O`` is built here (inside jit) so it stays
            on-device without per-step Python dispatch.  Returns
            ``(δθ, e_mean, var, g_norm)``.
            """
            n = walkers.shape[0]

            # Jacobian — (n_walkers, n_params).  Fits in ~100 MB
            # at n=128, P=10⁵ (float64).  For larger networks
            # consider chunking over walkers.
            o = jax.vmap(log_psi_grad_flat, in_axes=(0, None))(
                walkers, p_flat,
            )

            e_mean = jnp.mean(e_loc)
            de = e_loc - e_mean
            var = jnp.mean(de ** 2)

            # Centred Jacobian: ⟨O⟩ subtracted so S = ⟨do · doᵀ⟩ / N.
            o_mean = jnp.mean(o, axis=0)
            do = o - o_mean[None, :]

            # Force vector  f = 2 · ⟨de · O⟩ = 2 · ⟨de · do⟩  (since
            # ⟨de⟩ = 0 makes the O-mean term drop out).  Mixed
            # objective: add β · 2 · ⟨(de² − var) · do⟩.
            de_mix = de + var_weight_arr * (de ** 2 - var)
            f = 2.0 * (de_mix @ do) / n

            # S·v without materialising S:  S·v = do.T·(do·v) / n.
            def matvec(v):
                return (do.T @ (do @ v)) / n + damping_arr * v

            # CG solve for δθ.
            dtheta = _cg_solve(matvec, f, n_iters=n_cg_static)

            g_norm = jnp.sqrt(jnp.sum(f ** 2))
            return dtheta, e_mean, var, g_norm

        self._sr_update_core = sr_update_core

    # -----------------------------------------------------
    # Walker management
    # -----------------------------------------------------

    def initialize_walkers(self, rng_key, num_walkers):
        """Sample walker positions for the SR optimiser.

        Crystal-aware: when ``config.envelope_type == 'crystal_gaussian'``
        walkers are placed at the triangular Bravais sites with a small
        Gaussian noise so MCMC immediately samples the dominant region
        of |psi|^2 (uniform-init walkers fail to bridge the ``a_NN``
        gap to localised |psi|^2 peaks within typical decorrelation
        steps and SR then trains the localised character away).
        """
        envelope_type = getattr(self.config, 'envelope_type', 'plane_wave')
        if envelope_type == 'crystal_gaussian' and self.dim == 2:
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

    def _pot_batch(self, w):
        """Chunked Ewald pair energy (matches Adam driver)."""
        n = w.shape[0]
        chunk = self._pot_chunk_size
        if n <= chunk:
            return self._pot_chunk(w)
        return jnp.concatenate([
            self._pot_chunk(w[k:k + chunk])
            for k in range(0, n, chunk)
        ], axis=0)

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
        metropolis_allw = self._metropolis_move_allw
        kin_batch = self._kin_batch
        sr_update_core = self._sr_update_core

        rng_key, init_key = jax.random.split(rng_key)
        walkers = self.initialize_walkers(init_key, num_walkers)
        step_size = jnp.asarray((3 * mc_timestep) ** 0.5)

        # Equilibration.
        ar = 0.0
        for _ in range(num_equil_steps):
            rng_key, sub = jax.random.split(rng_key)
            keys = jax.random.split(sub, num_walkers)
            walkers, acc = metropolis_allw(
                keys, walkers, step_size, params_flat,
            )
            ar = float(jnp.mean(acc))
            step_size = _adapt_step_size(step_size, ar)

        if verbose >= 1:
            print(
                f"# SR-VMC training — {self.n_params} params, "
                f"lr={self.lr}, damping={self.damping}, "
                f"n_cg={self.n_cg}, β={self.var_weight:.3g}",
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

        for it in range(1, num_iters + 1):
            # Decorrelate walkers.
            for _ in range(mcmc_decorr_steps):
                rng_key, sub = jax.random.split(rng_key)
                keys = jax.random.split(sub, num_walkers)
                walkers, acc = metropolis_allw(
                    keys, walkers, step_size, params_flat,
                )
                step_size = _adapt_step_size(
                    step_size, float(jnp.mean(acc)),
                )

            # Local energies (kin + pot).
            e_kin = kin_batch(walkers, params_flat)
            e_pot = self._pot_batch(walkers)
            e_loc = e_kin + e_pot

            # SR update.
            dtheta, e_mean, var, g_norm = sr_update_core(
                walkers, params_flat, e_loc,
            )
            params_flat = params_flat - self.lr * dtheta

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

        if not (fname_log is None
                or (isinstance(fname_log, str) and fname_log == "")):
            fout.close()

        self.params_flat = params_flat
        params_pytree = self.unravel(params_flat)

        # Save checkpoint so downstream tools (density plot, eval
        # restart, etc.) can recover the trained wavefunction.  The
        # HEG SR driver previously announced the file in run_heg_psiformer
        # but never actually wrote it.
        if (self.ofname_chkpt is not None
                and self.ofname_chkpt != ""):
            from .nn_checkpoint import save_nn_checkpoint
            # save_nn_checkpoint is a molecular-code helper that wants a
            # mol_info object with .n_up/.n_down/.charges/.coords for
            # the metadata header.  HEG has no nuclei -> pass an empty
            # stub with the right attribute surface so the call
            # succeeds end-to-end (params + meta).

            class _HEGMolInfoStub:
                def __init__(self, n_up, n_down, dim):
                    self.n_up = int(n_up)
                    self.n_down = int(n_down)
                    self.charges = np.zeros((0,), dtype=np.float64)
                    self.coords = np.zeros((0, dim), dtype=np.float64)

            stub = _HEGMolInfoStub(self.n_up, self.n_down, self.dim)
            try:
                save_nn_checkpoint(
                    self.ofname_chkpt, params_pytree,
                    epoch=int(len(e_history)),
                    config_name='HEG_PsiFormer',
                    mol_info=stub,
                    energy=(e_history[-1] if e_history else None),
                )
            except Exception as e:
                # Don't crash a long training run on save failure;
                # surface the error.
                print(
                    f"[warn] failed to save checkpoint "
                    f"{self.ofname_chkpt}: {e}",
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
):
    """Construct an SR-VMC optimiser for a HEG ansatz.

    Args: see :class:`_HEGSROptimizer`.

    Returns:
        :class:`_HEGSROptimizer` — callable.
    """
    if prefix.endswith(".chk.h5"):
        prefix = prefix[: -len(".chk.h5")]
    ofname_chkpt = prefix + ".chk.h5" if prefix else None
    return _HEGSROptimizer(
        config, init_key,
        lr=lr,
        damping=damping,
        n_cg=n_cg,
        var_weight=var_weight,
        ewald_n_real=ewald_n_real,
        ewald_n_recip=ewald_n_recip,
        ewald_eta=ewald_eta,
        ofname_chkpt=ofname_chkpt,
    )
