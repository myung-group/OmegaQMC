"""Adam-based VMC optimiser for the HEG neural-network wavefunction.

Parallels :mod:`~OmegaQMC.vmcopt_nn_iradam` but specialised to
:mod:`~OmegaQMC.vmc_nn_heg` and the plane-wave/Jastrow ansatz
from :mod:`~OmegaQMC.psi.nn.heg_wf`:

  * Metropolis resampling on the torus between optimisation passes.
  * Local energy = Ewald Coulomb + neural Laplacian kinetic.
  * Gradient of ``<E_L>`` via the standard VMC identity
    ``∇<E> = 2 ⟨(E_L - ⟨E_L⟩) · ∇ log|ψ|⟩``.
  * Optax Adam update on the flat parameter pytree.

This is not a full Stochastic-Reconfiguration optimiser — SR for
HEG is a straightforward adaptation of
:mod:`~OmegaQMC.vmcopt_nn_sr` once the core training loop here
is validated, and is deferred to a follow-up.
"""

from __future__ import annotations

import sys
from datetime import datetime
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np
import optax

from .psi.nn.heg_wf import HEGConfig, make_heg_log_psi
from .psi.nn.periodic import wrap_to_cell, make_cubic_lattice
from .psi.nn.physics import laplacian
from .observables.ewald import build_ewald_tables, ewald_pair_energy


_TARGET_ACCEPTANCE_RATE = 0.5
_STEP_SIZE_ADAPTATION_RATE = 0.05


def _adapt_step_size(step_size, acceptance_rate):
    return step_size * (
        1.0 + _STEP_SIZE_ADAPTATION_RATE
        * (acceptance_rate - _TARGET_ACCEPTANCE_RATE)
    )


class _HEGAdamOptimizer:
    """Minimal Adam VMC optimiser for HEG.

    Args:
        config: :class:`HEGConfig`.
        init_key: JAX PRNG key for parameter init.
        lr: Adam learning rate.
        ewald_n_real, ewald_n_recip: Ewald cutoffs.
        ewald_eta: Ewald splitting parameter.
        ofname_chkpt: Checkpoint file (``<prefix>.chk.h5``).
    """

    def __init__(
        self,
        config: HEGConfig,
        init_key,
        *,
        lr: float = 1e-3,
        ewald_n_real: int = 3,
        ewald_n_recip: int = 6,
        ewald_eta: Optional[float] = None,
        ofname_chkpt: str = "heg_vmcopt.chk.h5",
    ):
        self.config = config
        self.L = float(config.L)
        self.n_up = int(config.n_up)
        self.n_down = int(config.n_down)
        self.nelec = self.n_up + self.n_down
        self.lr = lr
        self.ofname_chkpt = ofname_chkpt

        self.lattice = make_cubic_lattice(self.L)
        self.ewald = build_ewald_tables(
            self.L, eta=ewald_eta,
            n_real=ewald_n_real, n_recip=ewald_n_recip,
        )

        log_psi, init_params, graphdef = make_heg_log_psi(
            config, init_key,
        )
        self.log_psi = log_psi
        self.graphdef = graphdef
        self.params = init_params

        tables = self.ewald
        lattice = self.lattice
        nelec = self.nelec

        @jax.jit
        def local_energy(r, params):
            def f_flat(r_flat):
                return log_psi(r_flat.reshape(nelec, 3), params)
            lap_fn = laplacian(f_flat)
            lap_val, grad_val = lap_fn(r.reshape(-1))
            kin = -0.5 * (lap_val + jnp.dot(grad_val, grad_val))
            pot = ewald_pair_energy(r, tables)
            return kin + pot

        @jax.jit
        def metropolis_move(rng_key, r, step_size, params):
            key_prop, key_acc = jax.random.split(rng_key)
            proposed = r + step_size * jax.random.normal(
                key_prop, r.shape,
            )
            proposed = wrap_to_cell(proposed, lattice)
            lp_old = log_psi(r, params)
            lp_new = log_psi(proposed, params)
            accept = jax.random.uniform(key_acc) < jnp.exp(
                2.0 * (lp_new - lp_old),
            )
            new = jnp.where(accept, proposed, r)
            return new, accept

        self._metropolis_move_allw = jax.vmap(
            metropolis_move, in_axes=(0, 0, None, None),
        )
        self.local_energy = local_energy

        # Gradient of <E> via the standard VMC identity.  For
        # params p and sampled walkers r_k ~ |ψ|²,
        #   ∂<E>/∂p = 2 ⟨(E_L(r) - ⟨E_L⟩) · ∂ log|ψ|/∂p⟩
        # which is an unbiased estimator of the analytic gradient.
        log_psi_grad = jax.grad(log_psi, argnums=1)

        @jax.jit
        def e_loc_batch(walkers, params):
            return jax.vmap(local_energy, in_axes=(0, None))(
                walkers, params,
            )

        @jax.jit
        def vmc_grad(walkers, params):
            e_loc = e_loc_batch(walkers, params)
            e_mean = jnp.mean(e_loc)
            de = e_loc - e_mean

            def single_grad(r):
                return log_psi_grad(r, params)
            # per-walker ∂log|ψ|/∂p pytree with leading walker axis.
            o_grads = jax.vmap(single_grad)(walkers)

            # Weighted mean: <(E_L - <E>) · O> over walkers, then × 2.
            def weighted_mean(g):
                # g shape: (n_walkers, *param_shape)
                axes_reduce = tuple(range(1, g.ndim))
                # multiply de along walker axis; broadcast as needed.
                de_expand = de.reshape(
                    (-1,) + (1,) * (g.ndim - 1),
                )
                return jnp.mean(2.0 * de_expand * g, axis=0)

            g_pytree = jax.tree_util.tree_map(
                weighted_mean, o_grads,
            )
            return g_pytree, e_mean, jnp.var(e_loc)

        self._vmc_grad = vmc_grad

        self.opt = optax.adam(lr)
        self.opt_state = self.opt.init(self.params)

    # -----------------------------------------------------
    # Walker management
    # -----------------------------------------------------

    def initialize_walkers(self, rng_key, num_walkers):
        return self.L * jax.random.uniform(
            rng_key, (num_walkers, self.nelec, 3),
        )

    # -----------------------------------------------------
    # Training loop
    # -----------------------------------------------------

    def __call__(
        self,
        rng_key,
        num_iters: int = 200,
        num_walkers: int = 128,
        mcmc_decorr_steps: int = 20,
        num_equil_steps: int = 400,
        mc_timestep: float = 0.1,
        fname_log: Optional[str] = None,
        verbose: int = 1,
    ):
        """Run Adam-VMC optimisation.

        Args:
            rng_key: JAX PRNG key (int or array).
            num_iters: Number of parameter updates.
            num_walkers: MCMC walkers.
            mcmc_decorr_steps: MCMC steps between parameter updates.
            num_equil_steps: Initial equilibration steps.
            mc_timestep: Initial MC timestep.
            fname_log: Per-iter log path; ``None`` → stdout.
            verbose: Verbosity (0 silent).
        """
        if isinstance(rng_key, int):
            rng_key = jax.random.key(rng_key)
        if (fname_log is None
                or (isinstance(fname_log, str) and fname_log == "")):
            fout = sys.stdout
        else:
            fout = open(fname_log, 'w', 1)

        params = self.params
        opt_state = self.opt_state
        metropolis_allw = self._metropolis_move_allw
        vmc_grad = self._vmc_grad

        rng_key, init_key = jax.random.split(rng_key)
        walkers = self.initialize_walkers(init_key, num_walkers)
        step_size = jnp.asarray((3 * mc_timestep) ** 0.5)

        # Equilibration
        for _ in range(num_equil_steps):
            rng_key, sub = jax.random.split(rng_key)
            keys = jax.random.split(sub, num_walkers)
            walkers, acc = metropolis_allw(
                keys, walkers, step_size, params,
            )
            step_size = _adapt_step_size(step_size, float(jnp.mean(acc)))

        if verbose >= 1:
            print(
                "# iter       <E>/N            var(E)          "
                "|g|             step          dt",
                file=fout,
            )
        e_history = []
        timestamp_prev = datetime.now()

        for it in range(1, num_iters + 1):
            # Decorrelate walkers.
            for _ in range(mcmc_decorr_steps):
                rng_key, sub = jax.random.split(rng_key)
                keys = jax.random.split(sub, num_walkers)
                walkers, acc = metropolis_allw(
                    keys, walkers, step_size, params,
                )
                step_size = _adapt_step_size(
                    step_size, float(jnp.mean(acc)),
                )

            # VMC gradient.
            grads, e_mean, e_var = vmc_grad(walkers, params)
            updates, opt_state = self.opt.update(grads, opt_state)
            params = optax.apply_updates(params, updates)

            # Per-iter logging.
            g_norm = jnp.sqrt(sum(
                jnp.sum(g * g)
                for g in jax.tree_util.tree_leaves(grads)
            ))
            now = datetime.now()
            dt = (now - timestamp_prev).total_seconds()
            timestamp_prev = now
            e_history.append(float(e_mean) / self.nelec)
            if verbose >= 1 and (it <= 10 or it % 10 == 0):
                print(
                    f"{it:>6d}  {float(e_mean)/self.nelec:>14.8e}  "
                    f"{float(e_var):>12.4e}  "
                    f"{float(g_norm):>12.4e}  "
                    f"{float(step_size):>10.4f}  "
                    f"{dt:>8.3f}",
                    file=fout,
                )

        if not (fname_log is None
                or (isinstance(fname_log, str) and fname_log == "")):
            fout.close()

        self.params = params
        self.opt_state = opt_state
        return {
            'params': params,
            'E_per_elec_history': e_history,
            'E_final_ha': e_history[-1] if e_history else None,
        }


def get_vmcopt_nn_heg_func(
    config: HEGConfig,
    init_key,
    *,
    prefix: str = "heg_vmcopt",
    lr: float = 1e-3,
    ewald_n_real: int = 3,
    ewald_n_recip: int = 6,
    ewald_eta: Optional[float] = None,
):
    """Construct an Adam-VMC optimiser for a HEG ansatz.

    Args:
        config: :class:`HEGConfig`.
        init_key: JAX PRNG key.
        prefix: Checkpoint stem.
        lr: Adam learning rate.
        ewald_n_real, ewald_n_recip, ewald_eta: Ewald tuning.

    Returns:
        :class:`_HEGAdamOptimizer` — call with
        ``opt(rng_key, num_iters=..., ...)``.
    """
    if prefix.endswith(".chk.h5"):
        prefix = prefix[:-len(".chk.h5")]
    return _HEGAdamOptimizer(
        config, init_key,
        lr=lr,
        ewald_n_real=ewald_n_real,
        ewald_n_recip=ewald_n_recip,
        ewald_eta=ewald_eta,
        ofname_chkpt=prefix + ".chk.h5",
    )
