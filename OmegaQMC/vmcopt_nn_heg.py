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

from .psi.nn.heg_wf import HEGConfig, make_heg_log_psi_any as make_heg_log_psi
from .psi.nn.periodic import wrap_to_cell, make_cubic_lattice
from .psi.nn.physics import laplacian
from .observables.ewald import build_ewald_tables, ewald_pair_energy
from .observables.ewald_dispatch import (
    build_ewald_tables_dim, ewald_pair_energy_dim,
)


_TARGET_ACCEPTANCE_RATE = 0.5
_STEP_SIZE_ADAPTATION_RATE = 0.05


def _adapt_step_size(step_size, acceptance_rate):
    return step_size * (
        1.0 + _STEP_SIZE_ADAPTATION_RATE
        * (acceptance_rate - _TARGET_ACCEPTANCE_RATE)
    )


class _VMCOptDriverNNHEG_Adam:
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
        var_weight: float = 0.0,
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
        self.dim = int(getattr(config, 'dim', 3))
        self.lr = lr
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

        log_psi, init_params, graphdef = make_heg_log_psi(
            config, init_key,
        )
        self.log_psi = log_psi
        self.graphdef = graphdef
        self.params = init_params

        tables = self.ewald
        lattice = self.lattice
        nelec = self.nelec
        dim = self.dim

        # The Ewald reciprocal-sum at (n_real=3, n_recip=6) has 2196
        # G-vectors.  Under ``jax.vmap`` over walkers, XLA's fusion
        # heuristic on the first-ever compile in a Python process
        # takes ~130 s.  A direct batched call (no vmap — the
        # function broadcasts over leading dims natively) compiles
        # in ~0.6 s.  We therefore compute kinetic *per-walker with
        # vmap* and potential *in one batched call*, then add.
        if dim == 3:
            _ewald_pair = ewald_pair_energy
        else:
            from .observables.ewald_2d import ewald_2d_pair_energy
            _ewald_pair = ewald_2d_pair_energy

        def kin_only(r, params):
            def f_flat(r_flat):
                return log_psi(r_flat.reshape(nelec, dim), params)
            lap_fn = laplacian(f_flat)
            lap_val, grad_val = lap_fn(r.reshape(-1))
            return -0.5 * (lap_val + jnp.dot(grad_val, grad_val))

        def local_energy(r, params):
            """Per-walker local energy (kept for external callers)."""
            return kin_only(r, params) + _ewald_pair(r, tables)

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

        self._metropolis_move_allw = jax.jit(jax.vmap(
            metropolis_move, in_axes=(0, 0, None, None),
        ))
        self.local_energy = local_energy

        # Gradient of the *mixed* VMC objective
        #
        #   L(θ) = ⟨E⟩(θ)  +  β · Var(E_L)(θ)
        #
        # via the standard reweighting identities.  At β=0 this is
        # the plain energy gradient (bit-identical to the previous
        # implementation); at β>0 it adds an Umrigar-style variance
        # penalty.
        #
        # Derivation.  Applying the VMC identity
        #   ∇_θ ⟨f(r)⟩_{|ψ|²} = ⟨∂f/∂θ⟩ + 2 ⟨(f − ⟨f⟩) · O⟩
        # where O = ∇ log|ψ|, and dropping the direct ∂E_L/∂θ term
        # (requires an expensive second derivative and is small when
        # E_L is close to a local eigenvalue):
        #   ∇⟨E⟩  ≈ 2 ⟨(E − ⟨E⟩) · O⟩
        #   ∇Var ≈ 2 ⟨((E − ⟨E⟩)² − Var) · O⟩
        #        ≡ 2 ⟨(δE² − ⟨δE²⟩) · O⟩
        # so
        #   ∇L ≈ 2 ⟨ [δE + β·(δE² − ⟨δE²⟩)] · O ⟩
        # The bracket is computed once per walker and used as a
        # single weight in a per-leaf reweighted-average.
        #
        # Pipeline split into four small JITs to avoid the XLA
        # fusion bomb that fires if ``vmap(ewald_pair_energy)`` is
        # compiled anywhere (130 s cold-compile vs 0.6 s for the
        # un-vmapped direct batched call).
        log_psi_grad = jax.grad(log_psi, argnums=1)

        _kin_batch = jax.jit(jax.vmap(kin_only, in_axes=(0, None)))
        _grads_batch = jax.jit(
            jax.vmap(log_psi_grad, in_axes=(0, None)),
        )

        # Chunk the Ewald pot to avoid a ~130 s cold-compile.  At
        # (n_real=3, n_recip=6) XLA's fusion heuristic on a full
        # 128-walker batch takes ~130 s, while a 32-walker chunk
        # compiles in ~3 s and is reused across all chunks.
        _pot_chunk_size = 32
        _pot_chunk = jax.jit(lambda w: _ewald_pair(w, tables))

        def _pot_batch(w):
            n = w.shape[0]
            if n <= _pot_chunk_size:
                return _pot_chunk(w)
            return jnp.concatenate([
                _pot_chunk(w[k:k + _pot_chunk_size])
                for k in range(0, n, _pot_chunk_size)
            ], axis=0)

        @jax.jit
        def _combine(e_loc, o_grads, var_weight_scalar):
            e_mean = jnp.mean(e_loc)
            de = e_loc - e_mean
            var = jnp.mean(de ** 2)

            # Per-walker weight for the mixed gradient.  At β = 0
            # this collapses to ``de`` so the pure-energy path is
            # recovered exactly.
            w_per_walker = de + var_weight_scalar * (de ** 2 - var)

            def weighted_mean(g):
                w_expand = w_per_walker.reshape(
                    (-1,) + (1,) * (g.ndim - 1),
                )
                return jnp.mean(2.0 * w_expand * g, axis=0)

            g_pytree = jax.tree_util.tree_map(
                weighted_mean, o_grads,
            )
            return g_pytree, e_mean, var

        # Pass var_weight as a traced scalar so one JIT compilation
        # handles all values (including 0.0 which recovers the
        # pure-energy gradient).
        var_weight_arr = jnp.asarray(self.var_weight, dtype=jnp.float64)

        def vmc_grad(walkers, params):
            e_kin = _kin_batch(walkers, params)
            e_pot = _pot_batch(walkers)
            e_loc = e_kin + e_pot
            o_grads = _grads_batch(walkers, params)
            return _combine(e_loc, o_grads, var_weight_arr)

        self._vmc_grad = vmc_grad

        self.opt = optax.adam(lr)
        self.opt_state = self.opt.init(self.params)

    # -----------------------------------------------------
    # Walker management
    # -----------------------------------------------------

    def initialize_walkers(self, rng_key, num_walkers):
        """Sample walker positions for the Adam optimiser.

        Crystal-aware: see :class:`_VMCOptDriverNNHEG_SR.initialize_walkers`
        for rationale.  Walkers placed at triangular Bravais sites
        with small noise when ``envelope_type='crystal_gaussian'``.
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
            header = (
                "# iter       <E>/N            var(E)          "
                "|g|             step          dt"
            )
            if self.var_weight > 0.0:
                header = (
                    f"# iter       <E>/N            var(E)"
                    f"         L/N (β={self.var_weight:.3g})     "
                    f"|g|             step          dt"
                )
            print(header, file=fout)
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
                e_per = float(e_mean) / self.nelec
                if self.var_weight > 0.0:
                    loss_per = (
                        e_per + self.var_weight * float(e_var)
                        / self.nelec
                    )
                    print(
                        f"{it:>6d}  {e_per:>14.8e}  "
                        f"{float(e_var):>12.4e}  "
                        f"{loss_per:>14.8e}  "
                        f"{float(g_norm):>12.4e}  "
                        f"{float(step_size):>10.4f}  "
                        f"{dt:>8.3f}",
                        file=fout,
                    )
                else:
                    print(
                        f"{it:>6d}  {e_per:>14.8e}  "
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
    var_weight: float = 0.0,
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
        var_weight: Weight β of the Umrigar-style variance term in
            the mixed objective ``L = ⟨E⟩ + β · Var(E_L)``.  Default
            0.0 (pure energy, bit-identical to earlier runs).
            Typical values: 0.01-0.1 for stabilisation.  The exact
            eigenstate has both ``⟨E⟩ → E_0`` and ``Var(E_L) → 0``,
            so adding the variance term does not shift the true
            minimum — it reshapes the loss landscape to favour
            solutions that are simultaneously low-energy and
            low-variance, which usually converges faster.
        ewald_n_real, ewald_n_recip, ewald_eta: Ewald tuning.

    Returns:
        :class:`_VMCOptDriverNNHEG_Adam` — call with
        ``opt(rng_key, num_iters=..., ...)``.
    """
    if prefix.endswith(".chk.h5"):
        prefix = prefix[:-len(".chk.h5")]
    return _VMCOptDriverNNHEG_Adam(
        config, init_key,
        lr=lr,
        var_weight=var_weight,
        ewald_n_real=ewald_n_real,
        ewald_n_recip=ewald_n_recip,
        ewald_eta=ewald_eta,
        ofname_chkpt=prefix + ".chk.h5",
    )
