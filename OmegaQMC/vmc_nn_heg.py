"""VMC driver for the neural-network HEG trial wavefunction.

Companion to :mod:`~OmegaQMC.vmc_nn` but specialized for the
homogeneous electron gas:

  * No nuclei — no electron-nuclear potential and no nuclear
    repulsion term.
  * Electron-electron interaction uses Ewald summation on a cubic
    simulation cell (:mod:`~OmegaQMC.observables.ewald`).
  * Metropolis-Hastings walkers are wrapped to the primitive cell
    after each step to avoid unbounded drift over long runs.
  * No fragment symmetry / PGCS — jellium is translationally
    invariant, and symmetry averaging is supplied externally via
    twist-averaged boundary conditions if desired.

Designed to be callable with a :class:`~OmegaQMC.psi.nn.heg_wf.HEGConfig`
and produce a checkpointable driver with the same per-block
logging and binning-analysis output as the molecular driver.
"""

from __future__ import annotations

import sys
from datetime import datetime
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np

from .psi.nn.heg_wf import HEGConfig, make_heg_log_psi
from .psi.nn.periodic import wrap_to_cell, make_cubic_lattice
from .psi.nn.physics import laplacian
from .observables.ewald import (
    EwaldTables, build_ewald_tables, ewald_pair_energy,
)
from .utils import do_binning_analysis, _make_sharding


TARGET_ACCEPTANCE_RATE = 0.5
STEP_SIZE_ADAPTATION_RATE = 0.05


def _adapt_step_size(step_size, acceptance_rate):
    return step_size * (
        1.0 + STEP_SIZE_ADAPTATION_RATE
        * (acceptance_rate - TARGET_ACCEPTANCE_RATE)
    )


class _VMCDriverHEG:
    """VMC driver for a HEG trial wavefunction.

    Args:
        config: :class:`HEGConfig`.
        init_key: JAX PRNG key for parameter init.
        ewald_n_real: Real-space Ewald cutoff (lattice units).
        ewald_n_recip: Reciprocal-space Ewald cutoff (integer units).
        ewald_eta: Ewald splitting parameter (default ``√π / L``).
        ofname_chkpt: Checkpoint path stem.
    """

    def __init__(
        self,
        config: HEGConfig,
        init_key,
        *,
        ewald_n_real: int = 3,
        ewald_n_recip: int = 6,
        ewald_eta: Optional[float] = None,
        ofname_chkpt: str = "heg_vmc.chk.h5",
    ):
        self.config = config
        self.L = float(config.L)
        self.n_up = int(config.n_up)
        self.n_down = int(config.n_down)
        self.nelec = self.n_up + self.n_down
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

        lattice = self.lattice
        tables = self.ewald
        nelec = self.nelec

        @jax.jit
        def energy_potential(r):
            return ewald_pair_energy(r, tables)

        @jax.jit
        def energy_kinetic(r, params):
            def f_flat(r_flat):
                return log_psi(r_flat.reshape(nelec, 3), params)
            lap_fn = laplacian(f_flat)
            lap_val, grad_val = lap_fn(r.reshape(-1))
            return -0.5 * (lap_val + jnp.dot(grad_val, grad_val))

        @jax.jit
        def local_energy(r, params):
            return energy_potential(r) + energy_kinetic(r, params)

        self.energy_potential = energy_potential
        self.energy_kinetic = energy_kinetic
        self.local_energy = local_energy

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

        self._metropolis_move = metropolis_move
        self._metropolis_move_allw = jax.vmap(
            metropolis_move, in_axes=(0, 0, None, None),
        )

    # ---- Walkers ----

    def initialize_walkers(self, rng_key, num_walkers):
        """Uniformly sample electron positions inside the cell."""
        shape = (num_walkers, self.nelec, 3)
        return self.L * jax.random.uniform(rng_key, shape)

    def load_checkpoint(self, filepath):
        from .nn_checkpoint import load_nn_checkpoint
        params, meta = load_nn_checkpoint(filepath, self.params)
        self.params = params
        return meta

    # ---- Run ----

    def __call__(
        self,
        rng_key,
        num_walkers: int = 256,
        num_steps_per_block: int = 50,
        num_steps_decorr: int = 1,
        num_blocks: int = 50,
        num_blocks_equil: int = 10,
        mc_timestep: float = 0.1,
        fname_log: Optional[str] = None,
        verbose: int = 1,
    ):
        """Run the VMC simulation.

        Args:
            rng_key: JAX PRNG key (int or array).
            num_walkers: Number of MCMC walkers.
            num_steps_per_block: Moves per block.
            num_steps_decorr: Decorrelation moves between
                local-energy evaluations.
            num_blocks: Production blocks.
            num_blocks_equil: Equilibration blocks.
            mc_timestep: Initial MC timestep (step_size ≈ √(3 Δt)).
            fname_log: Per-block log path; ``None`` → stdout.
            verbose: Verbosity level (0 silent).

        Returns:
            Dict with keys ``'E_mean'``, ``'E_serr'``,
            ``'E_blocks'``, ``'E_per_elec_ha'``, ``'E_per_elec_ry'``.
        """
        if isinstance(rng_key, int):
            rng_key = jax.random.key(rng_key)
        if (fname_log is None
                or (isinstance(fname_log, str) and fname_log == "")):
            fout = sys.stdout
        else:
            fout = open(fname_log, 'w', 1)

        params = self.params
        metropolis_move_allw = self._metropolis_move_allw
        energy_potential = self.energy_potential
        energy_kinetic = self.energy_kinetic

        timestamp_init = datetime.now()

        rng_key, init_key = jax.random.split(rng_key)
        walkers = self.initialize_walkers(init_key, num_walkers)
        walkers_sharding, walker_keys_sharding = _make_sharding(
            num_walkers,
        )
        if walkers_sharding is not None:
            walkers = jax.device_put(walkers, walkers_sharding)
        mc_stepsize = (3 * mc_timestep) ** 0.5

        @jax.jit
        def eq_step(state, _):
            rk, w, s = state
            rk, key = jax.random.split(rk)
            keys = jax.random.split(key, num_walkers)
            if walker_keys_sharding is not None:
                keys = jax.lax.with_sharding_constraint(
                    keys, walker_keys_sharding,
                )
            nw, acc = metropolis_move_allw(keys, w, s, params)
            ar = acc.mean()
            ns = _adapt_step_size(s, ar)
            return (rk, nw, ns), ar

        for _ in range(num_blocks_equil):
            state = (rng_key, walkers, mc_stepsize)
            state, ratios = jax.lax.scan(
                eq_step, state, jnp.arange(num_steps_per_block),
            )
            rng_key, walkers, mc_stepsize = state

        if verbose >= 1:
            print(f"# Equilibration acceptance (last block): "
                  f"{float(ratios[-1]):.3f}", file=fout)
            print(f"# Adapted step size: {float(mc_stepsize):.4f} Bohr",
                  file=fout)

        @jax.jit
        def prod_step(state, _):
            rk, w, s = state
            for _ in range(num_steps_decorr):
                rk, key = jax.random.split(rk)
                keys = jax.random.split(key, num_walkers)
                if walker_keys_sharding is not None:
                    keys = jax.lax.with_sharding_constraint(
                        keys, walker_keys_sharding,
                    )
                nw, acc = metropolis_move_allw(keys, w, s, params)
                w = nw
            ar = acc.mean()
            e_pot = jax.vmap(energy_potential)(nw)
            e_kin = jax.vmap(
                energy_kinetic, in_axes=(0, None),
            )(nw, params)
            return (rk, nw, s), (ar, e_pot, e_kin)

        if verbose >= 1:
            print(
                "# block   E_loc_mean        E_loc_std"
                "       E_pot            E_kin            Δt_block",
                file=fout,
            )

        E_blocks = []
        timestamp_prev = datetime.now()

        for blk in range(1, num_blocks + 1):
            state = (rng_key, walkers, mc_stepsize)
            state, (ratios, e_pot, e_kin) = jax.lax.scan(
                prod_step, state, jnp.arange(num_steps_per_block),
            )
            rng_key, walkers, _ = state

            E_loc = e_pot + e_kin     # (steps, walkers)
            E_step = E_loc.mean(axis=1)
            E_mean = E_step.mean()
            E_std = E_step.std()
            E_blocks.append(float(E_mean))

            now = datetime.now()
            dt = (now - timestamp_prev).total_seconds()
            if verbose >= 1:
                print(
                    f"{blk:>7d}  {float(E_mean):>16.8e}  "
                    f"{float(E_std):>14.6e}  "
                    f"{float(e_pot.mean()):>14.6e}  "
                    f"{float(e_kin.mean()):>14.6e}  "
                    f"{dt:>10.4f}",
                    file=fout,
                )
            timestamp_prev = now

        if not (fname_log is None
                or (isinstance(fname_log, str) and fname_log == "")):
            fout.close()

        E_arr = jnp.array(E_blocks)
        e_mean, e_serr, _, e_kappa = do_binning_analysis(E_arr)
        N = self.nelec
        e_ha = float(e_mean) / N
        e_ry = e_ha * 2.0

        timestamp_fin = datetime.now()
        elapsed = (timestamp_fin - timestamp_init).total_seconds()
        if verbose >= 1:
            print(f"\nVMC energy: {float(e_mean):.8f} "
                  f"+/- {float(e_serr):.8f} Ha "
                  f"({float(E_arr.shape[0] / e_kappa):.1f} eff blocks)")
            print(f"E/N = {e_ha:.8f} Ha/elec  = {e_ry:.8f} Ry/elec")
            print(f"Total time: {elapsed:.2f} s")

        return {
            'E_mean': float(e_mean),
            'E_serr': float(e_serr),
            'E_blocks': E_blocks,
            'E_per_elec_ha': e_ha,
            'E_per_elec_ry': e_ry,
        }


def get_vmc_nn_heg_func(
    config: HEGConfig,
    init_key,
    *,
    prefix: str = "heg_vmc",
    ewald_n_real: int = 3,
    ewald_n_recip: int = 6,
    ewald_eta: Optional[float] = None,
):
    """Construct a VMC driver for a HEG NN trial wavefunction.

    Args:
        config: :class:`HEGConfig`.
        init_key: JAX PRNG key for parameter init.
        prefix: Output filename stem
            (``<prefix>.chk.h5`` for checkpoints).
        ewald_n_real: Real-space Ewald cutoff (lattice units).
        ewald_n_recip: Reciprocal-space Ewald cutoff (integer units).
        ewald_eta: Ewald splitting parameter (default ``√π / L``).

    Returns:
        :class:`_VMCDriverHEG` instance — call with ``driver(rng_key, ...)``.
    """
    if prefix.endswith(".chk.h5"):
        prefix = prefix[:-len(".chk.h5")]
    ofname_chkpt = prefix + ".chk.h5"
    return _VMCDriverHEG(
        config, init_key,
        ewald_n_real=ewald_n_real,
        ewald_n_recip=ewald_n_recip,
        ewald_eta=ewald_eta,
        ofname_chkpt=ofname_chkpt,
    )
