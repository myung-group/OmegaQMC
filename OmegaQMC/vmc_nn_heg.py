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

from .psi.nn.heg_wf import (
    HEGConfig,
    make_heg_log_psi_any as make_heg_log_psi,
    make_heg_log_psi_complex,
    transfer_jastrow_params,
    transfer_trained_params,
)
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
        self.dim = int(getattr(config, 'dim', 3))
        self.ofname_chkpt = ofname_chkpt

        if self.dim == 3:
            self.lattice = make_cubic_lattice(self.L)
            self.ewald = build_ewald_tables(
                self.L, eta=ewald_eta,
                n_real=ewald_n_real, n_recip=ewald_n_recip,
            )
            _ewald_pair = ewald_pair_energy
        else:
            from .psi.nn.periodic import make_square_lattice
            from .observables.ewald_2d import (
                build_ewald_2d_tables, ewald_2d_pair_energy,
            )
            self.lattice = make_square_lattice(self.L)
            self.ewald = build_ewald_2d_tables(
                self.L, eta=ewald_eta,
                n_real=ewald_n_real, n_recip=ewald_n_recip,
            )
            _ewald_pair = ewald_2d_pair_energy

        log_psi, init_params, graphdef = make_heg_log_psi(
            config, init_key,
        )
        self.log_psi = log_psi
        self.graphdef = graphdef
        self.params = init_params

        lattice = self.lattice
        tables = self.ewald
        nelec = self.nelec
        dim = self.dim
        # Closure for downstream chunked Ewald usage further down.
        self._ewald_pair_fn = _ewald_pair

        # Pure-Python closures: inner ``@jax.jit`` inside an outer
        # ``jax.jit(jax.vmap(f))`` triggers a pathological XLA fusion
        # on large reciprocal grids.  Jit only at the outermost layer.
        def energy_potential(r):
            return _ewald_pair(r, tables)

        def energy_kinetic(r, params):
            def f_flat(r_flat):
                return log_psi(r_flat.reshape(nelec, dim), params)
            lap_fn = laplacian(f_flat)
            lap_val, grad_val = lap_fn(r.reshape(-1))
            return -0.5 * (lap_val + jnp.dot(grad_val, grad_val))

        def local_energy(r, params):
            return energy_potential(r) + energy_kinetic(r, params)

        # Single-walker jitted wrappers for external use (e.g. the
        # integration test that calls ``driver.local_energy`` on one r).
        self.energy_potential = jax.jit(energy_potential)
        self.energy_kinetic = jax.jit(energy_kinetic)
        self.local_energy = jax.jit(local_energy)

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

        self._metropolis_move = jax.jit(metropolis_move)
        self._metropolis_move_allw = jax.jit(jax.vmap(
            metropolis_move, in_axes=(0, 0, None, None),
        ))

    # ---- Walkers ----

    def initialize_walkers(self, rng_key, num_walkers):
        """Sample electron positions inside the cell.

        Dispatch order:
        1. Explicit ``config.walker_init`` (``'uniform'`` or
           ``'crystal_perturbed'``) wins.
        2. Otherwise fall back to envelope-based default
           (``crystal_gaussian`` → ``crystal_perturbed``, else
           ``uniform``).
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
        shape = (num_walkers, self.nelec, self.dim)
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
        init_walkers=None,
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
        if init_walkers is not None:
            walkers = jnp.asarray(init_walkers["R"])
            # tile/truncate to num_walkers if needed
            if walkers.shape[0] != num_walkers:
                if walkers.shape[0] < num_walkers:
                    reps = (num_walkers + walkers.shape[0] - 1) // walkers.shape[0]
                    walkers = jnp.tile(
                        walkers, (reps,) + (1,) * (walkers.ndim - 1),
                    )[:num_walkers]
                else:
                    walkers = walkers[:num_walkers]
            mc_stepsize = float(init_walkers.get(
                "step_size", (3 * mc_timestep) ** 0.5,
            ))
            warm_start = True
        else:
            walkers = self.initialize_walkers(init_key, num_walkers)
            mc_stepsize = (3 * mc_timestep) ** 0.5
            warm_start = False
        walkers_sharding, walker_keys_sharding = _make_sharding(
            num_walkers,
        )
        if walkers_sharding is not None:
            walkers = jax.device_put(walkers, walkers_sharding)

        # Compiling a full ``lax.scan`` over ``num_steps_per_block`` with
        # a vmapped-Laplacian body fuses tens of thousands of ops into a
        # single XLA module (multi-minute compile).  Instead we JIT a
        # *single* MCMC/energy step and drive the loop from Python —
        # same wall-clock per step, tiny compile cost.
        #
        # Additional subtlety: wrapping a ``jax.vmap(inner_jit)`` inside
        # an *outer* ``@jax.jit`` triggers a pathologically-large fused
        # XLA graph for the Laplacian on large walker batches
        # (>10-minute compile at n_walkers=256 for N=14 electrons).
        # Keeping the vmap+inner-jit composition *outside* of any outer
        # jit avoids the blow-up; the inner ``energy_potential`` and
        # ``energy_kinetic`` are already jitted in ``__init__``.
        @jax.jit
        def _mcmc_one_step(rk, w, s, params):
            rk, key = jax.random.split(rk)
            keys = jax.random.split(key, num_walkers)
            if walker_keys_sharding is not None:
                keys = jax.lax.with_sharding_constraint(
                    keys, walker_keys_sharding,
                )
            nw, acc = metropolis_move_allw(keys, w, s, params)
            return rk, nw, acc.mean()

        # Potential uses ewald_pair_energy's broadcasting on the leading
        # dim, but a naive full-batch call at (n_real=3, n_recip=6) on
        # 256 walkers makes XLA's fusion heuristic explode — measured
        # >9 min compile time per call, vs <5 s if we split the walkers
        # into chunks.  Using ``jax.vmap`` here has the same pathology.
        # Kinetic vmap is unaffected (per-walker Laplacian fori_loop
        # dominates, compile ~2 s at 256 walkers).
        tables_ = self.ewald
        _ewald_pair_chunk_fn = self._ewald_pair_fn
        chunk = 32

        @jax.jit
        def _pot_chunk(w_chunk):
            return _ewald_pair_chunk_fn(w_chunk, tables_)

        def _pot_batch(w):
            n = w.shape[0]
            if n <= chunk:
                return _pot_chunk(w)
            return jnp.concatenate([
                _pot_chunk(w[k:k + chunk]) for k in range(0, n, chunk)
            ], axis=0)

        _kin_batch = jax.vmap(energy_kinetic, in_axes=(0, None))

        # Warm-started runs skip most equilibration since walkers come
        # pre-equilibrated from training; we still run ONE block to
        # decorrelate from training trajectory.  Cold start runs the
        # full requested equilibration.
        n_equil_blocks = 1 if warm_start else num_blocks_equil
        for _ in range(n_equil_blocks):
            ar = 0.0
            for _s in range(num_steps_per_block):
                rng_key, walkers, ar = _mcmc_one_step(
                    rng_key, walkers, mc_stepsize, params,
                )
                mc_stepsize = _adapt_step_size(mc_stepsize, float(ar))

        if verbose >= 1:
            print(f"# Equilibration acceptance (last block): "
                  f"{float(ar):.3f}", file=fout)
            print(f"# Adapted step size: {float(mc_stepsize):.4f} Bohr",
                  file=fout)

        if verbose >= 1:
            print(
                "# block   E_loc_mean        E_loc_std"
                "       E_pot            E_kin            Δt_block",
                file=fout,
            )

        E_blocks = []
        timestamp_prev = datetime.now()

        for blk in range(1, num_blocks + 1):
            e_pot_steps, e_kin_steps = [], []
            for _step in range(num_steps_per_block):
                # Decorrelation moves (kept as Python loop).
                for _ in range(num_steps_decorr):
                    rng_key, walkers, _ = _mcmc_one_step(
                        rng_key, walkers, mc_stepsize, params,
                    )
                e_pot = _pot_batch(walkers)
                e_kin = _kin_batch(walkers, params)
                e_pot_steps.append(e_pot)
                e_kin_steps.append(e_kin)
            e_pot = jnp.stack(e_pot_steps, axis=0)
            e_kin = jnp.stack(e_kin_steps, axis=0)

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


# =====================================================================
# Twist-averaged (complex-ansatz) driver
# =====================================================================

class _VMCDriverHEGTwist:
    """VMC driver for a complex (twist-averaged-ready) HEG ansatz.

    Identical in structure to :class:`_VMCDriverHEG` but wired to
    :func:`make_heg_log_psi_complex` instead of the real pathway, so
    it natively handles an arbitrary Bloch twist ``κ`` and a
    complex-valued trial wavefunction.

    Args:
        config: :class:`HEGConfig`.
        init_key: JAX PRNG key for parameter init.
        kappa: Twist ``(3,)`` in fractional coordinates
            ``[-0.5, 0.5)``.  Default ``(0, 0, 0)``.
        ewald_*: as :class:`_VMCDriverHEG`.
    """

    def __init__(
        self,
        config: HEGConfig,
        init_key,
        *,
        kappa=(0.0, 0.0, 0.0),
        ewald_n_real: int = 3,
        ewald_n_recip: int = 6,
        ewald_eta: Optional[float] = None,
    ):
        self.config = config
        self.L = float(config.L)
        self.n_up = int(config.n_up)
        self.n_down = int(config.n_down)
        self.nelec = self.n_up + self.n_down
        self.dim = int(getattr(config, 'dim', 3))
        self.kappa = tuple(float(x) for x in kappa)

        if self.dim == 3:
            self.lattice = make_cubic_lattice(self.L)
            self.ewald = build_ewald_tables(
                self.L, eta=ewald_eta,
                n_real=ewald_n_real, n_recip=ewald_n_recip,
            )
            _ewald_pair = ewald_pair_energy
        else:
            from .psi.nn.periodic import make_square_lattice
            from .observables.ewald_2d import (
                build_ewald_2d_tables, ewald_2d_pair_energy,
            )
            self.lattice = make_square_lattice(self.L)
            self.ewald = build_ewald_2d_tables(
                self.L, eta=ewald_eta,
                n_real=ewald_n_real, n_recip=ewald_n_recip,
            )
            _ewald_pair = ewald_2d_pair_energy
        self._ewald_pair_fn = _ewald_pair

        log_psi_complex, init_params, graphdef = (
            make_heg_log_psi_complex(config, init_key, kappa=kappa)
        )
        self.log_psi_complex = log_psi_complex
        self.graphdef = graphdef
        self.params = init_params

        tables = self.ewald
        lattice = self.lattice
        nelec = self.nelec
        dim = self.dim

        # ``u = Re(log ψ)``, ``v = Im(log ψ)`` — real-valued scalar
        # functions of real r used by the complex-kinetic formula
        # ``E_K = -½ (∇² u + |∇u|² − |∇v|²)``.
        def u_fn(r, params):
            return jnp.real(log_psi_complex(r, params))

        def v_fn(r, params):
            return jnp.imag(log_psi_complex(r, params))

        def energy_potential(r):
            return _ewald_pair(r, tables)

        def energy_kinetic(r, params):
            def u_flat(r_flat):
                return u_fn(r_flat.reshape(nelec, dim), params)

            def v_flat(r_flat):
                return v_fn(r_flat.reshape(nelec, dim), params)

            r_flat = r.reshape(-1)
            lap_u, grad_u = laplacian(u_flat)(r_flat)
            lap_v, grad_v = laplacian(v_flat)(r_flat)
            return -0.5 * (
                lap_u
                + jnp.dot(grad_u, grad_u)
                - jnp.dot(grad_v, grad_v)
            )

        def local_energy(r, params):
            return energy_potential(r) + energy_kinetic(r, params)

        self.energy_potential = jax.jit(energy_potential)
        self.energy_kinetic = jax.jit(energy_kinetic)
        self.local_energy = jax.jit(local_energy)

        # Metropolis uses log|ψ| = Re(log ψ).  The phase drops out of
        # the |ψ|² ratio so the sampler is unchanged from the real
        # pathway.
        def metropolis_move(rng_key, r, step_size, params):
            key_prop, key_acc = jax.random.split(rng_key)
            proposed = r + step_size * jax.random.normal(
                key_prop, r.shape,
            )
            proposed = wrap_to_cell(proposed, lattice)
            lp_old = u_fn(r, params)
            lp_new = u_fn(proposed, params)
            accept = jax.random.uniform(key_acc) < jnp.exp(
                2.0 * (lp_new - lp_old),
            )
            new = jnp.where(accept, proposed, r)
            return new, accept

        self._metropolis_move = jax.jit(metropolis_move)
        self._metropolis_move_allw = jax.jit(jax.vmap(
            metropolis_move, in_axes=(0, 0, None, None),
        ))

    def initialize_walkers(self, rng_key, num_walkers):
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
        init_walkers=None,
    ):
        """Run VMC at the fixed twist ``κ`` stored on this driver.

        Same per-block logging / binning analysis as the real driver.
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
        if init_walkers is not None:
            walkers = jnp.asarray(init_walkers["R"])
            if walkers.shape[0] != num_walkers:
                if walkers.shape[0] < num_walkers:
                    reps = (num_walkers + walkers.shape[0] - 1) // walkers.shape[0]
                    walkers = jnp.tile(
                        walkers, (reps,) + (1,) * (walkers.ndim - 1),
                    )[:num_walkers]
                else:
                    walkers = walkers[:num_walkers]
            mc_stepsize = float(init_walkers.get(
                "step_size", (3 * mc_timestep) ** 0.5,
            ))
            warm_start = True
        else:
            walkers = self.initialize_walkers(init_key, num_walkers)
            mc_stepsize = (3 * mc_timestep) ** 0.5
            warm_start = False

        @jax.jit
        def _mcmc_one_step(rk, w, s, params):
            rk, key = jax.random.split(rk)
            keys = jax.random.split(key, num_walkers)
            nw, acc = metropolis_move_allw(keys, w, s, params)
            return rk, nw, acc.mean()

        tables_ = self.ewald
        _ewald_pair_chunk_fn = self._ewald_pair_fn
        chunk = 32

        @jax.jit
        def _pot_chunk(w_chunk):
            return _ewald_pair_chunk_fn(w_chunk, tables_)

        def _pot_batch(w):
            n = w.shape[0]
            if n <= chunk:
                return _pot_chunk(w)
            return jnp.concatenate([
                _pot_chunk(w[k:k + chunk]) for k in range(0, n, chunk)
            ], axis=0)

        _kin_batch = jax.vmap(energy_kinetic, in_axes=(0, None))

        # Equilibration.
        ar = 0.0
        for _ in range(num_blocks_equil):
            for _s in range(num_steps_per_block):
                rng_key, walkers, ar = _mcmc_one_step(
                    rng_key, walkers, mc_stepsize, params,
                )
                mc_stepsize = _adapt_step_size(mc_stepsize, float(ar))

        if verbose >= 1:
            print(f"# Twist κ = {self.kappa}", file=fout)
            print(f"# Equilibration acceptance (last block): "
                  f"{float(ar):.3f}", file=fout)
            print(f"# Adapted step size: {float(mc_stepsize):.4f} Bohr",
                  file=fout)
            print(
                "# block   E_loc_mean        E_loc_std"
                "       E_pot            E_kin            Δt_block",
                file=fout,
            )

        E_blocks = []
        timestamp_prev = datetime.now()

        for blk in range(1, num_blocks + 1):
            e_pot_steps, e_kin_steps = [], []
            for _step in range(num_steps_per_block):
                for _ in range(num_steps_decorr):
                    rng_key, walkers, _ = _mcmc_one_step(
                        rng_key, walkers, mc_stepsize, params,
                    )
                e_pot = _pot_batch(walkers)
                e_kin = _kin_batch(walkers, params)
                e_pot_steps.append(e_pot)
                e_kin_steps.append(e_kin)
            e_pot = jnp.stack(e_pot_steps, axis=0)
            e_kin = jnp.stack(e_kin_steps, axis=0)

            E_loc = e_pot + e_kin
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

        elapsed = (datetime.now() - timestamp_init).total_seconds()
        if verbose >= 1:
            print(f"\nVMC energy at κ={self.kappa}: "
                  f"{float(e_mean):.8f} +/- {float(e_serr):.8f} Ha "
                  f"({float(E_arr.shape[0] / e_kappa):.1f} eff blocks)",
                  file=fout)
            print(f"E/N = {e_ha:.8f} Ha/elec  = {e_ry:.8f} Ry/elec",
                  file=fout)
            print(f"Total time: {elapsed:.2f} s", file=fout)

        return {
            'E_mean': float(e_mean),
            'E_serr': float(e_serr),
            'E_blocks': E_blocks,
            'E_per_elec_ha': e_ha,
            'E_per_elec_ry': e_ry,
            'kappa': self.kappa,
        }


def get_vmc_nn_heg_twist_func(
    config: HEGConfig,
    init_key,
    *,
    kappa=(0.0, 0.0, 0.0),
    ewald_n_real: int = 3,
    ewald_n_recip: int = 6,
    ewald_eta: Optional[float] = None,
):
    """Construct a twist-aware VMC driver at a specified twist ``κ``.

    Args:
        config: :class:`HEGConfig`.
        init_key: JAX PRNG key for parameter init.
        kappa: Twist angle ``(3,)`` in fractional coordinates.
        ewald_*: as :func:`get_vmc_nn_heg_func`.

    Returns:
        :class:`_VMCDriverHEGTwist` instance.
    """
    return _VMCDriverHEGTwist(
        config, init_key,
        kappa=kappa,
        ewald_n_real=ewald_n_real,
        ewald_n_recip=ewald_n_recip,
        ewald_eta=ewald_eta,
    )


# =====================================================================
# Twist-averaged (TABC) sweep
# =====================================================================

def run_twist_averaged_heg(
    config: HEGConfig,
    init_key,
    *,
    trained_params_real=None,
    n_twists: int = 4,
    twists=None,
    ewald_n_real: int = 3,
    ewald_n_recip: int = 6,
    ewald_eta: Optional[float] = None,
    num_walkers: int = 256,
    num_steps_per_block: int = 30,
    num_blocks: int = 40,
    num_blocks_equil: int = 20,
    mc_timestep: float = 0.1,
    eval_seed: int = 2026,
    fname_log: Optional[str] = None,
    verbose: int = 1,
):
    """Twist-averaged VMC for a trained Slater–Jastrow HEG ansatz.

    For each twist ``κ_m`` drawn from a Halton sequence, builds a
    fresh complex-envelope driver at that twist, transfers the
    trained Jastrow parameters, runs production VMC, and records
    the per-twist energy.  Returns the arithmetic-mean TABC energy
    with statistical error
    ``σ_TABC = √(Σ σ_m²) / N_tw`` (uncorrelated-twist estimator).

    Args:
        config: :class:`HEGConfig` — must match the one used for the
            Γ-point Jastrow training.
        init_key: JAX PRNG key used to instantiate the per-twist
            ansatz (envelope init is deterministic given the twist;
            this key seeds the Jastrow re-init that is then
            overwritten by the transferred trained parameters).
        trained_params_real: Params pytree from
            :class:`~OmegaQMC.vmcopt_nn_heg._HEGAdamOptimizer` or
            similar.  Its ``jastrow`` subtree is grafted into every
            per-twist complex ansatz; its envelope subtree is
            discarded (the envelope is re-initialised for each
            twist).  Pass ``None`` to run twist-averaged HF (no
            trained Jastrow).
        n_twists: Number of Halton twists.  Ignored if ``twists``
            is given.
        twists: Explicit ``(N_tw, 3)`` array of twist angles in
            fractional coordinates (overrides the Halton sampling).
        num_walkers, num_steps_per_block, num_blocks, num_blocks_equil,
        mc_timestep, ewald_*: Passed to each per-twist VMC run.
        eval_seed: Seed for the MCMC run at each twist (a distinct
            key is derived per twist from this seed).
        fname_log: Per-twist log path prefix; ``None`` → stdout.
        verbose: 0 silent, 1 per-twist progress, 2 per-block detail.

    Returns:
        Dict with:
          * ``E_mean_ha`` — twist-averaged energy (Ha).
          * ``E_serr_ha`` — statistical error (Ha).
          * ``E_per_elec_ha`` / ``E_per_elec_ry`` — per electron.
          * ``twists`` — ``(N_tw, 3)`` array used.
          * ``energies_per_twist`` / ``errors_per_twist`` — raw data.
    """
    dim = int(getattr(config, 'dim', 3))
    if twists is None:
        if dim == 3:
            from .afqmc_3deg import generate_halton_twists_3d
            twists = generate_halton_twists_3d(n_twists)
        else:
            from .heg_2d import generate_halton_twists_2d
            twists = generate_halton_twists_2d(n_twists)
    else:
        twists = np.asarray(twists, dtype=np.float64)
        n_twists = int(twists.shape[0])
        if twists.shape[1] != dim:
            raise ValueError(
                f"twists has shape {twists.shape}; "
                f"expected last-dim={dim}",
            )

    if fname_log is None or fname_log == "":
        fout = sys.stdout
    else:
        fout = open(fname_log, 'w', 1)

    N = config.n_up + config.n_down
    energies = np.zeros(n_twists)
    errors = np.zeros(n_twists)

    if verbose >= 1:
        print(
            f"# TABC sweep: {n_twists} twists, "
            f"N={N}, L={config.L:.4f}",
            file=fout,
        )
        print(
            f"# walkers={num_walkers}, blocks={num_blocks}, "
            f"equil={num_blocks_equil}, steps/block="
            f"{num_steps_per_block}",
            file=fout,
        )
        print(
            f"# {'i':>3} {'kx':>10} {'ky':>10} {'kz':>10} "
            f"{'E (Ha)':>14} {'σ (Ha)':>12} {'E/N (Ry)':>14}",
            file=fout,
        )

    rng = jax.random.key(eval_seed)
    for i, kappa in enumerate(twists):
        drv = _VMCDriverHEGTwist(
            config, init_key,
            kappa=tuple(kappa),
            ewald_n_real=ewald_n_real,
            ewald_n_recip=ewald_n_recip,
            ewald_eta=ewald_eta,
        )

        if trained_params_real is not None:
            # Copy every trained parameter except the envelope; the
            # envelope must stay as the fresh Hartree-Fock identity
            # init on the twisted k-grid.  For the Slater-Jastrow
            # ansatz this transfers only the pair Jastrow (equivalent
            # to the old jastrow-only transfer); for PsiFormer it
            # also carries the GNN, attention, backflow and cusp
            # weights into the twist-specific ansatz.
            drv.params = transfer_trained_params(
                trained_params_real, drv.params,
            )

        rng, sub = jax.random.split(rng)
        result = drv(
            sub,
            num_walkers=num_walkers,
            num_steps_per_block=num_steps_per_block,
            num_blocks=num_blocks,
            num_blocks_equil=num_blocks_equil,
            mc_timestep=mc_timestep,
            verbose=(verbose - 1 if verbose >= 2 else 0),
        )
        energies[i] = result['E_mean']
        errors[i] = result['E_serr']

        if verbose >= 1:
            print(
                f"  {i:>3} {float(kappa[0]):>10.5f} "
                f"{float(kappa[1]):>10.5f} {float(kappa[2]):>10.5f} "
                f"{energies[i]:>14.6f} {errors[i]:>12.6f} "
                f"{2.0 * energies[i] / N:>14.6f}",
                file=fout,
            )

    e_mean = float(np.mean(energies))
    e_serr = float(np.sqrt(np.sum(errors ** 2)) / n_twists)
    e_ha = e_mean / N
    e_ry = e_ha * 2.0

    if verbose >= 1:
        print(
            f"\n# TABC-averaged E = {e_mean:.6f} +/- {e_serr:.6f} Ha",
            file=fout,
        )
        print(
            f"# TABC-averaged E/N = {e_ha:.6f} Ha/elec "
            f"= {e_ry:.6f} Ry/elec",
            file=fout,
        )

    if fname_log is not None and fname_log != "":
        fout.close()

    return {
        'E_mean_ha': e_mean,
        'E_serr_ha': e_serr,
        'E_per_elec_ha': e_ha,
        'E_per_elec_ry': e_ry,
        'twists': np.asarray(twists),
        'energies_per_twist': energies,
        'errors_per_twist': errors,
    }
