"""VMC driver for neural network trial wavefunctions.

Runs Metropolis-Hastings Monte Carlo sampling of an NN
trial wavefunction built via
:func:`~OmegaQMC.psi.nn.adapter.make_nn_log_psi`.
Local energies are accumulated block by block and
analysed with binning to produce a statistical error
estimate.

Unlike :mod:`vmc_gto`, this driver has no fragment
symmetry operations, no MO relaxation, and no
gradient computation.  It takes a
:class:`~OmegaQMC.utils.Mole_custom` instance
directly.
"""

import sys
import time
from datetime import datetime
from statistics import mean, stdev
# from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import NamedSharding, PartitionSpec

from .psi.nn.adapter import make_nn_log_psi
from .constants import MIN_DIST_THRESHOLD
from .utils import (
    do_binning_analysis,
    _make_sharding,
    parse_molecular_inspheres,
    _autotune_prod_walkers,
)
from .symm.operations import populate_fragment_symmops
from .symm.fragments import (
    build_frag_transform_data,
    build_frag_symmops,
    build_single_frag_combos,
    make_apply_single_frag_symmop,
)

# VMC hyperparameters (matching vmc_gto)
TARGET_ACCEPTANCE_RATE = 0.4
STEP_SIZE_ADAPTATION_RATE = 0.05

# Maximum walkers per forward-Laplacian kinetic-energy
# evaluation.  ``jax.lax.map(..., batch_size=...)`` bounds the
# forward-Laplacian intermediate so the whole-batch KE eval does
# not overflow GPU memory / GEMM autotuning.  See
# vmcopt_nn_iradam for the detailed rationale.
_KE_WALKER_CHUNK = 256

# Automatic equilibration (``num_steps_equil='auto'``), following
# DeepQMC's ``sampling_utils.equilibrate``: record a cheap criterion
# (mean electron-electron distance) once per decorrelated sample, keep
# the last ``_EQ_AUTO_BLOCK * _EQ_AUTO_NBLOCKS`` values, and stop once
# the means of the oldest and newest ``_EQ_AUTO_BLOCK`` of them agree
# to within the smaller of their standard deviations.  In DeepQMC a
# "step" there is one ``DecorrSampler`` sample, i.e. 10-30 Metropolis
# moves, so a sample here is ``max(num_steps_decorr,
# _EQ_AUTO_STRIDE)`` moves: recording after every move instead makes
# the 50-point window span only 50 moves, and on a 2H2O PsiFormer
# that declared stationarity after 180 moves with the energy still
# ~0.5 Ha high.  ``_EQ_AUTO_MAX`` caps the number of samples, as
# DeepQMC's ``max_eq_steps`` does.
_EQ_AUTO_BLOCK = 10
_EQ_AUTO_NBLOCKS = 5
_EQ_AUTO_STRIDE = 10
_EQ_AUTO_MAX = 1000

# Fraction of the device budget the ZVZB force gradient may claim when
# ``force_chunk_size='auto'``, and the chunk used when the probe that
# sizes it is unavailable.  The rest of the budget holds the walkers,
# parameters and the other compiled kernels.
_FORCE_CHUNK_MEM_FRAC = 0.35
_FORCE_CHUNK_FALLBACK = 8

# Minimum number of production blocks when the run is sized from a
# time budget: the binning analysis needs enough block means to
# estimate the autocorrelation time at all.
_MIN_BLOCKS_DEFAULT = 32


def _adapt_step_size(step_size, acceptance_rate):
    """Adapt Metropolis step size toward target."""
    return step_size * (1.0 + STEP_SIZE_ADAPTATION_RATE
                        * (acceptance_rate - TARGET_ACCEPTANCE_RATE))


def _auto_force_chunk(grad_fn, nelec, batch_size):
    """Largest force-gradient vmap width that fits the device budget.

    AOT-compiles *grad_fn* for a single sample and divides
    ``_FORCE_CHUNK_MEM_FRAC`` of the allocator budget by its
    per-call buffers.  The one-sample probe also counts the
    sample-independent intermediates, so it over-estimates the
    per-sample cost and errs on the safe side.  Returns
    ``_FORCE_CHUNK_FALLBACK`` when the probe is unavailable.
    """
    from .gpu_memory import (
        allocator_config, compile_with_memory_stats,
        device_budget_bytes, gpu_devices, physical_gpu,
    )
    try:
        devs = gpu_devices()
        gpu = physical_gpu(devs[0]) if devs else None
        if gpu is None:
            return min(batch_size, _FORCE_CHUNK_FALLBACK)
        budget = device_budget_bytes(gpu, allocator_config())
        _, stats = compile_with_memory_stats(
            grad_fn, jnp.zeros((1, nelec, 3), dtype=jnp.float64),
        )
        if stats is None:
            return min(batch_size, _FORCE_CHUNK_FALLBACK)
        per = stats.temp_size_in_bytes + stats.output_size_in_bytes
        if per <= 0:
            return min(batch_size, _FORCE_CHUNK_FALLBACK)
        return max(1, min(
            batch_size, int(_FORCE_CHUNK_MEM_FRAC * budget // per),
        ))
    except Exception:  # noqa: BLE001 -- backend-dependent API
        return min(batch_size, _FORCE_CHUNK_FALLBACK)


def _chunk_calls(fn, chunk):
    """Evaluate *fn* over a batch in sub-batches of *chunk* samples.

    *fn* returns a tuple of arrays with a leading sample axis; the
    pieces are concatenated back, so the result is identical to one
    call on the full batch, with peak memory bounded by *chunk*.
    """
    def batched(batch_w):
        n = batch_w.shape[0]
        if n <= chunk:
            return fn(batch_w)
        outs = [fn(batch_w[i:i + chunk]) for i in range(0, n, chunk)]
        return tuple(jnp.concatenate(x, axis=0) for x in zip(*outs))
    return batched


class _VMCDriverNN:
    """Holds precompiled VMC kernels for an NN trial
    wavefunction and runs the simulation.
    """

    def __init__(
        self, mol_info, config, init_key,
        ofname_chkpt, ofname_grd,
        symmop_list=None,
    ):
        # Populate fragment metadata lazily for Mole_custom
        # instances that did not come through the GTO
        # factory (e.g. from Mole_custom.from_arrays).
        if not hasattr(mol_info, 'map_nuc_frag'):
            if not hasattr(mol_info, 'ignore_hydrogen_mass'):
                mol_info.ignore_hydrogen_mass = False
            parse_molecular_inspheres(mol_info)
        if not hasattr(mol_info, 'map_frag_symmops'):
            populate_fragment_symmops(mol_info)

        nuc_crds = jnp.asarray(
            mol_info.coords, dtype=jnp.float64,
        )
        charges = jnp.asarray(
            mol_info.charges, dtype=jnp.float64,
        )
        nelec = mol_info.n_up + mol_info.n_down
        n_nuc = len(charges)

        self.mol_info = mol_info
        self.nuc_crds = nuc_crds
        self.charges = charges
        self.nelec = nelec
        self.n_nuc = n_nuc
        self.ofname_chkpt = ofname_chkpt
        self.ofname_grd = ofname_grd

        # --- PGCS fragment plumbing ---
        if mol_info.map_frag_symmops:
            frag_ids = sorted(mol_info.map_frag_ctr.keys())
        else:
            frag_ids = [0]
        frag_symmops = build_frag_symmops(
            mol_info, symmop_list, frag_ids,
        )
        self.frag_ids = frag_ids
        self.frag_symmops = frag_symmops
        self.single_frag_combos = (
            build_single_frag_combos(
                frag_ids, frag_symmops,
            )
        )
        (frag_centroids, frag_inradii,
         frag_Vh, _is_planar) = build_frag_transform_data(
            mol_info, nuc_crds, frag_symmops=frag_symmops,
        )
        self._apply_single_frag_symmop = (
            make_apply_single_frag_symmop(
                frag_centroids, frag_Vh, frag_inradii,
            )
        )

        log_psi, init_params, graphdef, lap_grad \
            = make_nn_log_psi(config, mol_info, init_key)
        self.log_psi = log_psi
        self.params = init_params
        self.lap_grad = lap_grad

        # Precompute nuclear repulsion energy and gradient
        def _nuc_repulsion(R):
            enr = jnp.float64(0.0)
            for a in range(n_nuc):
                for b in range(a + 1, n_nuc):
                    rab = jnp.linalg.norm(R[a] - R[b])
                    enr = enr + (charges[a] * charges[b] / rab)
            return enr

        self.enr_nn = jnp.asarray(
            _nuc_repulsion(nuc_crds),
            dtype=jnp.float64,
        )
        self.grd_nn = jax.grad(
            _nuc_repulsion,
        )(nuc_crds)

        i_e, j_e = jnp.triu_indices(nelec, k=1)

        # --- Energy components ---
        @jax.jit
        def energy_ee(elec_crds):
            diffs = elec_crds[i_e] - elec_crds[j_e]
            dists = jnp.linalg.norm(diffs, axis=-1)
            return jnp.sum(1.0 / dists)

        @jax.jit
        def energy_en(elec_crds):
            diffs = elec_crds[:, None, :] - nuc_crds[None, :, :]
            dists = jnp.linalg.norm(diffs, axis=-1)
            return -jnp.sum(
                charges[None, :] / dists,
            )

        @jax.jit
        def energy_ke(elec_crds, params):
            lap_val, grad_val = lap_grad(
                elec_crds, nuc_crds, params,
            )
            return -0.5 * (
                lap_val + jnp.dot(grad_val, grad_val)
            )

        # Walker-chunked KE: lax.map over chunks (<=
        # _KE_WALKER_CHUNK) bounds the forward-Laplacian
        # intermediate.  Only the KE is chunked; the ee/en terms
        # are cheap 1/r sums.
        def batched_energy_ke(walkers, params):
            n = walkers.shape[0]
            chunk = min(_KE_WALKER_CHUNK, n)
            return jax.lax.map(
                lambda r: energy_ke(r, params),
                walkers,
                batch_size=chunk,
            )

        self.energy_ee = energy_ee
        self.energy_en = energy_en
        self.energy_ke = energy_ke
        self.batched_energy_ke = batched_energy_ke

        # --- Batched log-ψ and total local-energy for PGCS ---
        # The kernels take the parameters explicitly, and the
        # one-argument ``_log_psi_batch`` / ``_local_energy_batch``
        # read ``self.params`` at *call* time.  Binding the parameters
        # at construction instead (as this driver used to) meant that
        # after ``load_checkpoint`` the PGCS weights
        # psi^2(R') / psi^2(R) and the secondary local energies in
        # ``save_nn_gradients`` came from the initial random network,
        # while the force terms used the loaded one.  Passing params
        # as an argument costs no recompilation when they change.
        enr_nn_val = self.enr_nn

        @jax.jit
        def _log_psi_batch_p(batch, params):
            return jax.vmap(
                lambda r: log_psi(r, nuc_crds, params),
            )(batch)

        @jax.jit
        def _local_energy_batch_p(batch, params):
            ee = jax.vmap(energy_ee)(batch)
            en = jax.vmap(energy_en)(batch)
            ke = batched_energy_ke(batch, params)
            return ee + en + ke + enr_nn_val

        self._log_psi_batch_p = _log_psi_batch_p
        self._local_energy_batch_p = _local_energy_batch_p
        self._log_psi_batch = (
            lambda batch: _log_psi_batch_p(batch, self.params)
        )
        self._local_energy_batch = (
            lambda batch: _local_energy_batch_p(batch, self.params)
        )
        self._i_e, self._j_e = i_e, j_e

        # --- Metropolis move ---
        @jax.jit
        def metropolis_move(
            rng_key, elec_crds, step_size, params,
        ):
            key_prop, key_acc = jax.random.split(
                rng_key,
            )
            proposed = elec_crds + step_size * (
                jax.random.normal(
                    key_prop, elec_crds.shape,
                )
            )
            diffs_ee = proposed[i_e] - proposed[j_e]
            dists_ee = jnp.linalg.norm(
                diffs_ee, axis=-1,
            )
            diffs_en = (
                proposed[:, None, :]
                - nuc_crds[None, :, :]
            )
            dists_en = jnp.linalg.norm(
                diffs_en, axis=-1,
            )
            valid = (
                (dists_en.min() > MIN_DIST_THRESHOLD)
                & (dists_ee.min() > MIN_DIST_THRESHOLD)
            )
            lp_old = log_psi(
                elec_crds, nuc_crds, params,
            )
            lp_new = log_psi(
                proposed, nuc_crds, params,
            )
            accept = (
                jax.random.uniform(key_acc)
                < jnp.exp(2 * (lp_new - lp_old))
            ) & valid
            new_crds = jnp.where(
                accept, proposed, elec_crds,
            )
            return new_crds, accept

        self._metropolis_move = metropolis_move
        self._metropolis_move_allw = jax.vmap(
            metropolis_move,
            in_axes=(0, 0, None, None),
        )

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
        idx_cnt = idx_cnt[:total]
        idx_cnt = jnp.array(idx_cnt)
        centers = self.nuc_crds[idx_cnt]
        return (
            centers[None, :, :]
            + 0.05 * jax.random.normal(
                rng_key,
                (num_walkers, self.nelec, 3),
            )
        )

    def load_checkpoint(self, filepath):
        """Load optimised parameters from a checkpoint.

        Replaces ``self.params`` with the parameter
        values stored in the ``.chk.h5`` file.

        Args:
            filepath: Path to the HDF5 checkpoint
                created by
                :class:`~OmegaQMC.vmcopt_nn_iradam._VMCOptDriverNN_IRAdam`.

        Returns:
            Dict of checkpoint metadata (epoch,
            config_name, energy, etc.).
        """
        from .psi.nn.checkpoint import (
            load_nn_checkpoint,
        )
        params, meta = load_nn_checkpoint(
            filepath, self.params,
        )
        self.params = params
        return meta

    def __call__(
        self,
        rng_key,
        num_walkers=1000,
        num_steps_per_block=100,
        num_steps_decorr=1,
        num_blocks=100,
        num_blocks_equil=10,
        mc_timestep=0.1,
        compute_gradients=False,
        fname_log=None,
        verbose=1,
        dump_walkers_path=None,
        num_steps_equil=None,
        max_hours=None,
        target_stderr=None,
        min_blocks=_MIN_BLOCKS_DEFAULT,
        force_chunk_size='auto',
    ):
        """Execute a VMC run with fixed NN parameters.

        Runs Metropolis-Hastings sampling and accumulates local energies
        block by block. VMC results are appended to the checkpoint file
        set by :func:`get_vmc_nn_func`.

        The run length can be fixed (``num_blocks`` blocks of
        ``num_steps_per_block`` steps) or budgeted.  With
        ``max_hours`` set, ``num_steps_per_block='auto'`` sizes the
        blocks from a measured sampling rate so that at least
        ``min_blocks`` of them fit the budget; ``num_blocks`` is then
        only an upper bound.  ``max_hours`` and ``target_stderr``
        also stop the run early — whichever is met first — once
        ``min_blocks`` blocks exist.  Early stopping leaves the tail
        of a ``dump_walkers_path`` file unwritten.

        Args:
            rng_key: JAX PRNG key (int or array).
            num_walkers: Number of MC walkers.
            num_steps_per_block: Steps per block, or ``'auto'`` (needs
                ``max_hours``).
            num_steps_decorr: Decorrelation steps.
            num_blocks: Total production blocks (an upper bound when
                the run is budgeted).
            num_blocks_equil: Equilibration blocks of
                ``num_steps_per_block`` steps; used only when
                ``num_steps_equil`` is ``None``.
            mc_timestep: Initial MC timestep.
            compute_gradients: If ``True``,
                evaluate ZVZB nuclear force estimator each block
                and write to the gradient file set by :func:`get_vmc_nn_func`.
            fname_log: Optional path for the plain-text
                per-block log.  ``None`` or ``""`` writes
                to stdout; any other string opens that
                file in line-buffered mode.
            verbose: Verbosity (0 = silent).
            dump_walkers_path: Optional HDF5 path. When set, each
                production block streams the post-block walker
                configurations and ``log|Psi|`` values to that file via
                :class:`OmegaQMC.cs.walkers.WalkerDumper`. Used by the
                compressed-sensing CI extraction pipeline.
            num_steps_equil: Equilibration length in Metropolis steps,
                independent of the block length.  ``None`` keeps the
                ``num_blocks_equil * num_steps_per_block`` behaviour;
                an int runs that many steps; ``'auto'`` stops once the
                walker distribution is stationary (DeepQMC's
                criterion, checked every ``max(num_steps_decorr,
                _EQ_AUTO_STRIDE)`` steps and capped at
                ``_EQ_AUTO_MAX`` such checks).
            max_hours: Wall-clock budget for production, in hours.
            target_stderr: Stop once the binned standard error of the
                energy reaches this value (Ha).
            min_blocks: Blocks required before either early stop may
                fire, and the block count a budgeted run is sized
                for.
            force_chunk_size: Samples per vmap of the ZVZB force
                gradient when ``compute_gradients`` is set.
                ``'auto'`` sizes it from an AOT memory probe; an int
                fixes it; ``None`` evaluates each force batch in one
                call, as before.  Results do not depend on it.

        Returns:
            Dict with keys ``'E_mean'``, ``'E_serr'``, ``'E_blocks'``,
            ``'E_neff'``, ``'sigma'`` (spread of the individual local
            energies), ``'n_samples'``, ``'samples_per_sec'``,
            ``'num_steps_per_block'`` and ``'stop_reason'``.
        """
        if num_steps_per_block == 'auto' and max_hours is None:
            raise ValueError(
                "num_steps_per_block='auto' sizes blocks from a time "
                "budget, so max_hours must be given"
            )
        if isinstance(rng_key, int):
            rng_key = jax.random.key(rng_key)

        if fname_log is None \
                or (isinstance(fname_log, str)
                    and fname_log == ""):
            fout = sys.stdout
        else:
            fout = open(fname_log, 'w', 1)

        nelec = self.nelec
        nuc_crds = self.nuc_crds
        charges = self.charges
        params = self.params
        enr_nn = self.enr_nn
        energy_ee = self.energy_ee
        energy_en = self.energy_en
        energy_ke = self.energy_ke
        batched_energy_ke = self.batched_energy_ke
        metropolis_move_allw = self._metropolis_move_allw

        # --- Informational GPU capacity estimate ---
        # Does NOT modify num_walkers.
        try:
            from .vmcopt_gto_linear import _get_free_gpu_mb
            free_mb = _get_free_gpu_mb()
            n_rec, bpw = _autotune_prod_walkers(
                self._local_energy_batch, nelec, free_mb)
            free_txt = (f"{free_mb:.0f} MiB free"
                        if free_mb is not None
                        else "free GPU mem unknown")
            print(f"ℹ️\tEst. GPU capacity: {n_rec} walkers "
                  f"(user requested {num_walkers}; "
                  f"{bpw / 1e6:.2f} MB/walker, {free_txt})")
        except Exception as e:
            print(f"ℹ️\tGPU capacity estimate unavailable: {e}")

        timestamp_init = datetime.now()

        # --- Force setup ---
        if compute_gradients:
            import h5py
            import pathlib
            from .observables.force import (
                vmc_nn_gradients_zvzb,
                save_nn_gradients,
            )

            ofname_grd = self.ofname_grd
            grd_nn = self.grd_nn
            mol_info = self.mol_info

            nn_gradient_batch = vmc_nn_gradients_zvzb(
                self.log_psi, nuc_crds, charges,
                nelec, params,
                lap_grad=self.lap_grad,
            )

            p = pathlib.Path(ofname_grd)
            if p.exists():
                p.unlink()
            with h5py.File(ofname_grd, 'w') as f:
                f.create_dataset(
                    'grd_nn', data=grd_nn,
                )
                g = f.create_group("system")
                asym = [
                    mol_info.atom_symbol(i)
                    for i in range(self.n_nuc)
                ]
                g.create_dataset(
                    "atom_symbols",
                    data=" ".join(asym),
                )
                g.create_dataset(
                    "atom_coords", data=nuc_crds,
                )
                g.create_dataset(
                    "charges", data=charges,
                )
                g.create_dataset(
                    "units",
                    data=mol_info.unit.upper(),
                )
                g.create_dataset(
                    "atom_fragment_map",
                    data=mol_info.map_nuc_frag,
                )

            n_nuc = self.n_nuc
            base_batch_size = 500
            mem_factor = max(
                1, nelec * n_nuc // 1000,
            )
            batch_size = min(
                50, base_batch_size // mem_factor,
            )

            # Sub-chunk the force gradient inside each batch.  The
            # batch size above ignores the gradient's actual footprint,
            # which on the VGL path (jacfwd over nuclear coordinates
            # layered on the forward-Laplacian's electron tangents) is
            # one fused computation that XLA cannot rematerialise below
            # device memory for a 2H2O PsiFormer at 50 samples.  Only
            # the vmap width changes; the HDF5 layout and the
            # accumulation in save_nn_gradients do not.
            if force_chunk_size is None:
                force_chunk = batch_size
            elif force_chunk_size == 'auto':
                force_chunk = _auto_force_chunk(
                    nn_gradient_batch, nelec, batch_size,
                )
            else:
                force_chunk = max(
                    1, min(int(force_chunk_size), batch_size),
                )
            if force_chunk < batch_size:
                nn_gradient_batch = _chunk_calls(
                    nn_gradient_batch, force_chunk,
                )
            if verbose >= 1:
                print(f"Force gradient: batches of {batch_size},"
                      f" vmap chunks of {force_chunk}")

        rng_key, init_key = jax.random.split(rng_key)
        walkers = self.initialize_walkers(
            init_key, num_walkers,
        )
        walkers_sharding, walker_keys_sharding = \
            _make_sharding(num_walkers)
        if walkers_sharding is not None:
            walkers = jax.device_put(
                walkers, walkers_sharding)
        mc_stepsize = (3 * mc_timestep) ** 0.5

        # --- Equilibration ---
        @jax.jit
        def eq_step(state, _):
            rk, w, s = state
            rk, key = jax.random.split(rk)
            keys = jax.random.split(
                key, num_walkers,
            )
            if walker_keys_sharding is not None:
                keys = (
                    jax.lax
                    .with_sharding_constraint(
                        keys,
                        walker_keys_sharding,
                    )
                )
            nw, acc = metropolis_move_allw(
                keys, w, s, params,
            )
            ar = acc.mean()
            ns = _adapt_step_size(s, ar)
            return (rk, nw, ns), ar

        # Per-step stationarity criterion for automatic equilibration:
        # the walker-averaged mean electron-electron distance (DeepQMC's
        # ``pairwise_self_distance``), or the electron-nucleus distance
        # for a one-electron system.  It needs no local energy, so it
        # costs next to nothing next to a Metropolis step.
        i_e, j_e = self._i_e, self._j_e

        def _eq_criterion(w):
            if nelec > 1:
                d = jnp.linalg.norm(w[:, i_e] - w[:, j_e], axis=-1)
            else:
                d = jnp.linalg.norm(
                    w[:, :, None, :] - nuc_crds[None, None, :, :],
                    axis=-1,
                )
            return d.mean()

        # One criterion value per decorrelated sample of eq_stride
        # Metropolis moves (see _EQ_AUTO_STRIDE).
        eq_stride = max(int(num_steps_decorr), _EQ_AUTO_STRIDE)

        @jax.jit
        def eq_step_c(state, x):
            state, ar = jax.lax.scan(
                eq_step, state, jnp.arange(eq_stride),
            )
            return state, (ar[-1], _eq_criterion(state[1]))

        if num_steps_equil is None and num_steps_per_block == 'auto':
            num_steps_equil = 'auto'

        ratios = None
        eq_note = ""
        if num_steps_equil is None:
            for _ in range(num_blocks_equil):
                state = (rng_key, walkers, mc_stepsize)
                state, ratios = jax.lax.scan(
                    eq_step, state,
                    jnp.arange(num_steps_per_block),
                )
                rng_key, walkers, mc_stepsize = state
            n_eq_done = num_blocks_equil * num_steps_per_block
        else:
            # Chunks of _EQ_AUTO_BLOCK samples, so the scan compiles
            # once whatever the requested length (an int length, in
            # Metropolis moves, is rounded up to whole chunks).
            auto_eq = num_steps_equil == 'auto'
            n_eq_target = (_EQ_AUTO_MAX * eq_stride if auto_eq
                           else int(num_steps_equil))
            buf_size = _EQ_AUTO_BLOCK * _EQ_AUTO_NBLOCKS
            buf = []
            n_eq_done = 0
            converged = False
            while n_eq_done < n_eq_target:
                state = (rng_key, walkers, mc_stepsize)
                state, (ratios, crit) = jax.lax.scan(
                    eq_step_c, state, jnp.arange(_EQ_AUTO_BLOCK),
                )
                rng_key, walkers, mc_stepsize = state
                n_eq_done += _EQ_AUTO_BLOCK * eq_stride
                if not auto_eq:
                    continue
                buf = (buf + np.asarray(crit).tolist())[-buf_size:]
                if len(buf) == buf_size:
                    b1 = buf[:_EQ_AUTO_BLOCK]
                    b2 = buf[-_EQ_AUTO_BLOCK:]
                    if abs(mean(b1) - mean(b2)) < min(stdev(b1),
                                                      stdev(b2)):
                        converged = True
                        break
            if auto_eq:
                eq_note = (" (stationary)" if converged
                           else " (hit the cap, not stationary)")

        mc_timestep = mc_stepsize * mc_stepsize / 3
        if verbose >= 1:
            print(f"Equilibration: {n_eq_done} Metropolis"
                  f" steps{eq_note}")
            if ratios is not None:
                print(f"Equilibration acceptance rate:"
                      f" {ratios[-1]:.2f}")
            print(f"Adjusted step size:"
                  f" {mc_stepsize:.4f} bohr ~ {mc_timestep:.4f} Ha^-1")

        # --- Production ---
        if compute_gradients:
            @jax.jit
            def prod_step(state, _):
                rk, w, s = state
                for _ in range(num_steps_decorr):
                    rk, key = jax.random.split(rk)
                    keys = jax.random.split(
                        key, num_walkers,
                    )
                    if walker_keys_sharding is not None:
                        keys = (
                            jax.lax
                            .with_sharding_constraint(
                                keys,
                                walker_keys_sharding,
                            )
                        )
                    nw, acc = metropolis_move_allw(
                        keys, w, s, params,
                    )
                    w = nw
                ar = acc.mean()
                e_ee = jax.vmap(energy_ee)(nw)
                e_en = jax.vmap(energy_en)(nw)
                e_ke = batched_energy_ke(nw, params)
                return (
                    (rk, nw, s),
                    (ar, e_ee, e_en, e_ke, nw),
                )
        else:
            @jax.jit
            def prod_step(state, _):
                rk, w, s = state
                for _ in range(num_steps_decorr):
                    rk, key = jax.random.split(rk)
                    keys = jax.random.split(
                        key, num_walkers,
                    )
                    if walker_keys_sharding is not None:
                        keys = (
                            jax.lax
                            .with_sharding_constraint(
                                keys,
                                walker_keys_sharding,
                            )
                        )
                    nw, acc = metropolis_move_allw(
                        keys, w, s, params,
                    )
                    w = nw
                ar = acc.mean()
                e_ee = jax.vmap(energy_ee)(nw)
                e_en = jax.vmap(energy_en)(nw)
                e_ke = batched_energy_ke(nw, params)
                return (
                    (rk, nw, s),
                    (ar, e_ee, e_en, e_ke),
                )

        # --- Sampling rate and time-budgeted sizing ---
        # Measured on the real walkers with the kernels production
        # uses: one production step (Metropolis moves + local energy),
        # and — with gradients — one force batch plus the per-combo
        # PGCS work that save_nn_gradients does for every sample (a
        # force gradient, a local energy and a log|psi| on the
        # transformed batch, and a log|psi| on the original).  The
        # force and PGCS kernels compiled here are reused in the run.
        rate = None
        if max_hours is not None:
            def _timed(fn, *a):
                jax.block_until_ready(fn(*a))          # compile/warm
                t0 = time.perf_counter()
                out = fn(*a)
                jax.block_until_ready(out)
                return time.perf_counter() - t0, out

            t_step, (_, probe_out) = _timed(
                prod_step, (rng_key, walkers, mc_stepsize), None,
            )
            per_sample = t_step / num_walkers
            if compute_gradients:
                probe = walkers[:batch_size]
                nb = probe.shape[0]
                n_combo = len(self.single_frag_combos)
                t_g, _ = _timed(nn_gradient_batch, probe)
                per_sample += (1 + n_combo) * t_g / nb
                if n_combo:
                    t_lp, _ = _timed(
                        self._log_psi_batch_p, probe, params,
                    )
                    t_el, _ = _timed(
                        self._local_energy_batch_p, probe, params,
                    )
                    per_sample += (
                        (1 + n_combo) * t_lp + n_combo * t_el
                    ) / nb
            rate = 1.0 / per_sample
            affordable = max_hours * 3600.0 * rate
            if num_steps_per_block == 'auto':
                num_steps_per_block = max(1, int(
                    affordable // (min_blocks * num_walkers)
                ))
                num_blocks = min(num_blocks, max(min_blocks, int(
                    affordable // (num_steps_per_block * num_walkers)
                )))
            e_probe = (probe_out[1] + probe_out[2] + probe_out[3]
                       + enr_nn)
            sigma_probe = float(jnp.std(e_probe))
            n_plan = num_blocks * num_steps_per_block * num_walkers
            if verbose >= 1:
                print(f"Sampling rate: {rate:.2f} samples/s"
                      f" (gradients {'on' if compute_gradients else 'off'})")
                print(f"Budget {max_hours:g} h: {num_blocks} blocks x"
                      f" {num_steps_per_block} steps x {num_walkers}"
                      f" walkers = {n_plan} samples")
                # kappa = 2 as a guess for the residual autocorrelation;
                # the binning analysis reports what is actually achieved.
                print(f"  projected stderr ~"
                      f"{sigma_probe * (2.0 / n_plan) ** 0.5:.2e} Ha"
                      f" at sigma(E_L) ~ {sigma_probe:.3f} Ha")

        print(
            "# block_cnt        E_loc_mean      E_loc_std"
            "       eePotential     enPotential     Kinetic"
            "          ∆t_block",
            file=fout,
        )

        E_blocks = []
        E_cs_b = []
        timestamp_prev = datetime.now()
        e_sum = 0.0
        e_sumsq = 0.0
        n_tot = 0
        stop_reason = 'num_blocks'
        t_prod0 = time.perf_counter()

        walker_dumper = None
        if dump_walkers_path is not None:
            from .cs.walkers import WalkerDumper
            walker_dumper = WalkerDumper(
                dump_walkers_path,
                num_blocks=num_blocks,
                num_walkers=num_walkers,
                nelec=nelec,
                mc_timestep=mc_timestep,
                num_steps_decorr=num_steps_decorr,
            )

        for blk in range(1, num_blocks + 1):
            state = (rng_key, walkers, mc_stepsize)
            state, result = jax.lax.scan(
                prod_step, state,
                jnp.arange(num_steps_per_block),
            )
            rng_key, walkers, _ = state

            if compute_gradients:
                (ratios, e_ee, e_en, e_ke, sampled_w) = result
            else:
                ratios, e_ee, e_en, e_ke = result

            E_loc = e_ee + e_en + e_ke + enr_nn
            # E_loc: (num_steps_per_block,num_walkers)

            E_step = E_loc.mean(axis=1)
            E_mean = E_step.mean()
            E_std = E_step.std()
            E_blocks.append(float(E_mean))
            e_sum += float(E_loc.sum())
            e_sumsq += float((E_loc * E_loc).sum())
            n_tot += int(E_loc.size)

            ee_m = e_ee.mean()
            en_m = e_en.mean()
            ke_m = e_ke.mean()
            now = datetime.now()
            dt = (now - timestamp_prev).total_seconds()
            print(
                f"{blk:>8d}{E_mean:>24.8e}{E_std:>16.8e}"
                f"{ee_m:>16.8e}{en_m:>16.8e}{ke_m:>16.8e}"
                f"{dt:>16.6f}",
                file=fout,
            )
            timestamp_prev = now

            if walker_dumper is not None:
                walker_dumper.write_block(
                    jax.device_get(walkers),
                    jax.device_get(self._log_psi_batch(walkers)),
                )

            if compute_gradients:
                # sampled_w: (steps, walkers, nel, 3)
                if walkers_sharding is not None:
                    sampled_w = jax.device_put(
                        sampled_w,
                        NamedSharding(
                            walkers_sharding.mesh,
                            PartitionSpec(
                                None, None,
                                None, None,
                            ),
                        ),
                    )
                n_samples = num_steps_per_block * num_walkers
                num_batches = (n_samples + batch_size - 1) // batch_size
                combo_E = save_nn_gradients(
                    blk,
                    sampled_w.reshape(
                        -1, nelec, 3,
                    ),
                    E_loc,
                    batch_size,
                    num_batches,
                    self.single_frag_combos,
                    ofname_grd,
                    nn_gradient_batch,
                    log_psi_batch=self._log_psi_batch,
                    local_energy_batch=(
                        self._local_energy_batch
                    ),
                    apply_single_frag_symmop=(
                        self._apply_single_frag_symmop
                    ),
                )
                if combo_E:
                    all_E = (
                        [float(E_mean)]
                        + list(combo_E.values())
                    )
                    E_cs_b.append(
                        sum(all_E) / len(all_E),
                    )

            # --- Early stops, checked after the block's gradient pass
            # so the elapsed time includes it.
            if (target_stderr is not None and blk >= min_blocks
                    and blk < num_blocks):
                _, serr_now, _, _ = do_binning_analysis(
                    jnp.array(E_blocks),
                )
                if 0.0 < float(serr_now) <= target_stderr:
                    stop_reason = 'target_stderr'
                    break
            if max_hours is not None and blk < num_blocks:
                el = time.perf_counter() - t_prod0
                if el + el / blk > max_hours * 3600.0:
                    stop_reason = 'max_hours'
                    break

        t_prod = time.perf_counter() - t_prod0
        if walker_dumper is not None:
            walker_dumper.close()

        if not (fname_log is None
                or (isinstance(fname_log, str)
                    and fname_log == "")):
            fout.close()

        # --- Binning analysis ---
        E_arr = jnp.array(E_blocks)
        e_mean, e_serr, _, e_kappa = do_binning_analysis(E_arr)
        e_neff = E_arr.shape[0] / e_kappa

        timestamp_fin = datetime.now()
        elapsed = (timestamp_fin - timestamp_init).total_seconds()

        if verbose >= 1:
            print(f"\nVMC energy: {e_mean:.8f} +/- {e_serr:.8f} Ha"
                  f" (N_eff = {e_neff:.1f})")
            if compute_gradients and E_cs_b:
                E_cs_blocks = jnp.array(E_cs_b)
                ecs_mean, ecs_serr, _, ecs_kappa = (
                    do_binning_analysis(E_cs_blocks)
                )
                ecs_neff = E_cs_blocks.shape[0] / ecs_kappa
                print(
                    f"CS-averaged VMC energy: "
                    f"{ecs_mean:.8f} "
                    f"+/- {ecs_serr:.8f} Ha "
                    f"(N_eff = {ecs_neff:.1f})"
                )
            print(f"Total time: {elapsed:.2f} seconds")

        n_mean = e_sum / max(n_tot, 1)
        sigma = float(np.sqrt(max(e_sumsq / max(n_tot, 1)
                                  - n_mean * n_mean, 0.0)))
        rate_meas = n_tot / t_prod if t_prod > 0 else float('nan')
        if verbose >= 1:
            print(f"Production: {len(E_blocks)} blocks, {n_tot} samples"
                  f" in {t_prod / 3600:.2f} h ({rate_meas:.2f}"
                  f" samples/s), sigma(E_L) = {sigma:.4f} Ha;"
                  f" stopped on {stop_reason}")
            if len(E_blocks) < min_blocks:
                print(f"  warning: fewer than min_blocks={min_blocks}"
                      f" blocks, so the error bar is unreliable")
            if target_stderr is not None:
                if 0.0 < float(e_serr) <= target_stderr:
                    print(f"  target {target_stderr:g} Ha reached")
                else:
                    more = (float(e_serr) / target_stderr) ** 2
                    extra = n_tot * (more - 1.0)
                    print(f"  target {target_stderr:g} Ha NOT reached:"
                          f" ~{more:.1f}x the samples, i.e."
                          f" ~{extra / rate_meas / 3600:.1f} h more at"
                          f" this sigma; production cost scales as"
                          f" sigma^2, so optimising further first is"
                          f" usually cheaper")

        result = {
            'E_mean': float(e_mean),
            'E_serr': float(e_serr),
            'E_blocks': E_blocks,
            'E_neff': float(e_neff),
            'sigma': sigma,
            'n_samples': n_tot,
            'samples_per_sec': float(rate_meas),
            'num_steps_per_block': num_steps_per_block,
            'stop_reason': stop_reason,
        }

        from .psi.nn.checkpoint import append_vmc_results
        append_vmc_results(self.ofname_chkpt, result)
        if verbose >= 1:
            print(f"VMC results written to {self.ofname_chkpt}")

        return result


def get_vmc_nn_func(
    mol_info, config, init_key, prefix='vmc',
    symmop_list=None,
):
    """Construct a VMC driver for NN wavefunctions.

    Builds the NN trial wavefunction from *config*,
    compiles the Metropolis kernel and local-energy
    functions, and returns a callable driver.

    Args:
        mol_info: :class:`~OmegaQMC.utils.Mole_custom`
            instance describing the molecule.
        config: :class:`~OmegaQMC.psi.nn.config.NNAnsatzConfig`
            or a string (built-in name or YAML path).
        init_key: JAX PRNG key for parameter
            initialisation.
        prefix: Stem used for output file names
            (``<prefix>.chk.h5``,
            ``<prefix>.grd.h5``).
            Default is ``"vmc"``.
        symmop_list: Fragment symmetry operations to
            use for Point Group Correlated Sampling
            (PGCS).  ``None`` (default) disables PGCS;
            ``"auto"`` enables all detected operations;
            a list of strings or per-fragment dict
            restricts the set.  Matches the GTO
            driver's *symmop_list* semantics.

    Returns:
        :class:`_VMCDriverNN` instance.  Call it with
        ``driver(rng_key, ...)`` to run the VMC
        simulation.
    """
    for s in [".chk.h5", ".grd.h5"]:
        if prefix.endswith(s):
            prefix = prefix[:-len(s)]
    ofname_chkpt = prefix + ".chk.h5"
    ofname_grd = prefix + ".grd.h5"

    return _VMCDriverNN(
        mol_info, config, init_key,
        ofname_chkpt, ofname_grd,
        symmop_list=symmop_list,
    )
