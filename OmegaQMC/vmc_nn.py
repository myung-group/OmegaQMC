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
from datetime import datetime
# from functools import partial

import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec

from .psi.nn.adapter import make_nn_log_psi
from .psi.nn.physics import laplacian
from .constants import MIN_DIST_THRESHOLD
from .utils import (
    do_binning_analysis,
    _make_sharding,
    parse_molecular_inspheres,
    _autotune_prod_walkers,
)
from .symm.operations import populate_fragment_symmops
from .symm.fragments import (
    build_frag_reflect_data,
    build_frag_symmops,
    build_single_frag_combos,
    make_apply_single_frag_symmop,
)

# VMC hyperparameters (matching vmc_gto)
TARGET_ACCEPTANCE_RATE = 0.4
STEP_SIZE_ADAPTATION_RATE = 0.05


def _adapt_step_size(step_size, acceptance_rate):
    """Adapt Metropolis step size toward target."""
    return step_size * (1.0 + STEP_SIZE_ADAPTATION_RATE
                        * (acceptance_rate - TARGET_ACCEPTANCE_RATE))


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
         frag_Vh, _is_planar) = build_frag_reflect_data(
            mol_info, nuc_crds,
        )
        self._apply_single_frag_symmop = (
            make_apply_single_frag_symmop(
                frag_centroids, frag_Vh, frag_inradii,
            )
        )

        log_psi, init_params, graphdef \
            = make_nn_log_psi(config, mol_info, init_key)
        self.log_psi = log_psi
        self.params = init_params

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
            def f_flat(r_flat):
                r = r_flat.reshape(nelec, 3)
                return log_psi(r, nuc_crds, params)
            r_flat = elec_crds.reshape(-1)
            lap_fn = laplacian(f_flat)
            lap_val, grad_val = lap_fn(r_flat)
            return -0.5 * (lap_val + jnp.dot(grad_val, grad_val))

        self.energy_ee = energy_ee
        self.energy_en = energy_en
        self.energy_ke = energy_ke

        # --- Batched log-ψ and total local-energy for PGCS ---
        params_here = self.params
        enr_nn_val = self.enr_nn

        @jax.jit
        def _log_psi_batch(batch):
            return jax.vmap(
                lambda r: log_psi(
                    r, nuc_crds, params_here,
                ),
            )(batch)

        @jax.jit
        def _local_energy_batch(batch):
            ee = jax.vmap(energy_ee)(batch)
            en = jax.vmap(energy_en)(batch)
            ke = jax.vmap(
                energy_ke, in_axes=(0, None),
            )(batch, params_here)
            return ee + en + ke + enr_nn_val

        self._log_psi_batch = _log_psi_batch
        self._local_energy_batch = _local_energy_batch

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
        from .nn_checkpoint import (
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
    ):
        """Execute a VMC run with fixed NN parameters.

        Runs Metropolis-Hastings sampling and accumulates local energies
        block by block. VMC results are appended to the checkpoint file
        set by :func:`get_vmc_nn_func`.

        Args:
            rng_key: JAX PRNG key (int or array).
            num_walkers: Number of MC walkers.
            num_steps_per_block: Steps per block.
            num_steps_decorr: Decorrelation steps.
            num_blocks: Total production blocks.
            num_blocks_equil: Equilibration blocks.
            mc_timestep: Initial MC timestep.
            compute_gradients: If ``True``,
                evaluate ZVZB nuclear force estimator each block
                and write to the gradient file set by :func:`get_vmc_nn_func`.
            fname_log: Optional path for the plain-text
                per-block log.  ``None`` or ``""`` writes
                to stdout; any other string opens that
                file in line-buffered mode.
            verbose: Verbosity (0 = silent).

        Returns:
            Dict with keys ``'E_mean'``,
            ``'E_serr'``, ``'E_blocks'``.
        """
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

        for _ in range(num_blocks_equil):
            state = (rng_key, walkers, mc_stepsize)
            state, ratios = jax.lax.scan(
                eq_step, state,
                jnp.arange(num_steps_per_block),
            )
            rng_key, walkers, mc_stepsize = state

        mc_timestep = mc_stepsize * mc_stepsize / 3
        if verbose >= 1:
            ratio = ratios[-1]
            print(f"Equilibration acceptance rate: {ratio:.2f}")
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
                e_ke = jax.vmap(
                    energy_ke,
                    in_axes=(0, None),
                )(nw, params)
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
                e_ke = jax.vmap(
                    energy_ke,
                    in_axes=(0, None),
                )(nw, params)
                return (
                    (rk, nw, s),
                    (ar, e_ee, e_en, e_ke),
                )

        print(
            "# block_cnt        E_loc_mean      E_loc_std"
            "       eePotential     enPotential     Kinetic"
            "          ∆t_block",
            file=fout,
        )

        E_blocks = []
        E_cs_b = []
        timestamp_prev = datetime.now()

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

        result = {
            'E_mean': float(e_mean),
            'E_serr': float(e_serr),
            'E_blocks': E_blocks,
        }

        from .nn_checkpoint import append_vmc_results
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
