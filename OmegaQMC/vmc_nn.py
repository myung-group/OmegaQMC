"""VMC driver for neural network trial wavefunctions.

Runs Metropolis-Hastings Monte Carlo sampling of an NN trial
wavefunction built via :func:`~OmegaQMC.psi.nn.adapter.make_nn_log_psi`.
Local energies are accumulated block by block and analysed
with binning to produce a statistical error estimate.

Unlike :mod:`vmc_gto`, this driver has no PySCF dependency,
no fragment symmetry operations, no MO relaxation, and no
gradient computation.  It takes a :class:`MoleculeInfo`
directly.
"""

import sys
from datetime import datetime
from functools import partial

import jax
import jax.numpy as jnp

from .psi.nn.adapter import make_nn_log_psi
from .psi.nn.physics import laplacian
from .psi.nn.wf import MoleculeInfo
from .constants import MIN_DIST_THRESHOLD
from .utils import do_binning_analysis

# VMC hyperparameters (matching vmc_gto)
TARGET_ACCEPTANCE_RATE = 0.4
STEP_SIZE_ADAPTATION_RATE = 0.05


def _adapt_step_size(step_size, acceptance_rate):
    """Adapt Metropolis step size toward target."""
    return step_size * (
        1.0
        + STEP_SIZE_ADAPTATION_RATE
        * (acceptance_rate - TARGET_ACCEPTANCE_RATE)
    )


class _VMCDriverNN:
    """Holds precompiled VMC kernels for an NN trial
    wavefunction and runs the simulation.
    """

    def __init__(self, mol_info, config, init_key):
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

        log_psi, init_params, graphdef = (
            make_nn_log_psi(config, mol_info, init_key)
        )
        self.log_psi = log_psi
        self.params = init_params

        # Precompute nuclear repulsion energy and gradient
        def _nuc_repulsion(R):
            enr = jnp.float64(0.0)
            for a in range(n_nuc):
                for b in range(a + 1, n_nuc):
                    rab = jnp.linalg.norm(
                        R[a] - R[b],
                    )
                    enr = enr + (
                        charges[a] * charges[b] / rab
                    )
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
            diffs = (
                elec_crds[:, None, :]
                - nuc_crds[None, :, :]
            )
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
            return -0.5 * (
                lap_val
                + jnp.dot(grad_val, grad_val)
            )

        self.energy_ee = energy_ee
        self.energy_en = energy_en
        self.energy_ke = energy_ke

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
                :class:`~OmegaQMC.vmcopt_nn\
._VMCOptDriverNN`.

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
        verbose=1,
        prefix='nnopt',
    ):
        """Execute a VMC run with fixed NN parameters.

        Runs Metropolis-Hastings sampling and
        accumulates local energies block by block.
        VMC results are appended to the checkpoint
        file ``{prefix}.chk.h5``.

        Args:
            rng_key: JAX PRNG key (int or array).
            num_walkers: Number of MC walkers.
            num_steps_per_block: Steps per block.
            num_steps_decorr: Decorrelation steps.
            num_blocks: Total production blocks.
            num_blocks_equil: Equilibration blocks.
            mc_timestep: Initial MC timestep.
            compute_gradients: If ``True``, evaluate
                ZVZB nuclear force estimator each
                block and write to
                ``{prefix}.grd.h5``.
            verbose: Verbosity (0 = silent).
            prefix: Filename prefix for the HDF5
                checkpoint.  VMC results are appended
                to ``{prefix}.chk.h5``.

        Returns:
            Dict with keys ``'E_mean'``,
            ``'E_serr'``, ``'E_blocks'``.
        """
        if isinstance(rng_key, int):
            rng_key = jax.random.key(rng_key)

        nelec = self.nelec
        nuc_crds = self.nuc_crds
        charges = self.charges
        params = self.params
        enr_nn = self.enr_nn
        energy_ee = self.energy_ee
        energy_en = self.energy_en
        energy_ke = self.energy_ke
        metropolis_move_allw = (
            self._metropolis_move_allw
        )

        timestamp_init = datetime.now()

        # --- Force setup ---
        if compute_gradients:
            import h5py
            import pathlib
            from .observables.force import (
                vmc_nn_forces_zvzb,
                save_nn_forces,
            )

            ofname_grd = f"{prefix}.grd.h5"
            grd_nn = self.grd_nn

            zvzb_force_batch = vmc_nn_forces_zvzb(
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
                g.create_dataset(
                    "charges", data=charges,
                )
                g.create_dataset(
                    "atom_coords", data=nuc_crds,
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
        mc_stepsize = (3 * mc_timestep) ** 0.5

        # --- Equilibration ---
        @jax.jit
        def eq_step(state, _):
            rk, w, s = state
            rk, key = jax.random.split(rk)
            keys = jax.random.split(
                key, num_walkers,
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
            print(
                f"Equilibration acceptance rate:"
                f" {ratio:.2f}"
            )
            print(
                f"Adjusted step size:"
                f" {mc_stepsize:.4f} bohr"
                f" ~ {mc_timestep:.4f} Ha^-1"
            )

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

        if verbose >= 1:
            print(
                "# block     E_mean"
                "          E_std"
                "         ee_pot"
                "         en_pot"
                "        kinetic"
                "      dt_block"
            )

        E_blocks = []
        timestamp_prev = datetime.now()

        for blk in range(1, num_blocks + 1):
            state = (rng_key, walkers, mc_stepsize)
            state, result = jax.lax.scan(
                prod_step, state,
                jnp.arange(num_steps_per_block),
            )
            rng_key, walkers, _ = state

            if compute_gradients:
                (ratios, e_ee, e_en,
                 e_ke, sampled_w) = result
            else:
                ratios, e_ee, e_en, e_ke = result

            E_loc = e_ee + e_en + e_ke + enr_nn
            # E_loc: (num_steps_per_block,num_walkers)

            E_step = E_loc.mean(axis=1)
            E_mean = E_step.mean()
            E_std = E_step.std()
            E_blocks.append(float(E_mean))

            if verbose >= 1:
                ee_m = e_ee.mean()
                en_m = e_en.mean()
                ke_m = e_ke.mean()
                now = datetime.now()
                dt = (now - timestamp_prev
                      ).total_seconds()
                print(
                    f"{blk:>7d}"
                    f"{E_mean:>16.8e}"
                    f"{E_std:>14.8e}"
                    f"{ee_m:>14.8e}"
                    f"{en_m:>14.8e}"
                    f"{ke_m:>14.8e}"
                    f"{dt:>14.4f}"
                )
                timestamp_prev = now

            if compute_gradients:
                # sampled_w: (steps, walkers, nel, 3)
                n_samples = (
                    num_steps_per_block
                    * num_walkers
                )
                num_batches = (
                    (n_samples + batch_size - 1)
                    // batch_size
                )
                save_nn_forces(
                    blk,
                    sampled_w.reshape(
                        -1, nelec, 3,
                    ),
                    E_loc.reshape(-1),
                    float(E_mean),
                    batch_size,
                    num_batches,
                    ofname_grd,
                    zvzb_force_batch,
                )

        # --- Binning analysis ---
        E_arr = jnp.array(E_blocks)
        e_mean, e_serr, _, e_kappa = (
            do_binning_analysis(E_arr)
        )
        e_neff = E_arr.shape[0] / e_kappa

        timestamp_fin = datetime.now()
        elapsed = (
            timestamp_fin - timestamp_init
        ).total_seconds()

        if verbose >= 1:
            print(
                f"\nVMC energy: {e_mean:.8f}"
                f" +/- {e_serr:.8f} Ha"
                f" (N_eff = {e_neff:.1f})"
            )
            print(
                f"Total time: {elapsed:.2f} seconds"
            )

        result = {
            'E_mean': float(e_mean),
            'E_serr': float(e_serr),
            'E_blocks': E_blocks,
        }

        chkpt_file = f"{prefix}.chk.h5"
        from .nn_checkpoint import append_vmc_results
        append_vmc_results(chkpt_file, result)
        if verbose >= 1:
            print(
                f"VMC results written to"
                f" {chkpt_file}"
            )

        return result


def get_vmc_nn_func(mol_info, config, init_key):
    """Construct a VMC driver for NN wavefunctions.

    Builds the NN trial wavefunction from *config*,
    compiles the Metropolis kernel and local-energy
    functions, and returns a callable driver.

    Args:
        mol_info: :class:`~OmegaQMC.psi.nn.wf.MoleculeInfo`
            instance describing the molecule.
        config: :class:`~OmegaQMC.psi.nn.config.NNAnsatzConfig`
            or a string (built-in name or YAML path).
        init_key: JAX PRNG key for parameter
            initialisation.

    Returns:
        :class:`_VMCDriverNN` instance.  Call it with
        ``driver(rng_key, ...)`` to run the VMC
        simulation.
    """
    return _VMCDriverNN(mol_info, config, init_key)
