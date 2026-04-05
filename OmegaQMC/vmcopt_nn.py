"""Post-sampling Adam VMC optimizer for NN wavefunctions.

Implements Jastrow + backflow parameter optimization via a
two-phase approach:

1. **Sampling phase** — equilibrate walkers, then run
   production MC blocks and store final walker positions.
2. **Optimization phase** — minimize a combined
   energy-plus-variance loss on the stored snapshots
   using Adam (via Optax).

Unlike :mod:`vmcopt_gto_pssgd`, the local energy is computed
entirely from the NN trial wavefunction via the O(N) Laplacian
from :func:`~OmegaQMC.psi.nn.physics.laplacian`, with no
PySCF dependency.
"""

import os
import h5py
import jax
import jax.numpy as jnp
import optax
from functools import partial

from .nn_checkpoint import (
    save_nn_checkpoint,
    load_nn_checkpoint,
)
from .psi.nn.adapter import make_nn_log_psi
from .psi.nn.physics import laplacian
from .psi.nn.wf import MoleculeInfo
from .constants import MIN_DIST_THRESHOLD
from .vmcopt_gto_linear import _get_free_gpu_mb


def _autotune_nn_batch(
        compute_energy_fn, nelec, params,
        free_mb, mem_frac=0.75,
):
    """Choose walker batch size for NN energy eval.

    Compiles the vmapped energy function for a single
    walker to measure per-walker GPU memory via JAX AOT
    analysis.  Falls back to a 2 MB/walker heuristic
    when AOT is unavailable (NN walkers are costlier
    than GTO walkers due to the full Laplacian).

    Parameters
    ----------
    compute_energy_fn : callable
        Vmapped energy function
        ``(walkers, params) -> energies``.
    nelec : int
        Number of electrons.
    params : pytree
        Representative NN parameter pytree.
    free_mb : float or None
        Free GPU memory in MiB; ``None`` assumes 4096.
    mem_frac : float
        Fraction of free memory to target (0.75).

    Returns
    -------
    int
        Recommended walker batch size.
    """
    bytes_per_walker = None
    try:
        probe = jnp.zeros((1, nelec, 3))
        compiled = (
            jax.jit(compute_energy_fn)
            .lower(probe, params)
            .compile()
        )
        analysis = compiled.memory_analysis()
        bytes_per_walker = (
            analysis.alias_size
            + analysis.temp_size
        )
    except Exception:
        pass

    if not bytes_per_walker:
        bytes_per_walker = 2.0e6  # 2 MB fallback
    free_bytes = (
        (free_mb or 4096.0) * 1e6 * mem_frac
    )
    bs = int(free_bytes / bytes_per_walker)
    return max(10, min(bs, 8192))


class _VMCOptDriverNN:
    """Post-sampling Adam VMC optimizer for NN trials.

    Compiles the Metropolis kernel and local-energy
    function for a given molecule, collects walker
    snapshots in a sampling phase, then optimizes
    the full NN parameter set on those snapshots with
    Adam.
    """

    def __init__(self, mol_info, config, init_key):
        nuc_crds = jnp.asarray(
            mol_info.coords, dtype=jnp.float64,
        )
        charges = jnp.asarray(
            mol_info.charges, dtype=jnp.float64,
        )
        nelec = mol_info.n_up + mol_info.n_down
        n_up = mol_info.n_up
        n_down = mol_info.n_down

        self.mol_info = mol_info
        self.nuc_crds = nuc_crds
        self.charges = charges
        self.nelec = nelec
        self.config_name = (
            config if isinstance(config, str)
            else getattr(config, 'name', 'custom')
        )

        log_psi, init_params, graphdef = make_nn_log_psi(
            config, mol_info, init_key,
        )
        self.init_params = init_params

        # Precompute nuclear repulsion
        n_nuc = len(charges)
        enr_nn = 0.0
        for a in range(n_nuc):
            for b in range(a + 1, n_nuc):
                rab = jnp.linalg.norm(
                    nuc_crds[a] - nuc_crds[b],
                )
                enr_nn = enr_nn + (
                    charges[a] * charges[b] / rab
                )
        enr_nn = jnp.asarray(enr_nn, dtype=jnp.float64)

        i_e, j_e = jnp.triu_indices(nelec, k=1)

        # --- Electron-electron energy ---
        @jax.jit
        def energy_ee(elec_crds):
            diffs = elec_crds[i_e] - elec_crds[j_e]
            dists = jnp.linalg.norm(diffs, axis=-1)
            return jnp.sum(1.0 / dists)

        # --- Electron-nucleus energy ---
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

        # --- Kinetic energy via NN Laplacian ---
        @jax.jit
        def energy_ke(elec_crds, params):
            def f_flat(r_flat):
                r = r_flat.reshape(nelec, 3)
                return log_psi(r, nuc_crds, params)
            r_flat = elec_crds.reshape(-1)
            lap_fn = laplacian(f_flat)
            lap_val, grad_val = lap_fn(r_flat)
            return -0.5 * (
                lap_val + jnp.dot(grad_val, grad_val)
            )

        # --- Total local energy ---
        @jax.jit
        def total_local_energy(elec_crds, params):
            return (
                energy_ee(elec_crds)
                + energy_en(elec_crds)
                + energy_ke(elec_crds, params)
                + enr_nn
            )

        # --- Metropolis move ---
        @jax.jit
        def metropolis_move(
            rng_key, elec_crds, step_size, params,
        ):
            key_prop, key_accept = jax.random.split(
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
                jax.random.uniform(key_accept)
                < jnp.exp(2 * (lp_new - lp_old))
            ) & valid
            new_crds = jnp.where(
                accept, proposed, elec_crds,
            )
            return new_crds, accept

        # --- Equilibration scan ---
        @partial(jax.jit, static_argnums=(4, 5))
        def run_equilibration(
            rng_key, walkers, step_size, params,
            num_be, num_spb,
        ):
            def eq_step(carry, _):
                rk, w, s, p = carry
                rk0, rk1 = jax.random.split(rk)
                keys = jax.random.split(
                    rk1, w.shape[0],
                )
                nw, acc = jax.vmap(
                    metropolis_move,
                    in_axes=(0, 0, None, None),
                )(keys, w, s, p)
                ar = acc.mean()
                ns = s * (0.6 + ar)
                return (rk0, nw, ns, p), ar

            for _ in range(num_be):
                carry = (
                    rng_key, walkers,
                    step_size, params,
                )
                carry, acc = jax.lax.scan(
                    eq_step, carry,
                    jnp.arange(num_spb),
                )
            return carry, acc

        # --- Production scan ---
        @partial(jax.jit, static_argnums=(4, 5))
        def run_production(
            rng_key, walkers, step_size, params,
            num_spb, num_dc,
        ):
            def prod_step(carry, _):
                rk, w, s, p = carry
                for _ in range(num_dc):
                    rk0, rk1 = jax.random.split(rk)
                    keys = jax.random.split(
                        rk1, w.shape[0],
                    )
                    nw, acc = jax.vmap(
                        metropolis_move,
                        in_axes=(0, 0, None, None),
                    )(keys, w, s, p)
                    w = nw
                    rk = rk0
                ar = acc.mean()
                energies = jax.vmap(
                    total_local_energy,
                    in_axes=(0, None),
                )(nw, p)
                return (rk, nw, s, p), (ar, energies)

            carry = (
                rng_key, walkers,
                step_size, params,
            )
            carry, results = jax.lax.scan(
                prod_step, carry,
                jnp.arange(num_spb),
            )
            return carry, results

        # --- Batch energy ---
        @jax.jit
        def compute_batch_energy(walkers, params):
            return jax.vmap(
                total_local_energy,
                in_axes=(0, None),
            )(walkers, params)

        # --- Loss function ---
        @jax.jit
        def loss_fn(params, batch_walkers):
            energies = compute_batch_energy(
                batch_walkers, params,
            )
            return (
                0.2 * energies.mean()
                + 0.8 * energies.std()
            )

        self.run_equilibration = run_equilibration
        self.run_production = run_production
        self.compute_batch_energy = compute_batch_energy
        self.loss_fn = loss_fn

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

    def __call__(
        self,
        rng_key,
        num_epochs=20,
        num_walkers='auto',
        num_steps_per_block=200,
        num_steps_decorr=1,
        num_opt_samples='auto',
        num_blocks_equil=5,
        mc_timestep=0.1,
        lr=1e-3,
        train_split=0.8,
        batch_size=200,
        verbose=1,
        prefix='nnopt',
    ):
        """Run VMC optimization for NN wavefunctions.

        After each epoch the current parameters are
        written to ``{prefix}.chk.h5``.  Before each
        write the previous ``{prefix}.chk.h5`` is
        preserved as ``{prefix}.{epoch}.h5`` so that
        every completed epoch is recoverable.

        Args:
            rng_key: JAX PRNG key.
            num_epochs: Number of Adam optimization
                epochs.
            num_walkers: Number of MC walkers, or
                ``'auto'`` to set from GPU memory.
            num_steps_per_block: MC steps per production
                block.
            num_steps_decorr: Decorrelation steps between
                samples.
            num_opt_samples: Total walker snapshots to
                collect for optimization, or ``'auto'``
                to set to ``5 * num_walkers``.
            num_blocks_equil: Number of equilibration
                blocks.
            mc_timestep: Initial MC timestep.
            lr: Adam learning rate.
            train_split: Fraction of data for training.
            batch_size: Batch size for optimization.
            verbose: Verbosity level (0 = silent).
            prefix: Filename prefix for the HDF5
                checkpoint.  The live checkpoint is
                ``{prefix}.chk.h5``; superseded
                checkpoints are renamed
                ``{prefix}.{epoch}.h5``.

        Returns:
            Tuple ``(params_final, energy_data)`` where
            *params_final* is the optimized NNX parameter
            pytree and *energy_data* is a dict with key
            ``'energy'``.
        """
        params = self.init_params
        optimizer = optax.adam(learning_rate=lr)
        opt_state = optimizer.init(params)
        start_epoch = 0

        # Resume from checkpoint if one exists
        chkpt_path = f"{prefix}.chk.h5"
        if os.path.exists(chkpt_path):
            template_leaves = jax.tree.leaves(params)
            n_model = len(template_leaves)
            try:
                with h5py.File(chkpt_path, 'r') as f:
                    n_chk = int(
                        f['params'].attrs['num_leaves']
                    )
                    if n_chk != n_model:
                        print(
                            f"Error: checkpoint"
                            f" '{chkpt_path}' has"
                            f" {n_chk} parameter"
                            f" leaves but current"
                            f" model has {n_model}."
                            " Incompatible"
                            " architecture — stopping."
                        )
                        return None, {}
                    for i, leaf in enumerate(
                        template_leaves
                    ):
                        chk_shape = (
                            f['params'][str(i)].shape
                        )
                        if chk_shape != leaf.shape:
                            print(
                                f"Error: parameter"
                                f" leaf {i} shape"
                                f" mismatch:"
                                f" checkpoint"
                                f" {chk_shape} vs"
                                f" model {leaf.shape}."
                                " Incompatible"
                                " architecture"
                                " — stopping."
                            )
                            return None, {}
            except (KeyError, OSError) as exc:
                print(
                    f"Error reading checkpoint"
                    f" '{chkpt_path}': {exc}"
                    " — stopping."
                )
                return None, {}

            params, meta = load_nn_checkpoint(
                chkpt_path, params,
            )
            start_epoch = (
                int(meta.get('epoch', -1)) + 1
            )
            opt_state = optimizer.init(params)
            if verbose >= 1:
                print(
                    f"Resuming from '{chkpt_path}'"
                    f" (epoch {start_epoch - 1}"
                    f" completed,"
                    f" continuing from"
                    f" epoch {start_epoch})"
                )

        # Auto-tune walker counts to fit GPU memory
        _need_auto = (
            num_walkers == 'auto'
            or num_opt_samples == 'auto'
        )
        if _need_auto:
            free_mb = _get_free_gpu_mb()
            auto_bs = _autotune_nn_batch(
                self.compute_batch_energy,
                self.nelec, params, free_mb,
            )
            if num_walkers == 'auto':
                num_walkers = auto_bs
            if num_opt_samples == 'auto':
                num_opt_samples = 5 * num_walkers
            if verbose >= 1:
                print(
                    f"  Auto-tuned:"
                    f" num_walkers={num_walkers},"
                    f" num_opt_samples="
                    f"{num_opt_samples}"
                )

        # Ceiling division: blocks needed to collect
        # at least num_opt_samples walker snapshots
        num_sample_blocks = (
            -(-num_opt_samples // num_walkers)
        )

        rng_key, rng = jax.random.split(rng_key)
        walkers = self.initialize_walkers(
            rng, num_walkers,
        )
        mc_stepsize = (3 * mc_timestep) ** 0.5

        if verbose >= 1:
            print("Running equilibration...")
        (rng_key, walkers, mc_stepsize, _), acc = (
            self.run_equilibration(
                rng_key, walkers, mc_stepsize,
                params,
                num_blocks_equil,
                num_steps_per_block,
            )
        )
        if verbose >= 1:
            print(
                f"  acceptance rate: {acc[-1]:.2f}"
            )
            print(
                f"  step size: {mc_stepsize:.4f}"
            )

        if verbose >= 1:
            print(
                f"\nRunning {num_sample_blocks}"
                " production blocks..."
            )
        all_samples = []
        for blk in range(1, num_sample_blocks + 1):
            (rng_key, walkers, mc_stepsize, _), (
                acc_r, tw_e,
            ) = self.run_production(
                rng_key, walkers, mc_stepsize,
                params,
                num_steps_per_block,
                num_steps_decorr,
            )
            all_samples.append(walkers)
            if verbose >= 1:
                print(
                    f"  block"
                    f" {blk}/{num_sample_blocks}: "
                    f"{walkers.shape[0]} samples"
                )

        sampled = jnp.vstack(all_samples).reshape(
            -1, self.nelec, 3,
        )[:num_opt_samples]
        n_samples = sampled.shape[0]

        rng_key, rng1 = jax.random.split(rng_key)
        idx = jax.random.permutation(
            rng1, jnp.arange(n_samples),
        )
        n_train = int(train_split * n_samples)
        train_w = sampled[idx[:n_train]]
        valid_w = sampled[idx[n_train:]]

        if verbose >= 1:
            print(
                f"\nTraining on {n_train}, "
                f"validating on {n_samples - n_train}"
            )
            print(
                f"Starting {num_epochs} Adam epochs"
                f" (lr={lr})...\n"
            )

        for epoch in range(
            start_epoch, start_epoch + num_epochs,
        ):
            epoch_losses = []
            for si in range(0, n_train, batch_size):
                ei = min(si + batch_size, n_train)
                batch = train_w[si:ei]
                loss, grads = jax.value_and_grad(
                    self.loss_fn,
                )(params, batch)
                updates, opt_state = optimizer.update(
                    grads, opt_state, params,
                )
                params = optax.apply_updates(
                    params, updates,
                )
                epoch_losses.append(loss)

            train_loss = (
                jnp.array(epoch_losses).mean()
            )
            if verbose >= 1:
                v_losses = []
                for si in range(
                    0, valid_w.shape[0], batch_size,
                ):
                    ei = min(
                        si + batch_size,
                        valid_w.shape[0],
                    )
                    v_losses.append(
                        self.loss_fn(
                            params, valid_w[si:ei],
                        )
                    )
                valid_loss = (
                    jnp.array(v_losses).mean()
                )
                print(
                    f"Epoch {epoch:3d} | "
                    f"Loss: {train_loss:.6f} | "
                    f"Valid: {valid_loss:.6f}"
                )

            if os.path.exists(chkpt_path):
                os.rename(
                    chkpt_path,
                    f"{prefix}.{epoch}.h5",
                )
            save_nn_checkpoint(
                chkpt_path, params, epoch,
                self.config_name,
                self.mol_info,
                energy=float(train_loss),
            )

        # Final energy estimate
        v_energies = []
        for si in range(
            0, valid_w.shape[0], batch_size,
        ):
            ei = min(
                si + batch_size, valid_w.shape[0],
            )
            v_energies.append(
                self.compute_batch_energy(
                    valid_w[si:ei], params,
                )
            )
        all_e = jnp.concatenate(v_energies)
        final_e = float(all_e.mean())
        neff = all_e.size
        final_err = float(all_e.std()) / neff ** 0.5

        if verbose >= 1:
            print(
                f"\nFinal energy:"
                f" {final_e:.8f}"
                f" +/- {final_err:.8f}"
            )

        return params, {
            'energy': {
                'mean': final_e,
                'stderr': final_err,
            },
        }


def get_vmcopt_nn_func(mol_info, config, init_key):
    """Create a post-sampling Adam VMC optimizer for NN.

    Builds the NN trial wavefunction from *config*,
    compiles the Metropolis kernel and local-energy
    function, and returns a callable driver.

    Args:
        mol_info: :class:`~OmegaQMC.psi.nn.wf.MoleculeInfo`
            instance.
        config: :class:`~OmegaQMC.psi.nn.config.NNAnsatzConfig`
            or a string (built-in name or YAML path).
        init_key: JAX PRNG key for parameter
            initialisation.

    Returns:
        :class:`_VMCOptDriverNN` instance.  Call it with
        ``driver(rng_key, ...)`` to run the optimization.
    """
    return _VMCOptDriverNN(mol_info, config, init_key)


def pretrain_to_hf(nn_trial, mf, rng_key, steps=1000):
    """Pretrain NN wavefunction to match HF orbitals.

    Not yet implemented.  Will minimise the mean-squared
    error between the NN orbital matrix and the HF
    molecular orbitals on random electron configurations.

    Args:
        nn_trial: NN trial wavefunction (from
            :func:`~OmegaQMC.psi.nn.adapter.make_nn_log_psi`).
        mf: Converged PySCF mean-field object.
        rng_key: JAX PRNG key.
        steps: Number of pretraining steps.

    Raises:
        NotImplementedError: Always.
    """
    raise NotImplementedError(
        "pretrain_to_hf is not yet implemented"
    )
