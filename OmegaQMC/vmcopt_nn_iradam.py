"""Iteratively-resampled Adam VMC optimizer for NN
wavefunctions.

Optimises Jastrow + backflow parameters via an outer loop
that alternates between:

1. **Sampling** — collect fresh walker snapshots from the
   current |ψ(params)|² via Metropolis Monte Carlo.
2. **Optimization** — run several Adam epochs on those
   snapshots.

Periodic resampling keeps the training distribution
aligned with the evolving wavefunction, avoiding the
stale-sample problem of a single sample-then-optimize
cycle and converging comparably to the interleaved
approach used by DeepQMC.

Unlike :mod:`vmcopt_gto_irsgd`, the local energy is
computed entirely from the NN trial wavefunction via the
O(N) Laplacian from
:func:`~OmegaQMC.psi.nn.physics.laplacian`, with no
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
        compiled = jax.jit(compute_energy_fn) \
            .lower(probe, params).compile()
        analysis = compiled.memory_analysis()
        bytes_per_walker = analysis.alias_size + analysis.temp_size
    except Exception:
        pass

    if not bytes_per_walker:
        bytes_per_walker = 2.0e6  # 2 MB fallback
    free_bytes = (free_mb or 4096.0) * 1e6 * mem_frac
    bs = int(free_bytes / bytes_per_walker)
    return max(10, min(bs, 8192))


# Target total parameter updates when num_iters='auto'.
# DeepQMC typically needs 50 000–100 000 interleaved
# sample-then-update steps to converge a PsiFormer;
# 50 000 is a conservative default for iterative
# resampling with multiple Adam epochs per iteration.
_TARGET_UPDATES = 50000


class _VMCOptDriverNN_IRAdam:
    """Iteratively-resampled Adam VMC optimizer for NN.

    Compiles the Metropolis kernel and local-energy
    function for a given molecule, then runs an outer
    loop of (resample walkers → Adam epochs) until the
    target number of parameter updates is reached.
    """

    def __init__(self, mol_info, config, init_key):
        nuc_crds = jnp.asarray(
            mol_info.coords, dtype=jnp.float64,
        )
        charges = jnp.asarray(
            mol_info.charges, dtype=jnp.float64,
        )
        nelec = mol_info.n_up + mol_info.n_down
        # n_up = mol_info.n_up
        # n_down = mol_info.n_down

        self.mol_info = mol_info
        self.nuc_crds = nuc_crds
        self.charges = charges
        self.nelec = nelec
        self.config_name = \
            config if isinstance(config, str) \
            else getattr(config, 'name', 'custom')

        log_psi, init_params, graphdef, lap_grad = (
            make_nn_log_psi(
                config, mol_info, init_key,
            )
        )
        self.init_params = init_params
        self.lap_grad = lap_grad

        # Precompute nuclear repulsion
        n_nuc = len(charges)
        enr_nn = 0.0
        for a in range(n_nuc):
            for b in range(a + 1, n_nuc):
                rab = jnp.linalg.norm(
                    nuc_crds[a] - nuc_crds[b],
                )
                enr_nn = enr_nn + (charges[a] * charges[b] / rab)
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
            diffs = elec_crds[:, None, :] - nuc_crds[None, :, :]
            dists = jnp.linalg.norm(diffs, axis=-1)
            return -jnp.sum(
                charges[None, :] / dists,
            )

        # --- Kinetic energy via NN Laplacian ---
        @jax.jit
        def energy_ke(elec_crds, params):
            lap_val, grad_val = lap_grad(
                elec_crds, nuc_crds, params,
            )
            return -0.5 * (
                lap_val + jnp.dot(grad_val, grad_val)
            )

        # --- Total local energy ---
        @jax.jit
        def total_local_energy(elec_crds, params):
            return energy_ee(elec_crds) + energy_en(elec_crds) \
                + energy_ke(elec_crds, params) + enr_nn

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
            diffs_en = proposed[:, None, :] - nuc_crds[None, :, :]
            dists_en = jnp.linalg.norm(
                diffs_en, axis=-1,
            )
            valid = (dists_en.min() > MIN_DIST_THRESHOLD) \
                & (dists_ee.min() > MIN_DIST_THRESHOLD)
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
            return 0.2 * energies.mean() + 0.8 * energies.std()

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
        return centers[None, :, :] \
            + 0.05 * jax.random.normal(rng_key, (num_walkers, self.nelec, 3))

    def __call__(
        self,
        rng_key,
        num_iters='auto',
        num_epochs=20,
        num_walkers='auto',
        num_steps_per_block=200,
        num_steps_decorr=1,
        num_sample_blocks=5,
        num_blocks_equil=5,
        mc_timestep=0.1,
        lr=1e-3,
        train_split=0.8,
        batch_size=200,
        verbose=1,
        prefix='nnopt',
    ):
        """Run iterative VMC optimization for NN
        wavefunctions.

        Uses an outer loop that alternates between
        (1) sampling fresh walker snapshots from the
        current |ψ(params)|² and (2) running
        ``num_epochs`` Adam passes over those
        snapshots.  This avoids the stale-sample
        problem of a single sample-then-optimize
        cycle.

        After each iteration the current parameters
        are written to ``{prefix}.chk.h5``.  Before
        each write the previous file is preserved as
        ``{prefix}.{iter}.h5`` so every completed
        iteration is recoverable.

        Args:
            rng_key: JAX PRNG key.
            num_iters: Number of outer
                (resample + optimize) iterations,
                or ``'auto'`` to target
                ~50 000 total parameter updates.
            num_epochs: Adam passes over sampled
                data per iteration.
            num_walkers: Number of MC walkers, or
                ``'auto'`` to set from GPU memory.
            num_steps_per_block: MC steps per
                production block.
            num_steps_decorr: Decorrelation steps
                between samples.
            num_sample_blocks: Production blocks
                per iteration for collecting fresh
                samples.
            num_blocks_equil: Equilibration blocks
                (initial only).
            mc_timestep: Initial MC timestep.
            lr: Adam learning rate.
            train_split: Fraction of data for
                training (rest for validation).
            batch_size: Batch size for Adam.
            verbose: Verbosity level
                (0 = silent).
            prefix: Filename prefix for the HDF5
                checkpoint.

        Returns:
            Tuple ``(params_final, energy_data)``
            where *params_final* is the optimized
            NNX parameter pytree and *energy_data*
            is a dict with key ``'energy'``.
        """
        params = self.init_params
        optimizer = optax.adam(learning_rate=lr)
        opt_state = optimizer.init(params)
        start_iter = 0

        # Resume from checkpoint if one exists
        chkpt_path = f"{prefix}.chk.h5"
        if os.path.exists(chkpt_path):
            template_leaves = jax.tree.leaves(
                params,
            )
            n_model = len(template_leaves)
            try:
                with h5py.File(
                    chkpt_path, 'r',
                ) as f:
                    n_chk = int(f['params'].attrs['num_leaves'])
                    if n_chk != n_model:
                        print(f"Error: checkpoint '{chkpt_path}'"
                              f" has {n_chk} parameter leaves"
                              f" but current model has {n_model}."
                              " Incompatible architecture — stopping.")
                        return None, {}
                    for i, leaf in enumerate(template_leaves):
                        chk_shape = f['params'][str(i)].shape
                        if chk_shape != leaf.shape:
                            print(f"Error: parameter leaf {i} shape mismatch:"
                                  f" checkpoint {chk_shape}"
                                  f" vs model {leaf.shape}."
                                  " Incompatible architecture — stopping.")
                            return None, {}
            except (KeyError, OSError) as exc:
                print(f"Error reading checkpoint"
                      f" '{chkpt_path}': {exc} — stopping.")
                return None, {}

            params, meta = load_nn_checkpoint(
                chkpt_path, params,
            )
            start_iter = int(meta.get('epoch', -1)) + 1
            opt_state = optimizer.init(params)
            if verbose >= 1:
                print(f"Resuming from '{chkpt_path}'"
                      f" (iteration {start_iter - 1} completed,"
                      f" continuing from iteration {start_iter})")

        # Auto-tune walker count and iterations
        auto_walkers = num_walkers == 'auto'
        auto_iters = num_iters == 'auto'
        if auto_walkers:
            free_mb = _get_free_gpu_mb()
            num_walkers = _autotune_nn_batch(
                self.compute_batch_energy,
                self.nelec, params, free_mb,
            )
        n_train_per_iter = int(
            train_split
            * num_walkers
            * num_sample_blocks
        )
        updates_per_iter = num_epochs * max(1, n_train_per_iter // batch_size)
        if auto_iters:
            num_iters = max(1, _TARGET_UPDATES // updates_per_iter)
        if (auto_walkers or auto_iters) and verbose >= 1:
            print(f"  Auto-tuned:"
                  f" num_walkers={num_walkers}, num_iters={num_iters}")

        # Initialize walkers
        rng_key, rng = jax.random.split(rng_key)
        walkers = self.initialize_walkers(
            rng, num_walkers,
        )
        mc_stepsize = (3 * mc_timestep) ** 0.5

        # Initial equilibration
        if verbose >= 1:
            print("Running equilibration...")
        (rng_key, walkers, mc_stepsize, _), \
            acc = self.run_equilibration(
                    rng_key, walkers, mc_stepsize, params,
                    num_blocks_equil, num_steps_per_block,
                )
        if verbose >= 1:
            print(f"  acceptance rate: {acc[-1]:.2f}")
            print(f"  step size: {mc_stepsize:.4f}")
            total_updates = num_iters * updates_per_iter
            print(f"\nStarting {num_iters} iterations"
                  f" (~{total_updates} param updates, lr={lr})...\n")

        # ===== Main iterative loop =====
        for iteration in range(
            start_iter, start_iter + num_iters,
        ):
            # (a) Sample fresh walker snapshots
            all_samples = []
            for blk in range(num_sample_blocks):
                (
                    rng_key, walkers,
                    mc_stepsize, _,
                ), (acc_r, tw_e) = (
                    self.run_production(
                        rng_key, walkers,
                        mc_stepsize, params,
                        num_steps_per_block,
                        num_steps_decorr,
                    )
                )
                all_samples.append(walkers)
            sampled = jnp.vstack(
                all_samples,
            ).reshape(-1, self.nelec, 3)
            n_samples = sampled.shape[0]
            rng_key, rng1 = jax.random.split(
                rng_key,
            )
            idx = jax.random.permutation(
                rng1, jnp.arange(n_samples),
            )
            n_train = int(train_split * n_samples)
            train_w = sampled[idx[:n_train]]
            valid_w = sampled[idx[n_train:]]

            # (b) Adam optimization epochs
            epoch_losses = []
            for ep in range(num_epochs):
                for si in range(
                    0, n_train, batch_size,
                ):
                    ei = min(
                        si + batch_size,
                        n_train,
                    )
                    batch = train_w[si:ei]
                    loss, grads = (
                        jax.value_and_grad(
                            self.loss_fn,
                        )(params, batch)
                    )
                    updates, opt_state = (
                        optimizer.update(
                            grads, opt_state,
                            params,
                        )
                    )
                    params = optax.apply_updates(
                        params, updates,
                    )
                    epoch_losses.append(loss)

            # (c) Validation energy
            v_energies = []
            for si in range(
                0, valid_w.shape[0], batch_size,
            ):
                ei = min(
                    si + batch_size,
                    valid_w.shape[0],
                )
                v_energies.append(
                    self.compute_batch_energy(
                        valid_w[si:ei], params,
                    )
                )
            all_e = jnp.concatenate(v_energies)
            iter_e = float(all_e.mean())
            iter_err = float(all_e.std()) / all_e.size ** 0.5

            if verbose >= 1:
                iter_loss = float(jnp.array(epoch_losses).mean())
                print(f"Iter {iteration:5d}"
                      f" | E = {iter_e:.8f} +/- {iter_err:.8f}"
                      f" | Loss: {iter_loss:.6f}")

            # (d) Checkpoint
            if os.path.exists(chkpt_path):
                os.rename(
                    chkpt_path,
                    f"{prefix}.{iteration}.h5",
                )
            save_nn_checkpoint(
                chkpt_path, params, iteration,
                self.config_name,
                self.mol_info,
                energy=iter_e,
            )

        # ===== Final energy estimate =====
        if verbose >= 1:
            print("\nFinal energy evaluation...")
        (rng_key, walkers, mc_stepsize, _), \
            acc = (
                self.run_equilibration(
                    rng_key, walkers,
                    mc_stepsize, params,
                    num_blocks_equil,
                    num_steps_per_block,
                )
            )
        (rng_key, walkers, _, _), \
            (_, tw_e) = (
                self.run_production(
                    rng_key, walkers,
                    mc_stepsize, params,
                    num_steps_per_block,
                    num_steps_decorr,
                )
            )
        final_e = float(jnp.mean(tw_e))
        final_std = float(jnp.std(tw_e))
        neff = tw_e.size
        final_err = final_std / neff ** 0.5

        if verbose >= 1:
            print(f"Final energy: {final_e:.8f} +/- {final_err:.8f}")

        return params, {'energy': {'mean': final_e, 'stderr': final_err}}


def get_vmcopt_nn_func(mol_info, config, init_key):
    """Create an iteratively-resampled Adam VMC
    optimizer for NN.

    Builds the NN trial wavefunction from *config*,
    compiles the Metropolis kernel and local-energy
    function, and returns a callable driver.

    Args:
        mol_info: :class:`~OmegaQMC.utils.Mole_custom`
            instance.
        config: :class:`~OmegaQMC.psi.nn.config.NNAnsatzConfig`
            or a string (built-in name or YAML path).
        init_key: JAX PRNG key for parameter
            initialisation.

    Returns:
        :class:`_VMCOptDriverNN_IRAdam` instance.
        Call it with ``driver(rng_key, ...)`` to run
        the optimization.
    """
    return _VMCOptDriverNN_IRAdam(
        mol_info, config, init_key,
    )


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
    raise NotImplementedError("pretrain_to_hf is not yet implemented")
