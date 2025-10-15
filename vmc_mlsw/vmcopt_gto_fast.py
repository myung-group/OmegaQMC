import jax
import jax.numpy as jnp
import optax
from functools import partial
from vmc_mlsw.psi_gto import get_psi_fun


def get_vmcopt_func(mf, params_vmc, cgto_coeff=None):
    """Create VMC optimization function with improved efficiency."""

    # Precompute static quantities
    nuc_crds = jnp.array(mf.mol.atom_coords(unit='Bohr'))
    nelec = mf.mol.tot_electrons()
    Z_charges = mf.mol.atom_charges()
    i_e, j_e = jnp.triu_indices(nelec, k=1)

    # Get energy functions
    log_trial_wavefunction, local_energy, get_psi_mo = get_psi_fun(mf, cgto_coeff)
    local_energy_ee, local_energy_nn, local_energy_en, local_energy_ke = local_energy
    ener_nn = local_energy_nn(nuc_crds)

    # No Jastrow parameters (static)
    no_jastrow_params = {
        'J1_params': jnp.array([]),
        'J2_params': jnp.array([])
    }

    # Constants
    MIN_DIST_THRESHOLD = 1e-4

    @jax.jit
    def metropolis_move(rng_key, elec_crds, step_size):
        """Metropolis step with improved distance calculations."""
        key_prop, key_accept = jax.random.split(rng_key)

        # Propose new coordinates
        proposed_crds = elec_crds + step_size * jax.random.normal(key_prop, elec_crds.shape)

        # Vectorized distance checks
        diffs_ee = proposed_crds[i_e] - proposed_crds[j_e]
        dists_ee = jnp.linalg.norm(diffs_ee, axis=-1)

        diffs_en = proposed_crds[:, None, :] - nuc_crds[None, :, :]
        dists_en = jnp.linalg.norm(diffs_en, axis=-1)

        valid_move = (dists_en.min() > MIN_DIST_THRESHOLD) & (dists_ee.min() > MIN_DIST_THRESHOLD)

        # Compute acceptance probability
        log_psi_old = log_trial_wavefunction(elec_crds, nuc_crds, no_jastrow_params)
        log_psi_new = log_trial_wavefunction(proposed_crds, nuc_crds, no_jastrow_params)

        accept = (jax.random.uniform(key_accept) < jnp.exp(2 * (log_psi_new - log_psi_old))) & valid_move
        new_crds = jnp.where(accept, proposed_crds, elec_crds)

        return new_crds, accept

    @jax.jit
    def total_local_energy_fn(elec_crds, curr_params):
        """Compute total local energy."""
        return (local_energy_ee(elec_crds) +
                local_energy_en(elec_crds, nuc_crds) +
                local_energy_ke(elec_crds, nuc_crds, curr_params) +
                ener_nn)

    def initialize_walkers(rng_key, nwalkers):
        """Initialize electron positions efficiently."""
        idx_cnt = []
        for ia, iz in enumerate(Z_charges):
            idx_cnt.extend([ia] * int(iz))

        # Handle molecular charge
        charge = int(mf.mol.charge)
        if charge < 0:
            idx_cnt.extend([0] * abs(charge))
        elif charge > 0:
            idx_cnt = idx_cnt[:-charge]

        idx_cnt = jnp.array(idx_cnt)
        centers = nuc_crds[idx_cnt]
        walkers = centers[jnp.newaxis, :, :] + 0.05 * jax.random.normal(rng_key, (nwalkers, nelec, 3))

        return walkers

    #@jax.jit
    def run_equilibration(rng_key, walkers, num_equilibration, step_size):
        """Equilibration phase with adaptive step size."""

        @jax.jit
        def equilibration_step(carried_in, _):
            rkey, w, s = carried_in
            rkey0, rkey1 = jax.random.split(rkey)
            keys = jax.random.split(rkey1, w.shape[0])

            # The issue was here: s (step_size) is dynamic, but metropolis_move was jitted
            # with step_size as a static argument. Now metropolis_move is only jitted once
            # and takes s as a dynamic argument.
            new_w, accepted = jax.vmap(metropolis_move, in_axes=(0, 0, None))(keys, w, s)
            acceptance_rate = accepted.mean()
            new_s = s * (0.6 + acceptance_rate)

            return (rkey0, new_w, new_s), acceptance_rate

        carry_in = (rng_key, walkers, step_size)
        carry_out, acc_ratios = jax.lax.scan(equilibration_step, carry_in, jnp.arange(num_equilibration))

        return carry_out, acc_ratios

    @partial(jax.jit, static_argnums=(3, 4))
    def run_production(rng_key, walkers, step_size, num_steps, sample_every=10):
        """Production phase with sampling."""

        @jax.jit
        def production_step(carried_in, step_idx):
            rkey, w, s = carried_in
            rkey0, rkey1 = jax.random.split(rkey)
            keys = jax.random.split(rkey1, w.shape[0])

            # No issue here, but `metropolis_move` must be consistent
            new_w, accepted = jax.vmap(metropolis_move, in_axes=(0, 0, None))(keys, w, s)
            acceptance_rate = accepted.mean()
            new_s = s * (0.6 + acceptance_rate)

            # Calculate energies
            energies = jax.vmap(total_local_energy_fn, in_axes=(0, None))(new_w, no_jastrow_params)

            # Mark samples for saving
            should_sample = (step_idx + 1) % sample_every == 0

            return (rkey0, new_w, new_s), (energies, new_w, should_sample)

        carry_in = (rng_key, walkers, step_size)
        carry_out, results = jax.lax.scan(production_step, carry_in, jnp.arange(num_steps))

        return carry_out, results

    @jax.jit
    def compute_batch_energy(walkers, params):
        """Compute energy for a batch of walkers."""
        return jax.vmap(total_local_energy_fn, in_axes=(0, None))(walkers, params)

    @jax.jit
    def loss_fn(params, batch_walkers):
        """Combined mean and variance loss."""
        energies = compute_batch_energy(batch_walkers, params)
        enr_mean = energies.mean()
        enr_std = energies.std()
        return 0.2 * enr_mean + 0.8 * enr_std

    def compute_statistics(walkers, params, batch_size):
        """Compute energy statistics for dataset."""
        n_samples = walkers.shape[0]
        energies = []

        for start_idx in range(0, n_samples, batch_size):
            end_idx = min(start_idx + batch_size, n_samples)
            batch_energies = compute_batch_energy(walkers[start_idx:end_idx], params)
            energies.append(batch_energies.mean())

        return jnp.array(energies).mean()

    def vmcopt_run(rng_key,
                   nwalkers=100,
                   params_init=None,
                   num_steps=50000,
                   num_epochs=20,
                   num_equilibration=5000,
                   step_size=0.25,
                   lr=0.02,
                   optimizer="sgd",
                   num_production_runs=3,
                   sample_every=10,
                   train_split=0.8,
                   batch_size=1000,
                   verbose=True):
        """
        VMC optimization run with improved efficiency.

        Args:
            rng_key: JAX random key
            nwalkers: Number of walkers
            params_init: Initial parameters (if None, use params_vmc)
            num_steps: Steps per production run
            num_epochs: Number of training epochs
            num_equilibration: Equilibration steps
            step_size: Initial step size
            lr: Learning rate
            optimizer: "sgd" or "adam"
            num_production_runs: Number of production runs
            sample_every: Sample walkers every N steps
            train_split: Fraction of data for training
            batch_size: Batch size for training
            verbose: Print progress
        """

        # Initialize parameters
        params = params_vmc if params_init is None else jnp.array(params_init, dtype=jnp.float64)

        # Initialize optimizer
        optimizer_chosen = optax.adam(learning_rate=lr) if "adam" in optimizer.lower() else optax.sgd(learning_rate=lr)
        opt_state = optimizer_chosen.init(params)

        # Initialize walkers
        rng_key, rng = jax.random.split(rng_key)
        walkers = initialize_walkers(rng, nwalkers)

        # Equilibration
        if verbose:
            print("Running equilibration...")
        (rng_key, walkers, step_size), acc_ratios = run_equilibration(
            rng_key, walkers, num_equilibration, step_size
        )

        if verbose:
            print(f"Equilibration acceptance rate: {acc_ratios[-1]:.2f}")
            print(f"Adjusted step size: {step_size:.4f}")

        # Production runs - collect samples
        if verbose:
            print(f"\nRunning {num_production_runs} production runs...")

        all_samples = []
        for run_idx in range(num_production_runs):
            (rng_key, walkers, step_size), (tw_energies, sampled_walkers, sample_mask) = run_production(
                rng_key, walkers, step_size, num_steps, sample_every
            )

            # Extract sampled walkers
            samples = sampled_walkers[sample_mask]
            all_samples.append(samples)

            if verbose:
                print(f"  Run {run_idx + 1}/{num_production_runs}: collected {samples.shape[0]} samples")

        # Combine and shuffle samples
        sampled_walkers = jnp.vstack(all_samples).reshape(-1, nelec, 3)
        n_samples = sampled_walkers.shape[0]

        # Train/validation split
        rng_key, rng_key1 = jax.random.split(rng_key)
        idx = jax.random.permutation(rng_key1, jnp.arange(n_samples))

        n_train = int(train_split * n_samples)
        train_walkers = sampled_walkers[idx[:n_train]]
        valid_walkers = sampled_walkers[idx[n_train:]]

        if verbose:
            print(f"\nTraining on {n_train} samples, validating on {n_samples - n_train} samples")
            print(f"\nStarting optimization for {num_epochs} epochs...\n")

        # Training loop
        for epoch in range(num_epochs):
            # Training
            epoch_losses = []
            for start_idx in range(0, n_train, batch_size):
                end_idx = min(start_idx + batch_size, n_train)
                batch = train_walkers[start_idx:end_idx]

                loss, grad_mean = jax.value_and_grad(loss_fn)(params, batch)
                updates, opt_state = optimizer_chosen.update(grad_mean, opt_state, params)
                params = optax.apply_updates(params, updates)

                epoch_losses.append(loss)

            # Compute statistics
            train_loss = jnp.array(epoch_losses).mean()
            train_energy = compute_statistics(train_walkers, params, batch_size)
            valid_energy = compute_statistics(valid_walkers, params, batch_size)

            # Compute validation loss
            valid_losses = []
            for start_idx in range(0, valid_walkers.shape[0], batch_size):
                end_idx = min(start_idx + batch_size, valid_walkers.shape[0])
                valid_losses.append(loss_fn(params, valid_walkers[start_idx:end_idx]))
            valid_loss = jnp.array(valid_losses).mean()

            if verbose:
                print(f"Epoch {epoch:3d} | "
                      f"Loss: {train_loss:.6f} | "
                      f"Train E: {train_energy:.6f} | "
                      f"Valid Loss: {valid_loss:.6f} | "
                      f"Valid E: {valid_energy:.6f}")

        if verbose:
            print(f"\nOptimized parameters: {params}")

        return params

    return vmcopt_run
