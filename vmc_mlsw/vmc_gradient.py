import jax
import jax.numpy as jnp
import h5py


def create_vmc_gradient_save(
    log_trial_wavefunction,
    vmc_gradient_batch,
    rescale_fn,
    run_electron_exchange,
    nuc_crds,
    params_vmc,
    reflection_op_list,
    chkfile_grd='vmc_grd_chk.hdf5'
):
    """
    Create VMC gradient save function with importance reweighting.
    """

    def vmc_gradient_save_importance_reweighting(
        iter,
        sampled_walkers,
        local_energies,
        batch_size,
        num_batches,
        h5py_io='w'
    ):
        """
        Importance reweighting with consistent identity handling.

        All reflection operations (including 'I') are processed uniformly
        to ensure consistent coordinate transformations.
        """
        n_samples = sampled_walkers.shape[0]

        w_grd_ee_en_ke = []
        w_grd_logpsi = []
        w_grd_ke = []
        w_weights = []

        # Track statistics for each operation
        op_stats = {op: {'total_weight': 0.0, 'n_samples': 0}
                   for op in reflection_op_list}

        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, n_samples)
            batch_size_actual = end_idx - start_idx

            batch_samples = sampled_walkers[start_idx:end_idx]

            # Reference wavefunction for acceptance probability
            log_psi_old = jax.vmap(
                log_trial_wavefunction,
                in_axes=(0, None, None)
            )(batch_samples, nuc_crds, params_vmc)

            # Pre-compute rescale (used by all non-identity operations)
            rescale = jax.vmap(rescale_fn)(batch_samples)

            # Accumulate weighted gradients from all operations
            batch_grd_ee_en_ke = []
            batch_grd_logpsi = []
            batch_grd_ke = []
            batch_weights = []

            # Process ALL operations in a unified loop
            for reflection_op in reflection_op_list:

                if reflection_op == 'I':
                    # Identity: no transformation needed
                    elec_pos = batch_samples
                    accept_prob = jnp.ones(batch_size_actual)

                else:
                    # Non-identity: apply reflection transformation
                    elec_pos = run_electron_exchange(
                        batch_samples, rescale, reflection_op
                    )

                    # Calculate acceptance probability
                    log_psi_new = jax.vmap(
                        log_trial_wavefunction,
                        in_axes=(0, None, None)
                    )(elec_pos, nuc_crds, params_vmc)

                    log_acc = 2.0 * (log_psi_new - log_psi_old)
                    accept_prob = jnp.where(log_acc > 0.0, 1.0, jnp.exp(log_acc))

                # Calculate gradients for this operation
                g_ee, g_en, g_ke, g_logpsi = vmc_gradient_batch(elec_pos)

                # Weight by acceptance probability (importance reweighting)
                weighted_grd_ee_en_ke = (g_ee + g_en + g_ke) * accept_prob[:, None, None]
                weighted_grd_logpsi = g_logpsi * accept_prob[:, None, None]
                weighted_grd_ke = g_ke * accept_prob[:, None, None]

                batch_grd_ee_en_ke.append(weighted_grd_ee_en_ke)
                batch_grd_logpsi.append(weighted_grd_logpsi)
                batch_grd_ke.append(weighted_grd_ke)
                batch_weights.append(accept_prob)

                # Update statistics
                op_stats[reflection_op]['total_weight'] += float(accept_prob.sum())
                op_stats[reflection_op]['n_samples'] += batch_size_actual

            # Stack and normalize by total weight
            batch_grd_ee_en_ke = jnp.stack(batch_grd_ee_en_ke, axis=0)  # (n_ops, n_batch, n_nuc, 3)
            batch_grd_logpsi = jnp.stack(batch_grd_logpsi, axis=0)
            batch_grd_ke = jnp.stack(batch_grd_ke, axis=0)
            batch_weights = jnp.stack(batch_weights, axis=0)  # (n_ops, n_batch)

            # Normalize by sum of weights
            total_weight = batch_weights.sum(axis=0, keepdims=True)  # (1, n_batch)
            safe_weight = jnp.where(total_weight > 1e-10, total_weight, 1.0)

            avg_grd_ee_en_ke = batch_grd_ee_en_ke.sum(axis=0) / safe_weight[:, :, None, None]
            avg_grd_logpsi = batch_grd_logpsi.sum(axis=0) / safe_weight[:, :, None, None]
            avg_grd_ke = batch_grd_ke.sum(axis=0) / safe_weight[:, :, None, None]

            w_grd_ee_en_ke.append(avg_grd_ee_en_ke[0])
            w_grd_logpsi.append(avg_grd_logpsi[0])
            w_grd_ke.append(avg_grd_ke[0])
            w_weights.append(total_weight[0])

        # Stack all batches
        w_grd_ee_en_ke = jnp.vstack(w_grd_ee_en_ke)
        w_grd_logpsi = jnp.vstack(w_grd_logpsi)
        w_grd_ke = jnp.vstack(w_grd_ke)
        w_weights = jnp.concatenate(w_weights)

        # Print diagnostics on first iteration
        if iter == 1:
            print("\n" + "="*60)
            print("Reflection Operation Statistics (Iteration 1)")
            print("="*60)
            print(f"{'Operation':<12} {'Avg Weight':<15} {'% Contribution':<15}")
            print("-"*60)

            total_samples = sum(op_stats[op]['n_samples'] for op in reflection_op_list)
            for op in reflection_op_list:
                avg_w = op_stats[op]['total_weight'] / op_stats[op]['n_samples']
                # Contribution = weight / number of operations
                contrib_pct = (avg_w / len(reflection_op_list)) * 100
                print(f"{op:<12} {avg_w:<15.4f} {contrib_pct:<15.1f}%")

            '''
            print("\nInterpretation:")
            weights = [op_stats[op]['total_weight'] / op_stats[op]['n_samples']
                      for op in reflection_op_list]
            max_w, min_w = max(weights), min(weights)

            if max_w - min_w < 0.1:
                print("  • All operations have similar weights (~1.0)")
                print("  • Consider reducing to fewer operations (e.g., ['I', 'y'])")
                print("  • This would speed up calculation with minimal accuracy loss")
            else:
                print("  • Operations have varying acceptance rates")
                print("  • All operations are contributing useful information")
                print("  • Keep all four reflection operations")
            '''
            print("="*60 + "\n")

        # Save to HDF5
        with h5py.File(chkfile_grd, h5py_io) as f:
            f.create_dataset(f'grd_ee_en_ke_{iter}', data=w_grd_ee_en_ke)
            f.create_dataset(f'grd_logpsi_{iter}', data=w_grd_logpsi)
            f.create_dataset(f'local_energies_{iter}', data=local_energies)
            f.create_dataset(f'grd_ke_{iter}', data=w_grd_ke)
            f.create_dataset(f'importance_weights_{iter}', data=w_weights)


    return vmc_gradient_save_importance_reweighting
