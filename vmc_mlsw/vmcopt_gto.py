import jax
import jax.numpy as jnp
import optax
# from functools import partial
# import h5py
from vmc_mlsw.psi_gto import get_psi_fun


def get_vmcopt_func(mf,
                    params_vmc,
                    cgto_coeff=None):

    nuc_crds = jnp.array(mf.mol.atom_coords(unit='Bohr'))
    nelec = mf.mol.tot_electrons()
    # num_nuc = nuc_crds.shape[0]
    Z_charges = mf.mol.atom_charges()
    i_e, j_e = jnp.triu_indices(nelec, k=1)
    # atomic_masses = mf.mol.atom_mass_list()
    # mass_center = jnp.einsum('i,ij->j',
    #                          atomic_masses, nuc_crds)/atomic_masses.sum()
    # relative_nuc_pos = nuc_crds - mass_center

    log_trial_wavefunction, local_energy, get_psi_mo \
        = get_psi_fun(mf, cgto_coeff)

    local_energy_ee, local_energy_nn, local_energy_en, local_energy_ke \
        = local_energy
    ener_nn = local_energy_nn(nuc_crds)
    # grad_nn_nuc = jax.grad(local_energy_nn)(nuc_crds)

    @jax.jit
    def metropolis_move(rng_key, elec_crds, _step_size, curr_params):
        """Metropolis step."""
        key_prop, key_accept = jax.random.split(rng_key)

        proposed_crds = elec_crds + \
            _step_size * jax.random.normal(key_prop, elec_crds.shape)

        min_dist_threshold = 1e-4
        diffs_ee = proposed_crds[i_e] - proposed_crds[j_e]
        dists_ee = jnp.sqrt(jnp.sum(diffs_ee*diffs_ee, axis=-1))

        diffs_en = proposed_crds[:, None, :] - nuc_crds[None, :, :]
        dists_en = jnp.sqrt(jnp.sum(diffs_en*diffs_en, axis=-1))

        valid_move = (dists_en.min() > min_dist_threshold) & \
            (dists_ee.min() > min_dist_threshold)

        log_psi_old = log_trial_wavefunction(elec_crds, nuc_crds,
                                             curr_params)
        log_psi_new = log_trial_wavefunction(proposed_crds, nuc_crds,
                                             curr_params)

        accept = (jax.random.uniform(key_accept)
                  < jnp.exp(2 * (log_psi_new - log_psi_old))) & valid_move
        new_crds = jnp.where(accept, proposed_crds, elec_crds)

        return new_crds, accept

    @jax.jit
    def total_local_energy_fn(elec_crds, curr_params):
        return (local_energy_ee(elec_crds)
                + local_energy_en(elec_crds, nuc_crds)
                + local_energy_ke(elec_crds, nuc_crds, curr_params)
                + ener_nn)

    def vmcopt_run(rng_key,
                   nwalkers=100,
                   params_init=None,
                   num_steps=50000,
                   num_substeps=10,
                   num_epochs=20,
                   num_equilibration=5000,
                   step_size=0.25,
                   lr=0.02,
                   optimizer="sgd",
                   verbose=False):
        """VMC optimizatino run"""

        params = params_vmc if params_init is None \
            else jnp.array(params_init, dtype=jnp.float64)

        optimizer_chosen = optax.adam(learning_rate=lr) \
            if "adam" in optimizer.lower() \
            else optax.sgd(learning_rate=lr)
        opt_state = optimizer_chosen.init(params)

        # Initialize electron positions more efficiently
        rng_key, rng = jax.random.split(rng_key)
        idx_cnt = []
        for ia, iz in enumerate(Z_charges):
            idx_cnt.extend([ia] * iz)

        # Handle molecular charge
        if mf.mol.charge < 0:
            idx_cnt.extend([0] * abs(mf.mol.charge))
        elif mf.mol.charge > 0:
            idx_cnt = idx_cnt[:-mf.mol.charge]

        idx_cnt = jnp.array(idx_cnt)
        centers = nuc_crds[idx_cnt]
        walkers = centers[jnp.newaxis, :, :] \
            + 0.05 * jax.random.normal(rng, (nwalkers, nelec, 3))

        @jax.jit
        def equilibration_step(carried_in, _):
            rkey, w, s, curr_params = carried_in
            rkey0, rkey1 = jax.random.split(rkey)

            for _ in range(num_substeps):
                keys = jax.random.split(rkey1, nwalkers + 1)
                rkey1 = keys[0]
                keys = keys[1:]

                new_w, accepted \
                    = jax.vmap(metropolis_move,
                               in_axes=(0, 0, None, None))(keys, w, s,
                                                           curr_params)
                r = accepted.mean()
                new_s = s * (0.6 + r)

            return (rkey0, new_w, new_s, curr_params), r

        carry_in = (rng_key, walkers, step_size, params)
        carry_out, acc_ratios \
            = jax.lax.scan(equilibration_step, carry_in,
                           jnp.arange(num_equilibration))
        rng_key, walkers, step_size, _ = carry_out
        ratio = acc_ratios[-1]

        print(f"Equilibration Acceptance Rate: {ratio:.2f}")
        print(f"Step size: {step_size:.4f}")

        @jax.jit
        def production_step(carried_in, _):
            rkey, w, s, curr_params = carried_in
            rkey0, rkey1 = jax.random.split(rkey)

            for _ in range(num_substeps):
                keys = jax.random.split(rkey1, nwalkers + 1)
                rkey1 = keys[0]
                keys = keys[1:]

                new_w, accepted \
                    = jax.vmap(metropolis_move,
                               in_axes=(0, 0, None, None))(keys, w, s,
                                                           curr_params)
                r = accepted.mean()

                new_s = step_size * (0.6 + r)

            # calculate energy
            energies = jax.vmap(total_local_energy_fn,
                                in_axes=(0, None))(new_w,
                                                   curr_params)

            return (rkey0, new_w, new_s, curr_params), (r, energies)

        @jax.jit
        def grad_logpsi_vparams(elec_crds, _params):
            def _logpsi(p):
                return log_trial_wavefunction(elec_crds, nuc_crds, p)
            return jax.grad(_logpsi)(_params)

        @jax.jit
        def production_step_vpgrad(carried_in, _):
            rkey, w, s, curr_params = carried_in
            rkey0, rkey1 = jax.random.split(rkey)

            for _ in range(num_substeps):
                keys = jax.random.split(rkey1, nwalkers + 1)
                rkey1 = keys[0]
                keys = keys[1:]

                new_w, accepted \
                    = jax.vmap(metropolis_move,
                               in_axes=(0, 0, None, None))(keys, w, s,
                                                           curr_params)
                r = accepted.mean()

                new_s = step_size * (0.6 + r)

            # calculate energy
            energies = jax.vmap(total_local_energy_fn,
                                in_axes=(0, None))(new_w, curr_params)
            dlogpsi = jax.vmap(grad_logpsi_vparams,
                               in_axes=(0, None))(new_w, curr_params)
            E_mean = jnp.mean(energies)
            diff = energies - E_mean
            vp_grads = 2.0 * jnp.mean(diff[:, None] * dlogpsi, axis=0)

            return (rkey0, new_w, new_s, curr_params), (r, energies, vp_grads)

        carry_in = (rng_key, walkers, step_size, params)
        carry_out, results \
            = jax.lax.scan(production_step, carry_in,
                           jnp.arange(num_steps))
        rng_key, walkers, step_size, _ = carry_out
        acc_ratios, tw_energies = results
        # tw_energies.shape == num_steps, nwalkers

        def _acf_fft_jnp(x: jnp.ndarray) -> jnp.ndarray:
            """Normalized autocorrelation function via FFT (numpy)."""
            n = x.shape[0]
            if n == 0:
                return jnp.array([1.0], dtype=jnp.float64)
            x = x - x.mean()
            if n == 1:
                return jnp.array([1.0], dtype=jnp.float64)
            # Next power-of-two size for zero-padded convolution
            nfft = 1 << ((2*n - 1).bit_length())
            f = jnp.fft.rfft(x, nfft)
            acf = jnp.fft.irfft(f * jnp.conjugate(f), nfft)[:n]
            acf = jnp.real(acf)
            if acf.at[0] == 0.0:
                return jnp.ones(n, dtype=jnp.float64)
            return acf / acf[0]

        def _tau_int_from_acf(acf: jnp.ndarray) -> float:
            """Integrated autocorrelation time
            using initial positive sequence."""
            if acf.size <= 1:
                return 1.0
            rho = acf[1:]
            if rho.size < 2:
                return 1.0

            rho_pairs = rho[:(rho.size // 2) * 2].reshape(-1, 2)
            pair_sums = jnp.sum(rho_pairs, axis=1)
            positive_pairs_mask = pair_sums > 0.0

            initial_positive_run = jnp.cumprod(positive_pairs_mask)
            m_val = 2 * jnp.sum(initial_positive_run).astype(jnp.int32)

            indices = jnp.arange(rho.size)
            mask = indices < m_val
            rho_sum = jnp.sum(jnp.where(mask, rho, 0.0))

            tau_int = 1.0 + 2.0 * rho_sum
            return jnp.maximum(tau_int, 1.0)

        enr_mean = tw_energies.mean()
        # enr_wstd = tw_energies.mean(axis=1).std()

        def update_epoch(carried_in, epoch):
            # nonlocal tw_energies
            rng_key, walkers, step_size, params, opt_state = carried_in

            carry_in = (rng_key, walkers, step_size, params)
            carried_out, _ \
                = jax.lax.scan(equilibration_step, carry_in,
                               jnp.arange(num_equilibration))
            rng_key, walkers, step_size, _ = carried_out

            carry_in = (rng_key, walkers, step_size, params)
            carried_out, results \
                = jax.lax.scan(production_step_vpgrad, carry_in,
                               jnp.arange(num_steps))
            rng_key, walkers, step_size, _ = carried_out
            _, tw_energies, vparam_grads = results
            # tw_energies.shape == num_steps, nwalkers

            enr_mean = tw_energies.mean()
            grad_mean = vparam_grads.mean(axis=0)
            updates, opt_state = optimizer_chosen.update(grad_mean,
                                                         opt_state, params)
            params = optax.apply_updates(params, updates)

            jax.debug.print("[Epoch {epoch}/{ne}] "
                            "<E_loc>: {e:.6f}, params: {vp}",
                            epoch=epoch+1, ne=num_epochs,
                            e=enr_mean, vp=params)

            carry_out = (rng_key, walkers, step_size, params, opt_state)
            return carry_out, tw_energies

        carry_in_outer = (rng_key, walkers, step_size, params, opt_state)
        carried_out_outer, energies_opthist \
            = jax.lax.scan(update_epoch, carry_in_outer,
                           jnp.arange(num_epochs))
        energies_opthist.block_until_ready()

        rng_key, walkers, step_size, params, opt_state = carried_out_outer

        tw_energies = energies_opthist[-1]
        enr_mean = tw_energies.mean()
        w_std = jnp.std(tw_energies, axis=0)
        acf_axis1 = jax.jit(jax.vmap(_acf_fft_jnp, in_axes=1))
        w_acf = acf_axis1(tw_energies)
        tau_int_axis0 = jax.jit(jax.vmap(_tau_int_from_acf, in_axes=0))
        w_tau_int = tau_int_axis0(w_acf)
        w_ste = w_std * jnp.sqrt(w_tau_int / num_steps)
        enr_ste = jnp.sqrt(jnp.square(w_ste).sum()) / nwalkers

        return params, {'energy': {'mean': enr_mean, 'stderr': enr_ste}}
        # params, {'energy': jnp.array(energy_opthist)}

    return vmcopt_run
