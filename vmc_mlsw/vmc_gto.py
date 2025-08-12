import jax
import jax.numpy as jnp
# from functools import partial
import h5py
from vmc_mlsw.psi_gto import get_psi_fun
from vmc_mlsw.psi_gto_cusp import get_psi_cusp_fun


def get_vmc_func(mf,
                 params_vmc,
                 chkfile_mc='vmc_mc_chk.hdf5',
                 chkfile_enr='vmc_enr_chk.hdf5',
                 chkfile_grd='vmc_grd_chk.hdf5',
                 chkfile_elc='vmc_elc_chk.hdf5',
                 cgto_coeff=None):

    nuc_crds = jnp.array(mf.mol.atom_coords(unit='Bohr'))
    nelec = mf.mol.tot_electrons()
    Z_charges = mf.mol.atom_charges()
    i_e, j_e = jnp.triu_indices(nelec, k=1)
    # atomic_masses = mf.mol.atom_mass_list()
    # mass_center = jnp.einsum('i,ij->j',
    #                          atomic_masses, nuc_crds)/atomic_masses.sum()
    # relative_nuc_pos = nuc_crds - mass_center

    if cgto_coeff is None:
        log_trial_wavefunction, local_energy, get_psi_mo \
            = get_psi_fun(mf)
    else:
        log_trial_wavefunction, local_energy, get_psi_mo \
            = get_psi_cusp_fun(mf, cgto_coeff)

    local_energy_ee, local_energy_nn, local_energy_en, local_energy_ke \
        = local_energy
    ener_nn = local_energy_nn(nuc_crds)

    def vmc_run(rng_key,
                num_steps=5000,
                num_equilibration=1000,
                step_size=0.25):
        """Optimized VMC run with better memory management."""

        @jax.jit
        def metropolis_step(rng_key, elec_crds, _step_size):
            """Optimized Metropolis step."""
            rng_key, key_prop, key_accept \
                = jax.random.split(rng_key, 3)

            # More efficient proposal generation
            noise = jax.random.normal(key_prop, elec_crds.shape)
            proposed_crds = elec_crds + _step_size * noise

            # check the sigularity between electron-electron
            diffs_ee = proposed_crds[i_e] - proposed_crds[j_e]
            dists_ee = jnp.sqrt(jnp.sum(diffs_ee*diffs_ee, axis=-1))
            proposed_crds = jax.lax.select(dists_ee.min() < 0.001,
                                           elec_crds,
                                           proposed_crds)

            # check the singularity between electron and nuclei
            diffs_en = proposed_crds[:, None, :] - nuc_crds[None, :, :]
            dists_en = jnp.sqrt(jnp.sum(diffs_en*diffs_en, axis=-1))

            proposed_crds = jax.lax.select(dists_en.min() < 0.001,
                                           elec_crds,
                                           proposed_crds)

            # Vectorized acceptance calculation
            log_psi_old = log_trial_wavefunction(elec_crds, nuc_crds,
                                                 params_vmc)
            log_psi_new = log_trial_wavefunction(proposed_crds, nuc_crds,
                                                 params_vmc)

            acceptance_ratio = jnp.exp(2 * (log_psi_new - log_psi_old))
            accept_prob = jnp.minimum(1.0, acceptance_ratio)

            accept = jax.random.uniform(key_accept) < accept_prob
            new_crds = jnp.where(accept, proposed_crds, elec_crds)

            return rng_key, new_crds, accept

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
        elec_crds = centers + 0.05 * jax.random.normal(rng, (nelec, 3))

        # Equilibration phase
        @jax.jit
        def equilibration_step(state, step_number):
            rng_key, elec_crds, step_size, accepts, _ = state
            rng_key, elec_crds, accepted = metropolis_step(rng_key,
                                                           elec_crds,
                                                           step_size)
            accepts += accepted.sum()
            ratio = accepts/5000
            new_step_size = jax.lax.select(
                (step_number+1) % 5000 == 0,
                step_size * (0.5 + ratio),
                step_size
            )
            new_accepts = jax.lax.select(
                (step_number+1) % 5000 == 0,
                0,
                accepts
            )
            return (rng_key, elec_crds, new_step_size, new_accepts, ratio), \
                None

        initial_state = (rng_key, elec_crds, step_size, 0, 0.5)
        final_state, _ = jax.lax.scan(equilibration_step,
                                      initial_state,
                                      jnp.arange(num_equilibration))
        rng_key, elec_crds, step_size, _, ratio = final_state

        print(f"Equilibration Acceptance Rate: {ratio:.2f}")
        print(f"Step size: {step_size:.4f}")

        # Production phase
        @jax.jit
        def production_step(state, step_number):
            rng_key, elec_crds, step_size, accepts, _, samples, sample_idx \
                = state
            rng_key, elec_crds, accepted \
                = metropolis_step(rng_key, elec_crds, step_size)
            accepts += accepted.sum()
            ratio = accepts/50000
            new_step_size = jax.lax.select(
                (step_number+1) % 50000 == 0,
                step_size * (0.5 + ratio),
                step_size
            )
            new_accepts = jax.lax.select(
                (step_number+1) % 50000 == 0,
                0,
                accepts
            )

            is_collection = jnp.equal(jnp.mod(step_number+1, 10), 0)

            def write_samples(args):
                samples_loc, idx = args
                samples_loc = samples_loc.at[idx].set(elec_crds)
                idx = idx + 1
                return samples_loc, idx

            def no_write(args):
                return args

            samples, sample_idx = jax.lax.cond(is_collection,
                                               write_samples,
                                               no_write,
                                               operand=(samples, sample_idx))

            return (rng_key, elec_crds, new_step_size, new_accepts, ratio,
                    samples, sample_idx), None

        # Main sampling phase with pre-allocated arrays
        n_collect = num_steps//10
        samples = jnp.zeros((n_collect, elec_crds.shape[0], 3))
        sample_idx = 0
        for istep in range(num_steps//50000):
            initial_state = (rng_key, elec_crds, step_size, 0, 0.5,
                             samples, sample_idx)
            final_state, _ = jax.lax.scan(production_step,
                                          initial_state,
                                          jnp.arange(50000))
            rng_key, elec_crds, step_size, _, ratio, samples, sample_idx \
                = final_state

            print(f"MC {(istep+1)*50000}/{num_steps} "
                  f"({100*(istep+1)*50000/num_steps:.2f}%): "
                  f"Ratio {ratio:.2f} Step_size {step_size:.4f}")

        # More efficient stacking
        samples = samples[:sample_idx]

        # Save results
        with h5py.File(chkfile_mc, 'w') as f:
            f.create_dataset('stacked_samples', data=samples)
            f.create_dataset('nuc_crds', data=nuc_crds)

    def vmc_energy():
        """Optimized energy calculation with better batching."""

        # Load data
        with h5py.File(chkfile_mc, 'r') as f:
            stacked_samples = jnp.array(f['stacked_samples'][:])

        # Vectorized energy calculations
        print(f'NN energy: {ener_nn:.6f}')
        ener_ee_samples = jax.vmap(local_energy_ee)(stacked_samples)
        print(f'EE samples mean: {ener_ee_samples.mean():.6f}')
        ener_en_samples = jax.vmap(local_energy_en,
                                   in_axes=(0, None))(stacked_samples,
                                                      nuc_crds)
        print(f'EN samples mean: {ener_en_samples.mean():.6f}')

        # Optimized kinetic energy calculation with adaptive batching
        num_samples = stacked_samples.shape[0]
        batch_size = min(1000, num_samples)  # Adaptive batch size

        ener_ke_samples = jnp.zeros(num_samples)

        for i in range(0, num_samples, batch_size):
            end_idx = min(i + batch_size, num_samples)
            batch_samples = stacked_samples[i:end_idx]

            batch_ke = jax.vmap(local_energy_ke, in_axes=(0, None, None))(
                batch_samples, nuc_crds, params_vmc
            )
            ener_ke_samples = ener_ke_samples.at[i:end_idx].set(batch_ke)

        # Print results
        print(f'KE samples mean: {ener_ke_samples.mean():.6f}')

        total_electronic = ener_ee_samples + ener_en_samples + ener_ke_samples
        print(f'Total electronic energy: {total_electronic.mean():.6f}')
        print(f'Total energy [Ha]: {total_electronic.mean() + ener_nn:.6f}')

        # Save results
        with h5py.File(chkfile_enr, 'w') as f:
            f.create_dataset('ener_ee_samples', data=ener_ee_samples)
            f.create_dataset('ener_en_samples', data=ener_en_samples)
            f.create_dataset('ener_ke_samples', data=ener_ke_samples)

    def vmc_gradient_prep():
        """
        Highly optimized gradient preparation using JAX compilation,
        vectorization, and efficient memory management.
        """
        # Load data
        dict_elec_samples = {}
        with h5py.File(chkfile_elc, 'r') as f:
            for key, data in f.items():
                dict_elec_samples[key] = jnp.array(data[:])

        stacked_samples = dict_elec_samples['reflection_E']

        num_samples, num_electrons, num_nuc = (
            stacked_samples.shape[0],
            stacked_samples.shape[1],
            nuc_crds.shape[0]
        )

        print(f"Processing {num_samples} samples "
              f"with {num_electrons} electrons and {num_nuc} nuclei")

        # Pre-compile all gradient functions
        @jax.jit
        def grad_fn_ee(e_pos):
            return jax.grad(local_energy_ee)(e_pos)

        @jax.jit
        def grad_fn_en(e_pos):
            return jax.grad(local_energy_en, argnums=(0, 1))(e_pos, nuc_crds)

        @jax.jit
        def grad_fn_ke(e_pos):
            return jax.grad(local_energy_ke,
                            argnums=(0, 1))(e_pos, nuc_crds, params_vmc)

        @jax.jit
        def grad_fn_logpsi(e_pos):
            return jax.grad(log_trial_wavefunction,
                            argnums=(0, 1))(e_pos, nuc_crds, params_vmc)

        # @jax.jit
        def compute_gradient_batch(samples_batch):
            """JIT-compiled function for batch gradient computation."""
            # Compute gradients for the batch
            grad_ee = jax.vmap(grad_fn_ee)(samples_batch)
            grad_en_elc, grad_en_nuc = jax.vmap(grad_fn_en)(samples_batch)
            grad_ke_elc, grad_ke_nuc = jax.vmap(grad_fn_ke)(samples_batch)
            grad_logpsi_elc, grad_logpsi_nuc \
                = jax.vmap(grad_fn_logpsi)(samples_batch)

            return grad_ee, grad_en_elc, grad_en_nuc, \
                grad_ke_elc, grad_ke_nuc, grad_logpsi_elc, grad_logpsi_nuc

        # Compute nuclear-nuclear gradient (constant)
        grad_eloc_nn_nuc = jax.grad(local_energy_nn)(nuc_crds)
        with h5py.File(chkfile_grd, 'w') as f:
            f.create_dataset('grad_eloc_nn_nuc', data=grad_eloc_nn_nuc)

        # Adaptive batch size based on available memory and problem size
        base_batch_size = 500
        memory_factor = max(1, num_electrons * num_nuc // 1000)
        batch_size = max(50, base_batch_size // memory_factor)
        # Process samples in batches with progress tracking
        num_batches = (num_samples + batch_size - 1) // batch_size

        for key, elec_samples in dict_elec_samples.items():

            print(f"Using batch size: {key} {batch_size}")

            # elec_samples = stacked_samples

            # Pre-allocate output arrays
            grad_eloc_ee_elc = jnp.zeros((num_samples, num_electrons, 3))
            grad_eloc_en_elc = jnp.zeros((num_samples, num_electrons, 3))
            grad_eloc_en_nuc_base = jnp.zeros((num_samples, num_nuc, 3))
            grad_eloc_ke_elc = jnp.zeros((num_samples, num_electrons, 3))
            grad_eloc_ke_nuc_base = jnp.zeros((num_samples, num_nuc, 3))
            grad_logpsi_elc = jnp.zeros((num_samples, num_electrons, 3))
            grad_logpsi_nuc_base = jnp.zeros((num_samples, num_nuc, 3))

            for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, num_samples)

                if (batch_idx + 1) % max(1, num_batches // 5) == 0:
                    print(f"Processing batch {batch_idx + 1}/{num_batches} "
                          f"({100*(batch_idx+1)/num_batches:.1f}%)")

                batch_samples = elec_samples[start_idx:end_idx]

                # Compute all gradients for this batch
                # using JIT-compiled function
                (grad_ee_batch, grad_en_elc_batch, grad_en_nuc_batch,
                 grad_ke_elc_batch, grad_ke_nuc_batch,
                 grad_logpsi_elc_batch, grad_logpsi_nuc_batch) \
                    = compute_gradient_batch(batch_samples)

                # Store results
                grad_eloc_ee_elc = grad_eloc_ee_elc.at[start_idx:end_idx] \
                    .set(grad_ee_batch)
                grad_eloc_en_elc = grad_eloc_en_elc.at[start_idx:end_idx] \
                    .set(grad_en_elc_batch)
                grad_eloc_en_nuc_base = grad_eloc_en_nuc_base \
                    .at[start_idx:end_idx].set(grad_en_nuc_batch)
                grad_eloc_ke_elc = grad_eloc_ke_elc.at[start_idx:end_idx] \
                    .set(grad_ke_elc_batch)
                grad_eloc_ke_nuc_base = grad_eloc_ke_nuc_base \
                    .at[start_idx:end_idx].set(grad_ke_nuc_batch)
                grad_logpsi_elc = grad_logpsi_elc.at[start_idx:end_idx] \
                    .set(grad_logpsi_elc_batch)
                grad_logpsi_nuc_base = grad_logpsi_nuc_base \
                    .at[start_idx:end_idx].set(grad_logpsi_nuc_batch)

            print("Gradient computation completed. Saving results...")

            # Save results efficiently

            with h5py.File(chkfile_grd, 'a') as f:
                f.create_dataset('grad_eloc_ee_elc_'+key,
                                 data=grad_eloc_ee_elc)
                f.create_dataset('grad_eloc_en_elc_'+key,
                                 data=grad_eloc_en_elc)
                f.create_dataset('grad_eloc_en_nuc_base_'+key,
                                 data=grad_eloc_en_nuc_base)
                f.create_dataset('grad_eloc_ke_elc_'+key,
                                 data=grad_eloc_ke_elc)
                f.create_dataset('grad_eloc_ke_nuc_base_'+key,
                                 data=grad_eloc_ke_nuc_base)
                f.create_dataset('grad_logpsi_elc_'+key,
                                 data=grad_logpsi_elc)
                f.create_dataset('grad_logpsi_nuc_base_'+key,
                                 data=grad_logpsi_nuc_base)

        return

    @jax.jit
    def redistribute_samples_scheme1(elec_crds):
        mo_val, mo_val_s = get_psi_mo(elec_crds, nuc_crds)
        weight = jnp.einsum('neo,neo->en', mo_val_s, mo_val_s)  # **(1.0/4)
        return weight/jnp.sum(weight, axis=-1, keepdims=True)

    @jax.jit
    def redistribute_samples_scheme2(elec_crds):
        diff = elec_crds[:, None, :] - nuc_crds[None, :, :]
        dist = jnp.sqrt(jnp.sum(diff**2, axis=-1))
        dist = jnp.where(dist < 1e-12, 1e-12, dist)
        weight = dist**(-4.0)
        return weight/jnp.sum(weight, axis=-1, keepdims=True)

    def vmc_gradient_with_space_warping_and_symmetry(
            scheme='scheme1',
            mark_std=None):

        # Load data
        dict_elec_samples = {}
        with h5py.File(chkfile_elc, 'r') as f:
            for key, data in f.items():
                dict_elec_samples[key] = jnp.array(data[:])

        with h5py.File(chkfile_enr, 'r') as f:
            ener_ee_samples = jnp.array(f['ener_ee_samples'][:])
            ener_en_samples = jnp.array(f['ener_en_samples'][:])
            ener_ke_samples = jnp.array(f['ener_ke_samples'][:])

        local_energies = ener_nn + \
            ener_en_samples + \
            ener_ke_samples + \
            ener_ee_samples
        enr_mean = local_energies.mean()
        enr_std = local_energies.std()

        # --- Redistribute Gradient Samples ---
        if scheme in ['scheme1']:
            rescale_fn = redistribute_samples_scheme1
        elif scheme in ['scheme2']:
            rescale_fn = redistribute_samples_scheme2
        else:
            # default redistribute scheme
            rescale_fn = redistribute_samples_scheme1

        jac_rescale_fn = jax.jacobian(rescale_fn, argnums=0)
        #
        stacked_samples = dict_elec_samples['reflection_E']
        num_batches = 2000
        num_samples = stacked_samples.shape[0]
        num_elc = stacked_samples.shape[1]
        num_nuc = nuc_crds.shape[0]

        if mark_std is None:
            mark = jnp.ones((num_samples), dtype=int)
        else:
            enr_min = enr_mean - 3.0*enr_std
            enr_max = enr_mean + 3.0*enr_std
            mark = (local_energies > enr_min)*(local_energies < enr_max)

        d_enr = local_energies - local_energies.mean()
        f_h5 = h5py.File(chkfile_grd, 'r')
        # --- Nuclear Gradient ---
        grd_nn = jnp.array(f_h5['grad_eloc_nn_nuc'][:])
        grd_ee = []
        grd_en = []
        grd_ke = []
        grad_logpsi_nuc = []

        for key, elec_samples in dict_elec_samples.items():

            rescale = jnp.zeros((num_samples, num_elc, num_nuc))

            for ist in range(0, num_samples, num_batches):
                ied = min(ist+num_batches, num_samples)
                val = jax.vmap(rescale_fn)(
                        elec_samples[ist:ied]
                        )
                rescale = rescale.at[ist:ied].set(val)

            # --- Electron Gradient ---
            grad_eloc_ee_elc = jnp.array(f_h5['grad_eloc_ee_elc_'+key][:])
            grd_ee.append(jnp.einsum('seK,sen->snK',
                                     grad_eloc_ee_elc,
                                     rescale))

            # --- Electron-Nuclear Gradient ---
            grad_eloc_en_elc = jnp.array(f_h5['grad_eloc_en_elc_'+key][:])
            grad_eloc_en_nuc_base = jnp.array(f_h5['grad_eloc_en_nuc_base_'
                                                   + key][:])
            grd_en.append(grad_eloc_en_nuc_base +
                          jnp.einsum('seK,sen->snK',
                                     grad_eloc_en_elc,
                                     rescale))

            #    --- Electron-Kinetic Gradient ---
            grad_eloc_ke_elc = jnp.array(f_h5['grad_eloc_ke_elc_'+key][:])
            grad_eloc_ke_nuc_base = jnp.array(f_h5['grad_eloc_ke_nuc_base_'
                                                   + key][:])
            grd_ke.append(grad_eloc_ke_nuc_base +
                          jnp.einsum('seK,sen->snK',
                                     grad_eloc_ke_elc,
                                     rescale))

            # --- Electron-LogPsi Gradient ---
            grad_logpsi_elc = jnp.array(f_h5['grad_logpsi_elc_'+key][:])
            grad_logpsi_nuc_base = jnp.array(f_h5['grad_logpsi_nuc_base_'
                                                  + key][:])

            novel_correction = jnp.zeros((num_samples, num_nuc, 3))
            for ist in range(0, num_samples, num_batches):
                ied = min(ist+num_batches, num_samples)
                jac_rescale_elec = jax.vmap(jac_rescale_fn)(
                        elec_samples[ist:ied]
                        )
                val = 0.5*jnp.einsum('seneK->snK', jac_rescale_elec)
                novel_correction = novel_correction.at[ist:ied].set(val)

            grad_logpsi_nuc.append(grad_logpsi_nuc_base +
                                   jnp.einsum('seK,sen->snK',
                                              grad_logpsi_elc, rescale)
                                   + novel_correction)

        grd_ee = jnp.stack(grd_ee, axis=0)
        grd_ee = grd_ee.mean(axis=0)
        grd_en = jnp.stack(grd_en, axis=0)
        grd_en = grd_en.mean(axis=0)
        grd_ke = jnp.stack(grd_ke, axis=0)
        grd_ke = grd_ke.mean(axis=0)
        grad_logpsi_nuc = jnp.stack(grad_logpsi_nuc, axis=0)
        grad_logpsi_nuc = grad_logpsi_nuc.mean(axis=0)

        pulay_terms = 2.0 * jnp.einsum('s,snK->snK',
                                       d_enr,
                                       grad_logpsi_nuc)

        with jnp.printoptions(precision=5, suppress=True):
            print('grd_nn\n', grd_nn)
            print('grd_ee\n', grd_ee[mark].mean(axis=0))
            print('grd_en\n', grd_en[mark].mean(axis=0))
            print('grd_ke\n', grd_ke[mark].mean(axis=0))
            print('grd_pulay\n', pulay_terms[mark].mean(axis=0))
            print('grd_logpsi\n', grad_logpsi_nuc[mark].mean(axis=0))

        total_grad = grd_nn[None, ...] + \
            grd_ee + grd_en + grd_ke + pulay_terms
        '''
        loss_variance = jnp.mean(jnp.var(total_grad[mark], axis=0))
        torques_per_nucleus = jnp.cross(relative_nuc_pos,
                                        total_grad[mark])
        total_torque_per_sample = jnp.sum(torques_per_nucleus, axis=1)
        loss_torque = jnp.mean(jnp.sum(total_torque_per_sample**2, axis=-1))
        with jnp.printoptions(precision=5, suppress=True):
            print('loss_variance', loss_variance)
            print('loss_torque', loss_torque)
            print('loss', loss_variance+loss_torque)
            torque = jnp.cross(relative_nuc_pos,
                            total_grad[mark].mean(axis=0))
            print('torque', torque)
        '''
        return total_grad[mark].mean(axis=0)

    return vmc_run, vmc_energy, vmc_gradient_prep,  \
        vmc_gradient_with_space_warping_and_symmetry
