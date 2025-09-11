from threading import local
import jax
import jax.numpy as jnp
# from functools import partial
import h5py
from vmc_mlsw.psi_gto import get_psi_fun
from vmc_mlsw.symm.water_rotation_matrix import symmetrize_water_molecule

@jax.jit
def apply_reflection_I(coords):
    return coords

@jax.jit
def apply_reflection_x(coords):
    """Apply reflection across yz-plane (- x-coordinate)."""
    return coords.at[..., 0].multiply(-1)

@jax.jit
def apply_reflection_y(coords):
    """Apply reflection across xz-plane (- y-coordinate)."""
    return coords.at[..., 1].multiply(-1)

@jax.jit
def apply_reflection_xy(coords):
    """Apply reflection across yz-plane and xz-plane (- x,y-coordinate)."""
    coords = coords.at[..., 0].multiply(-1)
    coords = coords.at[..., 1].multiply(-1)
    return coords

reflection_map = {
    'I' : apply_reflection_I,
    'x' : apply_reflection_x,
    'y' : apply_reflection_y,
    'xy' : apply_reflection_xy
}


def get_vmc_func(mf,
                 params_vmc,
                 scheme='scheme1',
                 chkfile_grd='vmc_grd_chk.hdf5',
                 cgto_coeff=None,
                 reflection_op_list=None):

    if reflection_op_list is None:
        reflection_op_list = ['I']

    nuc_crds = jnp.array(mf.mol.atom_coords(unit='Bohr'))
    nelec = mf.mol.tot_electrons()
    num_nuc = nuc_crds.shape[0]
    Z_charges = mf.mol.atom_charges()
    i_e, j_e = jnp.triu_indices(nelec, k=1)

    # In case of a Water Molecule
    l_water = False
    rot_mat = jnp.eye (3)
    if tuple (Z_charges) == (8, 1, 1): # water molecule
        # The oxygen atom is placed on the origin.
        l_water = True
        # place the water structure on the yz-plane
        # R(H2) -  R(H1) : y-axis
        # vec (0.5*(R(H1)+R(H2) - R(O)): z-axis

        nuc_crds -= nuc_crds[0]
        nuc_crds_sym, rot_mat = symmetrize_water_molecule(nuc_crds)
        nuc_crds = jnp.einsum('...i,ij->...j', nuc_crds, rot_mat)
        nuc_crds_sym = jnp.einsum('...i,ij->...j', nuc_crds_sym, rot_mat)
        print ('translated and rotated water\n', nuc_crds)
        print ('water_sym\n', nuc_crds_sym)
        print ('water_rot_mat', rot_mat)

        nuc_crds_op = {}
        nuc_crds_sym_op = {}
        for reflection_op in reflection_op_list:
            nuc_crds_op[reflection_op] = reflection_map[reflection_op](nuc_crds)
            nuc_crds_sym_op[reflection_op] = reflection_map[reflection_op](nuc_crds_sym)

    # atomic_masses = mf.mol.atom_mass_list()
    # mass_center = jnp.einsum('i,ij->j',
    #                          atomic_masses, nuc_crds)/atomic_masses.sum()
    # relative_nuc_pos = nuc_crds - mass_center

    log_trial_wavefunction, local_energy, get_psi_mo \
        = get_psi_fun(mf, cgto_coeff)

    local_energy_ee, local_energy_nn, local_energy_en, local_energy_ke \
        = local_energy
    ener_nn = local_energy_nn(nuc_crds)
    grad_nn_nuc = jax.grad(local_energy_nn)(nuc_crds)

    # --- Redistribute Gradient Samples ---
    @jax.jit
    def redistribute_samples_scheme1(elec_crds):
        mo_val, mo_val_s = get_psi_mo(elec_crds, nuc_crds)
        weight = jnp.einsum('neo,neo->en', mo_val_s, mo_val_s) #**(1.0/4)
        return weight/jnp.sum(weight, axis=-1, keepdims=True)


    @jax.jit
    def redistribute_samples_scheme2(elec_crds):
        diff = elec_crds[:, None, :] - nuc_crds[None, :, :]
        dist = jnp.sqrt(jnp.sum(diff**2, axis=-1))
        dist = jnp.where (dist < 1e-12, 1e-12, dist)
        weight = dist**(-4.0)
        return weight/jnp.sum(weight, axis=-1, keepdims=True)

    if scheme in ['scheme1']:
        rescale_fn = redistribute_samples_scheme1
    elif scheme in ['scheme2']:
        rescale_fn = redistribute_samples_scheme2
    else:
        # default redistribute scheme
        rescale_fn = redistribute_samples_scheme1

    jac_rescale_fn = jax.jacobian(rescale_fn, argnums=0)


    @jax.jit
    def grad_fn_ee (e_pos):
        return jax.grad (local_energy_ee) (e_pos)

    @jax.jit
    def grad_fn_en(e_pos):
        return jax.grad(local_energy_en, argnums=(0, 1))(e_pos, nuc_crds)

    @jax.jit
    def grad_fn_ke(e_pos):
        return jax.grad(local_energy_ke, argnums=(0, 1))(e_pos, nuc_crds, params_vmc)

    @jax.jit
    def grad_fn_logpsi(e_pos):
        return jax.grad(log_trial_wavefunction, argnums=(0, 1))(e_pos, nuc_crds, params_vmc)


    @jax.jit
    def total_local_energy_fn (elec_crds):
        return (local_energy_ee(elec_crds)
                + local_energy_en(elec_crds, nuc_crds)
                + local_energy_ke(elec_crds, nuc_crds, params_vmc)
                + ener_nn)


    @jax.jit
    def metropolis_step (rng_key,
                         elec_crds,
                         _step_size):
        """Metropolis step."""
        key_prop, key_accept = jax.random.split(rng_key)

        # More efficient proposal generation
        noise = jax.random.normal(key_prop, elec_crds.shape)
        proposed_crds = elec_crds + _step_size * noise

        # check the sigularity between electron-electron // electron-nuclei
        min_dist_threshold = 0.0001
        diffs_ee = proposed_crds[i_e] - proposed_crds[j_e]
        dists_ee = jnp.sqrt (jnp.sum(diffs_ee*diffs_ee, axis=-1))

        diffs_en = proposed_crds[:,None,:] - nuc_crds[None, :, :]
        dists_en = jnp.sqrt (jnp.sum(diffs_en*diffs_en, axis=-1))

        valid_move = (dists_en.min() > min_dist_threshold) & \
                (dists_ee.min() > min_dist_threshold)


        # Vectorized acceptance calculation
        log_psi_old = log_trial_wavefunction(elec_crds, nuc_crds, params_vmc)
        log_psi_new = log_trial_wavefunction(proposed_crds, nuc_crds, params_vmc)

        acceptance_ratio = jnp.exp(2 * (log_psi_new - log_psi_old))
        accept_prob = jnp.minimum(1.0, acceptance_ratio)

        accept = (jax.random.uniform(key_accept) < accept_prob) & valid_move
        new_crds = jnp.where(accept, proposed_crds, elec_crds)

        return new_crds, accept

    def vmc_gradient_batch (batch_samples):

        grd_ee_elc = jax.vmap (grad_fn_ee) (batch_samples)
        grd_en_elc, grd_en_nuc = jax.vmap(grad_fn_en) (batch_samples)
        grd_ke_elc, grd_ke_nuc = jax.vmap(grad_fn_ke) (batch_samples)
        grd_logpsi_elc, grd_logpsi_nuc = jax.vmap(grad_fn_logpsi) (batch_samples)

        rescale = jax.vmap (rescale_fn) (batch_samples)
        jac_rescale_elc = jax.vmap(jac_rescale_fn) (batch_samples)
        novel_correction = 0.5*jnp.einsum('beneK->bnK', jac_rescale_elc)

        grd_ee = jnp.einsum('beK,ben->bnK', grd_ee_elc, rescale)
        grd_en = grd_en_nuc + jnp.einsum('beK,ben->bnK', grd_en_elc, rescale)
        grd_ke = grd_ke_nuc + jnp.einsum('beK,ben->bnK', grd_ke_elc, rescale)

        grd_logpsi = grd_logpsi_nuc + jnp.einsum('beK,ben->bnK',
                                                grd_logpsi_elc, rescale)
        grd_logpsi += novel_correction

        return grd_ee, grd_en, grd_ke, grd_logpsi

    def vmc_gradient_save (iter,
                           sampled_walkers,
                           local_energies,
                           batch_size,
                           num_batches,
                           h5py_io='w'):

        n_samples = sampled_walkers.shape[0]
        w_grd_ee_en_ke = []
        w_grd_logpsi = []
        if l_water:
            for batch_idx in range (num_batches):
                start_idx = batch_idx * batch_size
                end_idx = min (start_idx + batch_size, n_samples)

                #1) electrons on original water (which is translated and rotated)
                #   --> electrons on symmetrized water
                batch_samples_ref = sampled_walkers[start_idx:end_idx]
                weights = jax.vmap(rescale_fn) (batch_samples_ref)
                batch_samples_sym = batch_samples_ref + \
                        jnp.einsum('nk,sen->sek',
                                   nuc_crds_sym-nuc_crds,
                                   weights)

                grd_ee_en_ke = []
                grd_logpsi = []
                for reflection_op in reflection_op_list:
                    # 2) reflection of electrons based on symmetrized water
                    # 3) electrons on symmetrized water --> electrons on original water
                    batch_samples = reflection_map[reflection_op](batch_samples_sym) - \
                            jnp.einsum('nk,sen->sek',
                                       nuc_crds_sym_op[reflection_op]-nuc_crds_op[reflection_op],
                                       weights)
                    # 4) calculate gradients acting on original water and save them
                    g_ee, g_en, g_ke, g_logpsi = \
                        vmc_gradient_batch (batch_samples)
                    grd_ee_en_ke.append (g_ee+g_en+g_ke)
                    grd_logpsi.append (g_logpsi)

                grd_ee_en_ke = jnp.stack (grd_ee_en_ke, axis=0).mean(axis=0)
                grd_logpsi = jnp.stack(grd_logpsi, axis=0).mean(axis=0)

                w_grd_ee_en_ke.append (grd_ee_en_ke)
                w_grd_logpsi.append (grd_logpsi)

        else:

            for batch_idx in range (num_batches):
                start_idx = batch_idx * batch_size
                end_idx = min (start_idx + batch_size, n_samples)

                grd_ee_en_ke = []
                grd_logpsi = []
                for reflection_op in reflection_op_list:
                    batch_samples = reflection_map[reflection_op](sampled_walkers [start_idx:end_idx])
                    g_ee, g_en, g_ke, g_logpsi = \
                        vmc_gradient_batch (batch_samples)
                    grd_ee_en_ke.append (g_ee+g_en+g_ke)
                    grd_logpsi.append (g_logpsi)
                grd_ee_en_ke = jnp.stack (grd_ee_en_ke, axis=0).mean(axis=0)
                grd_logpsi = jnp.stack(grd_logpsi, axis=0).mean(axis=0)

                w_grd_ee_en_ke.append (grd_ee_en_ke)
                w_grd_logpsi.append (grd_logpsi)

        # (n_samples, 3)
        w_grd_ee_en_ke = jnp.vstack(w_grd_ee_en_ke)
        w_grd_logpsi = jnp.vstack (w_grd_logpsi)

        with h5py.File(chkfile_grd, h5py_io) as f:
            f.create_dataset(f'grd_ee_en_ek_{iter}', data=w_grd_ee_en_ke)
            f.create_dataset(f'grd_logpsi_{iter}', data=w_grd_logpsi)
            f.create_dataset(f'local_energies_{iter}', data=local_energies)

    def vmc_run(rng_key,
                nwalkers=100,
                num_mc_steps=1000,
                max_mc_iter=500,
                mc_step_size=0.25,
                tolerance_enr_std=0.01,
                fname_log='vmc_enr.log',
                l_grad=False):
        """VMC run with better memory management."""

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
        walkers = centers[jnp.newaxis,:,:] \
            + 0.05*jax.random.normal(rng, (nwalkers, nelec, 3))

        # Equilibration phase
        @jax.jit
        def equilibration_step (state, _):
            rng_key, walkers, step_size = state
            rng_key, key = jax.random.split(rng_key)
            keys = jax.random.split(key, nwalkers)
            new_walkers, accepted = jax.vmap(metropolis_step, in_axes=(0,0,None)) (
                                        keys,
                                        walkers,
                                        step_size)
            ratio = accepted.mean()
            new_step_size = step_size * (0.5 + ratio)

            return (rng_key, new_walkers, new_step_size), ratio

        initial_state = (rng_key, walkers, mc_step_size)
        final_state, ratios = jax.lax.scan(equilibration_step,
                                      initial_state,
                                      jnp.arange(num_mc_steps))
        rng_key, walkers, mc_step_size = final_state
        ratio = ratios[-1]

        print(f"Equilibration Acceptance Rate: {ratio:.2f}")
        print(f"Step size: {mc_step_size:.4f}")

        # Production phase
        @jax.jit
        def production_step (state, step_number):
            rng_key, walkers, step_size, _ = state
            rng_key, key = jax.random.split(rng_key)
            keys = jax.random.split(key, nwalkers)

            new_walkers, accepted = jax.vmap(metropolis_step, in_axes=(0,0,None)) (
                                        keys,
                                        walkers,
                                        step_size)

            ratio = accepted.mean()

            new_step_size = step_size * (0.5 + ratio)
            # calculate energy
            energies = jax.vmap (total_local_energy_fn) (new_walkers)

            return (rng_key, new_walkers, new_step_size, ratio), \
                (energies, new_walkers)

        # Main sampling phase with pre-allocated arrays
        base_batch_size = 500
        memory_factor = max(1, nelec * num_nuc // 1000)
        batch_size = max(50, base_batch_size // memory_factor)
        num_batches = (num_mc_steps*nwalkers//10 + batch_size - 1) // batch_size
        mark_samples = ((jnp.arange (num_mc_steps)+1)%10 == 0)

        ratio = 0.5

        initial_state = (rng_key, walkers, mc_step_size, ratio)
        final_state, samples = jax.lax.scan (production_step,
                                       initial_state,
                                       jnp.arange (num_mc_steps))

        rng_key, walkers, mc_step_size, ratio = final_state
        walkers_energies, sampled_walkers = samples

        enr_mean = walkers_energies.mean(axis=0).mean()
        enr_std = walkers_energies.mean(axis=0).std()
        iter = 1
        fout = open(fname_log, 'w', 1)

        print ('iter,enr,std: '
                    f'{iter:5d}  '
                    f'{enr_mean:.6f}  '
                    f'{enr_std:.6f}',
                    file=fout)

        if l_grad:
            sampled_walkers = sampled_walkers[mark_samples].reshape(-1, nelec, 3)
            local_energies = walkers_energies[mark_samples].reshape(-1)
            vmc_gradient_save (iter,
                           sampled_walkers,
                           local_energies,
                           batch_size,
                           num_batches,
                           h5py_io='w')

        while (iter < max_mc_iter) & (enr_std > tolerance_enr_std):
            initial_state = (rng_key, walkers, mc_step_size, ratio)
            final_state, samples = jax.lax.scan (production_step,
                                       initial_state,
                                       jnp.arange (num_mc_steps))

            rng_key, walkers, mc_step_size, ratio = final_state
            energies, sampled_walkers = samples
            iter += 1

            walkers_energies = jnp.append (walkers_energies, energies, axis=0)

            enr_mean = walkers_energies.mean(axis=0).mean()
            enr_std = walkers_energies.mean(axis=0).std()

            print ('iter,enr,std: '
                    f'{iter:5d}  '
                    f'{enr_mean:.6f}  '
                    f'{enr_std:.6f}',
                    file=fout)

            if l_grad:
                sampled_walkers = sampled_walkers[mark_samples].reshape(-1, nelec, 3)
                local_energies = energies[mark_samples].reshape(-1)

                vmc_gradient_save (iter,
                               sampled_walkers,
                               local_energies,
                               batch_size,
                               num_batches,
                               h5py_io='a')

        fout.close()

        if l_grad:
            sampled_iter = iter

            with h5py.File (chkfile_grd, 'a') as f:
                f.create_dataset ('enr_mean', data=enr_mean)
                f.create_dataset ('enr_std', data=enr_std)
                f.create_dataset ('sampled_iter', data=sampled_iter, dtype=jnp.int32)
                f.create_dataset ('grd_nn', data=grad_nn_nuc)


    def vmc_gradient_with_space_warping ():

        with h5py.File(chkfile_grd, 'r') as f:
            dict_grd_samples = {}
            for key, data in f.items():
                if key in ['enr_mean', 'enr_std', 'sampled_iter']:
                    dict_grd_samples[key] = data[()]
                else:
                    dict_grd_samples[key] = jnp.array(data[:])

            sampled_iter = int (dict_grd_samples['sampled_iter'])
            enr_mean = dict_grd_samples['enr_mean']
            enr_std = dict_grd_samples['enr_std']
            grd_nn = dict_grd_samples['grd_nn']

            grd_ee_en_ke_sum = 0.0
            grd_pulay_sum = 0.0
            valid_samples_count = 0
            for iter in range(sampled_iter):
                grd_ee_en_ke = dict_grd_samples[f'grd_ee_en_ke_{iter+1}']
                grd_logpsi = dict_grd_samples[f'grd_logpsi_{iter+1}']
                local_energies = dict_grd_samples[f'local_energies_{iter+1}']

                d_enr = local_energies - enr_mean
                grd_pulay = 2.0*jnp.einsum('s,snK->snK',
                                        d_enr,
                                        grd_logpsi)


                grd_ee_en_ke_sum += grd_ee_en_ke.sum(axis=0)
                grd_pulay_sum += grd_pulay.sum(axis=0)
                valid_samples_count += local_energies.shape[0]

            if valid_samples_count > 0:
                grd_ee_en_ke = grd_ee_en_ke_sum / valid_samples_count
                grd_pulay = grd_pulay_sum / valid_samples_count
            else:
                grd_ee_en_ke = jnp.zeros_like (grd_nn)
                grd_pulay = jnp.zeros_like (grd_nn)

            if l_water:
                # rotate the gradients
                grd_nn = jnp.einsum('...j,ij->...i', grd_nn, rot_mat)
                grd_ee_en_ke = jnp.einsum('...j,ij->...i', grd_ee_en_ke, rot_mat)
                grd_pulay = jnp.einsum('...j,ij->...i', grd_pulay, rot_mat)

            grd_tot = grd_nn + grd_ee_en_ke + grd_pulay
            with jnp.printoptions (precision=5, suppress=True):
                print('grd_nn\n', grd_nn)
                print('grd_ee_en_ke\n', grd_ee_en_ke)
                print('grd_pulay\n', grd_pulay)
                print('grd_tot\n', grd_tot)

            return grd_tot

    return vmc_run, vmc_gradient_with_space_warping
