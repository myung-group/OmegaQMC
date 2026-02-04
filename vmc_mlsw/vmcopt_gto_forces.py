import jax
import jax.numpy as jnp
from functools import partial
import h5py
import optax
from vmc_mlsw.psi_gto import get_psi_fun
#from vmc_mlsw.psi_gto_cusp import get_psi_cusp_fun


def get_vmc_func(
    mf,
    params_vmc,
    scheme="scheme1",
    chkfile_grd="vmc_grd_chk.hdf5",
    cgto_coeff=None,
):
    """
    Force-only training of J1_params (J2 fixed).
    - Uses your existing VMC nuclear force estimator (Pulay + rescale + novel correction).
    - Target force: RHF(PySCF) nuclear force = - mf.nuc_grad_method().kernel()
    - Logs: energy mean/std, force loss, ||J1||, ||ΔJ1||, ||J2||, ||ΔJ2||, J2_changed? and energy change.
    """

    # -------------------------
    # Geometry / system constants
    # -------------------------
    nuc_crds = jnp.array(mf.mol.atom_coords(unit="Bohr"))
    nelec = mf.mol.tot_electrons()
    num_nuc = nuc_crds.shape[0]
    Z_charges = mf.mol.atom_charges()
    i_e, j_e = jnp.triu_indices(nelec, k=1)

    # -------------------------
    # Wavefunction / local energy callables
    # -------------------------

    log_trial_wavefunction, local_energy, get_psi_mo = get_psi_fun(mf, cgto_coeff)

    local_energy_ee, local_energy_nn, local_energy_en, local_energy_ke = local_energy
    ener_nn = local_energy_nn(nuc_crds)
    grad_nn_nuc = jax.grad(local_energy_nn)(nuc_crds)  # dE_nn/dR

    # -------------------------
    # Params: expect dict with keys J1_params, J2_params
    # -------------------------
    if not (isinstance(params_vmc, dict) and "J1_params" in params_vmc and "J2_params" in params_vmc):
        raise ValueError(
            "params_vmc must be a dict with keys {'J1_params','J2_params'}.\n"
            "Example: params_vmc={'J1_params': jnp.zeros((...)), 'J2_params': jnp.zeros((...))}"
        )

    # Ensure float64 for stability
    params_vmc = {
        "J1_params": jnp.array(params_vmc["J1_params"], dtype=jnp.float64),
        "J2_params": jnp.array(params_vmc["J2_params"], dtype=jnp.float64),
    }

    # Freeze reference for J2
    J2_fixed = params_vmc["J2_params"]

    # -------------------------
    # Redistribute Gradient Samples (your code)
    # -------------------------
    @jax.jit
    def redistribute_samples_scheme1(elec_crds):
        mo_val, mo_val_s = get_psi_mo(elec_crds, nuc_crds)
        weight = jnp.einsum("neo,neo->en", mo_val_s, mo_val_s)
        return weight / jnp.sum(weight, axis=-1, keepdims=True)

    @jax.jit
    def redistribute_samples_scheme2(elec_crds):
        diff = elec_crds[:, None, :] - nuc_crds[None, :, :]
        dist = jnp.sqrt(jnp.sum(diff**2, axis=-1))
        dist = jnp.where(dist < 1e-12, 1e-12, dist)
        weight = dist ** (-4.0)
        return weight / jnp.sum(weight, axis=-1, keepdims=True)

    if scheme in ["scheme1"]:
        rescale_fn = redistribute_samples_scheme1
    elif scheme in ["scheme2"]:
        rescale_fn = redistribute_samples_scheme2
    else:
        rescale_fn = redistribute_samples_scheme1

    jac_rescale_fn = jax.jacobian(rescale_fn, argnums=0)

    # -------------------------
    # Helpers: grads wrt electron pos / nuclear pos (your code structure)
    # IMPORTANT: we will route params through wrappers to freeze J2 dynamically.
    # -------------------------
    @jax.jit
    def grad_fn_ee(e_pos):
        return jax.grad(local_energy_ee)(e_pos)

    @jax.jit
    def grad_fn_en(e_pos):
        return jax.grad(local_energy_en, argnums=(0, 1))(e_pos, nuc_crds)

    def _params_with_fixed_J2(params):
        """Return params where J2 is fixed (no grads), J1 trainable."""
        return {
            "J1_params": params["J1_params"],
            "J2_params": jax.lax.stop_gradient(J2_fixed),
        }

    @jax.jit
    def grad_fn_ke(e_pos, params):
        # grads wrt (e_pos, nuc_crds); params influences KE term
        p = _params_with_fixed_J2(params)
        return jax.grad(local_energy_ke, argnums=(0, 1))(e_pos, nuc_crds, p)

    @jax.jit
    def grad_fn_logpsi(e_pos, params):
        p = _params_with_fixed_J2(params)
        return jax.grad(log_trial_wavefunction, argnums=(0, 1))(e_pos, nuc_crds, p)

    @jax.jit
    def total_local_energy_fn(elec_crds, params):
        p = _params_with_fixed_J2(params)
        return (
            local_energy_ee(elec_crds)
            + local_energy_en(elec_crds, nuc_crds)
            + local_energy_ke(elec_crds, nuc_crds, p)
            + ener_nn
        )

    # -------------------------
    # Metropolis step (sampling uses fixed J2 as requested; J1 affects psi)
    # -------------------------
    @jax.jit
    def metropolis_step(rng_key, elec_crds, _step_size, params):
        key_prop, key_accept = jax.random.split(rng_key)

        noise = jax.random.normal(key_prop, elec_crds.shape)
        proposed_crds = elec_crds + _step_size * noise

        # singularity checks
        min_dist_threshold = 1e-4

        diffs_ee = proposed_crds[i_e] - proposed_crds[j_e]
        dists_ee = jnp.sqrt(jnp.sum(diffs_ee * diffs_ee, axis=-1))

        diffs_en = proposed_crds[:, None, :] - nuc_crds[None, :, :]
        dists_en = jnp.sqrt(jnp.sum(diffs_en * diffs_en, axis=-1))

        valid_move = (dists_en.min() > min_dist_threshold) & (dists_ee.min() > min_dist_threshold)

        p = _params_with_fixed_J2(params)
        log_psi_old = log_trial_wavefunction(elec_crds, nuc_crds, p)
        log_psi_new = log_trial_wavefunction(proposed_crds, nuc_crds, p)

        acceptance_ratio = jnp.exp(2 * (log_psi_new - log_psi_old))
        accept_prob = jnp.minimum(1.0, acceptance_ratio)

        accept = (jax.random.uniform(key_accept) < accept_prob) & valid_move
        new_crds = jnp.where(accept, proposed_crds, elec_crds)

        return new_crds, accept

    # -------------------------
    # Your batch force estimator pieces (rewired to accept params)
    # -------------------------
    def vmc_gradient_batch(batch_samples, params):
        # grads per-sample
        grd_ee_elc = jax.vmap(grad_fn_ee)(batch_samples)
        grd_en_elc, grd_en_nuc = jax.vmap(grad_fn_en)(batch_samples)
        grd_ke_elc, grd_ke_nuc = jax.vmap(lambda x: grad_fn_ke(x, params))(batch_samples)
        grd_logpsi_elc, grd_logpsi_nuc = jax.vmap(lambda x: grad_fn_logpsi(x, params))(batch_samples)

        # rescale + jacobian correction
        rescale = jax.vmap(rescale_fn)(batch_samples)
        jac_rescale_elc = jax.vmap(jac_rescale_fn)(batch_samples)
        novel_correction = 0.5 * jnp.einsum("beneK->bnK", jac_rescale_elc)

        grd_ee = jnp.einsum("beK,ben->bnK", grd_ee_elc, rescale)
        grd_en = grd_en_nuc + jnp.einsum("beK,ben->bnK", grd_en_elc, rescale)
        grd_ke = grd_ke_nuc + jnp.einsum("beK,ben->bnK", grd_ke_elc, rescale)

        grd_logpsi = grd_logpsi_nuc + jnp.einsum("beK,ben->bnK", grd_logpsi_elc, rescale)
        grd_logpsi += novel_correction

        return grd_ee, grd_en, grd_ke, grd_logpsi

    @jax.jit
    def vmc_force_estimator(batch_samples, params):
        """
        Return nuclear force (num_nuc,3) using your formula:
          grad_tot = grad_nn + <grd_ee> + <grd_en> + <grd_ke> + <pulay>
          force = -grad_tot
        """
        grd_ee, grd_en, grd_ke, grd_logpsi = vmc_gradient_batch(batch_samples, params)

        # Local energies for Pulay
        local_energies = jax.vmap(lambda x: total_local_energy_fn(x, params))(batch_samples)
        d_enr = local_energies - local_energies.mean()

        # Pulay term: 2 * < (E_L - <E_L>) * grad_logpsi >
        grd_pulay = 2.0 * jnp.einsum("s,snK->snK", d_enr, grd_logpsi).mean(axis=0)

        grad_tot = (
            grad_nn_nuc
            + grd_ee.mean(axis=0)
            + grd_en.mean(axis=0)
            + grd_ke.mean(axis=0)
            + grd_pulay
        )

        return -grad_tot

    # -------------------------
    # Force-only loss: J1 trainable, J2 fixed
    # -------------------------
    @jax.jit
    def force_only_loss(params, batch_samples, target_force):
        p = _params_with_fixed_J2(params)
        pred_force = vmc_force_estimator(batch_samples, p)
        return jnp.mean((pred_force[0][2] - target_force[0][2]) ** 2)

    @jax.jit
    def compute_pred_force(params, batch_samples):
        p = _params_with_fixed_J2(params)
        return vmc_force_estimator(batch_samples, p)
        
    def compute_force_all_walkers_batched(params, dataset, batch_size):
        forces = []
        for i in range(0, dataset.shape[0], batch_size):
            batch = dataset[i:i+batch_size]
            forces.append(compute_pred_force(params, batch))
        return jnp.mean(jnp.stack(forces), axis=0)
        
    @jax.jit
    def compute_force_all_walkers(params, dataset_walkers):
        p = _params_with_fixed_J2(params)
        return vmc_force_estimator(dataset_walkers, p)
        
    @jax.jit
    def force_only_loss_full(params, dataset_walkers, target_force):
        pred_force = compute_force_all_walkers_batched(params, dataset_walkers, batch_size_force)
        return jnp.mean((pred_force[0][2] - target_force[0][2]) ** 2)
        
    # -------------------------
    # Logging helpers
    # -------------------------
    def _norm(x):
        return jnp.sqrt(jnp.sum(jnp.asarray(x) ** 2))

    # -------------------------
    # VMC run (sampling) + optional gradient saving (your original)
    # -------------------------
    def vmc_gradient_save(
        iter,
        sampled_walkers,
        local_energies,
        batch_size,
        num_batches,
        h5py_io="w",
        params=None,
    ):
        """
        Save per-iter gradient samples (your format) using current params (with fixed J2).
        """
        if params is None:
            params = params_vmc

        n_samples = sampled_walkers.shape[0]
        batch_samples = sampled_walkers[:batch_size]
        grd_ee, grd_en, grd_ke, grd_logpsi = vmc_gradient_batch(batch_samples, _params_with_fixed_J2(params))

        for batch_idx in range(1, num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, n_samples)

            batch_samples = sampled_walkers[start_idx:end_idx]
            g_ee, g_en, g_ke, g_logpsi = vmc_gradient_batch(batch_samples, _params_with_fixed_J2(params))

            grd_ee = jnp.append(grd_ee, g_ee, axis=0)
            grd_en = jnp.append(grd_en, g_en, axis=0)
            grd_ke = jnp.append(grd_ke, g_ke, axis=0)
            grd_logpsi = jnp.append(grd_logpsi, g_logpsi, axis=0)

        with h5py.File(chkfile_grd, h5py_io) as f:
            f.create_dataset(f"grd_ee_{iter}", data=grd_ee)
            f.create_dataset(f"grd_en_{iter}", data=grd_en)
            f.create_dataset(f"grd_ke_{iter}", data=grd_ke)
            f.create_dataset(f"grd_logpsi_{iter}", data=grd_logpsi)
            f.create_dataset(f"local_energies_{iter}", data=local_energies)

    def vmc_run(
        rng_key,
        nwalkers=100,
        num_mc_steps=1000,
        tgt_enr_std=0.01,
        max_iter=500,
        step_size=0.25,
        fname_log="vmc_enr.log",
        l_grad=False,
        params=None,
    ):
        """Sampling run (energy stats) with current params (J2 fixed)."""
        if params is None:
            params = params_vmc

        # Initialize electron positions
        rng_key, rng = jax.random.split(rng_key)
        idx_cnt = []
        for ia, iz in enumerate(Z_charges):
            idx_cnt.extend([ia] * int(iz))

        # Handle molecular charge
        if mf.mol.charge < 0:
            idx_cnt.extend([0] * abs(int(mf.mol.charge)))
        elif mf.mol.charge > 0:
            idx_cnt = idx_cnt[: -int(mf.mol.charge)]

        idx_cnt = jnp.array(idx_cnt)
        centers = nuc_crds[idx_cnt]
        walkers = centers[jnp.newaxis, :, :] + 0.05 * jax.random.normal(rng, (nwalkers, nelec, 3))

        # Equilibration
        @jax.jit
        def equilibration_step(state, _):
            rng_key, walkers, step_size = state
            rng_key, key = jax.random.split(rng_key)
            keys = jax.random.split(key, nwalkers)

            new_walkers, accepted = jax.vmap(metropolis_step, in_axes=(0, 0, None, None))(
                keys, walkers, step_size, params
            )
            ratio = accepted.mean()
            new_step_size = step_size * (0.5 + ratio)
            return (rng_key, new_walkers, new_step_size), ratio

        initial_state = (rng_key, walkers, step_size)
        final_state, ratios = jax.lax.scan(equilibration_step, initial_state, jnp.arange(num_mc_steps))
        rng_key, walkers, step_size = final_state
        ratio = ratios[-1]

        print(f"Equilibration Acceptance Rate: {ratio:.2f}")
        print(f"Step size: {step_size:.4f}")

        # Production
        @jax.jit
        def production_step(state, step_number):
            rng_key, walkers, step_size, _ = state
            rng_key, key = jax.random.split(rng_key)
            keys = jax.random.split(key, nwalkers)

            new_walkers, accepted = jax.vmap(metropolis_step, in_axes=(0, 0, None, None))(
                keys, walkers, step_size, params
            )
            ratio = accepted.mean()
            new_step_size = step_size * (0.5 + ratio)

            energies = jax.vmap(lambda x: total_local_energy_fn(x, params))(new_walkers)
            return (rng_key, new_walkers, new_step_size, ratio), (energies, new_walkers)

        base_batch_size = 500
        memory_factor = max(1, nelec * num_nuc // 1000)
        batch_size = max(50, base_batch_size // memory_factor)
        num_batches = (num_mc_steps * nwalkers // 10 + batch_size - 1) // batch_size
        mark_samples = ((jnp.arange(num_mc_steps) + 1) % 10 == 0)

        ratio = 0.5
        initial_state = (rng_key, walkers, step_size, ratio)
        final_state, samples = jax.lax.scan(production_step, initial_state, jnp.arange(num_mc_steps))

        rng_key, walkers, step_size, ratio = final_state
        walkers_energies, sampled_walkers = samples

        enr_mean = walkers_energies.mean(axis=0).mean()
        enr_std = walkers_energies.mean(axis=0).std()

        iter = 1
        fout = open(fname_log, "w", 1)

        print(
            "iter,enr_mean,enr_std,accept_ratio,step_size",
            file=fout,
        )
        print(
            f"{iter:5d},{float(enr_mean):.10f},{float(enr_std):.10f},{float(ratio):.6f},{float(step_size):.6f}",
            file=fout,
        )

        if l_grad:
            sw = sampled_walkers[mark_samples].reshape(-1, nelec, 3)
            le = walkers_energies[mark_samples].reshape(-1)
            vmc_gradient_save(iter, sw, le, batch_size, num_batches, h5py_io="w", params=params)
            with h5py.File(chkfile_grd, "a") as f:
                f.create_dataset("enr_mean", data=enr_mean)
                f.create_dataset("enr_std", data=enr_std)
                f.create_dataset("sampled_iter", data=int(iter), dtype=jnp.int32)
                f.create_dataset("grd_nn", data=grad_nn_nuc)

        # Continue sampling until std criterion
        while (iter < max_iter) & (enr_std > tgt_enr_std):
            initial_state = (rng_key, walkers, step_size, ratio)
            final_state, samples = jax.lax.scan(production_step, initial_state, jnp.arange(num_mc_steps))

            rng_key, walkers, step_size, ratio = final_state
            energies, sampled_walkers = samples
            iter += 1

            walkers_energies = jnp.append(walkers_energies, energies, axis=0)
            enr_mean = walkers_energies.mean(axis=0).mean()
            enr_std = walkers_energies.mean(axis=0).std()

            print(
                f"{iter:5d},{float(enr_mean):.10f},{float(enr_std):.10f},{float(ratio):.6f},{float(step_size):.6f}",
                file=fout,
            )

            if l_grad:
                sw = sampled_walkers[mark_samples].reshape(-1, nelec, 3)
                le = energies[mark_samples].reshape(-1)
                vmc_gradient_save(iter, sw, le, batch_size, num_batches, h5py_io="a", params=params)

        #print(f"Total energy [Ha]: {enr_mean:.6f}")
        #print(f"Total energy std: {enr_std:.6f}")

        return {
            "rng_key": rng_key,
            "walkers": walkers,
            "step_size": step_size,
            "accept_ratio": ratio,
            "energies": walkers_energies,     # (steps_total, nwalkers)
            "sampled_walkers_last": sampled_walkers,  # (num_mc_steps, nwalkers, nelec,3)
            "mark_samples": mark_samples,     # (num_mc_steps,)
        }

    # -------------------------
    # Force-only training runner (NEW)
    # -------------------------
    def vmc_force_train(
        rng_key,
        # sampling
        nwalkers=100,
        num_mc_steps=2000,
        step_size=0.25,
        sample_every=10,          # use every N-th step from last production block
        # training
        num_epochs=50,
        batch_size_force=256,
        lr=1e-2,
        optimizer="adam",
        grad_clip=1.0,
        # logging
        fname_log="vmc_force_train.log",
        # resampling cadence
        resample_every=1,         # resample dataset every epoch
        # verbosity
        verbose=True,
    ):
        """
        Train J1_params using force-only objective, with J2 fixed.
        Logs include energy mean/std changes, J1/J2 norm and delta per epoch, and whether J2 changed.
        """

        # Target force from RHF
        target_force = -jnp.array(mf.nuc_grad_method().kernel(), dtype=jnp.float64)  # (num_nuc,3)

        # Optimizer (mask via stop_gradient on J2, plus explicit post-fix)
        base_opt = optax.adam(lr) if "adam" in optimizer.lower() else optax.sgd(lr)
        if grad_clip is not None and grad_clip > 0:
            opt = optax.chain(optax.clip_by_global_norm(grad_clip), base_opt)
        else:
            opt = base_opt

        params = {
            "J1_params": jnp.array(params_vmc["J1_params"], dtype=jnp.float64),
            "J2_params": jnp.array(J2_fixed, dtype=jnp.float64),  # fixed reference
        }
        opt_state = opt.init(params)

        # Logging header
        with open(fname_log, "w", 1) as f:
            f.write(
                "epoch,"
                "force_loss,"
                "E_mean,"
                "E_std,"
                "dE_mean,"
                "dE_std,"
                "J1_norm,"
                "dJ1_norm,"
                "J2_norm,"
                "dJ2_norm,"
                "J2_changed\n"
            )

        prev_E_mean = None
        prev_E_std = None
        prev_J1 = params["J1_params"]
        prev_J2 = params["J2_params"]

        # dataset placeholders
        dataset_walkers = None

        @jax.jit
        def compute_pred_force(params, batch_samples):
            p = _params_with_fixed_J2(params)
            return vmc_force_estimator(batch_samples, p)

        @jax.jit
        def compute_force_all_walkers(params, dataset_walkers):
            p = _params_with_fixed_J2(params)
            return vmc_force_estimator(dataset_walkers, p)

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
            
        @jax.jit
        def force_step(params, opt_state, batch, target_force):
            loss_f, grads_f = jax.value_and_grad(force_only_loss)(
                params, batch, target_force
            )
            updates, opt_state = opt.update(grads_f, opt_state, params)
            params = optax.apply_updates(params, updates)
            return params, opt_state, loss_f


        @jax.jit
        def energy_step(params, opt_state, batch):
            loss_e, grads_e = jax.value_and_grad(loss_fn)(params, batch)
            updates, opt_state = opt.update(grads_e, opt_state, params)
            params = optax.apply_updates(params, updates)
            return params, opt_state, loss_e

        # Helper to build dataset from a sampling block
        def _build_dataset(sample_block):
            # sample_block["sampled_walkers_last"]: (num_mc_steps, nwalkers, nelec,3)
            sw = sample_block["sampled_walkers_last"]
            mark = ((jnp.arange(sw.shape[0]) + 1) % sample_every) == 0
            ds = sw[mark].reshape(-1, nelec, 3)
            return ds

        # Main epochs
        for epoch in range(num_epochs):
            if (epoch % resample_every) == 0 or dataset_walkers is None or epoch == 0:
                # resample under current params (J2 fixed)
                sample_block = vmc_run(
                    rng_key=rng_key,
                    nwalkers=nwalkers,
                    num_mc_steps=num_mc_steps,
                    tgt_enr_std=1e9,  # don't early stop
                    max_iter=1,       # single production block
                    step_size=step_size,
                    fname_log=f"{fname_log}",
                    l_grad=False,
                    params=params,
                )
                rng_key = sample_block["rng_key"]
                step_size = float(sample_block["step_size"])  # keep adaptive step
                dataset_walkers = _build_dataset(sample_block)

                # energy stats for this epoch
                energies = sample_block["energies"]  # (num_mc_steps, nwalkers)
                E_mean = float(energies.mean(axis=0).mean())
                E_std = float(energies.mean(axis=0).std())
            else:
                # if not resampling, still compute E on a subset (cheap)
                # We'll just set to NaN to avoid misleading numbers
                E_mean = float(energies.mean(axis=0).mean())
                E_std = float(energies.mean(axis=0).std())
            # training minibatches
            # (if dataset smaller than batch, just use all)
            #params, opt_state, loss = energy_step(
            #    params, opt_state, dataset_walkers
            #)

            #params, opt_state, loss = force_step(
            #    params, opt_state, dataset_walkers, target_force
            #)
            params, opt_state, loss = force_step(
                params, opt_state, dataset_walkers, target_force
            )
            pred_force = compute_force_all_walkers_batched(params, dataset_walkers, batch_size_force)

            force_loss = float(jnp.mean(jnp.array(loss)))

            # deltas / norms
            J1_norm = float(_norm(params["J1_params"]))
            dJ1_norm = float(_norm(params["J1_params"] - prev_J1))

            J2_norm = float(_norm(params["J2_params"]))
            dJ2_norm = float(_norm(params["J2_params"] - prev_J2))
            J2_changed = bool(dJ2_norm > 0.0)

            # energy change
            if prev_E_mean is None or (jnp.isnan(prev_E_mean) or jnp.isnan(E_mean)):
                dE_mean = float("nan")
                dE_std = float("nan")
            else:
                dE_mean = float(E_mean - prev_E_mean)
                dE_std = float(E_std - prev_E_std)

            # write log
            with open(fname_log, "a", 1) as f:
                f.write(
                    f"{epoch},"
                    f"{force_loss:.10e},"
                    f"{E_mean:.10f},"
                    f"{E_std:.10f},"
                    f"{dE_mean:.10f},"
                    f"{dE_std:.10f},"
                    f"{J1_norm:.10e},"
                    f"{dJ1_norm:.10e},"
                    f"{J2_norm:.10e},"
                    f"{dJ2_norm:.10e},"
                    f"{int(J2_changed)}\n"
                )

            if verbose:
                print(
                    f"[epoch {epoch:03d}/{num_epochs}] | "
                    f"Pred_F={pred_force[0][2]:.6f} | "
                    f"F_loss={force_loss:.3e} | "
                    f"E_mean={E_mean:.6f} E_std={E_std:.6f} | "
                    f"J1={params['J1_params'][0]:.3e} , {params['J1_params'][1]:.3e} | "
                    f"J2={params['J2_params'][0]:.3e} , {params['J2_params'][1]:.3e} "
                )

            # update prev trackers
            prev_J1 = params["J1_params"]
            prev_J2 = params["J2_params"]
            prev_E_mean = E_mean
            prev_E_std = E_std

        # return trained params (J2 fixed)
        return params

    # -------------------------
    # Original gradient_calc (kept) - reads HDF5 outputs
    # -------------------------
    def vmc_gradient_calc():
        with h5py.File(chkfile_grd, "r") as f:
            dict_grd_samples = {}
            for key, data in f.items():
                if key in ["enr_mean", "enr_std", "sampled_iter"]:
                    dict_grd_samples[key] = data[()]
                else:
                    dict_grd_samples[key] = jnp.array(data[:])

            sampled_iter = int(dict_grd_samples["sampled_iter"])
            enr_mean = dict_grd_samples["enr_mean"]
            enr_std = dict_grd_samples["enr_std"]
            grd_nn = dict_grd_samples["grd_nn"]

            enr_max = enr_mean + 3.0 * enr_std
            enr_min = enr_mean - 3.0 * enr_std

            grd_ee_list = []
            grd_en_list = []
            grd_ke_list = []
            grd_pulay_list = []
            for it in range(sampled_iter):
                grd_ee = dict_grd_samples[f"grd_ee_{it+1}"]
                grd_en = dict_grd_samples[f"grd_en_{it+1}"]
                grd_ke = dict_grd_samples[f"grd_ke_{it+1}"]
                grd_logpsi = dict_grd_samples[f"grd_logpsi_{it+1}"]
                local_energies = dict_grd_samples[f"local_energies_{it+1}"]

                mark = (local_energies > enr_min) * (local_energies < enr_max)
                d_enr = local_energies - enr_mean
                grd_pulay = 2.0 * jnp.einsum("s,snK->snK", d_enr, grd_logpsi)

                grd_ee_list.append(grd_ee[mark].mean(axis=0))
                grd_en_list.append(grd_en[mark].mean(axis=0))
                grd_ke_list.append(grd_ke[mark].mean(axis=0))
                grd_pulay_list.append(grd_pulay[mark].mean(axis=0))

            grd_ee = jnp.stack(grd_ee_list, axis=0).mean(axis=0)
            grd_en = jnp.stack(grd_en_list, axis=0).mean(axis=0)
            grd_ke = jnp.stack(grd_ke_list, axis=0).mean(axis=0)
            grd_pulay = jnp.stack(grd_pulay_list, axis=0).mean(axis=0)

            grd_tot = grd_nn + grd_ee + grd_en + grd_ke + grd_pulay
            grd_tot_list = (
                jnp.stack(grd_ee_list, axis=0)
                + jnp.stack(grd_en_list, axis=0)
                + jnp.stack(grd_ke_list, axis=0)
                + jnp.stack(grd_pulay_list, axis=0)
            )

            with jnp.printoptions(precision=5, suppress=True):
                print("grd_nn\n", grd_nn)
                print("grd_ee\n", grd_ee)
                print("grd_en\n", grd_en)
                print("grd_ke\n", grd_ke)
                print("grd_pulay\n", grd_pulay)
                print("grd_tot\n", grd_tot)
                print("grd_tot_std\n", grd_tot_list.std(axis=0))

            return grd_tot

    # Return: sampling + post-analysis + force-only training
    return vmc_run, vmc_gradient_calc, vmc_force_train
