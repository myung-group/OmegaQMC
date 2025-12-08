"""
Variational Monte Carlo with Gaussian-Type Orbitals.

This module provides VMC sampling and gradient computation for molecular systems
using space-warping coordinate transformations for nuclear gradient estimation.
"""

import jax
import jax.numpy as jnp
import h5py
from typing import Tuple, Callable, Optional, List

from vmc_mlsw.psi_gto import get_psi_fun
from vmc_mlsw.symm.electron_reflection import (
    diatomic_reflection_electrons,
    water_reflection_electrons,
    water_dimer_reflection_electrons,
    water_cluster_reflection_electrons
)
from vmc_mlsw.vmc_utils import (
    date,
    batched_binning_analysis,
    batched_binning_analysis_grds,
    compute_torque_with_error
)
# Reflection operation name to ID mapping
REFLECTION_IDS = {'I': 0, 'x': 1, 'y': 2, 'xy': 3}

# VMC hyperparameters
EQUIL_MC_STEPS = 5000
TARGET_ACCEPTANCE_RATE = 0.4
STEP_SIZE_ADAPTATION_RATE = 0.05
MIN_DIST_THRESHOLD = 1e-6


def _get_electron_reflection_fn(Z_charges: jnp.ndarray,
                                 nuc_crds: jnp.ndarray,
                                 cluster_idx: Optional[List[int]]) -> Callable:
    """Select appropriate electron reflection function based on molecular composition."""
    charge_tuple = tuple(Z_charges)

    if cluster_idx is not None:
        return water_cluster_reflection_electrons(nuc_crds, cluster_idx)
    elif charge_tuple == (8, 1, 1):  # Water molecule
        return water_reflection_electrons(nuc_crds)
    elif charge_tuple == (8, 1, 1, 8, 1, 1):  # Water dimer
        return water_dimer_reflection_electrons(nuc_crds)
    else:  # Diatomic or other molecules
        return diatomic_reflection_electrons(nuc_crds)


def _initialize_walkers(rng_key: jax.Array,
                        nwalkers: int,
                        nelec: int,
                        Z_charges: jnp.ndarray,
                        nuc_crds: jnp.ndarray,
                        mol_charge: int) -> jnp.ndarray:
    """Initialize walker positions near nuclear centers."""
    # Assign electrons to atoms based on atomic number
    idx_cnt = []
    for ia, iz in enumerate(Z_charges):
        idx_cnt.extend([ia] * iz)

    # Adjust for molecular charge
    if mol_charge < 0:
        idx_cnt.extend([0] * abs(mol_charge))
    elif mol_charge > 0:
        idx_cnt = idx_cnt[:-mol_charge]

    idx_cnt = jnp.array(idx_cnt)
    centers = nuc_crds[idx_cnt]

    # Initialize with small Gaussian noise around centers
    noise = jax.random.normal(rng_key, (nwalkers, nelec, 3))
    walkers = centers[jnp.newaxis, :, :] + 0.05 * noise

    return walkers


def _adapt_step_size(step_size: float, acceptance_ratio: float) -> float:
    """Adapt step size based on acceptance ratio."""
    log_step = jnp.log(step_size) + STEP_SIZE_ADAPTATION_RATE * (
        acceptance_ratio - TARGET_ACCEPTANCE_RATE
    )
    return jnp.exp(log_step)


def get_vmc_func(mf,
                 params_vmc: dict,
                 scheme: str = 'scheme1',
                 chkfile: str = 'vmc_chk.hdf5',
                 cgto_coeff: Optional[jnp.ndarray] = None,
                 reflection_op_list: Optional[List[str]] = None,
                 cluster_idx: [List[int]] = None) -> Tuple[Callable, Callable]:
    """
    Create VMC sampling and gradient computation functions.

    Args:
        mf: PySCF mean-field object
        params_vmc: VMC parameters (Jastrow coefficients, etc.)
        scheme: Redistribution scheme ('scheme1' or 'scheme2')
        chkfile: HDF5 checkpoint file for to-restart data and gradient data
        cgto_coeff: Custom GTO coefficients (optional)
        reflection_op_list: List of reflection operations to use
        cluster_idx: List of indices of water molecules in n-H2O system (n=1,2,..)
                - Each molecule should be ordered as [O, H-1, H-2]
                - ex) [[0, 1, 2], [3, 4, 5]]
                    -> 0, 3: Oxygen, 1, 2, 4, 5: Hydrogen
                    -> [0, 1, 2]: 1st H2O, [3, 4, 5]: 2nd H2O

    Returns:
        Tuple of (vmc_run, vmc_gradient_with_space_warping) functions
    """
    # Default to identity reflection only
    if reflection_op_list is None:
        reflection_op_list = ['I']

    reflection_ID_list = [REFLECTION_IDS[op] for op in reflection_op_list]
    num_reflections = len(reflection_ID_list)

    # Extract molecular data
    nuc_crds = jnp.array(mf.mol.atom_coords(unit='Bohr'))
    nelec = mf.mol.tot_electrons()
    num_nuc = nuc_crds.shape[0]
    Z_charges = mf.mol.atom_charges()
    mol_charge = mf.mol.charge

    # Precompute electron pair indices for distance calculations
    i_e, j_e = jnp.triu_indices(nelec, k=1)

    # Get electron reflection function for this molecular type
    run_electron_exchange = _get_electron_reflection_fn(Z_charges, nuc_crds, cluster_idx)

    # Get wavefunction and energy functions
    log_trial_wavefunction, local_energy, get_psi_mo = get_psi_fun(mf, cgto_coeff)
    local_energy_ee, local_energy_nn, local_energy_en, local_energy_ke = local_energy

    # Precompute nuclear-nuclear energy and gradient
    ener_nn = local_energy_nn(nuc_crds)
    grad_nn_nuc = jax.grad(local_energy_nn)(nuc_crds)

    # --- Redistribution schemes for space warping ---

    @jax.jit
    def redistribute_scheme1(elec_crds: jnp.ndarray) -> jnp.ndarray:
        """MO-based redistribution weights."""
        _, mo_val_s = get_psi_mo(elec_crds, nuc_crds)
        weight = jnp.einsum('neo,neo->en', mo_val_s, mo_val_s)
        return weight / jnp.sum(weight, axis=-1, keepdims=True)

    @jax.jit
    def redistribute_scheme2(elec_crds: jnp.ndarray) -> jnp.ndarray:
        """Distance-based redistribution weights (1/r^4)."""
        diff = elec_crds[:, None, :] - nuc_crds[None, :, :]
        dist_sq = jnp.sum(diff**2, axis=-1)
        dist = jnp.sqrt(jnp.maximum(dist_sq, 1e-24))
        weight = dist**(-4.0)
        return weight / jnp.sum(weight, axis=-1, keepdims=True)

    rescale_fn = redistribute_scheme1 if scheme == 'scheme1' else redistribute_scheme2
    jac_rescale_fn = jax.jacobian(rescale_fn, argnums=0)

    # --- Gradient functions ---

    @jax.jit
    def grad_fn_ee(e_pos: jnp.ndarray) -> jnp.ndarray:
        return jax.grad(local_energy_ee)(e_pos)

    @jax.jit
    def grad_fn_en(e_pos: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        return jax.grad(local_energy_en, argnums=(0, 1))(e_pos, nuc_crds)

    @jax.jit
    def grad_fn_ke(e_pos: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        return jax.grad(local_energy_ke, argnums=(0, 1))(e_pos, nuc_crds, params_vmc)

    @jax.jit
    def grad_fn_logpsi(e_pos: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        return jax.grad(log_trial_wavefunction, argnums=(0, 1))(e_pos, nuc_crds, params_vmc)

    # --- Metropolis sampling functions ---

    @jax.jit
    def metropolis_step(rng_key: jax.Array,
                        elec_crds: jnp.ndarray,
                        step_size: float) -> Tuple[jnp.ndarray, bool]:
        """Single Metropolis-Hastings step with Gaussian proposal."""
        key_prop, key_accept = jax.random.split(rng_key)

        # Propose new positions
        noise = jax.random.normal(key_prop, elec_crds.shape)
        proposed_crds = elec_crds + step_size * noise

        # Check for singularities (electron-electron and electron-nuclear)
        diffs_ee = proposed_crds[i_e] - proposed_crds[j_e]
        dists_ee = jnp.linalg.norm(diffs_ee, axis=-1)

        diffs_en = proposed_crds[:, None, :] - nuc_crds[None, :, :]
        dists_en = jnp.linalg.norm(diffs_en, axis=-1)

        valid_move = (dists_en.min() > MIN_DIST_THRESHOLD) & \
                     (dists_ee.min() > MIN_DIST_THRESHOLD)

        # Compute acceptance probability
        log_psi_old = log_trial_wavefunction(elec_crds, nuc_crds, params_vmc)
        log_psi_new = log_trial_wavefunction(proposed_crds, nuc_crds, params_vmc)

        log_acc = 2.0 * (log_psi_new - log_psi_old)
        accept_prob = jnp.where(log_acc > 0.0, 1.0, jnp.exp(log_acc))
        #accept_prob = jnp.where(accept_prob < 1.0e-3, 0.0, accept_prob)
        accept = (jax.random.uniform(key_accept) < accept_prob) & valid_move
        new_crds = jnp.where(accept, proposed_crds, elec_crds)

        return new_crds, accept

    @jax.jit
    def metropolis_reflection(rng_key: jax.Array,
                              elec_crds: jnp.ndarray,
                              reflection_ID: int) -> jnp.ndarray:
        """Metropolis step with reflection move."""
        rescale = rescale_fn(elec_crds)
        proposed_crds = run_electron_exchange(elec_crds, rescale, reflection_ID)

        # Compute acceptance probability
        log_psi_old = log_trial_wavefunction(elec_crds, nuc_crds, params_vmc)
        log_psi_new = log_trial_wavefunction(proposed_crds, nuc_crds, params_vmc)

        log_acc = 2.0 * (log_psi_new - log_psi_old)
        accept_prob = jnp.where(log_acc > 0.0, 1.0, jnp.exp(log_acc))
        #accept_prob = jnp.where(accept_prob < 1.0e-3, 0.0, accept_prob)
        accept = jax.random.uniform(rng_key) < accept_prob
        new_crds = jnp.where(accept, proposed_crds, elec_crds)

        return new_crds

    # --- Gradient batch computation ---

    def vmc_gradient_batch(batch_samples: jnp.ndarray) -> Tuple[jnp.ndarray, ...]:
        """Compute gradients for a batch of samples."""
        grd_ee_elc = jax.vmap(grad_fn_ee)(batch_samples)
        grd_en_elc, grd_en_nuc = jax.vmap(grad_fn_en)(batch_samples)
        grd_ke_elc, grd_ke_nuc = jax.vmap(grad_fn_ke)(batch_samples)
        grd_logpsi_elc, grd_logpsi_nuc = jax.vmap(grad_fn_logpsi)(batch_samples)

        rescale = jax.vmap(rescale_fn)(batch_samples)
        jac_rescale_elc = jax.vmap(jac_rescale_fn)(batch_samples)
        novel_correction = 0.5 * jnp.einsum('beneK->bnK', jac_rescale_elc)

        grd_ee = jnp.einsum('beK,ben->bnK', grd_ee_elc, rescale)
        grd_en = grd_en_nuc + jnp.einsum('beK,ben->bnK', grd_en_elc, rescale)
        grd_ke = grd_ke_nuc + jnp.einsum('beK,ben->bnK', grd_ke_elc, rescale)

        grd_logpsi = grd_logpsi_nuc + jnp.einsum('beK,ben->bnK', grd_logpsi_elc, rescale)
        grd_logpsi += novel_correction

        return grd_ee, grd_en, grd_ke, grd_logpsi

    # --- Gradient saving ---

    def vmc_gradient_save(iteration: int,
                          sampled_walkers: jnp.ndarray,
                          local_energies: jnp.ndarray,
                          batch_size: int,
                          h5py_mode: str = 'w') -> None:
        """Compute and save gradients to HDF5 checkpoint."""
        n_samples = sampled_walkers.shape[0]
        num_batches = (n_samples + batch_size - 1) // batch_size

        w_grd_ee_en_ke = []
        w_grd_logpsi = []
        w_grd_ke = []

        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, n_samples)
            batch_samples = sampled_walkers[start_idx:end_idx]

            g_ee, g_en, g_ke, g_logpsi = vmc_gradient_batch(batch_samples)
            w_grd_ee_en_ke.append(g_ee + g_en + g_ke)
            w_grd_logpsi.append(g_logpsi)
            w_grd_ke.append(g_ke)

        # Stack all batches
        w_grd_ee_en_ke = jnp.vstack(w_grd_ee_en_ke)
        w_grd_logpsi = jnp.vstack(w_grd_logpsi)
        w_grd_ke = jnp.vstack(w_grd_ke)

        # Save to HDF5
        base, ext = chkfile.rsplit('.', 1)
        chkfile_grd = f"{base}_grd.{ext}"
        with h5py.File(chkfile_grd, h5py_mode) as f:
            f.create_dataset(f'grd_ee_en_ke_{iteration}', data=w_grd_ee_en_ke)
            f.create_dataset(f'grd_logpsi_{iteration}', data=w_grd_logpsi)
            f.create_dataset(f'local_energies_{iteration}', data=local_energies)
            f.create_dataset(f'grd_ke_{iteration}', data=w_grd_ke)

    # --- Main VMC run ---

    def vmc_run(rng_key: int = 888,
                nwalkers: int = 100,
                num_mc_steps: int = 1000,
                max_mc_iter: int = 500,
                mc_step_size: float = 0.25,
                tolerance_enr_std_per_elec: float = 0.01,
                fname_log: str = 'vmc_enr.log',
                l_grad: bool = False,
                batch_size: int = 50,
                restart: bool = False) -> None:
        """
        Run VMC sampling with optional gradient computation.

        Args:
            rng_key: JAX random key
            nwalkers: Number of parallel walkers
            num_mc_steps: MC steps per iteration
            max_mc_iter: Maximum number of iterations
            mc_step_size: Initial Metropolis step size
            tolerance_enr_std_per_elec: Convergence threshold (std per electron)
            fname_log: Log file path
            l_grad: Whether to compute and save gradients
            batch_size: Batch size for the gradient calculation
            restart: Restart the simulation from the saved *.hdf5 checkpoint
        """
        # Initialize walkers
        rng_key_to_restart = rng_key
        rng_key = jax.random.key(rng_key)
        rng_key, init_key = jax.random.split(rng_key)
        walkers = _initialize_walkers(
            init_key, nwalkers, nelec, Z_charges, nuc_crds, mol_charge
        )

        # Equilibration phase
        @jax.jit
        def equilibration_step(state, _):
            rng_key, walkers, step_size = state
            rng_key, key = jax.random.split(rng_key)
            walker_keys = jax.random.split(key, nwalkers)

            new_walkers, accepted = jax.vmap(
                metropolis_step, in_axes=(0, 0, None)
            )(walker_keys, walkers, step_size)

            ratio = accepted.mean()
            new_step_size = _adapt_step_size(step_size, ratio)

            return (rng_key, new_walkers, new_step_size), ratio

        if restart:
            with h5py.File(chkfile, 'r') as f:
                # Load metadata
                start_iter = int(f['sampled_iter'][()])
                mc_step_size = f['mc_step_size'][()]
                rng_key = int(f['rng_key'][()])
                rng_key_to_restart = rng_key
                rng_key = jax.random.key(rng_key)
                rng_key, init_key = jax.random.split(rng_key)
                walkers = jnp.array(f['walkers'][:])
                E_w = list(f['E_w'][:])
                print(f"Restarting ...")
                print(f"Step size: {mc_step_size:.4f}")
        else:
            initial_state = (rng_key, walkers, mc_step_size)
            final_state, ratios = jax.lax.scan(
                equilibration_step, initial_state, jnp.arange(EQUIL_MC_STEPS)
            )
            rng_key, walkers, mc_step_size = final_state
            print(f"Equilibration Acceptance Rate: {ratios[-1]:.2f}")
            print(f"Step size: {mc_step_size:.4f}")
            E_w = []
            start_iter = 0

        # Production phase
        @jax.jit
        def production_step(state, step_number):
            rng_key, walkers, step_size = state
            rng_key, key, key_reflection = jax.random.split(rng_key, 3)
            walker_keys = jax.random.split(key, nwalkers)

            is_move = step_number % min(2, num_reflections)

            def metropolis_branch(_):
                new_walkers, accepted = jax.vmap(
                    metropolis_step, in_axes=(0, 0, None)
                )(walker_keys, walkers, step_size)
                new_step_size = _adapt_step_size(step_size, accepted.mean())
                return new_walkers, new_step_size

            def reflection_branch(_):
                reflection_ID = \
                    jax.random.randint (key_reflection,minval=1,maxval=num_reflections,shape=(nwalkers,))
                new_walkers = jax.vmap(
                    metropolis_reflection, in_axes=(0, 0, 0)
                )(walker_keys, walkers, reflection_ID)
                return new_walkers, step_size

            new_walkers, new_step_size = jax.lax.cond(
                is_move == 0,
                metropolis_branch,
                reflection_branch,
                operand=None
            )

            out_walkers = jnp.where (
                is_move == 0,
                new_walkers,
                walkers
            )

            # Compute local energies
            enr_ee = jax.vmap(local_energy_ee)(new_walkers)
            enr_en = jax.vmap(local_energy_en, in_axes=(0, None))(new_walkers, nuc_crds)
            enr_ke = jax.vmap(local_energy_ke, in_axes=(0, None, None))(
                new_walkers, nuc_crds, params_vmc
            )

            return (rng_key, out_walkers, new_step_size), (enr_ee, enr_en, enr_ke, new_walkers)

        # Batch size computation for gradient calculation
        base_batch_size = 500
        memory_factor = max(1, nelec * num_nuc // 1000)
        batch_size = min(batch_size, base_batch_size // memory_factor)
        print("----------------------------------------------")
        print(f"Adjusted batch size: {batch_size}")
        print("----------------------------------------------")

        # Main sampling loop
        with open(fname_log, 'w', buffering=1) as fout:
            print(f"   Iter       Mean        Std            Loc            "
                  f"EE             EN             KE      MM/DD/YYYY HH/MM/SS", file=fout)
            for iteration in range(start_iter, max_mc_iter):
                h5py_mode = 'w' if iteration == 0 else 'a'

                initial_state = (rng_key, walkers, mc_step_size)
                final_state, samples = jax.lax.scan(
                    production_step, initial_state, jnp.arange(num_mc_steps)
                )

                rng_key, walkers, mc_step_size = final_state
                b_enr_ee, b_enr_en, b_enr_ke, b_walkers = samples
                # b_enr_loc [nbatch, nwalkers]
                b_enr_loc = b_enr_ee + b_enr_en + b_enr_ke + ener_nn

                # Accumulate energy statistics
                E_w.append(b_enr_loc.mean(axis=0))
                # E_w_arr [iteration+1, nwalkers]
                E_w_arr = jnp.array(E_w)
                enr_mean = E_w_arr.mean()
                enr_std = E_w_arr.mean(axis=0).std() / nelec
                       
                # Log progress
                print(
                    f'{iteration+1:5d}  '
                    f'{enr_mean:13.6f}  {enr_std:10.6f}  '
                    f'{b_enr_loc.mean(): 13.6f}  '
                    f'{b_enr_ee.mean():13.6f}  '
                    f'{b_enr_en.mean():13.6f}  '
                    f'{b_enr_ke.mean():13.6f}  '
                    f'{date()}  ',
                    file=fout
                )

                # Save gradients if requested
                if l_grad:
                    sampled_walkers = b_walkers.reshape(-1, nelec, 3)
                    local_energies = b_enr_loc#.reshape(-1)
                    vmc_gradient_save(
                        iteration + 1,
                        sampled_walkers,
                        local_energies,
                        batch_size,
                        h5py_mode=h5py_mode
                    )

                # Check convergence
                if enr_std < tolerance_enr_std_per_elec:
                    break
                
                with h5py.File(chkfile, 'w') as f:
                    # to restart
                    f.create_dataset('E_w', data=E_w)
                    f.create_dataset('rng_key', data=rng_key_to_restart)
                    f.create_dataset('walkers', data=walkers)
                    f.create_dataset('mc_step_size', data=mc_step_size)
                    f.create_dataset('sampled_iter', data=iteration + 1, dtype=jnp.int32)
                    f.create_dataset('grd_nn', data=grad_nn_nuc)
                    f.create_dataset('enr_mean', data=enr_mean)

        xbar, serr, s, kappa = batched_binning_analysis(
            E_w_arr, walker_based_batch_size
        )
        e_mean = jnp.array(xbar).mean()
        e_err = jnp.linalg.norm(serr) / len(serr)
        print(f'Total energy | error [Ha]: {e_mean:.6f} | {e_err:.6f}')

    # --- Gradient post-processing ---
    def vmc_gradient_with_space_warping(
        fname_log: str = 'vmc_grad.log',
        compute_error: bool = False,
        walker_based_batch_size: int = 10) -> jnp.ndarray:
        """
        Compute nuclear gradients from saved checkpoint data.

        Args:
            fname_log: Output log file path
            compute_error: Compute errors or not of forces and torques 
                    - forces error calculation requires a large amount of memory
            walker_based_batch_size: Batch size for error computation based on walker

        Returns:
            Total nuclear gradient and gradient's error array (num_nuc, 3), (num_nuc, 3)
        """
        with h5py.File(chkfile, 'r') as f:
            # Load metadata
            sampled_iter = int(f['sampled_iter'][()])
            enr_mean = f['enr_mean'][()]
            grd_nn = jnp.array(f['grd_nn'][:])

        base, ext = chkfile.rsplit('.', 1)
        chkfile_grd = f"{base}_grd.{ext}"
        with h5py.File(chkfile_grd, 'r') as f:
            # Accumulate gradients from all iterations
            grd_ke_sum = 0.0
            valid_samples_count = 0
            if compute_error:
                grd_ee_en_ke_sum = []
                grd_pulay_sum = []
            else:
                grd_ee_en_ke_sum = 0.0
                grd_pulay_sum = 0.0

            for iter_idx in range(1, sampled_iter + 1):
                grd_ee_en_ke = jnp.array(f[f'grd_ee_en_ke_{iter_idx}'][:])
                grd_logpsi = jnp.array(f[f'grd_logpsi_{iter_idx}'][:])
                local_energies = jnp.array(f[f'local_energies_{iter_idx}'][:])
                grd_ke = jnp.array(f[f'grd_ke_{iter_idx}'][:])

                # Pulay force contribution
                d_enr = local_energies - enr_mean
                s_nw, n, _ = grd_ee_en_ke.shape
                s, nw = local_energies.shape
                if compute_error:
                    grd_logpsi = grd_logpsi.reshape(s, nw, n, 3)
                    grd_pulay = 2.0 * jnp.einsum('sb,sbnK->sbnK', d_enr, grd_logpsi)
                    # Regroup
                    grd_ee_en_ke_rg = grd_ee_en_ke.reshape(s, nw, n, 3)
                    grd_pulay_rg = grd_pulay.reshape(s, nw, n, 3)

                    grd_ee_en_ke_sum.append(grd_ee_en_ke_rg)
                    grd_pulay_sum.append(grd_pulay_rg)
                else:
                    d_enr = d_enr.reshape(-1)
                    grd_pulay = 2.0 * jnp.einsum('s,snK->snK', d_enr, grd_logpsi)
                    grd_ee_en_ke_sum += grd_ee_en_ke.sum(axis=0)
                    grd_pulay_sum += grd_pulay.sum(axis=0)
                
                grd_ke_sum += grd_ke.sum(axis=0)
                valid_samples_count += local_energies.reshape(-1).shape[0]

            if compute_error:
                grd_ee_en_ke_sum = jnp.vstack(grd_ee_en_ke_sum)
                grd_pulay_sum = jnp.vstack(grd_pulay_sum)

            # Compute averages
            if valid_samples_count > 0:
                grd_ke = grd_ke_sum / valid_samples_count

                if compute_error:
                    grd_arrays = [grd_ee_en_ke_sum, grd_pulay_sum]
                    grd_tot_ls = jnp.sum(jnp.stack(grd_arrays, axis=0), axis=0)

                    # Compute forces and error
                    xbar, serr, s, kappa = batched_binning_analysis_grds(
                        grd_tot_ls, walker_based_batch_size
                    )
                    grd_tot = jnp.mean(xbar, axis=0) + grd_nn
                    grd_err = jnp.linalg.norm(serr, axis=0) / serr.shape[0]

                    # Compute torques and error
                    torque, dtau = compute_torque_with_error(mf.mol, grd_tot, grd_err)

                    grd_ee_en_ke = jnp.mean(grd_ee_en_ke_sum, axis=0).mean(axis=0)
                    grd_pulay = jnp.mean(grd_pulay_sum, axis=0).mean(axis=0)
                else:
                    grd_ee_en_ke = grd_ee_en_ke_sum / valid_samples_count
                    grd_pulay = grd_pulay_sum / valid_samples_count
                    grd_tot = grd_nn + grd_ee_en_ke + grd_pulay
            else:
                grd_ee_en_ke = jnp.zeros_like(grd_nn)
                grd_pulay = jnp.zeros_like(grd_nn)
                grd_ke = jnp.zeros_like(grd_nn)


            # Write results
            with open(fname_log, 'w', buffering=1) as fout:
                with jnp.printoptions(precision=5, suppress=True):
                    print('grd_nn\n', grd_nn, file=fout)
                    print('grd_ee_en_ke\n', grd_ee_en_ke, file=fout)
                    print('grd_ke\n', grd_ke, file=fout)
                    print('grd_pulay\n', grd_pulay, file=fout)
                    print('grd_tot\n', grd_tot, file=fout)
                    print('grd_tot\n', grd_tot)
                    if compute_error:
                        print('grd_err\n', grd_err, file=fout)
                        print('grd_err\n', grd_err)
                        print('torque\n', torque, file=fout)
                        print('trq_err\n', dtau, file=fout)
                        print('torque\n', torque)
                        print('trq_err\n', dtau)
            if compute_error:
                return grd_tot, grd_err
            else:
                return grd_tot

    return vmc_run, vmc_gradient_with_space_warping
