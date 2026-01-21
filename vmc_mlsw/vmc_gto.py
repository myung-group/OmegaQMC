import sys
import pathlib
from typing import Callable, List, Optional, Tuple
from datetime import datetime
import jax
import jax.numpy as jnp
# from functools import partial
import h5py
from .psi_gto import get_psi_fun
from .cusp import get_cusp_params
from .symm.water_rotation_matrix import symmetrize_water_molecule
# from .constants import CHEMICAL_ACCURACY
from .constants import MIN_DIST_THRESHOLD

from .symm.electron_reflection import (
    diatomic_reflection_electrons,
    water_reflection_electrons,
    water_dimer_reflection_electrons,
    water_cluster_reflection_electrons
)
from .utils import (
    batched_binning_analysis,
    batched_binning_analysis_grds,
    compute_torque,
    compute_torque_with_error
)

# VMC hyperparameters
TARGET_ACCEPTANCE_RATE = 0.4
STEP_SIZE_ADAPTATION_RATE = 0.05

jax.config.update("jax_enable_x64", True)


def _get_electron_reflection_fn(Z_charges: jnp.ndarray,
                                nuc_crds: jnp.ndarray,
                                cluster_idx: Optional[List[int]]) -> Callable:
    """Select appropriate electron reflection function
    based on molecular composition."""
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
                        num_walkers: int,
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
    noise = jax.random.normal(rng_key, (num_walkers, nelec, 3))
    walkers = centers[jnp.newaxis, :, :] + 0.05 * noise

    return walkers


def _adapt_step_size(step_size: float, acceptance_ratio: float) -> float:
    """Adapt step size based on acceptance ratio."""
    log_step = jnp.log(step_size) + STEP_SIZE_ADAPTATION_RATE * (
        acceptance_ratio - TARGET_ACCEPTANCE_RATE
    )
    return jnp.exp(log_step)


@jax.jit
def apply_identity(coords):
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
def apply_reflection_z(coords):
    """Apply reflection across xy-plane (- z-coordinate)."""
    return coords.at[..., 2].multiply(-1)


@jax.jit
def apply_rotation_2(coords):
    """Apply 180-degree rotation about z-axis (- x,y-coordinate)."""
    return coords.at[..., [0, 1]].multiply(-1)


@jax.jit
def apply_rotation_p4(coords):
    """Apply 90-degree ccw rotation about z-axis (-y, x)."""
    return coords.at[..., [0, 1]].set(coords[..., [1, 0]]) \
        .at[..., 0].multiply(-1)


@jax.jit
def apply_rotation_m4(coords):
    """Apply 90-degree cw rotation about z-axis (y, -x)."""
    return coords.at[..., [0, 1]].set(coords[..., [1, 0]]) \
        .at[..., 1].multiply(-1)


@jax.jit
def apply_inversion(coords):
    """Negate all coordinates."""
    return coords.at[..., [0, 1, 2]].multiply(-1)


symmetry_operations_map = {
    'I': apply_identity,
    '1': apply_identity,
    '-I': apply_inversion,
    '-1': apply_inversion,
    'inv': apply_inversion,
    'x': apply_reflection_x,
    'y': apply_reflection_y,
    'z': apply_reflection_z,
    'sigma_x': apply_reflection_x,
    'sigma_y': apply_reflection_y,
    'sigma_z': apply_reflection_z,
    'Cp4': apply_rotation_p4,
    'Cm4': apply_rotation_m4,
    'xy': apply_rotation_2,
    'C2': apply_rotation_2
}


def get_vmc_func(mf,
                 params_corr,
                 cusp_scheme='Quady2025',
                 gr_scheme='scheme1',
                 chkfile_prefix='vmc',
                 symmop_list=["I"],
                 cluster_idx: [List[int]] = None) -> Tuple[Callable, Callable]:
    assert symmop_list != []

    chkfile_name = chkfile_prefix + ".chk.h5"
    chkfile_name_grd = chkfile_prefix + "_grd.chk.h5"

    nuc_crds = jnp.array(mf.mol.atom_coords(unit='Bohr'))
    eps = jnp.finfo(nuc_crds.dtype).eps     # softwired epsilon
    nelec = mf.mol.tot_electrons()
    num_nuc = mf.mol.natm
    Z_charges = mf.mol.atom_charges()
    mol_charge = mf.mol.charge

    # Precompute electron pair indices for distance calculations
    i_e, j_e = jnp.triu_indices(nelec, k=1)

    # Get electron reflection function for this molecular type
    run_electron_exchange \
        = _get_electron_reflection_fn(Z_charges, nuc_crds, cluster_idx)

    # In case of a Water Molecule
    l_water = False
    rot_mat = jnp.eye(3)
    nuc_crds_sym = None
    nuc_crds_op = {}
    nuc_crds_sym_op = {}

    # TODO: this needs to be soft-wired ... I'll come back to this
    #       after polishing the runs with (pre-oriented) H2
    if tuple(Z_charges) == (8, 1, 1):   # water molecule
        # The oxygen atom is placed on the origin.
        l_water = True
        # place the water structure on the yz-plane
        # R(H2) -  R(H1) : y-axis
        # vec (0.5*(R(H1)+R(H2) - R(O)): z-axis

        nuc_crds -= nuc_crds[0]
        nuc_crds_sym, rot_mat = symmetrize_water_molecule(nuc_crds)
        nuc_crds = jnp.einsum('...i,ij->...j', nuc_crds, rot_mat)
        nuc_crds_sym = jnp.einsum('...i,ij->...j', nuc_crds_sym, rot_mat)
        print('translated and rotated water\n', nuc_crds)
        print('water_sym\n', nuc_crds_sym)
        print('water_rot_mat', rot_mat)

        for s_op in symmop_list:
            nuc_crds_op[s_op] = symmetry_operations_map[s_op](nuc_crds)
            nuc_crds_sym_op[s_op] = symmetry_operations_map[s_op](nuc_crds_sym)

    # atomic_masses = mf.mol.atom_mass_list()
    # mass_center = jnp.einsum('i,ij->j',
    #                          atomic_masses, nuc_crds)/atomic_masses.sum()
    # relative_nuc_pos = nuc_crds - mass_center
    timestamp_init = datetime.now()
    print("Begin time: {}".format(timestamp_init))

    if cusp_scheme == "Quady2025":
        params_cusp = {}
        for i in range(num_nuc):
            atom_symbol = mf.mol.atom_symbol(i)
            if atom_symbol not in params_cusp:
                if isinstance(mf.mol.basis, str):
                    p = get_cusp_params(atom_symbol, mf.mol.basis)
                else:
                    p = get_cusp_params(atom_symbol, mf.mol.basis[atom_symbol])
                params_cusp[atom_symbol] = p[atom_symbol]
    else:
        params_cusp = None

    # Get wavefunction and energy functions
    log_trial_wavefunction, local_energy, get_psi_mo \
        = get_psi_fun(mf, params_cusp=params_cusp)

    # Precompute nuclear-nuclear energy and gradient
    local_energy_ee, local_energy_nn, local_energy_en, local_energy_ke \
        = local_energy
    enr_nn = local_energy_nn(nuc_crds)
    grad_nn_nuc = jax.grad(local_energy_nn)(nuc_crds)

    # --- Gradient sample redistribution schemes for space warping ---
    @jax.jit
    def redistribute_scheme1(elec_crds: jnp.ndarray) -> jnp.ndarray:
        _, mo_val_s = get_psi_mo(elec_crds, nuc_crds)
        weight = jnp.einsum('neo,neo->en', mo_val_s, mo_val_s)  # **(1.0/4)
        return weight / jnp.sum(weight, axis=-1, keepdims=True)

    @jax.jit
    def redistribute_scheme2(elec_crds: jnp.ndarray) -> jnp.ndarray:
        diff = elec_crds[:, None, :] - nuc_crds[None, :, :]
        dist = jnp.sqrt(jnp.sum(diff**2, axis=-1))
        dist = jnp.where(dist < eps, eps, dist)
        weight = dist**(-4.0)
        return weight / jnp.sum(weight, axis=-1, keepdims=True)

    rescale_fn = redistribute_scheme2 \
        if 'scheme2' in gr_scheme \
        else redistribute_scheme1
    jac_rescale_fn = jax.jacobian(rescale_fn, argnums=0)

    @jax.jit
    def grad_fn_ee(e_pos: jnp.ndarray) -> jnp.ndarray:
        return jax.grad(local_energy_ee)(e_pos)

    @jax.jit
    def grad_fn_en(e_pos: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        return jax.grad(local_energy_en, argnums=(0, 1))(e_pos, nuc_crds)

    @jax.jit
    def grad_fn_ke(e_pos: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        return jax.grad(local_energy_ke,
                        argnums=(0, 1))(e_pos, nuc_crds, params_corr)

    @jax.jit
    def grad_fn_logpsi(e_pos: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        return jax.grad(log_trial_wavefunction,
                        argnums=(0, 1))(e_pos, nuc_crds, params_corr)

    # @jax.jit
    # def total_local_energy_fn(elec_crds):
    #     return (local_energy_ee(elec_crds)
    #             + local_energy_en(elec_crds, nuc_crds)
    #             + local_energy_ke(elec_crds, nuc_crds, params_corr)
    #             + enr_nn)

    @jax.jit
    def metropolis_move_alle(rng_key: jax.Array,
                             elec_crds: jnp.ndarray,
                             _step_size: float) -> Tuple[jnp.ndarray, bool]:
        """Single Metropolis-Hastings step with Gaussian proposal."""
        key_displace, key_accept = jax.random.split(rng_key)

        # More efficient proposal generation
        proposed_crds = elec_crds \
            + _step_size * jax.random.normal(key_displace, elec_crds.shape)

        # Check for sigularities (electron-electron and electron-nuclei)
        diffs_ee = proposed_crds[i_e] - proposed_crds[j_e]
        dists_ee = jnp.linalg.norm(diffs_ee, axis=-1)

        diffs_en = proposed_crds[:, None, :] - nuc_crds[None, :, :]
        dists_en = jnp.linalg.norm(diffs_en, axis=-1)

        # whether this is a valid move
        is_invalid_move = (dists_en.min() < MIN_DIST_THRESHOLD) \
            | (dists_ee.min() < MIN_DIST_THRESHOLD)
        return jax.lax.cond(is_invalid_move,
                            lambda: (proposed_crds, 0.0),
                            lambda: (proposed_crds, 1.0))

    @jax.jit
    def metropolis_reflection(rng_key: jax.Array,
                              elec_crds: jnp.ndarray,
                              reflection_ID: int) -> jnp.ndarray:
        """Metropolis step with reflection move."""
        rescale = rescale_fn(elec_crds)
        proposed_crds = run_electron_exchange(elec_crds,
                                              rescale, reflection_ID)

        # Compute acceptance probability
        log_psi_old = log_trial_wavefunction(elec_crds, nuc_crds,
                                             params_corr)
        log_psi_new = log_trial_wavefunction(proposed_crds, nuc_crds,
                                             params_corr)

        accept = jax.random.uniform(rng_key) \
            < jnp.exp(2.0 * (log_psi_new - log_psi_old))
        new_crds = jnp.where(accept, proposed_crds, elec_crds)

        return new_crds, 1.0

    @jax.jit
    def metropolis_move_1w(rng_key: jax.Array,
                           elec_crds: jnp.ndarray,
                           _step_size: float) -> Tuple[jnp.ndarray, bool]:
        """Single-walker displacements"""
        key_dtype, key_prop, key_accept = jax.random.split(rng_key, 3)

        # TODO: extend to more displacement types
        # trial_displacements = [metropolis_move_alle, metropolis_reflection]
        # displacement_idx = jax.random.choice(key_dtype, jnp.arange(2))

        trial_displacements = [metropolis_move_alle]
        displacement_idx = 0

        proposed_crds, proposal_ratio = jax.lax.switch(
                displacement_idx,
                trial_displacements,
                key_prop,
                elec_crds,
                _step_size
                )

        # Vectorized acceptance calculation
        log_psi_old = log_trial_wavefunction(elec_crds, nuc_crds,
                                             params_corr)
        log_psi_new = log_trial_wavefunction(proposed_crds, nuc_crds,
                                             params_corr)

        accept = jax.random.uniform(key_accept) \
            < jnp.exp(2.0 * (log_psi_new - log_psi_old)) * proposal_ratio
        new_crds = jnp.where(accept, proposed_crds, elec_crds)

        return new_crds, accept

    metropolis_move_allw = jax.vmap(metropolis_move_1w,
                                    in_axes=(0, 0, None))

    # --- Gradient batch computation ---
    def vmc_gradient_batch(batch_samples: jnp.ndarray) \
            -> Tuple[jnp.ndarray, ...]:
        grd_ee_elc = jax.vmap(grad_fn_ee)(batch_samples)
        grd_en_elc, grd_en_nuc = jax.vmap(grad_fn_en)(batch_samples)
        grd_ke_elc, grd_ke_nuc = jax.vmap(grad_fn_ke)(batch_samples)
        grd_logpsi_elc, grd_logpsi_nuc \
            = jax.vmap(grad_fn_logpsi)(batch_samples)

        rescale = jax.vmap(rescale_fn)(batch_samples)
        jac_rescale_elc = jax.vmap(jac_rescale_fn)(batch_samples)
        novel_correction = 0.5 * jnp.einsum('beneK->bnK', jac_rescale_elc)

        grd_ee = jnp.einsum('beK,ben->bnK', grd_ee_elc, rescale)
        grd_en = grd_en_nuc + jnp.einsum('beK,ben->bnK', grd_en_elc, rescale)
        grd_ke = grd_ke_nuc + jnp.einsum('beK,ben->bnK', grd_ke_elc, rescale)

        grd_logpsi = grd_logpsi_nuc + jnp.einsum('beK,ben->bnK',
                                                 grd_logpsi_elc, rescale)
        grd_logpsi += novel_correction

        return grd_ee, grd_en, grd_ke, grd_logpsi

    # --- Gradient saving ---
    def vmc_gradient_save(block_cnt: int,
                          sampled_walkers: jnp.ndarray,
                          local_energies: jnp.ndarray,
                          batch_size: int, num_batches: int):
        # sampled_walkers enter flattened along the first two axes.
        # ie. num_samples_per_block == num_steps_per_block * num_walkers
        # sampled_walkers.shape
        #   == (num_steps_per_block * num_walkers, nelec, 3)
        # local_energies.shape == (num_steps_per_block, num_walkers)
        num_samples_per_block = sampled_walkers.shape[0]
        w_grd_ee_en = []
        w_grd_ke = []
        w_grd_logpsi = []

        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, num_samples_per_block)

            # batch_samples = sampled_walkers[start_idx:end_idx, :, :]
            grd_ee_en = []
            grd_logpsi = []
            grd_ke = []
            for s_op in symmop_list:
                batch_samples = symmetry_operations_map[s_op](
                    sampled_walkers[start_idx:end_idx, :, :]
                    )
                g_ee, g_en, g_ke, g_logpsi = vmc_gradient_batch(batch_samples)
                grd_ee_en.append(g_ee + g_en)
                grd_ke.append(g_ke)
                grd_logpsi.append(g_logpsi)

            grd_ee_en = jnp.stack(grd_ee_en, axis=0).mean(axis=0)
            grd_ke = jnp.stack(grd_ke, axis=0).mean(axis=0)
            grd_logpsi = jnp.stack(grd_logpsi, axis=0).mean(axis=0)

            w_grd_ee_en.append(grd_ee_en)
            w_grd_ke.append(grd_ke)
            w_grd_logpsi.append(grd_logpsi)

        # Stack all batches
        w_grd_ee_en = jnp.vstack(w_grd_ee_en)
        w_grd_ke = jnp.vstack(w_grd_ke)
        w_grd_logpsi = jnp.vstack(w_grd_logpsi)

        # Save to HDF5
        with h5py.File(chkfile_name_grd, "a") as f:
            if f'grd_ee_en_{block_cnt}' in f.keys():
                del f[f'grd_ee_en_{block_cnt}']
            if f'grd_logpsi_{block_cnt}' in f.keys():
                del f[f'grd_logpsi_{block_cnt}']
            if f'local_energies_{block_cnt}' in f.keys():
                del f[f'local_energies_{block_cnt}']
            if f'grd_ke_{block_cnt}' in f.keys():
                del f[f'grd_ke_{block_cnt}']
            f.create_dataset(f'grd_ee_en_{block_cnt}', data=w_grd_ee_en)
            f.create_dataset(f'grd_logpsi_{block_cnt}', data=w_grd_logpsi)
            f.create_dataset(f'local_energies_{block_cnt}',
                             data=local_energies)
            f.create_dataset(f'grd_ke_{block_cnt}', data=w_grd_ke)

    # --- Main VMC run ---
    def vmc_run(rng_key: int | jnp.ndarray,
                num_walkers: int = 1000,
                num_steps_per_block: int = 100, num_steps_decorr: int = 1,
                num_blocks: int = 1000, num_blocks_equil: int = 100,
                mc_timestep: float = 0.1,
                fname_log: str = None,
                mode_restart: bool = False, l_grad: bool = False) -> None:
        """VMC run with better memory management."""
        # tolerance_enr_std_per_elec=CHEMICAL_ACCURACY,
        if isinstance(rng_key, int):
            rng_key = jax.random.key(rng_key)
        rng_key_to_restart = rng_key.copy()

        # Initialize electron positions more efficiently
        rng_key, init_key = jax.random.split(rng_key)
        walkers = _initialize_walkers(init_key,
                                      num_walkers, nelec,
                                      Z_charges, nuc_crds, mol_charge)
        mc_stepsize = (3 * mc_timestep)**0.5

        # Equilibration phase
        @jax.jit
        def equilibration_step(state, _):
            rng_key, walkers, step_size = state
            rng_key, key = jax.random.split(rng_key)
            walker_keys = jax.random.split(key, num_walkers)
            new_walkers, accepted \
                = metropolis_move_allw(walker_keys, walkers, step_size)

            ratio = accepted.mean()
            new_step_size = _adapt_step_size(step_size, ratio)

            return (rng_key, new_walkers, new_step_size), ratio
        # walkers.shape == (num_walkers, nelec, 3)

        if mode_restart:
            with h5py.File(chkfile_name, 'r') as f:
                # Load metadata
                block_cnt_start = int(f['block_count'][()])
                mc_stepsize = f['mc_stepsize'][()]
                mc_timestep = mc_stepsize * mc_stepsize / 3
                rng_key = jax.random.key(int(f['rng_key'][()]))
                rng_key_to_restart = rng_key.copy()
                rng_key, init_key = jax.random.split(rng_key)
                walkers = jnp.array(f['walkers'][:])
                E_b = list(f['E_blocks'][:])
                print("Restarting ...")

            # for _ in range(num_blocks_equil):
            #     initial_state = (rng_key, walkers, mc_stepsize)
            #     final_state, ratios = jax.lax.scan(equilibration_step,
            #                                        initial_state,
            #                                        jnp.arange(num_steps_per_block))
            #     rng_key, walkers, mc_stepsize = final_state
        else:
            for _ in range(num_blocks_equil):
                initial_state = (rng_key, walkers, mc_stepsize)
                final_state, ratios \
                    = jax.lax.scan(equilibration_step,
                                   initial_state,
                                   jnp.arange(num_steps_per_block))
                rng_key, walkers, mc_stepsize = final_state
            block_cnt_start = 1
            mc_timestep = mc_stepsize * mc_stepsize / 3
            E_b = []
            # std_E_b = []

        ratio = ratios[-1]

        print(f"ℹ️ Equilibration acceptance rate: {ratio:.2f}")
        print(f"ℹ️ Adjusted step size: {mc_stepsize:.4f} bohr "
              f"~ {mc_timestep:.4f} Ha⁻¹ in Brownian time")

        # Production phase
        @jax.jit
        def production_step(state, step_number):
            rng_key, walkers, step_size = state

            for _ in range(num_steps_decorr):
                rng_key, key_displace = jax.random.split(rng_key)
                walker_keys = jax.random.split(key_displace, num_walkers)
                new_walkers, accepted \
                    = metropolis_move_allw(walker_keys, walkers, step_size)
                walkers = new_walkers

            ratio = accepted.mean()
            # new_step_size = step_size * (0.5 + ratio)

            # calculate energy
            # energies = jax.vmap(total_local_energy_fn)(new_walkers)
            enr_ee = jax.vmap(local_energy_ee)(new_walkers)
            enr_en = jax.vmap(local_energy_en,
                              in_axes=(0, None))(new_walkers,
                                                 nuc_crds)
            enr_ke = jax.vmap(local_energy_ke,
                              in_axes=(0, None, None))(new_walkers,
                                                       nuc_crds,
                                                       params_corr)

            return (rng_key, walkers, step_size), \
                (ratio, enr_ee, enr_en, enr_ke, new_walkers)
        # walkers.shape == (num_walkers, nelec, 3)
        # ratios.shape == (num_steps_per_block,)
        # energies.shape == (num_steps_per_block, num_walkers)

        # initial_state = (rng_key, walkers, mc_stepsize)
        # final_state, result = jax.lax.scan(production_step,
        #                                    initial_state,
        #                                    jnp.arange(num_steps_per_block))

        # rng_key, walkers, mc_stepsize = final_state
        # ratios, walkers_energies = result

        if fname_log is None \
                or (isinstance(fname_log, str) and fname_log == ""):
            fout = sys.stdout
        else:
            fout = open(fname_log, 'w', 1)

        # could turn this into a one-liner if requirement becomes Python 3.8+
        p = pathlib.Path(chkfile_name)
        if p.exists():
            p.unlink()
        p = pathlib.Path(chkfile_name_grd)
        if p.exists():
            p.unlink()

        base_batch_size = 500
        memory_factor = max(1, nelec * num_nuc // 1000)
        batch_size = min(50, base_batch_size // memory_factor)
        num_batches = (num_steps_per_block * num_walkers + batch_size - 1) \
            // batch_size
        # mark_samples = ((jnp.arange(num_steps_per_block)+1) == 0)
        print("ℹ️ Adjusted batch size, number of batches: "
              f"{batch_size}, {num_batches}")
        print("# block_cnt        E_loc_mean      E_loc_std"
              "       eePotential     enPotential     Kinetic"
              "          ∆t_block",
              file=fout)

        timestamp_prev = datetime.now()
        # Main sampling phase with pre-allocated arrays
        for block_cnt in range(block_cnt_start,
                               block_cnt_start+num_blocks):
            initial_state = (rng_key, walkers, mc_stepsize)
            final_state, result \
                = jax.lax.scan(production_step,
                               initial_state,
                               jnp.arange(num_steps_per_block))

            rng_key, walkers, _ = final_state
            ratios, enr_ee_sw, enr_en_sw, enr_ke_sw, sampled_walkers = result
            E_loc_sw = enr_ee_sw + enr_en_sw + enr_ke_sw + enr_nn
            # sampled_walkers.shape
            #   == (num_steps_per_block, num_walkers, nelec, 3)
            # E_loc_sw.shape == (num_steps_per_block, num_walkers)

            # mean over walkers
            enr_ee_s = enr_ee_sw.mean(axis=1)
            enr_en_s = enr_en_sw.mean(axis=1)
            enr_ke_s = enr_ke_sw.mean(axis=1)
            E_loc_s = enr_ee_s + enr_en_s + enr_ke_s + enr_nn

            # mean over steps in block
            enr_ee = enr_ee_s.mean()
            enr_en = enr_en_s.mean()
            enr_ke = enr_ke_s.mean()

            E_mean = E_loc_s.mean()             # mean over steps
            std_E_s = E_loc_s.std()             # std over steps

            E_b.append(E_mean)              # append to block data
            # std_E_b.append(std_E_s)

            timestamp_curr = datetime.now()
            tdelta_block = (timestamp_curr - timestamp_prev).total_seconds()
            print(f"{block_cnt:>8d}{E_mean:>24.8e}{std_E_s:>16.8e}"
                  f"{enr_ee:>16.8e}{enr_en:>16.8e}{enr_ke:>16.8e}"
                  f"{tdelta_block:>16.6f}",
                  file=fout)

            if l_grad:
                vmc_gradient_save(block_cnt,
                                  sampled_walkers.reshape(-1, nelec, 3),
                                  E_loc_sw,
                                  batch_size, num_batches)

            # if std_E_s < tolerance_enr_std_per_elec * nelec:
            #     break

            timestamp_prev = timestamp_curr

        if not (fname_log is None
                or (isinstance(fname_log, str) and fname_log == "")):
            fout.close()

        if l_grad:
            with h5py.File(chkfile_name, 'a') as f:
                f.create_dataset('E_blocks', data=E_b)
                f.create_dataset('rng_key',
                                 data=jax.random.key_data(rng_key_to_restart))
                # f.create_dataset('enr_mean', data=enr_mean)
                f.create_dataset('block_count', data=block_cnt,
                                 dtype=jnp.int32)
                f.create_dataset('grd_nn', data=grad_nn_nuc)

        timestamp_fin = datetime.now()
        print("End time: {}\t({:.6f} seconds total)"
              .format(timestamp_fin,
                      (timestamp_fin-timestamp_init).total_seconds()))

    # --- Gradient post-processing ---
    def vmc_gradient_with_space_warping(
            fname_log: str = None,
            compute_errors: bool = False,
            walker_based_batch_size: int = 10
            ) -> jnp.ndarray:
        with h5py.File(chkfile_name, 'r') as f:
            # Load metadata
            # block_cnt_start = int(f['block_count'][()])
            # enr_mean = f['enr_mean'][()]
            enr_mean = f["E_blocks"][:].mean()
            grd_nn = jnp.array(f['grd_nn'][:])

        with h5py.File(chkfile_name_grd, 'r') as f:
            dict_grd_samples = {}
            for key, data in f.items():
                if data.ndim == 0:
                    dict_grd_samples[key] = data[()]    # scalar
                else:
                    dict_grd_samples[key] = jnp.array(data[:])

            # num_blocks = int(dict_grd_samples['num_blocks'])
            grad_list \
                = list(filter(lambda x: x.startswith("grd_logpsi_"), f.keys()))
            block_nums = []
            for g in grad_list:
                block_nums.append(int(g[len("grd_logpsi_"):]))
            block_nums.sort()
            # num_blocks = len(block_nums)
            # block_cnt_start = block_nums[0]
            # enr_mean = dict_grd_samples['enr_mean']
            # enr_std = dict_grd_samples['enr_std']
            # grd_nn = dict_grd_samples['grd_nn']

            valid_samples_count = 0
            grd_ke_sum = 0.0
            grd_ee_en_sum = 0.0
            grd_pulay_sum = 0.0
            if compute_errors:
                grd_tot_list = []
                grd_err_list = []

            for block_cnt in block_nums:
                grd_ee_en = dict_grd_samples[f'grd_ee_en_{block_cnt}']
                grd_logpsi = dict_grd_samples[f'grd_logpsi_{block_cnt}']
                local_energies \
                    = dict_grd_samples[f'local_energies_{block_cnt}']
                grd_ke = jnp.array(f[f'grd_ke_{block_cnt}'][:])

                # Pulay force contribution
                d_enr = local_energies - enr_mean

                # num_samples_per_block, num_nuc, 3 == grd_ee_en.shape
                num_steps_per_block, num_walkers = local_energies.shape
                if compute_errors:
                    # Regroup
                    grd_ee_en = grd_ee_en.reshape(num_steps_per_block,
                                                  num_walkers, num_nuc, 3)
                    grd_ke = grd_ke.reshape(num_steps_per_block,
                                            num_walkers, num_nuc, 3)
                    grd_logpsi = grd_logpsi.reshape(num_steps_per_block,
                                                    num_walkers, num_nuc, 3)
                    grd_pulay = 2.0 * jnp.einsum('sw,swnK->swnK',
                                                 d_enr, grd_logpsi)
                    # grd_pulay = grd_pulay.reshape(num_steps_per_block,
                    #                               num_walkers, num_nuc, 3)

                    grd_nn_sw = jnp.broadcast_to(
                        grd_nn[jnp.newaxis, jnp.newaxis, :, :],
                        (num_steps_per_block, num_walkers, num_nuc, 3)
                        )
                    grd_arrays = [grd_nn_sw,
                                  grd_ee_en, grd_ke,
                                  grd_pulay]
                    grd_tot_sw = jnp.stack(grd_arrays, axis=0).sum(axis=0)

                    # Compute forces and error
                    xbar, serr, sdev, kappa = batched_binning_analysis_grds(
                        grd_tot_sw, walker_based_batch_size
                    )
                    grd_tot_list.append(xbar[None, :, :, :])
                    grd_err_list.append(serr[None, :, :, :])

                    grd_ee_en_sum += grd_ee_en.sum(axis=0)
                    grd_ke_sum += grd_ke.sum(axis=0)
                    grd_pulay_sum += grd_pulay.sum(axis=0)

                    valid_samples_count += local_energies.shape[0]
                    # do not include num_walkers factor
                else:
                    d_enr = d_enr.reshape(-1)
                    grd_pulay = 2.0 * jnp.einsum('s,snK->snK',
                                                 d_enr, grd_logpsi)
                    grd_ee_en_sum += grd_ee_en.sum(axis=0)
                    grd_ke_sum += grd_ke.sum(axis=0)
                    grd_pulay_sum += grd_pulay.sum(axis=0)

                    valid_samples_count += local_energies.reshape(-1).shape[0]
                    # include num_walkers factor

            # Compute averages
            if valid_samples_count > 0:
                grd_ee_en = grd_ee_en_sum / valid_samples_count
                grd_ke = grd_ke_sum / valid_samples_count
                grd_pulay = grd_pulay_sum / valid_samples_count

                if compute_errors:
                    grd_tot_bw = jnp.concatenate(grd_tot_list, axis=0)
                    grd_err_bw = jnp.concatenate(grd_err_list, axis=0)

                    # mean over blocks, then walkers
                    grd_tot = grd_tot_bw.mean(axis=0).squeeze()
                    grd_tot = grd_tot.mean(axis=0)

                    grd_err = jnp.linalg.norm(grd_err_bw, axis=0).squeeze() \
                        / grd_err_bw.shape[0]
                    grd_err = jnp.linalg.norm(grd_err, axis=0) \
                        / grd_err.shape[0]

                    # Compute torques and error
                    torque, dtau \
                        = compute_torque_with_error(mf.mol, grd_tot, grd_err)

                    grd_ee_en = jnp.mean(grd_ee_en, axis=0)
                    grd_ke = jnp.mean(grd_ke, axis=0)
                    grd_pulay = jnp.mean(grd_pulay, axis=0)
                else:
                    grd_tot = grd_nn + grd_ee_en + grd_ke + grd_pulay
                    grd_err = None

                    torque = compute_torque(mf.mol, grd_tot)
            else:
                grd_ee_en = jnp.zeros_like(grd_nn)
                grd_ke = jnp.zeros_like(grd_nn)
                grd_pulay = jnp.zeros_like(grd_nn)

            # Write results
            with jnp.printoptions(precision=12, suppress=True):
                if fname_log is None \
                        or (isinstance(fname_log, str) and fname_log == ""):
                    fout = sys.stdout
                else:
                    fout = open(fname_log, 'w', 1)

                print('NN gradients\n', grd_nn, file=fout)
                print('ee+eN gradients\n', grd_ee_en, file=fout)
                print('KE gradients\n', grd_ke, file=fout)
                print('Pulay gradients\n', grd_pulay, file=fout)
                print('Total gradients\n', grd_tot, file=fout)
                if compute_errors:
                    fout.write("Total forces (-gradients)\n")
                    for i in range(num_nuc):
                        fout.write("{:4s}{:>16.6g} ± {:>12.6g}"
                                   "{:>16.6g} ± {:>12.6g}"
                                   "{:>16.6g} ± {:>12.6g}\n"
                                   .format(mf.mol.atom_symbol(i),
                                           -grd_tot[i, 0], grd_err[i, 0],
                                           -grd_tot[i, 1], grd_err[i, 1],
                                           -grd_tot[i, 2], grd_err[i, 2]))
                    fout.write("Total torque\n")
                    fout.write("    {:>16.6g} ± {:>12.6g}"
                               "{:>16.6g} ± {:>12.6g}"
                               "{:>16.6g} ± {:>12.6g}\n"
                               .format(torque[0], dtau[0],
                                       torque[1], dtau[1],
                                       torque[2], dtau[2]))
                else:
                    fout.write("Total forces (-gradients)\n")
                    for i in range(num_nuc):
                        fout.write("{:4s}{:>16.6g}{:>16.6g}{:>16.6g}\n"
                                   .format(mf.mol.atom_symbol(i),
                                           -grd_tot[i, 0],
                                           -grd_tot[i, 1],
                                           -grd_tot[i, 2]))
                    fout.write("Total torque\n")
                    fout.write("    {:>16.6g}{:>16.6g}{:>16.6g}\n"
                               .format(torque[0], torque[1], torque[2]))
                fout.write("\n")

                if not (fname_log is None
                        or (isinstance(fname_log, str) and fname_log == "")):
                    fout.close()

            return -grd_tot, grd_err

    return vmc_run, vmc_gradient_with_space_warping
