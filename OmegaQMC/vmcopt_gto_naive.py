"""Naïve (serial) VMC optimizer.

This module provides a reference implementation of the
VMC Jastrow-factor optimizer that differentiates through
the full production MC trajectory at each epoch.
The gradient of a combined energy-plus-variance loss is
computed via JAX automatic differentiation.

This approach is exact but memory-intensive because JAX
must retain the entire trajectory for backpropagation
(hence ``jax.checkpoint`` in the inner loop).  For
production use, prefer :mod:`vmcopt_gto_pssgd` (which
samples first, then optimizes on the stored snapshots)
or :mod:`vmcopt_gto_linear` (the linear method).
"""

import jax
import jax.numpy as jnp
import optax
from .psi_gto import get_psi_fun
from .cusp import get_cusp_params
from .constants import MIN_DIST_THRESHOLD
from .vmcopt_gto_pssgd import (
    _build_opt_mask,
    _zero_frozen_grads,
    _init_params_corr,
    _check_j2_cusps,
)
from .utils import _make_sharding


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


class _VMCOptNaiveDriver:
    """Naïve VMC optimizer — differentiates through MC.

    Compiles the Metropolis kernel and local-energy
    function for a given molecule, then at each epoch
    differentiates through the entire production MC
    trajectory to compute parameter gradients.
    """

    def __init__(self, mf, params_cusp,
                 bspline_config=None):
        nuc_crds = jnp.array(mf.mol.atom_coords(unit='Bohr'))
        eps = jnp.finfo(nuc_crds.dtype).eps
        nelec = mf.mol.tot_electrons()
        Z_charges = mf.mol.atom_charges()
        i_e, j_e = jnp.triu_indices(nelec, k=1)

        self.mf = mf
        self.nuc_crds = nuc_crds
        self.eps = eps
        self.nelec = nelec
        self.Z_charges = Z_charges

        log_trial_wavefunction, local_energy, _, _ \
            = get_psi_fun(mf, params_cusp=params_cusp,
                          bspline_config=bspline_config)
        local_energy_ee, local_energy_nn, local_energy_en, local_energy_ke \
            = local_energy
        enr_nn = local_energy_nn(nuc_crds)

        @jax.jit
        def metropolis_move(rng_key, elec_crds, _step_size, curr_params):
            """Metropolis step."""
            key_prop, key_accept = jax.random.split(rng_key)

            proposed_crds = elec_crds \
                + _step_size * jax.random.normal(key_prop, elec_crds.shape)

            diffs_ee = proposed_crds[i_e] - proposed_crds[j_e]
            dists_ee = jnp.linalg.norm(diffs_ee, axis=-1)

            diffs_en = proposed_crds[:, None, :] - nuc_crds[None, :, :]
            dists_en = jnp.linalg.norm(diffs_en, axis=-1)

            valid_move = (dists_en.min() > MIN_DIST_THRESHOLD) & \
                (dists_ee.min() > MIN_DIST_THRESHOLD)

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
                    + enr_nn)

        self.metropolis_move = metropolis_move
        self.total_local_energy_fn = total_local_energy_fn

    def initialize_walkers(self, rng_key, num_walkers):
        """Initialize electron positions near nuclei."""
        idx_cnt = []
        for ia, iz in enumerate(self.Z_charges):
            idx_cnt.extend([ia] * iz)
        if self.mf.mol.charge < 0:
            idx_cnt.extend([0] * abs(self.mf.mol.charge))
        elif self.mf.mol.charge > 0:
            idx_cnt = idx_cnt[:-self.mf.mol.charge]
        idx_cnt = jnp.array(idx_cnt)
        centers = self.nuc_crds[idx_cnt]
        return centers[jnp.newaxis, :, :] \
            + 0.05 * jax.random.normal(rng_key, (num_walkers, self.nelec, 3))

    def __call__(self, rng_key,
                 params_corr_init=None, frozen_keys=None,
                 num_epochs=20, num_walkers=1000,
                 num_steps_per_block=1000,
                 num_steps_decorr=1,
                 num_blocks=10, num_blocks_equil=10,
                 mc_timestep=0.1, fname_log=None,
                 lr=0.02, optimizer="sgd",
                 verbose=False):
        """Run the naïve VMC optimization loop.

        Parameters
        ----------
        rng_key : jax.Array
            JAX random key.
        params_corr_init : dict or None
            Initial Jastrow parameters.
        frozen_keys : dict or None
            Parameters to keep frozen.
        num_epochs : int
            Number of gradient-descent epochs.
        num_walkers : int
            Number of MC walkers.
        num_steps_per_block : int
            MC steps per production block.
        num_steps_decorr : int
            Decorrelation sub-steps per block.
        num_blocks : int
            Number of production blocks per epoch.
        num_blocks_equil : int
            Equilibration blocks (per epoch).
        mc_timestep : float
            Initial MC timestep.
        fname_log : str or None
            Unused; retained for API compatibility.
        lr : float
            Optimizer learning rate.
        optimizer : str
            ``"sgd"`` or ``"adam"``.
        verbose : bool
            Print progress.

        Returns
        -------
        params_corr : dict
            Optimized Jastrow parameters.
        stats : dict
            Energy statistics from the final epoch.
        """

        params_corr = _init_params_corr(params_corr_init)
        _check_j2_cusps(params_corr, self.eps)

        base_optimizer = optax.adam(learning_rate=lr) \
            if "adam" in optimizer.lower() \
            else optax.sgd(learning_rate=lr)
        mask = _build_opt_mask(params_corr, frozen_keys)
        if mask is not None:
            optimizer_chosen = optax.chain(
                _zero_frozen_grads(mask), base_optimizer)
        else:
            optimizer_chosen = base_optimizer
        opt_state = optimizer_chosen.init(params_corr)

        walkers_sharding, walker_keys_sharding = _make_sharding(num_walkers)

        rng_key, rng = jax.random.split(rng_key)
        walkers = self.initialize_walkers(rng, num_walkers)
        if walkers_sharding is not None:
            walkers = jax.device_put(walkers, walkers_sharding)
        mc_stepsize = (3 * mc_timestep)**0.5

        @jax.jit
        def equilibration_step(carried_in, _):
            rkey, w, s, curr_params = carried_in
            rkey0, rkey1 = jax.random.split(rkey)
            keys = jax.random.split(rkey1, num_walkers + 1)
            rkey1 = keys[0]
            keys = keys[1:]
            if walker_keys_sharding is not None:
                keys = jax.lax.with_sharding_constraint(
                    keys, walker_keys_sharding)

            new_w, accepted \
                = jax.vmap(self.metropolis_move,
                           in_axes=(0, 0, None, None))(keys, w, s,
                                                       curr_params)
            r = accepted.mean()
            new_s = s * (0.6 + r)

            return (rkey0, new_w, new_s, curr_params), r

        for _ in range(num_blocks_equil):
            carry_in = (rng_key, walkers, mc_stepsize, params_corr)
            carry_out, acc_ratios \
                = jax.lax.scan(equilibration_step, carry_in,
                               jnp.arange(num_steps_per_block))
            rng_key, walkers, mc_stepsize, _ = carry_out

        mc_timestep = mc_stepsize * mc_stepsize / 3
        ratio = acc_ratios[-1]

        print(f"ℹ️ Equilibration acceptance rate: {ratio:.2f}")
        print(f"ℹ️ Adjusted step size: {mc_stepsize:.4f} bohr "
              f"~ {mc_timestep:.4f} Ha⁻¹ in Brownian time")

        @jax.jit
        def production_step(carried_in, _):
            rkey, w, s, curr_params = carried_in

            for _ in range(num_steps_decorr):
                rkey0, rkey1 = jax.random.split(rkey)
                keys = jax.random.split(rkey1, num_walkers + 1)
                rkey1 = keys[0]
                keys = keys[1:]
                if walker_keys_sharding is not None:
                    keys = jax.lax.with_sharding_constraint(
                        keys, walker_keys_sharding)
                new_w, accepted \
                    = jax.vmap(self.metropolis_move,
                               in_axes=(0, 0, None, None))(keys, w, s,
                                                           curr_params)
                w = new_w

            r = accepted.mean()
            # new_s = step_size * (0.6 + r)

            # calculate energy
            energies = jax.vmap(self.total_local_energy_fn,
                                in_axes=(0, None))(new_w,
                                                   curr_params)

            return (rkey0, new_w, s, curr_params), (r, energies)

        def update_epoch(carried_in, epoch):
            # nonlocal tw_energies
            nonlocal mc_stepsize
            rng_key, walkers, \
                params_corr_epoch, opt_state = carried_in

            for t in range(num_blocks_equil):
                carry_in_eq = (rng_key, walkers, mc_stepsize,
                               params_corr_epoch)
                carried_out_eq, acc_ratios \
                    = jax.lax.scan(equilibration_step, carry_in_eq,
                                   jnp.arange(num_steps_per_block))
                rng_key, walkers, mc_stepsize, _ = carried_out_eq

            def loss_fn(p):
                nonlocal rng_key, walkers
                for block_cnt in range(1, num_blocks+1):
                    carry_in_prod = (rng_key, walkers, mc_stepsize, p)
                    carried_out_prod, (acc_ratios, energies_sw) \
                        = jax.lax.scan(jax.checkpoint(production_step),
                                       carry_in_prod,
                                       jnp.arange(num_steps_per_block))
                    rng_key, walkers, _, _ = carried_out_prod

                E_s = energies_sw.mean(axis=1)      # mean over walkers

                E_mean = E_s.mean()                 # mean over steps
                std_E_s = E_s.std()                 # std over steps

                loss = 0.2 * E_mean + 0.8 * std_E_s

                return loss, E_mean

            (loss_val, enr_mean), grad_mean \
                = jax.value_and_grad(loss_fn, has_aux=True)(params_corr_epoch)

            updates, opt_state = optimizer_chosen.update(grad_mean,
                                                         opt_state,
                                                         params_corr_epoch)
            params_corr_epoch = optax.apply_updates(params_corr_epoch, updates)

            jax.debug.print("[Epoch {epoch}/{ne}] "
                            "loss: {l:.6f}, <E_loc>: {e:.6f}, "
                            "params_corr_epoch: {vp}",
                            epoch=epoch+1, ne=num_epochs,
                            l=loss_val, e=enr_mean, vp=params_corr_epoch)

            carry_in_final = (rng_key, walkers, mc_stepsize, params_corr_epoch)
            carried_out_final, (_, final_energies) \
                = jax.lax.scan(production_step, carry_in_final,
                               jnp.arange(num_steps_per_block))
            rng_key, walkers, _, _ = carried_out_final

            carry_out = (rng_key, walkers,
                         params_corr_epoch, opt_state)
            return carry_out, final_energies

        carry_epoch = (rng_key, walkers, params_corr, opt_state)
        energies_list = []
        for epoch in range(num_epochs):
            carry_epoch, final_energies = update_epoch(carry_epoch, epoch)
            final_energies.block_until_ready()
            energies_list.append(final_energies)
        energies_opthist = jnp.stack(energies_list)

        rng_key, walkers, params_corr, opt_state = carry_epoch

        energies_sw = energies_opthist[-1]

        E_mean = energies_sw.mean()
        std_E_w = jnp.std(energies_sw, axis=0)

        acf_axis1 = jax.jit(jax.vmap(_acf_fft_jnp, in_axes=1))
        w_acf = acf_axis1(energies_sw)
        tau_int_axis0 = jax.jit(jax.vmap(_tau_int_from_acf, in_axes=0))
        w_tau_int = tau_int_axis0(w_acf)
        ste_E_w = std_E_w * jnp.sqrt(w_tau_int / num_steps_per_block)
        E_stderr = jnp.sqrt(jnp.square(ste_E_w).sum()) / num_walkers

        return params_corr, {'energy': {'mean': E_mean, 'stderr': E_stderr}}


def get_vmcopt_func(mf, cusp_scheme="Quady2025",
                    bspline_config=None):
    """Create a naïve VMC optimizer.

    Builds cusp-corrected trial wave-function parameters
    and returns a :class:`_VMCOptNaiveDriver` that
    optimizes Jastrow coefficients by differentiating
    through the entire MC trajectory at each epoch.

    Parameters
    ----------
    mf : pyscf.scf.RHF
        Converged mean-field object.
    cusp_scheme : str or None
        ``"Quady2025"`` (default) or ``None``.
    bspline_config : dict or None
        B-spline Jastrow cutoff radii.

    Returns
    -------
    driver : _VMCOptNaiveDriver
        Callable optimizer.
    """
    num_nuc = mf.mol.natm
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
    return _VMCOptNaiveDriver(
        mf, params_cusp,
        bspline_config=bspline_config
    )
