"""Post-sampling SGD VMC optimizer.

This module implements Jastrow-factor optimization via a
two-phase approach:

1. **Sampling phase** — run several production MC blocks
   and store the final walker positions from each block.
2. **Optimization phase** — minimize a combined
   energy-plus-variance loss on the stored snapshots
   using SGD or Adam (via Optax).

Because the gradient is computed only over the stored
snapshots (not through the MC trajectory itself), this
driver is faster and more memory-efficient than the
naïve approach in :mod:`vmcopt_gto_naive`.  For the
best convergence, prefer :mod:`vmcopt_gto_linear`
(the linear method).
"""

import warnings
import jax
import jax.numpy as jnp
import optax
from functools import partial
from .cusp import get_cusp_params
from .psi_gto import get_psi_fun, _sanitize_J3_eeI_params
from .constants import MIN_DIST_THRESHOLD


def _build_opt_mask(params_corr, frozen_keys):
    """Build boolean pytree: True = optimize, False = freeze."""
    if frozen_keys is None:
        return None
    if isinstance(frozen_keys, list):
        frozen_keys = {k: True for k in frozen_keys}

    mask = {}
    for k, v in params_corr.items():
        if k not in frozen_keys:
            if isinstance(v, dict):
                mask[k] = {k2: jnp.ones_like(v2, dtype=bool)
                           for k2, v2 in v.items()}
            else:
                mask[k] = jnp.ones_like(v, dtype=bool)
        elif frozen_keys[k] is True:
            if isinstance(v, dict):
                mask[k] = {k2: jnp.zeros_like(v2, dtype=bool)
                           for k2, v2 in v.items()}
            else:
                mask[k] = jnp.zeros_like(v, dtype=bool)
        elif isinstance(frozen_keys[k], dict):
            mask[k] = {}
            for k2, v2 in v.items():
                if k2 not in frozen_keys[k]:
                    mask[k][k2] = jnp.ones_like(v2, dtype=bool)
                else:
                    m = jnp.ones_like(v2, dtype=bool)
                    for idx in frozen_keys[k][k2]:
                        m = m.at[idx].set(False)
                    mask[k][k2] = m
    return mask


def _zero_frozen_grads(mask):
    """Optax transformation that zeros gradients for frozen elements."""
    def init_fn(params):
        return optax.EmptyState()

    def update_fn(updates, state, params=None):
        return jax.tree.map(
            lambda u, m: jnp.where(m, u, jnp.zeros_like(u)),
            updates, mask), state

    return optax.GradientTransformation(init_fn, update_fn)


def _init_params_corr(params_corr_init):
    """Convert params_corr_init dict to jnp arrays, or return empty dict."""
    if params_corr_init is None:
        return dict()
    params_corr = {}
    for k, v in params_corr_init.items():
        if isinstance(v, dict):
            params_corr[k] = {
                k2: jnp.asarray(v2, dtype=jnp.float64)
                for k2, v2 in v.items()
            }
        else:
            params_corr[k] = jnp.array(
                v, dtype=jnp.float64
            )
    if "J3_eeI" in params_corr:
        params_corr["J3_eeI"] = (
            _sanitize_J3_eeI_params(
                params_corr["J3_eeI"]
            )
        )
    return params_corr


def _check_j2_cusps(params_corr, eps):
    """Warn if J2 cusp coefficients violate cusp conditions."""
    if "J2_pade" not in params_corr:
        return
    j2 = params_corr["J2_pade"]
    if "like" in j2 and abs(float(j2["like"][0]) - 0.25) > eps:
        warnings.warn(
            f"J2_pade['like'][0] = {float(j2['like'][0]):.8f}, "
            "expected 0.25 (same-spin cusp condition)")
    if "unlike" in j2 and abs(float(j2["unlike"][0]) - 0.5) > eps:
        warnings.warn(
            f"J2_pade['unlike'][0] = {float(j2['unlike'][0]):.8f}, "
            "expected 0.5 (opposite-spin cusp condition)")


class _VMCOptDriver:
    """Post-sampling SGD VMC optimizer.

    Compiles the Metropolis kernel and local-energy
    function for a given molecule, collects walker
    snapshots in a sampling phase, then optimizes
    Jastrow parameters on those snapshots with SGD
    or Adam.
    """

    def __init__(self, mf, params_cusp,
                 jastrow_config=None):
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
                          jastrow_config=jastrow_config)
        local_energy_ee, local_energy_nn, local_energy_en, local_energy_ke \
            = local_energy
        enr_nn = local_energy_nn(nuc_crds)

        @jax.jit
        def metropolis_move(rng_key, elec_crds, _step_size, curr_params):
            """Metropolis step with improved distance calculations."""
            key_prop, key_accept = jax.random.split(rng_key)

            proposed_crds = elec_crds \
                + _step_size * jax.random.normal(key_prop, elec_crds.shape)

            diffs_ee = proposed_crds[i_e] - proposed_crds[j_e]
            dists_ee = jnp.linalg.norm(diffs_ee, axis=-1)

            diffs_en = proposed_crds[:, None, :] - nuc_crds[None, :, :]
            dists_en = jnp.linalg.norm(diffs_en, axis=-1)

            valid_move = (dists_en.min() > MIN_DIST_THRESHOLD) \
                & (dists_ee.min() > MIN_DIST_THRESHOLD)

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
            """Compute total local energy."""
            return (local_energy_ee(elec_crds)
                    + local_energy_en(elec_crds, nuc_crds)
                    + local_energy_ke(elec_crds, nuc_crds, curr_params)
                    + enr_nn)

        @partial(jax.jit, static_argnums=(4, 5))
        def run_equilibration(rng_key, walkers, step_size, params_corr,
                              num_be, num_spb):
            """Equilibration phase with adaptive step size."""

            @jax.jit
            def equilibration_step(carried_in, _):
                rkey, w, s, curr_params = carried_in
                rkey0, rkey1 = jax.random.split(rkey)
                keys = jax.random.split(rkey1, w.shape[0])

                new_w, accepted \
                    = jax.vmap(metropolis_move,
                               in_axes=(0, 0, None, None))(keys, w, s,
                                                           curr_params)
                acceptance_rate = accepted.mean()
                new_s = s * (0.6 + acceptance_rate)

                return (rkey0, new_w, new_s, curr_params), acceptance_rate

            for _ in range(num_be):
                carry_in = (rng_key, walkers, step_size, params_corr)
                carry_out, acc_ratios \
                    = jax.lax.scan(equilibration_step, carry_in,
                                   jnp.arange(num_spb))

            return carry_out, acc_ratios

        @partial(jax.jit, static_argnums=(4, 5))
        def run_production(rng_key, walkers, step_size, params_corr,
                           num_spb, num_dc):
            """Production phase with sampling."""

            @jax.jit
            def production_step(carried_in, step_idx):
                rkey, w, s, curr_params = carried_in

                for _ in range(num_dc):
                    rkey0, rkey1 = jax.random.split(rkey)
                    keys = jax.random.split(rkey1, w.shape[0])

                    new_w, accepted \
                        = jax.vmap(metropolis_move,
                                   in_axes=(0, 0, None, None))(keys, w, s,
                                                               curr_params)
                    w = new_w

                r = accepted.mean()

                energies = jax.vmap(total_local_energy_fn,
                                    in_axes=(0, None))(new_w,
                                                       curr_params)

                return (rkey0, new_w, s, curr_params), (r, energies)

            carry_in = (rng_key, walkers, step_size, params_corr)
            carried_out, results \
                = jax.lax.scan(production_step, carry_in, jnp.arange(num_spb))

            return carried_out, results

        @jax.jit
        def compute_batch_energy(walkers, params):
            """Compute energy for a batch of walkers."""
            return jax.vmap(total_local_energy_fn,
                            in_axes=(0, None))(walkers, params)

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
                batch_energies \
                    = compute_batch_energy(walkers[start_idx:end_idx], params)
                energies.append(batch_energies.mean())

            return jnp.array(energies).mean()

        self.run_equilibration = run_equilibration
        self.run_production = run_production
        self.compute_batch_energy = compute_batch_energy
        self.loss_fn = loss_fn
        self.compute_statistics = compute_statistics

    def initialize_walkers(self, rng_key, num_walkers):
        """Initialize electron positions near nuclei."""
        idx_cnt = []
        for ia, iz in enumerate(self.Z_charges):
            idx_cnt.extend([ia] * int(iz))

        if self.mf.mol.charge < 0:
            idx_cnt.extend([0] * abs(self.mf.mol.charge))
        elif self.mf.mol.charge > 0:
            idx_cnt = idx_cnt[:-self.mf.mol.charge]

        idx_cnt = jnp.array(idx_cnt)
        centers = self.nuc_crds[idx_cnt]
        walkers = centers[jnp.newaxis, :, :] \
            + 0.05 * jax.random.normal(rng_key, (num_walkers, self.nelec, 3))

        return walkers

    def __call__(self,
                 rng_key,
                 params_corr_init=None,
                 frozen_keys=None,
                 num_epochs=20,
                 num_walkers=1000,
                 num_steps_per_block=1000,
                 num_steps_decorr=1,
                 num_blocks=10,
                 num_blocks_equil=10,
                 mc_timestep=0.1,
                 lr=0.02,
                 optimizer="sgd",
                 train_split=0.8,
                 batch_size=1000,
                 verbose=1):
        """
        VMC optimization run with improved efficiency.

        Args:
            rng_key: JAX random key
            num_walkers: Number of walkers
            params_corr_init: Initial parameters (if None, use empty dict)
            num_steps_per_block: Steps per production block
            num_epochs: Number of training epochs
            num_blocks_equil: Equilibration blocks
            mc_timestep: Initial MC timestep
            lr: Learning rate
            optimizer: "sgd" or "adam"
            num_blocks: Number of production blocks
            train_split: Fraction of data for training
            batch_size: Batch size for training
            verbose: Verbosity level (0 = silent, 1 = progress)
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

        rng_key, rng = jax.random.split(rng_key)
        walkers = self.initialize_walkers(rng, num_walkers)
        mc_stepsize = (3 * mc_timestep)**0.5

        if verbose >= 1:
            print("Running equilibration...")
        (rng_key, walkers, mc_stepsize, _), acc_ratios \
            = self.run_equilibration(rng_key, walkers, mc_stepsize,
                                     params_corr,
                                     num_blocks_equil, num_steps_per_block)

        if verbose >= 1:
            print(f"Equilibration acceptance rate: {acc_ratios[-1]:.2f}")
            print("Adjusted step size: {:.4f} bohr "
                  "~ {:.4f} Ha⁻¹ in Brownian time"
                  .format(mc_stepsize, mc_stepsize * mc_stepsize / 3))

        if verbose >= 1:
            print(f"\nRunning {num_blocks} production blocks...")

        all_samples = []
        for block_cnt in range(1, num_blocks+1):
            (rng_key, walkers, mc_stepsize, _), (acc_ratios, tw_energies) \
                = self.run_production(rng_key, walkers, mc_stepsize,
                                      params_corr,
                                      num_steps_per_block, num_steps_decorr)

            all_samples.append(walkers)

            if verbose >= 1:
                print(f"  Run {block_cnt}/{num_blocks}: "
                      f"collected {walkers.shape[0]} samples")

        sampled_walkers = jnp.vstack(all_samples).reshape(-1, self.nelec, 3)
        n_samples = sampled_walkers.shape[0]

        rng_key, rng_key1 = jax.random.split(rng_key)
        idx = jax.random.permutation(rng_key1, jnp.arange(n_samples))

        n_train = int(train_split * n_samples)
        train_walkers = sampled_walkers[idx[:n_train]]
        valid_walkers = sampled_walkers[idx[n_train:]]

        if verbose >= 1:
            print(f"\nTraining on {n_train} samples, "
                  f"validating on {n_samples-n_train} samples")
            if frozen_keys:
                print(f"Frozen parameters: {frozen_keys}")
            print(f"\nStarting optimization for {num_epochs} epochs...\n")

        for epoch in range(num_epochs):
            epoch_losses = []
            for start_idx in range(0, n_train, batch_size):
                end_idx = min(start_idx + batch_size, n_train)
                batch = train_walkers[start_idx:end_idx]

                loss, grad_mean \
                    = jax.value_and_grad(self.loss_fn)(params_corr, batch)
                updates, opt_state \
                    = optimizer_chosen.update(grad_mean, opt_state,
                                              params_corr)
                params_corr = optax.apply_updates(params_corr, updates)

                epoch_losses.append(loss)

            train_loss = jnp.array(epoch_losses).mean()
            train_energy = self.compute_statistics(train_walkers, params_corr,
                                                   batch_size)
            valid_energy = self.compute_statistics(valid_walkers, params_corr,
                                                   batch_size)

            valid_losses = []
            for start_idx in range(0, valid_walkers.shape[0], batch_size):
                end_idx = min(start_idx + batch_size, valid_walkers.shape[0])
                valid_losses.append(
                    self.loss_fn(params_corr, valid_walkers[start_idx:end_idx])
                    )
            valid_loss = jnp.array(valid_losses).mean()

            if verbose >= 1:
                print(f"Epoch {epoch:3d} | "
                      f"Loss: {train_loss:.6f} | "
                      f"Train E: {train_energy:.6f} | "
                      f"Valid Loss: {valid_loss:.6f} | "
                      f"Valid E: {valid_energy:.6f}")

        if verbose >= 1:
            print(f"\nOptimized parameters: {params_corr}")

        return params_corr, {'energy': {'mean': valid_energy}}


def get_vmcopt_gto_func(mf, cusp_scheme="Quady2025",
                    jastrow_config=None):
    """Create a post-sampling SGD VMC optimizer.

    Builds cusp-corrected trial wave-function parameters
    and returns a :class:`_VMCOptDriver` that optimizes
    Jastrow coefficients by minimizing a combined
    energy-plus-variance loss on stored walker snapshots
    using SGD or Adam.

    Parameters
    ----------
    mf : pyscf.scf.RHF
        Converged mean-field object as returned by
        :func:`generate_molecular_orbitals`.
    cusp_scheme : str or None, optional
        Cusp-correction scheme.  ``"Quady2025"``
        (default) applies the scheme from Quady *et al.*
        (2025).  Pass ``None`` to skip cusp corrections.
    jastrow_config : dict or None, optional
        Cutoff radii for B-spline Jastrow factors.
        Example::

            {"J1": {"H": {"r_cut": 5.0}},
             "J2": {"r_cut": 10.0}}

    Returns
    -------
    driver : _VMCOptDriver
        A callable optimizer.  Call it with
        ``driver(rng_key, ...)`` to run the optimization
        loop and obtain optimized Jastrow parameters.
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
                    p = get_cusp_params(atom_symbol,
                                        mf.mol.basis[atom_symbol])
                params_cusp[atom_symbol] = p[atom_symbol]
    else:
        params_cusp = None
    return _VMCOptDriver(mf, params_cusp,
                         jastrow_config=jastrow_config)
