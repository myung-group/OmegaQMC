"""Pfau et al. 2024 NES-VMC for K=2 excited states (Science 385, 6711).

This driver implements the natural-excited-states variational Monte
Carlo (NES-VMC) approach for two states (K=2). The K-state
wavefunction is the determinant of single-state ansatze evaluated
at K independent configurations,

    Psi(x^1, x^2) = det [[psi_1(x^1), psi_2(x^1)],
                         [psi_1(x^2), psi_2(x^2)]]
                  = psi_1(x^1) psi_2(x^2) - psi_1(x^2) psi_2(x^1).

Joint walkers (x^1, x^2) are sampled from |Psi(x^1, x^2)|^2. At each
walker, the matrix local energy is

    E_Psi = M^{-1} (H M)

where M[i,j] = psi_i(x^j) and (HM)[i,j] = H[psi_i](x^j); writing
local_E[i,j] = (HM)[i,j] / M[i,j] gives HM = M (elementwise *) local_E,
so

    E_Psi = M^{-1} (M (*) local_E),
    loss  = Tr(E_Psi).

For K=2 specifically,
    Tr(E_Psi) = (a d (e11 + e22) - b c (e12 + e21)) / (a d - b c),
where M = [[a,b],[c,d]] and e_ij = local_E[i,j].

Orthogonality is enforced *structurally* by the determinantal form:
if psi_1 = psi_2, M is rank-1 and Psi = 0 everywhere, so the
sampling distribution collapses to zero and the energy explodes.
There are no penalty terms, no hyperparameters, no scale issues.

Reference: Pfau, Axelrod, Sutterud, von Glehn, Spencer, ``Accurate
computation of quantum excited states with neural networks'',
Science 385, 6711 (2024); arXiv:2308.16848.
"""

import os
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import optax

from .vmcopt_nn_iradam import _VMCOptDriverNN_IRAdam
from .psi.nn.checkpoint import save_nn_checkpoint, load_nn_checkpoint

MIN_DIST_THRESHOLD = 1e-4


class _VMCOptDriverNN_Pfau_K2:
    """Pfau-NES driver for K=2 states (one ground + one excited).

    The driver instantiates two independent single-state drivers (one
    per state) but maintains a *joint* walker bank of shape
    ``(num_walkers, 2, nelec, 3)`` and a custom MCMC sampler that
    moves in the joint configuration space with acceptance ratio
    ``|Psi_new(x_a, x_b) / Psi_old(x_a, x_b)|^2``.

    The two state ansatze share their architecture (config_name)
    but have independent parameter sets. Both parameter sets are
    optimised jointly by Adam on the trace of the matrix local
    energy.
    """

    def __init__(
        self,
        mol_info,
        config,
        init_key_1,
        init_key_2,
    ):
        # State 1
        self.driver_1 = _VMCOptDriverNN_IRAdam(mol_info, config, init_key_1)
        # State 2 with a different init key
        self.driver_2 = _VMCOptDriverNN_IRAdam(mol_info, config, init_key_2)

        self.mol_info = mol_info
        self.config_name = self.driver_1.config_name
        self.nelec = self.driver_1.nelec
        self.nuc_crds = self.driver_1.nuc_crds
        self.params_1 = self.driver_1.init_params
        self.params_2 = self.driver_2.init_params

        log_psi_1 = self.driver_1.log_psi
        log_psi_2 = self.driver_2.log_psi
        log_psi_1_signed = log_psi_1.signed
        log_psi_2_signed = log_psi_2.signed
        nuc_crds = self.nuc_crds

        def signed_psi_1(x, params):
            sgn, lam = log_psi_1_signed(x, nuc_crds, params)
            return sgn * jnp.exp(lam)

        def signed_psi_2(x, params):
            sgn, lam = log_psi_2_signed(x, nuc_crds, params)
            return sgn * jnp.exp(lam)
        self._signed_psi_1 = signed_psi_1
        self._signed_psi_2 = signed_psi_2

        e_local_1 = self.driver_1.compute_batch_energy  # vmap'd
        e_local_2 = self.driver_2.compute_batch_energy

        # Single-walker local energy for state 1 and 2
        # Reach into driver_1/2 for the scalar local energy function
        nuc_crds_jx = nuc_crds
        i_e, j_e = jnp.triu_indices(self.nelec, k=1)
        charges = self.driver_1.charges
        enr_nn_jx = jnp.asarray(0.0)
        n_nuc = len(charges)
        enr = 0.0
        for a in range(n_nuc):
            for b in range(a + 1, n_nuc):
                rab = float(jnp.linalg.norm(nuc_crds[a] - nuc_crds[b]))
                enr += float(charges[a]) * float(charges[b]) / rab
        enr_nn_jx = jnp.asarray(enr, dtype=jnp.float64)

        lap_grad_1 = self.driver_1.lap_grad
        lap_grad_2 = self.driver_2.lap_grad

        @jax.jit
        def _energy_ee_en(elec_crds):
            d_ee = elec_crds[i_e] - elec_crds[j_e]
            d_en = elec_crds[:, None, :] - nuc_crds_jx[None, :, :]
            return (jnp.sum(1.0 / jnp.linalg.norm(d_ee, axis=-1))
                    - jnp.sum(charges[None, :]
                              / jnp.linalg.norm(d_en, axis=-1)))

        @jax.jit
        def local_E_1(elec, p):
            lap, grad = lap_grad_1(elec, nuc_crds_jx, p)
            return _energy_ee_en(elec) - 0.5 * (
                lap + jnp.dot(grad, grad)
            ) + enr_nn_jx

        @jax.jit
        def local_E_2(elec, p):
            lap, grad = lap_grad_2(elec, nuc_crds_jx, p)
            return _energy_ee_en(elec) - 0.5 * (
                lap + jnp.dot(grad, grad)
            ) + enr_nn_jx
        self._local_E_1 = local_E_1
        self._local_E_2 = local_E_2

        # --- Joint Psi (K=2 determinant) ---
        @jax.jit
        def joint_psi(x_a, x_b, p1, p2):
            """det [[psi_1(x_a), psi_2(x_a)], [psi_1(x_b), psi_2(x_b)]]
            evaluated at a single (x_a, x_b) walker pair.
            """
            a = signed_psi_1(x_a, p1)
            b = signed_psi_2(x_a, p2)
            c = signed_psi_1(x_b, p1)
            d = signed_psi_2(x_b, p2)
            return a * d - b * c
        self.joint_psi = joint_psi

        @jax.jit
        def log_abs_joint_psi(x_a, x_b, p1, p2):
            val = joint_psi(x_a, x_b, p1, p2)
            return jnp.log(jnp.abs(val) + 1e-300)
        self.log_abs_joint_psi = log_abs_joint_psi

        # --- Joint Metropolis move (move both x_a and x_b in one step) ---
        @jax.jit
        def joint_metropolis(rng_key, x_a, x_b, step_size, p1, p2):
            key_prop, key_accept = jax.random.split(rng_key)
            key_a, key_b = jax.random.split(key_prop)
            xa_new = x_a + step_size * jax.random.normal(key_a, x_a.shape)
            xb_new = x_b + step_size * jax.random.normal(key_b, x_b.shape)
            # Validity: all electron-nucleus and electron-electron
            # distances must exceed the threshold
            def is_valid(elec):
                d_en = elec[:, None, :] - nuc_crds_jx[None, :, :]
                d_ee = elec[i_e] - elec[j_e]
                return ((jnp.linalg.norm(d_en, axis=-1).min()
                         > MIN_DIST_THRESHOLD)
                        & (jnp.linalg.norm(d_ee, axis=-1).min()
                           > MIN_DIST_THRESHOLD))
            valid = is_valid(xa_new) & is_valid(xb_new)
            lp_old = log_abs_joint_psi(x_a, x_b, p1, p2)
            lp_new = log_abs_joint_psi(xa_new, xb_new, p1, p2)
            accept = ((jax.random.uniform(key_accept)
                       < jnp.exp(2.0 * (lp_new - lp_old)))
                      & valid)
            x_a_out = jnp.where(accept, xa_new, x_a)
            x_b_out = jnp.where(accept, xb_new, x_b)
            return x_a_out, x_b_out, accept
        self._joint_metropolis = joint_metropolis

        # --- Sweeps ---
        @partial(jax.jit, static_argnums=(4,))
        def joint_sweep(rng_key, walkers, step_size, params_pair, num_steps):
            """walkers shape (N, 2, nelec, 3). params_pair = (p1, p2)."""
            p1, p2 = params_pair

            def step(carry, _):
                rk, wlk, ss = carry
                rk0, rk1 = jax.random.split(rk)
                keys = jax.random.split(rk1, wlk.shape[0])
                xa = wlk[:, 0]
                xb = wlk[:, 1]
                xa_new, xb_new, acc = jax.vmap(
                    joint_metropolis,
                    in_axes=(0, 0, 0, None, None, None),
                )(keys, xa, xb, ss, p1, p2)
                wlk_new = jnp.stack([xa_new, xb_new], axis=1)
                ar = acc.mean()
                ss_new = ss * (0.6 + ar)
                return (rk0, wlk_new, ss_new), ar

            carry, _ = jax.lax.scan(
                step, (rng_key, walkers, step_size),
                jnp.arange(num_steps),
            )
            return carry
        self.joint_sweep = joint_sweep

        # --- Matrix local energy + trace loss ---
        @jax.jit
        def trace_loss_one_walker(x_a, x_b, p1, p2):
            """Single-walker contribution to Tr(M^{-1}(M (*) local_E)).

            For K=2:
              det(M) = a d - b c
              Tr = (a d (e11 + e22) - b c (e12 + e21)) / det(M)
            where e_ij = local_E_i(x_j, p_i).
            """
            a = signed_psi_1(x_a, p1)
            b = signed_psi_2(x_a, p2)
            c = signed_psi_1(x_b, p1)
            d = signed_psi_2(x_b, p2)
            ad = a * d
            bc = b * c
            det = ad - bc
            e_11 = local_E_1(x_a, p1)   # state 1 at config x_a
            e_22 = local_E_2(x_b, p2)   # state 2 at config x_b
            e_12 = local_E_2(x_a, p2)   # state 2 at config x_a
            e_21 = local_E_1(x_b, p1)   # state 1 at config x_b
            return (ad * (e_11 + e_22) - bc * (e_12 + e_21)) / det
        self.trace_loss_one_walker = trace_loss_one_walker

        @jax.jit
        def loss_fn(p1, p2, batch_walkers):
            """Pfau-NES loss = mean over walkers of Tr(E_matrix).

            Note this is the *standard* VMC estimator on a complex
            energy-like quantity. Walkers must be sampled from
            |det(M)|^2 for the expectation to give the correct
            sum-of-eigenvalues estimate.
            """
            xa = batch_walkers[:, 0]
            xb = batch_walkers[:, 1]
            traces = jax.vmap(
                trace_loss_one_walker,
                in_axes=(0, 0, None, None),
            )(xa, xb, p1, p2)
            # Mean (no variance term -- Pfau doesn't use one; the
            # determinantal structure provides regularisation).
            return jnp.mean(traces)
        self.loss_fn = loss_fn

    def initialize_joint_walkers(self, rng_key, num_walkers):
        """Initialise a joint walker bank by drawing each state's
        configurations from a Gaussian around the nuclei, indep'tly."""
        key_a, key_b = jax.random.split(rng_key)
        w_a = self.driver_1.initialize_walkers(key_a, num_walkers)
        w_b = self.driver_2.initialize_walkers(key_b, num_walkers)
        return jnp.stack([w_a, w_b], axis=1)

    def __call__(
        self,
        rng_key,
        num_iters: int = 200,
        num_walkers: int = 256,
        num_steps_per_block: int = 100,
        num_blocks_equil: int = 5,
        num_sample_blocks: int = 2,
        mc_timestep: float = 0.1,
        lr: float = 1e-3,
        train_split: float = 0.8,
        batch_size: int = 256,
        num_epochs: int = 1,
        verbose: int = 1,
        prefix: str = "pfau_nes_k2",
    ):
        """Joint Adam loop on (params_1, params_2)."""
        p1, p2 = self.params_1, self.params_2

        optimizer = optax.adam(learning_rate=lr)
        opt_state = optimizer.init((p1, p2))
        mc_stepsize = (3 * mc_timestep) ** 0.5

        rng_key, init_walker_key = jax.random.split(rng_key)
        walkers = self.initialize_joint_walkers(init_walker_key, num_walkers)

        if verbose >= 1:
            print(f"Pfau-NES K=2: 2 x PsiFormer, joint walkers="
                  f"{num_walkers}, equilibrating {num_blocks_equil} blocks")

        # Equilibrate
        for _ in range(num_blocks_equil):
            rng_key, sub = jax.random.split(rng_key)
            carry = self.joint_sweep(
                sub, walkers, mc_stepsize, (p1, p2),
                num_steps_per_block,
            )
            _, walkers, mc_stepsize = carry

        chkpt_path = f"{prefix}.chk.h5"
        for iteration in range(num_iters):
            # Sample
            all_samples = []
            for _ in range(num_sample_blocks):
                rng_key, sub = jax.random.split(rng_key)
                carry = self.joint_sweep(
                    sub, walkers, mc_stepsize, (p1, p2),
                    num_steps_per_block,
                )
                _, walkers, mc_stepsize = carry
                all_samples.append(walkers)
            sampled = jnp.concatenate(all_samples, axis=0)
            n_samples = sampled.shape[0]

            # Permute + split
            rng_key, perm_key = jax.random.split(rng_key)
            idx = jax.random.permutation(perm_key, jnp.arange(n_samples))
            n_train = int(train_split * n_samples)
            train_w = sampled[idx[:n_train]]
            valid_w = sampled[idx[n_train:]]

            # Adam epochs
            epoch_losses = []
            for _ in range(num_epochs):
                for si in range(0, n_train, batch_size):
                    ei = min(si + batch_size, n_train)
                    loss, grads = jax.value_and_grad(
                        self.loss_fn, argnums=(0, 1),
                    )(p1, p2, train_w[si:ei])
                    updates, opt_state = optimizer.update(
                        grads, opt_state, (p1, p2),
                    )
                    (p1, p2) = optax.apply_updates((p1, p2), updates)
                    epoch_losses.append(loss)

            # Diagnostics: compute single-state energies from joint walkers
            # (these are biased because walkers come from |det M|^2, not
            # |psi_i|^2, but the trend is informative; we'll do clean
            # single-state sampling at the end)
            xa_v = valid_w[:, 0]
            xb_v = valid_w[:, 1]
            traces_v = jax.vmap(
                self.trace_loss_one_walker,
                in_axes=(0, 0, None, None),
            )(xa_v, xb_v, p1, p2)
            tr_mean = float(jnp.mean(traces_v))
            tr_err = float(jnp.std(traces_v)) / max(1, traces_v.size) ** 0.5

            if verbose >= 1:
                iter_loss = float(jnp.array(epoch_losses).mean())
                print(f"Iter {iteration:5d} | "
                      f"Tr(E) = {tr_mean:.6f} +/- {tr_err:.5f} | "
                      f"Loss: {iter_loss:.6f}")

            # Save both checkpoints
            if os.path.exists(f"{prefix}_1.chk.h5"):
                os.rename(f"{prefix}_1.chk.h5", f"{prefix}_1.{iteration}.h5")
            if os.path.exists(f"{prefix}_2.chk.h5"):
                os.rename(f"{prefix}_2.chk.h5", f"{prefix}_2.{iteration}.h5")
            save_nn_checkpoint(
                f"{prefix}_1.chk.h5", p1, iteration,
                self.config_name, self.mol_info, energy=tr_mean / 2,
            )
            save_nn_checkpoint(
                f"{prefix}_2.chk.h5", p2, iteration,
                self.config_name, self.mol_info, energy=tr_mean / 2,
            )

        self.params_1 = p1
        self.params_2 = p2
        return (p1, p2), {"trace_E": {"mean": tr_mean, "stderr": tr_err}}


def get_vmcopt_nn_pfau_k2_func(
    mol_info,
    config,
    init_key,
    init_from_ground_checkpoint: str = None,
):
    """Factory for the Pfau-NES K=2 driver.

    ``init_key`` is split into two sub-keys for the two state
    parameter inits.

    ``init_from_ground_checkpoint`` (optional): if supplied, both
    state 1 and state 2 NN parameters are initialised from the
    ground-state checkpoint. The determinantal training will then
    break the degeneracy via the matrix-energy gradient. Recommended
    for well-conditioned starting points.
    """
    k1, k2 = jax.random.split(init_key)
    driver = _VMCOptDriverNN_Pfau_K2(mol_info, config, k1, k2)
    if init_from_ground_checkpoint is not None:
        ground_params, _ = load_nn_checkpoint(
            init_from_ground_checkpoint, driver.driver_1.init_params,
        )
        driver.params_1 = ground_params
        # State 2: also init from ground but with slight perturbation
        # (otherwise det(M) = 0 identically). Add small Gaussian noise.
        import jax.tree as tree
        rng = jax.random.split(init_key, 3)[2]
        keys = jax.random.split(rng, len(jax.tree.leaves(ground_params)))
        leaves = jax.tree.leaves(ground_params)
        treedef = jax.tree.structure(ground_params)
        perturbed = [
            leaf + 0.01 * jax.random.normal(k, leaf.shape)
            for leaf, k in zip(leaves, keys)
        ]
        driver.params_2 = jax.tree.unflatten(treedef, perturbed)
    return driver
