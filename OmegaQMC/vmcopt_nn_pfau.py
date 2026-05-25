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
from jax.flatten_util import ravel_pytree

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
        num_steps_decorr: int = 10,
        mc_timestep: float = 0.1,
        lr: float = 1e-2,
        damping: float = 1e-3,
        cg_maxiter: int = 100,
        max_param_change: float = 0.2,
        jac_batch_size: int = 32,
        verbose: int = 1,
        prefix: str = "pfau_nes_k2",
        # Adam-era kwargs retained for backward compatibility (ignored):
        num_sample_blocks: int = None,
        num_epochs: int = None,
        batch_size: int = None,
        train_split: float = None,
    ):
        """Joint stochastic-reconfiguration loop on (params_1, params_2).

        This matches the natural-gradient training used in Pfau et al.
        (Science 2024). Adam is the wrong optimiser for the K-state
        determinantal loss: it has no notion of the curved Riemannian
        geometry of the variational manifold and gets trapped in the
        gerade subspace local minimum. SR uses the wavefunction-overlap
        matrix S_ij = E[O_i O_j] - E[O_i]E[O_j] (with
        O_i = d log|Psi|/d theta_i evaluated at the JOINT
        wavefunction Psi(x_a, x_b) = det[psi_i(x_j)]) to precondition
        the gradient.

        Walkers (x_a, x_b) are sampled jointly from |Psi(x_a, x_b)|^2
        via the existing self.joint_sweep MCMC. The per-walker
        ``local loss'' is the trace ``trace_loss_one_walker``; the
        per-walker derivative is the gradient of log|Psi| with respect
        to a CONCATENATED flat parameter vector (p1_flat, p2_flat).
        The SR force f = mean(dL * dO) is solved against
        (dO^T dO / N + damping * I) by conjugate gradient (identical
        machinery to vmcopt_nn_sr).

        Kwargs ``num_sample_blocks``, ``num_epochs``, ``batch_size``,
        ``train_split`` from the previous Adam call signature are kept
        for backward compatibility but ignored: SR does one walker
        decorrelation + one preconditioned update per outer iter.
        """
        p1, p2 = self.params_1, self.params_2
        p1_flat, unravel_1 = ravel_pytree(p1)
        p2_flat, unravel_2 = ravel_pytree(p2)
        n_p1 = p1_flat.shape[0]
        n_p2 = p2_flat.shape[0]
        n_total = n_p1 + n_p2

        mc_stepsize = (3 * mc_timestep) ** 0.5

        signed_psi_1 = self._signed_psi_1
        signed_psi_2 = self._signed_psi_2
        local_E_1 = self._local_E_1
        local_E_2 = self._local_E_2

        # log|Psi(x_a, x_b)| as a function of CONCATENATED flat params
        def log_abs_joint_psi_of_concat(p_concat, x_a, x_b):
            p1_now = unravel_1(p_concat[:n_p1])
            p2_now = unravel_2(p_concat[n_p1:])
            a = signed_psi_1(x_a, p1_now)
            b = signed_psi_2(x_a, p2_now)
            c = signed_psi_1(x_b, p1_now)
            d = signed_psi_2(x_b, p2_now)
            val = a * d - b * c
            return jnp.log(jnp.abs(val) + 1e-300)

        # Per-walker local loss (trace) on concatenated flat params
        def local_loss_concat(p_concat, x_a, x_b):
            p1_now = unravel_1(p_concat[:n_p1])
            p2_now = unravel_2(p_concat[n_p1:])
            a = signed_psi_1(x_a, p1_now)
            b = signed_psi_2(x_a, p2_now)
            c = signed_psi_1(x_b, p1_now)
            d = signed_psi_2(x_b, p2_now)
            ad = a * d
            bc = b * c
            det = ad - bc
            e_11 = local_E_1(x_a, p1_now)
            e_22 = local_E_2(x_b, p2_now)
            e_12 = local_E_2(x_a, p2_now)
            e_21 = local_E_1(x_b, p1_now)
            return (ad * (e_11 + e_22) - bc * (e_12 + e_21)) / det

        # Fused per-walker (local loss, log|psi| gradient) for SR
        @jax.jit
        def jac_and_loss_batch_fn(batch_xa, batch_xb, p_concat):
            def per_walker(xa, xb):
                L = local_loss_concat(p_concat, xa, xb)
                O = jax.grad(log_abs_joint_psi_of_concat)(p_concat, xa, xb)
                return L, O.astype(jnp.float32)
            return jax.vmap(per_walker)(batch_xa, batch_xb)

        # CG solve for natural gradient
        @jax.jit
        def sr_solve(dO, f, x0, dmp, maxiter):
            nw = dO.shape[0]
            dmp_f = jnp.float32(dmp)
            def s_matvec(v):
                t = dO @ v
                return (dO.T @ t) / nw + dmp_f * v
            delta_p, _ = jax.scipy.sparse.linalg.cg(
                s_matvec, f, x0=x0, maxiter=maxiter,
            )
            return delta_p

        # Walker init + equilibration
        rng_key, init_walker_key = jax.random.split(rng_key)
        walkers = self.initialize_joint_walkers(init_walker_key, num_walkers)

        if verbose >= 1:
            print(f"Pfau-NES K=2 (SR): 2 x PsiFormer, joint walkers="
                  f"{num_walkers}, n_params = {n_p1} + {n_p2} = {n_total}")
            print(f"  lr = {lr}, damping = {damping}, "
                  f"cg_maxiter = {cg_maxiter}, jac_batch = {jac_batch_size}")
            print(f"  equilibrating {num_blocks_equil} blocks of "
                  f"{num_steps_per_block} steps...")
        for _ in range(num_blocks_equil):
            rng_key, sub = jax.random.split(rng_key)
            carry = self.joint_sweep(
                sub, walkers, mc_stepsize, (p1, p2), num_steps_per_block,
            )
            _, walkers, mc_stepsize = carry

        prev_delta = jnp.zeros(n_total, dtype=jnp.float32)
        from datetime import datetime
        t_prev = datetime.now()

        for iteration in range(num_iters):
            # (a) Decorrelate walkers
            rng_key, sub = jax.random.split(rng_key)
            carry = self.joint_sweep(
                sub, walkers, mc_stepsize, (p1, p2), num_steps_decorr,
            )
            _, walkers, mc_stepsize = carry
            xa = walkers[:, 0]
            xb = walkers[:, 1]

            # (b) Per-walker (L, O) in batches
            p_concat = jnp.concatenate([p1_flat, p2_flat])
            L_parts = []
            O_parts = []
            for i in range(0, num_walkers, jac_batch_size):
                end = min(i + jac_batch_size, num_walkers)
                L_b, O_b = jac_and_loss_batch_fn(xa[i:end], xb[i:end],
                                                  p_concat)
                L_parts.append(L_b)
                O_parts.append(O_b)
            L = jnp.concatenate(L_parts, axis=0)
            O = jnp.concatenate(O_parts, axis=0)
            L_mean = float(jnp.mean(L))
            L_err = float(jnp.std(L)) / max(1, L.size) ** 0.5

            # (c) Centered force f = mean((L - <L>)(O - <O>))
            O_mean = jnp.mean(O, axis=0)
            dO = O - O_mean[None, :]
            dL = (L - jnp.mean(L)).astype(jnp.float32)
            f = (dL @ dO) / dO.shape[0]

            # (d) CG solve
            delta_p = sr_solve(dO, f, prev_delta, damping, cg_maxiter)
            prev_delta = delta_p

            # (e) Clip + apply
            max_dp = float(jnp.max(jnp.abs(delta_p)))
            scale = min(1.0, max_param_change / max(max_dp, 1e-30))
            p_concat = p_concat - lr * scale * delta_p
            p1_flat = p_concat[:n_p1]
            p2_flat = p_concat[n_p1:]
            p1 = unravel_1(p1_flat)
            p2 = unravel_2(p2_flat)

            # (f) Logging
            if verbose >= 1:
                now = datetime.now()
                dt = (now - t_prev).total_seconds()
                t_prev = now
                print(f"Iter {iteration:5d} | "
                      f"Tr(E) = {L_mean:.6f} +/- {L_err:.5f} | "
                      f"max_dp = {max_dp:.4e} | scale = {scale:.3f} | "
                      f"dt = {dt:.1f}s")

            # (g) Save checkpoints periodically
            if (iteration + 1) % 50 == 0 or iteration == num_iters - 1:
                if os.path.exists(f"{prefix}_1.chk.h5"):
                    os.rename(f"{prefix}_1.chk.h5",
                              f"{prefix}_1.{iteration}.h5")
                if os.path.exists(f"{prefix}_2.chk.h5"):
                    os.rename(f"{prefix}_2.chk.h5",
                              f"{prefix}_2.{iteration}.h5")
                save_nn_checkpoint(
                    f"{prefix}_1.chk.h5", p1, iteration,
                    self.config_name, self.mol_info, energy=L_mean / 2,
                )
                save_nn_checkpoint(
                    f"{prefix}_2.chk.h5", p2, iteration,
                    self.config_name, self.mol_info, energy=L_mean / 2,
                )

        self.params_1 = p1
        self.params_2 = p2
        return (p1, p2), {"trace_E": {"mean": L_mean, "stderr": L_err}}


def get_vmcopt_nn_pfau_k2_func(
    mol_info,
    config,
    init_key,
    init_from_ground_checkpoint: str = None,
    init_perturbation: float = 0.5,
    init_state2_random: bool = False,
):
    """Factory for the Pfau-NES K=2 driver.

    ``init_key`` is split into two sub-keys for the two state
    parameter inits.

    ``init_from_ground_checkpoint`` (optional): if supplied, state 1
    is initialised from the ground-state checkpoint. State 2 is then
    either (a) a random PsiFormer init if ``init_state2_random=True``
    (independent draw from the PsiFormer prior, completely different
    parameters from the ground state, and the strongest available
    symmetry-breaker), or (b) the same ground-state checkpoint with
    Gaussian noise of std ``init_perturbation`` added per parameter.

    The perturbation default was 0.01 in earlier versions, which left
    the K=2 manifold essentially one-dimensional ({GS + tiny noise})
    and made the orthogonal direction within the recovered span
    unphysical. Increased to 0.5 (50% per-parameter relative noise)
    to give state 2 a meaningful initial distance from the ground.
    For a really clean break of the gerade-trap, set
    ``init_state2_random=True``.
    """
    k1, k2 = jax.random.split(init_key)
    driver = _VMCOptDriverNN_Pfau_K2(mol_info, config, k1, k2)
    if init_from_ground_checkpoint is not None:
        ground_params, _ = load_nn_checkpoint(
            init_from_ground_checkpoint, driver.driver_1.init_params,
        )
        driver.params_1 = ground_params
        if init_state2_random:
            # Leave driver.params_2 at its independent random init;
            # do nothing further.
            pass
        else:
            rng = jax.random.split(init_key, 3)[2]
            keys = jax.random.split(rng, len(jax.tree.leaves(ground_params)))
            leaves = jax.tree.leaves(ground_params)
            treedef = jax.tree.structure(ground_params)
            perturbed = [
                leaf + float(init_perturbation)
                       * jax.random.normal(k, leaf.shape)
                for leaf, k in zip(leaves, keys)
            ]
            driver.params_2 = jax.tree.unflatten(treedef, perturbed)
    return driver
