"""Stochastic Reconfiguration optimizer for QED-NN-VMC.

Lean SR optimizer dedicated to cavity-coupled NN trial wavefunctions.
Implements the same SR algorithm as :mod:`OmegaQMC.vmcopt_nn_sr`
(Sorella, Becca-Sorella) but with a smaller surface area, joint
discrete-continuous Metropolis sampling delegated to
:class:`OmegaQMC.qed_vmc_nn._QEDVMCDriverNN`, and explicit handling of
the learnable coherent-state shift α as a separate parameter group.

Algorithm (per iteration):
  (a) Decorrelate walkers via the joint :math:`(\\mathbf r, n)` MH sampler
      (inherited from the QED driver).
  (b) Compute local energies via
      :func:`OmegaQMC.psi.nn.qed_physics.pauli_fierz_local_energy`.
  (c) Compute Jacobian of :math:`\\log|\\Psi(r, n)|` w.r.t. flat-pytree
      parameters in batches (NN params + α together).
  (d) Centered force vector :math:`f = \\langle \\delta E_L \\, \\delta O\\rangle`.
  (e) CG solve :math:`(S + \\varepsilon I)\\,\\delta p = f` with
      :math:`S = O^T O / N` (overlap / Fisher matrix).
  (f) Apply update with optional separate learning rate for α.

Compared to ``vmcopt_nn_sr``:
  * No HDF5 checkpointing (plain in-memory state for v1 — add later).
  * No autotuning of walker counts or Jacobian batch size.
  * Joint :math:`(\\mathbf r, n)` walker state.
  * Optional ``alpha_lr_scale``: multiplies the learning rate applied to
    the α component (default 1.0). This lets α evolve at a different
    pace than NN params if needed for stability.

Notes:
  * The flat pytree includes both NN params and α as a single
    parameter vector for the SR linear algebra. Separation by group
    happens after the CG solve via the unravel function.
  * Targets the **mango** GH200 cluster for production runs.

References:
  - Sorella, "Generalized Lanczos algorithm for variational quantum
    Monte Carlo," PRB 2001.
  - Becca-Sorella, "Quantum Monte Carlo Approaches for Correlated
    Systems," 2017.
"""
from __future__ import annotations

from datetime import datetime
from functools import partial

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree

from .qed_vmc_nn import _QEDVMCDriverNN, get_qed_vmc_nn_func
from .utils import Mole_custom


__all__ = [
    "get_qed_vmcopt_nn_sr_func",
]


def _flatten_params(params):
    """Flatten the QED params dict {'nn': ..., 'alpha': ...} to a 1D vector.

    Returns ``(flat, unravel)`` where ``unravel`` reconstructs the dict.
    """
    return ravel_pytree(params)


class _QEDVMCOptDriverNN_SR:
    """SR optimizer for QED-NN-VMC. Use :func:`get_qed_vmcopt_nn_sr_func`."""

    def __init__(
        self,
        mol_info: Mole_custom,
        config,
        init_key,
        *,
        omega: float,
        coupling_vec,
        alpha_init: float = 0.0,
        alpha_train: bool = True,
        nph_max: int = 10,
        n_aware: bool = False,
        fock_hidden_dim: int = 64,
        arch: str | None = None,
    ):
        # Build a QED driver — provides log_psi, sampler, local energy.
        self.driver = get_qed_vmc_nn_func(
            mol_info, config, init_key,
            omega=omega, coupling_vec=coupling_vec,
            alpha_init=alpha_init, alpha_train=alpha_train,
            nph_max=nph_max,
            n_aware=n_aware, fock_hidden_dim=fock_hidden_dim,
            arch=arch,
        )
        self.arch = self.driver.arch
        self.alpha_train = (
            alpha_train and self.arch in ("factorized", "hybrid")
            and "alpha" in self.driver.params
        )
        self.n_aware = (self.arch == "n_aware")
        self.init_params = self.driver.params
        self.log_psi = self.driver.log_psi
        self.nuc_crds = self.driver.nuc_crds

    # ---- SR step kernels ----
    def _build_sr_kernels(self):
        log_psi = self.log_psi
        nuc = self.nuc_crds

        # Flatten template params to get the unravel function (closure).
        _, unravel_fn = _flatten_params(self.init_params)

        def log_psi_of_flat(fp, r, n):
            return log_psi(r, nuc, n, unravel_fn(fp))

        @jax.jit
        def jac_batch_fn(flat_params, r_batch, n_batch):
            """∂ log|Ψ| / ∂ flat_params for a batch of walkers."""
            return jax.vmap(
                jax.grad(log_psi_of_flat),
                in_axes=(None, 0, 0),
            )(flat_params, r_batch, n_batch).astype(jnp.float32)

        @jax.jit
        def sr_cg_solve(dO, f, x0, damping, maxiter):
            nw = dO.shape[0]
            dmp = jnp.float32(damping)

            def s_matvec(v):
                t = dO @ v
                Sv = (dO.T @ t) / nw
                return Sv + dmp * v

            delta_p, info = jax.scipy.sparse.linalg.cg(
                s_matvec, f, x0=x0, maxiter=maxiter,
            )
            return delta_p, info

        return jac_batch_fn, sr_cg_solve, unravel_fn

    # ---- Optimization loop ----
    def __call__(
        self,
        rng_key,
        num_iters: int = 200,
        num_walkers: int = 256,
        num_steps_per_block: int = 50,
        num_blocks_equil: int = 5,
        num_steps_decorr: int = 5,
        mc_timestep: float = 0.1,
        lr: float = 0.05,
        damping: float = 1e-3,
        cg_maxiter: int = 200,
        max_param_change: float = 0.5,
        jac_batch_size: int = 64,
        alpha_lr_scale: float = 1.0,
        verbose: int = 1,
    ):
        """Run SR iterations.

        Args:
            rng_key: JAX PRNG key.
            num_iters: Number of SR update steps.
            num_walkers: Walker batch size.
            num_steps_per_block: MH steps in equilibration block.
            num_blocks_equil: Equilibration blocks before SR begins.
            num_steps_decorr: Decorrelation MH steps between SR updates.
            mc_timestep: Initial MH stepsize (Gaussian σ for r).
            lr: SR learning rate (scalar).
            damping: Diagonal damping for the SR linear system.
            cg_maxiter: Conjugate-gradient max iterations.
            max_param_change: Cap on max |δp| element after the solve
                (clips before applying lr).
            jac_batch_size: Walkers per Jacobian micro-batch (memory).
            alpha_lr_scale: Multiplier on the learning rate applied to α
                only (NN params get plain lr).
            verbose: 0 = silent, 1 = per-iter logs, 2 = + acceptance.

        Returns:
            ``(params_final, history)`` where history contains
            ``energies``, ``acceptances``, ``alpha_history`` (list of
            float per iter) if ``alpha_train``.
        """
        if isinstance(rng_key, int):
            rng_key = jax.random.key(rng_key)

        params = self.init_params
        flat_params, unravel_fn_init = _flatten_params(params)
        n_params = int(flat_params.shape[0])
        if verbose:
            print(
                f"QED-SR: {n_params} parameters "
                f"({'incl' if self.alpha_train else 'excl'} alpha)",
                flush=True,
            )

        jac_batch_fn, sr_cg_solve, unravel_fn = self._build_sr_kernels()

        # Initialize walkers (r, n) and equilibrate.
        rng_key, key_init = jax.random.split(rng_key)
        elec, n_ph = self.driver.initialize_walkers(key_init, num_walkers)

        sigma_r = float(mc_timestep)
        # Equilibration: just run MH steps for `num_blocks_equil`
        # blocks of `num_steps_per_block` steps each.
        mh_step_batch = self.driver._mh_step_batch
        for _blk in range(num_blocks_equil):
            for _step in range(num_steps_per_block):
                rng_key, key_step = jax.random.split(rng_key)
                step_keys = jax.random.split(key_step, num_walkers)
                elec, n_ph, acc_r, acc_n, was_r = mh_step_batch(
                    step_keys, elec, n_ph, params, sigma_r, 0.5,
                )

        # --- Main SR loop ---
        history = {
            "energies": [],
            "energy_serrs": [],
            "acceptances_r": [],
            "acceptances_n": [],
            "alpha_history": [],
            "param_change_max": [],
        }
        prev_delta = jnp.zeros(n_params, dtype=jnp.float32)

        # Mask in flat-param space telling which entries are α (for
        # alpha_lr_scale). We need the unravel structure to identify them.
        if self.alpha_train and "alpha" in params:
            template_zero = jax.tree.map(jnp.zeros_like, params)
            alpha_marker = dict(template_zero)
            alpha_marker["alpha"] = jnp.ones_like(params["alpha"])
            alpha_mask, _ = ravel_pytree(alpha_marker)
            alpha_mask = alpha_mask.astype(jnp.float32)
        else:
            alpha_mask = jnp.zeros(n_params, dtype=jnp.float32)

        for iteration in range(num_iters):
            t0 = datetime.now()

            # (a) decorrelate walkers
            n_decorr_acc_r = 0.0
            n_decorr_acc_n = 0.0
            n_decorr_was_r = 0.0
            n_decorr_was_n = 0.0
            for _step in range(num_steps_decorr):
                rng_key, key_step = jax.random.split(rng_key)
                step_keys = jax.random.split(key_step, num_walkers)
                elec, n_ph, acc_r, acc_n, was_r = mh_step_batch(
                    step_keys, elec, n_ph, params, sigma_r, 0.5,
                )
                n_decorr_acc_r += float(jnp.sum(acc_r & was_r))
                n_decorr_was_r += float(jnp.sum(was_r))
                n_decorr_acc_n += float(jnp.sum(acc_n & ~was_r))
                n_decorr_was_n += float(jnp.sum(~was_r))

            # (b) local energies
            e_loc = self.driver._local_energy_batch(elec, n_ph, params)
            E_mean = float(jnp.mean(e_loc))
            E_serr = float(jnp.std(e_loc) / jnp.sqrt(num_walkers))

            # (c) Jacobian of log|Ψ| in batches
            O_parts = []
            for i_start in range(0, num_walkers, jac_batch_size):
                i_end = min(i_start + jac_batch_size, num_walkers)
                O_parts.append(
                    jac_batch_fn(
                        flat_params,
                        elec[i_start:i_end],
                        n_ph[i_start:i_end],
                    )
                )
            O = jnp.concatenate(O_parts, axis=0)

            # (d) centered force vector
            O_mean = jnp.mean(O, axis=0)
            dO = O - O_mean[None, :]
            dE = (e_loc - jnp.mean(e_loc)).astype(jnp.float32)
            f = (dE @ dO) / num_walkers

            # (e) SR CG solve
            delta_p, _cg_info = sr_cg_solve(
                dO, f, prev_delta, damping, cg_maxiter,
            )
            prev_delta = delta_p

            # (f) apply update with optional α scaling and clip
            max_abs = float(jnp.max(jnp.abs(delta_p)))
            if max_abs > max_param_change:
                delta_p = delta_p * (max_param_change / max_abs)

            # Per-component learning rate: lr for NN, lr*alpha_lr_scale for α
            if self.alpha_train and alpha_lr_scale != 1.0:
                lr_vec = (
                    lr * (1.0 - alpha_mask)
                    + lr * alpha_lr_scale * alpha_mask
                ).astype(jnp.float32)
                flat_params = (
                    flat_params - lr_vec * delta_p.astype(flat_params.dtype)
                )
            else:
                flat_params = (
                    flat_params - lr * delta_p.astype(flat_params.dtype)
                )

            params = unravel_fn(flat_params)

            history["energies"].append(E_mean)
            history["energy_serrs"].append(E_serr)
            history["acceptances_r"].append(
                n_decorr_acc_r / max(n_decorr_was_r, 1.0)
            )
            history["acceptances_n"].append(
                n_decorr_acc_n / max(n_decorr_was_n, 1.0)
            )
            history["param_change_max"].append(max_abs)
            if "alpha" in params:
                history["alpha_history"].append(float(params["alpha"]))

            if verbose:
                dt = (datetime.now() - t0).total_seconds()
                msg = (
                    f"[QED-SR iter {iteration + 1}/{num_iters}] "
                    f"E={E_mean:.6f}±{E_serr:.6f}  "
                    f"|δp|max={max_abs:.4f}  "
                    f"dt={dt:.2f}s"
                )
                if "alpha" in params:
                    msg += f"  α={float(params['alpha']):.4f}"
                if verbose >= 2:
                    msg += (
                        f"  acc_r={history['acceptances_r'][-1]:.2f}"
                        f"  acc_n={history['acceptances_n'][-1]:.2f}"
                    )
                print(msg, flush=True)

        return params, history


def get_qed_vmcopt_nn_sr_func(
    mol_info: Mole_custom,
    config,
    init_key,
    *,
    omega: float,
    coupling_vec,
    alpha_init: float = 0.0,
    alpha_train: bool = True,
    nph_max: int = 10,
    n_aware: bool = False,
    fock_hidden_dim: int = 64,
    arch: str | None = None,
) -> _QEDVMCOptDriverNN_SR:
    """Construct an SR optimizer driver for QED-NN-VMC.

    Architecture selection (in order of precedence):
      * ``arch='factorized'``    — Phase 2b form Ψ_e(r)·⟨n|α⟩.
      * ``arch='n_aware'``       — Phase 2f-1 Path A: Ψ_e(r) + Fock head.
      * ``arch='hybrid'``        — Phase 2f-1 Path B: Ψ_e(r)·⟨n|α⟩
        + Fock head; positive-Ψ ansatz.
      * ``arch='signed_hybrid'`` — Phase 2g: same factorisation as
        hybrid but with PsiFormer's psi.sign threaded through and a
        SignedFockHead that outputs both magnitude and sign
        corrections. Allows the joint Ψ(r, n) to flip sign across Fock
        sectors with r-dependence — required for capturing polariton
        ground-state stabilisation in symmetric systems.

    For backward compatibility, ``arch`` defaults to ``None`` and is
    inferred from the legacy ``n_aware`` bool when omitted.

    ``alpha_*`` arguments apply when α is part of the chosen architecture
    (factorized, hybrid, signed_hybrid). With ``arch='n_aware'`` they are
    ignored.

    See :class:`_QEDVMCOptDriverNN_SR.__call__` for run-time hyperparameters.
    """
    return _QEDVMCOptDriverNN_SR(
        mol_info, config, init_key,
        omega=omega, coupling_vec=coupling_vec,
        alpha_init=alpha_init, alpha_train=alpha_train,
        nph_max=nph_max,
        n_aware=n_aware, fock_hidden_dim=fock_hidden_dim,
        arch=arch,
    )
