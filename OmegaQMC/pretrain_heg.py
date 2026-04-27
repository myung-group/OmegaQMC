"""Supervised Hartree–Fock pre-training for HEG PsiFormer.

Problem it solves.  Starting HEG PsiFormer training from the
randomly-initialised network directly via energy VMC falls into a
well-known obstacle: the default variance-scaled init gives an
initial wavefunction so far from any smooth target that Metropolis
acceptance collapses and walkers get stuck; the tiny rescaling
needed to recover acceptance in turn leaves the backflow
parameters with almost-zero Jacobian on the log-density, so the
VMC energy gradient is below the walker-sampling noise floor and
both Adam and SR random-walk around the HF plateau without
descending.

The pre-training stage below sidesteps this entirely by training
the network **against a known analytic target** — the Hartree-Fock
Slater determinant — using supervised regression:

    θ*  =  argmin_θ  𝔼_{r ~ |ψ_HF|²}  ‖ Φ_NN(r; θ)  −  Φ_HF(r) ‖²

where ``Φ_NN`` is the network's post-backflow orbital matrix (see
:meth:`HEGPsiFormerWaveFunction.orbital_matrices`) and ``Φ_HF`` is
the free-electron Slater matrix of filled plane waves.

The target is deterministic — no Hamiltonian evaluation, no local
energies, no variance in the per-walker signal — so the gradient
signal is 3-4 orders of magnitude larger than the VMC energy
gradient on the same walkers, and ordinary Adam converges in
several hundred iterations.

Outcome.  After pre-training:
  * ``|ψ_θ|²  ≈  |ψ_HF|²`` (measured by small MSE).
  * The GNN, attention, and backflow weights hold **non-trivial
    learned values** — they've been trained to *cancel* out any
    departure from HF.  In contrast, the pre-pretraining init has
    all backflow layers barely activated; small subsequent
    perturbations have no effect on the wavefunction.
  * Subsequent energy VMC (Adam or SR) sees a meaningful gradient
    from iteration 1 and actually descends below HF.

Usage.

.. code-block:: python

    from OmegaQMC.pretrain_heg import pretrain_heg_psiformer

    trained_params = pretrain_heg_psiformer(
        config,
        init_key,
        num_iters=500,
        num_walkers=256,
    )
    # Feed ``trained_params`` into the Adam/SR energy optimizer as
    # its initial parameter pytree.
"""

from __future__ import annotations

import sys
from datetime import datetime
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx

from .psi.nn.env_periodic import enumerate_real_pw_basis
from .psi.nn.heg_psiformer import (
    _make_heg_phys_conf,
    build_heg_psiformer_wf,
)
from .psi.nn.heg_wf import HEGPsiFormerConfig
from .psi.nn.periodic import make_cubic_lattice, wrap_to_cell


TARGET_ACCEPTANCE = 0.5
STEP_SIZE_ADAPT_RATE = 0.05


def _adapt_step(step_size, acceptance):
    return step_size * (
        1.0 + STEP_SIZE_ADAPT_RATE * (acceptance - TARGET_ACCEPTANCE)
    )


# =====================================================================
# Hartree–Fock orbital target (analytic, frozen)
# =====================================================================

def _hf_orb_matrices_one(r, basis):
    """Evaluate HF plane-wave orbitals at one spin block.

    Orbital ``j`` of the Fermi sea is either ``cos(k_j·r)`` or
    ``sin(k_j·r)`` depending on ``basis.basis_is_sin[j]``.
    The matrix is ``(n_spin, n_orb_spin)`` with entry ``[i, j]`` =
    orbital ``j`` evaluated at electron ``i``.

    Args:
        r: ``(n_spin, 3)`` positions for one spin.
        basis: :class:`~.env_periodic.RealPWBasis` for this spin
            (from :func:`enumerate_real_pw_basis`).
    """
    if basis.basis_idx.shape[0] == 0:
        return jnp.zeros((0, 0), dtype=r.dtype)
    kvecs = basis.kvecs[basis.basis_idx]           # (n_orb, 3)
    is_sin = basis.basis_is_sin                    # (n_orb,)
    kr = r @ kvecs.T                                # (n_spin, n_orb)
    return jnp.where(
        is_sin[None, :] == 1,
        jnp.sin(kr),
        jnp.cos(kr),
    )


def _hf_orb_matrices(r, n_up, basis_up, basis_down):
    """Return ``(orb_up, orb_down)`` HF orbital matrices at ``r``.

    ``r`` has shape ``(n_elec, 3)`` in grouped up-then-down order.
    """
    orb_up = _hf_orb_matrices_one(r[:n_up], basis_up)
    if basis_down is not None and basis_down.basis_idx.shape[0] > 0:
        orb_down = _hf_orb_matrices_one(r[n_up:], basis_down)
    else:
        orb_down = jnp.zeros((0, 0), dtype=r.dtype)
    return orb_up, orb_down


# =====================================================================
# Pre-training driver
# =====================================================================

class _HEGPreTrainDriver:
    """Adam-on-MSE pre-training for HEG PsiFormer."""

    def __init__(
        self,
        config: HEGPsiFormerConfig,
        init_key,
        *,
        lr: float = 1e-3,
    ):
        self.config = config
        self.L = float(config.L)
        self.n_up = int(config.n_up)
        self.n_down = int(config.n_down)
        self.nelec = self.n_up + self.n_down
        self.n_det = int(config.n_det)
        self.dim = int(getattr(config, 'dim', 3))
        self.lr = float(lr)

        if self.dim == 3:
            self.lattice = make_cubic_lattice(self.L)
        else:
            from .psi.nn.periodic import make_square_lattice
            self.lattice = make_square_lattice(self.L)

        rngs = nnx.Rngs(init_key)
        model = build_heg_psiformer_wf(config, rngs)
        self.graphdef, self.params, self.other = nnx.split(
            model, nnx.Param, ...,
        )

        if self.dim == 3:
            self.basis_up = enumerate_real_pw_basis(self.n_up, self.L)
            if self.n_down > 0:
                self.basis_down = enumerate_real_pw_basis(
                    self.n_down, self.L,
                )
            else:
                self.basis_down = None
        else:
            from .psi.nn.env_periodic import enumerate_real_pw_basis_2d
            self.basis_up = enumerate_real_pw_basis_2d(
                self.n_up, self.L,
            )
            if self.n_down > 0:
                self.basis_down = enumerate_real_pw_basis_2d(
                    self.n_down, self.L,
                )
            else:
                self.basis_down = None

        graphdef = self.graphdef
        other = self.other
        n_up = self.n_up
        n_down = self.n_down
        basis_up = self.basis_up
        basis_down = self.basis_down
        lattice = self.lattice

        # ---- Network orbital extractor ----
        def nn_orb(r, params):
            mdl = nnx.merge(graphdef, params, other)
            return mdl.orbital_matrices(r)

        # ---- HF orbital target (analytic) ----
        def hf_orb(r):
            return _hf_orb_matrices(r, n_up, basis_up, basis_down)

        # ---- MCMC on |ψ_HF|² ----
        # Pre-training samples from the HF distribution (not the
        # current network's |ψ|², which is broken at the random
        # init).  Once pre-training converges the two distributions
        # coincide, so the subsequent energy VMC can start
        # MCMC from the same walker set without a long burn-in.
        def log_psi_hf(r):
            orb_u, orb_d = hf_orb(r)
            _, log_u = jnp.linalg.slogdet(orb_u)
            if n_down > 0:
                _, log_d = jnp.linalg.slogdet(orb_d)
                return log_u + log_d
            return log_u

        def hf_metropolis(rng_key, r, step_size):
            kp, ka = jax.random.split(rng_key)
            proposed = r + step_size * jax.random.normal(
                kp, r.shape,
            )
            proposed = wrap_to_cell(proposed, lattice)
            lp_old = log_psi_hf(r)
            lp_new = log_psi_hf(proposed)
            accept = jax.random.uniform(ka) < jnp.exp(
                2.0 * (lp_new - lp_old),
            )
            return jnp.where(accept, proposed, r), accept

        self._hf_mcmc_allw = jax.jit(jax.vmap(
            hf_metropolis, in_axes=(0, 0, None),
        ))

        # ---- MSE loss and gradient ----
        def mse_loss(params, walkers):
            orb_nn_u, orb_nn_d = jax.vmap(
                nn_orb, in_axes=(0, None),
            )(walkers, params)
            orb_hf_u, orb_hf_d = jax.vmap(hf_orb)(walkers)
            # Broadcast HF (no det axis) over n_det.
            loss = jnp.mean(
                (orb_nn_u - orb_hf_u[:, None, :, :]) ** 2,
            )
            if n_down > 0:
                loss = loss + jnp.mean(
                    (orb_nn_d - orb_hf_d[:, None, :, :]) ** 2,
                )
            return loss

        self._loss = jax.jit(mse_loss)
        self._loss_grad = jax.jit(jax.grad(mse_loss))

        self.opt = optax.adam(lr)
        self.opt_state = self.opt.init(self.params)

    # -------------------------------------------------
    # Main training call
    # -------------------------------------------------

    def __call__(
        self,
        rng_key,
        num_iters: int = 500,
        num_walkers: int = 256,
        num_equil_steps: int = 200,
        mcmc_decorr_steps: int = 5,
        mc_timestep: float = 0.1,
        fname_log: Optional[str] = None,
        verbose: int = 1,
    ):
        """Run supervised pre-training.

        Args:
            rng_key: JAX PRNG key (int or array).
            num_iters: Adam iterations on the MSE loss.
            num_walkers: MCMC walkers (sampled from |ψ_HF|²).
            num_equil_steps: Burn-in MCMC steps before training.
            mcmc_decorr_steps: MCMC steps between updates.
            mc_timestep: Initial Metropolis timestep.
            fname_log: Optional log file path.
            verbose: 0 silent, 1 per-iter summary.

        Returns:
            Dict with ``'params'`` (trained pytree, ready to feed
            into Adam/SR energy optimiser), ``'loss_history'``,
            ``'final_loss'``.
        """
        if isinstance(rng_key, int):
            rng_key = jax.random.key(rng_key)
        if (fname_log is None
                or (isinstance(fname_log, str) and fname_log == "")):
            fout = sys.stdout
        else:
            fout = open(fname_log, 'w', 1)

        params = self.params
        opt_state = self.opt_state

        # Walker initialisation.
        rng_key, init_key = jax.random.split(rng_key)
        walkers = self.L * jax.random.uniform(
            init_key, (num_walkers, self.nelec, self.dim),
        )
        step_size = jnp.asarray((3 * mc_timestep) ** 0.5)

        # Burn-in on |ψ_HF|².
        for _ in range(num_equil_steps):
            rng_key, sub = jax.random.split(rng_key)
            keys = jax.random.split(sub, num_walkers)
            walkers, acc = self._hf_mcmc_allw(
                keys, walkers, step_size,
            )
            step_size = _adapt_step(step_size, float(jnp.mean(acc)))

        if verbose >= 1:
            loss_init = float(self._loss(params, walkers))
            print(
                f"# Pre-training HEG PsiFormer: "
                f"{num_iters} iters, {num_walkers} walkers, "
                f"lr={self.lr}",
                file=fout,
            )
            print(
                f"# Burn-in acceptance: {float(jnp.mean(acc)):.3f}, "
                f"step size: {float(step_size):.4f} Bohr",
                file=fout,
            )
            print(
                f"# initial MSE = {loss_init:.4e}",
                file=fout,
            )
            print(
                "# iter       MSE             step size      dt",
                file=fout,
            )

        loss_history = []
        t_prev = datetime.now()

        for it in range(1, num_iters + 1):
            # MCMC decorrelate on |ψ_HF|².
            for _ in range(mcmc_decorr_steps):
                rng_key, sub = jax.random.split(rng_key)
                keys = jax.random.split(sub, num_walkers)
                walkers, acc = self._hf_mcmc_allw(
                    keys, walkers, step_size,
                )
                step_size = _adapt_step(step_size, float(jnp.mean(acc)))

            # Supervised MSE gradient step.
            loss = float(self._loss(params, walkers))
            g = self._loss_grad(params, walkers)
            updates, opt_state = self.opt.update(g, opt_state)
            params = optax.apply_updates(params, updates)

            loss_history.append(loss)
            now = datetime.now()
            dt = (now - t_prev).total_seconds()
            t_prev = now
            if verbose >= 1 and (it <= 10 or it % 25 == 0
                                 or it == num_iters):
                print(
                    f"{it:>6d}  {loss:>14.6e}  "
                    f"{float(step_size):>10.4f}  "
                    f"{dt:>8.3f}",
                    file=fout,
                )

        final_loss = loss_history[-1] if loss_history else float('nan')

        if verbose >= 1:
            print(
                f"# Pre-training done.  MSE: "
                f"{loss_history[0]:.4e} → {final_loss:.4e} "
                f"(×{loss_history[0] / final_loss:.1e} reduction)",
                file=fout,
            )

        if not (fname_log is None
                or (isinstance(fname_log, str) and fname_log == "")):
            fout.close()

        self.params = params
        self.opt_state = opt_state
        return {
            'params': params,
            'loss_history': loss_history,
            'final_loss': final_loss,
            'graphdef': self.graphdef,
        }


def pretrain_heg_psiformer(
    config: HEGPsiFormerConfig,
    init_key,
    *,
    mcmc_key=None,
    num_iters: int = 500,
    num_walkers: int = 256,
    lr: float = 1e-3,
    num_equil_steps: int = 200,
    mcmc_decorr_steps: int = 5,
    mc_timestep: float = 0.1,
    fname_log: Optional[str] = None,
    verbose: int = 1,
):
    """One-shot supervised pre-training for HEG PsiFormer.

    Convenience wrapper around :class:`_HEGPreTrainDriver`.

    Args:
        config: :class:`HEGPsiFormerConfig`.
        init_key: JAX PRNG key for the network parameter init.
        mcmc_key: JAX PRNG key for the pretraining MCMC chain.
            Defaults to a deterministic split from ``init_key`` so
            the pretraining stage is fully seeded by the caller's
            seed — with both keys derived from the user's YAML
            ``seed:``, changing the seed changes every random
            quantity in the pretraining stage (NN params and MCMC
            walker trajectories).
        num_iters: Adam iterations on the MSE loss.
        num_walkers: MCMC walkers (sampled from |ψ_HF|²).
        lr: Adam learning rate.
        num_equil_steps: Burn-in MCMC steps before training.
        mcmc_decorr_steps: MCMC steps between updates.
        mc_timestep: Initial Metropolis timestep.
        fname_log, verbose: Logging controls.

    Returns:
        Dict with ``'params'`` (trained pytree), ``'loss_history'``,
        ``'final_loss'``, ``'graphdef'``.
    """
    if mcmc_key is None:
        init_key, mcmc_key = jax.random.split(init_key)
    drv = _HEGPreTrainDriver(config, init_key, lr=lr)
    return drv(
        mcmc_key,
        num_iters=num_iters,
        num_walkers=num_walkers,
        num_equil_steps=num_equil_steps,
        mcmc_decorr_steps=mcmc_decorr_steps,
        mc_timestep=mc_timestep,
        fname_log=fname_log,
        verbose=verbose,
    )
