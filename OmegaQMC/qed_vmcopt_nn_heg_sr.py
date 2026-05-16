"""Stochastic-Reconfiguration VMC optimiser for the HEG trial WF.

Natural-gradient optimisation via Stochastic Reconfiguration (SR),
the standard VMC optimiser for deep neural-network wavefunctions
(Sorella 2001; used by FermiNet, PsiFormer, DeepErwin, etc.).  For
the 3D HEG Adam optimisation typically fails to escape the
initial Hartree-Fock plateau within any reasonable iter budget
because the VMC loss has a highly anisotropic Fisher metric —
Adam's diagonal rescaling is blind to this, while SR works in the
metric-aware natural-gradient direction.

The algorithm per iteration:

  1. Decorrelate walkers via Metropolis MCMC.
  2. Compute per-walker local energies ``E_L`` and Jacobian rows
     ``O_n = ∇_θ log|ψ(r_n)|`` (the ``O`` matrix).
  3. Force vector
     ``f_θ = 2 · ⟨(E_L − ⟨E_L⟩) · O_θ⟩_walkers``
     — the standard VMC energy gradient (a linear functional of
     ``O`` and ``E_L``).
  4. Solve ``(S + εI)·δθ = f`` for ``δθ`` via conjugate gradient,
     where ``S = cov(O) = ⟨(O − ⟨O⟩) · (O − ⟨O⟩)ᵀ⟩`` is the Fisher
     matrix and ε is a damping parameter.  ``S`` itself is never
     materialised — only ``S·v`` matvecs, which are O(N_walkers ·
     N_params).
  5. Update ``θ ← θ − η · δθ``.

Typical gain over Adam: convergence in 500-2000 iters to
90-99% of the trial's capacity, vs 10⁴-10⁵ iters for Adam (which
often plateaus earlier because it's direction-blind, not
step-size-blind).  Wall-clock cost per iter is 3-5× Adam due to
Jacobian computation + CG, but the iter-budget reduction
dominates.

Matches the interface of :mod:`vmcopt_nn_heg` — drop-in
replacement by swapping the import.
"""

from __future__ import annotations

import sys
from datetime import datetime
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np
from jax.flatten_util import ravel_pytree

from .psi.nn.heg_wf import (
    HEGConfig,
    make_heg_log_psi_any as make_heg_log_psi,
)
from .psi.nn.periodic import wrap_to_cell, make_cubic_lattice
from .psi.nn.physics import laplacian
from .observables.ewald import build_ewald_tables, ewald_pair_energy
from .observables.ewald_dispatch import (
    build_ewald_tables_dim, ewald_pair_energy_dim,
)


TARGET_ACCEPTANCE_RATE = 0.5
STEP_SIZE_ADAPTATION_RATE = 0.05


def _adapt_step_size(step_size, acceptance_rate):
    return step_size * (
        1.0 + STEP_SIZE_ADAPTATION_RATE
        * (acceptance_rate - TARGET_ACCEPTANCE_RATE)
    )


# ---------------------------------------------------------------------
# Step 4: paramagnetic local-energy formula (closed form, JIT-able).
# Tested independently in tests/test_qed_para_local_energy.py against
# both an analytical reference and a finite-difference ground truth.
# ---------------------------------------------------------------------

def _para_eloc_from_components(
    walkers, eps_grad_w, n_ph_local,
    phase_alpha, coh_alpha, K_vectors, eps_dot_K_vec, g, nph_max,
):
    """Velocity-gauge paramagnetic E_loc per walker.

    Step 5 ansatz (Option C, multi-K, coherent-state amplitude):
        c_n(α_coh) = α_coh^n / √n! · exp(−α_coh²/2)
        θ_n(R)    = n · Σ_k phase_alpha_k · F_k(R),  F_k(R) = Σᵢ sin(K_k·rᵢ)

    Algebraic simplifications:
        √(n+1)·(c_{n+1}/c_n) = α_coh   (Fock-state-independent!)
        √n·(c_{n−1}/c_n)     = n/α_coh
        θ_{n+1} − θ_n        = f(R) := Σ_k phase_alpha_k · F_k(R)
        ε·∂_R θ_m            = m · ∂_ε f(R)
                              = m · Σ_k phase_alpha_k · (ε·K_k) · G_k(R)

    Local energy:
        E_loc_para(R, n) = i·g · [
            α_coh · exp(+i·f) · B_{n+1}(R)
          + (n/α_coh) · exp(−i·f) · B_{n−1}(R)
        ]
    with B_m = (ε·∇R log|ψ_e|) + i·m·∂_ε f.

    Out-of-bound channels (n+1>nph_max or n−1<0) are masked to zero.

    Args:
      walkers:       (n_w, n_elec, dim)
      eps_grad_w:    (n_w,)         — ε·Σᵢ∇ᵢ log|ψ_e| per walker
      n_ph_local:    (n_w,) int     — Fock state index per walker
      phase_alpha:   (n_K,)         — phase coefficients α_k (Option C)
      coh_alpha:     scalar         — coherent-state amplitude α_coh
      K_vectors:     (n_K, dim)     — list of K vectors
      eps_dot_K_vec: (n_K,)         — ε·K_k for each k
      g:             scalar         — coupling g = λ/√(2Ω)
      nph_max:       int (static)

    Returns:
      (e_para_re, e_para_im), each (n_w,) real-valued.
    """
    # Phase features per K vector.
    K_dot_r = jnp.einsum("wid,kd->wki", walkers, K_vectors)  # (n_w, n_K, n_elec)
    F_per_k = jnp.sum(jnp.sin(K_dot_r), axis=-1)             # (n_w, n_K)
    G_per_k = jnp.sum(jnp.cos(K_dot_r), axis=-1)             # (n_w, n_K)

    # f(R) = Σ_k phase_alpha_k · F_k(R)
    f_w = F_per_k @ phase_alpha                              # (n_w,)
    # ∂_ε f(R) = Σ_k phase_alpha_k · (ε·K_k) · G_k(R)
    df_w = G_per_k @ (phase_alpha * eps_dot_K_vec)           # (n_w,)

    eps_grad_c = eps_grad_w.astype(jnp.complex128)
    f_w_c = f_w.astype(jnp.complex128)
    df_w_c = df_w.astype(jnp.complex128)

    # m_up = n+1 channel.  Mask off when n+1 > nph_max.
    n_up = n_ph_local + 1
    m_up_ok = (n_up <= nph_max).astype(jnp.complex128)
    n_up_f = n_up.astype(jnp.float64).astype(jnp.complex128)
    # B_{n+1} = ε·∇log|ψ_e| + i·(n+1)·df
    B_up = eps_grad_c + 1j * n_up_f * df_w_c
    contrib_up = coh_alpha * jnp.exp(1j * f_w_c) * B_up * m_up_ok

    # m_dn = n−1 channel.  Mask off when n=0 (√n already zero).
    n_dn = n_ph_local - 1
    m_dn_ok = (n_dn >= 0).astype(jnp.complex128)
    n_f = n_ph_local.astype(jnp.float64).astype(jnp.complex128)
    n_dn_f = jnp.maximum(n_ph_local - 1, 0).astype(jnp.float64).astype(jnp.complex128)
    # B_{n−1} = ε·∇log|ψ_e| + i·(n−1)·df
    B_dn = eps_grad_c + 1j * n_dn_f * df_w_c
    contrib_dn = (n_f / coh_alpha) * jnp.exp(-1j * f_w_c) * B_dn * m_dn_ok

    e_para_complex = 1j * g * (contrib_up + contrib_dn)
    return jnp.real(e_para_complex), jnp.imag(e_para_complex)


# ---------------------------------------------------------------------
# Conjugate gradient for S·δθ = f (S positive-definite with damping)
# ---------------------------------------------------------------------

def _cg_solve(matvec, rhs, n_iters, tol=1e-6):
    """Conjugate-gradient solver for a symmetric-positive system.

    Args:
        matvec: Callable ``v → A·v`` where ``A`` is
            symmetric-positive-definite (SPD).
        rhs: Right-hand side ``f`` of shape ``(P,)``.
        n_iters: Maximum CG iterations.
        tol: Relative residual tolerance for early termination.

    Returns:
        ``x`` with ``A·x ≈ rhs`` to the requested tolerance.

    Notes:
        Uses ``jax.lax.fori_loop`` to avoid Python-side iteration
        overhead inside JIT.  Does not early-exit inside the loop
        (would require ``lax.while_loop``); instead runs all
        ``n_iters`` iterations.  For typical SR at P~10⁵ params
        20-50 iters suffice.
    """
    def body(i, state):
        x, r, p, rr = state
        Ap = matvec(p)
        alpha = rr / (jnp.dot(p, Ap) + 1e-30)
        x_new = x + alpha * p
        r_new = r - alpha * Ap
        rr_new = jnp.dot(r_new, r_new)
        beta = rr_new / (rr + 1e-30)
        p_new = r_new + beta * p
        return (x_new, r_new, p_new, rr_new)

    x0 = jnp.zeros_like(rhs)
    r0 = rhs  # residual at x=0
    p0 = rhs  # initial conj. direction
    rr0 = jnp.dot(r0, r0)
    x, _, _, _ = jax.lax.fori_loop(
        0, n_iters, body, (x0, r0, p0, rr0),
    )
    return x


# ---------------------------------------------------------------------
# SR driver
# ---------------------------------------------------------------------

class _QEDHEGSROptimizer:
    """Natural-gradient (SR) VMC optimiser for HEG ansätze.

    Args:
        config: :class:`HEGConfig` or :class:`HEGPsiFormerConfig`.
        init_key: JAX PRNG key for parameter init.
        lr: SR learning rate ``η`` (applied to the natural-gradient
            step).  Typical: 0.05-0.1 (larger than Adam's because SR
            rescales by the Fisher metric internally).
        damping: SR damping ``ε`` — added to the diagonal of ``S``
            before CG.  Typical: 1e-3 to 1e-4.  Larger ε → step
            closer to plain gradient descent (more stable, slower);
            smaller ε → closer to pure natural gradient (faster at
            the risk of numerical instability).
        n_cg: CG iterations per SR step.  20-50 is standard.
        var_weight: Optional Umrigar-style ``β`` for the mixed
            ``⟨E⟩ + β · Var(E_L)`` objective (0 = pure energy).
        ewald_n_real, ewald_n_recip, ewald_eta: Ewald tuning.
    """

    def __init__(
        self,
        config,
        init_key,
        *,
        lr: float = 0.05,
        damping: float = 1e-3,
        n_cg: int = 30,
        var_weight: float = 0.0,
        ewald_n_real: int = 3,
        ewald_n_recip: int = 6,
        ewald_eta: Optional[float] = None,
        ofname_chkpt: Optional[str] = None,
        lr_schedule: str = 'auto',
        lr_decay_T: Optional[float] = None,
        lr_min: float = 0.0,
        lr_T_max: Optional[int] = None,
        lr_n_restarts: int = 0,
        spring_mu: float = 0.0,
        spring_norm_clip: Optional[float] = None,
        damping_adapt: bool = False,
        damping_min: float = 1.0e-5,
        damping_max: float = 1.0e-1,
        damping_factor: float = 2.0,
        damping_lookback: int = 50,
        sampler: str = 'metropolis',
        mala_grad_clip: Optional[float] = 1.0,
        save_every: int = 0,
        # ---- QED extension (Steps 2-3) ----
        omega: float = 0.0,
        nph_max: int = 0,
        fock_log_amp_slope: float = 3.0,
        # Step 3: cavity coupling. λ = |coupling_vec|, ε = direction.
        # At λ=0 (default) all cavity terms are zero — Step 3 reduces
        # exactly to Step 2.
        coupling_lambda: float = 0.0,
        coupling_polarization=None,   # tuple/list of length dim; default ε = x̂
        # Step 5: Option C multi-K phase + coherent-state amplitude.
        #
        # phase_K_vectors: list of (dim,) K vectors defining the basis for
        #   the phase ansatz θ_n(R) = n · Σ_k α_k · sin(K_k·rᵢ).  None →
        #   default to single K = 2π·ε/L (recovers Phase 4's behaviour).
        # phase_alpha_init: (n_K,) initial phase coefficients; None → zeros.
        # alpha_step_clip: per-iter clip on |δα_k| (cubic-landscape safety).
        # coh_alpha_init: initial coherent amplitude.  Default 0.05 ≈ exp(-3),
        #   matching the prior closure-baked exp(-3n) Fock weight at small n.
        # coh_alpha_step_clip: per-iter clip on |δα_coh|.
        # coh_alpha_floor: lower bound on α_coh (avoid log/division by zero).
        phase_K_vectors=None,
        phase_alpha_init=None,
        alpha_step_clip: float = 0.005,
        coh_alpha_init: float = 0.05,
        coh_alpha_step_clip: float = 0.005,
        coh_alpha_floor: float = 1.0e-3,
    ):
        self.config = config
        self.L = float(config.L)
        self.n_up = int(config.n_up)
        self.n_down = int(config.n_down)
        self.nelec = self.n_up + self.n_down
        self.dim = int(getattr(config, 'dim', 3))
        self.lr = float(lr)
        self.damping = float(damping)
        self.n_cg = int(n_cg)
        self.var_weight = float(var_weight)
        self.ofname_chkpt = ofname_chkpt
        # ---- QED extension (Step 2) ----
        # Photon Fock walker variable + ω·n photon energy.
        # Trial Fock weights c_n = exp(-slope·n) → vacuum-biased.
        # At λ=0 with n_ph initialised to 0 and the n-MCMC's vacuum
        # bias, walkers stay at n=0 almost always → ω·n_ph ≈ 0 →
        # trajectory matches bare HEG within MCMC noise.
        self.omega = float(omega)
        self.nph_max = int(nph_max)
        # ``fock_log_amp_slope`` was the slope of the prior closure-baked
        # c_n = exp(-slope·n).  Step 5 replaces this with a coherent-state
        # form c_n = α_coh^n/√n! · exp(-α_coh²/2) with trainable α_coh.
        # The legacy slope kwarg is mapped to coh_alpha_init = exp(-slope)
        # iff the caller didn't override coh_alpha_init explicitly.
        self.fock_log_amp_slope = float(fock_log_amp_slope)
        # log_factorial table for the coherent c_n.  Cheap precompute.
        import math
        self._log_fact_table = jnp.asarray(
            [math.lgamma(n + 1) for n in range(self.nph_max + 1)],
            dtype=jnp.float64,
        )
        self.coupling_lambda = float(coupling_lambda)
        if coupling_polarization is None:
            _eps = [1.0] + [0.0] * (self.dim - 1)
        else:
            _eps = list(coupling_polarization)
        _eps_arr = jnp.asarray(_eps, dtype=jnp.float64)
        _eps_norm = float(jnp.linalg.norm(_eps_arr))
        if _eps_norm > 1e-30:
            _eps_arr = _eps_arr / _eps_norm
        self._coupling_eps = _eps_arr   # unit vector
        if self.omega > 0:
            _g = self.coupling_lambda / jnp.sqrt(2.0 * self.omega)
        else:
            _g = 0.0
        _g = float(_g)
        self._coupling_g = _g
        # Per-electron coefficient on the diamagnetic bb_sq term.
        self._diamag_coeff = 0.5 * self.nelec * _g * _g

        # ---- Step 5: Option C multi-K phase basis ----
        # Default to a single K = 2π·ε/L (recovers Phase 4 single-K).
        if phase_K_vectors is None:
            _K_default = (
                2.0 * jnp.pi * _eps_arr / self.L
            )
            self._K_vectors = _K_default[None, :]            # (1, dim)
        else:
            self._K_vectors = jnp.asarray(
                phase_K_vectors, dtype=jnp.float64,
            )                                                # (n_K, dim)
            if self._K_vectors.shape[-1] != self.dim:
                raise ValueError(
                    f"phase_K_vectors must have shape (n_K, {self.dim}); "
                    f"got {self._K_vectors.shape}"
                )
        n_K = int(self._K_vectors.shape[0])
        self._n_K = n_K
        # ε·K_k for each k (used for ∂_ε θ projection).
        self._eps_dot_K_vec = self._K_vectors @ _eps_arr     # (n_K,)

        # ---- Step 5: phase-coefficient state α_k (shape (n_K,))  ----
        # α_0 (Fock vacuum) is implicitly pinned by the n·... prefactor.
        if phase_alpha_init is None:
            self.phase_alpha = jnp.zeros(n_K, dtype=jnp.float64)
        else:
            init_arr = jnp.asarray(phase_alpha_init, dtype=jnp.float64)
            if init_arr.shape != (n_K,):
                raise ValueError(
                    f"phase_alpha_init must have shape ({n_K},), "
                    f"got {init_arr.shape}"
                )
            self.phase_alpha = init_arr
        self.alpha_step_clip = float(alpha_step_clip)

        # ---- Step 5: coherent-state amplitude α_coh ----
        # c_n = α_coh^n / √n! · exp(-α_coh²/2).  α_coh is real, positive,
        # floored at coh_alpha_floor.  Updated by its own scalar-SR step
        # (Re(E_loc) Jacobian, see __call__).
        self.coh_alpha = jnp.asarray(
            max(float(coh_alpha_init), float(coh_alpha_floor)),
            dtype=jnp.float64,
        )
        self.coh_alpha_step_clip = float(coh_alpha_step_clip)
        self.coh_alpha_floor = float(coh_alpha_floor)
        # ---- Learning-rate scheduler ----
        # Schedule names:
        #   'fixed'   : constant lr (vanilla SR).
        #   'inverse' : Smith-2024-style lr_t = lr / (1 + t / T).
        #               Set ``lr_decay_T``.
        #   'cosine'  : Cosine annealing from lr (max) to lr_min over
        #               lr_T_max iters.  ``lr_n_restarts``=0 → simple
        #               cosine; >0 → SGDR-style equal-length cycles.
        # 'auto' (default) → 'inverse' iff lr_decay_T set, else 'fixed'.
        # All schedules: at iter > T_max stay at the floor (lr_min for
        # cosine; lr/(1+T_max/T) for inverse).
        if lr_schedule == 'auto':
            lr_schedule = 'inverse' if lr_decay_T is not None else 'fixed'
        if lr_schedule not in ('fixed', 'inverse', 'cosine'):
            raise ValueError(
                f"lr_schedule must be 'fixed'/'inverse'/'cosine', "
                f"got {lr_schedule!r}"
            )
        self.lr_schedule = lr_schedule
        self.lr_decay_T = (
            None if lr_decay_T is None else float(lr_decay_T)
        )
        self.lr_min = float(lr_min)
        self.lr_T_max = (
            None if lr_T_max is None else int(lr_T_max)
        )
        self.lr_n_restarts = int(lr_n_restarts)
        if self.lr_schedule == 'cosine' and self.lr_T_max is None:
            raise ValueError(
                "lr_schedule='cosine' requires lr_T_max (set it to the "
                "total number of optimization iters, or to the desired "
                "cycle length when used with lr_n_restarts>0)"
            )
        # SPRING (Goldshlager 2024) momentum coefficient.  0 → vanilla
        # SR.  Smith uses 0.9.  Formula:
        #     dθ_t = (S + λI)^{-1} (g + λμ dθ_{t-1})
        self.spring_mu = float(spring_mu)
        # F-norm clip on dθ (KFAC-style).  Smith uses C = η_0 (the
        # initial lr).  None → take initial lr; 0 → disabled; float →
        # explicit value.
        if spring_norm_clip is None:
            self.spring_norm_clip = float(self.lr)
        elif float(spring_norm_clip) <= 0.0:
            self.spring_norm_clip = 0.0  # disabled
        else:
            self.spring_norm_clip = float(spring_norm_clip)
        # Adaptive damping (Levenberg-Marquardt-style).  Off by
        # default — Smith holds λ fixed at 0.001.
        self.damping_adapt = bool(damping_adapt)
        self.damping_min = float(damping_min)
        self.damping_max = float(damping_max)
        self.damping_factor = float(damping_factor)
        self.damping_lookback = int(damping_lookback)
        # MCMC sampler: 'metropolis' (Gaussian random walk, default) or
        # 'mala' (Metropolis-adjusted Langevin, Smith 2024).
        self.sampler = str(sampler).lower()
        if self.sampler not in ('metropolis', 'mala'):
            raise ValueError(
                f"sampler must be 'metropolis' or 'mala', got {sampler!r}"
            )
        self.mala_grad_clip = (
            None if mala_grad_clip is None else float(mala_grad_clip)
        )
        # Periodic checkpoint save during training (in addition to the
        # one written at end-of-training).  0 → save only at end (old
        # behaviour).  Recommended: 500 — limits worst-case data loss
        # on SIGTERM/SIGINT to ~500 iters.
        self.save_every = int(save_every)

        if self.dim == 3:
            self.lattice = make_cubic_lattice(self.L)
        else:
            from .psi.nn.periodic import make_square_lattice
            self.lattice = make_square_lattice(self.L)
        self.ewald = build_ewald_tables_dim(
            self.L, dim=self.dim, eta=ewald_eta,
            n_real=ewald_n_real, n_recip=ewald_n_recip,
        )

        log_psi_pytree, init_params, graphdef = make_heg_log_psi(
            config, init_key,
        )
        self.log_psi_pytree = log_psi_pytree
        self.graphdef = graphdef

        # Flatten the parameter pytree once — SR works on the flat
        # representation so the Jacobian is a plain ``(N, P)`` matrix.
        p_flat, unravel = ravel_pytree(init_params)
        self.params_flat = p_flat
        self.unravel = unravel
        self.n_params = int(p_flat.shape[0])

        def log_psi_flat(r, p_flat):
            return log_psi_pytree(r, unravel(p_flat))

        self.log_psi_flat = log_psi_flat

        tables = self.ewald
        lattice = self.lattice
        nelec = self.nelec
        dim = self.dim

        # Per-walker local energy (kin + pot) plus ε·Σᵢ∇ᵢ log|ψ_e|
        # which is the electronic-gradient input to the paramagnetic
        # local energy (Step 4 Phase 1).  The gradient is already
        # computed inside ``laplacian``, so the projection is free.
        eps_const = jnp.asarray(
            self._coupling_eps, dtype=jnp.float64,
        )                                                          # (dim,)

        def kin_and_eps_grad(r, p_flat):
            def f_flat(r_flat):
                return log_psi_flat(
                    r_flat.reshape(nelec, dim), p_flat,
                )
            r_flat = r.reshape(-1)
            lap_val, grad_val = laplacian(f_flat)(r_flat)
            e_kin = -0.5 * (lap_val + jnp.dot(grad_val, grad_val))
            # ε·Σᵢ∇ᵢ log|ψ_e| = Σ_{i,d} grad[i,d]·ε[d]
            eps_dot_grad = jnp.einsum(
                "id,d->", grad_val.reshape(nelec, dim), eps_const,
            )
            return e_kin, eps_dot_grad

        # Backwards-compat alias for the kin-only path (used by MCMC
        # tests that don't need the gradient projection).
        def kin_only(r, p_flat):
            e_kin, _ = kin_and_eps_grad(r, p_flat)
            return e_kin

        # MCMC step (same as Adam path).
        def metropolis_move(rng_key, r, step_size, p_flat):
            key_prop, key_acc = jax.random.split(rng_key)
            proposed = r + step_size * jax.random.normal(
                key_prop, r.shape,
            )
            proposed = wrap_to_cell(proposed, lattice)
            lp_old = log_psi_flat(r, p_flat)
            lp_new = log_psi_flat(proposed, p_flat)
            accept = jax.random.uniform(key_acc) < jnp.exp(
                2.0 * (lp_new - lp_old),
            )
            new = jnp.where(accept, proposed, r)
            return new, accept

        # Multi-GPU: each device handles n_walkers/n_devices walkers.
        # Outer pmap distributes across devices (axis_name='dev'),
        # inner vmap parallelises over the device-local walker batch.
        # in_axes=(0, 0, None, None): rng_key + walkers are sharded
        # along device axis; step_size + p_flat are replicated.
        self._metropolis_move_pmap = jax.pmap(
            jax.vmap(metropolis_move, in_axes=(0, 0, None, None)),
            in_axes=(0, 0, None, None),
            axis_name='dev',
        )

        # ---- MALA (Metropolis-adjusted Langevin) sampler ------------
        # Smith 2024 Eq. 22:
        #   R̃ = R + τ ∇ log|Ψ|² + sqrt(2τ) ξ
        # where ξ ~ N(0, I) and the ∇ log|Ψ|² = 2 ∇ log|Ψ|.  Use the
        # log-psi gradient already available, multiplied by 2.  The
        # Metropolis-Hastings ratio includes both the wavefunction
        # ratio and the Langevin proposal asymmetry.
        log_psi_grad_r = jax.grad(log_psi_flat, argnums=0)
        mala_clip = self.mala_grad_clip

        def mala_move(rng_key, r, step_size, p_flat):
            # Convention: step_size in MALA = sqrt(2 τ).  τ = ½ step².
            tau = 0.5 * step_size * step_size

            key_prop, key_acc = jax.random.split(rng_key)
            grad_r = log_psi_grad_r(r, p_flat)
            drift_r = 2.0 * grad_r          # ∇ log|Ψ|² = 2 ∇ log|Ψ|
            if mala_clip is not None:
                gnorm = jnp.sqrt(jnp.sum(drift_r ** 2) + 1e-12)
                scale = jnp.minimum(1.0, mala_clip / gnorm)
                drift_r = drift_r * scale

            noise = jax.random.normal(key_prop, r.shape)
            proposed = r + tau * drift_r + step_size * noise
            proposed = wrap_to_cell(proposed, lattice)

            # Reverse drift for the M-H ratio.
            grad_p = log_psi_grad_r(proposed, p_flat)
            drift_p = 2.0 * grad_p
            if mala_clip is not None:
                gnorm_p = jnp.sqrt(jnp.sum(drift_p ** 2) + 1e-12)
                scale_p = jnp.minimum(1.0, mala_clip / gnorm_p)
                drift_p = drift_p * scale_p

            lp_old = log_psi_flat(r, p_flat)
            lp_new = log_psi_flat(proposed, p_flat)

            # log q(R | R̃) − log q(R̃ | R) (Gaussian, mean = R + τ drift).
            d_fwd = proposed - r - tau * drift_r
            d_bwd = r - proposed - tau * drift_p
            two_tau = 2.0 * tau
            log_q_ratio = (
                jnp.sum(d_fwd ** 2) - jnp.sum(d_bwd ** 2)
            ) / (2.0 * two_tau)

            log_alpha = 2.0 * (lp_new - lp_old) + log_q_ratio
            accept = jax.random.uniform(key_acc) < jnp.exp(log_alpha)
            new = jnp.where(accept, proposed, r)
            return new, accept

        self._mala_move_pmap = jax.pmap(
            jax.vmap(mala_move, in_axes=(0, 0, None, None)),
            in_axes=(0, 0, None, None),
            axis_name='dev',
        )

        log_psi_grad_flat = jax.grad(log_psi_flat, argnums=1)

        # Ewald potential: chunked over walkers to avoid XLA's
        # fusion bomb at (n_real=3, n_recip=6).  See vmcopt_nn_heg.
        if dim == 3:
            pot_fn = lambda w: ewald_pair_energy(w, tables)
        else:
            from .observables.ewald_2d import ewald_2d_pair_energy
            pot_fn = lambda w: ewald_2d_pair_energy(w, tables)

        # Multi-GPU device set-up.  pmap distributes across leading
        # axis = device axis.
        self.devices = jax.devices()
        self.n_devices = len(self.devices)

        # Per-device kinetic+potential (no cross-device communication
        # needed; each device computes its own walkers' E_loc).
        omega_const = self.omega
        diamag_coeff_const = self._diamag_coeff
        g_para_const = jnp.asarray(
            self._coupling_g, dtype=jnp.float64,
        )
        K_vectors_const = self._K_vectors
        eps_dot_K_vec_const = self._eps_dot_K_vec
        nph_max_static = int(self.nph_max)
        n_arr_const = jnp.arange(
            self.nph_max + 1, dtype=jnp.float64,
        )

        def _bb_sq_from_coh_alpha(coh_alpha):
            """Diamagnetic table at runtime, from coherent-state c_n.
                bb_sq[n] = (2n+1)
                         + α_coh² · I_{n+2≤nph_max}
                         + n(n−1)/α_coh² · I_{n≥2}
            """
            p2_mask = (
                n_arr_const + 2.0 <= nph_max_static
            ).astype(jnp.float64)
            m2_mask = (n_arr_const >= 2.0).astype(jnp.float64)
            return (
                (2.0 * n_arr_const + 1.0)
                + (coh_alpha ** 2) * p2_mask
                + (n_arr_const * (n_arr_const - 1.0))
                / (coh_alpha ** 2)
                * m2_mask
            )

        def kin_pot_local(
            walkers, p_flat, n_ph_local, phase_alpha, coh_alpha,
        ):
            (e_kin, eps_grad_w) = jax.vmap(
                kin_and_eps_grad, in_axes=(0, None),
            )(walkers, p_flat)
            e_pot = pot_fn(walkers)
            # Photon energy with Ω/2 zero-point.  Constant offset, no SR.
            e_phot = omega_const * (
                n_ph_local.astype(e_kin.dtype) + 0.5
            )
            bb_sq = _bb_sq_from_coh_alpha(coh_alpha)
            e_diamag = (
                diamag_coeff_const
                * bb_sq[n_ph_local].astype(e_kin.dtype)
            )
            # Full paramagnetic E_loc formula (Option C multi-K).
            e_para_re, _ = _para_eloc_from_components(
                walkers, eps_grad_w, n_ph_local,
                phase_alpha, coh_alpha,
                K_vectors_const, eps_dot_K_vec_const,
                g_para_const, nph_max_static,
            )
            return e_kin + e_pot + e_phot + e_diamag + e_para_re

        def kin_pot_breakdown(
            walkers, p_flat, n_ph_local, phase_alpha, coh_alpha,
        ):
            (e_kin, eps_grad_w) = jax.vmap(
                kin_and_eps_grad, in_axes=(0, None),
            )(walkers, p_flat)
            e_pot = pot_fn(walkers)
            e_phot = omega_const * (
                n_ph_local.astype(e_kin.dtype) + 0.5
            )
            bb_sq = _bb_sq_from_coh_alpha(coh_alpha)
            e_diamag = (
                diamag_coeff_const
                * bb_sq[n_ph_local].astype(e_kin.dtype)
            )
            e_para_re, e_para_im = _para_eloc_from_components(
                walkers, eps_grad_w, n_ph_local,
                phase_alpha, coh_alpha,
                K_vectors_const, eps_dot_K_vec_const,
                g_para_const, nph_max_static,
            )
            return (
                e_kin, e_pot, e_phot, e_diamag, e_para_re, e_para_im,
            )

        # in_axes adds None for phase_alpha and coh_alpha (replicated).
        self._eloc_pmap = jax.pmap(
            kin_pot_local,
            in_axes=(0, None, 0, None, None),
            axis_name='dev',
        )
        self._eloc_breakdown_pmap = jax.pmap(
            kin_pot_breakdown,
            in_axes=(0, None, 0, None, None),
            axis_name='dev',
        )

        # ---- SR step (pmap'd, all-gather o across devices) ----
        var_weight_arr = jnp.asarray(
            self.var_weight, dtype=jnp.float64,
        )

        # Closure-capture the global walker count (statically known
        # at __call__ time; we re-create the pmap'd function in
        # __call__ so this is a Python int, suitable for jnp.eye).
        n_devices_static = self.n_devices

        def make_sr_step(num_walkers):
            n_w_local = num_walkers // n_devices_static

            def sr_step(
                walkers, p_flat, e_loc,
                prev_dtheta, damping, lambda_mu, c_clip,
            ):
                """One SR step, per-device.  Cross-device collectives
                via psum / all_gather under axis_name='dev'.

                Inputs (per device):
                  walkers:    (n_w_local, nelec, dim)
                  p_flat:     (n_p,)              replicated
                  e_loc:      (n_w_local,)
                  prev_dtheta:(n_p,)              replicated
                  damping, lambda_mu, c_clip:     scalars, replicated

                Returns (per device, all replicated outputs):
                  dtheta:    (n_p,)
                  e_mean:    scalar
                  var:       scalar
                  g_norm:    scalar
                  scale:     scalar
                """
                # Per-walker o = ∇_θ log|Ψ| (local walkers).
                o_local = jax.vmap(
                    log_psi_grad_flat, in_axes=(0, None),
                )(walkers, p_flat)                          # (n_w_local, n_p)

                # Global mean energy (across all devices).
                e_sum_local = jnp.sum(e_loc)
                e_sum = jax.lax.psum(e_sum_local, axis_name='dev')
                e_mean = e_sum / num_walkers
                de_local = e_loc - e_mean
                var_sum = jax.lax.psum(
                    jnp.sum(de_local ** 2), axis_name='dev',
                )
                var = var_sum / num_walkers

                # Global mean of o (across all devices).
                o_sum_local = jnp.sum(o_local, axis=0)      # (n_p,)
                o_sum = jax.lax.psum(o_sum_local, axis_name='dev')
                o_mean = o_sum / num_walkers
                do_local = o_local - o_mean[None, :]        # (n_w_local, n_p)

                # Force g (gradient).
                de_mix_local = de_local + var_weight_arr * (
                    de_local ** 2 - var
                )
                g_partial = de_mix_local @ do_local         # (n_p,)
                g_full = jax.lax.psum(g_partial, axis_name='dev')
                g = 2.0 * g_full / num_walkers

                # SPRING RHS.
                rhs = g + lambda_mu * prev_dtheta

                # ===== SMW dual-form SR =====
                # Need full do (n_w, n_p) on every device for the
                # K = do @ do.T solve.  all_gather along device axis
                # tiles the local do shards into the global matrix.
                # Memory: n_w × n_p × 8 bytes (e.g. 1024 × 53k × 8 =
                # 437 MB at production, fits easily in 80 GB A100).
                do = jax.lax.all_gather(
                    do_local, axis_name='dev', tiled=True,
                )                                           # (n_w, n_p)

                inv_λn = 1.0 / (damping * num_walkers)
                K = (do @ do.T) * inv_λn                    # (n_w, n_w)
                I_plus_K = K + jnp.eye(num_walkers, dtype=K.dtype)
                o_rhs = (do @ rhs) * inv_λn                 # (n_w,)
                u = jnp.linalg.solve(I_plus_K, o_rhs)       # (n_w,)
                dtheta = (rhs - do.T @ u) / damping         # (n_p,)

                # F-norm clip: S dθ via (1/n) do^T (do dθ).
                s_dtheta = (do.T @ (do @ dtheta)) / num_walkers
                f_norm_sq = jnp.maximum(
                    jnp.dot(dtheta, s_dtheta), 1e-20,
                )
                f_norm = jnp.sqrt(f_norm_sq)
                raw_scale = c_clip / (f_norm + 1e-20)
                scale = jnp.where(
                    c_clip > 0.0,
                    jnp.minimum(1.0, raw_scale),
                    jnp.ones_like(raw_scale),
                )
                dtheta = scale * dtheta

                g_norm = jnp.sqrt(jnp.sum(g ** 2))
                return dtheta, e_mean, var, g_norm, scale

            return jax.pmap(
                sr_step,
                in_axes=(0, None, 0, None, None, None, None),
                axis_name='dev',
            )

        self._make_sr_step = make_sr_step

    # -----------------------------------------------------
    # Learning-rate scheduler
    # -----------------------------------------------------

    def _compute_lr(self, it: int) -> float:
        """Return effective lr at iteration ``it`` (1-indexed).

        Dispatches on ``self.lr_schedule``:
          * 'fixed'   → ``self.lr``
          * 'inverse' → ``self.lr / (1 + (it-1) / lr_decay_T)``
          * 'cosine'  → cosine anneal (with optional warm restarts)
        """
        import math
        lr_max = self.lr
        if self.lr_schedule == 'fixed':
            return lr_max
        if self.lr_schedule == 'inverse':
            T = self.lr_decay_T
            return lr_max / (1.0 + (it - 1) / T)
        # cosine (with optional restarts)
        T_max = float(self.lr_T_max)
        n_cycles = max(1, self.lr_n_restarts + 1)
        T_cycle = T_max / n_cycles
        t = float(it - 1)
        if t >= T_max:
            return self.lr_min
        cycle_idx = int(t // T_cycle)
        t_in_cycle = t - cycle_idx * T_cycle
        phase = t_in_cycle / T_cycle      # in [0, 1)
        return self.lr_min + 0.5 * (lr_max - self.lr_min) * (
            1.0 + math.cos(math.pi * phase)
        )

    # -----------------------------------------------------
    # Checkpoint helper (re-used by periodic save + end-of-training)
    # -----------------------------------------------------

    def _save_checkpoint(self, params_flat, epoch, energy=None,
                         note: str = "", path: Optional[str] = None):
        """Persist params + meta to ``path`` (default ``self.ofname_chkpt``).

        Failures are logged, never raised — long training runs must
        not crash on a transient I/O hiccup.
        """
        target = path if path is not None else self.ofname_chkpt
        if not target:
            return
        from .nn_checkpoint import save_nn_checkpoint

        class _HEGMolInfoStub:
            def __init__(self, n_up, n_down, dim):
                self.n_up = int(n_up)
                self.n_down = int(n_down)
                self.charges = np.zeros((0,), dtype=np.float64)
                self.coords = np.zeros((0, dim), dtype=np.float64)

        stub = _HEGMolInfoStub(self.n_up, self.n_down, self.dim)
        try:
            save_nn_checkpoint(
                target,
                self.unravel(params_flat),
                epoch=int(epoch),
                config_name='HEG_PsiFormer',
                mol_info=stub,
                energy=energy,
            )
            return True
        except Exception as e:
            print(
                f"[warn] checkpoint save failed "
                f"({target}{', ' + note if note else ''}): {e}",
            )
            return False

    # -----------------------------------------------------
    # Walker management
    # -----------------------------------------------------

    def initialize_walkers(self, rng_key, num_walkers):
        """Sample walker positions for the SR optimiser.

        Dispatch order:
        1. Explicit ``config.walker_init`` (``'uniform'`` or
           ``'crystal_perturbed'``) wins.  Smith 2024 uses
           ``crystal_perturbed`` for both fluid and crystal phases.
        2. Otherwise fall back to envelope-based default: crystal_gaussian
           → crystal_perturbed; everything else → uniform.
        """
        walker_init = str(getattr(
            self.config, 'walker_init', 'auto',
        )).lower()
        envelope_type = getattr(self.config, 'envelope_type', 'plane_wave')

        if walker_init == 'auto':
            walker_init = (
                'crystal_perturbed'
                if (envelope_type == 'crystal_gaussian' and self.dim == 2)
                else 'uniform'
            )

        if walker_init == 'crystal_perturbed' and self.dim == 2:
            from .psi.nn.env_localized_2d import crystal_init_walkers_2d
            return crystal_init_walkers_2d(
                rng_key, num_walkers,
                n_up=self.n_up, n_down=self.n_down, L=self.L,
                sigma_init=float(getattr(
                    self.config, 'crystal_sigma_init', 0.25,
                )),
                spin_pattern=str(getattr(
                    self.config, 'crystal_spin_pattern', 'neel',
                )),
            )
        return self.L * jax.random.uniform(
            rng_key, (num_walkers, self.nelec, self.dim),
        )

    # -----------------------------------------------------
    # MCMC primitives shared by training and eval
    # -----------------------------------------------------

    def _build_step_fns(self, num_walkers: int):
        """Return jitted (R-step, n-step) MCMC functions for the
        given ``num_walkers``.

        Both functions capture ``n_w_local = num_walkers / n_devices``
        as a Python static so XLA can specialise. Called once each by
        training and by ``evaluate``; the JIT cache makes repeated
        calls with the same shape free.
        """
        n_dev = self.n_devices
        if num_walkers % n_dev != 0:
            raise ValueError(
                f"num_walkers ({num_walkers}) must be divisible by "
                f"n_devices ({n_dev}) for pmap walker sharding."
            )
        n_w_local = num_walkers // n_dev

        if self.sampler == 'mala':
            mcmc_move_pmap = self._mala_move_pmap
        else:
            mcmc_move_pmap = self._metropolis_move_pmap

        nph_max_static = int(self.nph_max)
        log_fact_static = self._log_fact_table

        @jax.jit
        def _mcmc_step(rng_key, walkers, step_size, params_flat):
            keys_per_dev = jax.vmap(
                lambda k: jax.random.split(k, n_w_local),
            )(rng_key)
            walkers, acc = mcmc_move_pmap(
                keys_per_dev, walkers, step_size, params_flat,
            )
            ar = jnp.mean(acc).astype(jnp.float64)
            new_step = _adapt_step_size(step_size, ar)
            return walkers, new_step, ar

        @jax.jit
        def _n_mcmc_step(rng_key, n_ph, coh_alpha):
            """Photon-Fock Metropolis using coherent-state c_n.

            log|c_n|² = 2·(n·log(α_coh) − ½·log(n!) − α_coh²/2)
            log_acc(n→n') = 2·(log c_{n'} − log c_n)
                           = 2·(n'−n)·log(α_coh) − log(n'!) + log(n!)
            """
            log_alpha = jnp.log(coh_alpha)

            def one(key, n):
                kp, ka = jax.random.split(key)
                delta = (
                    2 * jax.random.bernoulli(kp).astype(jnp.int32) - 1
                )
                n_prop = n + delta
                in_bounds = (
                    (n_prop >= 0) & (n_prop <= nph_max_static)
                )
                n_eval = jnp.where(in_bounds, n_prop, n)
                # log_c_n_diff = (n_eval−n)·log(α_coh)
                #                − ½·(log_fact[n_eval] − log_fact[n])
                # → log_acc = 2 · log_c_n_diff
                log_c_diff = (
                    (n_eval - n).astype(jnp.float64) * log_alpha
                    - 0.5 * (
                        log_fact_static[n_eval] - log_fact_static[n]
                    )
                )
                log_acc = 2.0 * log_c_diff
                accept = (
                    (jnp.log(jax.random.uniform(ka)) < log_acc)
                    & in_bounds
                )
                return jnp.where(accept, n_prop, n), accept
            keys_per_dev = jax.vmap(
                lambda k: jax.random.split(k, n_w_local),
            )(rng_key)
            n_new, acc = jax.vmap(jax.vmap(one))(keys_per_dev, n_ph)
            ar_n = jnp.mean(acc).astype(jnp.float64)
            return n_new, ar_n

        return _mcmc_step, _n_mcmc_step

    # -----------------------------------------------------
    # Evaluation loop (non-gradient).  Same MCMC + E_loc as training,
    # no SR step.  Reports the full Pauli-Fierz local energy with the
    # photonic + diamagnetic contributions baked in.
    # -----------------------------------------------------

    def evaluate(
        self,
        rng_key,
        params_flat=None,
        num_walkers: int = 256,
        num_blocks: int = 50,
        num_blocks_equil: int = 10,
        num_steps_per_block: int = 30,
        mc_timestep: float = 0.1,
        fname_log: Optional[str] = None,
        verbose: int = 1,
    ):
        """Block-averaged eval-mode VMC for the full QED Hamiltonian.

        Initialises walkers at ``n_ph = 0`` (cavity vacuum), runs the
        same composite (R, n) Metropolis chain as training without
        gradient updates, and returns per-block ``E_loc`` plus the
        kin/pot/phot/diamag breakdown and ``⟨n_ph⟩``.
        """
        from .utils import do_binning_analysis

        if isinstance(rng_key, int):
            rng_key = jax.random.key(rng_key)
        if (fname_log is None
                or (isinstance(fname_log, str) and fname_log == "")):
            fout = sys.stdout
        else:
            fout = open(fname_log, 'w', 1)

        if params_flat is None:
            params_flat = self.params_flat

        n_dev = self.n_devices
        if num_walkers % n_dev != 0:
            raise ValueError(
                f"num_walkers ({num_walkers}) must be divisible by "
                f"n_devices ({n_dev}) for pmap walker sharding."
            )
        n_w_local = num_walkers // n_dev

        _mcmc_step, _n_mcmc_step = self._build_step_fns(num_walkers)
        eloc_breakdown_pmap = self._eloc_breakdown_pmap

        rng_key, init_key = jax.random.split(rng_key)
        walkers = self.initialize_walkers(init_key, num_walkers)
        walkers = walkers.reshape(
            n_dev, n_w_local, self.nelec, self.dim,
        )
        n_ph_walkers = jnp.zeros(
            (n_dev, n_w_local), dtype=jnp.int32,
        )

        step_size = jnp.asarray(
            (3 * mc_timestep) ** 0.5, dtype=jnp.float64,
        )

        # Equilibration.
        ar_jax = jnp.zeros((), dtype=jnp.float64)
        for blk in range(num_blocks_equil):
            for _s in range(num_steps_per_block):
                rng_key, sub = jax.random.split(rng_key)
                keys_dev = jax.random.split(sub, n_dev)
                walkers, step_size, ar_jax = _mcmc_step(
                    keys_dev, walkers, step_size, params_flat,
                )
                rng_key, sub_n = jax.random.split(rng_key)
                keys_n_dev = jax.random.split(sub_n, n_dev)
                n_ph_walkers, _ = _n_mcmc_step(
                    keys_n_dev, n_ph_walkers, self.coh_alpha,
                )

        if verbose >= 1:
            print(
                f"# Equilibration acceptance (last block): "
                f"{float(ar_jax):.3f}",
                file=fout,
            )
            print(
                f"# Adapted step size: {float(step_size):.4f} Bohr",
                file=fout,
            )
            print(
                "# block   E_loc_mean        E_loc_std       E_kin"
                "            E_pot            E_phot          "
                "E_diamag         E_para_re        E_para_im"
                "        <n_ph>     Δt_block",
                file=fout,
            )

        E_blocks = []
        kin_blocks, pot_blocks = [], []
        phot_blocks, diamag_blocks = [], []
        para_re_blocks, para_im_blocks = [], []
        nph_blocks = []
        timestamp_init = datetime.now()
        timestamp_prev = timestamp_init

        for blk in range(1, num_blocks + 1):
            e_kin_steps, e_pot_steps = [], []
            e_phot_steps, e_diamag_steps = [], []
            e_para_re_steps, e_para_im_steps = [], []
            n_ph_steps = []
            for _s in range(num_steps_per_block):
                rng_key, sub = jax.random.split(rng_key)
                keys_dev = jax.random.split(sub, n_dev)
                walkers, step_size, _ = _mcmc_step(
                    keys_dev, walkers, step_size, params_flat,
                )
                rng_key, sub_n = jax.random.split(rng_key)
                keys_n_dev = jax.random.split(sub_n, n_dev)
                n_ph_walkers, _ = _n_mcmc_step(
                    keys_n_dev, n_ph_walkers, self.coh_alpha,
                )
                (
                    e_kin, e_pot, e_phot, e_diamag,
                    e_para_re, e_para_im,
                ) = eloc_breakdown_pmap(
                    walkers, params_flat, n_ph_walkers,
                    self.phase_alpha, self.coh_alpha,
                )
                e_kin_steps.append(e_kin)
                e_pot_steps.append(e_pot)
                e_phot_steps.append(e_phot)
                e_diamag_steps.append(e_diamag)
                e_para_re_steps.append(e_para_re)
                e_para_im_steps.append(e_para_im)
                n_ph_steps.append(n_ph_walkers)

            e_kin = jnp.stack(e_kin_steps)
            e_pot = jnp.stack(e_pot_steps)
            e_phot = jnp.stack(e_phot_steps)
            e_diamag = jnp.stack(e_diamag_steps)
            e_para_re = jnp.stack(e_para_re_steps)
            e_para_im = jnp.stack(e_para_im_steps)
            n_ph_stk = jnp.stack(n_ph_steps).astype(jnp.float64)

            # Re(E_loc) goes into the variational sum; Im is diagnostic.
            E_loc = e_kin + e_pot + e_phot + e_diamag + e_para_re
            E_step = E_loc.mean(axis=(1, 2))      # (steps,)
            E_mean = E_step.mean()
            E_std = E_step.std()
            E_blocks.append(float(E_mean))
            kin_blocks.append(float(e_kin.mean()))
            pot_blocks.append(float(e_pot.mean()))
            phot_blocks.append(float(e_phot.mean()))
            diamag_blocks.append(float(e_diamag.mean()))
            para_re_blocks.append(float(e_para_re.mean()))
            para_im_blocks.append(float(e_para_im.mean()))
            nph_blocks.append(float(n_ph_stk.mean()))

            now = datetime.now()
            dt = (now - timestamp_prev).total_seconds()
            timestamp_prev = now
            if verbose >= 1:
                print(
                    f"{blk:>7d}  {float(E_mean):>16.8e}  "
                    f"{float(E_std):>14.6e}  "
                    f"{float(e_kin.mean()):>14.6e}  "
                    f"{float(e_pot.mean()):>14.6e}  "
                    f"{float(e_phot.mean()):>14.6e}  "
                    f"{float(e_diamag.mean()):>14.6e}  "
                    f"{float(e_para_re.mean()):>14.6e}  "
                    f"{float(e_para_im.mean()):>14.6e}  "
                    f"{float(n_ph_stk.mean()):>10.4e}  "
                    f"{dt:>10.4f}",
                    file=fout,
                )

        if not (fname_log is None
                or (isinstance(fname_log, str) and fname_log == "")):
            fout.close()

        E_arr = jnp.array(E_blocks)
        e_mean, e_serr, _, e_kappa = do_binning_analysis(E_arr)
        N = self.nelec
        e_ha = float(e_mean) / N
        e_ry = e_ha * 2.0

        elapsed = (datetime.now() - timestamp_init).total_seconds()
        if verbose >= 1:
            print(
                f"\nVMC QED energy: {float(e_mean):.8f} +/- "
                f"{float(e_serr):.8f} Ha "
                f"({float(E_arr.shape[0] / e_kappa):.1f} eff blocks)",
            )
            print(f"E_QED/N = {e_ha:.8f} Ha/elec  = "
                  f"{e_ry:.8f} Ry/elec")
            para_re_mean = float(np.mean(para_re_blocks))
            para_im_mean = float(np.mean(para_im_blocks))
            para_im_serr = float(np.std(para_im_blocks)
                                  / np.sqrt(len(para_im_blocks)))
            phase_str = ", ".join(
                f"a{k}={float(self.phase_alpha[k]):+.4e}"
                for k in range(len(self.phase_alpha))
            )
            print(
                f"  <E_kin>     = {float(np.mean(kin_blocks)):+.8e} Ha\n"
                f"  <E_pot>     = {float(np.mean(pot_blocks)):+.8e} Ha\n"
                f"  <E_phot>    = {float(np.mean(phot_blocks)):+.8e} Ha\n"
                f"  <E_diamag>  = {float(np.mean(diamag_blocks)):+.8e} Ha\n"
                f"  <E_para_re> = {para_re_mean:+.8e} Ha   "
                f"(SR uses Re(E_loc))\n"
                f"  <E_para_im> = {para_im_mean:+.4e} ± "
                f"{para_im_serr:.4e} Ha   "
                f"(Hermiticity: must average to 0)\n"
                f"  <n_ph>      = {float(np.mean(nph_blocks)):.6e}\n"
                f"  α_coh       = {float(self.coh_alpha):+.6e}\n"
                f"  phase_α     = [{phase_str}]"
            )
            print(f"Total eval time: {elapsed:.2f} s")

        return {
            'E_mean': float(e_mean),
            'E_serr': float(e_serr),
            'E_blocks': E_blocks,
            'E_per_elec_ha': e_ha,
            'E_per_elec_ry': e_ry,
            'E_kin_blocks': kin_blocks,
            'E_pot_blocks': pot_blocks,
            'E_phot_blocks': phot_blocks,
            'E_diamag_blocks': diamag_blocks,
            'E_para_re_blocks': para_re_blocks,
            'E_para_im_blocks': para_im_blocks,
            'n_ph_blocks': nph_blocks,
            'phase_alpha': np.asarray(self.phase_alpha).tolist(),
            'coh_alpha': float(self.coh_alpha),
        }

    # -----------------------------------------------------
    # Training loop
    # -----------------------------------------------------

    # NOTE: _pot_batch removed in pmap port — kinetic+potential
    # are computed together by the pmap'd ``_eloc_pmap``.

    def __call__(
        self,
        rng_key,
        num_iters: int = 500,
        num_walkers: int = 256,
        mcmc_decorr_steps: int = 20,
        num_equil_steps: int = 400,
        mc_timestep: float = 0.1,
        fname_log: Optional[str] = None,
        verbose: int = 1,
    ):
        """Run SR-VMC optimisation.

        Args:
            rng_key: JAX PRNG key (int or array).
            num_iters: SR parameter updates.
            num_walkers: MCMC walkers per iteration.
            mcmc_decorr_steps: MCMC steps between SR updates.
            num_equil_steps: Initial equilibration.
            mc_timestep: Initial MC timestep.
            fname_log: Log file path (None = stdout).
            verbose: Verbosity level.

        Returns:
            Dict with ``'params'``, ``'params_flat'``,
            ``'E_per_elec_history'``, ``'Var_history'``,
            ``'E_final_ha'``.
        """
        if isinstance(rng_key, int):
            rng_key = jax.random.key(rng_key)
        if (fname_log is None
                or (isinstance(fname_log, str) and fname_log == "")):
            fout = sys.stdout
        else:
            fout = open(fname_log, 'w', 1)

        params_flat = self.params_flat
        eloc_pmap = self._eloc_pmap

        # ---- pmap walker shape -----------------------------------
        # Reshape sharded data to (n_dev, n_w_local, ...) so pmap
        # distributes the leading axis across devices.  Replicated
        # data (params, scalars) keep their bare shape; pmap is told
        # via in_axes=None in each pmap'd function.
        n_dev = self.n_devices
        n_w_local = num_walkers // n_dev  # _build_step_fns validates

        rng_key, init_key = jax.random.split(rng_key)
        walkers = self.initialize_walkers(init_key, num_walkers)
        walkers = walkers.reshape(
            n_dev, n_w_local, self.nelec, self.dim,
        )
        # n_ph_walkers[dev, w] is a discrete int32 in [0, nph_max]. We
        # initialise to 0 (cavity vacuum). The n-MCMC suppresses moves
        # away from vacuum via |c_{n'}/c_n|² = exp(-2·slope·(n'-n)).
        n_ph_walkers = jnp.zeros(
            (n_dev, n_w_local), dtype=jnp.int32,
        )

        # Build the SR step pmap with statically-known num_walkers.
        sr_update_pmap = self._make_sr_step(num_walkers)

        # Composite (R, n) MCMC steps — shared between training and
        # evaluation. The R-step and n-step are independent and
        # individually detailed-balanced; we apply them sequentially
        # (Option B) per outer iteration.
        _mcmc_step, _n_mcmc_step = self._build_step_fns(num_walkers)

        # Step size is a scalar; pmap'd MCMC takes it via in_axes=None.
        step_size = jnp.asarray(
            (3 * mc_timestep) ** 0.5, dtype=jnp.float64,
        )

        # Equilibration.  Python loop over jitted steps; per-iter
        # sync via float() at the end blocks just enough to keep
        # async dispatch from over-staging.
        ar_jax = jnp.zeros((), dtype=jnp.float64)
        ar_n_jax = jnp.zeros((), dtype=jnp.float64)
        for i in range(num_equil_steps):
            rng_key, sub = jax.random.split(rng_key)
            keys_dev = jax.random.split(sub, n_dev)
            walkers, step_size, ar_jax = _mcmc_step(
                keys_dev, walkers, step_size, params_flat,
            )
            # QED Step 2: photon Fock n-MCMC after the R-step.
            rng_key, sub_n = jax.random.split(rng_key)
            keys_n_dev = jax.random.split(sub_n, n_dev)
            n_ph_walkers, ar_n_jax = _n_mcmc_step(
                keys_n_dev, n_ph_walkers, self.coh_alpha,
            )
            if i % 20 == 19:
                step_size.block_until_ready()
        ar = float(ar_jax)

        if verbose >= 1:
            extras = []
            if self.lr_schedule == 'inverse':
                extras.append(f"lr_decay_T={self.lr_decay_T:g}")
            elif self.lr_schedule == 'cosine':
                extras.append(
                    f"cosine[lr_min={self.lr_min:g}, "
                    f"T_max={self.lr_T_max}, "
                    f"n_restarts={self.lr_n_restarts}]"
                )
            if self.spring_mu > 0.0:
                extras.append(f"spring_mu={self.spring_mu}")
            if self.spring_norm_clip > 0.0:
                extras.append(
                    f"spring_norm_clip={self.spring_norm_clip:g}"
                )
            if self.damping_adapt:
                extras.append(
                    f"damping_adapt=[{self.damping_min:g},"
                    f"{self.damping_max:g}]"
                )
            if self.sampler != 'metropolis':
                extras.append(f"sampler={self.sampler}")
            extras_str = (' ' + ' '.join(extras)) if extras else ''
            print(
                f"# SR-VMC training — {self.n_params} params, "
                f"lr={self.lr}, damping={self.damping}, "
                f"n_cg={self.n_cg}, β={self.var_weight:.3g}"
                f"{extras_str}",
                file=fout,
            )
            print(
                f"# Equilibration acceptance: {ar:.3f}, "
                f"step size: {float(step_size):.4f} Bohr",
                file=fout,
            )
            obj_tag = (
                f"L/N (β={self.var_weight:.3g})"
                if self.var_weight > 0
                else ""
            )
            header = (
                f"# iter       <E>/N            Var(E)          "
                f"|f|             dt"
            )
            print(header, file=fout)

        e_history = []
        var_history = []
        timestamp_prev = datetime.now()

        # ---- Graceful shutdown on SIGTERM/SIGINT ----
        # When the user (or the OS) sends SIGTERM (e.g. ``kill <pid>``)
        # or SIGINT (Ctrl-C), set a flag that the training loop polls
        # at the end of each iter.  When set, we save the latest
        # checkpoint and break cleanly.  Important for long runs that
        # the user wants to terminate early without losing progress.
        import signal
        _stop_requested = {'flag': False, 'signo': None}

        def _stop_handler(signo, frame):
            _stop_requested['flag'] = True
            _stop_requested['signo'] = signo
            print(
                f"\n[signal] received {signal.Signals(signo).name}; "
                f"will save checkpoint and exit at end of iter.",
            )

        prev_sigterm = signal.signal(signal.SIGTERM, _stop_handler)
        prev_sigint = signal.signal(signal.SIGINT, _stop_handler)

        # SPRING momentum buffer (zero-init).  Replicated across devices
        # by virtue of being a single-host array passed through pmap
        # with in_axes=None.
        prev_dtheta = jnp.zeros_like(params_flat)
        spring_mu = self.spring_mu
        c_clip = self.spring_norm_clip   # 0.0 → disabled inside core
        damping_now = float(self.damping)
        damping_adapt = self.damping_adapt
        damping_factor = self.damping_factor
        damping_min = self.damping_min
        damping_max = self.damping_max
        damping_lookback = self.damping_lookback
        clip_scale = 1.0

        # Step 5: α-SR for the Option C multi-K phase + α_coh amplitude.
        # Two independent natural-gradient steps per iter:
        #
        # (A) phase α_k update — Im(E_loc) gradient, dense n_K × n_K Fisher
        #     Jacobian O_{α_k} = i · n_walker · F_k(R)
        #     g_α_k = 2·⟨(n_walker·F_k − ⟨...⟩) · Im(ΔE_loc)⟩
        #     S_{α_k α_k'} = Cov(n_walker·F_k, n_walker·F_k')
        #
        # (B) α_coh amplitude update — Re(E_loc) gradient, scalar Fisher
        #     Jacobian O_{coh} = n_walker/α_coh − α_coh  (real, no phase)
        #     g_coh = 2·⟨(O_coh − ⟨O_coh⟩) · Re(ΔE_loc)⟩
        #     S_coh = Var(O_coh)
        #
        # The two blocks (phase-Im vs amplitude-Re) decouple from WF SR
        # because:  Re-block and Im-block are orthogonal (different
        # Re/Im components of E_loc).  Within-block cross-correlations
        # are zero in expectation (electronic/photonic factorisation).
        K_vectors_for_alpha = self._K_vectors
        eps_dot_K_vec_for_alpha = self._eps_dot_K_vec
        n_K_static = int(self._n_K)

        @jax.jit
        def _phase_alpha_update(
            walkers, n_ph, e_para_im, phase_alpha,
            lr_now, damping, step_clip,
        ):
            """One phase-α SR step (Option C, multi-K, dense Fisher)."""
            # F_k(R) per walker, per K vector.
            K_dot_r = jnp.einsum(
                "nwid,kd->nwki", walkers, K_vectors_for_alpha,
            )                                                # (n_dev,n_w,n_K,n_e)
            F_per_k = jnp.sum(
                jnp.sin(K_dot_r), axis=-1,
            )                                                # (n_dev,n_w,n_K)

            # Jacobian factor: O_{α_k}(R,n) = n · F_k(R)  (imaginary i·n·F_k
            # but the i and the conjugate pair so the gradient formula sees
            # the real product directly).
            n_w_factor = n_ph.astype(jnp.float64)            # (n_dev,n_w)
            O_per_k = (
                n_w_factor[..., None] * F_per_k
            )                                                # (n_dev,n_w,n_K)

            # Centre Im(E_loc) and the Jacobian factor.
            de_im = e_para_im - jnp.mean(e_para_im)          # (n_dev,n_w)
            O_mean = jnp.mean(O_per_k, axis=(0, 1))          # (n_K,)
            dO = O_per_k - O_mean                            # (n_dev,n_w,n_K)

            # Force g_α_k = 2·⟨dO_k · ΔIm(E_loc)⟩
            g_alpha = 2.0 * jnp.mean(
                dO * de_im[..., None], axis=(0, 1),
            )                                                # (n_K,)
            # Dense Fisher S_{k,k'} = ⟨dO_k · dO_k'⟩
            dO_flat = dO.reshape(-1, n_K_static)             # (n_dev*n_w, n_K)
            S = (dO_flat.T @ dO_flat) / dO_flat.shape[0]     # (n_K, n_K)
            S_reg = S + damping * jnp.eye(n_K_static)

            # Natural-gradient step (small dense solve).
            delta_raw = lr_now * jnp.linalg.solve(S_reg, g_alpha)
            delta = jnp.clip(delta_raw, -step_clip, +step_clip)
            new_alpha = phase_alpha - delta
            return new_alpha, g_alpha, S

        @jax.jit
        def _coh_alpha_update(
            n_ph, e_loc_re, coh_alpha,
            lr_now, damping, step_clip, coh_floor,
        ):
            """One α_coh scalar-SR step (Re(E_loc) gradient)."""
            # O_coh = n/α_coh − α_coh
            n_f = n_ph.astype(jnp.float64)                   # (n_dev,n_w)
            O_w = n_f / coh_alpha - coh_alpha
            dO_w = O_w - jnp.mean(O_w)
            de_re = e_loc_re - jnp.mean(e_loc_re)
            g_coh = 2.0 * jnp.mean(dO_w * de_re)
            S_coh = jnp.mean(dO_w ** 2)
            delta_raw = lr_now * g_coh / (S_coh + damping)
            delta = jnp.clip(delta_raw, -step_clip, +step_clip)
            new_coh = coh_alpha - delta
            new_coh = jnp.maximum(new_coh, coh_floor)
            return new_coh, g_coh, S_coh

        for it in range(1, num_iters + 1):
            # Decorrelate walkers (pmap'd, distributes across devices).
            for _ in range(mcmc_decorr_steps):
                rng_key, sub = jax.random.split(rng_key)
                keys_dev = jax.random.split(sub, n_dev)
                walkers, step_size, _ = _mcmc_step(
                    keys_dev, walkers, step_size, params_flat,
                )
                # QED Step 2: photon-Fock n-step alongside R-step.
                rng_key, sub_n = jax.random.split(rng_key)
                keys_n_dev = jax.random.split(sub_n, n_dev)
                n_ph_walkers, _ = _n_mcmc_step(
                    keys_n_dev, n_ph_walkers, self.coh_alpha,
                )

            # Local energies + breakdown.  We use the breakdown pmap so
            # we can split off Im(E_loc) for the phase-α SR and
            # Re(E_loc) for the α_coh SR.
            (
                e_kin_dev, e_pot_dev, e_phot_dev, e_diamag_dev,
                e_para_re_dev, e_para_im_dev,
            ) = self._eloc_breakdown_pmap(
                walkers, params_flat, n_ph_walkers,
                self.phase_alpha, self.coh_alpha,
            )
            e_loc = (
                e_kin_dev + e_pot_dev + e_phot_dev
                + e_diamag_dev + e_para_re_dev
            )
            # e_loc shape (n_dev, n_w_local) — leave in pmap shape;
            # the SR step's pmap wants this shape via in_axes=0.

            # SPRING coupling: λμ.  When μ=0 the RHS is just g (vanilla SR).
            lambda_mu = damping_now * spring_mu

            # SR update — pmap'd; outputs are per-device but
            # all-reduced internally so every device returns the same
            # values.  Take [0] to get the host-side scalar/array.
            dtheta_pdev, e_mean_pdev, var_pdev, g_norm_pdev, clip_scale_pdev = (
                sr_update_pmap(
                    walkers, params_flat, e_loc,
                    prev_dtheta,
                    jnp.asarray(damping_now, dtype=jnp.float64),
                    jnp.asarray(lambda_mu, dtype=jnp.float64),
                    jnp.asarray(c_clip, dtype=jnp.float64),
                )
            )
            # Outputs from pmap have leading device axis; replicated
            # values are identical across [0..n_dev], so we just take
            # device 0.
            dtheta = dtheta_pdev[0]
            e_mean = e_mean_pdev[0]
            var = var_pdev[0]
            g_norm = g_norm_pdev[0]
            clip_scale = float(clip_scale_pdev[0])
            if spring_mu > 0.0:
                prev_dtheta = dtheta

            # Learning-rate scheduler (fixed / inverse / cosine).
            lr_now = self._compute_lr(it)

            params_flat = params_flat - lr_now * dtheta

            # ---- Step 5: phase-α + α_coh SR updates ----
            # Block-diagonal Fisher → independent steps on Im(E_loc)
            # (phase α_k) and Re(E_loc) (α_coh).  Both clipped to
            # prevent runaway under finite-sample noise.
            self.phase_alpha, phase_g_now, phase_S_now = (
                _phase_alpha_update(
                    walkers, n_ph_walkers, e_para_im_dev,
                    self.phase_alpha,
                    jnp.asarray(lr_now, dtype=jnp.float64),
                    jnp.asarray(damping_now, dtype=jnp.float64),
                    jnp.asarray(
                        self.alpha_step_clip, dtype=jnp.float64,
                    ),
                )
            )
            # α_coh-SR uses Re(E_loc).  e_loc was already built as the
            # sum kin+pot+phot+diamag+para_re above (Re only).
            self.coh_alpha, coh_g_now, coh_S_now = _coh_alpha_update(
                n_ph_walkers, e_loc, self.coh_alpha,
                jnp.asarray(lr_now, dtype=jnp.float64),
                jnp.asarray(damping_now, dtype=jnp.float64),
                jnp.asarray(
                    self.coh_alpha_step_clip, dtype=jnp.float64,
                ),
                jnp.asarray(self.coh_alpha_floor, dtype=jnp.float64),
            )

            # Levenberg-Marquardt-style adaptive damping.  Compare
            # the most recent ``damping_lookback`` energies' mean to
            # the prior window of the same width: if energy increased,
            # SR is over-confident — increase damping.  If it
            # decreased substantially, decrease damping.  Window-based
            # rather than per-step so noise doesn't whipsaw λ.
            if damping_adapt and it > 2 * damping_lookback:
                e_recent = float(np.mean(
                    e_history[-damping_lookback:]
                ))
                e_prev = float(np.mean(
                    e_history[
                        -2 * damping_lookback:-damping_lookback
                    ]
                ))
                # Compare on a per-electron basis (e_history already is).
                e_diff = e_recent - e_prev
                noise_floor = max(1e-6, 0.5 * np.sqrt(
                    float(np.mean(var_history[-damping_lookback:]))
                    / num_walkers,
                ) / self.nelec)
                if e_diff > noise_floor:
                    damping_now = min(
                        damping_max, damping_now * damping_factor,
                    )
                elif e_diff < -noise_floor:
                    damping_now = max(
                        damping_min, damping_now / damping_factor,
                    )

            # Per-iter logging.
            now = datetime.now()
            dt = (now - timestamp_prev).total_seconds()
            timestamp_prev = now
            e_per = float(e_mean) / self.nelec
            e_history.append(e_per)
            var_history.append(float(var))

            if verbose >= 1 and (it <= 10 or it % 10 == 0):
                print(
                    f"{it:>6d}  {e_per:>14.8e}  "
                    f"{float(var):>12.4e}  "
                    f"{float(g_norm):>12.4e}  "
                    f"{dt:>8.3f}",
                    file=fout,
                )

            # Step 5: phase-α + α_coh diagnostics every 100 iters.
            # At λ=0, both phase α should remain at 0 (no Im force)
            # and α_coh should drift to minimise diamag.
            if verbose >= 1 and (it == 1 or it % 100 == 0):
                phase_str = ", ".join(
                    f"a{k}={float(self.phase_alpha[k]):+.3e}"
                    for k in range(n_K_static)
                )
                phase_g_max = float(jnp.max(jnp.abs(phase_g_now)))
                print(
                    f"  [α] iter {it:>5d}  "
                    f"coh_α={float(self.coh_alpha):+.3e} "
                    f"(g_coh={float(coh_g_now):+.3e})  "
                    f"phase=[{phase_str}]  "
                    f"|g_phase|_max={phase_g_max:.3e}",
                    file=fout,
                )

            # BF diagnostics (every 100 iters when coord BF is active).
            # Tracks displacement magnitude and min quasiparticle pair
            # separation — direct evidence for coord-collapse pathology.
            # min_pair_sep_min → 0 = electrons colliding in coord space →
            # singular Slater determinant → biased E_L estimator.
            bf_diag_fn = getattr(self.log_psi_pytree, 'bf_diagnostics', None)
            if (verbose >= 1 and bf_diag_fn is not None
                    and (it == 1 or it % 100 == 0)):
                stats = bf_diag_fn(walkers, self.unravel(params_flat))
                print(
                    f"  [bf] iter {it:>5d}  "
                    f"disp(mean/walker)={float(stats['mean_disp_avg']):.3f}  "
                    f"disp(max/walker)={float(stats['max_disp_avg']):.3f}  "
                    f"disp_max_overall={float(stats['max_disp_max']):.3f}  "
                    f"min_qp_sep(avg/walker)={float(stats['min_pair_sep_avg']):.4f}  "
                    f"min_qp_sep(global)={float(stats['min_pair_sep_min']):.4f}",
                    file=fout,
                )

            # Periodic checkpoint save (in-loop) — unique filename per
            # save (no overwrite) so the full training history is
            # preserved.  E.g. ``<project>.chk_000500.h5``.
            if (self.save_every > 0
                    and self.ofname_chkpt
                    and it % self.save_every == 0):
                if self.ofname_chkpt.endswith('.chk.h5'):
                    iter_path = self.ofname_chkpt.replace(
                        '.chk.h5', f'.chk_{it:06d}.h5',
                    )
                else:
                    iter_path = f'{self.ofname_chkpt}_{it:06d}'
                ok = self._save_checkpoint(
                    params_flat, epoch=it, energy=e_per,
                    note=f"periodic@iter{it}",
                    path=iter_path,
                )
                if ok and verbose >= 1:
                    print(
                        f"  [chkpt] saved at iter {it}, "
                        f"E/N={e_per:+.6f} Ha → {iter_path}",
                        file=fout,
                    )

            # Honour SIGTERM / SIGINT — save and break loop.
            if _stop_requested['flag']:
                if self.ofname_chkpt:
                    ok = self._save_checkpoint(
                        params_flat, epoch=it, energy=e_per,
                        note=f"signal@iter{it}",
                    )
                    if ok and verbose >= 1:
                        print(
                            f"  [chkpt] saved on signal at iter {it}, "
                            f"E/N={e_per:+.6f} Ha → {self.ofname_chkpt}",
                            file=fout,
                        )
                break

        # Restore previous signal handlers so a subsequent eval
        # phase doesn't silently inherit our flags.
        signal.signal(signal.SIGTERM, prev_sigterm)
        signal.signal(signal.SIGINT, prev_sigint)

        if not (fname_log is None
                or (isinstance(fname_log, str) and fname_log == "")):
            fout.close()

        self.params_flat = params_flat
        params_pytree = self.unravel(params_flat)

        # End-of-training checkpoint save (also dumped periodically
        # during training, so this is a final-state snapshot).
        self._save_checkpoint(
            params_flat,
            epoch=len(e_history),
            energy=(e_history[-1] if e_history else None),
            note='final',
        )

        return {
            'params': params_pytree,
            'params_flat': params_flat,
            'E_per_elec_history': e_history,
            'Var_history': var_history,
            'E_final_ha': e_history[-1] if e_history else None,
        }


def get_qed_vmcopt_nn_heg_sr_func(
    config,
    init_key,
    *,
    prefix: str = "heg_sr",
    lr: float = 0.05,
    damping: float = 1e-3,
    n_cg: int = 30,
    var_weight: float = 0.0,
    ewald_n_real: int = 3,
    ewald_n_recip: int = 6,
    ewald_eta: Optional[float] = None,
    lr_schedule: str = 'auto',
    lr_decay_T: Optional[float] = None,
    lr_min: float = 0.0,
    lr_T_max: Optional[int] = None,
    lr_n_restarts: int = 0,
    spring_mu: float = 0.0,
    spring_norm_clip: Optional[float] = None,
    damping_adapt: bool = False,
    damping_min: float = 1.0e-5,
    damping_max: float = 1.0e-1,
    damping_factor: float = 2.0,
    damping_lookback: int = 50,
    sampler: str = 'metropolis',
    mala_grad_clip: Optional[float] = 1.0,
    save_every: int = 0,
    # QED extension
    omega: float = 0.0,
    nph_max: int = 0,
    fock_log_amp_slope: float = 3.0,
    coupling_lambda: float = 0.0,
    coupling_polarization=None,
    # Step 5: Option C multi-K phase + coherent-state amplitude.
    phase_K_vectors=None,
    phase_alpha_init=None,
    alpha_step_clip: float = 0.005,
    coh_alpha_init: float = 0.05,
    coh_alpha_step_clip: float = 0.005,
    coh_alpha_floor: float = 1.0e-3,
):
    """Construct an SR-VMC optimiser for a HEG ansatz.

    Args: see :class:`_QEDHEGSROptimizer`.

    Returns:
        :class:`_QEDHEGSROptimizer` — callable.
    """
    if prefix.endswith(".chk.h5"):
        prefix = prefix[: -len(".chk.h5")]
    ofname_chkpt = prefix + ".chk.h5" if prefix else None
    return _QEDHEGSROptimizer(
        config, init_key,
        lr=lr,
        damping=damping,
        n_cg=n_cg,
        var_weight=var_weight,
        ewald_n_real=ewald_n_real,
        ewald_n_recip=ewald_n_recip,
        ewald_eta=ewald_eta,
        ofname_chkpt=ofname_chkpt,
        lr_schedule=lr_schedule,
        lr_decay_T=lr_decay_T,
        lr_min=lr_min,
        lr_T_max=lr_T_max,
        lr_n_restarts=lr_n_restarts,
        spring_mu=spring_mu,
        spring_norm_clip=spring_norm_clip,
        damping_adapt=damping_adapt,
        damping_min=damping_min,
        damping_max=damping_max,
        damping_factor=damping_factor,
        damping_lookback=damping_lookback,
        sampler=sampler,
        mala_grad_clip=mala_grad_clip,
        save_every=save_every,
        omega=omega,
        nph_max=nph_max,
        fock_log_amp_slope=fock_log_amp_slope,
        coupling_lambda=coupling_lambda,
        coupling_polarization=coupling_polarization,
        phase_K_vectors=phase_K_vectors,
        phase_alpha_init=phase_alpha_init,
        alpha_step_clip=alpha_step_clip,
        coh_alpha_init=coh_alpha_init,
        coh_alpha_step_clip=coh_alpha_step_clip,
        coh_alpha_floor=coh_alpha_floor,
    )
