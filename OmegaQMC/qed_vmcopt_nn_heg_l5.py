"""Level 5 cavity-QED HEG: position-rep photon, dense-Fourier + MLP
non-factorised complex Ψ.

Velocity-gauge Pauli-Fierz Hamiltonian, single q=0 cavity mode, photon
treated as a continuous coordinate q_c:

    H_PF = T_e + V_ee + ½·π_c² + ½·Ω_eff²·q_c² − λ·q_c·(ε·P̂_total)

with Ω_eff² = Ω² + N·λ² and P̂_total = Σᵢ p̂ᵢ.

Trial:  log Ψ(R, q_c) = u(R, q_c) + i·v(R, q_c).  Architecture
(Phase 5a-2 rewrite — NO handpicked K-basis, NO linear-in-q_c assumption):

    u(R, q_c) = log_psi_e_FermiNet(R)
                − ½·s·q_c² + ½·log(s/π)          # photon HO Gaussian
                + MLP_mag(features)               # magnitude coupling

    v(R, q_c) = MLP_phase(features)               # complex phase

    features = [Σᵢ sin(K·rᵢ),  Σᵢ cos(K·rᵢ)  for K in K_grid,  q_c]

K_grid is the **dense** reciprocal-lattice grid at all K = (2π/L)·(nx,ny)
with nx²+ny² ≤ K_max² (excluding origin).  K_max is a single
hyperparameter analogous to a plane-wave cutoff — NOT a handpicked
choice of which K vectors matter.

Both MLPs have **zero-initialised final layers** so that at init:
    MLP_mag = 0,  MLP_phase = 0  →  trial = bare HEG · HO photon Gaussian
    (real, factorised).  Regression to bare HEG at λ=0 is exact at init.

Inductive biases baked in (physical requirements, not arbitrary choices):
  - PBC: all sin/cos features at reciprocal-lattice K are periodic.
  - Permutation symmetry over electrons: features are Σᵢ pooled.
"""
from __future__ import annotations

from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np
from jax.flatten_util import ravel_pytree

import sys
from datetime import datetime

from .psi.nn.heg_wf import (
    HEGConfig,
    make_heg_log_psi_any as make_heg_log_psi,
)
from .psi.nn.periodic import (
    wrap_to_cell,
    make_cubic_lattice,
    make_square_lattice,
)
from .observables.ewald_dispatch import (
    build_ewald_tables_dim,
    ewald_pair_energy_dim,
)
from .observables.ewald_2d import ewald_2d_pair_energy
from .psi.nn.physics import laplacian
from .qed_vmcopt_nn_heg_sr import (
    _cg_solve,
    _adapt_step_size,
    TARGET_ACCEPTANCE_RATE,
)
from .utils import do_binning_analysis


# ---------------------------------------------------------------------
# Helpers: K-grid and MLP
# ---------------------------------------------------------------------

def build_K_grid_2d(L: float, K_max: int):
    """Dense 2D reciprocal-lattice grid up to ‖K‖ ≤ K_max·(2π/L).

    Returns (n_K, 2) array.  Excludes the origin K=(0,0)
    (sin gives 0, cos gives N — a constant; we don't need it as a feature).
    """
    coords = []
    for nx in range(-K_max, K_max + 1):
        for ny in range(-K_max, K_max + 1):
            if nx == 0 and ny == 0:
                continue
            if nx * nx + ny * ny > K_max * K_max:
                continue
            coords.append((nx, ny))
    if not coords:
        raise ValueError(
            f"K_max={K_max} gives no K vectors (need K_max ≥ 1)"
        )
    arr = np.asarray(coords, dtype=np.float64)
    return jnp.asarray(2.0 * np.pi * arr / L, dtype=jnp.float64)


def init_mlp_params(rng_key, layer_sizes, zero_init_last: bool = True):
    """Initialise dense-MLP params as a list of {'W', 'b'} dicts.

    layer_sizes: (in_dim, h1, ..., h_n, out_dim) — full chain.
    Xavier init for non-final layers; zero init for the final layer
    if `zero_init_last`.
    """
    layers = []
    keys = jax.random.split(rng_key, max(1, len(layer_sizes) - 1))
    for i, (n_in, n_out) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
        is_last = (i == len(layer_sizes) - 2)
        if is_last and zero_init_last:
            W = jnp.zeros((n_in, n_out), dtype=jnp.float64)
        else:
            stddev = jnp.sqrt(2.0 / (n_in + n_out))
            W = stddev * jax.random.normal(
                keys[i], (n_in, n_out), dtype=jnp.float64,
            )
        b = jnp.zeros((n_out,), dtype=jnp.float64)
        layers.append({"W": W, "b": b})
    return layers


def mlp_apply(params, x, activation=jnp.tanh):
    """Apply stacked dense MLP; final layer is linear (no activation)."""
    for i, layer in enumerate(params):
        x = x @ layer["W"] + layer["b"]
        if i < len(params) - 1:
            x = activation(x)
    return x


# ---------------------------------------------------------------------
# Main factory
# ---------------------------------------------------------------------

def build_l5_log_psi(
    config,
    init_key,
    *,
    omega_init: float,
    K_max: int = 5,
    phase_mlp_hidden=(64, 64),
    mag_mlp_hidden=(64, 64),
    activation: str = "tanh",
):
    """Construct the Level 5 complex log Ψ + initial flat params.

    Args:
      config: HEGConfig or HEGPsiFormerConfig (electronic ansatz spec).
      init_key: JAX PRNG key for all parameter initialisation.
      omega_init: initial photon HO width s.  Set to cavity Ω so the
        trial starts at the HO ground state.
      K_max: Fourier-feature cutoff (in units of 2π/L).  Default 5
        gives ~78 K vectors in 2D → 156 sin/cos features.
      phase_mlp_hidden: tuple of hidden-layer widths for the phase MLP.
        Final scalar output appended automatically.
      mag_mlp_hidden: hidden widths for the magnitude-coupling MLP.
      activation: 'tanh' or 'silu' (default 'tanh').

    Returns dict with:
      log_psi_l5         — callable (R, q_c, p_pytree) → (log_mag, phase)
      log_psi_l5_flat    — callable (r_flat, q_c, p_flat) → (log_mag, phase)
      init_params_flat   — initial flat parameter vector
      init_params_pytree — pytree form of the same
      unravel            — flat → pytree
      n_params           — number of flat parameters
      n_electronic       — # of electronic FermiNet parameters
      n_mag_mlp          — # of magnitude-MLP parameters
      n_phase_mlp        — # of phase-MLP parameters
      n_K                — # of K vectors in dense Fourier basis
      n_features         — input dim of MLPs (= 2·n_K + 1)
      K_grid             — (n_K, dim) array of K vectors
      electronic_log_psi — the bare FermiNet log_psi (for reuse)
    """
    nelec = int(config.n_up) + int(config.n_down)
    dim = int(getattr(config, "dim", 3))
    L = float(config.L)

    if dim != 2:
        raise ValueError(
            f"build_l5_log_psi currently supports dim=2 only "
            f"(got dim={dim})"
        )

    K_grid = build_K_grid_2d(L, K_max)
    n_K = int(K_grid.shape[0])
    n_features = 2 * n_K + 1   # sin(K·r), cos(K·r), q_c

    if activation == "tanh":
        act_fn = jnp.tanh
    elif activation == "silu":
        act_fn = jax.nn.silu
    elif activation == "gelu":
        act_fn = jax.nn.gelu
    else:
        raise ValueError(f"Unknown activation: {activation}")

    # Param-init keys: split into electronic + mag MLP + phase MLP
    key_e, key_mag, key_phase = jax.random.split(init_key, 3)

    # Electronic FermiNet (existing infrastructure)
    log_psi_e_pytree, e_init_params, graphdef = make_heg_log_psi(
        config, key_e,
    )

    # MLPs.  Final output dim is 1 (scalar).
    mag_layer_sizes = (n_features,) + tuple(mag_mlp_hidden) + (1,)
    mag_init = init_mlp_params(
        key_mag, mag_layer_sizes, zero_init_last=True,
    )
    phase_layer_sizes = (n_features,) + tuple(phase_mlp_hidden) + (1,)
    phase_init = init_mlp_params(
        key_phase, phase_layer_sizes, zero_init_last=True,
    )

    init_params_pytree = {
        "e": e_init_params,
        "s": jnp.asarray(omega_init, dtype=jnp.float64),
        "mag_mlp": mag_init,
        "phase_mlp": phase_init,
    }
    init_params_flat, unravel = ravel_pytree(init_params_pytree)
    n_params = int(init_params_flat.shape[0])

    K_grid_const = K_grid
    log_inv_pi = -0.5 * jnp.log(jnp.pi)

    def compute_features(R, q_c):
        """[Σᵢ sin(K·rᵢ), Σᵢ cos(K·rᵢ), q_c] → (2·n_K + 1,)."""
        K_dot_r = jnp.einsum(
            "id,kd->ki", R, K_grid_const,
        )                                                  # (n_K, n_e)
        sin_feat = jnp.sum(jnp.sin(K_dot_r), axis=-1)      # (n_K,)
        cos_feat = jnp.sum(jnp.cos(K_dot_r), axis=-1)      # (n_K,)
        return jnp.concatenate(
            [sin_feat, cos_feat, jnp.atleast_1d(q_c)]
        )

    def log_psi_l5(R, q_c, p_pytree):
        """log Ψ(R, q_c) = u + i·v, returned as two real scalars."""
        s = p_pytree["s"]
        log_psi_e_val = log_psi_e_pytree(R, p_pytree["e"])
        log_chi = (
            -0.5 * s * q_c * q_c
            + 0.5 * jnp.log(jnp.abs(s))
            + log_inv_pi
        )
        feats = compute_features(R, q_c)
        # Both MLPs output (1,); take scalar via [0].
        mag_coupling = mlp_apply(
            p_pytree["mag_mlp"], feats, activation=act_fn,
        )[0]
        phase = mlp_apply(
            p_pytree["phase_mlp"], feats, activation=act_fn,
        )[0]
        log_mag = log_psi_e_val + log_chi + mag_coupling
        return log_mag, phase

    def log_psi_l5_flat(r_flat, q_c, p_flat):
        R = r_flat.reshape(nelec, dim)
        p_pytree = unravel(p_flat)
        return log_psi_l5(R, q_c, p_pytree)

    # Diagnostics on parameter splits
    e_flat, _ = ravel_pytree(e_init_params)
    n_electronic = int(e_flat.shape[0])
    mag_flat, _ = ravel_pytree(mag_init)
    n_mag_mlp = int(mag_flat.shape[0])
    phase_flat, _ = ravel_pytree(phase_init)
    n_phase_mlp = int(phase_flat.shape[0])

    return {
        "log_psi_l5": log_psi_l5,
        "log_psi_l5_flat": log_psi_l5_flat,
        "init_params_flat": init_params_flat,
        "init_params_pytree": init_params_pytree,
        "unravel": unravel,
        "n_params": n_params,
        "n_electronic": n_electronic,
        "n_mag_mlp": n_mag_mlp,
        "n_phase_mlp": n_phase_mlp,
        "n_K": n_K,
        "n_features": n_features,
        "K_grid": K_grid_const,
        "electronic_log_psi": log_psi_e_pytree,
        "electronic_graphdef": graphdef,
        "nelec": nelec,
        "dim": dim,
    }


# ---------------------------------------------------------------------
# Phase 5a-3: full Pauli-Fierz local energy
# ---------------------------------------------------------------------
#
# Returns (Re, Im) of E_loc per walker, NOT including V_ee (Ewald).
# Caller adds V_ee separately.
#
# Re(E_loc) = -½ Σᵢ [∇ᵢ²u + (∇ᵢu)² - (∇ᵢv)²]                ← T_e (Re part)
#             -½ [∂²u/∂q_c² + (∂u/∂q_c)² - (∂v/∂q_c)²]      ← T_phot (Re part)
#             + ½·Ω_eff²·q_c²                                ← V_phot
#             - λ·q_c·(ε·Σᵢ ∇ᵢv)                             ← bilinear (Re)
#
# Im(E_loc) = -½ Σᵢ [∇ᵢ²v + 2·∇ᵢu·∇ᵢv]                      ← T_e (Im part)
#             -½ [∂²v/∂q_c² + 2·(∂u/∂q_c)·(∂v/∂q_c)]        ← T_phot (Im part)
#             + λ·q_c·(ε·Σᵢ ∇ᵢu)                             ← bilinear (Im)
#
# Im(E_loc) is the Hermiticity diagnostic — must average to 0.

# ---------------------------------------------------------------------
# Phase 5a-4: SR primitives for complex log Ψ with real parameters
# ---------------------------------------------------------------------
#
# For real-valued θ and complex log Ψ = u + i·v:
#   O_θ = ∂(log Ψ)/∂θ = ∂u/∂θ + i·∂v/∂θ
#   ΔE_loc = (Re(E_loc) − ⟨Re⟩) + i·(Im(E_loc) − ⟨Im⟩)
#
# Standard complex-Ψ SR formulas (Sorella; NetKet; DeepErwin):
#   g_θ      = 2·Re[ ⟨ΔO_θ* · ΔE_loc⟩ ]
#            = 2·⟨ ∂u/∂θ · ΔRe + ∂v/∂θ · ΔIm ⟩
#   S_{θθ'}  = Re[ ⟨ΔO_θ* · ΔO_θ'⟩ ]
#            = ⟨ ∂u/∂θ · ∂u/∂θ' + ∂v/∂θ · ∂v/∂θ' ⟩    (after centering)
#
# Both real (PSD).  CG solves (S + ε·I)·δθ = g.
#
# Matvec used in CG (avoid forming dense S):
#   S·v = ⟨ ∂u/∂θ · (∂u/∂θ' · v) + ∂v/∂θ · (∂v/∂θ' · v) ⟩

def make_l5_sr_primitives(log_psi_l5_flat, nelec: int, dim: int):
    """Build per-walker Jacobian + batched SR primitives for Level 5.

    log_psi_l5_flat: callable (r_flat, q_c, p_flat) → (log_mag, phase)
    """
    def log_mag_flat(r_flat, q_c, p_flat):
        return log_psi_l5_flat(r_flat, q_c, p_flat)[0]

    def phase_flat(r_flat, q_c, p_flat):
        return log_psi_l5_flat(r_flat, q_c, p_flat)[1]

    # Per-walker Jacobians w.r.t. parameters (real).
    grad_u_param = jax.grad(log_mag_flat, argnums=2)
    grad_v_param = jax.grad(phase_flat, argnums=2)

    def per_walker_jacobian(r_flat, q_c, p_flat):
        """Returns (du_dθ, dv_dθ) for one walker — both (n_params,) real.
        """
        du = grad_u_param(r_flat, q_c, p_flat)
        dv = grad_v_param(r_flat, q_c, p_flat)
        return du, dv

    # Batched: maps over walkers (R, q_c are per-walker; p_flat shared).
    batched_jacobian = jax.jit(
        jax.vmap(
            per_walker_jacobian,
            in_axes=(0, 0, None),
        )
    )

    def sr_force_and_S_matvec(
        Jac_u, Jac_v, e_re, e_im,
    ):
        """Compute force vector and matvec closure for S.

        Inputs (n_walkers shape):
          Jac_u:  (n_walkers, n_params)  ∂u/∂θ per walker
          Jac_v:  (n_walkers, n_params)  ∂v/∂θ per walker
          e_re:   (n_walkers,)            Re(E_loc) per walker
          e_im:   (n_walkers,)            Im(E_loc) per walker

        Returns:
          g:       (n_params,)            SR force vector
          S_matvec: callable v → S·v
          e_mean:  scalar                 ⟨Re(E_loc)⟩
          e_var:   scalar                 Var(Re(E_loc))
          im_mean: scalar                 ⟨Im(E_loc)⟩  (Hermiticity check)
        """
        n_walkers = Jac_u.shape[0]
        e_re_mean = jnp.mean(e_re)
        e_im_mean = jnp.mean(e_im)
        e_re_var = jnp.var(e_re)
        de_re = e_re - e_re_mean
        de_im = e_im - e_im_mean
        # Center Jacobians (subtract per-param mean across walkers)
        u_mean = jnp.mean(Jac_u, axis=0)
        v_mean = jnp.mean(Jac_v, axis=0)
        dJu = Jac_u - u_mean
        dJv = Jac_v - v_mean

        # g_θ = 2·⟨dJu · de_re + dJv · de_im⟩
        g = 2.0 * (
            (dJu.T @ de_re) / n_walkers
            + (dJv.T @ de_im) / n_walkers
        )

        def S_matvec(v):
            proj_u = dJu @ v                            # (n_walkers,)
            proj_v = dJv @ v                            # (n_walkers,)
            return (
                (dJu.T @ proj_u) / n_walkers
                + (dJv.T @ proj_v) / n_walkers
            )

        return g, S_matvec, e_re_mean, e_re_var, e_im_mean

    return {
        "per_walker_jacobian": per_walker_jacobian,
        "batched_jacobian": batched_jacobian,
        "sr_force_and_S_matvec": sr_force_and_S_matvec,
    }


def make_l5_sr_smw_solver():
    """SMW dual-form SR solver for Level 5 complex log Ψ.

    Solves (S + λI) δθ = rhs in walker space — exact, no CG.

    For complex log Ψ = u + iv with real parameters θ,
        S = (1/n_w) · (dJu.T @ dJu + dJv.T @ dJv)
          = (1/n_w) · do_stacked.T @ do_stacked
    where do_stacked = concat([dJu, dJv], axis=0) ∈ (2n_w, n_p).
    The Sherman-Morrison-Woodbury identity gives
        (S + λI)^{-1} rhs = (1/λ) (rhs - do_s.T @ u_smw)
        u_smw = (I_2n + K)^{-1} (do_s @ rhs / (λ n_w))
        K = do_s @ do_s.T / (λ n_w)              ∈ (2n_w, 2n_w)
    Single dense solve on a 2n_w-by-2n_w matrix; for n_w=512 that's
    1024×1024, runs in O(2n_w²·n_p) + O((2n_w)³) ≈ tens of ms on GH200.

    Returns a jit-compiled callable
        solver(Jac_u, Jac_v, e_re, e_im, prev_delta, damping, mu, c_clip)
        → (delta, e_mean, e_var, im_mean, g_norm, scale)
    matching the SPRING + F-norm clip semantics of the CG path.
    """
    @jax.jit
    def solver(Jac_u, Jac_v, e_re, e_im,
               prev_delta, damping, mu, c_clip):
        n_w = Jac_u.shape[0]
        # Center Jacobians and E_loc
        e_re_mean = jnp.mean(e_re)
        e_im_mean = jnp.mean(e_im)
        e_re_var = jnp.var(e_re)
        de_re = e_re - e_re_mean
        de_im = e_im - e_im_mean
        dJu = Jac_u - jnp.mean(Jac_u, axis=0)
        dJv = Jac_v - jnp.mean(Jac_v, axis=0)

        # Force g_θ = 2·⟨dJu·de_re + dJv·de_im⟩
        g = 2.0 * (
            (dJu.T @ de_re) / n_w
            + (dJv.T @ de_im) / n_w
        )

        # SPRING-modified RHS: rhs = g + (μ·λ)·prev_delta
        rhs = g + (mu * damping) * prev_delta

        # SMW solve in walker space.  Note: divisor in K and o_rhs is
        # (λ·n_w) where n_w is the ELECTRONIC walker count, not 2n_w,
        # because Fisher is normalised by n_w (one sum each over the
        # n_w Re-rows and n_w Im-rows, both divided by n_w).
        do_s = jnp.concatenate([dJu, dJv], axis=0)   # (2n_w, n_p)
        inv_λn = 1.0 / (damping * n_w)
        K = (do_s @ do_s.T) * inv_λn                  # (2n_w, 2n_w)
        I_plus_K = K + jnp.eye(K.shape[0], dtype=K.dtype)
        o_rhs = (do_s @ rhs) * inv_λn                 # (2n_w,)
        u_smw = jnp.linalg.solve(I_plus_K, o_rhs)
        delta = (rhs - do_s.T @ u_smw) / damping      # (n_p,)

        # F-norm clip: ‖δ‖_S² = δ·(S δ) = (1/n_w)·‖do_s δ‖²
        proj = do_s @ delta                            # (2n_w,)
        f_norm_sq = jnp.maximum(jnp.sum(proj * proj) / n_w, 1e-20)
        f_norm = jnp.sqrt(f_norm_sq)
        raw_scale = c_clip / (f_norm + 1e-20)
        scale = jnp.where(
            c_clip > 0.0,
            jnp.minimum(1.0, raw_scale),
            jnp.ones_like(raw_scale),
        )
        delta = scale * delta

        g_norm = jnp.sqrt(jnp.sum(g * g))
        return delta, e_re_mean, e_re_var, e_im_mean, g_norm, scale

    return solver


def _laplacian_vmap(f, x):
    """O(N) Laplacian via vmap-of-JVP of grad.

    For a scalar f(x) with x ∈ R^n, returns
        ∇²f(x) = Σᵢ ∂²f/∂xᵢ² = Σᵢ (H eᵢ)ᵢ
        ∇ f(x) = grad f(x)
    using ``vmap(jax.jvp(grad_f, x, eᵢ))[i]`` over the standard basis.

    Equivalent to ``laplacian_linearize`` (in OmegaQMC/utils.py) but
    uses ``vmap`` + ``jax.jvp`` instead of ``linearize`` + ``fori_loop``.
    The latter has a latent bug that surfaces under outer jit at
    certain batch sizes (vmap walker counts ~256): some walkers'
    Laplacian becomes NaN even though the gradient is finite.  This
    vmap form is numerically robust under nested jit.

    Cost analysis (per scalar f-call):
      - 1 grad evaluation for ``df``
      - n vmap'd jvp evaluations for diag(H) — each is one extra
        forward+backward pass of grad_f at a unit-direction tangent.
      - Total flops ~ 2n × (forward pass of grad_f) = O(n²)
      - But all n JVPs share the same linearization point, so XLA can
        fuse most of it.  Wall time is closer to O(n) than O(n²).
    """
    grad_f = jax.grad(f)
    df = grad_f(x)
    n = x.shape[0]
    eye = jnp.eye(n, dtype=x.dtype)

    def hvp_diag_i(e_i, i):
        # primal output discarded; we want tangent at index i
        _, hvp_e = jax.jvp(grad_f, (x,), (e_i,))
        return hvp_e[i]

    diag = jax.vmap(hvp_diag_i, in_axes=(0, 0))(eye, jnp.arange(n))
    return jnp.sum(diag), df


def make_l5_eloc_no_vee(
    log_psi_l5,
    *,
    eps,
    lam: float,
    omega_eff: float,
    nelec: int,
    dim: int,
):
    """Build (Re, Im) local-energy function for Level 5 trial.

    Returns callable ``eloc(R, q_c, p_pytree) → (re, im)`` where
    `re` excludes V_ee — add Ewald separately.

    Args:
      log_psi_l5: callable (R, q_c, p_pytree) → (log_mag, phase)
      eps: (dim,) polarisation unit vector
      lam: coupling λ
      omega_eff: renormalised photon frequency √(Ω² + N·λ²)
      nelec, dim: system dims
    """
    eps_arr = jnp.asarray(eps, dtype=jnp.float64)

    def u_fn(R, q_c, p):
        return log_psi_l5(R, q_c, p)[0]

    def v_fn(R, q_c, p):
        return log_psi_l5(R, q_c, p)[1]

    def eloc(R, q_c, p_pytree):
        r_flat = R.reshape(-1)

        # Wrap u, v as functions of flat r and scalar q (for jax.grad)
        def u_r_flat(rf):
            return u_fn(rf.reshape(nelec, dim), q_c, p_pytree)

        def v_r_flat(rf):
            return v_fn(rf.reshape(nelec, dim), q_c, p_pytree)

        def u_q(q):
            return u_fn(R, q, p_pytree)

        def v_q(q):
            return v_fn(R, q, p_pytree)

        # Electronic Laplacians + gradients via O(N) vmap-of-JVP.
        # ``laplacian_linearize`` (linearize + fori_loop) is faster
        # standalone but produces NaN for some walkers under outer
        # jit at certain batch sizes — see _laplacian_vmap below.
        lap_u_r, grad_u_r = _laplacian_vmap(u_r_flat, r_flat)
        lap_v_r, grad_v_r = _laplacian_vmap(v_r_flat, r_flat)

        # Squared electronic gradients (sum over all coordinates)
        grad_u_r_sq = jnp.dot(grad_u_r, grad_u_r)
        grad_v_r_sq = jnp.dot(grad_v_r, grad_v_r)
        grad_u_dot_v = jnp.dot(grad_u_r, grad_v_r)

        # ε·Σᵢ ∇ᵢ u and v (per-electron gradient projected on ε, summed)
        grad_u_per_elec = grad_u_r.reshape(nelec, dim)
        grad_v_per_elec = grad_v_r.reshape(nelec, dim)
        eps_dot_grad_u = jnp.einsum("id,d->", grad_u_per_elec, eps_arr)
        eps_dot_grad_v = jnp.einsum("id,d->", grad_v_per_elec, eps_arr)

        # Photon (q_c) derivatives via 1D grad
        du_dq = jax.grad(u_q)(q_c)
        dv_dq = jax.grad(v_q)(q_c)
        d2u_dq2 = jax.grad(jax.grad(u_q))(q_c)
        d2v_dq2 = jax.grad(jax.grad(v_q))(q_c)

        # Assemble Re(E_loc) and Im(E_loc) per Phase 0 formula
        re = (
            -0.5 * (lap_u_r + grad_u_r_sq - grad_v_r_sq)
            - 0.5 * (d2u_dq2 + du_dq ** 2 - dv_dq ** 2)
            + 0.5 * omega_eff ** 2 * q_c ** 2
            - lam * q_c * eps_dot_grad_v
        )
        im = (
            -0.5 * (lap_v_r + 2.0 * grad_u_dot_v)
            - 0.5 * (d2v_dq2 + 2.0 * du_dq * dv_dq)
            + lam * q_c * eps_dot_grad_u
        )
        return re, im

    return eloc


# ---------------------------------------------------------------------
# Phase 5a-4 (full): Level 5 SR optimizer
# ---------------------------------------------------------------------
#
# Wraps log_psi + walker + composite MCMC + E_loc + SR into a callable.
# Single-device for now (no pmap); composite Metropolis MCMC on
# (R, q_c).  Standard SR with cosine learning-rate schedule.

class _QEDL5Optimizer:
    """Level 5 SR-VMC optimizer for cavity-QED HEG.

    Position-rep photon, non-factorised complex trial wavefunction,
    dense-Fourier + MLP architecture.
    """

    def __init__(
        self,
        config,
        init_key,
        *,
        lr: float = 0.5,
        damping: float = 1e-3,
        n_cg: int = 30,
        ewald_n_real: int = 3,
        ewald_n_recip: int = 6,
        ewald_eta=None,
        ofname_chkpt=None,
        lr_schedule: str = "cosine",
        lr_min: float = 1e-5,
        lr_T_max=None,
        # SPRING (Goldshlager 2024) momentum + F-norm clip.
        # μ=0 → vanilla SR.  Step 5 proven recipe: μ=0.9, clip=0.1, lr=1.0.
        spring_mu: float = 0.0,
        spring_norm_clip: float = 0.0,
        # Plan-B SMW dual-form SR (exact walker-space solve).  When
        # True (default), bypasses CG and uses the closed-form SMW
        # update — exact, ~8× faster than CG for n_w ≪ n_p.  Set to
        # False to fall back to the legacy primal-CG path.
        use_smw_sr: bool = True,
        # Plan-C fused @jax.jit train_step (MCMC + eloc + Jac + SMW +
        # update inline).  Eliminates ~34 Python↔device boundaries per
        # iter.  Requires use_smw_sr=True.  (NaN bug at walker=256
        # traced to laplacian_linearize misbehaving under outer jit;
        # fixed by switching eloc to full-Hessian Laplacian.)
        use_fused_step: bool = True,
        # Diagnostic: freeze mag_mlp + phase_mlp params at their init
        # values by zeroing the corresponding entries of the SR update
        # vector.  At λ=0 the MLPs have no physical driving signal but
        # still get random-walked by SR noise — adding variance burden
        # without improving variational energy.  Setting this True
        # tests whether the L5 plateau at λ=0 is caused by that noise:
        # if frozen → reaches bare_HEG + Ω/2.
        freeze_mlps: bool = False,
        # Cavity parameters
        omega: float = 0.1,
        coupling_lambda: float = 0.0,
        coupling_polarization=None,
        # Level-5 architecture knobs
        K_max: int = 5,
        phase_mlp_hidden=(64, 64),
        mag_mlp_hidden=(64, 64),
        activation: str = "tanh",
    ):
        self.config = config
        self.L = float(config.L)
        self.n_up = int(config.n_up)
        self.n_down = int(config.n_down)
        self.nelec = self.n_up + self.n_down
        self.dim = int(getattr(config, "dim", 3))
        if self.dim != 2:
            raise ValueError(
                f"_QEDL5Optimizer currently supports dim=2 only "
                f"(got dim={self.dim})"
            )
        self.lr = float(lr)
        self.damping = float(damping)
        self.n_cg = int(n_cg)
        self.ofname_chkpt = ofname_chkpt

        # Cavity parameters
        self.omega = float(omega)
        self.coupling_lambda = float(coupling_lambda)
        self.omega_eff = float(
            jnp.sqrt(
                self.omega ** 2 + self.nelec * self.coupling_lambda ** 2
            )
        )
        if coupling_polarization is None:
            eps_list = [1.0] + [0.0] * (self.dim - 1)
        else:
            eps_list = list(coupling_polarization)
        eps_arr = jnp.asarray(eps_list, dtype=jnp.float64)
        eps_arr = eps_arr / jnp.linalg.norm(eps_arr)
        self.eps = eps_arr

        # Build Level 5 trial machinery
        self.l5 = build_l5_log_psi(
            config, init_key,
            omega_init=self.omega,        # init HO width = bare Ω
            K_max=K_max,
            phase_mlp_hidden=phase_mlp_hidden,
            mag_mlp_hidden=mag_mlp_hidden,
            activation=activation,
        )
        self.params_flat = self.l5["init_params_flat"]
        self.unravel = self.l5["unravel"]
        self.n_params = self.l5["n_params"]

        # SR primitives
        self.sr = make_l5_sr_primitives(
            self.l5["log_psi_l5_flat"],
            nelec=self.nelec, dim=self.dim,
        )

        # E_loc (no Ewald)
        self.eloc_fn = make_l5_eloc_no_vee(
            self.l5["log_psi_l5"],
            eps=self.eps,
            lam=self.coupling_lambda,
            omega_eff=self.omega_eff,
            nelec=self.nelec, dim=self.dim,
        )

        # Lattice + Ewald
        if self.dim == 2:
            self.lattice = make_square_lattice(self.L)
        else:
            self.lattice = make_cubic_lattice(self.L)
        self.ewald = build_ewald_tables_dim(
            self.L, dim=self.dim, eta=ewald_eta,
            n_real=ewald_n_real, n_recip=ewald_n_recip,
        )

        # Pre-jitted Ewald per-walker (vmap)
        if self.dim == 2:
            _ewald_one = lambda r: ewald_2d_pair_energy(r[None], self.ewald)[0]
        else:
            from .observables.ewald import ewald_pair_energy
            _ewald_one = lambda r: ewald_pair_energy(r[None], self.ewald)[0]
        self._ewald_per_walker = jax.jit(jax.vmap(_ewald_one))

        # LR schedule
        self.lr_schedule = lr_schedule
        self.lr_min = float(lr_min)
        self.lr_T_max = lr_T_max
        # SPRING
        self.spring_mu = float(spring_mu)
        self.spring_norm_clip = float(spring_norm_clip)
        # SMW dual-form SR
        self.use_smw_sr = bool(use_smw_sr)
        if self.use_smw_sr:
            self.smw_solver = make_l5_sr_smw_solver()
        else:
            self.smw_solver = None
        # Fused jit train_step (built lazily — needs num_walkers and
        # mcmc_decorr_steps which come at call time).
        self.use_fused_step = bool(use_fused_step)
        if self.use_fused_step and not self.use_smw_sr:
            raise ValueError(
                "use_fused_step=True requires use_smw_sr=True "
                "(fused path only supports SMW solver)."
            )
        self._fused_train_step_cache = None
        # MLP-freeze mask (zeros on mag_mlp + phase_mlp slices, ones
        # elsewhere).  Used to mask the SR update before applying.
        self.freeze_mlps = bool(freeze_mlps)
        if self.freeze_mlps:
            mask_pytree = jax.tree.map(
                lambda x: jnp.ones_like(x),
                self.l5["init_params_pytree"],
            )
            mask_pytree["mag_mlp"] = jax.tree.map(
                lambda x: jnp.zeros_like(x), mask_pytree["mag_mlp"],
            )
            mask_pytree["phase_mlp"] = jax.tree.map(
                lambda x: jnp.zeros_like(x), mask_pytree["phase_mlp"],
            )
            self._freeze_mask_flat, _ = jax.flatten_util.ravel_pytree(
                mask_pytree
            )
        else:
            self._freeze_mask_flat = None

    def _compute_lr(self, it):
        if self.lr_schedule == "cosine":
            if self.lr_T_max is None:
                T = 500
            else:
                T = int(self.lr_T_max)
            if it >= T:
                return self.lr_min
            cos = 0.5 * (1.0 + jnp.cos(jnp.pi * it / T))
            return self.lr_min + (self.lr - self.lr_min) * float(cos)
        return self.lr  # fixed

    def initialize_walkers(self, rng_key, num_walkers):
        """Joint (R, q_c) walker init.

        R uniform in [0, L]^(N×dim);
        q_c sampled from N(0, 1/√(2Ω)) — HO ground state.
        """
        k_R, k_q = jax.random.split(rng_key)
        R = self.L * jax.random.uniform(
            k_R, (num_walkers, self.nelec, self.dim),
            dtype=jnp.float64,
        )
        sigma_q = 1.0 / jnp.sqrt(2.0 * self.omega)
        q_c = sigma_q * jax.random.normal(
            k_q, (num_walkers,), dtype=jnp.float64,
        )
        return R, q_c

    def _build_mcmc_steps(self, num_walkers):
        """Returns jitted (R_step, qc_step) MCMC for given walker batch size."""
        log_psi_l5_flat = self.l5["log_psi_l5_flat"]
        nelec = self.nelec
        dim = self.dim
        lattice = self.lattice

        def log_mag_one(r_flat, q_c, p_flat):
            return log_psi_l5_flat(r_flat, q_c, p_flat)[0]

        def R_move_one(key, R, q_c, step, p_flat):
            kp, ka = jax.random.split(key)
            R_prop = R + step * jax.random.normal(
                kp, R.shape, dtype=jnp.float64,
            )
            R_prop = wrap_to_cell(R_prop, lattice)
            r_old = R.reshape(-1)
            r_new = R_prop.reshape(-1)
            lp_old = log_mag_one(r_old, q_c, p_flat)
            lp_new = log_mag_one(r_new, q_c, p_flat)
            accept = jax.random.uniform(ka) < jnp.exp(
                2.0 * (lp_new - lp_old)
            )
            R_new = jnp.where(accept, R_prop, R)
            return R_new, accept

        def qc_move_one(key, R, q_c, step, p_flat):
            kp, ka = jax.random.split(key)
            q_prop = q_c + step * jax.random.normal(
                kp, (), dtype=jnp.float64,
            )
            r_flat = R.reshape(-1)
            lp_old = log_mag_one(r_flat, q_c, p_flat)
            lp_new = log_mag_one(r_flat, q_prop, p_flat)
            accept = jax.random.uniform(ka) < jnp.exp(
                2.0 * (lp_new - lp_old)
            )
            q_new = jnp.where(accept, q_prop, q_c)
            return q_new, accept

        R_move_batch = jax.vmap(
            R_move_one, in_axes=(0, 0, 0, None, None),
        )
        qc_move_batch = jax.vmap(
            qc_move_one, in_axes=(0, 0, 0, None, None),
        )

        @jax.jit
        def R_step(rng_key, R, q_c, step_R, p_flat):
            keys = jax.random.split(rng_key, num_walkers)
            R_new, acc = R_move_batch(keys, R, q_c, step_R, p_flat)
            ar = jnp.mean(acc).astype(jnp.float64)
            new_step = _adapt_step_size(step_R, ar)
            return R_new, new_step, ar

        @jax.jit
        def qc_step(rng_key, R, q_c, step_q, p_flat):
            keys = jax.random.split(rng_key, num_walkers)
            q_new, acc = qc_move_batch(keys, R, q_c, step_q, p_flat)
            ar = jnp.mean(acc).astype(jnp.float64)
            new_step = _adapt_step_size(step_q, ar)
            return q_new, new_step, ar

        return R_step, qc_step

    def _batched_eloc(self, R, q_c, p_pytree):
        """Per-walker (re, im) including V_ee."""
        eloc_v = jax.vmap(
            lambda Ri, qi: self.eloc_fn(Ri, qi, p_pytree),
            in_axes=(0, 0),
        )
        re_no_vee, im = eloc_v(R, q_c)
        V_ee = self._ewald_per_walker(R)
        return re_no_vee + V_ee, im

    def _build_fused_train_step(self, num_walkers, mcmc_decorr_steps):
        """Plan C: one jitted train_step that runs the entire iter.

        Composition:
          1. lax.scan over ``mcmc_decorr_steps`` of (R_move, qc_move)
          2. batched E_loc (Re + Im) + Ewald V_ee
          3. batched Jacobian (∂u, ∂v) per walker
          4. SMW dual-form SR solve (inline)
          5. parameter update

        carry := (rng_key, R, q_c, step_R, step_q, params_flat, prev_delta)
        scanned input := lr (a traced scalar — recomputed each iter
                           from cosine schedule on Python side)
        Eliminates ~34 host↔device boundaries per iter.

        Returns: ``train_step(carry, lr) → (new_carry, metrics)`` where
        ``metrics`` is a dict with e_mean, e_var, im_mean, g_norm,
        scale, ar_R, ar_q, step_R, step_q (all scalars).
        """
        # --- Closures ---
        log_psi_l5_flat = self.l5["log_psi_l5_flat"]
        lattice = self.lattice
        unravel = self.unravel
        eloc_fn = self.eloc_fn
        ewald = self.ewald
        damping_c = jnp.float64(self.damping)
        mu_c = jnp.float64(self.spring_mu)
        c_clip_c = jnp.float64(self.spring_norm_clip)
        freeze_mask_flat = self._freeze_mask_flat   # None or (n_params,)

        def log_mag_one(r_flat, q_c, p_flat):
            return log_psi_l5_flat(r_flat, q_c, p_flat)[0]

        # Per-walker MCMC moves
        def R_move_one(key, R, q_c, step, p_flat):
            kp, ka = jax.random.split(key)
            R_prop = R + step * jax.random.normal(
                kp, R.shape, dtype=jnp.float64,
            )
            R_prop = wrap_to_cell(R_prop, lattice)
            lp_old = log_mag_one(R.reshape(-1), q_c, p_flat)
            lp_new = log_mag_one(R_prop.reshape(-1), q_c, p_flat)
            accept = jax.random.uniform(ka) < jnp.exp(
                2.0 * (lp_new - lp_old)
            )
            return jnp.where(accept, R_prop, R), accept

        def qc_move_one(key, R, q_c, step, p_flat):
            kp, ka = jax.random.split(key)
            q_prop = q_c + step * jax.random.normal(
                kp, (), dtype=jnp.float64,
            )
            r_flat = R.reshape(-1)
            lp_old = log_mag_one(r_flat, q_c, p_flat)
            lp_new = log_mag_one(r_flat, q_prop, p_flat)
            accept = jax.random.uniform(ka) < jnp.exp(
                2.0 * (lp_new - lp_old)
            )
            return jnp.where(accept, q_prop, q_c), accept

        R_move_batch = jax.vmap(R_move_one, in_axes=(0, 0, 0, None, None))
        qc_move_batch = jax.vmap(qc_move_one, in_axes=(0, 0, 0, None, None))

        # Per-walker SR Jacobian
        per_walker_jacobian = self.sr["per_walker_jacobian"]
        batched_jac = jax.vmap(
            per_walker_jacobian, in_axes=(0, 0, None),
        )

        # Per-walker E_loc + Ewald
        def eloc_one(R_i, q_c_i, p_pytree):
            return eloc_fn(R_i, q_c_i, p_pytree)
        eloc_batched = jax.vmap(eloc_one, in_axes=(0, 0, None))

        def ewald_batched(R):
            _one = lambda r: ewald_2d_pair_energy(r[None], ewald)[0]
            return jax.vmap(_one)(R)

        # ---- MCMC inner scan body ----
        def mcmc_one_step(carry, _):
            rng, R, q_c, step_R, step_q, p_flat = carry
            # R move
            rng, sub = jax.random.split(rng)
            keys = jax.random.split(sub, num_walkers)
            R, acc_R = R_move_batch(keys, R, q_c, step_R, p_flat)
            ar_R = jnp.mean(acc_R).astype(jnp.float64)
            step_R = _adapt_step_size(step_R, ar_R)
            # qc move
            rng, sub = jax.random.split(rng)
            keys = jax.random.split(sub, num_walkers)
            q_c, acc_q = qc_move_batch(keys, R, q_c, step_q, p_flat)
            ar_q = jnp.mean(acc_q).astype(jnp.float64)
            step_q = _adapt_step_size(step_q, ar_q)
            return (rng, R, q_c, step_R, step_q, p_flat), (ar_R, ar_q)

        # ---- Full fused step ----
        # Unrolled MCMC (Python loop inside the jit, 15× XLA code but
        # all one program) — avoids lax.scan numerical issues observed
        # at production scale (N=10, walker=512).
        @jax.jit
        def fused_train_step(carry, lr):
            (rng, R, q_c, step_R, step_q,
             params_flat, prev_delta) = carry

            # 1) MCMC decorrelation (unrolled inside jit)
            ar_R_sum = jnp.float64(0.0)
            ar_q_sum = jnp.float64(0.0)
            mcmc_carry = (rng, R, q_c, step_R, step_q, params_flat)
            for _ in range(mcmc_decorr_steps):
                mcmc_carry, (ar_R_i, ar_q_i) = mcmc_one_step(
                    mcmc_carry, None,
                )
                ar_R_sum = ar_R_sum + ar_R_i
                ar_q_sum = ar_q_sum + ar_q_i
            rng, R, q_c, step_R, step_q, _ = mcmc_carry
            ar_R = ar_R_sum / mcmc_decorr_steps
            ar_q = ar_q_sum / mcmc_decorr_steps

            # 2) E_loc + Ewald
            p_pytree = unravel(params_flat)
            re_no_vee, e_im = eloc_batched(R, q_c, p_pytree)
            V_ee = ewald_batched(R)
            e_re = re_no_vee + V_ee

            # 3) Jacobian per walker
            r_flat = R.reshape(num_walkers, -1)
            Jac_u, Jac_v = batched_jac(r_flat, q_c, params_flat)

            # 4) Inline SMW solve (matches make_l5_sr_smw_solver)
            n_w = num_walkers
            e_re_mean = jnp.mean(e_re)
            e_im_mean = jnp.mean(e_im)
            e_re_var = jnp.var(e_re)
            de_re = e_re - e_re_mean
            de_im = e_im - e_im_mean
            dJu = Jac_u - jnp.mean(Jac_u, axis=0)
            dJv = Jac_v - jnp.mean(Jac_v, axis=0)
            g = 2.0 * (
                (dJu.T @ de_re) / n_w + (dJv.T @ de_im) / n_w
            )
            rhs = g + (mu_c * damping_c) * prev_delta
            do_s = jnp.concatenate([dJu, dJv], axis=0)   # (2n_w, n_p)
            inv_λn = 1.0 / (damping_c * n_w)
            K = (do_s @ do_s.T) * inv_λn                  # (2n_w, 2n_w)
            I_plus_K = K + jnp.eye(K.shape[0], dtype=K.dtype)
            o_rhs = (do_s @ rhs) * inv_λn                 # (2n_w,)
            u_smw = jnp.linalg.solve(I_plus_K, o_rhs)
            delta = (rhs - do_s.T @ u_smw) / damping_c

            # F-norm clip
            proj = do_s @ delta
            f_norm_sq = jnp.maximum(jnp.sum(proj * proj) / n_w, 1e-20)
            f_norm = jnp.sqrt(f_norm_sq)
            raw_scale = c_clip_c / (f_norm + 1e-20)
            scale = jnp.where(
                c_clip_c > 0.0,
                jnp.minimum(1.0, raw_scale),
                jnp.ones_like(raw_scale),
            )
            delta = scale * delta
            g_norm = jnp.sqrt(jnp.sum(g * g))

            # 5) Update (optional: zero MLP-param slice of delta)
            if freeze_mask_flat is not None:
                delta = delta * freeze_mask_flat
            params_flat = params_flat - lr * delta
            prev_delta = delta

            new_carry = (rng, R, q_c, step_R, step_q,
                         params_flat, prev_delta)
            metrics = {
                "e_mean": e_re_mean,
                "e_var": e_re_var,
                "im_mean": e_im_mean,
                "g_norm": g_norm,
                "scale": scale,
                "ar_R": ar_R,
                "ar_q": ar_q,
                "step_R": step_R,
                "step_q": step_q,
            }
            return new_carry, metrics

        return fused_train_step

    def __call__(
        self,
        rng_key,
        num_iters: int = 500,
        num_walkers: int = 1024,
        mcmc_decorr_steps: int = 20,
        num_equil_steps: int = 400,
        mc_timestep_R: float = 0.1,
        mc_timestep_qc: float = 0.5,
        fname_log=None,
        verbose: int = 1,
    ):
        """SR-VMC optimisation run.

        Returns dict with params, E history, etc.
        """
        if isinstance(rng_key, int):
            rng_key = jax.random.key(rng_key)
        if fname_log is None or fname_log == "":
            fout = sys.stdout
        else:
            fout = open(fname_log, "w", 1)

        rng_key, init_key = jax.random.split(rng_key)
        R, q_c = self.initialize_walkers(init_key, num_walkers)

        R_step_size = jnp.asarray((3 * mc_timestep_R) ** 0.5, dtype=jnp.float64)
        qc_step_size = jnp.asarray(mc_timestep_qc, dtype=jnp.float64)

        R_step_fn, qc_step_fn = self._build_mcmc_steps(num_walkers)

        # Equilibration
        ar_R = jnp.asarray(0.0)
        ar_q = jnp.asarray(0.0)
        for i in range(num_equil_steps):
            rng_key, sub = jax.random.split(rng_key)
            R, R_step_size, ar_R = R_step_fn(
                sub, R, q_c, R_step_size, self.params_flat,
            )
            rng_key, sub = jax.random.split(rng_key)
            q_c, qc_step_size, ar_q = qc_step_fn(
                sub, R, q_c, qc_step_size, self.params_flat,
            )
            if i % 50 == 49:
                R_step_size.block_until_ready()

        if verbose >= 1:
            print(
                f"# L5 SR-VMC training — {self.n_params} params  "
                f"(elec={self.l5['n_electronic']}, "
                f"mag_mlp={self.l5['n_mag_mlp']}, "
                f"phase_mlp={self.l5['n_phase_mlp']}, "
                f"n_K={self.l5['n_K']})",
                file=fout,
            )
            print(
                f"# Equilibration: ar_R={float(ar_R):.3f}, "
                f"ar_q={float(ar_q):.3f}; "
                f"step_R={float(R_step_size):.3f}, "
                f"step_q={float(qc_step_size):.3f}",
                file=fout,
            )
            print(
                f"# Ω={self.omega:.4f}, λ={self.coupling_lambda:.4f}, "
                f"Ω_eff={self.omega_eff:.4f}",
                file=fout,
            )
            print(
                "# iter   <Re E>/N        Var(Re E)      <Im E>/N        |g|",
                file=fout,
            )

        e_history = []
        var_history = []
        im_history = []
        timestamp_prev = datetime.now()
        # SPRING momentum buffer
        prev_delta = jnp.zeros_like(self.params_flat)

        # Build the fused jit train_step once (closes over num_walkers,
        # mcmc_decorr_steps).  Cached for repeat __call__ s with the
        # same walker batch / decorr length.
        if self.use_fused_step:
            cache_key = (num_walkers, mcmc_decorr_steps)
            if (self._fused_train_step_cache is None
                or self._fused_train_step_cache[0] != cache_key):
                fused = self._build_fused_train_step(
                    num_walkers, mcmc_decorr_steps,
                )
                self._fused_train_step_cache = (cache_key, fused)
            fused_train_step = self._fused_train_step_cache[1]
            carry = (rng_key, R, q_c, R_step_size, qc_step_size,
                     self.params_flat, prev_delta)

        for it in range(1, num_iters + 1):
            if self.use_fused_step:
                # Plan C: one jit call per iter — MCMC + eloc + Jac +
                # SMW + update all fused, lax.scan inside for decorr.
                lr_now = self._compute_lr(it)
                carry, m = fused_train_step(carry, jnp.float64(lr_now))
                (rng_key, R, q_c, R_step_size, qc_step_size,
                 self.params_flat, prev_delta) = carry
                e_mean = m["e_mean"]
                e_var = m["e_var"]
                im_mean = m["im_mean"]
                g_norm = float(m["g_norm"])
            else:
                # Legacy unfused path (kept for debugging / fallback).
                for _ in range(mcmc_decorr_steps):
                    rng_key, sub = jax.random.split(rng_key)
                    R, R_step_size, _ = R_step_fn(
                        sub, R, q_c, R_step_size, self.params_flat,
                    )
                    rng_key, sub = jax.random.split(rng_key)
                    q_c, qc_step_size, _ = qc_step_fn(
                        sub, R, q_c, qc_step_size, self.params_flat,
                    )
                p_pytree = self.unravel(self.params_flat)
                e_re, e_im = self._batched_eloc(R, q_c, p_pytree)
                r_flat = R.reshape(num_walkers, -1)
                Jac_u, Jac_v = self.sr["batched_jacobian"](
                    r_flat, q_c, self.params_flat,
                )
                damping = self.damping
                mu = self.spring_mu
                c_clip = self.spring_norm_clip
                if self.use_smw_sr:
                    delta, e_mean, e_var, im_mean, g_norm_j, scale = \
                        self.smw_solver(
                            Jac_u, Jac_v, e_re, e_im,
                            prev_delta,
                            jnp.float64(damping),
                            jnp.float64(mu),
                            jnp.float64(c_clip),
                        )
                    g_norm = float(g_norm_j)
                else:
                    g, S_matvec, e_mean, e_var, im_mean = self.sr[
                        "sr_force_and_S_matvec"
                    ](Jac_u, Jac_v, e_re, e_im)
                    rhs = g + (mu * damping) * prev_delta
                    def mv(v):
                        return S_matvec(v) + damping * v
                    delta = _cg_solve(mv, rhs, n_iters=self.n_cg, tol=1e-6)
                    if c_clip > 0.0:
                        s_delta = S_matvec(delta)
                        f_norm = jnp.sqrt(
                            jnp.maximum(jnp.dot(delta, s_delta), 1e-20)
                        )
                        scale_cg = jnp.minimum(1.0, c_clip / (f_norm + 1e-20))
                        delta = scale_cg * delta
                    g_norm = float(jnp.linalg.norm(g))
                if self._freeze_mask_flat is not None:
                    delta = delta * self._freeze_mask_flat
                prev_delta = delta
                lr_now = self._compute_lr(it)
                self.params_flat = self.params_flat - lr_now * delta

            e_per = float(e_mean) / self.nelec
            im_per = float(im_mean) / self.nelec
            e_history.append(e_per)
            var_history.append(float(e_var))
            im_history.append(im_per)

            now = datetime.now()
            dt = (now - timestamp_prev).total_seconds()
            timestamp_prev = now
            if verbose >= 1 and (it <= 10 or it % 10 == 0):
                print(
                    f"{it:>5d}  {e_per:>+.6e}  {float(e_var):>.4e}  "
                    f"{im_per:>+.3e}  {g_norm:>.3e}  ({dt:.2f}s)",
                    file=fout,
                )

        if fname_log is not None and fname_log != "":
            fout.close()

        return {
            "params_flat": self.params_flat,
            "params_pytree": self.unravel(self.params_flat),
            "E_per_elec_history": e_history,
            "Var_history": var_history,
            "Im_per_elec_history": im_history,
            "E_final_ha": e_history[-1] if e_history else None,
        }

    def _build_fused_eval_step(self, num_walkers):
        """Jit-fused 1-step MCMC + E_loc for evaluation.

        carry  := (rng, R, q_c, step_R, step_q, params_flat)
        output := (e_re_mean, e_im_mean, q_c_sq_mean)  — scalars/walker

        Same composition as ``_build_fused_train_step`` but without
        Jacobian + SR (eval is non-gradient).  Eliminates the per-step
        Python↔device boundaries that make the unfused eval ~50× slower.
        """
        log_psi_l5_flat = self.l5["log_psi_l5_flat"]
        lattice = self.lattice
        unravel = self.unravel
        eloc_fn = self.eloc_fn
        ewald = self.ewald

        def log_mag_one(r_flat, q_c, p_flat):
            return log_psi_l5_flat(r_flat, q_c, p_flat)[0]

        def R_move_one(key, R, q_c, step, p_flat):
            kp, ka = jax.random.split(key)
            R_prop = R + step * jax.random.normal(
                kp, R.shape, dtype=jnp.float64,
            )
            R_prop = wrap_to_cell(R_prop, lattice)
            lp_old = log_mag_one(R.reshape(-1), q_c, p_flat)
            lp_new = log_mag_one(R_prop.reshape(-1), q_c, p_flat)
            accept = jax.random.uniform(ka) < jnp.exp(
                2.0 * (lp_new - lp_old)
            )
            return jnp.where(accept, R_prop, R), accept

        def qc_move_one(key, R, q_c, step, p_flat):
            kp, ka = jax.random.split(key)
            q_prop = q_c + step * jax.random.normal(
                kp, (), dtype=jnp.float64,
            )
            r_flat = R.reshape(-1)
            lp_old = log_mag_one(r_flat, q_c, p_flat)
            lp_new = log_mag_one(r_flat, q_prop, p_flat)
            accept = jax.random.uniform(ka) < jnp.exp(
                2.0 * (lp_new - lp_old)
            )
            return jnp.where(accept, q_prop, q_c), accept

        R_move_batch = jax.vmap(R_move_one, in_axes=(0, 0, 0, None, None))
        qc_move_batch = jax.vmap(qc_move_one, in_axes=(0, 0, 0, None, None))

        eloc_batched = jax.vmap(
            lambda Ri, qi, p: eloc_fn(Ri, qi, p), in_axes=(0, 0, None),
        )

        def ewald_batched(R):
            _one = lambda r: ewald_2d_pair_energy(r[None], ewald)[0]
            return jax.vmap(_one)(R)

        @jax.jit
        def eval_step(carry, _):
            rng, R, q_c, step_R, step_q, params_flat = carry
            # R move
            rng, sub = jax.random.split(rng)
            keys = jax.random.split(sub, num_walkers)
            R, acc_R = R_move_batch(keys, R, q_c, step_R, params_flat)
            ar_R = jnp.mean(acc_R).astype(jnp.float64)
            step_R = _adapt_step_size(step_R, ar_R)
            # qc move
            rng, sub = jax.random.split(rng)
            keys = jax.random.split(sub, num_walkers)
            q_c, acc_q = qc_move_batch(keys, R, q_c, step_q, params_flat)
            ar_q = jnp.mean(acc_q).astype(jnp.float64)
            step_q = _adapt_step_size(step_q, ar_q)
            # E_loc + Ewald + reductions
            p_pytree = unravel(params_flat)
            re_no_vee, e_im = eloc_batched(R, q_c, p_pytree)
            V_ee = ewald_batched(R)
            e_re = re_no_vee + V_ee
            e_re_mean = jnp.mean(e_re)
            e_im_mean = jnp.mean(e_im)
            qc_sq_mean = jnp.mean(q_c * q_c)
            new_carry = (rng, R, q_c, step_R, step_q, params_flat)
            return new_carry, (e_re_mean, e_im_mean, qc_sq_mean)

        return eval_step

    def evaluate(
        self,
        rng_key,
        params_flat=None,
        num_walkers: int = 512,
        num_blocks: int = 50,
        num_blocks_equil: int = 10,
        num_steps_per_block: int = 20,
        mc_timestep_R: float = 0.1,
        mc_timestep_qc: float = 0.5,
        fname_log=None,
        verbose: int = 1,
    ):
        """Non-gradient block-averaged evaluation.

        ``fname_log`` is opened in APPEND mode so the runner can pass
        the same file as training to keep both sections in one log.
        """
        if isinstance(rng_key, int):
            rng_key = jax.random.key(rng_key)
        if fname_log is None or fname_log == "":
            fout = sys.stdout
        else:
            fout = open(fname_log, "a", 1)

        if params_flat is None:
            params_flat = self.params_flat

        rng_key, init_key = jax.random.split(rng_key)
        R, q_c = self.initialize_walkers(init_key, num_walkers)
        R_step_size = jnp.asarray((3 * mc_timestep_R) ** 0.5, dtype=jnp.float64)
        qc_step_size = jnp.asarray(mc_timestep_qc, dtype=jnp.float64)

        # Jit-fused 1-step MCMC + eloc, used inside lax.scan for both
        # equilibration and per-block sampling.  Eliminates Python loop
        # overhead — same trick as the fused training step.
        eval_step = self._build_fused_eval_step(num_walkers)
        carry = (rng_key, R, q_c, R_step_size, qc_step_size, params_flat)

        # Equilibration (no metric collection)
        n_equil = num_blocks_equil * num_steps_per_block
        if n_equil > 0:
            carry, _ = jax.lax.scan(eval_step, carry, None, length=n_equil)

        E_blocks = []
        Im_blocks = []
        qc_sq_blocks = []
        timestamp_init = datetime.now()

        for blk in range(num_blocks):
            # Each block: scan over num_steps_per_block MCMC+eloc steps
            carry, (re_arr, im_arr, qc_sq_arr) = jax.lax.scan(
                eval_step, carry, None, length=num_steps_per_block,
            )
            E_blocks.append(float(jnp.mean(re_arr)))
            Im_blocks.append(float(jnp.mean(im_arr)))
            qc_sq_blocks.append(float(jnp.mean(qc_sq_arr)))

            if verbose >= 1 and (blk + 1) % 10 == 0:
                print(
                    f"#  block {blk + 1}/{num_blocks}  "
                    f"E/N = {E_blocks[-1] / self.nelec:+.6e}  "
                    f"Im/N = {Im_blocks[-1] / self.nelec:+.3e}  "
                    f"<q²>={qc_sq_blocks[-1]:.4f}",
                    file=fout,
                )

        E_arr = jnp.array(E_blocks)
        e_mean, e_serr, _, e_kappa = do_binning_analysis(E_arr)
        N = self.nelec
        e_per = float(e_mean) / N
        e_serr_per = float(e_serr) / N
        im_mean = float(jnp.mean(jnp.array(Im_blocks))) / N
        im_serr = float(jnp.std(jnp.array(Im_blocks)) / jnp.sqrt(num_blocks)) / N
        qc_sq_mean = float(jnp.mean(jnp.array(qc_sq_blocks)))

        elapsed = (datetime.now() - timestamp_init).total_seconds()
        if verbose >= 1:
            print(
                f"\nVMC L5 energy: {float(e_mean):.6e} ± "
                f"{float(e_serr):.6e} Ha  "
                f"({float(E_arr.shape[0] / e_kappa):.1f} eff blocks)",
                file=fout,
            )
            print(
                f"E_QED/N      = {e_per:+.8e} ± {e_serr_per:.2e} Ha",
                file=fout,
            )
            print(
                f"<Im E_loc>/N = {im_mean:+.4e} ± {im_serr:.2e} Ha  "
                f"(Hermiticity check)",
                file=fout,
            )
            print(
                f"<q_c²>       = {qc_sq_mean:.4e}  "
                f"(expected ≈ 1/(2·Ω_eff) = "
                f"{1.0 / (2.0 * self.omega_eff):.4e}"
                f"   [Ω_eff=√(Ω²+N·λ²)={self.omega_eff:.4f}])",
                file=fout,
            )
            print(f"Total eval time: {elapsed:.2f} s", file=fout)

        if fname_log is not None and fname_log != "":
            fout.close()

        return {
            "E_per_elec_ha": e_per,
            "E_serr_per_e_ha": e_serr_per,
            "Im_per_e_ha": im_mean,
            "Im_serr_per_e_ha": im_serr,
            "qc_sq_mean": qc_sq_mean,
            "E_blocks": E_blocks,
            "Im_blocks": Im_blocks,
            "qc_sq_blocks": qc_sq_blocks,
        }
