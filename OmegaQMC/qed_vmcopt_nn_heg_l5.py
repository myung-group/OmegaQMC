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

from .psi.nn.heg_wf import (
    HEGConfig,
    make_heg_log_psi_any as make_heg_log_psi,
)


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

        # Electronic gradients (n_e*dim,)
        grad_u_r = jax.grad(u_r_flat)(r_flat)
        grad_v_r = jax.grad(v_r_flat)(r_flat)
        # Electronic Laplacians via Hessian trace
        # NOTE: hessian is O((n_e·dim)²) memory; fine for our small N
        # in HEG.  For production scaling, use Hutchinson trick instead.
        hess_u_r = jax.hessian(u_r_flat)(r_flat)
        hess_v_r = jax.hessian(v_r_flat)(r_flat)
        lap_u_r = jnp.trace(hess_u_r)
        lap_v_r = jnp.trace(hess_v_r)

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
