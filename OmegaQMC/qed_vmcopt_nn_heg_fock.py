"""Level 8 cavity-QED HEG: Fock-basis ψ(R, n) ansatz.

Tang-style architecture (arxiv 2503.15644v1): no physical priors on the
matter–photon factorisation.  The photon mode is in the discrete Fock
basis truncated at n ∈ {0, …, N_max}, and the trial is a vector-valued
amplitude

    ψ_n(R)  for n = 0, …, N_max

with no a-priori coherent-state / Gaussian / Lang-Firsov structure.  All
correlation between matter and photon is learned: each Fock sector has
its own arbitrary matter wavefunction.

Hamiltonian (velocity-gauge Pauli-Fierz, single q=0 cavity mode):

    H_PF = T_e + V_ee + v_ext(R)
         + Ω_eff·(b†b + ½)          ← dressed photon HO (diagonal in n)
         − λ·q_c·(ε̂·P̂_total)         ← bilinear coupling (off-diagonal n↔n±1)

where Ω_eff² = Ω² + N·λ² absorbs the diamagnetic A² term and the
dressed photon position operator is q_c = √(1/(2Ω_eff))·(b + b†).

Trial (per-n head on shared FermiNet trunk):

    ψ_n(R) = exp[ log_ψ_e(R) + mag_mlp_n(F) + offset_n + i·phase_mlp_n(F) ]

    F = [ Σᵢ sin(K·rᵢ),  Σᵢ cos(K·rᵢ),  CoM ]      # matter-only features
    offset = (0, -50, -50, …)                       # log-amplitude floor

At init (zero-init last layers), mag_mlp_n = phase_mlp_n = 0, so
    ψ_0(R) ≈ exp(log_ψ_e(R))    (matter HF)
    ψ_{n>0}(R) ≈ 0              (Fock vacuum)
and Ψ = ψ_HF(R)·|0⟩ regardless of N_max.

MCMC: R-only (no photon coordinate to sample).  Sampling density
    π(R) ∝ Σ_n |ψ_n(R)|²

Local energy (Fock sum, exact — no n MCMC):
    E_loc(R) = ( Σ_{n,n'} ψ_n*(R) · H_{n,n'}(R) · ψ_{n'}(R) ) / Σ_n |ψ_n(R)|²

SR primitives: natural log-derivative for vector Ψ at fixed R,
    ∂_θ log Ψ(R) ≡ ( Σ_n ψ_n*(R)·∂_θ ψ_n(R) ) / Σ_n |ψ_n(R)|²

See design/L8_fock_spec.md for the full design rationale.
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
# Generic SR numerical primitives — same math as L5; not L5-specific.
from .qed_vmcopt_nn_heg_l5 import make_l5_sr_smw_solver
from .qed_vmcopt_nn_heg_sr import _adapt_step_size, TARGET_ACCEPTANCE_RATE


# ---------------------------------------------------------------------
# Helpers (local copies — kept here so the Fock architecture has no
# import-time dependency on the L5/L7 module)
# ---------------------------------------------------------------------

def build_K_grid_2d(L: float, K_max: int):
    """Dense 2D reciprocal-lattice grid up to ‖K‖ ≤ K_max·(2π/L).

    Returns (n_K, 2) array, excluding the origin (sin gives 0, cos gives N
    — constant; not useful as a feature).
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

    layer_sizes: (in_dim, h1, ..., h_n, out_dim).
    Xavier-normal for non-final layers; final layer is zero-init when
    ``zero_init_last``.
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
# Trial wavefunction factory — step 1 of the L8 spec
# ---------------------------------------------------------------------

def build_fock_log_psi(
    config,
    init_key,
    *,
    N_max: int = 6,
    K_max: int = 5,
    mag_mlp_hidden=(64, 64),
    phase_mlp_hidden=(64, 64),
    activation: str = "tanh",
    offset_floor: float = -50.0,
    eps_arr=None,
):
    """Construct the Level 8 vector-valued log Ψ + initial flat params.

    Args:
      config:    HEGConfig or HEGPsiFormerConfig (electronic ansatz spec).
      init_key:  JAX PRNG key for all parameter initialisation.
      N_max:     Fock truncation.  Network has (N_max+1) per-n outputs.
      K_max:     Fourier-feature cutoff for the matter features
                 (= dense reciprocal-lattice cutoff in units of 2π/L).
      mag_mlp_hidden, phase_mlp_hidden: hidden widths.  Each MLP outputs
                 a vector of size (N_max+1) — one scalar per Fock sector.
      activation: 'tanh' | 'silu' | 'gelu'.
      offset_floor: log-amplitude offset added to mag_mlp output for
                 n > 0 sectors so the trial starts at the vacuum (ψ_0 ≈
                 ψ_HF, ψ_{n>0} ≈ 0).  Negative; default −50 ≈ 1e−22 ratio.
      eps_arr:   polarisation unit vector (only stored as metadata for
                 downstream local energy use).

    Returns dict with:
      psi_vec            — callable (R, p_pytree) → (N_max+1,) complex
      psi_vec_flat       — callable (r_flat, p_flat) → same
      log_psi_total      — callable (R, p_pytree) → real scalar
                           = log|Ψ(R)| = ½ log Σ_n |ψ_n(R)|²  (for MCMC)
      log_psi_total_flat — same on flat r and p
      init_params_flat   — initial flat parameter vector (real)
      init_params_pytree — pytree form of the same
      unravel            — flat → pytree
      n_params           — # of flat parameters
      n_electronic       — # of electronic FermiNet parameters
      n_mag_mlp          — # of magnitude-MLP parameters
      n_phase_mlp        — # of phase-MLP parameters
      n_K                — # of K vectors in the Fourier basis
      n_features         — input dim of the heads (= 2·n_K + dim)
      K_grid             — (n_K, dim) array of K vectors
      offset             — (N_max+1,) constant amplitude offsets
      N_max              — Fock truncation
      nelec, dim         — system dims (echoed for caller convenience)
    """
    nelec = int(config.n_up) + int(config.n_down)
    dim = int(getattr(config, "dim", 3))
    L = float(config.L)
    if dim != 2:
        raise ValueError(
            f"build_fock_log_psi currently supports dim=2 only "
            f"(got dim={dim})"
        )
    if N_max < 0:
        raise ValueError(f"N_max must be ≥ 0 (got {N_max})")

    if activation == "tanh":
        act_fn = jnp.tanh
    elif activation == "silu":
        act_fn = jax.nn.silu
    elif activation == "gelu":
        act_fn = jax.nn.gelu
    else:
        raise ValueError(f"Unknown activation: {activation}")

    K_grid = build_K_grid_2d(L, K_max)
    n_K = int(K_grid.shape[0])
    # Matter-only features.  No q_c — that variable does not exist in
    # the Fock representation.  CoM is included so the network can
    # express dipole-style couplings if they emerge.
    n_features = 2 * n_K + dim

    key_e, key_mag, key_phase = jax.random.split(init_key, 3)

    # Electronic FermiNet trunk (existing infrastructure).
    log_psi_e_pytree, e_init_params, graphdef = make_heg_log_psi(
        config, key_e,
    )

    # Per-n heads — single MLP each, but with output dim (N_max+1).
    n_out = N_max + 1
    mag_layer_sizes = (n_features,) + tuple(mag_mlp_hidden) + (n_out,)
    phase_layer_sizes = (n_features,) + tuple(phase_mlp_hidden) + (n_out,)
    mag_init = init_mlp_params(
        key_mag, mag_layer_sizes, zero_init_last=True,
    )
    phase_init = init_mlp_params(
        key_phase, phase_layer_sizes, zero_init_last=True,
    )

    # Non-trainable amplitude offset.  log|ψ_n| starts at offset[n], so
    # with offset = [0, -floor, -floor, …] only the n=0 sector has
    # non-negligible amplitude at init.  The optimizer can grow
    # mag_mlp_n past offset[n] where needed.
    offset = jnp.concatenate([
        jnp.zeros((1,), dtype=jnp.float64),
        jnp.full((N_max,), float(offset_floor), dtype=jnp.float64),
    ])  # shape (N_max+1,)

    init_params_pytree = {
        "e":         e_init_params,
        "mag_mlp":   mag_init,
        "phase_mlp": phase_init,
    }
    init_params_flat, unravel = ravel_pytree(init_params_pytree)
    n_params = int(init_params_flat.shape[0])

    # Polarization (metadata only — used by downstream local-energy code).
    if eps_arr is None:
        eps_for_features = jnp.array([1.0] + [0.0] * (dim - 1),
                                     dtype=jnp.float64)
    else:
        eps_for_features = jnp.asarray(eps_arr, dtype=jnp.float64)

    K_grid_const = K_grid
    offset_const = offset

    def compute_matter_features(R):
        """[Σᵢ sin(K·rᵢ), Σᵢ cos(K·rᵢ), CoM] → (2·n_K + dim,)."""
        K_dot_r = jnp.einsum("id,kd->ki", R, K_grid_const)
        sin_feat = jnp.sum(jnp.sin(K_dot_r), axis=-1)
        cos_feat = jnp.sum(jnp.cos(K_dot_r), axis=-1)
        com = jnp.mean(R, axis=0)
        return jnp.concatenate([sin_feat, cos_feat, com])

    def psi_vec(R, p_pytree):
        """Vector amplitude ψ_n(R) for n=0..N_max.  Returns (N_max+1,)
        complex array.
        """
        log_psi_e_val = log_psi_e_pytree(R, p_pytree["e"])
        feats = compute_matter_features(R)
        log_mag_vec = (
            mlp_apply(p_pytree["mag_mlp"], feats, activation=act_fn)
            + offset_const
        )  # (N_max+1,)
        phase_vec = mlp_apply(
            p_pytree["phase_mlp"], feats, activation=act_fn,
        )  # (N_max+1,)
        # Common matter prefactor.  Real-valued (log_ψ_e returns
        # log|Ψ_e| with no phase since the electronic ansatz is real
        # for the Γ-point HEG used here).
        log_amp = log_psi_e_val + log_mag_vec       # (N_max+1,)
        amp = jnp.exp(log_amp)                       # real
        return amp * (jnp.cos(phase_vec)
                      + 1j * jnp.sin(phase_vec))     # complex

    def psi_vec_flat(r_flat, p_flat):
        R = r_flat.reshape(nelec, dim)
        p_pytree = unravel(p_flat)
        return psi_vec(R, p_pytree)

    def log_psi_total(R, p_pytree):
        """log|Ψ(R)| = ½ log Σ_n |ψ_n(R)|².  Used as the MCMC log-density
        target (so accept ratio uses 2·log|Ψ|).  Numerically stable via
        log-sum-exp on the squared amplitudes.
        """
        log_psi_e_val = log_psi_e_pytree(R, p_pytree["e"])
        feats = compute_matter_features(R)
        log_mag_vec = (
            mlp_apply(p_pytree["mag_mlp"], feats, activation=act_fn)
            + offset_const
        )
        # log|ψ_n|² = 2·(log_ψ_e + log_mag_n).  Common factor 2·log_ψ_e
        # pulled out for stability.
        a = 2.0 * log_mag_vec                        # (N_max+1,)
        # log Σ exp(a) = max(a) + log Σ exp(a − max(a))
        a_max = jnp.max(a)
        log_sum = a_max + jnp.log(jnp.sum(jnp.exp(a - a_max)))
        return log_psi_e_val + 0.5 * log_sum

    def log_psi_total_flat(r_flat, p_flat):
        R = r_flat.reshape(nelec, dim)
        p_pytree = unravel(p_flat)
        return log_psi_total(R, p_pytree)

    # Diagnostics on parameter splits
    e_flat, _ = ravel_pytree(e_init_params)
    n_electronic = int(e_flat.shape[0])
    mag_flat, _ = ravel_pytree(mag_init)
    n_mag_mlp = int(mag_flat.shape[0])
    phase_flat, _ = ravel_pytree(phase_init)
    n_phase_mlp = int(phase_flat.shape[0])

    return {
        "psi_vec":            psi_vec,
        "psi_vec_flat":       psi_vec_flat,
        "log_psi_total":      log_psi_total,
        "log_psi_total_flat": log_psi_total_flat,
        "init_params_flat":   init_params_flat,
        "init_params_pytree": init_params_pytree,
        "unravel":            unravel,
        "n_params":           n_params,
        "n_electronic":       n_electronic,
        "n_mag_mlp":          n_mag_mlp,
        "n_phase_mlp":        n_phase_mlp,
        "n_K":                n_K,
        "n_features":         n_features,
        "K_grid":             K_grid_const,
        "offset":             offset_const,
        "N_max":              int(N_max),
        "nelec":              nelec,
        "dim":                dim,
        "eps":                eps_for_features,
        "electronic_log_psi": log_psi_e_pytree,
        "electronic_graphdef": graphdef,
    }


# ---------------------------------------------------------------------
# Convenience: photon-occupation expectation under the trial
# ---------------------------------------------------------------------

def mean_n(R, p_pytree, psi_vec_fn):
    """⟨n⟩(R) = Σ_n n·|ψ_n(R)|² / Σ_n |ψ_n(R)|² at fixed R.

    Useful diagnostic for whether the trial is actually putting weight
    on n>0 sectors (i.e. whether the Fock truncation N_max is large
    enough).
    """
    psi = psi_vec_fn(R, p_pytree)
    p_n = jnp.abs(psi) ** 2
    Z = jnp.sum(p_n)
    n_vec = jnp.arange(p_n.shape[0], dtype=jnp.float64)
    return jnp.sum(n_vec * p_n) / Z


def n_population(R, p_pytree, psi_vec_fn):
    """|ψ_n|² / Σ_m |ψ_m|² at fixed R — photon-number distribution."""
    psi = psi_vec_fn(R, p_pytree)
    p_n = jnp.abs(psi) ** 2
    return p_n / jnp.sum(p_n)


# ---------------------------------------------------------------------
# Laplacian primitive (local copy — keeps this module decoupled from L5)
# ---------------------------------------------------------------------

def _laplacian_vmap(f, x):
    """O(N) Laplacian + gradient via vmap-of-JVP of grad.

    For a scalar f(x) with x ∈ R^n, returns (∇²f(x), ∇f(x)).
    Numerically robust under outer jit (unlike the linearize+fori_loop
    form, which produces NaN at some vmap batch sizes).
    """
    grad_f = jax.grad(f)
    df = grad_f(x)
    n = x.shape[0]
    eye = jnp.eye(n, dtype=x.dtype)

    def hvp_diag_i(e_i, i):
        _, hvp_e = jax.jvp(grad_f, (x,), (e_i,))
        return hvp_e[i]

    diag = jax.vmap(hvp_diag_i, in_axes=(0, 0))(eye, jnp.arange(n))
    return jnp.sum(diag), df


# ---------------------------------------------------------------------
# Local energy — step 3 of the L8 spec
# ---------------------------------------------------------------------

def make_fock_eloc_no_vee(
    psi_vec_fn,
    *,
    eps,
    lam: float,
    omega_eff: float,
    N_max: int,
    nelec: int,
    dim: int,
    coupling_op: str = "P",
):
    """Build the Fock-basis local energy (excludes V_ee — caller adds Ewald).

    E_loc(R) = ( Σ_{n,n'} ψ_n*·H_{n,n'}·ψ_{n'} ) / Σ_n |ψ_n|²

    where H_{n,n'} decomposes into

      (a) matter diagonal:        −½ ∇² ψ_n      (T_e — V_ee added by caller)
      (b) photon HO diagonal:      (n+½)·Ω_eff · ψ_n
      (c) bilinear off-diagonal:   −λ·q_c·(ε̂·P̂_total)
                                   with q_c = √(1/(2Ω_eff))·(b + b†)
                                   couples n ↔ n±1
                                   ε̂·P̂_total ψ = −i·(ε̂·∇)ψ

    The full off-diagonal contribution to the numerator is

      i·λ·√(1/(2Ω_eff)) ·
        Σ_{m=1}^{N_max} √m · [ ψ_{m−1}*·(ε̂·∇ψ_m) + ψ_m*·(ε̂·∇ψ_{m−1}) ]

    Returns:
      eloc(R, p_pytree) → (re, im) of E_loc per walker (no V_ee).

    Note: caller adds v_ext and V_ee externally.
    """
    eps_arr = jnp.asarray(eps, dtype=jnp.float64)
    if coupling_op != "P":
        raise NotImplementedError(
            f"L8 only supports coupling_op='P' at the moment "
            f"(got {coupling_op!r})"
        )

    n_axis = jnp.arange(N_max + 1, dtype=jnp.float64)
    bilinear_prefactor = jnp.sqrt(1.0 / (2.0 * omega_eff))
    m_idx = jnp.arange(1, N_max + 1, dtype=jnp.float64)
    sqrt_m = jnp.sqrt(m_idx)

    def eloc(R, p_pytree):
        r_flat = R.reshape(-1)

        # Per-n scalar wrappers around psi_vec.  We use the
        # default-arg trick so the lambda captures the current n_idx
        # value (not a late-binding reference).
        def re_at_n(rf, n_idx):
            psi_v = psi_vec_fn(rf.reshape(nelec, dim), p_pytree)
            return jnp.real(psi_v[n_idx])

        def im_at_n(rf, n_idx):
            psi_v = psi_vec_fn(rf.reshape(nelec, dim), p_pytree)
            return jnp.imag(psi_v[n_idx])

        # For each n, compute ψ_n, ∇ψ_n (complex), ∇²ψ_n (complex).
        # Python loop is fine — N_max+1 small, JAX traces+JITs unrolled.
        psi_list = []
        grad_list = []
        lap_list = []
        for n in range(N_max + 1):
            re_fn = lambda rf, _n=n: re_at_n(rf, _n)
            im_fn = lambda rf, _n=n: im_at_n(rf, _n)
            lap_re, grad_re = _laplacian_vmap(re_fn, r_flat)
            lap_im, grad_im = _laplacian_vmap(im_fn, r_flat)
            psi_n = re_fn(r_flat) + 1j * im_fn(r_flat)
            grad_n = grad_re + 1j * grad_im        # (n_dofs,) complex
            lap_n = lap_re + 1j * lap_im
            psi_list.append(psi_n)
            grad_list.append(grad_n)
            lap_list.append(lap_n)

        psi = jnp.stack(psi_list)                  # (N_max+1,)
        grad = jnp.stack(grad_list)                # (N_max+1, n_dofs)
        lap = jnp.stack(lap_list)                  # (N_max+1,)

        psi_conj = jnp.conj(psi)
        Z = jnp.sum(jnp.abs(psi) ** 2)             # real, > 0

        # (a) Matter kinetic: −½ Σ_n ψ_n*·∇²ψ_n.
        T_e_num = -0.5 * jnp.sum(psi_conj * lap)   # complex

        # (b) Photon HO diagonal: Σ_n (n+½)·Ω_eff·|ψ_n|² (real).
        H_phot_num = omega_eff * jnp.sum(
            (n_axis + 0.5) * jnp.abs(psi) ** 2,
        )                                            # real

        # (c) Bilinear off-diagonal.  ε̂·∇ψ_n is a per-n complex scalar.
        grad_per_elec = grad.reshape(N_max + 1, nelec, dim)
        g_eps = jnp.einsum("nid,d->n", grad_per_elec, eps_arr)
        # (N_max+1,) complex
        bilin_sum = jnp.sum(
            sqrt_m * (
                jnp.conj(psi[:-1]) * g_eps[1:]
                + jnp.conj(psi[1:])  * g_eps[:-1]
            )
        )
        bilinear_num = 1j * lam * bilinear_prefactor * bilin_sum

        numer = T_e_num + H_phot_num + bilinear_num
        E_loc = numer / Z
        return jnp.real(E_loc), jnp.imag(E_loc)

    return eloc


# ---------------------------------------------------------------------
# SR primitives — step 4 of the L8 spec
# ---------------------------------------------------------------------
#
# Vector log-derivative for the Fock-basis state:
#
#     ∂_θ log Ψ(R)  ≡  ( Σ_n ψ_n*(R) · ∂_θ ψ_n(R) ) / Σ_n |ψ_n(R)|²
#
# This is the natural generalisation of ∂_θ log Ψ = ∂_θ Ψ / Ψ when the
# state at fixed R is a *vector* (over Fock sectors).  With this single
# substitution, every L5/L7 SR formula carries over unchanged:
#
#     g_θ   = 2 Re ⟨(∂_θ log Ψ)* · (E_loc − ⟨E_loc⟩)⟩
#     S_θθ' = Re ⟨(∂_θ log Ψ)* · (∂_θ' log Ψ)⟩_centred
#
# so the same SMW dual-form solver works after only the per-walker
# Jacobian computation is replaced.

def make_fock_sr_primitives(psi_vec_flat_fn, *, nelec: int, dim: int):
    """Per-walker (du/dθ, dv/dθ) for the vector log-derivative.

    psi_vec_flat_fn: callable (r_flat, p_flat) → (N_max+1,) complex.
    Returns dict with the same keys as the L5 SR primitives — drop-in
    substitute for downstream SMW / CG solvers.
    """
    def re_im_vec(r_flat, p_flat):
        """Stack Re(ψ) and Im(ψ) into a (2, N_max+1) real tensor so JAX
        can differentiate without complex-output friction."""
        psi = psi_vec_flat_fn(r_flat, p_flat)
        return jnp.stack([jnp.real(psi), jnp.imag(psi)])

    jac_re_im = jax.jacrev(re_im_vec, argnums=1)

    def per_walker_d_log_psi(r_flat, p_flat):
        """Returns (du/dθ, dv/dθ) — both (n_params,) real — for one walker.

        Implements
            ∂_θ log Ψ = ( Σ_n ψ_n* · ∂_θ ψ_n ) / Σ_n |ψ_n|²
        with real/imag parts extracted at the end.
        """
        ri = re_im_vec(r_flat, p_flat)          # (2, N_max+1)
        psi = ri[0] + 1j * ri[1]                 # (N_max+1,) complex
        jac = jac_re_im(r_flat, p_flat)          # (2, N_max+1, n_p)
        jac_c = jac[0] + 1j * jac[1]             # (N_max+1, n_p) complex

        Z = jnp.sum(jnp.abs(psi) ** 2)
        psi_conj = jnp.conj(psi)
        # Σ_n psi_n* · jac_{n,k} / Z, result (n_p,) complex.
        d_log = jnp.einsum("n,nk->k", psi_conj, jac_c) / Z
        return jnp.real(d_log), jnp.imag(d_log)

    batched_jacobian = jax.jit(
        jax.vmap(per_walker_d_log_psi, in_axes=(0, None))
    )

    return {
        "per_walker_d_log_psi": per_walker_d_log_psi,
        "batched_jacobian":     batched_jacobian,
    }


# ---------------------------------------------------------------------
# Optimizer — step 5 of the L8 spec (lean version, no fused JIT yet)
# ---------------------------------------------------------------------

class _QEDFockOptimizer:
    """SR-VMC optimizer for the L8 Fock-basis cavity-QED HEG.

    Lean version — Python orchestrates JIT primitives per iter; no
    fused-JIT train step yet.  Per-iter cost is ~1.3× the fully-fused
    version due to host↔device boundaries; train loop is much simpler.

    MCMC: R-only Metropolis with adaptive step size targeting
    ``TARGET_ACCEPTANCE_RATE``; sampling density π(R) ∝ Σ_n |ψ_n(R)|².
    """

    def __init__(
        self,
        config,
        init_key,
        *,
        lr: float = 0.005,
        damping: float = 1e-3,
        n_cg: int = 20,        # not used in SMW path; kept for API compat
        ewald_n_real: int = 3,
        ewald_n_recip: int = 6,
        ewald_eta=None,
        ofname_chkpt=None,
        lr_schedule: str = "cosine",
        lr_min: float = 1e-5,
        lr_T_max=None,
        # SPRING momentum + F-norm clip (SR is μ=0, clip=large)
        spring_mu: float = 0.0,
        spring_norm_clip: float = 0.0,
        # Cavity
        omega: float = 0.1,
        coupling_lambda: float = 0.0,
        coupling_polarization=None,
        coupling_op: str = "P",
        # External one-body potential
        v_ext_amp: float = 0.0,
        v_ext_a=None,
        include_vee: bool = True,
        # L8 architecture knobs
        N_max: int = 6,
        K_max: int = 5,
        mag_mlp_hidden=(64, 64),
        phase_mlp_hidden=(64, 64),
        activation: str = "tanh",
        offset_floor: float = -50.0,
    ):
        self.config = config
        L_y_attr = getattr(config, "L_y", None)
        if L_y_attr is not None:
            self.L_x = float(config.L)
            self.L_y = float(L_y_attr)
        else:
            self.L_x = self.L_y = float(config.L)
        self.L = float(config.L)
        self.n_up = int(config.n_up)
        self.n_down = int(config.n_down)
        self.nelec = self.n_up + self.n_down
        self.dim = int(getattr(config, "dim", 3))
        if self.dim != 2:
            raise ValueError(
                f"_QEDFockOptimizer currently supports dim=2 only "
                f"(got dim={self.dim})"
            )
        self.lr = float(lr)
        self.damping = float(damping)
        self.n_cg = int(n_cg)
        self.ofname_chkpt = ofname_chkpt

        # Cavity setup (P-coupling only for L8 v1)
        self.omega = float(omega)
        self.coupling_lambda = float(coupling_lambda)
        self.coupling_op = str(coupling_op)
        if self.coupling_op != "P":
            raise NotImplementedError(
                "L8 v1 supports coupling_op='P' only "
                f"(got {self.coupling_op!r})"
            )
        self.omega_eff = float(jnp.sqrt(
            self.omega ** 2 + self.nelec * self.coupling_lambda ** 2
        ))
        if coupling_polarization is None:
            eps_list = [1.0] + [0.0] * (self.dim - 1)
        else:
            eps_list = list(coupling_polarization)
        eps_arr = jnp.asarray(eps_list, dtype=jnp.float64)
        eps_arr = eps_arr / jnp.linalg.norm(eps_arr)
        self.eps = eps_arr

        # Build trial
        self.fock = build_fock_log_psi(
            config, init_key,
            N_max=N_max, K_max=K_max,
            mag_mlp_hidden=mag_mlp_hidden,
            phase_mlp_hidden=phase_mlp_hidden,
            activation=activation,
            offset_floor=offset_floor,
            eps_arr=self.eps,
        )
        self.params_flat = self.fock["init_params_flat"]
        self.unravel = self.fock["unravel"]
        self.n_params = self.fock["n_params"]
        self.N_max = int(N_max)

        # SR primitives
        self.sr = make_fock_sr_primitives(
            self.fock["psi_vec_flat"],
            nelec=self.nelec, dim=self.dim,
        )

        # Local energy (no V_ee)
        self.eloc_no_vee_fn = make_fock_eloc_no_vee(
            self.fock["psi_vec"],
            eps=self.eps,
            lam=self.coupling_lambda,
            omega_eff=self.omega_eff,
            N_max=self.N_max,
            nelec=self.nelec, dim=self.dim,
            coupling_op=self.coupling_op,
        )

        # Lattice + Ewald
        if abs(self.L_x - self.L_y) > 1e-9:
            from .psi.nn.periodic import make_rectangular_lattice
            self.lattice = make_rectangular_lattice(self.L_x, self.L_y)
            ewald_L = (self.L_x, self.L_y)
        else:
            self.lattice = make_square_lattice(self.L_x)
            ewald_L = self.L_x
        self.ewald = build_ewald_tables_dim(
            ewald_L, dim=self.dim, eta=ewald_eta,
            n_real=ewald_n_real, n_recip=ewald_n_recip,
        )
        self._ewald_per_walker = jax.jit(jax.vmap(
            lambda r: ewald_2d_pair_energy(r[None], self.ewald)[0]
        ))
        self.include_vee = bool(include_vee)

        # v_ext (Weber cosine TI-breaking)
        self.v_ext_amp = float(v_ext_amp)
        self.v_ext_a = float(v_ext_a) if v_ext_a is not None else float(self.L_x)
        k_ext = 2.0 * jnp.pi / self.v_ext_a
        amp = self.v_ext_amp

        def _vext_one(R):
            return -amp * jnp.sum(jnp.cos(k_ext * R))
        self._vext_per_walker = jax.jit(jax.vmap(_vext_one))

        # LR schedule + SR solver
        self.lr_schedule = lr_schedule
        self.lr_min = float(lr_min)
        self.lr_T_max = lr_T_max
        self.spring_mu = float(spring_mu)
        self.spring_norm_clip = float(spring_norm_clip)
        self.smw_solver = make_l5_sr_smw_solver()

        # Batched eloc
        def _eloc_one(R_i, p_pytree):
            return self.eloc_no_vee_fn(R_i, p_pytree)
        self._eloc_batched = jax.jit(jax.vmap(
            _eloc_one, in_axes=(0, None),
        ))

        # MCMC step (jitted)
        log_psi_total_flat = self.fock["log_psi_total_flat"]
        nelec_c, dim_c = self.nelec, self.dim
        lattice_c = self.lattice

        def R_move_one(key, R, step, p_flat):
            kp, ka = jax.random.split(key)
            R_prop = R + step * jax.random.normal(
                kp, R.shape, dtype=jnp.float64,
            )
            R_prop = wrap_to_cell(R_prop, lattice_c)
            lp_old = log_psi_total_flat(R.reshape(-1), p_flat)
            lp_new = log_psi_total_flat(R_prop.reshape(-1), p_flat)
            accept = jax.random.uniform(ka) < jnp.exp(
                2.0 * (lp_new - lp_old)
            )
            return jnp.where(accept, R_prop, R), accept

        self._R_move_batch = jax.vmap(
            R_move_one, in_axes=(0, 0, None, None),
        )

        def _mcmc_step(rng_key, R, step_R, p_flat, num_walkers):
            keys = jax.random.split(rng_key, num_walkers)
            R_new, acc = self._R_move_batch(keys, R, step_R, p_flat)
            ar = jnp.mean(acc).astype(jnp.float64)
            new_step = _adapt_step_size(step_R, ar)
            return R_new, new_step, ar
        # Make num_walkers a closure variable; JIT will recompile per batch.
        self._mcmc_step_uncompiled = _mcmc_step

    def _compute_lr(self, it):
        if self.lr_schedule == "cosine":
            T = int(self.lr_T_max) if self.lr_T_max is not None else 500
            if it >= T:
                return self.lr_min
            cos = 0.5 * (1.0 + jnp.cos(jnp.pi * it / T))
            return self.lr_min + (self.lr - self.lr_min) * float(cos)
        return self.lr

    def initialize_walkers(self, rng_key, num_walkers):
        """R walkers — uniform in [0, L]^(N×dim)."""
        L_arr = jnp.asarray([self.L_x, self.L_y], dtype=jnp.float64)
        u = jax.random.uniform(
            rng_key, (num_walkers, self.nelec, self.dim),
            dtype=jnp.float64,
        )
        return u * L_arr

    def _batched_eloc_with_vee(self, R, p_pytree):
        re_no_vee, im = self._eloc_batched(R, p_pytree)
        V_ee = self._ewald_per_walker(R) if self.include_vee else 0.0
        V_ext = (
            self._vext_per_walker(R) if self.v_ext_amp != 0.0 else 0.0
        )
        return re_no_vee + V_ee + V_ext, im

    def train(
        self,
        rng_key,
        num_walkers: int,
        n_iters: int,
        *,
        mcmc_decorr_steps: int = 15,
        mc_timestep_R: float = 0.1,
        equil_steps: int = 50,
        save_every: int = 0,
        verbose: int = 1,
        chkpt_path=None,
        log_file=None,
    ):
        """Run SR-VMC training for ``n_iters`` iterations.

        Returns (params_flat, R_walkers) at the end.  Writes a single
        chkpt with name ``chkpt_path`` (or self.ofname_chkpt) every
        ``save_every`` iters when > 0.
        """
        if chkpt_path is None:
            chkpt_path = self.ofname_chkpt

        key_init, key_train = jax.random.split(rng_key)
        R = self.initialize_walkers(key_init, num_walkers)
        step_R = jnp.float64(mc_timestep_R)
        params_flat = self.params_flat
        prev_delta = jnp.zeros_like(params_flat)

        # Equilibration
        for _ in range(equil_steps):
            key_train, sub = jax.random.split(key_train)
            R, step_R, ar = self._mcmc_step_uncompiled(
                sub, R, step_R, params_flat, num_walkers,
            )
        if verbose >= 1:
            print(
                f"# L8 SR-VMC training — {self.n_params} params  "
                f"(elec={self.fock['n_electronic']}, "
                f"mag_mlp={self.fock['n_mag_mlp']}, "
                f"phase_mlp={self.fock['n_phase_mlp']}, "
                f"N_max={self.N_max}, n_K={self.fock['n_K']})",
                file=log_file,
            )
            print(
                f"# Equilibration: ar_R={float(ar):.3f}; "
                f"step_R={float(step_R):.3f}",
                file=log_file,
            )
            print(
                f"# Ω={self.omega:.4f}, λ={self.coupling_lambda:.4f}, "
                f"Ω_eff={self.omega_eff:.4f}",
                file=log_file,
            )
            print(
                "# iter   <Re E>/N        Var(Re E)      <Im E>/N        |g|",
                file=log_file,
            )
            if log_file is not None:
                log_file.flush()

        import time
        damping = jnp.float64(self.damping)
        mu = jnp.float64(self.spring_mu)
        c_clip = jnp.float64(self.spring_norm_clip)

        for it in range(1, n_iters + 1):
            t0 = time.time()

            # 1) MCMC decorrelation (R-only)
            for _ in range(mcmc_decorr_steps):
                key_train, sub = jax.random.split(key_train)
                R, step_R, _ = self._mcmc_step_uncompiled(
                    sub, R, step_R, params_flat, num_walkers,
                )

            # 2) E_loc + Ewald + V_ext
            p_pytree = self.unravel(params_flat)
            e_re, e_im = self._batched_eloc_with_vee(R, p_pytree)

            # 3) SR Jacobian
            r_flat = R.reshape(num_walkers, -1)
            Jac_u, Jac_v = self.sr["batched_jacobian"](r_flat, params_flat)

            # 4) SMW solve
            delta, e_mean, e_var, im_mean, g_norm, scale = self.smw_solver(
                Jac_u, Jac_v, e_re, e_im,
                prev_delta, damping, mu, c_clip,
            )

            # 5) parameter update
            lr_now = self._compute_lr(it - 1)
            params_flat = params_flat - lr_now * delta
            prev_delta = delta

            dt = time.time() - t0
            if verbose >= 1:
                # log every iter for first 10, then every 10
                if it <= 10 or it % 10 == 0:
                    print(
                        f"{it:5d}  {float(e_mean) / self.nelec:+.6e}  "
                        f"{float(e_var):.4e}  "
                        f"{float(im_mean) / self.nelec:+.3e}  "
                        f"{float(g_norm):.3e}  ({dt:.2f}s)",
                        file=log_file,
                    )
                    if log_file is not None:
                        log_file.flush()

            # Chkpt
            if save_every > 0 and chkpt_path is not None and it % save_every == 0:
                np.savez(
                    chkpt_path,
                    params_flat=np.asarray(params_flat),
                    R=np.asarray(R),
                    R_step_size=np.asarray(step_R),
                    E_final_ha=np.asarray(e_mean / self.nelec),
                    n_iters_trained=int(it),
                )
                if verbose >= 1:
                    print(
                        f"# [chkpt] saved at iter {it}, "
                        f"E/N={float(e_mean)/self.nelec:+.6f} Ha "
                        f"→ {chkpt_path}",
                        file=log_file,
                    )
                    if log_file is not None:
                        log_file.flush()

        self.params_flat = params_flat
        return params_flat, R

    # ---------------------------------------------------------------
    # Fused JIT train step — speed up over the Python-orchestrated path
    # ---------------------------------------------------------------
    def _build_fused_train_step(self, num_walkers, mcmc_decorr_steps):
        """One @jax.jit train_step that fuses MCMC + eloc + Jac + SMW + update.

        Mirrors the L5 _build_fused_train_step pattern but drops the q_c
        MCMC branch (Fock-basis has no continuous photon coordinate).

        Returns ``train_step(carry, lr) → (new_carry, metrics)`` with
        carry = (rng, R, step_R, params_flat, prev_delta).
        """
        log_psi_total_flat = self.fock["log_psi_total_flat"]
        lattice = self.lattice
        unravel = self.unravel
        eloc_no_vee_fn = self.eloc_no_vee_fn
        ewald = self.ewald
        damping_c = jnp.float64(self.damping)
        mu_c = jnp.float64(self.spring_mu)
        c_clip_c = jnp.float64(self.spring_norm_clip)
        include_vee = self.include_vee
        v_ext_amp = self.v_ext_amp
        k_ext = 2.0 * jnp.pi / self.v_ext_a
        per_walker_d_log_psi = self.sr["per_walker_d_log_psi"]

        # ---- per-walker primitives ----
        def R_move_one(key, R, step, p_flat):
            kp, ka = jax.random.split(key)
            R_prop = R + step * jax.random.normal(
                kp, R.shape, dtype=jnp.float64,
            )
            R_prop = wrap_to_cell(R_prop, lattice)
            lp_old = log_psi_total_flat(R.reshape(-1), p_flat)
            lp_new = log_psi_total_flat(R_prop.reshape(-1), p_flat)
            accept = jax.random.uniform(ka) < jnp.exp(
                2.0 * (lp_new - lp_old)
            )
            return jnp.where(accept, R_prop, R), accept

        R_move_batch = jax.vmap(
            R_move_one, in_axes=(0, 0, None, None),
        )
        batched_jac = jax.vmap(
            per_walker_d_log_psi, in_axes=(0, None),
        )

        def eloc_one(R_i, p_pytree):
            return eloc_no_vee_fn(R_i, p_pytree)
        eloc_batched = jax.vmap(eloc_one, in_axes=(0, None))

        def ewald_batched(R):
            _one = lambda r: ewald_2d_pair_energy(r[None], ewald)[0]
            return jax.vmap(_one)(R)

        def vext_batched(R):
            if v_ext_amp == 0.0:
                return jnp.zeros(R.shape[0], dtype=R.dtype)
            return -v_ext_amp * jnp.sum(jnp.cos(k_ext * R), axis=(1, 2))

        # ---- MCMC inner step (one Metropolis sweep on R) ----
        def mcmc_one_step(carry, _):
            rng, R, step_R, p_flat = carry
            rng, sub = jax.random.split(rng)
            keys = jax.random.split(sub, num_walkers)
            R, acc_R = R_move_batch(keys, R, step_R, p_flat)
            ar_R = jnp.mean(acc_R).astype(jnp.float64)
            step_R = _adapt_step_size(step_R, ar_R)
            return (rng, R, step_R, p_flat), ar_R

        # ---- Full fused step ----
        @jax.jit
        def fused_train_step(carry, lr):
            (rng, R, step_R, params_flat, prev_delta) = carry

            # 1) MCMC decorrelation (Python loop unrolled inside jit
            #    — same convention as L5; avoids lax.scan issues
            #    observed at production scale).
            ar_R_sum = jnp.float64(0.0)
            mcmc_carry = (rng, R, step_R, params_flat)
            for _ in range(mcmc_decorr_steps):
                mcmc_carry, ar_R_i = mcmc_one_step(mcmc_carry, None)
                ar_R_sum = ar_R_sum + ar_R_i
            rng, R, step_R, _ = mcmc_carry
            ar_R = ar_R_sum / mcmc_decorr_steps

            # 2) E_loc + Ewald + V_ext
            p_pytree = unravel(params_flat)
            re_no_vee, e_im = eloc_batched(R, p_pytree)
            V_ee = (
                ewald_batched(R) if include_vee
                else jnp.zeros(num_walkers)
            )
            V_ext = vext_batched(R)
            e_re = re_no_vee + V_ee + V_ext

            # 3) SR Jacobian per walker (vector log-derivative form)
            r_flat = R.reshape(num_walkers, -1)
            Jac_u, Jac_v = batched_jac(r_flat, params_flat)

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
            do_s = jnp.concatenate([dJu, dJv], axis=0)
            inv_lambda_n = 1.0 / (damping_c * n_w)
            K = (do_s @ do_s.T) * inv_lambda_n
            I_plus_K = K + jnp.eye(K.shape[0], dtype=K.dtype)
            o_rhs = (do_s @ rhs) * inv_lambda_n
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

            # 5) Update
            params_flat = params_flat - lr * delta
            prev_delta = delta

            new_carry = (rng, R, step_R, params_flat, prev_delta)
            metrics = {
                "e_mean": e_re_mean,
                "e_var": e_re_var,
                "im_mean": e_im_mean,
                "g_norm": g_norm,
                "scale": scale,
                "ar_R": ar_R,
                "step_R": step_R,
            }
            return new_carry, metrics

        return fused_train_step

    def train_fused(
        self,
        rng_key,
        num_walkers: int,
        n_iters: int,
        *,
        mcmc_decorr_steps: int = 15,
        mc_timestep_R: float = 0.1,
        equil_steps: int = 50,
        save_every: int = 0,
        verbose: int = 1,
        chkpt_path=None,
        log_file=None,
    ):
        """Fused-JIT version of train.  Drop-in replacement that should
        be 2–3× faster than the Python-orchestrated train() once the
        first iter has paid the JIT compile cost.
        """
        if chkpt_path is None:
            chkpt_path = self.ofname_chkpt

        key_init, key_train = jax.random.split(rng_key)
        R = self.initialize_walkers(key_init, num_walkers)
        step_R = jnp.float64(mc_timestep_R)
        params_flat = self.params_flat
        prev_delta = jnp.zeros_like(params_flat)

        # Equilibration (re-uses uncompiled mcmc — small overhead)
        for _ in range(equil_steps):
            key_train, sub = jax.random.split(key_train)
            R, step_R, ar = self._mcmc_step_uncompiled(
                sub, R, step_R, params_flat, num_walkers,
            )
        if verbose >= 1:
            print(
                f"# L8 SR-VMC training (fused JIT) — {self.n_params} params  "
                f"(elec={self.fock['n_electronic']}, "
                f"mag_mlp={self.fock['n_mag_mlp']}, "
                f"phase_mlp={self.fock['n_phase_mlp']}, "
                f"N_max={self.N_max}, n_K={self.fock['n_K']})",
                file=log_file,
            )
            print(
                f"# Equilibration: ar_R={float(ar):.3f}; "
                f"step_R={float(step_R):.3f}",
                file=log_file,
            )
            print(
                f"# Ω={self.omega:.4f}, λ={self.coupling_lambda:.4f}, "
                f"Ω_eff={self.omega_eff:.4f}",
                file=log_file,
            )
            print(
                "# iter   <Re E>/N        Var(Re E)      <Im E>/N        |g|",
                file=log_file,
            )
            if log_file is not None:
                log_file.flush()

        # Build the fused step (compiled once, num_walkers and
        # mcmc_decorr_steps are baked in as static shapes).
        fused_step = self._build_fused_train_step(
            num_walkers, mcmc_decorr_steps,
        )

        import time
        carry = (key_train, R, step_R, params_flat, prev_delta)
        for it in range(1, n_iters + 1):
            t0 = time.time()
            lr_now = jnp.float64(self._compute_lr(it - 1))
            carry, metrics = fused_step(carry, lr_now)
            # Block to get accurate per-iter timing
            metrics["e_mean"].block_until_ready()
            dt = time.time() - t0

            if verbose >= 1 and (it <= 10 or it % 10 == 0):
                print(
                    f"{it:5d}  "
                    f"{float(metrics['e_mean']) / self.nelec:+.6e}  "
                    f"{float(metrics['e_var']):.4e}  "
                    f"{float(metrics['im_mean']) / self.nelec:+.3e}  "
                    f"{float(metrics['g_norm']):.3e}  ({dt:.2f}s)",
                    file=log_file,
                )
                if log_file is not None:
                    log_file.flush()

            if save_every > 0 and chkpt_path is not None and it % save_every == 0:
                _, R_now, step_R_now, params_now, _ = carry
                np.savez(
                    chkpt_path,
                    params_flat=np.asarray(params_now),
                    R=np.asarray(R_now),
                    R_step_size=np.asarray(step_R_now),
                    E_final_ha=np.asarray(metrics["e_mean"] / self.nelec),
                    n_iters_trained=int(it),
                )
                if verbose >= 1:
                    print(
                        f"# [chkpt] saved at iter {it}, "
                        f"E/N={float(metrics['e_mean']) / self.nelec:+.6f} Ha "
                        f"→ {chkpt_path}",
                        file=log_file,
                    )
                    if log_file is not None:
                        log_file.flush()

        # Unpack final state
        _, R_final, _, params_final, _ = carry
        self.params_flat = params_final
        return params_final, R_final

    def evaluate(
        self,
        rng_key,
        num_walkers: int,
        *,
        num_blocks: int,
        steps_per_block: int = 10,
        equil_blocks: int = 5,
        mc_timestep_R: float = 0.1,
        params_flat=None,
        R_init=None,
        verbose: int = 1,
        log_file=None,
    ):
        """Compute ⟨H⟩/N via blocked sampling.  Returns dict of stats.
        """
        if params_flat is None:
            params_flat = self.params_flat

        key_init, key_eval = jax.random.split(rng_key)
        if R_init is None:
            R = self.initialize_walkers(key_init, num_walkers)
        else:
            R = R_init
        step_R = jnp.float64(mc_timestep_R)

        # Equilibration blocks
        for _ in range(equil_blocks * steps_per_block):
            key_eval, sub = jax.random.split(key_eval)
            R, step_R, _ = self._mcmc_step_uncompiled(
                sub, R, step_R, params_flat, num_walkers,
            )

        block_re = []
        block_im = []
        p_pytree = self.unravel(params_flat)
        import time
        t0 = time.time()
        for b in range(num_blocks):
            for _ in range(steps_per_block):
                key_eval, sub = jax.random.split(key_eval)
                R, step_R, _ = self._mcmc_step_uncompiled(
                    sub, R, step_R, params_flat, num_walkers,
                )
            e_re, e_im = self._batched_eloc_with_vee(R, p_pytree)
            block_re.append(float(jnp.mean(e_re)) / self.nelec)
            block_im.append(float(jnp.mean(e_im)) / self.nelec)
            if verbose >= 2:
                print(
                    f"  block {b+1}/{num_blocks}: <E>/N = {block_re[-1]:+.6e} Ha",
                    file=log_file,
                )
        dt = time.time() - t0

        e_re_arr = np.asarray(block_re)
        e_im_arr = np.asarray(block_im)
        mean_re = float(e_re_arr.mean())
        std_re = float(e_re_arr.std(ddof=1)) if len(e_re_arr) > 1 else 0.0
        sem_re = std_re / max(1.0, np.sqrt(len(e_re_arr)))

        if verbose >= 1:
            print(
                f"# Eval: {num_blocks} blocks × {steps_per_block} steps × "
                f"{num_walkers} walkers, time {dt:.1f}s",
                file=log_file,
            )
            print(
                f"  E/N = {mean_re:+.6e} ± {sem_re:.2e} Ha  "
                f"(im = {float(e_im_arr.mean()):+.3e} ± "
                f"{float(e_im_arr.std(ddof=1))/np.sqrt(len(e_im_arr)):.2e})",
                file=log_file,
            )

        return {
            "E_per_e_ha":     mean_re,
            "E_per_e_sem":    sem_re,
            "E_per_e_std":    std_re,
            "Im_per_e_ha":    float(e_im_arr.mean()),
            "block_re":       e_re_arr.tolist(),
            "block_im":       e_im_arr.tolist(),
            "wall_time_s":    dt,
        }




