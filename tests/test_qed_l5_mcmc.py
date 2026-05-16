"""Phase 5a-1 validation: joint (R, q_c) walker with composite MCMC.

Verifies the new walker structure + Metropolis moves on R and q_c
sample the trial distribution |Ψ(R, q_c)|² correctly.

Toy trial:
    Ψ(R, q_c) = exp(−α·Σᵢⱼ rᵢⱼ²)·exp(−Ω/2·q_c²)        # factorised, real

Analytical references:
    R-marginal:  3D Gaussian product of pairwise distances; complicated.
                 We sidestep with a simpler "non-interacting" toy:
                     log_psi_e(R) = −α·Σᵢ rᵢ·rᵢ           (single-particle Gaussian)
                 → ⟨rᵢ²⟩_marginal = D/(2α) per electron        (D=2 for 2D)
    q_c-marginal: HO ground-state Gaussian width 1/√Ω →
                 ⟨q_c²⟩ = 1/(2·Ω)
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)


# ---------------------------------------------------------------------
# Toy log|Ψ| function (real, factorised)
# ---------------------------------------------------------------------

def log_psi_toy(R, q_c, alpha, omega):
    """log|Ψ(R, q_c)| = −α·Σᵢ rᵢ·rᵢ − Ω/2·q_c².

    R: (N_e, dim);  q_c: scalar.   Returns: real scalar.
    """
    return -alpha * jnp.sum(R * R) - 0.5 * omega * q_c * q_c


# ---------------------------------------------------------------------
# MCMC primitives
# ---------------------------------------------------------------------

def metropolis_R_step(rng_key, R, q_c, step_size, alpha, omega):
    """One R-step: Gaussian proposal on R, accept/reject on |Ψ|² ratio."""
    key_prop, key_acc = jax.random.split(rng_key)
    R_prop = R + step_size * jax.random.normal(key_prop, R.shape)
    log_old = log_psi_toy(R, q_c, alpha, omega)
    log_new = log_psi_toy(R_prop, q_c, alpha, omega)
    log_ratio = 2.0 * (log_new - log_old)
    accept = jax.random.uniform(key_acc) < jnp.exp(log_ratio)
    R_new = jnp.where(accept, R_prop, R)
    return R_new, accept


def metropolis_qc_step(rng_key, R, q_c, step_size, alpha, omega):
    """One q_c-step: Gaussian proposal on q_c, accept/reject on |Ψ|²."""
    key_prop, key_acc = jax.random.split(rng_key)
    q_prop = q_c + step_size * jax.random.normal(key_prop)
    log_old = log_psi_toy(R, q_c, alpha, omega)
    log_new = log_psi_toy(R, q_prop, alpha, omega)
    log_ratio = 2.0 * (log_new - log_old)
    accept = jax.random.uniform(key_acc) < jnp.exp(log_ratio)
    q_new = jnp.where(accept, q_prop, q_c)
    return q_new, accept


# Vectorise over walker batch
metropolis_R_batch = jax.jit(
    jax.vmap(metropolis_R_step, in_axes=(0, 0, 0, None, None, None)),
)
metropolis_qc_batch = jax.jit(
    jax.vmap(metropolis_qc_step, in_axes=(0, 0, 0, None, None, None)),
)


# ---------------------------------------------------------------------
# Composite MCMC chain (Option B: sequential R-step + q_c-step per iter)
# ---------------------------------------------------------------------

def run_chain(
    rng_key, n_walkers, N_e, dim, n_equil, n_sample, alpha, omega,
    R_step_size=0.5, qc_step_size=0.5,
):
    """Returns (R_samples, q_c_samples) of shape
    (n_sample, n_walkers, ...).  Walkers initialised at origin/zero."""
    rng_key, init_key = jax.random.split(rng_key)
    R = 0.1 * jax.random.normal(init_key, (n_walkers, N_e, dim))
    q_c = jnp.zeros(n_walkers)

    R_samples = []
    qc_samples = []

    for it in range(n_equil + n_sample):
        rng_key, sub_R = jax.random.split(rng_key)
        keys_R = jax.random.split(sub_R, n_walkers)
        R, _ = metropolis_R_batch(
            keys_R, R, q_c, R_step_size, alpha, omega,
        )

        rng_key, sub_q = jax.random.split(rng_key)
        keys_q = jax.random.split(sub_q, n_walkers)
        q_c, _ = metropolis_qc_batch(
            keys_q, R, q_c, qc_step_size, alpha, omega,
        )

        if it >= n_equil:
            R_samples.append(R)
            qc_samples.append(q_c)

    return jnp.stack(R_samples), jnp.stack(qc_samples)


# ---------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------

def test_qc_marginal_matches_HO_ground_state():
    """At the toy trial, the q_c marginal is HO ground state with
    width 1/√Ω.  ⟨q_c²⟩ = 1/(2Ω)."""
    rng_key = jax.random.key(42)
    alpha = 0.05
    omega = 1.0
    n_walkers = 1024
    R_samples, qc_samples = run_chain(
        rng_key, n_walkers, N_e=4, dim=2,
        n_equil=500, n_sample=500,
        alpha=alpha, omega=omega,
        R_step_size=2.5, qc_step_size=1.0,
    )
    qc_sq_mean = float(jnp.mean(qc_samples ** 2))
    expected = 1.0 / (2.0 * omega)
    rel_err = abs(qc_sq_mean - expected) / expected
    assert rel_err < 0.02, (
        f"⟨q_c²⟩={qc_sq_mean}, expected {expected}, rel_err={rel_err}"
    )


def test_qc_mean_is_zero():
    """Symmetric trial → ⟨q_c⟩ = 0."""
    rng_key = jax.random.key(7)
    R_samples, qc_samples = run_chain(
        rng_key, n_walkers=1024, N_e=4, dim=2,
        n_equil=500, n_sample=500,
        alpha=0.05, omega=1.0,
        R_step_size=2.5, qc_step_size=1.0,
    )
    qc_mean = float(jnp.mean(qc_samples))
    qc_serr = float(jnp.std(qc_samples)) / np.sqrt(qc_samples.size)
    assert abs(qc_mean) < 5 * qc_serr, (
        f"⟨q_c⟩={qc_mean} ± {qc_serr} not consistent with 0"
    )


def test_R_marginal_matches_single_particle_gaussian():
    """log|Ψ| = −α·Σ x²  →  |Ψ|² = exp(−2α·Σx²)  →  ⟨x²⟩_1D = 1/(4α).
    Total Σ over (N_e × dim) coordinates: N_e·D / (4α)."""
    rng_key = jax.random.key(13)
    alpha = 0.05
    N_e = 4
    dim = 2
    R_samples, qc_samples = run_chain(
        rng_key, n_walkers=1024, N_e=N_e, dim=dim,
        n_equil=500, n_sample=500,
        alpha=alpha, omega=1.0,
        R_step_size=2.5, qc_step_size=1.0,
    )
    R_sq_total = float(jnp.mean(jnp.sum(R_samples ** 2, axis=(2, 3))))
    expected = N_e * dim / (4.0 * alpha)
    rel_err = abs(R_sq_total - expected) / expected
    assert rel_err < 0.05, (
        f"Σᵢ⟨rᵢ²⟩={R_sq_total}, expected {expected}, rel_err={rel_err}"
    )


def test_R_qc_independence_for_factorised_trial():
    """For a factorised trial, R and q_c samples should be uncorrelated.
    The cross-correlation ⟨q_c·Σrᵢ²⟩ − ⟨q_c⟩·⟨Σrᵢ²⟩ = 0."""
    rng_key = jax.random.key(99)
    R_samples, qc_samples = run_chain(
        rng_key, n_walkers=1024, N_e=4, dim=2,
        n_equil=500, n_sample=500,
        alpha=0.05, omega=1.0,
        R_step_size=2.5, qc_step_size=1.0,
    )
    R_sq = jnp.sum(R_samples ** 2, axis=(2, 3))     # (n_sample, n_walkers)
    cov = float(jnp.mean(qc_samples * R_sq) - jnp.mean(qc_samples) * jnp.mean(R_sq))
    # serr of cov ~ std(qc·R_sq)/√n
    serr = float(jnp.std(qc_samples * R_sq)) / np.sqrt(qc_samples.size)
    assert abs(cov) < 5 * serr, (
        f"cov(q_c, R²)={cov} ± {serr} not consistent with 0 "
        f"(factorised trial → should be 0)"
    )
