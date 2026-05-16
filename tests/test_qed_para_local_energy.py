"""Unit tests for the Step 5 paramagnetic local-energy formula.

Pauli-Fierz velocity-gauge cavity-QED Hamiltonian, q=0 mode:

    H_para = +i·g·(b+b†)·Σᵢ(ε·∇ᵢ)        # acting on Ψ

Step 5 trial wavefunction (Option C multi-K phase + coherent c_n):

    log|c_n(α_coh)| = n·log(α_coh) − ½·log(n!) − α_coh²/2
    θ_n(R)         = n · Σ_k α_k · F_k(R),   F_k(R) = Σᵢ sin(K_k·rᵢ)
    log Ψ(R, n)    = log|ψ_e(R)| + log c_n + i·θ_n(R)

Algebraic simplifications (n-linear θ + coherent c_n):
    √(n+1)·(c_{n+1}/c_n) = α_coh        # n-independent!
    √n  ·(c_{n−1}/c_n)   = n/α_coh
    θ_{n+1} − θ_n         = f(R)        # n-independent
    ε·∂_R θ_m            = m · ∂_ε f(R)

Closed-form local-energy at walker (R, n):

    E_loc_para(R, n) = i·g · [
        α_coh        · exp(+i·f) · B_{n+1}(R) · I_{n+1≤nph_max}
      + (n/α_coh)    · exp(−i·f) · B_{n−1}(R) · I_{n−1≥0}
    ]

with  B_m(R) = (ε·∇R log|ψ_e|) + i·m·∂_ε f(R)
      f(R)  = Σ_k α_k · F_k(R)
      ∂_ε f = Σ_k α_k · (ε·K_k) · G_k(R),  G_k(R) = Σᵢ cos(K_k·rᵢ)

This file implements that formula and an FD ground truth, then verifies
the production-code path matches it.
"""
from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)


# ---------------------------------------------------------------------
# Reference implementations
# ---------------------------------------------------------------------

def _log_c_n(n, coh_alpha, nph_max):
    """Coherent-state amplitude log c_n."""
    return (
        n * jnp.log(coh_alpha)
        - 0.5 * math.lgamma(n + 1)
        - 0.5 * coh_alpha ** 2
    )


def _theta_n(R, n, phase_alpha, K_vectors):
    """θ_n(R) = n · Σ_k α_k · Σᵢ sin(K_k·rᵢ).  Option C."""
    K_dot_r = jnp.einsum("id,kd->ki", R, K_vectors)        # (n_K, n_elec)
    F_per_k = jnp.sum(jnp.sin(K_dot_r), axis=-1)           # (n_K,)
    return float(n) * jnp.dot(phase_alpha, F_per_k)


def _log_psi_complex(R, n, phase_alpha, coh_alpha, K_vectors,
                     log_psi_e_fn, nph_max):
    return (
        log_psi_e_fn(R)
        + _log_c_n(n, coh_alpha, nph_max)
        + 1j * _theta_n(R, n, phase_alpha, K_vectors)
    )


def E_loc_para_analytical(
    R, n, phase_alpha, coh_alpha, K_vectors, eps,
    log_psi_e_fn, g, nph_max,
):
    """E_loc_para from the closed-form formula above."""
    # ε · ∂_R log|ψ_e|
    grad_log_psi_e = jax.grad(log_psi_e_fn)(R)             # (n_elec, dim)
    p = jnp.einsum("id,d->", grad_log_psi_e, eps)          # scalar

    # f(R) = Σ_k α_k · Σᵢ sin(K_k·rᵢ);  ∂_ε f = Σ_k α_k·(ε·K_k)·Σᵢ cos(K_k·rᵢ)
    K_dot_r = jnp.einsum("id,kd->ki", R, K_vectors)        # (n_K, n_elec)
    F_per_k = jnp.sum(jnp.sin(K_dot_r), axis=-1)           # (n_K,)
    G_per_k = jnp.sum(jnp.cos(K_dot_r), axis=-1)           # (n_K,)
    eps_dot_K = jnp.einsum("kd,d->k", K_vectors, eps)      # (n_K,)
    f_val = jnp.dot(phase_alpha, F_per_k)
    df_val = jnp.dot(phase_alpha * eps_dot_K, G_per_k)

    contrib = jnp.zeros((), dtype=jnp.complex128)

    # m = n+1 channel
    if n + 1 <= nph_max:
        m_up = n + 1
        B_up = p + 1j * float(m_up) * df_val
        contrib = contrib + (
            coh_alpha
            * jnp.exp(1j * f_val.astype(jnp.complex128))
            * B_up
        )

    # m = n−1 channel
    if n - 1 >= 0:
        m_dn = n - 1
        B_dn = p + 1j * float(m_dn) * df_val
        contrib = contrib + (
            (float(n) / coh_alpha)
            * jnp.exp(-1j * f_val.astype(jnp.complex128))
            * B_dn
        )

    return 1j * g * contrib


def E_loc_para_numerical(
    R, n, phase_alpha, coh_alpha, K_vectors, eps,
    log_psi_e_fn, g, nph_max, h=1e-4,
):
    """FD ground truth: directly evaluate
        E_loc = +i·g · Σ_m ⟨n|b+b†|m⟩ · (Ψ(R, m)/Ψ(R, n)) · ∂_ε log Ψ(R, m)
    via central differences on the complex log Ψ.
    """
    def log_psi_at(R_, m):
        return _log_psi_complex(
            R_, m, phase_alpha, coh_alpha, K_vectors,
            log_psi_e_fn, nph_max,
        )

    R_plus = R + h * eps[None, :]
    R_minus = R - h * eps[None, :]

    def eps_dot_grad_log_psi(m):
        return (
            log_psi_at(R_plus, m) - log_psi_at(R_minus, m)
        ) / (2.0 * h)

    contrib = jnp.zeros((), dtype=jnp.complex128)
    if n + 1 <= nph_max:
        m_up = n + 1
        log_ratio = log_psi_at(R, m_up) - log_psi_at(R, n)
        contrib = contrib + (
            math.sqrt(m_up)
            * jnp.exp(log_ratio)
            * eps_dot_grad_log_psi(m_up)
        )
    if n - 1 >= 0:
        m_dn = n - 1
        log_ratio = log_psi_at(R, m_dn) - log_psi_at(R, n)
        contrib = contrib + (
            math.sqrt(n)
            * jnp.exp(log_ratio)
            * eps_dot_grad_log_psi(m_dn)
        )

    return 1j * g * contrib


# ---------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------

@pytest.fixture
def simple_system():
    """Tiny synthetic 2D system for unit tests."""
    N_el = 4
    dim = 2
    L = 10.0
    nph_max = 3
    eps = jnp.array([1.0, 0.0])

    # Three K vectors for multi-K testing.
    K_vectors = jnp.array(
        [[2.0 * jnp.pi / L,                 0.0],
         [0.0,                              2.0 * jnp.pi / L],
         [2.0 * jnp.pi / L, 2.0 * jnp.pi / L]],
        dtype=jnp.float64,
    )
    n_K = K_vectors.shape[0]

    phase_alpha = jnp.array([0.3, -0.2, 0.1], dtype=jnp.float64)
    coh_alpha = jnp.asarray(0.4, dtype=jnp.float64)

    def log_psi_e(R_):
        # Anisotropic Gaussian so ε·∇ ≠ 0 generically.
        return -0.03 * jnp.sum(R_ ** 2) - 0.02 * jnp.sum(
            R_[:, 0] * R_[:, 1]
        )

    g = 0.12
    return dict(
        N_el=N_el, dim=dim, L=L, nph_max=nph_max, n_K=n_K,
        eps=eps, K_vectors=K_vectors,
        phase_alpha=phase_alpha, coh_alpha=coh_alpha,
        log_psi_e=log_psi_e, g=g,
    )


def _sample_walker(seed, N_el, dim, L):
    rng = np.random.default_rng(seed)
    return jnp.asarray(
        rng.uniform(0.0, L, size=(N_el, dim)), dtype=jnp.float64,
    )


# ---------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------

@pytest.mark.parametrize("seed", [0, 1, 2, 3])
@pytest.mark.parametrize("n_walker", [0, 1, 2, 3])
def test_analytical_matches_finite_difference(
    simple_system, seed, n_walker,
):
    """E_loc_para closed-form vs central-difference ground truth."""
    s = simple_system
    R = _sample_walker(seed, s["N_el"], s["dim"], s["L"])

    ea = E_loc_para_analytical(
        R, n_walker, s["phase_alpha"], s["coh_alpha"], s["K_vectors"],
        s["eps"], s["log_psi_e"], s["g"], s["nph_max"],
    )
    en = E_loc_para_numerical(
        R, n_walker, s["phase_alpha"], s["coh_alpha"], s["K_vectors"],
        s["eps"], s["log_psi_e"], s["g"], s["nph_max"], h=1e-4,
    )
    assert jnp.allclose(ea, en, atol=1e-6), (
        f"seed={seed} n={n_walker}: analytical={ea}, "
        f"numerical={en}, diff={ea - en}"
    )


def test_zero_at_lambda_zero(simple_system):
    """g=0 → E_loc_para must be exactly zero, for any R, n."""
    s = simple_system
    for seed in [0, 1, 2]:
        R = _sample_walker(seed, s["N_el"], s["dim"], s["L"])
        for n_walker in range(s["nph_max"] + 1):
            ea = E_loc_para_analytical(
                R, n_walker, s["phase_alpha"], s["coh_alpha"],
                s["K_vectors"], s["eps"], s["log_psi_e"],
                g=0.0, nph_max=s["nph_max"],
            )
            assert ea == 0.0 + 0.0j, (
                f"g=0 but E_loc_para={ea} at seed={seed} n={n_walker}"
            )


def test_zero_real_part_when_phase_alpha_zero_at_vacuum(simple_system):
    """phase_α=0 ∀k AND walker at n=0:
        Re[E_loc_para] = 0 exactly (only Im piece via ε·∇log|ψ_e|).
    """
    s = simple_system
    phase_zero = jnp.zeros_like(s["phase_alpha"])
    for seed in [0, 1, 2, 3]:
        R = _sample_walker(seed, s["N_el"], s["dim"], s["L"])
        ea = E_loc_para_analytical(
            R, 0, phase_zero, s["coh_alpha"], s["K_vectors"],
            s["eps"], s["log_psi_e"], s["g"], s["nph_max"],
        )
        assert abs(float(jnp.real(ea))) < 1e-12, (
            f"seed={seed}: Re(E_loc_para) = {float(jnp.real(ea))}"
        )


def test_production_function_matches_analytical(simple_system):
    """The JIT'd production code path (called from kin_pot_*) must
    agree with the in-file analytical reference on a batch of
    walkers spanning all Fock states with non-zero phase_alpha and
    non-zero coh_alpha."""
    from OmegaQMC.qed_vmcopt_nn_heg_sr import _para_eloc_from_components

    s = simple_system
    eps_dot_K_vec = s["K_vectors"] @ s["eps"]

    n_walkers = 8
    walkers = jnp.stack([
        _sample_walker(seed, s["N_el"], s["dim"], s["L"])
        for seed in range(n_walkers)
    ])
    n_ph = jnp.array(
        [0, 1, 2, 3, 0, 1, 2, 3], dtype=jnp.int32,
    )

    # Compute ε·∇log|ψ_e| via jax.grad (matches what kin_and_eps_grad
    # does internally, sans the WF plumbing).
    grad_fn = jax.vmap(jax.grad(s["log_psi_e"]))
    eps_grad_w = jnp.einsum(
        "wid,d->w", grad_fn(walkers), s["eps"],
    )

    re_prod, im_prod = _para_eloc_from_components(
        walkers, eps_grad_w, n_ph,
        s["phase_alpha"], s["coh_alpha"],
        s["K_vectors"], eps_dot_K_vec, s["g"], s["nph_max"],
    )

    for w in range(n_walkers):
        ea = E_loc_para_analytical(
            walkers[w], int(n_ph[w]), s["phase_alpha"], s["coh_alpha"],
            s["K_vectors"], s["eps"], s["log_psi_e"], s["g"],
            s["nph_max"],
        )
        got_re = float(re_prod[w])
        got_im = float(im_prod[w])
        ref_re = float(jnp.real(ea))
        ref_im = float(jnp.imag(ea))
        assert abs(got_re - ref_re) < 1e-6, (
            f"walker {w} (n={int(n_ph[w])}): "
            f"Re prod={got_re} vs ref={ref_re}"
        )
        assert abs(got_im - ref_im) < 1e-6, (
            f"walker {w} (n={int(n_ph[w])}): "
            f"Im prod={got_im} vs ref={ref_im}"
        )


def test_production_function_zero_re_at_phase_zero(simple_system):
    """phase_alpha = 0 ∀ k, walker at n=0 → Re[E_loc_para] = 0 exactly
    from production code path."""
    from OmegaQMC.qed_vmcopt_nn_heg_sr import _para_eloc_from_components

    s = simple_system
    phase_zero = jnp.zeros_like(s["phase_alpha"])
    eps_dot_K_vec = s["K_vectors"] @ s["eps"]

    walkers = jnp.stack([
        _sample_walker(seed, s["N_el"], s["dim"], s["L"])
        for seed in range(6)
    ])
    # All walkers at n=0 (only m=n+1 channel active, gives pure Im).
    n_ph = jnp.zeros(6, dtype=jnp.int32)
    grad_fn = jax.vmap(jax.grad(s["log_psi_e"]))
    eps_grad_w = jnp.einsum(
        "wid,d->w", grad_fn(walkers), s["eps"],
    )

    re_prod, _ = _para_eloc_from_components(
        walkers, eps_grad_w, n_ph,
        phase_zero, s["coh_alpha"],
        s["K_vectors"], eps_dot_K_vec, s["g"], s["nph_max"],
    )
    for w in range(walkers.shape[0]):
        assert abs(float(re_prod[w])) < 1e-12, (
            f"walker {w}: Re(E_para)={float(re_prod[w])} (expected 0)"
        )


def test_single_K_recovers_phase4_form(simple_system):
    """With n_K=1, phase_alpha=[α], the new Option-C formula at n=0
    should match the Phase 4 form (i·g·α_coh·p) when α_1 = 0.

    Sanity: with phase_alpha=[0] and walker at n=0:
        E_loc_para = i·g·α_coh·(p + 0) = i·g·α_coh·p  → pure imaginary.
    """
    from OmegaQMC.qed_vmcopt_nn_heg_sr import _para_eloc_from_components

    s = simple_system
    K_single = s["K_vectors"][:1]                          # (1, dim)
    eps_dot_K_single = K_single @ s["eps"]
    phase_single = jnp.array([0.0], dtype=jnp.float64)

    walkers = jnp.stack([
        _sample_walker(seed, s["N_el"], s["dim"], s["L"])
        for seed in range(4)
    ])
    n_ph = jnp.zeros(4, dtype=jnp.int32)
    grad_fn = jax.vmap(jax.grad(s["log_psi_e"]))
    eps_grad_w = jnp.einsum("wid,d->w", grad_fn(walkers), s["eps"])

    re_prod, im_prod = _para_eloc_from_components(
        walkers, eps_grad_w, n_ph,
        phase_single, s["coh_alpha"],
        K_single, eps_dot_K_single, s["g"], s["nph_max"],
    )
    # Expected: Im(E_para) = g · α_coh · p_walker
    expected_im = s["g"] * float(s["coh_alpha"]) * eps_grad_w
    for w in range(walkers.shape[0]):
        assert abs(float(re_prod[w])) < 1e-12, (
            f"w={w}: Re should be 0, got {float(re_prod[w])}"
        )
        assert abs(float(im_prod[w]) - float(expected_im[w])) < 1e-10, (
            f"w={w}: Im prod={float(im_prod[w])}, "
            f"expected={float(expected_im[w])}"
        )
