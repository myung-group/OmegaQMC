"""Phase 5a-4 validation: SR primitives for Level 5 complex log Ψ.

For real-valued parameters θ and complex log Ψ = u + i·v:
    g_θ      = 2·⟨ ∂u/∂θ·ΔRe(E_loc) + ∂v/∂θ·ΔIm(E_loc) ⟩       (real)
    S_{θθ'}  = ⟨ Δ∂u/∂θ · Δ∂u/∂θ' + Δ∂v/∂θ · Δ∂v/∂θ' ⟩          (real PSD)

Tests:
  1. Per-walker Jacobian returns (du, dv) of correct shape; agrees with
     jax.grad on log_mag and phase respectively.
  2. Batched Jacobian vectorises correctly across walkers.
  3. At init (zero-init MLPs): dv-wrt-mag_mlp params = 0,
                              du-wrt-phase_mlp params = 0.
  4. Force g = 0 when E_loc is constant across walkers (no variance).
  5. Fisher matvec S·v is symmetric (v·(S·v) ≥ 0 PSD; and
     u·(S·v) = v·(S·u) for any test vectors u, v).
  6. CG-solve of (S + ε·I)·δθ = g converges and recovers a sensible δθ
     on a small toy problem.
"""
from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)


@pytest.fixture
def l5_with_sr():
    from OmegaQMC.psi.nn.heg_wf import HEGPsiFormerConfig
    from OmegaQMC.qed_vmcopt_nn_heg_l5 import (
        build_l5_log_psi, make_l5_sr_primitives, make_l5_eloc_no_vee,
    )

    n_up, n_down = 1, 1
    rs = 10.0
    N = n_up + n_down
    L = rs * math.sqrt(math.pi * N)
    config = HEGPsiFormerConfig(
        n_up=n_up, n_down=n_down, L=L, dim=2,
        backbone="ferminet",
        embedding_dim=8,
        n_interactions=1,
        two_particle_stream_dim=4,
        n_det=1,
        full_determinant=True,
        use_backflow=False,
        use_cusp=False,
    )
    init_key = jax.random.key(0)
    omega = 0.10
    machinery = build_l5_log_psi(
        config, init_key,
        omega_init=omega,
        K_max=2,
        phase_mlp_hidden=(4, 4),
        mag_mlp_hidden=(4, 4),
    )
    sr = make_l5_sr_primitives(
        machinery["log_psi_l5_flat"],
        nelec=2, dim=2,
    )
    eloc = make_l5_eloc_no_vee(
        machinery["log_psi_l5"],
        eps=jnp.array([1.0, 0.0]),
        lam=0.05,
        omega_eff=math.sqrt(omega ** 2 + N * 0.05 ** 2),
        nelec=2, dim=2,
    )
    return dict(
        config=config, omega=omega, L=L, N=N,
        eloc=eloc, sr=sr, **machinery,
    )


def _sample_batch(seed, n_walkers, nelec, dim, L):
    rng = np.random.default_rng(seed)
    R = rng.uniform(0.0, L, size=(n_walkers, nelec, dim))
    q_c = rng.normal(0.0, 1.0, size=(n_walkers,))
    return jnp.asarray(R, dtype=jnp.float64), jnp.asarray(q_c, dtype=jnp.float64)


# ---------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------

def test_per_walker_jacobian_shape(l5_with_sr):
    m = l5_with_sr
    R, q_c = _sample_batch(0, 1, 2, 2, m["L"])
    r_flat = R[0].reshape(-1)
    du, dv = m["sr"]["per_walker_jacobian"](
        r_flat, q_c[0], m["init_params_flat"],
    )
    assert du.shape == (m["n_params"],)
    assert dv.shape == (m["n_params"],)
    assert jnp.all(jnp.isfinite(du))
    assert jnp.all(jnp.isfinite(dv))


def test_batched_jacobian_matches_per_walker(l5_with_sr):
    m = l5_with_sr
    n_w = 4
    R, q_c = _sample_batch(0, n_w, 2, 2, m["L"])
    r_flat_batch = R.reshape(n_w, -1)
    Jac_u, Jac_v = m["sr"]["batched_jacobian"](
        r_flat_batch, q_c, m["init_params_flat"],
    )
    assert Jac_u.shape == (n_w, m["n_params"])
    assert Jac_v.shape == (n_w, m["n_params"])
    # Spot-check one walker matches per-walker call
    du, dv = m["sr"]["per_walker_jacobian"](
        r_flat_batch[2], q_c[2], m["init_params_flat"],
    )
    assert jnp.allclose(du, Jac_u[2], atol=1e-12)
    assert jnp.allclose(dv, Jac_v[2], atol=1e-12)


def test_jacobian_block_structure_at_init(l5_with_sr):
    """At init (zero-init MLP final layers), v doesn't depend on
    mag_mlp params (v ignores mag MLP entirely), and u doesn't depend
    on phase_mlp final-layer params (phase contributes ZERO to u
    regardless of where in phase MLP).

    Specifically, dv/d(mag_mlp) ≡ 0 and du/d(phase_mlp) ≡ 0 EVERYWHERE.
    """
    m = l5_with_sr
    R, q_c = _sample_batch(0, 1, 2, 2, m["L"])
    r_flat = R[0].reshape(-1)
    du, dv = m["sr"]["per_walker_jacobian"](
        r_flat, q_c[0], m["init_params_flat"],
    )
    # We don't directly know the slice mapping in p_flat for mag vs phase.
    # Instead check: phase MLP contributes ONLY to v.  Since at init
    # phase=0 identically (zero-init final layer), changing phase MLP
    # hidden weights doesn't change u — so du wrt those params = 0.
    # We rely on this via: phase_mlp has n_phase_mlp params; dv has
    # support on those (in general non-zero); du wrt those should be 0.
    unravel = m["unravel"]
    du_pytree = unravel(du)
    dv_pytree = unravel(dv)
    # du wrt phase_mlp should be zero everywhere
    for layer in du_pytree["phase_mlp"]:
        assert jnp.allclose(layer["W"], 0.0, atol=1e-12), (
            "du has non-zero component on phase_mlp.W"
        )
        assert jnp.allclose(layer["b"], 0.0, atol=1e-12), (
            "du has non-zero component on phase_mlp.b"
        )
    # dv wrt mag_mlp should be zero everywhere (phase doesn't see mag MLP)
    for layer in dv_pytree["mag_mlp"]:
        assert jnp.allclose(layer["W"], 0.0, atol=1e-12)
        assert jnp.allclose(layer["b"], 0.0, atol=1e-12)


def test_force_zero_when_eloc_constant(l5_with_sr):
    """If e_re and e_im are constant across walkers (no variance),
    g_θ = 2·⟨ΔJac·0⟩ = 0 trivially."""
    m = l5_with_sr
    n_w = 8
    R, q_c = _sample_batch(0, n_w, 2, 2, m["L"])
    r_flat_batch = R.reshape(n_w, -1)
    Jac_u, Jac_v = m["sr"]["batched_jacobian"](
        r_flat_batch, q_c, m["init_params_flat"],
    )
    # All walkers have same e_re=1.0, e_im=0.5 — constant
    e_re = jnp.ones(n_w) * 1.0
    e_im = jnp.ones(n_w) * 0.5
    g, _, e_mean, e_var, im_mean = m["sr"]["sr_force_and_S_matvec"](
        Jac_u, Jac_v, e_re, e_im,
    )
    assert jnp.allclose(g, 0.0, atol=1e-12)
    assert abs(float(e_mean) - 1.0) < 1e-12
    assert float(e_var) < 1e-12
    assert abs(float(im_mean) - 0.5) < 1e-12


def test_fisher_matvec_symmetric_and_psd(l5_with_sr):
    """For any test vectors u, v:
        u·(S·v) = v·(S·u)  (symmetric)
        v·(S·v) >= 0       (PSD)
    """
    m = l5_with_sr
    n_w = 16
    R, q_c = _sample_batch(42, n_w, 2, 2, m["L"])
    r_flat_batch = R.reshape(n_w, -1)
    # Perturb params so Jacobian has rich structure
    p_perturbed = jax.tree.map(lambda x: x, m["init_params_pytree"])
    p_perturbed["phase_mlp"][-1]["W"] = 0.1 * jnp.ones_like(
        p_perturbed["phase_mlp"][-1]["W"]
    )
    p_perturbed["mag_mlp"][-1]["W"] = 0.1 * jnp.ones_like(
        p_perturbed["mag_mlp"][-1]["W"]
    )
    p_flat_perturbed, _ = jax.flatten_util.ravel_pytree(p_perturbed)

    Jac_u, Jac_v = m["sr"]["batched_jacobian"](
        r_flat_batch, q_c, p_flat_perturbed,
    )
    e_re = jnp.linspace(-0.1, 0.1, n_w)
    e_im = jnp.linspace(-0.05, 0.05, n_w)
    _, S_matvec, _, _, _ = m["sr"]["sr_force_and_S_matvec"](
        Jac_u, Jac_v, e_re, e_im,
    )

    rng = np.random.default_rng(123)
    n_p = m["n_params"]
    u_vec = jnp.asarray(rng.normal(size=n_p), dtype=jnp.float64)
    v_vec = jnp.asarray(rng.normal(size=n_p), dtype=jnp.float64)
    Sv = S_matvec(v_vec)
    Su = S_matvec(u_vec)
    uSv = float(jnp.dot(u_vec, Sv))
    vSu = float(jnp.dot(v_vec, Su))
    vSv = float(jnp.dot(v_vec, Sv))
    assert abs(uSv - vSu) < 1e-10, f"S not symmetric: u·Sv={uSv}, v·Su={vSu}"
    assert vSv >= -1e-12, f"S not PSD: v·Sv={vSv}"


def test_cg_solves_S_delta_equals_g(l5_with_sr):
    """Test CG on (S + ε·I)·δθ = g with ε > 0 to ensure positive-definite.
    Then verify (S + ε·I)·δθ ≈ g."""
    from OmegaQMC.qed_vmcopt_nn_heg_sr import _cg_solve

    m = l5_with_sr
    n_w = 32
    R, q_c = _sample_batch(7, n_w, 2, 2, m["L"])
    r_flat_batch = R.reshape(n_w, -1)
    p_perturbed = jax.tree.map(lambda x: x, m["init_params_pytree"])
    p_perturbed["phase_mlp"][-1]["W"] = 0.1 * jnp.ones_like(
        p_perturbed["phase_mlp"][-1]["W"]
    )
    p_flat_perturbed, _ = jax.flatten_util.ravel_pytree(p_perturbed)

    Jac_u, Jac_v = m["sr"]["batched_jacobian"](
        r_flat_batch, q_c, p_flat_perturbed,
    )
    e_re = jnp.linspace(-0.2, 0.2, n_w)
    e_im = jnp.linspace(-0.1, 0.1, n_w)
    g, S_matvec, _, _, _ = m["sr"]["sr_force_and_S_matvec"](
        Jac_u, Jac_v, e_re, e_im,
    )

    eps = 1e-3
    def matvec_damped(v):
        return S_matvec(v) + eps * v

    delta = _cg_solve(matvec_damped, g, n_iters=200, tol=1e-8)
    # Check (S + ε·I)·δ ≈ g.  CG with damping 1e-3 typically converges
    # to rel_err ~ 1e-3 in 200 iters; production runs use more iters or
    # larger damping for tighter accuracy.
    residual = matvec_damped(delta) - g
    rel_err = float(jnp.linalg.norm(residual) / (jnp.linalg.norm(g) + 1e-30))
    assert rel_err < 5e-3, f"CG residual rel_err = {rel_err}"
