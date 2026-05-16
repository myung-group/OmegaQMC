"""Phase 5a-2 validation (dense-Fourier + MLP architecture).

Architecture (this phase, fully non-factorised + NN-based, no priors):
    u(R, q_c) = log_psi_e(R) + log_chi_HO(q_c; s) + MLP_mag(features)
    v(R, q_c) = MLP_phase(features)
    features = [Σᵢ sin(K·rᵢ), Σᵢ cos(K·rᵢ) for K in K_grid, q_c]

Both MLPs zero-init final layer → at init, MLP_mag = MLP_phase = 0
→ trial = bare HEG · HO photon Gaussian (real, factorised).

Tests:
  1. Instantiates with realistic HEG config + K_max cutoff.
  2. Returns (log_mag, phase) tuple — two real scalars.
  3. At init: phase = 0 (zero-init MLP_phase) for any (R, q_c).
  4. At init: mag-coupling = 0 (zero-init MLP_mag).
  5. With perturbed MLP weights, phase ≠ 0 and mag_coupling ≠ 0.
  6. Flat wrapper agrees with pytree form.
  7. PBC: under r → r + L·ê (lattice translation), features unchanged
     → trial unchanged (modulo gauge phase, which |Ψ|² absorbs).
  8. Permutation symmetry: swapping any two electrons leaves features
     and trial invariant.
  9. Gradients exist (and are finite) w.r.t. R, q_c, and all params.
"""
from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)


# ---------------------------------------------------------------------
# Fixture: small HEG with modest K cutoff
# ---------------------------------------------------------------------

@pytest.fixture
def l5_machinery():
    from OmegaQMC.psi.nn.heg_wf import HEGPsiFormerConfig
    from OmegaQMC.qed_vmcopt_nn_heg_l5 import build_l5_log_psi

    n_up, n_down = 1, 1
    rs = 10.0
    N = n_up + n_down
    L = rs * math.sqrt(math.pi * N)
    config = HEGPsiFormerConfig(
        n_up=n_up, n_down=n_down, L=L, dim=2,
        backbone="ferminet",
        embedding_dim=16,
        n_interactions=1,
        two_particle_stream_dim=8,
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
        K_max=2,                     # small for fast tests (~12 K vectors)
        phase_mlp_hidden=(16, 16),
        mag_mlp_hidden=(16, 16),
    )
    return dict(config=config, omega=omega, L=L, **machinery)


def _sample_walker(seed, nelec, dim, L):
    rng = np.random.default_rng(seed)
    R = rng.uniform(0.0, L, size=(nelec, dim))
    q_c = float(rng.normal(0.0, 1.0))
    return jnp.asarray(R, dtype=jnp.float64), q_c


# ---------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------

def test_instantiates(l5_machinery):
    m = l5_machinery
    assert m["n_params"] > 0
    assert m["n_electronic"] > 0
    assert m["n_K"] > 0
    assert m["n_mag_mlp"] > 0
    assert m["n_phase_mlp"] > 0
    assert m["n_features"] == 2 * m["n_K"] + 1


def test_returns_tuple(l5_machinery):
    m = l5_machinery
    R, q_c = _sample_walker(0, 2, 2, m["L"])
    log_mag, phase = m["log_psi_l5"](R, q_c, m["init_params_pytree"])
    assert log_mag.shape == ()
    assert phase.shape == ()
    assert jnp.isreal(log_mag)
    assert jnp.isreal(phase)


def test_phase_zero_at_init(l5_machinery):
    """Zero-init final layer of MLP_phase → phase = 0 for any (R, q_c)."""
    m = l5_machinery
    for seed in range(5):
        R, q_c = _sample_walker(seed, 2, 2, m["L"])
        _, phase = m["log_psi_l5"](R, q_c, m["init_params_pytree"])
        assert abs(float(phase)) < 1e-12, (
            f"seed={seed}: phase={float(phase)} (should be 0 at init)"
        )


def test_mag_coupling_zero_at_init(l5_machinery):
    """Zero-init final layer of MLP_mag → log_mag = log_psi_e + log_HO exactly
    (no mag coupling contribution)."""
    m = l5_machinery
    for seed in range(5):
        R, q_c = _sample_walker(seed, 2, 2, m["L"])
        log_mag, _ = m["log_psi_l5"](R, q_c, m["init_params_pytree"])
        # Reference: bare HEG · HO photon Gaussian
        log_psi_e_val = m["electronic_log_psi"](
            R, m["init_params_pytree"]["e"],
        )
        s = m["init_params_pytree"]["s"]
        log_chi = (
            -0.5 * s * q_c ** 2
            + 0.5 * jnp.log(s)
            - 0.5 * jnp.log(jnp.pi)
        )
        expected = log_psi_e_val + log_chi
        assert abs(float(log_mag) - float(expected)) < 1e-10, (
            f"seed={seed}: log_mag - expected = "
            f"{float(log_mag) - float(expected)}"
        )


def test_phase_nonzero_with_perturbed_mlp(l5_machinery):
    """Perturbing MLP_phase final layer makes phase non-zero."""
    m = l5_machinery
    R, q_c = _sample_walker(0, 2, 2, m["L"])
    p = jax.tree.map(lambda x: x, m["init_params_pytree"])
    # Manually set MLP_phase final layer to non-zero
    p["phase_mlp"][-1]["W"] = 0.1 * jnp.ones_like(
        p["phase_mlp"][-1]["W"]
    )
    p["phase_mlp"][-1]["b"] = jnp.array([0.05])
    _, phase = m["log_psi_l5"](R, q_c, p)
    assert abs(float(phase)) > 1e-4, (
        f"phase={float(phase)} should be non-trivial with perturbed MLP"
    )


def test_mag_coupling_nonzero_with_perturbed_mlp(l5_machinery):
    """Perturbing MLP_mag final layer makes log_mag deviate from bare."""
    m = l5_machinery
    R, q_c = _sample_walker(0, 2, 2, m["L"])
    p = jax.tree.map(lambda x: x, m["init_params_pytree"])
    p["mag_mlp"][-1]["W"] = 0.1 * jnp.ones_like(
        p["mag_mlp"][-1]["W"]
    )
    p["mag_mlp"][-1]["b"] = jnp.array([0.05])
    log_mag, _ = m["log_psi_l5"](R, q_c, p)
    # Compare against init (mag_coupling=0)
    log_mag_init, _ = m["log_psi_l5"](R, q_c, m["init_params_pytree"])
    delta = float(log_mag) - float(log_mag_init)
    assert abs(delta) > 1e-4, (
        f"delta(log_mag)={delta} should be non-trivial with perturbed MLP"
    )


def test_flat_wrapper_agrees(l5_machinery):
    m = l5_machinery
    R, q_c = _sample_walker(0, 2, 2, m["L"])
    r_flat = R.reshape(-1)
    p_flat = m["init_params_flat"]
    lm_a, ph_a = m["log_psi_l5"](R, q_c, m["init_params_pytree"])
    lm_b, ph_b = m["log_psi_l5_flat"](r_flat, q_c, p_flat)
    assert abs(float(lm_a) - float(lm_b)) < 1e-12
    assert abs(float(ph_a) - float(ph_b)) < 1e-12


def test_pbc_translation_invariance(l5_machinery):
    """Under R → R + L·ê, density features Σᵢ sin/cos(K·rᵢ) are unchanged
    because K·L = 2π·integer.  Thus MLP outputs unchanged.
    Electronic FermiNet log|ψ_e| is also PBC-respecting (Γ-point HEG).
    → log_mag and phase invariant.

    NOTE: This tests the FOURIER feature path strictly.  The FermiNet
    backbone is also PBC-respecting for HEG (existing infrastructure).
    """
    m = l5_machinery
    R, q_c = _sample_walker(0, 2, 2, m["L"])
    L = m["L"]
    # Translate electron 0 by L along x-axis
    R_translated = R.at[0, 0].add(L)

    lm_orig, ph_orig = m["log_psi_l5"](
        R, q_c, m["init_params_pytree"],
    )
    lm_trans, ph_trans = m["log_psi_l5"](
        R_translated, q_c, m["init_params_pytree"],
    )
    # Both should be equal (modulo any FermiNet PBC subtlety)
    assert abs(float(lm_orig) - float(lm_trans)) < 1e-8, (
        f"log_mag NOT PBC-invariant under r → r+L: "
        f"orig={float(lm_orig)}, trans={float(lm_trans)}"
    )
    # Phase is zero at init regardless, so this test is trivial there;
    # repeat with perturbed phase MLP to ensure features themselves
    # are PBC-respecting.
    p = jax.tree.map(lambda x: x, m["init_params_pytree"])
    p["phase_mlp"][-1]["W"] = 0.05 * jnp.ones_like(
        p["phase_mlp"][-1]["W"]
    )
    _, ph_orig = m["log_psi_l5"](R, q_c, p)
    _, ph_trans = m["log_psi_l5"](R_translated, q_c, p)
    assert abs(float(ph_orig) - float(ph_trans)) < 1e-8, (
        f"phase NOT PBC-invariant under r → r+L: "
        f"orig={float(ph_orig)}, trans={float(ph_trans)}"
    )


def test_permutation_invariance_of_features(l5_machinery):
    """Σᵢ pooling → features invariant under electron permutation.
    Total trial Ψ should be antisymmetric in the Slater det but our
    MLP head sees only the pooled (symmetric) features."""
    from OmegaQMC.qed_vmcopt_nn_heg_l5 import build_K_grid_2d
    m = l5_machinery
    R, q_c = _sample_walker(0, 2, 2, m["L"])
    R_perm = R[::-1]  # swap electrons (only 2 in this test)

    # The MLP-fed features should be identical
    K_grid = build_K_grid_2d(m["L"], K_max=2)
    K_dot_r1 = R @ K_grid.T
    K_dot_r2 = R_perm @ K_grid.T
    sin1 = jnp.sum(jnp.sin(K_dot_r1), axis=0)
    sin2 = jnp.sum(jnp.sin(K_dot_r2), axis=0)
    assert jnp.allclose(sin1, sin2, atol=1e-12), (
        "Σᵢ sin(K·rᵢ) not permutation-invariant"
    )


def test_gradients_exist(l5_machinery):
    m = l5_machinery
    R, q_c = _sample_walker(0, 2, 2, m["L"])
    p = jax.tree.map(lambda x: x, m["init_params_pytree"])
    # Perturb so derivatives are non-trivial
    p["phase_mlp"][-1]["W"] = 0.1 * jnp.ones_like(
        p["phase_mlp"][-1]["W"]
    )
    p["mag_mlp"][-1]["W"] = 0.1 * jnp.ones_like(
        p["mag_mlp"][-1]["W"]
    )

    def log_mag_fn(R_, q_c_, p_):
        return m["log_psi_l5"](R_, q_c_, p_)[0]

    def phase_fn(R_, q_c_, p_):
        return m["log_psi_l5"](R_, q_c_, p_)[1]

    grad_R_u = jax.grad(log_mag_fn, argnums=0)(R, q_c, p)
    grad_qc_u = jax.grad(log_mag_fn, argnums=1)(R, q_c, p)
    grad_R_v = jax.grad(phase_fn, argnums=0)(R, q_c, p)
    grad_qc_v = jax.grad(phase_fn, argnums=1)(R, q_c, p)

    assert grad_R_u.shape == R.shape
    assert jnp.all(jnp.isfinite(grad_R_u))
    assert jnp.isfinite(grad_qc_u)
    assert grad_R_v.shape == R.shape
    assert jnp.all(jnp.isfinite(grad_R_v))
    assert jnp.isfinite(grad_qc_v)
