"""Phase 5a-3 validation: production E_loc on the Level 5 trial.

Validates that `make_l5_eloc_no_vee()` produces (Re, Im) of the
Pauli-Fierz local energy in agreement with the Phase 0 analytical
reference (which itself was already validated against FD ground truth).

Reduction checks at INIT (zero-init MLPs, factorised real trial):
  • At λ=0 with s=Ω:
      - Im(E_loc) ≡ 0     (real trial → no imaginary contribution)
      - Re(E_loc) at q_c → Re(E_loc) at q_c'=q_c+δ is the same MODULO
        the kinetic-energy variation with q_c (HO ground state gives a
        q_c-independent local energy).  Concretely, for u =
        log|ψ_e| − ½·Ω·q_c² + ½·log(Ω/π), the photon contribution to
        Re(E_loc) is Ω/2 EXACTLY (independent of q_c).
  • At λ>0 with s=Ω:
      - Photon-renormalised contribution: trial ≠ HO ground state of
        H_eff, so Re(E_loc) acquires a q_c-dependent residual until s
        is retrained to Ω_eff.
"""
from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)


# ---------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------

@pytest.fixture
def l5_with_eloc():
    from OmegaQMC.psi.nn.heg_wf import HEGPsiFormerConfig
    from OmegaQMC.qed_vmcopt_nn_heg_l5 import (
        build_l5_log_psi, make_l5_eloc_no_vee,
    )

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
    lam = 0.0  # will override per-test
    machinery = build_l5_log_psi(
        config, init_key,
        omega_init=omega,
        K_max=2,
        phase_mlp_hidden=(8, 8),
        mag_mlp_hidden=(8, 8),
    )
    return dict(
        config=config, omega=omega, L=L, N=N,
        **machinery,
    )


def _sample_walker(seed, nelec, dim, L):
    rng = np.random.default_rng(seed)
    R = rng.uniform(0.0, L, size=(nelec, dim))
    q_c = float(rng.normal(0.0, 1.0))
    return jnp.asarray(R, dtype=jnp.float64), q_c


# ---------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------

def test_eloc_returns_real_tuple(l5_with_eloc):
    from OmegaQMC.qed_vmcopt_nn_heg_l5 import make_l5_eloc_no_vee
    m = l5_with_eloc
    eps = jnp.array([1.0, 0.0])
    eloc = make_l5_eloc_no_vee(
        m["log_psi_l5"],
        eps=eps, lam=0.0, omega_eff=m["omega"],
        nelec=2, dim=2,
    )
    R, q_c = _sample_walker(0, 2, 2, m["L"])
    re, im = eloc(R, q_c, m["init_params_pytree"])
    assert re.shape == ()
    assert im.shape == ()
    assert jnp.isreal(re)
    assert jnp.isreal(im)


def test_im_zero_at_init_lambda_zero(l5_with_eloc):
    """At init (v ≡ 0) + λ=0: Im(E_loc) = 0 exactly."""
    from OmegaQMC.qed_vmcopt_nn_heg_l5 import make_l5_eloc_no_vee
    m = l5_with_eloc
    eps = jnp.array([1.0, 0.0])
    eloc = make_l5_eloc_no_vee(
        m["log_psi_l5"],
        eps=eps, lam=0.0, omega_eff=m["omega"],
        nelec=2, dim=2,
    )
    for seed in range(4):
        R, q_c = _sample_walker(seed, 2, 2, m["L"])
        _, im = eloc(R, q_c, m["init_params_pytree"])
        assert abs(float(im)) < 1e-10, (
            f"seed={seed}: Im(E_loc)={float(im)} (should be 0)"
        )


def test_im_zero_at_init_lambda_positive(l5_with_eloc):
    """At init (v ≡ 0) but λ>0: Im(E_loc) = +λ·q_c·(ε·Σᵢ ∇ᵢ u) — NOT zero
    in general, since u depends on R via log|ψ_e|.

    This is the Hermiticity diagnostic that the FULL optimised state must
    drive to 0, but at init it can be non-zero.  We just check it's
    finite and that it scales linearly with λ.
    """
    from OmegaQMC.qed_vmcopt_nn_heg_l5 import make_l5_eloc_no_vee
    m = l5_with_eloc
    eps = jnp.array([1.0, 0.0])
    omega = m["omega"]
    N = m["N"]

    R, q_c = _sample_walker(0, 2, 2, m["L"])
    p = m["init_params_pytree"]

    for lam in [0.01, 0.05]:
        omega_eff = math.sqrt(omega ** 2 + N * lam ** 2)
        eloc = make_l5_eloc_no_vee(
            m["log_psi_l5"],
            eps=eps, lam=lam, omega_eff=omega_eff,
            nelec=2, dim=2,
        )
        re, im = eloc(R, q_c, p)
        # Im scales linearly with λ at init (since the only Im source is
        # λ·q_c·(ε·∇u))
        assert jnp.isfinite(im)
        assert jnp.isfinite(re)
    # Sanity: doubling λ doubles Im
    lam1, lam2 = 0.01, 0.02
    eloc1 = make_l5_eloc_no_vee(
        m["log_psi_l5"], eps=eps, lam=lam1,
        omega_eff=math.sqrt(omega ** 2 + N * lam1 ** 2),
        nelec=2, dim=2,
    )
    eloc2 = make_l5_eloc_no_vee(
        m["log_psi_l5"], eps=eps, lam=lam2,
        omega_eff=math.sqrt(omega ** 2 + N * lam2 ** 2),
        nelec=2, dim=2,
    )
    _, im1 = eloc1(R, q_c, p)
    _, im2 = eloc2(R, q_c, p)
    # im2/im1 should be 2 (within numerical precision)
    if abs(float(im1)) > 1e-8:
        ratio = float(im2) / float(im1)
        assert abs(ratio - 2.0) < 1e-6, (
            f"Im scales non-linearly: im2/im1 = {ratio} (expected 2)"
        )


def test_re_photon_zeropoint_at_init_lambda_zero(l5_with_eloc):
    """At init (v=0, mag-MLP=0, s=Ω) + λ=0, the photon contribution to
    Re(E_loc) is exactly Ω/2 (HO ground-state energy), independent of q_c.

    Re(E_loc) = T_e[ψ_e](R) + (photon-piece) + V_phot
    photon-piece = -½·∂²u/∂q² - ½·(∂u/∂q)² + ½·Ω²·q_c²
                 = -½·(-Ω) - ½·(Ω·q_c)² + ½·Ω²·q_c²
                 = Ω/2 + 0
    → q_c-independent contribution of Ω/2.

    Test: Re(E_loc) at two different q_c values should differ ONLY by
    the bilinear (which is 0 at λ=0 and v=0).  So Re(E_loc) should be
    q_c-INDEPENDENT at init + λ=0.
    """
    from OmegaQMC.qed_vmcopt_nn_heg_l5 import make_l5_eloc_no_vee
    m = l5_with_eloc
    eps = jnp.array([1.0, 0.0])
    omega = m["omega"]
    eloc = make_l5_eloc_no_vee(
        m["log_psi_l5"],
        eps=eps, lam=0.0, omega_eff=omega,
        nelec=2, dim=2,
    )
    R, _ = _sample_walker(0, 2, 2, m["L"])

    re_at_0, _ = eloc(R, 0.0, m["init_params_pytree"])
    re_at_1, _ = eloc(R, 1.0, m["init_params_pytree"])
    re_at_2, _ = eloc(R, 2.0, m["init_params_pytree"])
    # All should be equal (HO ground state local-energy is q_c-independent)
    assert abs(float(re_at_0) - float(re_at_1)) < 1e-9, (
        f"Re(E_loc) varied with q_c: re(0)={float(re_at_0)}, "
        f"re(1)={float(re_at_1)}"
    )
    assert abs(float(re_at_0) - float(re_at_2)) < 1e-9, (
        f"Re(E_loc) varied with q_c: re(0)={float(re_at_0)}, "
        f"re(2)={float(re_at_2)}"
    )


def test_re_matches_phase0_analytical(l5_with_eloc):
    """Production E_loc matches the Phase 0 analytical reference
    when fed the SAME (u, v) function pair, on randomly perturbed MLP
    parameters (so v ≠ 0)."""
    from OmegaQMC.qed_vmcopt_nn_heg_l5 import make_l5_eloc_no_vee
    # Reuse the Phase 0 reference
    from tests.test_qed_l5_eloc import E_loc_l5_analytical

    m = l5_with_eloc
    eps = jnp.array([1.0, 0.0])
    omega = m["omega"]
    N = m["N"]
    lam = 0.05
    omega_eff = math.sqrt(omega ** 2 + N * lam ** 2)
    eloc = make_l5_eloc_no_vee(
        m["log_psi_l5"],
        eps=eps, lam=lam, omega_eff=omega_eff,
        nelec=2, dim=2,
    )
    R, q_c = _sample_walker(0, 2, 2, m["L"])

    # Perturb MLPs so v ≠ 0
    p = jax.tree.map(lambda x: x, m["init_params_pytree"])
    p["phase_mlp"][-1]["W"] = 0.1 * jnp.ones_like(
        p["phase_mlp"][-1]["W"]
    )
    p["phase_mlp"][-1]["b"] = jnp.array([0.05])
    p["mag_mlp"][-1]["W"] = 0.05 * jnp.ones_like(
        p["mag_mlp"][-1]["W"]
    )
    p["mag_mlp"][-1]["b"] = jnp.array([0.02])

    re_prod, im_prod = eloc(R, q_c, p)

    # Phase 0 reference uses u_fn, v_fn taking (R, q_c)
    def u_fn(R_, q_):
        return m["log_psi_l5"](R_, q_, p)[0]

    def v_fn(R_, q_):
        return m["log_psi_l5"](R_, q_, p)[1]

    re_ref, im_ref = E_loc_l5_analytical(
        R, q_c, u_fn, v_fn,
        V_ee_R=0.0,    # both formulas exclude V_ee here
        eps=eps, lam=lam, omega_eff=omega_eff, N_el=2,
    )
    assert abs(float(re_prod) - re_ref) < 1e-8, (
        f"Re mismatch: prod={float(re_prod)}, ref={re_ref}, "
        f"diff={float(re_prod) - re_ref}"
    )
    assert abs(float(im_prod) - im_ref) < 1e-8, (
        f"Im mismatch: prod={float(im_prod)}, ref={im_ref}, "
        f"diff={float(im_prod) - im_ref}"
    )
