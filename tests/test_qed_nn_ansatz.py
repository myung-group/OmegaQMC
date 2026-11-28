"""QED neural-network ansatzes.

Pauli-Fierz primitives, the coherent-state and n-aware adapters, and
the Tang-native joint electron-photon ansatz, whose log psi, local
energy and gradients must be finite.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import unittest.mock as mock
from flax import nnx
import dataclasses

jax.config.update('jax_enable_x64', True)

from OmegaQMC.psi.nn.qed_physics import (  # noqa: E402
    real_space_dipole_electronic,
    real_space_dipole_total,
    coulomb_ee,
    coulomb_en,
    coulomb_nn,
    coherent_state_log_amplitude,
    pauli_fierz_local_energy,
    pauli_fierz_local_energy_signed,
)
from OmegaQMC.psi.nn.qed_adapter import (  # noqa: E402
    make_qed_nn_log_psi,
    analytical_alpha_perturbative,
    make_qed_nn_log_psi_n_aware,
    FockHead,
    make_qed_nn_log_psi_tang_native,
)
from OmegaQMC.utils import Mole_custom  # noqa: E402
from OmegaQMC.psi.nn.config import load_nn_config  # noqa: E402


# ---------------------------------------------------------------------------
# From test_qed_physics.py
#
# Unit tests for OmegaQMC.psi.nn.qed_physics primitives.
#
# Tests the Pauli-Fierz local-energy components against analytical
# ground-truth values on toy configurations:
#   1. Real-space dipole operator (electronic + total).
#   2. Coulomb energy components (e-e, e-n, n-n).
#   3. Coherent-state log amplitude.
#   4. Pauli-Fierz local energy on a Gaussian trial wavefunction.
#   5. Fock-ladder coupling at n=0 (single up branch) and n>0 (both branches).
# ---------------------------------------------------------------------------


@pytest.fixture
def h2_config():
    """Symmetric H2 config (electron and nuclear positions in Bohr)."""
    nuc = jnp.array([[0., 0., -0.7], [0., 0., 0.7]])
    charges = jnp.array([1., 1.])
    elec_sym = jnp.array([[0., 0., -0.5], [0., 0., 0.5]])
    return nuc, charges, elec_sym


def test_dipole_electronic_symmetric(h2_config):
    """Electronic dipole vanishes for centrosymmetric H2 config."""
    _, _, elec = h2_config
    d = real_space_dipole_electronic(elec)
    np.testing.assert_allclose(d, [0., 0., 0.], atol=1e-12)


def test_dipole_total_neutral_symmetric(h2_config):
    """Total dipole vanishes for neutral symmetric system."""
    nuc, charges, elec = h2_config
    d = real_space_dipole_total(elec, nuc, charges)
    np.testing.assert_allclose(d, [0., 0., 0.], atol=1e-12)


def test_coulomb_ee_known(h2_config):
    """E-E Coulomb for two electrons at z=-0.5, z=0.5 (separation 1.0)."""
    _, _, elec = h2_config
    e_ee = coulomb_ee(elec)
    np.testing.assert_allclose(float(e_ee), 1.0, atol=1e-10)


def test_coulomb_en_known(h2_config):
    """E-N Coulomb on H2 with electrons at z=±0.5, nuclei at z=±0.7."""
    nuc, charges, elec = h2_config
    e_en = coulomb_en(elec, nuc, charges)
    # |-0.5 - (-0.7)| = 0.2; |-0.5 - 0.7| = 1.2; etc.
    expected = -(1/0.2 + 1/1.2 + 1/1.2 + 1/0.2)
    np.testing.assert_allclose(float(e_en), expected, atol=1e-8)


def test_coulomb_nn_known(h2_config):
    """N-N Coulomb on H2 (Z=1, Z=1, separation 1.4 Bohr)."""
    nuc, charges, _ = h2_config
    e_nn = coulomb_nn(nuc, charges)
    np.testing.assert_allclose(float(e_nn), 1.0/1.4, atol=1e-8)


def test_coherent_state_amplitude_alpha_zero():
    """At alpha=0: <0|0> = 1 (log = 0); <n|0> = 0 (log → -inf)."""
    log_a = coherent_state_log_amplitude(0, jnp.array(0.0))
    np.testing.assert_allclose(float(log_a), 0.0, atol=1e-10)


def test_coherent_state_amplitude_known():
    """Verify <n=2 | alpha=1.0> = e^{-0.5} * 1 / sqrt(2!) = e^{-0.5}/sqrt(2).
    """
    log_a = coherent_state_log_amplitude(2, jnp.array(1.0))
    expected = -0.5 + 0.0 - 0.5 * np.log(2.0)  # -0.5 + log(1) - 0.5 ln(2)
    np.testing.assert_allclose(float(log_a), expected, atol=1e-10)


def test_pauli_fierz_local_energy_gaussian_lambda_zero(h2_config):
    """Local energy on Gaussian trial Psi at lambda=0.

    For log Psi(r) = -0.5 r^2 (independent of n):
      E_K = -0.5 * (lap + |grad|^2)
          = -0.5 * (-3*n_elec + sum r^2)
          = -0.5 * (-6 + 0.5) = 2.75
      Total = E_K + E_ee + E_en + E_nn (other terms zero).
    """
    nuc, charges, elec = h2_config

    def gauss_log_psi(r, R, n, params):
        return -0.5 * jnp.sum(r ** 2)

    e_loc = pauli_fierz_local_energy(
        gauss_log_psi, None, elec, jnp.array(0), nuc, charges,
        omega=0.5, coupling_vec=jnp.array([0., 0., 0.0]),
        nph_max=5, enuc=None,
    )
    # E_K = 2.75, E_ee = 1.0, E_en = -11.6667, E_nn = 0.71428
    expected = 2.75 + 1.0 - (1/0.2 + 1/1.2 + 1/1.2 + 1/0.2) + 1/1.4
    np.testing.assert_allclose(float(e_loc), expected, atol=1e-6)


def test_pauli_fierz_bilinear_coupling_n_zero():
    """Bilinear ladder at n=0: only +1 branch contributes (sqrt(0)*term=0)."""
    nuc = jnp.array([[0., 0., -0.7], [0., 0., 0.7]])
    charges = jnp.array([1., 1.])
    elec = jnp.array(
        [[0.0, 0.0, -0.6], [0.0, 0.0, 0.4]]
    )  # asymmetric -> nonzero dipole z

    def n_dep_log_psi(r, R, n, params):
        # Psi = electron Gaussian * coherent-state envelope at alpha=0.3
        alpha = 0.3
        n_f = jnp.asarray(n, dtype=jnp.float64)
        log_chi = (
            -0.5 * alpha**2
            + n_f * jnp.log(alpha)
            - 0.5 * jax.lax.lgamma(n_f + 1.0)
        )
        return -0.5 * jnp.sum(r**2) + log_chi

    omega, lam = 0.5, 0.1
    e_loc = pauli_fierz_local_energy(
        n_dep_log_psi, None, elec, jnp.array(0), nuc, charges,
        omega=omega, coupling_vec=jnp.array([0., 0., lam]),
        nph_max=5, enuc=None,
    )
    # Manual breakdown:
    #   Electronic dipole z = -(-0.6 + 0.4) = 0.2  →  eps_dot_d = 0.2
    # E_K (Gauss): grad=−r ⇒ |grad|^2 = 0.36+0.16 = 0.52, lap = -6 → E_K = 2.74
    #   E_ee = 1/1.0 = 1.0
    #   E_en = -(1/0.1 + 1/1.3 + 1/1.1 + 1/0.3)
    #   E_nn = 1/1.4
    #   E_ph = 0
    #   Bilinear (n=0): sqrt(omega/2)*lam*eps_dot_d * sqrt(1)*Psi(1)/Psi(0)
    #     ratio = chi(1)/chi(0) = alpha = 0.3
    #     = sqrt(0.25) * 0.1 * 0.2 * 0.3 = 0.5 * 0.1 * 0.2 * 0.3 = 0.003
    #   DSE = 0.5 * 0.01 * 0.04 = 0.0002
    e_K = 2.74
    e_ee = 1.0
    e_en = -(1/0.1 + 1/1.3 + 1/1.1 + 1/0.3)
    e_nn = 1/1.4
    # Bilinear has Tang/standard convention sign: H_bilin =
    # +√(ω/2)·λ·(ε·∑rᵢ)·(b+b†).
    # In our local-energy estimator this enters as −√(ω/2)·λ·(ε·d_e)·... since
    # ε·d_e = -ε·∑rᵢ. Hence the leading minus on e_bilin below.
    e_bilin = -np.sqrt(omega/2) * lam * 0.2 * 0.3
    e_dse = 0.5 * lam**2 * 0.2**2
    expected = e_K + e_ee + e_en + e_nn + 0.0 + e_bilin + e_dse
    np.testing.assert_allclose(float(e_loc), expected, atol=1e-4)


def test_pauli_fierz_bilinear_coupling_n_two():
    """At n=2 both up (sqrt(3)) and down (sqrt(2)) branches are active."""
    nuc = jnp.array([[0., 0., -0.7], [0., 0., 0.7]])
    charges = jnp.array([1., 1.])
    elec = jnp.array([[0., 0., -0.6], [0., 0., 0.4]])

    alpha_cs = 0.3

    def n_dep_log_psi(r, R, n, params):
        n_f = jnp.asarray(n, dtype=jnp.float64)
        log_chi = (
            -0.5 * alpha_cs**2
            + n_f * jnp.log(alpha_cs)
            - 0.5 * jax.lax.lgamma(n_f + 1.0)
        )
        return -0.5 * jnp.sum(r**2) + log_chi

    omega, lam = 0.5, 0.1
    e_loc = pauli_fierz_local_energy(
        n_dep_log_psi, None, elec, jnp.array(2), nuc, charges,
        omega=omega, coupling_vec=jnp.array([0., 0., lam]),
        nph_max=5, enuc=None,
    )
    # ratio_up = chi(3)/chi(2) = alpha/sqrt(3) = 0.3/sqrt(3)
    # ratio_dn = chi(1)/chi(2) = sqrt(2)/alpha = sqrt(2)/0.3
    ratio_up = alpha_cs / np.sqrt(3.0)
    ratio_dn = np.sqrt(2.0) / alpha_cs
    bilin_factor = np.sqrt(3.0) * ratio_up + np.sqrt(2.0) * ratio_dn
    eps_dot_d = 0.2

    e_K = 2.74
    e_ee = 1.0
    e_en = -(1/0.1 + 1/1.3 + 1/1.1 + 1/0.3)
    e_nn = 1/1.4
    e_ph = omega * 2
    # Bilinear has Tang/standard convention sign (see other test for context).
    e_bilin = -np.sqrt(omega/2) * lam * eps_dot_d * bilin_factor
    e_dse = 0.5 * lam**2 * eps_dot_d**2
    expected = e_K + e_ee + e_en + e_nn + e_ph + e_bilin + e_dse
    np.testing.assert_allclose(float(e_loc), expected, atol=1e-4)


def test_pauli_fierz_at_truncation_boundary():
    """At n = nph_max: up branch must be zeroed (no |n+1> state available)."""
    nuc = jnp.array([[0., 0., -0.7], [0., 0., 0.7]])
    charges = jnp.array([1., 1.])
    elec = jnp.array([[0., 0., -0.6], [0., 0., 0.4]])

    def constant_n_dep_log_psi(r, R, n, params):
        # Constant in n so any ratio = 1; both branches contribute their sqrt
        # factors
        return -0.5 * jnp.sum(r**2)

    nph_max = 3
    # At n = nph_max = 3: up term is zeroed (n+1=4 > 3)
    omega, lam = 0.5, 0.1
    e_loc_truncated = pauli_fierz_local_energy(
        constant_n_dep_log_psi, None, elec, jnp.array(nph_max),
        nuc, charges,
        omega=omega, coupling_vec=jnp.array([0., 0., lam]),
        nph_max=nph_max, enuc=None,
    )
    # Bilinear should only have the down-branch sqrt(n) = sqrt(3) contribution
    bilin_factor = np.sqrt(float(nph_max))  # only down branch
    eps_dot_d = 0.2
    expected_bilin = -np.sqrt(omega/2) * lam * eps_dot_d * bilin_factor
    # Test indirectly: compare to nph_max=10 (no truncation) result
    e_loc_full = pauli_fierz_local_energy(
        constant_n_dep_log_psi, None, elec, jnp.array(nph_max),
        nuc, charges,
        omega=omega, coupling_vec=jnp.array([0., 0., lam]),
        nph_max=10, enuc=None,
    )
    # Difference should be exactly the up-branch contribution that got removed
    diff = float(e_loc_full) - float(e_loc_truncated)
    expected_up_branch = (
        -np.sqrt(omega / 2) * lam * eps_dot_d * np.sqrt(float(nph_max + 1))
    )
    np.testing.assert_allclose(diff, expected_up_branch, atol=1e-6)


# ---------------------------------------------------------------------------
# From test_qed_adapter.py
#
# Tests for the coherent-state-shifted QED-NN ansatz adapter.
#
# We focus on adapter logic and analytical invariants rather than end-to-end
# PsiFormer integration (which needs full NN-VMC compute and is exercised by
# the QED-VMC driver tests in test_qed_vmc.py).
#
# Key invariant tested
# --------------------
# For the factorized form Ψ(r, n) = Ψ_e(r) ⟨n|α⟩, the Fock-ladder ratio that
# appears in the bilinear Pauli-Fierz coupling reduces to a closed-form:
#
#     Ψ(r, n+1)/Ψ(r, n)  =  ⟨n+1|α⟩/⟨n|α⟩  =  α / sqrt(n+1)
#     Ψ(r, n-1)/Ψ(r, n)  =  ⟨n-1|α⟩/⟨n|α⟩  =  sqrt(n) / α  (n ≥ 1)
#
# These are independent of r and serve as an exact invariant for testing.
# ---------------------------------------------------------------------------


def _stub_nn_factory(mock_log_psi):
    """Patch make_nn_log_psi inside qed_adapter to return a known stub.

    ``make_nn_log_psi`` returns ``(log_psi, params, graphdef,
    lap_grad)``; the adapter ignores ``lap_grad``.
    """
    return mock.patch(
        "OmegaQMC.psi.nn.qed_adapter.make_nn_log_psi",
        return_value=(
            mock_log_psi,
            {"weight": jnp.array(1.5)},
            "fake_graphdef",
            None,
        ),
    )


def test_adapter_returns_callable_and_combined_params():
    """make_qed_nn_log_psi should expose log_psi callable + alpha-augmented
    params.
    """

    def mock_elec_log_psi(r, R, params):
        return -0.5 * jnp.sum(r ** 2)

    with _stub_nn_factory(mock_elec_log_psi):
        log_psi, params, graphdef = make_qed_nn_log_psi(
            config=None, mol_info=None, rng_key=jax.random.key(0),
            omega=0.5, coupling_vec=jnp.array([0., 0., 0.05]),
            alpha_init=0.3, alpha_train=True,
        )

    assert callable(log_psi)
    assert isinstance(params, dict)
    assert "nn" in params and "alpha" in params
    assert float(params["alpha"]) == pytest.approx(0.3)
    assert graphdef == "fake_graphdef"


def test_adapter_alpha_train_false_omits_alpha_from_params():
    """When alpha_train=False, params dict should NOT contain 'alpha'."""

    def mock_elec_log_psi(r, R, params):
        return jnp.array(0.0)

    with _stub_nn_factory(mock_elec_log_psi):
        _, params, _ = make_qed_nn_log_psi(
            config=None, mol_info=None, rng_key=jax.random.key(0),
            omega=0.5, coupling_vec=jnp.array([0., 0., 0.05]),
            alpha_init=0.3, alpha_train=False,
        )
    assert "alpha" not in params


def test_factorized_ansatz_log_value_decomposition():
    """log Ψ(r, n) = log Ψ_e(r) + log ⟨n|α⟩."""

    def mock_elec_log_psi(r, R, params):
        return -0.5 * jnp.sum(r ** 2) + jnp.array(2.71)

    with _stub_nn_factory(mock_elec_log_psi):
        log_psi, params, _ = make_qed_nn_log_psi(
            config=None, mol_info=None, rng_key=jax.random.key(0),
            omega=0.5, coupling_vec=jnp.array([0., 0., 0.1]),
            alpha_init=0.3,
        )

    r = jnp.array([[0., 0., 0.1], [0., 0., -0.2]])
    R = jnp.array([[0., 0., 0.0]])
    n = jnp.array(2)
    val = float(log_psi(r, R, n, params))
    expected_elec = -0.5 * (0.01 + 0.04) + 2.71
    expected_chi = float(coherent_state_log_amplitude(2, jnp.array(0.3)))
    np.testing.assert_allclose(val, expected_elec + expected_chi, atol=1e-12)


def test_fock_ladder_ratio_invariant():
    """Ψ(r, n+1)/Ψ(r, n) = α / sqrt(n+1) regardless of r — analytical
    invariant.
    """

    def mock_elec_log_psi(r, R, params):
        # Pick something nontrivial in r so we'd notice if it leaked into n.
        return jnp.tanh(jnp.sum(r))

    with _stub_nn_factory(mock_elec_log_psi):
        log_psi, params, _ = make_qed_nn_log_psi(
            config=None, mol_info=None, rng_key=jax.random.key(0),
            omega=0.5, coupling_vec=jnp.array([0., 0., 0.1]),
            alpha_init=0.4,
        )

    r = jnp.array([[0.1, 0.2, 0.3], [-0.4, 0.5, -0.6]])
    R = jnp.array([[0., 0., 0.0]])
    alpha = 0.4

    for n_val in [0, 1, 2, 3, 5]:
        log_n = log_psi(r, R, jnp.array(n_val), params)
        log_np1 = log_psi(r, R, jnp.array(n_val + 1), params)
        ratio = float(jnp.exp(log_np1 - log_n))
        expected = alpha / np.sqrt(n_val + 1)
        np.testing.assert_allclose(
            ratio, expected, atol=1e-10,
            err_msg=f"Up-ratio mismatch at n={n_val}",
        )

    for n_val in [1, 2, 3, 5]:
        log_n = log_psi(r, R, jnp.array(n_val), params)
        log_nm1 = log_psi(r, R, jnp.array(n_val - 1), params)
        ratio = float(jnp.exp(log_nm1 - log_n))
        expected = np.sqrt(n_val) / alpha
        np.testing.assert_allclose(
            ratio, expected, atol=1e-10,
            err_msg=f"Down-ratio mismatch at n={n_val}",
        )


def test_alpha_zero_recovers_pure_vacuum():
    """At α=0: log Ψ(r, 0) = log Ψ_e(r); log Ψ(r, n>0) = -inf (zero amplitude).
    """

    def mock_elec_log_psi(r, R, params):
        return jnp.array(1.5)

    with _stub_nn_factory(mock_elec_log_psi):
        log_psi, params, _ = make_qed_nn_log_psi(
            config=None, mol_info=None, rng_key=jax.random.key(0),
            omega=0.5, coupling_vec=jnp.array([0., 0., 0.0]),
            alpha_init=0.0,
        )

    r = jnp.zeros((2, 3))
    R = jnp.zeros((1, 3))
    log0 = float(log_psi(r, R, jnp.array(0), params))
    np.testing.assert_allclose(log0, 1.5, atol=1e-12)

    # n=1 with α=0 should give a very negative log (effectively log 0)
    log1 = float(log_psi(r, R, jnp.array(1), params))
    assert (
        log1 < -50.0
    )  # implementation uses an _EPS floor; just verify it's tiny


def test_adapter_integrates_with_pauli_fierz_local_energy():
    """Smoke test: adapter output flows through pauli_fierz_local_energy."""

    def mock_elec_log_psi(r, R, params):
        # Standard 2-electron Gaussian — has nontrivial kinetic energy.
        return -0.5 * jnp.sum(r ** 2)

    with _stub_nn_factory(mock_elec_log_psi):
        log_psi, params, _ = make_qed_nn_log_psi(
            config=None, mol_info=None, rng_key=jax.random.key(0),
            omega=0.5, coupling_vec=jnp.array([0., 0., 0.1]),
            alpha_init=0.3,
        )

    nuc = jnp.array([[0., 0., -0.7], [0., 0., 0.7]])
    charges = jnp.array([1., 1.])
    elec = jnp.array([[0., 0., -0.6], [0., 0., 0.4]])

    # Should produce a finite real value.
    e_loc = pauli_fierz_local_energy(
        log_psi, params, elec, jnp.array(0), nuc, charges,
        omega=0.5, coupling_vec=jnp.array([0., 0., 0.1]),
        nph_max=5,
    )
    assert jnp.isfinite(e_loc)
    # Sanity: should be order O(1-10) for this small system at small λ.
    assert -50.0 < float(e_loc) < 10.0


def test_analytical_alpha_perturbative_zero_dipole_returns_zero():
    """For symmetric systems (⟨ε·d⟩ = 0), perturbative α = 0."""
    alpha = analytical_alpha_perturbative(
        omega=0.5, coupling_vec=jnp.array([0., 0., 0.1]),
        dipole_expectation=0.0,
    )
    assert alpha == pytest.approx(0.0)


def test_analytical_alpha_perturbative_polar_system():
    """α = -λ·⟨ε·d⟩/sqrt(2ω); test with known input."""
    omega, lam, mu = (
        0.5,
        0.1,
        1.85,
    )  # water-like dipole magnitude in atomic units
    alpha = analytical_alpha_perturbative(
        omega=omega, coupling_vec=jnp.array([0., 0., lam]),
        dipole_expectation=mu,
    )
    expected = -lam * mu / np.sqrt(2.0 * omega)
    np.testing.assert_allclose(alpha, expected, atol=1e-12)


# ---------------------------------------------------------------------------
# From test_qed_adapter_n_aware.py
#
# Tests for the Tang-style n-aware QED-NN ansatz adapter (Phase 2f-1).
#
# Verifies the Fock-head architecture and key invariants:
#   1. Zero-initialised head means initial log Ψ(r, n) = log Ψ_e(r) for all n
#      (n-aware ansatz reduces to standard NN at iter 0).
#   2. Once the head's parameters are perturbed, the Fock-ladder ratio
#      log Ψ(r, n+1) − log Ψ(r, n) becomes r-dependent — this is the
#      architectural fix from Phase 2e (the factorized ansatz had ratios
#      independent of r, which zeroed the bilinear-coupling contribution
#      for symmetric systems).
#   3. log Ψ flows through pauli_fierz_local_energy without errors.
# ---------------------------------------------------------------------------


def _build_h2():
    L = 1.4010
    return Mole_custom.from_arrays(
        charges=[1, 1],
        coords=[[0.0, 0.0, -L / 2], [0.0, 0.0, L / 2]],
        n_up=1, n_down=1,
    )


@pytest.fixture
def h2_n_aware():
    mol = _build_h2()
    log_psi, params, graphdef = make_qed_nn_log_psi_n_aware(
        config="psiformer",
        mol_info=mol,
        rng_key=jax.random.key(99),
        omega=0.5,
        coupling_vec=jnp.array([0.0, 0.0, 0.05]),
        nph_max=4,
        fock_hidden_dim=32,
    )
    return log_psi, params, graphdef, mol


def test_n_aware_returns_combined_params(h2_n_aware):
    """Params dict should expose both 'nn' and 'fock_head'."""
    _, params, _, _ = h2_n_aware
    assert "nn" in params
    assert "fock_head" in params


def test_zero_init_head_means_no_n_dependence(h2_n_aware):
    """At iter 0 (zero-init output), log Ψ(r, n) must equal log Ψ_e(r)
    for *all* n. The Fock head adds exactly 0 at initialization."""
    log_psi, params, _, _ = h2_n_aware

    r = jnp.array([[0.1, 0.0, -0.5], [-0.05, 0.1, 0.6]])
    R = jnp.array([[0.0, 0.0, -0.7005], [0.0, 0.0, 0.7005]])

    val_n0 = float(log_psi(r, R, jnp.array(0), params))
    for n_val in [1, 2, 3, 4]:
        val_n = float(log_psi(r, R, jnp.array(n_val), params))
        np.testing.assert_allclose(
            val_n, val_n0, atol=1e-12,
            err_msg=f"n={n_val}: head should add 0 at zero-init",
        )


def test_perturbed_head_breaks_n_independence(h2_n_aware):
    """After perturbing the Fock head's output layer, log Ψ(r, n+1) −
    log Ψ(r, n) becomes r-dependent — the key architectural fix."""
    log_psi, params, _, _ = h2_n_aware

    # Perturb the output layer of the head with a small random kernel.
    head_params = params["fock_head"]
    rng = jax.random.key(7)
    flat, treedef = jax.tree.flatten(head_params)
    new_flat = []
    for leaf in flat:
        rng, sub = jax.random.split(rng)
        new_flat.append(leaf + 0.1 * jax.random.normal(sub, leaf.shape))
    head_params_pert = jax.tree.unflatten(treedef, new_flat)
    params_pert = {**params, "fock_head": head_params_pert}

    R = jnp.array([[0.0, 0.0, -0.7005], [0.0, 0.0, 0.7005]])

    # Check ratio depends on r (the architectural fix from Phase 2e).
    r1 = jnp.array([[0.1, 0.0, -0.5], [-0.05, 0.1, 0.6]])
    r2 = jnp.array([[-0.3, 0.2, 0.4], [0.5, -0.1, -0.6]])

    diff_n01_r1 = (
        float(log_psi(r1, R, jnp.array(1), params_pert))
        - float(log_psi(r1, R, jnp.array(0), params_pert))
    )
    diff_n01_r2 = (
        float(log_psi(r2, R, jnp.array(1), params_pert))
        - float(log_psi(r2, R, jnp.array(0), params_pert))
    )
    assert abs(diff_n01_r1 - diff_n01_r2) > 1e-6, (
        f"Fock-ladder ratio still r-independent after head perturbation: "
        f"r1 diff={diff_n01_r1:.6f}, r2 diff={diff_n01_r2:.6f}"
    )


def test_n_aware_integrates_with_pauli_fierz_local_energy(h2_n_aware):
    """Smoke test: n-aware log_psi flows through Pauli-Fierz local energy."""
    log_psi, params, _, _ = h2_n_aware

    nuc = jnp.array([[0.0, 0.0, -0.7005], [0.0, 0.0, 0.7005]])
    charges = jnp.array([1.0, 1.0])
    elec = jnp.array([[0.0, 0.0, -0.5], [0.0, 0.0, 0.5]])

    e_loc = pauli_fierz_local_energy(
        log_psi, params, elec, jnp.array(0), nuc, charges,
        omega=0.5,
        coupling_vec=jnp.array([0.0, 0.0, 0.05]),
        nph_max=4,
    )
    # Untrained PsiFormer at fixed coords can produce very large kinetic
    # energies; only assert finiteness here. Physical reasonableness is
    # checked at the full-pipeline (MH-sampled) level in
    # test_qed_vmc_nn_n_aware_smoke.
    assert jnp.isfinite(e_loc)


def test_fock_head_zero_init_alone():
    """FockHead module on its own returns 0 at initialisation
    (independent of any inputs), since the output layer is zeros_init."""
    rngs = nnx.Rngs(jax.random.key(123))
    head = FockHead(
        nph_max=5, n_features=1, hidden_dim=16, rngs=rngs,
    )
    for n_val in [0, 1, 3, 5]:
        for f in [-2.0, 0.0, 0.5, 5.0]:
            out = float(head(jnp.array(n_val), jnp.array([f])))
            np.testing.assert_allclose(out, 0.0, atol=1e-12)


# ---------------------------------------------------------------------------
# From test_qed_tang_native.py
#
# Tests for the Tang-native joint electron-photon ansatz (Phase 2g).
#
# Verifies the per-electron one-hot(n) injection inside the GNN
# (Tang 2025, arXiv:2503.15644 Sec. II.C):
#
#   1. Builds end-to-end on FermiNet+Jastrow+backflow without shape errors.
#   2. log|Psi(r,n)| varies with n (the network actually consumes n).
#   3. Native Slater determinant sign is +/-1 for all n.
#   4. Pauli-Fierz signed local energy is finite for random configs at
#      lambda=0 (decoupling sanity) and at lambda=0.1.
#   5. At lambda=0, n only enters via the photon energy term omega*n;
#      the local-energy difference E_loc(n=k) - E_loc(n=0) - k*omega is
#      bounded by floating-point noise on a *constant-in-n* trial wf.
# ---------------------------------------------------------------------------


def _h2():
    L = 1.4010
    return Mole_custom.from_arrays(
        charges=[1, 1],
        coords=[[0.0, 0.0, -L / 2], [0.0, 0.0, L / 2]],
        n_up=1, n_down=1,
    )


def _build_tang_native(rng_seed=0, nph_max=4):
    cfg = load_nn_config('ferminet')
    cfg = dataclasses.replace(cfg, use_jastrow=True)
    mol = _h2()
    log_psi, params, gd = make_qed_nn_log_psi_tang_native(
        cfg, mol, jax.random.PRNGKey(rng_seed), nph_max=nph_max,
    )
    return log_psi, params, gd, mol


@pytest.fixture(scope="module")
def tang_h2():
    log_psi, params, gd, mol = _build_tang_native()
    return log_psi, params, mol


def test_build_and_forward_finite(tang_h2):
    """log|Psi| and sign are finite scalars for every n in [0, nph_max]."""
    log_psi, params, mol = tang_h2
    nph_max = 4
    elec = jax.random.normal(jax.random.PRNGKey(1), (2, 3)) * 0.5
    nuc = jnp.asarray(mol.coords, dtype=jnp.float64)

    for n_val in range(nph_max + 1):
        logp, sgn = log_psi(elec, nuc, jnp.array(n_val), params)
        assert jnp.isfinite(logp), f"log|Psi| not finite at n={n_val}"
        assert jnp.isfinite(sgn), f"sign not finite at n={n_val}"
        assert float(jnp.abs(sgn)) == pytest.approx(
            1.0, abs=1e-12
        ), f"sign should be +/-1 (native Slater determinant), got {float(sgn)}"


def test_log_psi_varies_with_n(tang_h2):
    """Per-electron one-hot(n) actually influences the network output.

    If the GNN ignored n, log|Psi(r,n)| would be constant in n. Tang
    Sec. II.C: n must enter through every layer of the GNN.
    """
    log_psi, params, mol = tang_h2
    elec = jax.random.normal(jax.random.PRNGKey(2), (2, 3)) * 0.5
    nuc = jnp.asarray(mol.coords, dtype=jnp.float64)

    vals = jnp.array([
        log_psi(elec, nuc, jnp.array(n), params)[0]
        for n in range(5)
    ])
    spread = float(jnp.max(vals) - jnp.min(vals))
    assert spread > 1e-3, (
        f"log|Psi(r,n)| barely varies with n (spread={spread:.2e}); "
        "the n-injection is broken or n is unused by the GNN."
    )


def test_local_energy_finite_at_lambda_zero(tang_h2):
    """Pauli-Fierz local energy is finite at lambda=0 for random configs.

    At lambda=0 the photon couples only via the omega*n term; if the
    bilinear / DSE pathways have any latent NaN, this test catches it.
    """
    log_psi, params, mol = tang_h2
    nuc = jnp.asarray(mol.coords, dtype=jnp.float64)
    charges = jnp.asarray(mol.charges, dtype=jnp.float64)

    rng = jax.random.PRNGKey(3)
    omega = 0.5
    cv0 = jnp.array([0.0, 0.0, 0.0])

    for k in range(4):
        rng, sub = jax.random.split(rng)
        elec = jax.random.normal(sub, (2, 3)) * 0.4
        for n_val in [0, 1, 3]:
            e_loc = pauli_fierz_local_energy_signed(
                log_psi, params, elec, jnp.array(n_val),
                nuc, charges, omega, cv0,
                nph_max=4, enuc=None,
            )
            assert jnp.isfinite(e_loc), (
                f"E_loc not finite at n={n_val} (cfg #{k})"
            )


def test_local_energy_finite_at_lambda_finite(tang_h2):
    """Local energy is finite at lambda=0.1 (bilinear + DSE active)."""
    log_psi, params, mol = tang_h2
    nuc = jnp.asarray(mol.coords, dtype=jnp.float64)
    charges = jnp.asarray(mol.charges, dtype=jnp.float64)

    rng = jax.random.PRNGKey(4)
    cv = jnp.array([0.0, 0.0, 0.1])
    for k in range(4):
        rng, sub = jax.random.split(rng)
        elec = jax.random.normal(sub, (2, 3)) * 0.4
        for n_val in [0, 1, 3]:
            e_loc = pauli_fierz_local_energy_signed(
                log_psi, params, elec, jnp.array(n_val),
                nuc, charges, omega=0.5, coupling_vec=cv,
                nph_max=4, enuc=None,
            )
            assert jnp.isfinite(e_loc), (
                f"E_loc not finite at n={n_val} (cfg #{k}, lambda=0.1)"
            )


def test_grad_through_log_psi_finite(tang_h2):
    """JAX grad through log_psi w.r.t. params is finite for any n."""
    log_psi, params, mol = tang_h2
    nuc = jnp.asarray(mol.coords, dtype=jnp.float64)
    elec = jax.random.normal(jax.random.PRNGKey(5), (2, 3)) * 0.4

    def loss(p, n):
        log, sgn = log_psi(elec, nuc, n, p)
        return log * sgn  # scalar

    for n_val in [0, 2, 4]:
        g = jax.grad(loss)(params, jnp.array(n_val))
        leaves = jax.tree_util.tree_leaves(g)
        for L in leaves:
            assert bool(jnp.all(jnp.isfinite(L))), (
                f"grad NaN/inf at n={n_val}"
            )
