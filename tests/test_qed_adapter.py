"""Tests for the coherent-state-shifted QED-NN ansatz adapter.

We focus on adapter logic and analytical invariants rather than end-to-end
PsiFormer integration (which needs full NN-VMC compute and is exercised by
``test_qed_vmc_nn_h2`` once Phase 2c lands).

Key invariant tested
--------------------
For the factorized form Ψ(r, n) = Ψ_e(r) ⟨n|α⟩, the Fock-ladder ratio that
appears in the bilinear Pauli-Fierz coupling reduces to a closed-form:

    Ψ(r, n+1)/Ψ(r, n)  =  ⟨n+1|α⟩/⟨n|α⟩  =  α / sqrt(n+1)
    Ψ(r, n-1)/Ψ(r, n)  =  ⟨n-1|α⟩/⟨n|α⟩  =  sqrt(n) / α  (n ≥ 1)

These are independent of r and serve as an exact invariant for testing.
"""
import unittest.mock as mock

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from OmegaQMC.psi.nn.qed_adapter import (
    make_qed_nn_log_psi,
    analytical_alpha_perturbative,
    QEDLogPsiParams,
)
from OmegaQMC.psi.nn.qed_physics import (
    coherent_state_log_amplitude,
    pauli_fierz_local_energy,
)


def _stub_nn_factory(mock_log_psi):
    """Patch make_nn_log_psi inside qed_adapter to return a known stub."""
    return mock.patch(
        "OmegaQMC.psi.nn.qed_adapter.make_nn_log_psi",
        return_value=(mock_log_psi, {"weight": jnp.array(1.5)}, "fake_graphdef"),
    )


def test_adapter_returns_callable_and_combined_params():
    """make_qed_nn_log_psi should expose log_psi callable + alpha-augmented params."""

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
    """Ψ(r, n+1)/Ψ(r, n) = α / sqrt(n+1) regardless of r — analytical invariant."""

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
    """At α=0: log Ψ(r, 0) = log Ψ_e(r); log Ψ(r, n>0) = -inf (zero amplitude)."""

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
    assert log1 < -50.0  # implementation uses an _EPS floor; just verify it's tiny


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
    omega, lam, mu = 0.5, 0.1, 1.85   # water-like dipole magnitude in atomic units
    alpha = analytical_alpha_perturbative(
        omega=omega, coupling_vec=jnp.array([0., 0., lam]),
        dipole_expectation=mu,
    )
    expected = -lam * mu / np.sqrt(2.0 * omega)
    np.testing.assert_allclose(alpha, expected, atol=1e-12)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
