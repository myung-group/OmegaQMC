"""Unit tests for OmegaQMC.psi.nn.qed_physics primitives.

Tests the Pauli-Fierz local-energy components against analytical
ground-truth values on toy configurations:
  1. Real-space dipole operator (electronic + total).
  2. Coulomb energy components (e-e, e-n, n-n).
  3. Coherent-state log amplitude.
  4. Pauli-Fierz local energy on a Gaussian trial wavefunction.
  5. Fock-ladder coupling at n=0 (single up branch) and n>0 (both branches).
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from OmegaQMC.psi.nn.qed_physics import (
    real_space_dipole_electronic,
    real_space_dipole_total,
    coulomb_ee, coulomb_en, coulomb_nn,
    coherent_state_log_amplitude,
    pauli_fierz_local_energy,
)


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
    """Verify <n=2 | alpha=1.0> = e^{-0.5} * 1 / sqrt(2!) = e^{-0.5}/sqrt(2)."""
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
    elec = jnp.array([[0., 0., -0.6], [0., 0., 0.4]])  # asymmetric -> nonzero dipole z

    def n_dep_log_psi(r, R, n, params):
        # Psi = electron Gaussian * coherent-state envelope at alpha=0.3
        alpha = 0.3
        n_f = jnp.asarray(n, dtype=jnp.float64)
        log_chi = -0.5 * alpha**2 + n_f * jnp.log(alpha) - 0.5 * jax.lax.lgamma(n_f + 1.0)
        return -0.5 * jnp.sum(r**2) + log_chi

    omega, lam = 0.5, 0.1
    e_loc = pauli_fierz_local_energy(
        n_dep_log_psi, None, elec, jnp.array(0), nuc, charges,
        omega=omega, coupling_vec=jnp.array([0., 0., lam]),
        nph_max=5, enuc=None,
    )
    # Manual breakdown:
    #   Electronic dipole z = -(-0.6 + 0.4) = 0.2  →  eps_dot_d = 0.2
    #   E_K (Gauss): grad=−r ⇒ |grad|^2 = 0.36+0.16 = 0.52, lap = -6 → E_K = 2.74
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
    e_bilin = np.sqrt(omega/2) * lam * 0.2 * 0.3
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
        log_chi = -0.5 * alpha_cs**2 + n_f * jnp.log(alpha_cs) - 0.5 * jax.lax.lgamma(n_f + 1.0)
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
    e_bilin = np.sqrt(omega/2) * lam * eps_dot_d * bilin_factor
    e_dse = 0.5 * lam**2 * eps_dot_d**2
    expected = e_K + e_ee + e_en + e_nn + e_ph + e_bilin + e_dse
    np.testing.assert_allclose(float(e_loc), expected, atol=1e-4)


def test_pauli_fierz_at_truncation_boundary():
    """At n = nph_max: up branch must be zeroed (no |n+1> state available)."""
    nuc = jnp.array([[0., 0., -0.7], [0., 0., 0.7]])
    charges = jnp.array([1., 1.])
    elec = jnp.array([[0., 0., -0.6], [0., 0., 0.4]])

    def constant_n_dep_log_psi(r, R, n, params):
        # Constant in n so any ratio = 1; both branches contribute their sqrt factors
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
    expected_bilin = np.sqrt(omega/2) * lam * eps_dot_d * bilin_factor
    # Test indirectly: compare to nph_max=10 (no truncation) result
    e_loc_full = pauli_fierz_local_energy(
        constant_n_dep_log_psi, None, elec, jnp.array(nph_max),
        nuc, charges,
        omega=omega, coupling_vec=jnp.array([0., 0., lam]),
        nph_max=10, enuc=None,
    )
    # Difference should be exactly the up-branch contribution that got removed
    diff = float(e_loc_full) - float(e_loc_truncated)
    expected_up_branch = np.sqrt(omega/2) * lam * eps_dot_d * np.sqrt(float(nph_max + 1))
    np.testing.assert_allclose(diff, expected_up_branch, atol=1e-6)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
