"""Unit tests for 2D plane-wave envelopes.

Verifies real and complex 2D PW basis enumeration matches the 2D
closed-shell sequence (1, 5, 9, 13, 21, 25, 29, 37, 45, 57, ...) and
that the envelopes evaluate to the correct shape on random walkers.
"""

import numpy as np
import jax.numpy as jnp
import pytest

from OmegaQMC.psi.nn.env_periodic import (
    ComplexPlaneWaveEnvelope,
    PlaneWaveEnvelope,
    enumerate_complex_pw_basis_2d,
    enumerate_real_pw_basis_2d,
)
from OmegaQMC.psi.nn.types import PhysicalConfiguration


# ---------------------------------------------------------------------
# Real-valued 2D PW basis
# ---------------------------------------------------------------------

@pytest.mark.parametrize('n_orb', [1, 5, 9, 13, 21, 25, 29, 37, 45, 57])
def test_real_pw_basis_2d_size(n_orb):
    basis = enumerate_real_pw_basis_2d(n_orb, L=10.0)
    assert basis.basis_idx.shape == (n_orb,)
    assert basis.kvecs.shape[1] == 2


def test_real_pw_basis_2d_first_orbital_is_constant():
    basis = enumerate_real_pw_basis_2d(1, L=5.0)
    np.testing.assert_allclose(basis.kvecs, jnp.zeros((1, 2)))
    assert int(basis.basis_is_sin[0]) == 0


def test_real_pw_basis_2d_n5_has_first_shell():
    """nup=5 fills shells |n|^2 = 0, 1.  Five basis functions:
    constant + (cos kx, sin kx) + (cos ky, sin ky)."""
    basis = enumerate_real_pw_basis_2d(5, L=5.0)
    assert basis.basis_idx.shape == (5,)
    # 1 constant + 2 cos + 2 sin
    n_cos = int(np.sum(np.asarray(basis.basis_is_sin) == 0))
    n_sin = int(np.sum(np.asarray(basis.basis_is_sin) == 1))
    assert n_cos == 3 and n_sin == 2


# ---------------------------------------------------------------------
# Complex 2D PW basis
# ---------------------------------------------------------------------

def test_complex_pw_basis_2d_shape():
    basis = enumerate_complex_pw_basis_2d(29, L=10.0, kappa=(0.0, 0.0))
    assert basis.kvecs.shape == (29, 2)
    assert basis.k_sq.shape == (29,)
    assert basis.kappa.shape == (2,)


def test_complex_pw_basis_2d_kappa_shifts():
    basis_g = enumerate_complex_pw_basis_2d(5, L=10.0, kappa=(0.0, 0.0))
    basis_k = enumerate_complex_pw_basis_2d(5, L=10.0, kappa=(0.1, 0.2))
    # Kappa shift breaks degeneracy: at twist all 5 k_sq differ;
    # at Γ the shells are degenerate.
    assert len(set(np.asarray(basis_k.k_sq).tolist())) == 5


def test_complex_pw_basis_2d_kappa_wrong_dim_raises():
    with pytest.raises(ValueError):
        enumerate_complex_pw_basis_2d(5, L=10.0, kappa=(0.1, 0.2, 0.3))


# ---------------------------------------------------------------------
# Envelope evaluation
# ---------------------------------------------------------------------

def _make_2d_phys_conf(n_elec, L, seed=0):
    rng = np.random.default_rng(seed)
    return PhysicalConfiguration(
        R=jnp.zeros((0, 2)),
        r=jnp.asarray(rng.uniform(0, L, size=(n_elec, 2))),
        mol_idx=jnp.asarray(0),
    )


def test_real_envelope_2d_shape_unpolarized():
    L = 10.0
    env = PlaneWaveEnvelope(
        n_up=29, n_down=29, n_det=1, L=L, dim=2,
    )
    pc = _make_2d_phys_conf(58, L)
    orb = env(pc)
    assert orb.shape == (1, 58, 58)
    assert orb.dtype == jnp.float64


def test_real_envelope_2d_shape_polarized():
    L = 10.0
    env = PlaneWaveEnvelope(
        n_up=29, n_down=0, n_det=1, L=L, dim=2,
    )
    pc = _make_2d_phys_conf(29, L)
    orb = env(pc)
    assert orb.shape == (1, 29, 29)


def test_complex_envelope_2d_shape():
    L = 10.0
    env = ComplexPlaneWaveEnvelope(
        n_up=29, n_down=29, n_det=1, L=L, dim=2, kappa=(0.1, 0.2),
    )
    pc = _make_2d_phys_conf(58, L)
    orb = env(pc)
    assert orb.shape == (1, 58, 58)
    assert jnp.iscomplexobj(orb)


def test_real_envelope_2d_periodic():
    """Envelope output must be invariant under integer-cell shifts of
    the full electron configuration (because the envelope coefficients
    are initialised to plane-wave eigenstates)."""
    L = 10.0
    env = PlaneWaveEnvelope(
        n_up=29, n_down=29, n_det=1, L=L, dim=2,
    )
    pc0 = _make_2d_phys_conf(58, L, seed=42)
    pc_shift = pc0._replace(r=pc0.r + L * jnp.array([1.0, 0.0]))
    orb0 = env(pc0)
    orb_shift = env(pc_shift)
    np.testing.assert_allclose(orb0, orb_shift, atol=1e-9)


def test_real_envelope_2d_invalid_dim_raises():
    with pytest.raises(ValueError):
        PlaneWaveEnvelope(
            n_up=5, n_down=5, n_det=1, L=10.0, dim=4,
        )


# ---------------------------------------------------------------------
# Slater determinant of the 2D envelope reproduces the HEG kinetic energy
# ---------------------------------------------------------------------

def test_2d_envelope_slater_det_kinetic_matches_analytic():
    """At init, the Slater det of the 2D PW envelope is the
    free-electron Fermi sea.  The kinetic energy <T> = sum_k |k|^2/2
    over occupied k must match the analytical sum using the integer
    grid we generated.  This catches off-by-one in the basis ordering.
    """
    from OmegaQMC.heg_2d import build_2deg_system, hf_energy_2d_finite

    sys = build_2deg_system(rs=2.0, N_elec=58, polarization='unpolarized')
    L = sys['L']

    # Analytical kinetic via heg_2d
    hf = hf_energy_2d_finite(sys)
    expected_T_per_elec = hf['kinetic']

    # Envelope-based T: sum |k|^2/2 over the closed-shell basis chosen
    # by the envelope (k = 2pi*n/L, n in the Fermi sea).  This must
    # agree to floating-point precision because the envelope
    # initializes orbitals to those exact plane waves.
    basis = enumerate_real_pw_basis_2d(29, L=L)
    # The basis includes one zero-vector (the constant) + first-shell
    # ks paired (cos, sin).  We need the per-orbital |k|^2: for orbital
    # i we look up basis.basis_idx[i] -> kvec, take its |k|^2.
    k_sq_per_orb = np.asarray(basis.k_sq)[np.asarray(basis.basis_idx)]
    T_envelope = 0.5 * float(np.sum(k_sq_per_orb))
    # Two spins (unpolarized), per-electron average:
    T_envelope_per_elec = 2.0 * T_envelope / sys['N_elec']
    np.testing.assert_allclose(T_envelope_per_elec, expected_T_per_elec, atol=1e-12)
