"""Unit tests for the complex plane-wave envelope.

Covers the TABC prerequisites:
  * Basis enumeration at κ=0 recovers the Γ-point closed shells
    (1, 7, 19, 27, …).
  * Basis enumeration at κ≠0 breaks the shell degeneracy and is
    consistent with analytical expectations.
  * The envelope is lattice-periodic (shift electrons by any lattice
    vector, the complex orbital matrix picks up a global phase that
    drops out of |ψ|²).
  * At κ=0, ``|det(complex-envelope orbitals)|²`` equals
    ``|det(real-envelope orbitals)|²`` — same physical Slater
    determinant up to a global phase.
  * ``jax.grad`` through a complex log-det is finite.
"""

from typing import NamedTuple

import numpy as np
import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from OmegaQMC.psi.nn.env_periodic import (
    ComplexPlaneWaveEnvelope,
    PlaneWaveEnvelope,
    enumerate_complex_pw_basis,
)


class _PhysConfHEG(NamedTuple):
    R: jax.Array
    r: jax.Array
    mol_idx: jax.Array


def _pc(r):
    return _PhysConfHEG(R=jnp.zeros((0, 3)),
                        r=r, mol_idx=jnp.asarray(0))


# -----------------------------------------------------------
# Basis enumeration
# -----------------------------------------------------------

def test_complex_basis_gamma_closed_shells():
    """At κ=0, 7 orbitals → k=0 + first shell of 6 integer vectors."""
    basis = enumerate_complex_pw_basis(7, L=4.0, kappa=(0, 0, 0))
    assert basis.kvecs.shape == (7, 3)
    n_ints = np.asarray(basis.n_ints)
    # Orbital 0 must be k=0.
    np.testing.assert_array_equal(n_ints[0], [0, 0, 0])
    # Remaining 6 must be ±e_x, ±e_y, ±e_z (any order).
    rest = {tuple(n) for n in n_ints[1:]}
    expected = {
        (+1, 0, 0), (-1, 0, 0),
        (0, +1, 0), (0, -1, 0),
        (0, 0, +1), (0, 0, -1),
    }
    assert rest == expected


def test_complex_basis_kappa_breaks_degeneracy():
    """At κ ≠ 0 the 7 lowest-|k|² integer triples are no longer a
    symmetric ±-pair set; instead we pick the 7 whose |n+κ|² values
    are smallest."""
    kappa = (0.1, 0.0, 0.0)
    basis = enumerate_complex_pw_basis(7, L=4.0, kappa=kappa)
    k_sq = np.asarray(basis.k_sq)
    assert np.all(np.diff(k_sq) >= -1e-12)
    # |(-1, 0, 0) + (0.1, 0, 0)|² = 0.81  <  |(1, 0, 0) + (0.1, 0, 0)|² = 1.21
    # So the basis should include (-1, 0, 0) before (+1, 0, 0).
    n_ints = np.asarray(basis.n_ints)
    idx_neg = next(i for i, n in enumerate(n_ints)
                   if tuple(n) == (-1, 0, 0))
    idx_pos = next((i for i, n in enumerate(n_ints)
                    if tuple(n) == (1, 0, 0)), len(n_ints))
    assert idx_neg < idx_pos


# -----------------------------------------------------------
# Envelope forward
# -----------------------------------------------------------

def test_complex_envelope_shape():
    env = ComplexPlaneWaveEnvelope(n_up=7, n_down=7, n_det=4, L=4.0)
    rng = np.random.default_rng(0)
    r = jnp.asarray(rng.uniform(0, 4.0, size=(14, 3)))
    out = env(_pc(r))
    assert out.shape == (4, 14, 14)
    assert out.dtype == jnp.complex128


def test_complex_envelope_lattice_periodicity():
    """Shifting all electrons by a lattice vector multiplies each
    orbital by exp(i k · a) — a global *k-dependent* phase.  The
    determinant picks up the product of these phases but its
    modulus is unchanged."""
    L = 4.0
    n_up = n_down = 7
    env = ComplexPlaneWaveEnvelope(n_up=n_up, n_down=n_down, n_det=1, L=L)
    rng = np.random.default_rng(1)
    r = jnp.asarray(rng.uniform(0, L, size=(n_up + n_down, 3)))
    out0 = env(_pc(r))
    shift = L * jnp.asarray([1.0, -2.0, 3.0])
    out1 = env(_pc(r + shift))
    # Up block Slater determinants must have identical modulus.
    det0 = jnp.linalg.det(out0[0, :n_up, :n_up])
    det1 = jnp.linalg.det(out1[0, :n_up, :n_up])
    np.testing.assert_allclose(
        float(jnp.abs(det0)), float(jnp.abs(det1)), atol=1e-10,
    )


def test_complex_vs_real_envelope_same_physical_density():
    """At κ = 0, the two Slater determinants differ only by a linear
    change of basis ``{1, cos, sin} ↔ {1, e^{±ik·r}}`` which is
    invertible but not unit-determinant.  Their ratio
    ``|det_c|² / |det_r|²`` is therefore a geometry-independent
    constant — the |det(basis change)|² factor — and the ratio of
    ratios at two configurations must be 1.  That constant factor
    drops out of VMC observables (it's a wavefunction-normalization
    convention)."""
    L = 4.0
    n_up = n_down = 7

    env_c = ComplexPlaneWaveEnvelope(n_up=n_up, n_down=n_down, n_det=1, L=L)
    env_r = PlaneWaveEnvelope(n_up=n_up, n_down=n_down, n_det=1, L=L)

    def det_ratio(seed):
        r = jnp.asarray(
            np.random.default_rng(seed).uniform(0, L, size=(n_up + n_down, 3))
        )
        det_c = jnp.linalg.det(env_c(_pc(r))[0, :n_up, :n_up])
        det_r = jnp.linalg.det(env_r(_pc(r))[0, :n_up, :n_up])
        return float(jnp.abs(det_c) ** 2) / float(jnp.abs(det_r) ** 2)

    ratios = [det_ratio(s) for s in range(5)]
    # The ratio must be constant across configurations.  For the 7-orbital
    # unpolarised basis at Γ the analytic value is 4³ = 64 (from three
    # ±k-pair 2×2 change-of-basis blocks each with |det|² = 4).
    np.testing.assert_allclose(ratios, [ratios[0]] * len(ratios),
                               rtol=1e-10)
    assert abs(ratios[0] - 64.0) < 1e-6


def test_complex_envelope_grad_finite():
    """jax.grad of log|det(orb_up)| through the complex Slater block
    produces finite real gradients."""
    L = 4.0
    n_up = 7
    env = ComplexPlaneWaveEnvelope(n_up=n_up, n_down=7, n_det=1, L=L)
    graphdef, params, other = nnx.split(env, nnx.Param, ...)

    def log_abs_det(r):
        m = nnx.merge(graphdef, params, other)
        out = m(_pc(r))
        d = jnp.linalg.det(out[0, :n_up, :n_up])
        return jnp.log(jnp.abs(d))

    rng = np.random.default_rng(3)
    r = jnp.asarray(rng.uniform(0, L, size=(14, 3)))
    g = jax.grad(log_abs_det)(r)
    assert jnp.all(jnp.isfinite(g))
    assert g.dtype == jnp.float64
