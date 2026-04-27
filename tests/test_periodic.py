"""Unit tests for ``OmegaQMC.psi.nn.periodic``.

Verifies the four properties the HEG PsiFormer relies on:
  * Lattice-translation invariance of the sin/cos feature.
  * Smooth short-range recovery of the Euclidean norm (up to 2π).
  * Numerical equality of the metric-tensor form at several offsets.
  * Finite gradients across the cell boundary (no minimum-image kink).
"""

import numpy as np
import jax
import jax.numpy as jnp
import pytest

from OmegaQMC.psi.nn.periodic import (
    PeriodicLattice,
    make_cubic_lattice,
    make_lattice,
    fractional_coords,
    wrap_to_cell,
    minimum_image_diff,
    periodic_sincos_features,
    periodic_norm_sq,
    periodic_norm,
    periodic_pairwise_diffs,
)


@pytest.fixture
def cubic():
    return make_cubic_lattice(4.0)


@pytest.fixture
def triclinic():
    A = jnp.asarray([
        [3.0, 0.3, 0.1],
        [0.2, 2.8, 0.4],
        [0.0, 0.1, 3.5],
    ], dtype=jnp.float64)
    return make_lattice(A)


# -------------------------------------------------------------
# Basic lattice arithmetic
# -------------------------------------------------------------

def test_cubic_lattice_builds(cubic):
    assert cubic.A.shape == (3, 3)
    np.testing.assert_allclose(
        cubic.A @ cubic.A_inv, np.eye(3), atol=1e-12,
    )
    np.testing.assert_allclose(
        np.asarray(cubic.metric), 16.0 * np.eye(3), atol=1e-12,
    )
    np.testing.assert_allclose(float(cubic.volume), 64.0)


def test_triclinic_inverse_and_metric(triclinic):
    np.testing.assert_allclose(
        np.asarray(triclinic.A) @ np.asarray(triclinic.A_inv),
        np.eye(3), atol=1e-12,
    )
    np.testing.assert_allclose(
        np.asarray(triclinic.metric),
        np.asarray(triclinic.A).T @ np.asarray(triclinic.A),
        atol=1e-12,
    )


def test_fractional_roundtrip(triclinic):
    rng = np.random.default_rng(0)
    r = jnp.asarray(rng.normal(size=(4, 3)))
    s = fractional_coords(r, triclinic)
    r_back = s @ triclinic.A.T
    np.testing.assert_allclose(r, r_back, atol=1e-12)


def test_wrap_to_cell_stays_in_cell(triclinic):
    rng = np.random.default_rng(1)
    r = jnp.asarray(rng.normal(size=(32, 3)) * 10.0)
    r_wrapped = wrap_to_cell(r, triclinic)
    s = fractional_coords(r_wrapped, triclinic)
    assert float(jnp.min(s)) >= -1e-12
    assert float(jnp.max(s)) < 1.0 + 1e-12


def test_minimum_image_is_shortest_orthorhombic():
    """For orthorhombic lattices the fractional MI == Cartesian shortest."""
    A = jnp.diag(jnp.asarray([3.0, 4.0, 2.5]))
    lat = make_lattice(A)
    rng = np.random.default_rng(2)
    diff = jnp.asarray(rng.normal(size=(16, 3)) * 6.0)
    mi = minimum_image_diff(diff, lat)
    shifts = np.stack(np.meshgrid(
        [-1, 0, 1], [-1, 0, 1], [-1, 0, 1], indexing='ij',
    ), axis=-1).reshape(-1, 3)
    A_np = np.asarray(lat.A)
    for d_i, mi_i in zip(np.asarray(diff), np.asarray(mi)):
        cands = d_i + (A_np @ shifts.T).T
        best = np.min(np.sum(cands ** 2, axis=-1))
        assert np.sum(mi_i ** 2) <= best + 1e-10


def test_minimum_image_fractional_bounded(triclinic):
    """For arbitrary lattices the fractional coords are in [-0.5, 0.5)."""
    rng = np.random.default_rng(22)
    diff = jnp.asarray(rng.normal(size=(16, 3)) * 6.0)
    mi = minimum_image_diff(diff, triclinic)
    s = fractional_coords(mi, triclinic)
    assert float(jnp.min(s)) >= -0.5 - 1e-12
    assert float(jnp.max(s)) < 0.5 + 1e-12


# -------------------------------------------------------------
# Sin/cos feature: periodicity
# -------------------------------------------------------------

def test_sincos_periodic_cubic(cubic):
    """sin/cos feature is invariant under lattice shifts."""
    rng = np.random.default_rng(3)
    r1 = jnp.asarray(rng.normal(size=(5, 3)))
    r2 = jnp.asarray(rng.normal(size=(3, 3)))
    diff = r1[:, None, :] - r2[None, :, :]

    f0 = periodic_sincos_features(diff, cubic)
    # Shift one particle by a lattice vector.
    shift = cubic.A @ jnp.asarray([2.0, -1.0, 3.0])
    f_shift = periodic_sincos_features(diff + shift, cubic)
    np.testing.assert_allclose(f0, f_shift, atol=1e-10)


def test_sincos_periodic_triclinic(triclinic):
    rng = np.random.default_rng(4)
    diff = jnp.asarray(rng.normal(size=(7, 3)) * 3.0)
    f0 = periodic_sincos_features(diff, triclinic)
    for n in [(1, 0, 0), (0, -1, 0), (2, 3, -1)]:
        shift = triclinic.A @ jnp.asarray(n, dtype=jnp.float64)
        f_shift = periodic_sincos_features(diff + shift, triclinic)
        np.testing.assert_allclose(f0, f_shift, atol=1e-10)


# -------------------------------------------------------------
# Periodic norm: short-range limit and smoothness
# -------------------------------------------------------------

def test_periodic_norm_small_limit(cubic):
    """Near origin, ``periodic_norm_sq`` approaches the Euclidean
    squared distance (in Bohr^2).  The implementation uses the
    FermiNet 1/(2pi)^2 normalization so it matches Bohr-unit
    Euclidean distances at short range, not the un-normalised
    Cassella formula's (2pi)^2 |r|^2 — see the ``periodic_norm_sq``
    docstring for context.  This test confirms the rescaling."""
    diff = jnp.asarray([1e-4, 2e-4, -1.5e-4])
    n2 = periodic_norm_sq(diff, cubic)
    euclidean_sq = jnp.sum(diff ** 2, axis=-1)
    # Leading order: n2 ~= euclidean_sq (Bohr^2 units).
    ratio = float(n2 / euclidean_sq)
    assert abs(ratio - 1.0) < 1e-4


def test_periodic_norm_periodicity(triclinic):
    rng = np.random.default_rng(5)
    diff = jnp.asarray(rng.normal(size=(6, 3)) * 2.0)
    n0 = periodic_norm_sq(diff, triclinic)
    shift = triclinic.A @ jnp.asarray([3.0, -2.0, 1.0])
    n_shift = periodic_norm_sq(diff + shift, triclinic)
    np.testing.assert_allclose(n0, n_shift, atol=1e-10)


def test_periodic_norm_gradient_finite_at_origin(cubic):
    """Gradient of ‖·‖_p² is well-defined at the origin (= 0)."""
    grad = jax.grad(
        lambda d: periodic_norm_sq(d, cubic),
    )(jnp.zeros(3))
    assert jnp.all(jnp.isfinite(grad))


def test_periodic_norm_gradient_smooth_across_boundary(cubic):
    """Unlike raw minimum-image norm, the smooth periodic norm has
    no gradient jump when a particle crosses the cell face."""
    # Walk across s_x = 0.5 (the minimum-image kink for cubic).
    L = 4.0
    offsets = jnp.linspace(-0.55, 0.55, 31) * L
    grad_fn = jax.grad(
        lambda d: periodic_norm_sq(d, cubic),
    )
    grads = jnp.stack([
        grad_fn(jnp.asarray([x, 0.1, 0.1])) for x in offsets
    ])
    # First differences along the trajectory: no spike > sensible bound.
    step = offsets[1] - offsets[0]
    dgrad = jnp.diff(grads, axis=0) / step
    assert jnp.all(jnp.isfinite(dgrad))
    # The second derivative is bounded because the only trigonometric
    # pieces that appear are sin/cos of 2π s, so |∂²/∂x²| ≤ C·(2π)²·L
    # with C an O(1) constant from the metric factors.  Use a loose
    # sanity bound: no value exceeds 8·(2π)²·L.
    assert float(jnp.max(jnp.abs(dgrad))) < 8.0 * (2 * np.pi) ** 2 * L


# -------------------------------------------------------------
# Integrated pairwise diffs
# -------------------------------------------------------------

def test_periodic_pairwise_diffs_shape(cubic):
    r1 = jnp.asarray(np.random.default_rng(6).normal(size=(4, 3)))
    r2 = jnp.asarray(np.random.default_rng(7).normal(size=(3, 3)))
    out = periodic_pairwise_diffs(r1, r2, cubic)
    assert out.shape == (4, 3, 4)
    # Last channel must equal periodic_norm_sq of the raw diff.
    diff = r1[:, None, :] - r2[None, :, :]
    np.testing.assert_allclose(
        out[..., -1], periodic_norm_sq(diff, cubic), atol=1e-10,
    )


def test_periodic_pairwise_diffs_minimum_image(cubic):
    """First three channels return minimum-image Cartesian diffs."""
    r1 = jnp.asarray([[3.9, 0.0, 0.0]])
    r2 = jnp.asarray([[0.1, 0.0, 0.0]])
    out = periodic_pairwise_diffs(r1, r2, cubic)
    # Raw diff is +3.8 along x but the minimum image is -0.2.
    np.testing.assert_allclose(out[0, 0, 0], -0.2, atol=1e-10)
