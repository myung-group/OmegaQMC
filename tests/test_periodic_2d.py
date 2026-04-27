"""Unit tests for 2D periodic lattice primitives.

Verifies that the helpers in :mod:`OmegaQMC.psi.nn.periodic` work
correctly with a 2D :class:`PeriodicLattice` constructed by
:func:`make_square_lattice`, and that :func:`periodic_norm` reduces
to the Euclidean distance at short range in 2D.
"""

import numpy as np
import jax.numpy as jnp
import pytest

from OmegaQMC.psi.nn.periodic import (
    fractional_coords,
    make_cubic_lattice,
    make_square_lattice,
    minimum_image_diff,
    periodic_norm,
    periodic_norm_sq,
    periodic_pairwise_diffs,
    periodic_sincos_features,
    wrap_to_cell,
)


# ---------------------------------------------------------------------
# Lattice constructor
# ---------------------------------------------------------------------

def test_square_lattice_dimensions():
    lat = make_square_lattice(3.0)
    assert lat.A.shape == (2, 2)
    assert lat.A_inv.shape == (2, 2)
    assert lat.metric.shape == (2, 2)
    np.testing.assert_allclose(lat.A, jnp.eye(2) * 3.0)
    np.testing.assert_allclose(lat.metric, jnp.eye(2) * 9.0)
    np.testing.assert_allclose(lat.volume, 9.0)


def test_square_lattice_inverse():
    lat = make_square_lattice(2.5)
    np.testing.assert_allclose(lat.A @ lat.A_inv, jnp.eye(2), atol=1e-12)


# ---------------------------------------------------------------------
# Position helpers
# ---------------------------------------------------------------------

def test_fractional_coords_2d():
    lat = make_square_lattice(4.0)
    r = jnp.asarray([[2.0, 1.0], [-3.0, 2.5]])
    s = fractional_coords(r, lat)
    np.testing.assert_allclose(s, r / 4.0)


def test_wrap_to_cell_2d():
    lat = make_square_lattice(2.0)
    r = jnp.asarray([[3.5, -0.5], [-1.0, 2.7]])
    r_w = wrap_to_cell(r, lat)
    s_w = fractional_coords(r_w, lat)
    assert jnp.all(s_w >= 0)
    assert jnp.all(s_w < 1.0)


def test_minimum_image_diff_2d():
    lat = make_square_lattice(2.0)
    diff = jnp.asarray([[1.5, 0.0], [0.0, -1.7]])
    d_mi = minimum_image_diff(diff, lat)
    np.testing.assert_allclose(
        d_mi, jnp.asarray([[-0.5, 0.0], [0.0, 0.3]]), atol=1e-10,
    )


# ---------------------------------------------------------------------
# Periodic features
# ---------------------------------------------------------------------

def test_periodic_sincos_features_2d_shape():
    lat = make_square_lattice(2.0)
    diff = jnp.asarray([[0.3, 0.7], [-0.5, 0.1], [1.0, -0.3]])
    feats = periodic_sincos_features(diff, lat)
    assert feats.shape == (3, 4)


def test_periodic_norm_sq_short_range_limit_2d():
    """At short range periodic_norm_sq reduces to |diff|^2 (Bohr^2)."""
    lat = make_square_lattice(10.0)
    diff = jnp.asarray([[0.001, 0.002], [0.005, -0.003]])
    n2 = periodic_norm_sq(diff, lat)
    expected = jnp.sum(diff ** 2, axis=-1)
    np.testing.assert_allclose(n2, expected, rtol=1e-6, atol=1e-12)


def test_periodic_norm_smooth_at_origin_2d():
    """periodic_norm with safe=True returns finite gradient at r=0."""
    import jax
    lat = make_square_lattice(5.0)
    diff0 = jnp.zeros(2)
    g = jax.grad(lambda d: periodic_norm(d[None, :], lat, safe=True).sum())(diff0)
    assert jnp.all(jnp.isfinite(g))


def test_periodic_norm_periodic_2d():
    """periodic_norm(r + L*e_x) == periodic_norm(r)."""
    lat = make_square_lattice(3.0)
    diff = jnp.asarray([[0.7, -0.4]])
    shift = jnp.asarray([[3.0, 0.0]])
    n0 = periodic_norm(diff, lat)
    n_shift = periodic_norm(diff + shift, lat)
    np.testing.assert_allclose(n0, n_shift, atol=1e-10)


# ---------------------------------------------------------------------
# Periodic pairwise diffs
# ---------------------------------------------------------------------

def test_periodic_pairwise_diffs_2d_shape():
    """In 2D, periodic_pairwise_diffs should return (n1, n2, dim+1=3)."""
    lat = make_square_lattice(4.0)
    r1 = jnp.asarray([[0.0, 0.0], [1.0, 0.5]])
    r2 = jnp.asarray([[0.5, 0.5], [-0.3, 0.7], [1.5, -0.2]])
    feats = periodic_pairwise_diffs(r1, r2, lat)
    assert feats.shape == (2, 3, 3)


# ---------------------------------------------------------------------
# 3D regression — 3D code path must still work
# ---------------------------------------------------------------------

def test_3d_regression_lattice():
    lat = make_cubic_lattice(3.0)
    assert lat.A.shape == (3, 3)
    np.testing.assert_allclose(lat.volume, 27.0)


def test_3d_regression_periodic_pairwise_diffs():
    lat = make_cubic_lattice(4.0)
    r1 = jnp.asarray([[0.0, 0.0, 0.0]])
    r2 = jnp.asarray([[1.0, 1.0, 1.0]])
    feats = periodic_pairwise_diffs(r1, r2, lat)
    assert feats.shape == (1, 1, 4)
