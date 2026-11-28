"""Periodic lattice primitives and edge features, 3D and 2D.

Lattice vectors, minimum image, periodic sin/cos features and the
periodic norm, whose gradient must stay finite at the origin and smooth
across the cell boundary (the sqrt(0) NaN), plus the periodic edge
features built from them.
"""
import numpy as np
import jax
import jax.numpy as jnp
import pytest

from OmegaQMC.psi.nn.periodic import (  # noqa: E402
    make_cubic_lattice,
    make_lattice,
    fractional_coords,
    wrap_to_cell,
    minimum_image_diff,
    periodic_sincos_features,
    periodic_norm_sq,
    periodic_norm,
    periodic_pairwise_diffs,
    make_square_lattice,
)
from OmegaQMC.psi.nn.gnn.edge_features_periodic import (  # noqa: E402
    PeriodicDifferenceEdgeFeature,
    PeriodicSinCosFeature,
    PeriodicDistancePowerEdgeFeature,
    PeriodicCombinedEdgeFeature,
)


# ---------------------------------------------------------------------------
# From test_periodic.py
#
# Unit tests for ``OmegaQMC.psi.nn.periodic``.
#
# Verifies the four properties the HEG PsiFormer relies on:
#   * Lattice-translation invariance of the sin/cos feature.
#   * Smooth short-range recovery of the Euclidean norm (up to 2π).
#   * Numerical equality of the metric-tensor form at several offsets.
#   * Finite gradients across the cell boundary (no minimum-image kink).
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# From test_periodic_2d.py
#
# Unit tests for 2D periodic lattice primitives.
#
# Verifies that the helpers in :mod:`OmegaQMC.psi.nn.periodic` work
# correctly with a 2D :class:`PeriodicLattice` constructed by
# :func:`make_square_lattice`, and that :func:`periodic_norm` reduces
# to the Euclidean distance at short range in 2D.
# ---------------------------------------------------------------------------


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
    g = jax.grad(lambda d: periodic_norm(d[None, :], lat, safe=True).sum())(
        diff0
    )
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


# ---------------------------------------------------------------------------
# From test_edge_features_periodic.py
#
# Tests for the periodic edge-feature classes.
#
# Checks:
#   * Output shapes and ``__len__`` match the molecular contract,
#     so the periodic classes are drop-in replacements for the
#     molecular edge features inside
#     :class:`~OmegaQMC.psi.nn.gnn.electron_gnn.ElectronGNN`.
#   * PeriodicSinCosFeature is invariant under lattice translations
#     of the pairwise difference.
#   * PeriodicDistancePowerEdgeFeature is smooth (no gradient kink)
#     across the cell face.
#   * PeriodicDifferenceEdgeFeature gives minimum-image diffs.
# ---------------------------------------------------------------------------


def test_difference_shape_and_mi():
    lat = make_cubic_lattice(4.0)
    f = PeriodicDifferenceEdgeFeature(lattice=lat)
    assert len(f) == 3
    # Pair diff outside the minimum image → must wrap.
    d = jnp.asarray([[3.8, 0.0, 0.0]])
    out = f(d)
    assert out.shape == (1, 3)
    # Expected MI: 3.8 - 4.0 = -0.2.
    np.testing.assert_allclose(out[0], [-0.2, 0.0, 0.0], atol=1e-12)


def test_sincos_shape_and_periodicity():
    lat = make_cubic_lattice(4.0)
    f = PeriodicSinCosFeature(lattice=lat)
    assert len(f) == 6
    rng = np.random.default_rng(0)
    d = jnp.asarray(rng.normal(size=(8, 3)))
    out = f(d)
    assert out.shape == (8, 6)
    # Shift by a lattice vector → feature unchanged.
    shift = 4.0 * jnp.asarray([1.0, -2.0, 3.0])
    out_shift = f(d + shift)
    np.testing.assert_allclose(out, out_shift, atol=1e-10)


def test_distance_power_shape_and_smoothness():
    lat = make_cubic_lattice(4.0)
    f = PeriodicDistancePowerEdgeFeature(lattice=lat, powers=[1, 2])
    assert len(f) == 2
    d = jnp.asarray([[0.1, 0.2, -0.05]])
    out = f(d)
    assert out.shape == (1, 2)
    # Gradient finite at a point near the cell face — where the
    # MI distance has a kink but the smooth periodic norm does not.
    d_face = jnp.asarray([1.99, 0.3, 0.1])  # near s_x = 0.5.
    g = jax.grad(lambda x: jnp.sum(
        f(x[None, :])
    ))(d_face)
    assert jnp.all(jnp.isfinite(g))


def test_distance_power_matches_molecular_at_short_range():
    """periodic_norm carries the 1/(2π) normalisation, so at short
    range it is |r| in Bohr; PeriodicDistancePowerEdgeFeature(
    powers=[1])(d) then equals the molecular distance |d| for small
    d."""
    lat = make_cubic_lattice(4.0)
    f = PeriodicDistancePowerEdgeFeature(lattice=lat, powers=[1])
    d = jnp.asarray([1e-3, 2e-3, -1e-3])
    out = float(f(d[None, :])[0, 0])
    expected = float(jnp.linalg.norm(d))
    assert abs(out - expected) / expected < 1e-4


def test_combined_feature_concatenates():
    lat = make_cubic_lattice(4.0)
    sincos = PeriodicSinCosFeature(lattice=lat)
    dist = PeriodicDistancePowerEdgeFeature(lattice=lat, powers=[1])
    combined = PeriodicCombinedEdgeFeature(
        features=[sincos, dist],
    )
    assert len(combined) == 6 + 1
    d = jnp.asarray([[0.3, -0.1, 0.2]])
    out = combined(d)
    assert out.shape == (1, 7)
    # Check individual contributions.
    np.testing.assert_allclose(out[:, :6], sincos(d), atol=1e-12)
    np.testing.assert_allclose(out[:, 6:7], dist(d), atol=1e-12)


def test_drop_in_compatibility_with_molecular_interface():
    """Periodic edge features must expose the same call signature
    and ``__len__`` as the molecular ones so they can be used
    inside ``ElectronGNN`` without changes to the GNN code."""
    from OmegaQMC.psi.nn.gnn.edge_features import (
        DifferenceEdgeFeature,
        DistancePowerEdgeFeature,
    )
    lat = make_cubic_lattice(4.0)

    mol_diff = DifferenceEdgeFeature()
    mol_dist = DistancePowerEdgeFeature(powers=[1])
    per_diff = PeriodicDifferenceEdgeFeature(lattice=lat)
    per_dist = PeriodicDistancePowerEdgeFeature(
        lattice=lat, powers=[1],
    )
    # Same output rank (3-vec and k-vec respectively).
    assert len(mol_diff) == len(per_diff) == 3
    assert len(mol_dist) == len(per_dist) == 1
    # Both accept (..., 3) diffs and return (..., len(f)) features.
    d = jnp.asarray([[0.1, 0.2, 0.3]])
    assert mol_diff(d).shape == per_diff(d).shape
    assert mol_dist(d).shape == per_dist(d).shape
