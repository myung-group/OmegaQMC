"""Tests for the periodic edge-feature classes.

Checks:
  * Output shapes and ``__len__`` match the molecular contract,
    so the periodic classes are drop-in replacements for the
    molecular edge features inside
    :class:`~OmegaQMC.psi.nn.gnn.electron_gnn.ElectronGNN`.
  * PeriodicSinCosFeature is invariant under lattice translations
    of the pairwise difference.
  * PeriodicDistancePowerEdgeFeature is smooth (no gradient kink)
    across the cell face.
  * PeriodicDifferenceEdgeFeature gives minimum-image diffs.
"""

import numpy as np
import jax
import jax.numpy as jnp

from OmegaQMC.psi.nn.periodic import make_cubic_lattice
from OmegaQMC.psi.nn.gnn.edge_features_periodic import (
    PeriodicDifferenceEdgeFeature,
    PeriodicSinCosFeature,
    PeriodicDistancePowerEdgeFeature,
    PeriodicCombinedEdgeFeature,
)


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
    """At short range, periodic_norm ≈ 2π |r|.  So
    PeriodicDistancePowerEdgeFeature(powers=[1])(d) ≈ 2π |d| for
    small d — consistent with the molecular distance scaling up
    to the 2π factor."""
    lat = make_cubic_lattice(4.0)
    f = PeriodicDistancePowerEdgeFeature(lattice=lat, powers=[1])
    d = jnp.asarray([1e-3, 2e-3, -1e-3])
    out = float(f(d[None, :])[0, 0])
    expected = 2 * np.pi * float(jnp.linalg.norm(d))
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
