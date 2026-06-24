"""Regression tests for ``_project_force_sum_rules``.

The fragment-wise PGCS average in ``postproc_h5_pgcs``
combines different atoms over different state subsets, so
the assembled force vector no longer satisfies the exact
translational and rotational sum rules of an isolated
molecule (each per-state estimator does, by SWCT, but the
mosaic recombination breaks the cancellation).

``_project_force_sum_rules`` restores both sum rules,

    sum_i F_i = 0          (translational)
    sum_i r_i x F_i = 0    (rotational)

via a constrained generalized-least-squares projection
weighted by the per-component error bars.  These tests pin
the defining properties of that projection:

* both sum rules are satisfied to machine precision;
* error bars are never inflated;
* the map is idempotent and a leaves a sum-rule-consistent
  field untouched;
* the residual is absorbed mostly by the least
  well-determined components (the GLS weighting);
* degenerate geometries (collinear atoms, whose torque
  about the molecular axis has a vanishing constraint row)
  are handled without NaNs.

The function is pure NumPy, so no SCF / JAX machinery is
needed.
"""
import numpy as np

from OmegaQMC.observables.force import _project_force_sum_rules


def _total_force(grd):
    # Forces are -gradients; the translational sum rule is
    # identical for forces and gradients.
    return (-grd).sum(axis=0)


def _total_torque(grd, coords, ref):
    rel = coords - ref
    return np.cross(rel, -grd).sum(axis=0)


def _bent_trimer():
    # A genuinely 3-D, low-symmetry arrangement so all six
    # constraint rows are linearly independent.
    coords = np.array([
        [0.00, 0.00, 0.00],
        [1.43, 1.11, 0.20],
        [-1.40, 1.05, -0.15],
        [5.50, 0.20, 0.40],
        [6.40, 1.00, 1.40],
        [6.35, 0.95, -1.45],
    ])
    return coords


def test_both_sum_rules_enforced():
    coords = _bent_trimer()
    ref = coords.mean(axis=0)
    rng = np.random.default_rng(7)
    grd = rng.normal(scale=0.01, size=coords.shape)
    err = rng.uniform(0.001, 0.005, size=coords.shape)

    # Pre-projection: both sum rules are violated.
    assert np.linalg.norm(_total_force(grd)) > 1e-3
    assert np.linalg.norm(_total_torque(grd, coords, ref)) > 1e-3

    grd_p, err_p = _project_force_sum_rules(grd, err, coords, ref)

    assert np.allclose(_total_force(grd_p), 0.0, atol=1e-12)
    assert np.allclose(
        _total_torque(grd_p, coords, ref), 0.0, atol=1e-12,
    )


def test_torque_reference_independent_after_projection():
    # Once sum F = 0 is enforced, the total torque is the
    # same about any reference point, so it must vanish about
    # an arbitrary origin too, not only the one used to build
    # the constraint.
    coords = _bent_trimer()
    ref = coords.mean(axis=0)
    rng = np.random.default_rng(21)
    grd = rng.normal(scale=0.01, size=coords.shape)
    err = rng.uniform(0.001, 0.005, size=coords.shape)

    grd_p, _ = _project_force_sum_rules(grd, err, coords, ref)

    other = np.array([3.3, -1.7, 2.9])
    assert np.allclose(
        _total_torque(grd_p, coords, other), 0.0, atol=1e-12,
    )


def test_errors_never_inflated():
    coords = _bent_trimer()
    ref = coords.mean(axis=0)
    rng = np.random.default_rng(3)
    grd = rng.normal(scale=0.01, size=coords.shape)
    err = rng.uniform(0.001, 0.005, size=coords.shape)

    _, err_p = _project_force_sum_rules(grd, err, coords, ref)

    assert np.all(err_p <= err + 1e-12)
    assert np.all(err_p >= 0.0)


def test_idempotent_and_fixed_point():
    coords = _bent_trimer()
    ref = coords.mean(axis=0)
    rng = np.random.default_rng(99)
    grd = rng.normal(scale=0.01, size=coords.shape)
    err = rng.uniform(0.001, 0.005, size=coords.shape)

    grd_p, err_p = _project_force_sum_rules(grd, err, coords, ref)
    # Re-projecting with the SAME error metric is a no-op:
    # the projector is idempotent and grd_p already lies in
    # the constraint null space.  (Feeding the reduced err_p
    # back would change Sigma and is not how the map is used.)
    grd_pp, err_pp = _project_force_sum_rules(
        grd_p, err, coords, ref,
    )
    assert np.allclose(grd_pp, grd_p, atol=1e-12)
    assert np.allclose(err_pp, err_p, atol=1e-12)


def test_consistent_field_unchanged():
    # A field that already satisfies both sum rules must be
    # returned untouched (up to round-off).
    coords = _bent_trimer()
    ref = coords.mean(axis=0)
    rng = np.random.default_rng(42)
    grd0 = rng.normal(scale=0.01, size=coords.shape)
    err = rng.uniform(0.001, 0.005, size=coords.shape)
    # Make grd0 sum-rule-consistent by projecting it once.
    grd0, _ = _project_force_sum_rules(grd0, err, coords, ref)

    grd_p, _ = _project_force_sum_rules(grd0, err, coords, ref)
    assert np.allclose(grd_p, grd0, atol=1e-12)


def test_gls_weighting_protects_precise_components():
    # Inject a pure net force (no torque) and make one atom
    # far better determined than the rest.  The GLS solution
    # must move the precise atom much less than a noisy one.
    coords = _bent_trimer()
    ref = coords.mean(axis=0)
    grd = np.zeros_like(coords)
    grd[:, 0] = 0.01                       # uniform x residual
    err = np.full(coords.shape, 0.01)
    err[0, 0] = 1e-4                        # atom 0 x: precise

    grd_p, _ = _project_force_sum_rules(grd, err, coords, ref)
    shift = np.abs(grd_p - grd)

    # Translational x residual must be removed ...
    assert abs(_total_force(grd_p)[0]) < 1e-12
    # ... almost entirely off the noisy atoms, sparing the
    # precise one by orders of magnitude.
    assert shift[0, 0] < 1e-5
    assert shift[1, 0] > 1e-3
    assert shift[0, 0] < 0.01 * shift[1, 0]


def test_collinear_molecule_no_nan():
    # Linear molecule along z: the torque-about-z constraint
    # row vanishes identically, making C Sigma C^T singular.
    # The pseudo-inverse must keep the result finite while
    # still zeroing the well-defined sum rules.
    coords = np.array([
        [0.0, 0.0, -2.1],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 2.1],
    ])
    ref = coords.mean(axis=0)
    rng = np.random.default_rng(5)
    grd = rng.normal(scale=0.01, size=coords.shape)
    err = rng.uniform(0.001, 0.005, size=coords.shape)

    grd_p, err_p = _project_force_sum_rules(grd, err, coords, ref)

    assert np.all(np.isfinite(grd_p))
    assert np.all(np.isfinite(err_p))
    assert np.allclose(_total_force(grd_p), 0.0, atol=1e-12)
    # The two well-defined torque components (about x, y)
    # vanish; the z component is identically zero for any
    # collinear-along-z field.
    tau = _total_torque(grd_p, coords, ref)
    assert np.allclose(tau, 0.0, atol=1e-12)


if __name__ == "__main__":
    test_both_sum_rules_enforced()
    test_torque_reference_independent_after_projection()
    test_errors_never_inflated()
    test_idempotent_and_fixed_point()
    test_consistent_field_unchanged()
    test_gls_weighting_protects_precise_components()
    test_collinear_molecule_no_nan()
    print("OK")
