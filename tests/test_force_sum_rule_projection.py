"""Regression tests for ``_project_force_sum_rules``.

The fragment-wise PGCS average in ``postproc_h5_pgcs``
combines different atoms over different state subsets, so
the assembled force vector no longer satisfies the exact
translational sum rule of an isolated molecule (each
per-state estimator does, by SWCT, but the mosaic
recombination breaks the cancellation).

``_project_force_sum_rules`` restores it,

    sum_i F_i = 0          (translational)

via a constrained generalized-least-squares projection
weighted by the components' covariance.  The rotational rule
is not enforced: with MO coefficients frozen on lab-frame AOs
the total torque of the estimand is -dE_V/d theta, nonzero
unless the orbitals are stationary w.r.t. rotation.  These
tests pin the defining properties of that projection:

* the net force vanishes to machine precision, while the
  torque is left alone (a field with zero net force and a
  nonzero torque passes through unchanged);
* error bars are never inflated (independent components);
* with the components' correlations, a field whose every sample
  already obeys the sum rule keeps its error bars, which the
  independent propagation would shrink;
* weighted by the full covariance (enough bins, positive-definite
  correlations) no error bar grows and chi^2 under V is minimal;
  otherwise the diagonal weighting is kept and reported;
* the map is idempotent and leaves a sum-rule-consistent
  field untouched;
* the residual is absorbed mostly by the least
  well-determined components (the GLS weighting);
* after projection the net-force error is zero and the torque
  error is the propagated one;
* collinear atoms are handled without NaNs.

The function is pure NumPy, so no SCF / JAX machinery is
needed.
"""
import numpy as np
import pytest

from OmegaQMC.observables.force import (
    _block_correlation,
    _project_force_sum_rules,
)


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


def test_translational_sum_rule_enforced_torque_kept():
    coords = _bent_trimer()
    ref = coords.mean(axis=0)
    rng = np.random.default_rng(7)
    grd = rng.normal(scale=0.01, size=coords.shape)
    err = rng.uniform(0.001, 0.005, size=coords.shape)

    # Pre-projection: the net force and the torque are nonzero.
    assert np.linalg.norm(_total_force(grd)) > 1e-3
    assert np.linalg.norm(_total_torque(grd, coords, ref)) > 1e-3

    grd_p, err_p = _project_force_sum_rules(grd, err, coords, ref)

    assert np.allclose(_total_force(grd_p), 0.0, atol=1e-12)
    # the torque is not a constraint
    assert np.linalg.norm(_total_torque(grd_p, coords, ref)) > 1e-3


@pytest.mark.parametrize('weighting', ['diagonal', 'full'])
def test_zero_net_force_with_torque_unchanged(weighting):
    # A field with sum F = 0 and a sizeable torque is consistent
    # with the only constraint, so it passes through unchanged, and
    # so do its errors when the correlations make every sample obey
    # the sum rule.
    coords = _bent_trimer()
    ref = coords.mean(axis=0)
    rng = np.random.default_rng(11)
    q = _sum_rule_null_projector(coords, ref, rotations=False)
    series = rng.normal(size=(400, coords.size)) @ q + 0.02 * (
        q @ rng.normal(size=coords.size))
    grd = series.mean(axis=0).reshape(coords.shape)
    err = (series.std(axis=0, ddof=1)
           / np.sqrt(len(series))).reshape(coords.shape)
    assert np.abs(_total_force(grd)).max() < 1e-12
    tau = _total_torque(grd, coords, ref)
    assert np.linalg.norm(tau) > 1e-2
    kw = ({} if weighting == 'diagonal'
          else {'corr': _block_correlation(series, 1),
                'n_bins': len(series)})
    grd_p, err_p, info = _project_force_sum_rules(
        grd, err, coords, ref, return_info=True, **kw,
    )
    np.testing.assert_allclose(grd_p, grd, atol=1e-14)
    np.testing.assert_allclose(
        _total_torque(grd_p, coords, ref), tau, rtol=1e-12,
    )
    if weighting == 'full':
        # corr is singular along sum F (zero variance): diagonal W,
        # but V keeps every error
        assert 'not positive definite' in info['weighting']
        np.testing.assert_allclose(err_p, err, rtol=1e-8)
        assert np.all(info['net_err'][1] > 0)


def test_torque_reference_independent_after_projection():
    # Once sum F = 0 is enforced, the total torque is the same about
    # any reference point, and so is its error.
    from OmegaQMC.observables.force import _net_errors_from_cov
    coords = _bent_trimer()
    ref = coords.mean(axis=0)
    rng = np.random.default_rng(21)
    grd = rng.normal(scale=0.01, size=coords.shape)
    err = rng.uniform(0.001, 0.005, size=coords.shape)

    grd_p, _, info = _project_force_sum_rules(
        grd, err, coords, ref, return_info=True,
    )

    other = np.array([3.3, -1.7, 2.9])
    np.testing.assert_allclose(
        _total_torque(grd_p, coords, other),
        _total_torque(grd_p, coords, ref), rtol=1e-10,
    )
    np.testing.assert_allclose(
        _net_errors_from_cov(info['cov'], coords, other)[1],
        info['net_err'][1], rtol=1e-6,
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

    grd_p, err_p, info = _project_force_sum_rules(
        grd, err, coords, ref, return_info=True,
    )

    assert np.all(np.isfinite(grd_p))
    assert np.all(np.isfinite(err_p))
    assert np.allclose(_total_force(grd_p), 0.0, atol=1e-12)
    # The torque about x, y is left alone; the z component is
    # identically zero for any collinear-along-z field, with a zero
    # error.
    tau = _total_torque(grd_p, coords, ref)
    assert np.all(np.abs(tau[:2]) > 1e-4)
    assert tau[2] == 0.0
    net = info['net_err']
    assert np.all(net[0] < 1e-15) and np.all(net[1, :2] > 0)
    assert net[1, 2] == 0.0


def _sum_rule_null_projector(coords, ref, rotations=True):
    """Orthogonal projector onto fields with zero net force and, with
    *rotations*, zero torque about *ref*."""
    rel = coords - ref
    rows = []
    for k in range(3):
        t = np.zeros(coords.shape)
        t[:, k] = 1.0
        rows.append(t.ravel())
        if rotations:
            r = np.cross(rel, np.eye(3)[k]).ravel()   # d tau_k / d F
            rows.append(r)
    c = np.array(rows)
    return np.eye(c.shape[1]) - np.linalg.pinv(c) @ c


def test_correlated_errors_kept_for_consistent_samples():
    # Every block obeys the translational sum rule exactly, as each
    # SWCT sample does, so the components are strongly correlated and
    # the mean is already consistent: projecting must change neither
    # the mean nor, given the correlations, the error.
    coords = _bent_trimer()
    ref = coords.mean(axis=0)
    rng = np.random.default_rng(5)
    q = _sum_rule_null_projector(coords, ref, rotations=False)
    series = rng.normal(size=(400, coords.size)) @ q   # (T, 3N)
    grd = series.mean(axis=0).reshape(coords.shape)
    err = (series.std(axis=0, ddof=1)
           / np.sqrt(len(series))).reshape(coords.shape)
    corr = _block_correlation(series, 1)

    grd_c, err_c = _project_force_sum_rules(
        grd, err, coords, ref, corr=corr,
    )
    np.testing.assert_allclose(grd_c, grd, atol=1e-12)
    np.testing.assert_allclose(err_c, err, rtol=1e-8)

    # The independent propagation understates these errors.
    grd_d, err_d = _project_force_sum_rules(grd, err, coords, ref)
    np.testing.assert_allclose(grd_d, grd_c, atol=1e-12)
    assert np.all(err_d <= err + 1e-12)
    # one constraint per direction over 6 atoms of similar error:
    # ~ sqrt(5 / 6) = 0.91
    assert np.mean(err_d / err) < 0.95


def test_block_correlation_zero_variance_and_few_bins():
    rng = np.random.default_rng(6)
    series = np.column_stack([
        rng.normal(size=20), np.zeros(20), rng.normal(size=20),
    ])
    corr = _block_correlation(series, 2)
    assert corr.shape == (3, 3)
    np.testing.assert_allclose(np.diag(corr), 1.0)
    assert corr[1, 0] == corr[1, 2] == 0.0
    assert _block_correlation(series, 11) is None


def _project_w(*args, **kwargs):
    """Projection, with the weighting description in place of info."""
    grd_p, err_p, info = _project_force_sum_rules(*args, **kwargs)
    return grd_p, err_p, info['weighting']


def _correlated_case(seed=0, n_blocks=400):
    """Generic correlated block series of a 6-atom field."""
    coords = _bent_trimer()
    rng = np.random.default_rng(seed)
    m = coords.size
    mix = rng.normal(size=(m, m)) * rng.uniform(0.2, 2.0, size=m)
    series = rng.normal(size=(n_blocks, m)) @ mix + 0.01
    grd = series.mean(axis=0).reshape(coords.shape)
    err = (series.std(axis=0, ddof=1)
           / np.sqrt(n_blocks)).reshape(coords.shape)
    return coords, coords.mean(axis=0), grd, err, series


@pytest.mark.parametrize('seed', range(4))
def test_full_covariance_weighting_never_inflates_errors(seed):
    coords, ref, grd, err, series = _correlated_case(seed)
    corr = _block_correlation(series, 1)
    grd_f, err_f, how_f = _project_w(
        grd, err, coords, ref, corr=corr, n_bins=len(series),
        return_info=True,
    )
    assert how_f.startswith('full covariance')
    assert np.all(err_f <= err * (1.0 + 1e-10))
    assert np.allclose(_total_force(grd_f), 0.0, atol=1e-12)
    assert np.linalg.norm(_total_torque(grd_f, coords, ref)) > 1e-3

    # The diagonal weighting propagated with the same correlations
    # inflates some errors and has the larger chi^2 under V.
    grd_d, err_d, how_d = _project_w(
        grd, err, coords, ref, corr=corr, n_bins=50,
        return_info=True,
    )
    assert how_d.startswith('diagonal (only 50 bins')
    assert np.any(err_d > err)
    v_inv = np.linalg.inv(err.ravel()[:, None] * corr * err.ravel())

    def chi2(g):
        r = (g - grd).ravel()
        return r @ v_inv @ r

    assert chi2(grd_f) <= chi2(grd_d)


def test_full_covariance_falls_back_to_diagonal():
    coords, ref, grd, err, series = _correlated_case(0)
    corr = _block_correlation(series, 1)
    grd_0, err_0 = _project_force_sum_rules(
        grd, err, coords, ref, corr=corr,
    )
    # too few bins (5 per component needed), or no bin count: the
    # diagonal weighting, unchanged from before
    for n_bins, reason in [(89, 'only 89 bins for 18 components'),
                           (None, 'number of bins unknown')]:
        grd_p, err_p, how = _project_w(
            grd, err, coords, ref, corr=corr, n_bins=n_bins,
            return_info=True,
        )
        assert how.startswith('diagonal') and reason in how
        np.testing.assert_array_equal(grd_p, grd_0)
        np.testing.assert_array_equal(err_p, err_0)
    _, _, how = _project_w(
        grd, err, coords, ref, corr=corr, n_bins=90,
        return_info=True,
    )
    assert how.startswith('full covariance')

    # a singular correlation (every block obeys the sum rule)
    q = _sum_rule_null_projector(coords, ref, rotations=False)
    series = np.random.default_rng(5).normal(size=(400, 18)) @ q
    corr = _block_correlation(series, 1)
    grd_p, err_p, how = _project_w(
        grd, err, coords, ref, corr=corr, n_bins=400,
        return_info=True,
    )
    assert 'not positive definite' in how
    grd_d, _ = _project_force_sum_rules(grd, err, coords, ref)
    np.testing.assert_array_equal(grd_p, grd_d)

    # no correlations at all
    _, _, how = _project_w(
        grd, err, coords, ref, return_info=True,
    )
    assert how == 'diagonal (no correlations available)'


def test_net_force_error_vanishes_torque_error_kept():
    from OmegaQMC.observables.force import _net_errors_from_cov
    coords, ref, grd, err, series = _correlated_case(1)
    corr = _block_correlation(series, 1)
    for n_bins in (len(series), 10):           # full and diagonal W
        grd_p, _, info = _project_force_sum_rules(
            grd, err, coords, ref, corr=corr, n_bins=n_bins,
            return_info=True,
        )
        net = info['net_err']
        assert net.shape == (2, 3)
        assert np.all(net[0] < 1e-13 * err.max())
        # the torque keeps the error propagated through
        # (I - P) V (I - P)^T, which the covariance reproduces (the
        # net force up to the square root of its round-off)
        assert np.all(net[1] > 1e-3 * err.max())
        net_c = _net_errors_from_cov(info['cov'], coords, ref)
        assert np.all(net_c[0] < 1e-7 * err.max())
        np.testing.assert_allclose(net_c[1], net[1], rtol=1e-8)
        # ... and the projected per-block series: the projection is a
        # fixed linear map of the series
        if n_bins == 10:
            continue
        sd = err.ravel()
        v = sd[:, None] * corr * sd
        c = np.zeros((3, coords.size))
        for k in range(3):
            c[k, k::3] = 1.0
        p = v @ c.T @ np.linalg.pinv(c @ v @ c.T) @ c
        proj = series @ (np.eye(coords.size) - p).T
        tau_t = np.cross(coords - ref,
                         proj.reshape(len(series), -1, 3)).sum(axis=1)
        np.testing.assert_allclose(
            net[1], tau_t.std(axis=0, ddof=1) / np.sqrt(len(series)),
            rtol=1e-8,
        )
        np.testing.assert_allclose(
            grd_p, proj.mean(axis=0).reshape(coords.shape),
            rtol=1e-8, atol=1e-12,
        )
    # unprojected: the error of the net series itself
    v = err.ravel()[:, None] * corr * err.ravel()
    net = _net_errors_from_cov(v, coords, ref)
    tot = series.reshape(len(series), -1, 3).sum(axis=1)
    np.testing.assert_allclose(
        net[0], tot.std(axis=0, ddof=1) / np.sqrt(len(series)),
        rtol=1e-10,
    )


def test_torque_note_warns_only_beyond_nsigma():
    import warnings
    from OmegaQMC.observables.force import TORQUE_NSIGMA, _torque_note
    assert TORQUE_NSIGMA == 4.0
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        # within 4 sigma, or a zero error (axial torque of a linear
        # molecule): no note, no warning
        assert _torque_note('x', [0.039, -0.01, 1e-17],
                            [0.01, 0.01, 0.0]) is None
    with pytest.warns(UserWarning,
                      match='not stationary w.r.t. rotation') as rec:
        note = _torque_note('reference state', [0.05, 0.0, 0.0],
                            [0.01, 0.01, 0.01])
    assert note == str(rec[0].message)
    assert note.startswith(
        'Note: total torque of the reference state is 5.0 sigma')
    assert 'diagnostic, not a constraint' in note


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, '-q']))
