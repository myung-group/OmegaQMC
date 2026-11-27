"""Error of the fragment-wise BLUE combine for correlated states.

PGCS states are images of the same samples, so their block series can
be identical (e.g. a force component every operation leaves unchanged).
The combined error must then equal the single-series error, not
err / sqrt(K) as an independence assumption would give.  With few bins
:func:`OmegaQMC.utils.blue_combine_states` falls back to fixed weights,
but it propagates them through the full bin-mean covariance, so the
combined error is still exact here.
"""
import numpy as np
import pytest

from OmegaQMC.utils import blue_combine_states


@pytest.mark.parametrize('K', [2, 3, 4])
def test_identical_series_keep_single_series_error(K):
    rng = np.random.default_rng(1)
    series = rng.normal(size=10)                  # 10 blocks, kappa 1
    labels = list(range(K))
    kappa = {k: 1.0 for k in labels}
    _, err_1, _, _, bin_1 = blue_combine_states(
        {0: series}, [0], kappa_per_label={0: 1.0},
    )
    with pytest.warns(UserWarning, match='falling back'):
        mean, err, _, w, bin_k = blue_combine_states(
            {k: series for k in labels}, labels, kappa_per_label=kappa,
        )
    assert bin_k == bin_1 == 2
    np.testing.assert_allclose(w.sum(), 1.0)
    np.testing.assert_allclose(mean, series.mean())
    np.testing.assert_allclose(err, err_1, rtol=1e-12)
    bins = series.reshape(5, 2).mean(axis=1)
    np.testing.assert_allclose(err_1, bins.std(ddof=1) / np.sqrt(5))
