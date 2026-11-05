"""Tests for the local-energy clipping of the KFAC optimiser.

:func:`OmegaQMC.vmcopt_nn_kfac.clip_local_energies` must reproduce
FermiNet's ``loss.clip_local_values`` (window ``c +- scale * mean|E_L -
c|``, ``c`` the batch mean or median; residuals centred at the mean of
the clipped or of the unclipped energies), and the driver must feed
the clipped residuals to the KFAC step only while logging unclipped
energies.  Clipping off (the default) must leave the residuals
untouched.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from OmegaQMC.utils import Mole_custom  # noqa: E402
from OmegaQMC.vmcopt_nn_kfac import (  # noqa: E402
    clip_local_energies,
    get_vmcopt_nn_func,
)


def _ferminet_reference(e, scale, median, center_at_clip):
    """NumPy transcription of ferminet.loss.clip_local_values."""
    mean = e.mean()
    c = np.median(e) if median else mean
    tv = np.abs(e - c).mean()
    ec = np.clip(e, c - scale * tv, c + scale * tv)
    return ec, ec - (ec.mean() if center_at_clip else mean)


@pytest.fixture(scope="module")
def energies():
    rng = np.random.default_rng(0)
    e = -1.17 + 0.05 * rng.standard_normal(1000)
    e[3] = 40.0               # outliers on both sides, as near nodes
    e[17] = -25.0
    e[500] = -3.0
    return e


@pytest.mark.parametrize("median", [False, True])
@pytest.mark.parametrize("center_at_clip", [False, True])
@pytest.mark.parametrize("scale", [5.0, 2.0])
def test_matches_ferminet(energies, median, center_at_clip, scale):
    ec, de = clip_local_energies(jnp.asarray(energies), scale,
                                 clip_median=median,
                                 center_at_clip=center_at_clip)
    ec_ref, de_ref = _ferminet_reference(energies, scale, median,
                                         center_at_clip)
    np.testing.assert_allclose(np.asarray(ec), ec_ref, rtol=0,
                               atol=1e-12)
    np.testing.assert_allclose(np.asarray(de), de_ref, rtol=0,
                               atol=1e-12)
    # the outliers are the ones clipped
    clipped = np.flatnonzero(np.asarray(ec) != energies)
    assert {3, 17, 500} <= set(clipped)


def test_center_at_clip_residuals_sum_to_zero(energies):
    _, de = clip_local_energies(jnp.asarray(energies), 5.0)
    assert abs(float(jnp.mean(de))) < 1e-13
    _, de = clip_local_energies(jnp.asarray(energies), 5.0,
                                center_at_clip=False)
    assert abs(float(jnp.mean(de))) > 1e-3


def test_no_outliers_unchanged():
    e = jnp.linspace(-1.2, -1.1, 64)
    ec, de = clip_local_energies(e, 5.0)
    np.testing.assert_array_equal(np.asarray(ec), np.asarray(e))
    np.testing.assert_allclose(np.asarray(de),
                               np.asarray(e - jnp.mean(e)), atol=1e-15)


# --- driver ---------------------------------------------------------

def _h2():
    return Mole_custom.from_arrays(
        charges=[1, 1], coords=[[0, 0, -0.7], [0, 0, 0.7]],
        n_up=1, n_down=1,
    )


@pytest.fixture(scope="module")
def drv():
    return get_vmcopt_nn_func(_h2(), 'ferminet', jax.random.key(0))


def _run_with_outlier(drv, tmp_path, monkeypatch, name):
    """Two iterations with walker 0's E_L shifted by +1000 Ha.

    Returns the E_L batches, the residuals handed to the KFAC step,
    the driver output and the log text.
    """
    orig_e, orig_apply = drv.compute_batch_energy, drv._kfac_apply
    seen = {'e': [], 'de': []}

    def with_outlier(walkers, params):
        e = orig_e(walkers, params).at[0].add(1000.0)
        seen['e'].append(np.asarray(e))
        return e

    def spy(params, A_hat, G_hat, dW_hat, generic_flat, de_eff, *rest):
        seen['de'].append(np.asarray(de_eff))
        return orig_apply(params, A_hat, G_hat, dW_hat, generic_flat,
                          de_eff, *rest)

    monkeypatch.setattr(drv, 'compute_batch_energy', with_outlier)
    monkeypatch.setattr(drv, '_kfac_apply', spy)
    log = str(tmp_path / f'{name}.log')
    out = drv(jax.random.key(1), num_iters=2, num_walkers=32,
              num_blocks_equil=1, num_steps_per_block=5,
              factor_chunk_size=16, prefix=str(tmp_path / name),
              verbose=1, fname_log=log)
    with open(log) as f:
        return seen, out, f.read()


def test_driver_default_is_unclipped(drv, tmp_path, monkeypatch):
    assert drv.clip_local_energy == 0.0
    seen, (_, edata), log = _run_with_outlier(drv, tmp_path,
                                              monkeypatch, 'off')
    for e, de in zip(seen['e'], seen['de']):
        np.testing.assert_allclose(de, e - e.mean(), atol=1e-10)
    assert edata['n_clipped_history'] == []
    assert '# clip_local_energy=0 (off)' in log


@pytest.mark.parametrize("median", [False, True])
def test_driver_clips_gradient_only(drv, tmp_path, monkeypatch, median):
    monkeypatch.setattr(drv, 'clip_local_energy', 5.0)
    monkeypatch.setattr(drv, 'clip_median', median)
    seen, (params, edata), log = _run_with_outlier(
        drv, tmp_path, monkeypatch, f'on{int(median)}')
    for e, de in zip(seen['e'], seen['de']):
        _, de_ref = _ferminet_reference(e, 5.0, median, True)
        np.testing.assert_allclose(de, de_ref, atol=1e-10)
        assert de[0] < 0.5 * (e[0] - e.mean())     # the outlier is cut
    # logged energies are the unclipped batch means (seen['e'] also
    # holds the final estimate after the loop)
    assert len(seen['de']) == 2
    np.testing.assert_allclose(edata['E_history'],
                               [e.mean() for e in seen['e'][:2]],
                               rtol=1e-12)
    assert len(edata['n_clipped_history']) == 2
    assert all(n >= 1 for n in edata['n_clipped_history'])
    assert all(bool(jnp.all(jnp.isfinite(x)))
               for x in jax.tree.leaves(params))
    centre = 'median' if median else 'mean'
    assert (f'# clip_local_energy=5 (window centred at the {centre},'
            ' center_at_clip=True)') in log
