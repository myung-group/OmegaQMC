"""Tests for the non-finite guard of the KFAC optimiser.

A walker with a non-finite local energy, or a step that comes out
non-finite, must not poison the parameters: the guard replaces such
walkers by copies of finite ones, rejects non-finite steps, records the
events in ``{prefix}.nonfinite.h5`` and gives up after too many
consecutive rejections.  The faults are injected by wrapping the
driver's compiled kernels.
"""
import h5py
import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

import OmegaQMC.vmcopt_nn_kfac as kfac_mod  # noqa: E402
from OmegaQMC.utils import Mole_custom  # noqa: E402
from OmegaQMC.vmcopt_nn_kfac import get_vmcopt_nn_func  # noqa: E402


def _h2():
    return Mole_custom.from_arrays(
        charges=[1, 1], coords=[[0, 0, -0.7], [0, 0, 0.7]],
        n_up=1, n_down=1,
    )


@pytest.fixture(scope="module")
def drv():
    return get_vmcopt_nn_func(_h2(), 'ferminet', jax.random.key(0))


def _run(drv, tmp_path, num_iters, name):
    prefix = str(tmp_path / name)
    log = str(tmp_path / f'{name}.log')
    out = drv(jax.random.key(1), num_iters=num_iters, num_walkers=32,
              num_blocks_equil=1, num_steps_per_block=5,
              factor_chunk_size=16, prefix=prefix, verbose=1,
              fname_log=log)
    with open(log) as f:
        return out, prefix, f.read()


def _all_finite(tree):
    return all(bool(jnp.all(jnp.isfinite(x)))
               for x in jax.tree.leaves(tree))


def test_nonfinite_local_energy_walkers_replaced(drv, tmp_path,
                                                 monkeypatch):
    orig = drv.compute_batch_energy
    calls = {'n': 0}

    def faulty(walkers, params):
        e = orig(walkers, params)
        calls['n'] += 1
        if calls['n'] == 2:                    # second iteration
            e = e.at[:3].set(jnp.nan).at[3].set(jnp.inf)
        return e

    monkeypatch.setattr(drv, 'compute_batch_energy', faulty)
    (params, edata), prefix, log = _run(drv, tmp_path, 3, 'eloc')
    assert _all_finite(params)
    assert np.all(np.isfinite(edata['E_history']))
    assert '4 walker(s) with a non-finite local energy replaced' in log
    with h5py.File(prefix + '.nonfinite.h5') as f:
        (key,) = list(f)
        assert f[key].attrs['kind'] == 'e_loc'
        assert f[key].attrs['iteration'] == 1
        assert f[key]['walkers'].shape == (4, 2, 3)


def test_nonfinite_step_rejected(drv, tmp_path, monkeypatch):
    orig = drv._kfac_apply
    calls = {'n': 0}
    kept = {}

    def faulty(params, *args):
        out = orig(params, *args)
        calls['n'] += 1
        if calls['n'] == 2:
            kept['params'] = params
            bad = jax.tree.map(lambda x: x * jnp.nan, out[0])
            return (bad,) + tuple(out[1:])
        if calls['n'] == 3:
            kept['next_in'] = params
        return out

    monkeypatch.setattr(drv, '_kfac_apply', faulty)
    (params, _), prefix, log = _run(drv, tmp_path, 3, 'step')
    assert _all_finite(params)
    assert 'non-finite KFAC step rejected' in log
    # The rejected iteration left the parameters unchanged.
    for a, b in zip(jax.tree.leaves(kept['params']),
                    jax.tree.leaves(kept['next_in'])):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
    with h5py.File(prefix + '.nonfinite.h5') as f:
        (key,) = list(f)
        assert f[key].attrs['kind'] == 'step'
        assert f[key]['grad_r_finite'].shape == (32,)


def test_too_many_rejections_raise(drv, tmp_path, monkeypatch):
    orig = drv._kfac_apply

    def always_bad(params, *args):
        out = orig(params, *args)
        return (jax.tree.map(lambda x: x * jnp.nan, out[0]),) \
            + tuple(out[1:])

    monkeypatch.setattr(drv, '_kfac_apply', always_bad)
    monkeypatch.setattr(kfac_mod, '_NONFINITE_MAX_CONSECUTIVE', 3)
    with pytest.raises(FloatingPointError):
        _run(drv, tmp_path, 10, 'raise')


def test_mostly_nonfinite_raises(drv, tmp_path, monkeypatch):
    orig = drv.compute_batch_energy

    def faulty(walkers, params):
        e = orig(walkers, params)
        return e.at[: e.shape[0] * 3 // 4].set(jnp.nan)

    monkeypatch.setattr(drv, 'compute_batch_energy', faulty)
    with pytest.raises(FloatingPointError):
        _run(drv, tmp_path, 2, 'mostly')
