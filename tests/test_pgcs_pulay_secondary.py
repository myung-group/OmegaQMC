"""Pulay term of PGCS secondary states in ``postproc_h5_pgcs``.

A secondary state's gradient terms are evaluated at the transformed
configuration T R, so its Pulay term must use E_L(T R), which both
gradient writers store as ``local_energies/<label>/<blk>``; E_V stays
the reference mean.  On a synthetic gradient file these tests check

* that the printed Pulay gradients of the reference and the secondary
  state equal 2 <(E_L - E_V) dlnpsi/dR> with E_L(R) and E_L(T R)
  respectively, where the two energies differ;
* that a file without ``local_energies/<label>`` warns and falls back
  to the reference E_L(R);
* the correlated-sampling diagnostics printed for the secondary state
  (mean weight, Kish ESS/N, energy offset) and their warnings.
"""
import re
import warnings

import h5py
import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from OmegaQMC.observables.force import postproc_h5_pgcs  # noqa: E402

N_STEPS, N_WALKERS, N_NUC, N_BLOCKS = 4, 8, 2, 3
LABEL = '1:Rz180'
GRD_KEYS = ('grd_ee_en', 'grd_ke', 'grd_logpsi')


def _write_file(path, with_secondary_el=True, seed=0, exact=False,
                weights=None):
    """H2 gradient file with one secondary state.

    The secondary state shares the reference gradients and has unit
    weights, so its Pulay term differs from the reference one only
    through the local energies.  With *exact* E_L(T R) = E_L(R), as
    for an exact symmetry.  *weights(rng, n)* replaces the unit
    weights.  Returns the reference and secondary local energies and
    dlnpsi/dR, flattened over blocks.
    """
    rng = np.random.default_rng(seed)
    el_ref, el_sec, logpsi = [], [], []
    with h5py.File(path, 'w') as f:
        g = f.create_group('system')
        g['atom_symbols'] = np.bytes_('H H')
        g['atom_coords'] = np.array([[0., 0., -0.7], [0., 0., 0.7]])
        g['units'] = np.bytes_('Bohr')
        f['grd_nn'] = rng.normal(size=(N_NUC, 3))
        for grp in ('local_energies', 'fragment_weights') + GRD_KEYS:
            f.create_group(grp)
            if grp != 'local_energies' or with_secondary_el:
                f[grp].create_group(LABEL)
        for b in range(N_BLOCKS):
            bcs = f'{b}'
            el = rng.normal(size=(N_STEPS, N_WALKERS)) - 1.1
            el_t = el.ravel() + rng.normal(0.5, 0.2, size=el.size)
            if exact:
                el_t = el.ravel().copy()
            f['local_energies'][bcs] = el
            if with_secondary_el:
                f['local_energies'][LABEL][bcs] = el_t
            for grp in GRD_KEYS:
                x = rng.normal(size=(N_STEPS * N_WALKERS, N_NUC, 3))
                f[grp][bcs] = x
                f[grp][LABEL][bcs] = x
                if grp == 'grd_logpsi':
                    logpsi.append(x)
            f['fragment_weights'][LABEL][bcs] = (
                np.ones(el.size) if weights is None
                else weights(rng, el.size)
            )
            el_ref.append(el.ravel())
            el_sec.append(el_t)
    return (np.concatenate(el_ref), np.concatenate(el_sec),
            np.concatenate(logpsi))


def _pulay(el, e_v, logpsi):
    return 2.0 * np.einsum('s,snk->nk', el - e_v, logpsi) / len(el)


def _printed_pulay(log_text):
    """Pulay gradients of each state, in printed order."""
    out = []
    for m in re.finditer(
        r'Pulay gradients\n(.*?)\nTotal gradients', log_text, re.S,
    ):
        nums = re.findall(r'-?\d+\.\d*(?:e[-+]?\d+)?', m.group(1))
        out.append(np.array(nums, dtype=float).reshape(N_NUC, 3))
    return out


def _postproc(tmp_path, **kw):
    pre = str(tmp_path / 'syn')
    data = _write_file(pre + '.grd.h5', **kw)
    log = str(tmp_path / 'syn.log')
    postproc_h5_pgcs(prefix=pre, logfile=log)
    with open(log) as fh:
        return data, fh.read()


def _run(tmp_path, **kw):
    data, text = _postproc(tmp_path, **kw)
    printed = _printed_pulay(text)
    assert len(printed) == 2
    return data, printed


def _cs_diagnostics(text):
    """Diagnostic values printed for the secondary state."""
    num = r'(-?[\d.]+(?:e[-+]?\d+)?)'
    vals = {}
    for key, pat in [
        ('mean_w', r'mean w = ' + num),
        ('ess', r'ESS/N = ' + num),
        ('offset', r'- E_V = ' + num),
        ('serr', r'- E_V = \S+ ± ' + num),
        ('dmax', r'max \|E_L\(TR\) - E_L\(R\)\| = ' + num),
    ]:
        m = re.search(pat, text)
        vals[key] = float(m.group(1)) if m else None
    return vals


def _cs_warnings(caught):
    return [str(w.message) for w in caught
            if str(w.message).startswith(f'PGCS state {LABEL}')]


def test_secondary_pulay_uses_transformed_energies(tmp_path):
    (el_ref, el_sec, logpsi), (p_ref, p_sec) = _run(tmp_path)
    e_v = el_ref.mean()
    np.testing.assert_allclose(
        p_ref, _pulay(el_ref, e_v, logpsi), atol=1e-10,
    )
    np.testing.assert_allclose(
        p_sec, _pulay(el_sec, e_v, logpsi), atol=1e-10,
    )
    # E_L(T R) - E_L(R) ~ 0.5 must show up in the secondary state
    assert np.abs(p_sec - p_ref).max() > 1e-2


def test_missing_secondary_energies_fall_back(tmp_path):
    with pytest.warns(UserWarning, match=f'local_energies/{LABEL}'):
        (el_ref, _, logpsi), (p_ref, p_sec) = _run(
            tmp_path, with_secondary_el=False,
        )
    expected = _pulay(el_ref, el_ref.mean(), logpsi)
    np.testing.assert_allclose(p_ref, expected, atol=1e-10)
    np.testing.assert_allclose(p_sec, expected, atol=1e-10)


def test_cs_diagnostics_exact_symmetry(tmp_path):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        _, text = _postproc(tmp_path, exact=True)
    d = _cs_diagnostics(text)
    assert d['mean_w'] == 1.0 and d['ess'] == 1.0 and d['dmax'] == 0.0
    assert abs(d['offset']) < 1e-12 and d['serr'] > 0.0
    assert _cs_warnings(caught) == []


def test_cs_diagnostics_offset_and_ess_warn(tmp_path):
    def skewed(rng, n):
        w = np.full(n, 1e-3)
        w[0] = 1.0                                # one dominant sample
        return w

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        (el_ref, el_sec, _), text = _postproc(tmp_path)
    d = _cs_diagnostics(text)
    # unit weights: offset = <E_L(TR)> - <E_L(R)>, about 0.5 Ha
    np.testing.assert_allclose(
        d['offset'], el_sec.mean() - el_ref.mean(), rtol=1e-5,
    )
    msgs = _cs_warnings(caught)
    assert d['ess'] == 1.0
    assert len(msgs) == 1 and 'standard errors from zero' in msgs[0]

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        _, text = _postproc(tmp_path, weights=skewed)
    assert _cs_diagnostics(text)['ess'] < 0.1
    assert any('Kish ESS/N' in m for m in _cs_warnings(caught))


def test_cs_diagnostics_without_secondary_energies(tmp_path):
    with warnings.catch_warnings(record=True):
        warnings.simplefilter('always')
        _, text = _postproc(tmp_path, with_secondary_el=False)
    d = _cs_diagnostics(text)
    assert d['ess'] == 1.0 and d['offset'] is None
    assert f'local_energies/{LABEL} absent' in text
