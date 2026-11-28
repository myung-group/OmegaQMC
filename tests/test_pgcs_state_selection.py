"""Low-ESS PGCS states are dropped from the BLUE, alike for E and F.

:func:`OmegaQMC.utils.select_cs_states` decides from the fragment
weights alone which PGCS states enter the BLUE, and both
``postproc_h5_pgcs`` functions (energy and force) use it on the same
blocks.  On a synthetic two-fragment gradient file, with one secondary
state of near-unit weights and one dominated by a single sample per
block, these tests check

* that the energy and force logs print identical selection lines, and
  the force diagnostics the same ESS/N, for several cutoffs;
* that the reference state is never dropped, and that dropping every
  secondary state gives the reference alone without raising;
* that ``ess_min = 0`` (or ``None``) combines every state, the energy
  BLUE then equalling a direct :func:`blue_combine_states` over all;
* that an atom whose fragment lost its secondary state is averaged
  over the reference alone;
* that with every fragment reference-only the translational sum-rule
  projection is skipped by default (``project_sum_rules="auto"``), the
  averaged rows being the reference rows, while ``True`` still
  projects;
* that net-force errors come from their own block series, and that after
  the projection the averaged net force is 0 +- 0 while the torque
  keeps a finite error;
* that the energy log always ends with the reported energy.
"""
import re
import warnings

import h5py
import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from OmegaQMC.observables import energy, force  # noqa: E402
from OmegaQMC.utils import (  # noqa: E402
    CS_ESS_MIN,
    blue_combine_states,
    do_binning_analysis,
    select_cs_states,
)

N_STEPS, N_WALKERS, N_NUC, N_BLOCKS = 4, 8, 2, 24
GOOD = '1:E,2:Rz180'          # secondary state of fragment 2 (atom 1)
BAD = '1:Rz180,2:E'           # secondary state of fragment 1 (atom 0)
LABELS = (GOOD, BAD)
GRD_KEYS = ('grd_ee_en', 'grd_ke', 'grd_logpsi')


def _write_file(path, seed=0):
    """H2 gradient file, one fragment per atom, two secondary states."""
    rng = np.random.default_rng(seed)
    n = N_STEPS * N_WALKERS
    with h5py.File(path, 'w') as f:
        g = f.create_group('system')
        g['atom_symbols'] = np.bytes_('H H')
        g['atom_coords'] = np.array([[0., 0., -0.7], [0., 0., 0.7]])
        g['units'] = np.bytes_('Bohr')
        g['atom_fragment_map'] = np.array([1, 2])
        f['grd_nn'] = rng.normal(size=(N_NUC, 3))
        for grp in ('local_energies', 'fragment_weights') + GRD_KEYS:
            f.create_group(grp)
            for label in LABELS:
                f[grp].create_group(label)
        for b in range(N_BLOCKS):
            bcs = f'{b}'
            # slow drift, so that equil_cutoff='auto' has work to do
            el = (rng.normal(size=(N_STEPS, N_WALKERS)) - 1.1
                  + 0.5 * np.exp(-b / 3.0))
            f['local_energies'][bcs] = el
            for grp in GRD_KEYS:
                x = rng.normal(size=(n, N_NUC, 3))
                f[grp][bcs] = x
                for label in LABELS:
                    f[grp][label][bcs] = (
                        x + 0.1 * rng.normal(size=x.shape)
                    )
            w_good = np.exp(0.1 * rng.normal(size=n))
            w_bad = np.full(n, 1e-3)
            w_bad[rng.integers(n)] = 1.0      # one dominant sample
            for label, w in ((GOOD, w_good), (BAD, w_bad)):
                f['fragment_weights'][label][bcs] = w
                f['local_energies'][label][bcs] = (
                    el.ravel() + 0.05 * rng.normal(size=n)
                )


def _run(tmp_path, **kw):
    """Energy and force post-processing of the same file."""
    pre = str(tmp_path / 'syn')
    if not (tmp_path / 'syn.grd.h5').exists():
        _write_file(pre + '.grd.h5')
    out = {}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        for name, mod in (('energy', energy), ('force', force)):
            log = str(tmp_path / f'{name}.log')
            kw_m = {k: v for k, v in kw.items()
                    if name == 'force' or k != 'project_sum_rules'}
            res = mod.postproc_h5_pgcs(prefix=pre, logfile=log, **kw_m)
            with open(log) as fh:
                out[name] = (res, fh.read())
    return out, [str(w.message) for w in caught]


def _selection_lines(text):
    return [
        ln for ln in text.splitlines()
        if ln.startswith(('PGCS state selection', 'BLUE:'))
        or re.match(r'^  \S+\s+ESS/N = ', ln)
    ]


def _averaged_states(text):
    """Number of states of each averaged force row, by atom."""
    return [int(m) for m in re.findall(r'over (\d+) states\)', text)]


@pytest.mark.parametrize('equil_cutoff', [0, 5, 'auto'])
def test_energy_and_force_drop_the_same_states(tmp_path, equil_cutoff):
    out, msgs = _run(tmp_path, equil_cutoff=equil_cutoff)
    sel_e = _selection_lines(out['energy'][1])
    sel_f = _selection_lines(out['force'][1])
    assert sel_e == sel_f and len(sel_e) == 4
    assert sel_e[-1] == f'BLUE: dropped states {BAD} (ESS/N < 0.1)'
    assert 'kept' in sel_e[1] and GOOD in sel_e[1]
    assert 'dropped' in sel_e[2] and BAD in sel_e[2]
    # the force diagnostics print the selection's ESS/N
    ess_sel = [float(re.search(r'ESS/N = (\S+)', ln).group(1))
               for ln in sel_e[1:3]]
    ess_diag = [float(x) for x in re.findall(
        r'Correlated sampling: .* ESS/N = (\S+)', out['force'][1],
    )]
    np.testing.assert_allclose(ess_diag, ess_sel, rtol=1e-3)
    assert ess_sel[0] > 0.9 and ess_sel[1] < CS_ESS_MIN
    assert any(m.startswith(f'PGCS state {BAD}: Kish ESS/N')
               and 'dropped' in m for m in msgs)
    assert not any(m.startswith(f'PGCS state {GOOD}') for m in msgs)
    # fragment 1 (atom 0) lost its only secondary state
    assert _averaged_states(out['force'][1]) == [1, 2]
    assert 'over 2 states)' in out['energy'][1]


def test_reference_never_dropped(tmp_path):
    path = str(tmp_path / 'syn.grd.h5')
    _write_file(path)
    with pytest.warns(UserWarning, match='dropped from the BLUE'):
        kept, stats = select_cs_states(
            path, [None, GOOD, BAD], range(N_BLOCKS), ess_min=2.0,
        )
    assert kept == [None] and set(stats) == {GOOD, BAD}
    with h5py.File(path, 'r') as f:
        kept, _ = select_cs_states(
            f, [None, GOOD, BAD], range(N_BLOCKS), ess_min=0,
        )
    assert kept == [None, GOOD, BAD]


def test_all_dropped_gives_reference_only(tmp_path):
    out, _ = _run(tmp_path, ess_min=2.0)
    for name in ('energy', 'force'):
        assert 'BLUE: reference state only' in out[name][1]
    assert (_selection_lines(out['energy'][1])
            == _selection_lines(out['force'][1]))
    (enr, enr_err), text = out['energy']
    assert 'BLUE-combined E' not in text
    ref = re.search(r'E\[reference\] = (\S+) ± (\S+)', text)
    np.testing.assert_allclose(
        [enr, enr_err], [float(ref.group(1)), float(ref.group(2))],
        rtol=1e-7,
    )
    (grd, grd_err), text = out['force']
    assert _averaged_states(text) == [1, 1]
    assert np.all(np.isfinite(grd)) and np.all(np.isfinite(grd_err))


def test_ess_min_zero_combines_every_state(tmp_path):
    out, _ = _run(tmp_path, ess_min=0)
    out_none, _ = _run(tmp_path, ess_min=None)
    out_tiny, _ = _run(tmp_path, ess_min=1e-12)
    for name in ('energy', 'force'):
        assert 'BLUE: state dropping disabled' in out[name][1]
        np.testing.assert_array_equal(
            np.asarray(out[name][0]), np.asarray(out_none[name][0]),
        )
        np.testing.assert_array_equal(
            np.asarray(out[name][0]), np.asarray(out_tiny[name][0]),
        )
    assert _averaged_states(out['force'][1]) == [2, 2]

    # the energy BLUE over all three states, recomputed directly
    series, kappa = {s: [] for s in (None,) + LABELS}, {}
    with h5py.File(str(tmp_path / 'syn.grd.h5'), 'r') as f:
        for b in range(N_BLOCKS):
            series[None].append(
                float(np.asarray(f['local_energies'][f'{b}']).mean())
            )
            for label in LABELS:
                w = f['fragment_weights'][label][f'{b}'][()]
                e = f['local_energies'][label][f'{b}'][()]
                series[label].append(float((w * e).sum() / w.sum()))
    for s in series:
        kappa[s] = float(do_binning_analysis(np.asarray(series[s]))[3])
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        mean, err, *_ = blue_combine_states(
            series, [None, *LABELS], kappa_per_label=kappa,
        )
    np.testing.assert_allclose(out['energy'][0], (mean, err),
                               rtol=1e-10)
    assert 'over 3 states)' in out['energy'][1]


_ROW = r'^H\s+(\S+) ± +(\S+)\s+(\S+) ± +(\S+)\s+(\S+) ± +(\S+)'


def _force_rows(text, header):
    """Rows (as printed strings) of the force block after *header*."""
    block = text.split(header + '\n', 1)[1]
    rows = []
    for ln in block.splitlines():
        m = re.match(_ROW, ln)
        if not m:
            break
        rows.append(m.groups())
    assert len(rows) == N_NUC
    return rows


def test_all_reference_skips_projection_by_default(tmp_path):
    out, _ = _run(tmp_path, ess_min=2.0)
    text = out['force'][1]
    assert ('Sum-rule projection (translational only): skipped'
            ' (every fragment is') in text
    # the reference state is printed first
    ref_rows = _force_rows(text, 'Total forces (-gradients)')
    avg_rows = _force_rows(text, 'Total forces (-gradients, averaged)')
    assert avg_rows == ref_rows
    forces, errs = out['force'][0]
    np.testing.assert_allclose(
        np.asarray(forces), np.array(ref_rows, float)[:, ::2],
        rtol=1e-5,
    )
    np.testing.assert_allclose(
        np.asarray(errs), np.array(ref_rows, float)[:, 1::2],
        rtol=1e-5,
    )
    # the raw reference estimate does not obey the sum rule
    assert np.abs(np.asarray(forces).sum(axis=0)).max() > 1e-6

    out_f, _ = _run(tmp_path, ess_min=2.0, project_sum_rules=False)
    assert 'disabled (project_sum_rules=False)' in out_f['force'][1]
    np.testing.assert_array_equal(
        np.asarray(out_f['force'][0][0]), np.asarray(forces),
    )


def test_project_sum_rules_true_still_projects(tmp_path):
    out, _ = _run(tmp_path, ess_min=2.0, project_sum_rules=True)
    text = out['force'][1]
    assert ('Sum-rule projection (translational only):'
            ' weighting = diagonal') in text
    forces = np.asarray(out['force'][0][0])
    np.testing.assert_allclose(forces.sum(axis=0), 0.0, atol=1e-10)
    # a kept secondary state: "auto" projects as before
    out, _ = _run(tmp_path)
    assert ('Sum-rule projection (translational only): weighting = '
            in out['force'][1])
    np.testing.assert_allclose(
        np.asarray(out['force'][0][0]).sum(axis=0), 0.0, atol=1e-10,
    )
    with pytest.raises(ValueError, match='project_sum_rules'):
        force.postproc_h5_pgcs(prefix=str(tmp_path / 'syn'),
                               project_sum_rules='yes')


def _reported_energy(text):
    m = re.findall(r'^Reported E = (\S+) ± (\S+)  \((.*)\)$', text, re.M)
    assert len(m) == 1 and text.rstrip().splitlines()[-1].startswith(
        'Reported E = ')
    return float(m[0][0]), float(m[0][1]), m[0][2]


@pytest.mark.parametrize('ess_min, how', [
    (2.0, 'reference only'), (CS_ESS_MIN, 'BLUE over 2 states'),
    (0, 'BLUE over 3 states'),
])
def test_energy_log_states_reported_energy(tmp_path, ess_min, how):
    out, _ = _run(tmp_path, ess_min=ess_min)
    (enr, err), text = out['energy']
    e, de, got = _reported_energy(text)
    assert got == how
    np.testing.assert_allclose([e, de], [enr, err], rtol=1e-7)


def test_energy_log_without_pgcs_states(tmp_path):
    pre = str(tmp_path / 'syn')
    _write_file(pre + '.grd.h5')
    with h5py.File(pre + '.grd.h5', 'a') as f:
        del f['fragment_weights']
    log = str(tmp_path / 'e.log')
    enr, err = energy.postproc_h5_pgcs(prefix=pre, logfile=log)
    with open(log) as fh:
        text = fh.read()
    e, de, how = _reported_energy(text)
    assert how == 'reference only'
    np.testing.assert_allclose([e, de], [enr, err], rtol=1e-7)


def _net_line(text, header, nth=0):
    """(values, errors) of the net line after the *nth* *header*."""
    line = text.split(header + '\n')[nth + 1].splitlines()[0]
    nums = [float(x) for x in line.replace('±', ' ').split()]
    assert len(nums) == 6
    return np.array(nums[::2]), np.array(nums[1::2])


def test_net_force_error_from_its_own_series(tmp_path):
    # The synthetic gradients do not sum to zero, so the reference
    # state's net force has a finite error, which must be that of its
    # walker-averaged block series sum_I F_I, binned like the atoms.
    from OmegaQMC.utils import binned_block_error
    out, _ = _run(tmp_path, ess_min=0)
    text = out['force'][1]
    _, dnet = _net_line(text, 'Total force on system')
    series = []
    with h5py.File(str(tmp_path / 'syn.grd.h5'), 'r') as f:
        e_v = np.mean([f['local_energies'][f'{b}'][()]
                       for b in range(N_BLOCKS)])
        for b in range(N_BLOCKS):
            el = f['local_energies'][f'{b}'][()]
            g = [f[k][f'{b}'][()].reshape(N_STEPS, N_WALKERS, N_NUC, 3)
                 for k in GRD_KEYS]
            tot = (f['grd_nn'][()] + g[0] + g[1]
                   + 2.0 * (el - e_v)[..., None, None] * g[2])
            series.append(tot.mean(axis=(0, 1)).sum(axis=0))
    np.testing.assert_allclose(
        dnet, binned_block_error(np.array(series))[0], rtol=2e-5,
    )
    assert np.all(dnet > 1e-3)
    assert 'Total torque (about center of mass)\n' in text


def test_averaged_net_force_zero_torque_kept_after_projection(tmp_path):
    out, _ = _run(tmp_path)
    text = out['force'][1]
    assert 'Sum-rule projection (translational only): weighting = ' in text
    val, err = _net_line(text, 'Total force on system (averaged)')
    assert np.all(np.abs(val) < 1e-10) and np.all(err < 1e-10)
    # the torque is not constrained: finite value and error about x
    # and y; about the molecular (z) axis both vanish identically
    val, err = _net_line(
        text, 'Total torque (about center of mass, averaged)',
    )
    assert np.all(np.abs(val[:2]) > 1e-6) and np.all(err[:2] > 1e-6)
    assert val[2] == 0.0 and err[2] == 0.0

    # skipped projection: the reference state's net lines
    out, _ = _run(tmp_path, ess_min=2.0)
    text = out['force'][1]
    for h_ref, h_avg in (
        ('Total force on system', 'Total force on system (averaged)'),
        ('Total torque (about center of mass)',
         'Total torque (about center of mass, averaged)'),
    ):
        ref = _net_line(text, h_ref)
        avg = _net_line(text, h_avg)
        np.testing.assert_array_equal(ref[0], avg[0])
        np.testing.assert_array_equal(ref[1], avg[1])
        # finite, except the torque about the molecular (z) axis
        assert np.all(ref[1][:2] > 0)
