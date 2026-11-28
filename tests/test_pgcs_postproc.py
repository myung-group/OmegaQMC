"""Statistics of the PGCS post-processing (energy and forces).

On synthetic gradient files: low-ESS states dropped alike for E and F,
error bars binned from the walker-averaged block series (AR(1) data
with known errors), the correlated-sampling energy offset and its
error, the Pulay term of secondary states, and the BLUE combination of
correlated states.
"""
import re
import warnings
import h5py
import jax
import numpy as np
import pytest

jax.config.update('jax_enable_x64', True)

from OmegaQMC.observables import (  # noqa: E402
    energy,
    force,
)
from OmegaQMC.utils import (  # noqa: E402
    CS_ESS_MIN,
    blue_combine_states,
    do_binning_analysis,
    select_cs_states,
    binned_block_error,
    blue_bin_size,
    do_binning_analysis_grds,
)
from OmegaQMC.observables.force import (  # noqa: E402
    CS_OFFSET_WARN_Z,
    _cs_accumulate,
    _cs_diagnostic_lines,
    _new_cs_accumulator,
    postproc_h5_pgcs,
)


# ---------------------------------------------------------------------------
# From test_pgcs_state_selection.py
#
# Low-ESS PGCS states are dropped from the BLUE, alike for E and F.
#
# :func:`OmegaQMC.utils.select_cs_states` decides from the fragment
# weights alone which PGCS states enter the BLUE, and both
# ``postproc_h5_pgcs`` functions (energy and force) use it on the same
# blocks.  On a synthetic two-fragment gradient file, with one secondary
# state of near-unit weights and one dominated by a single sample per
# block, these tests check
#
# * that the energy and force logs print identical selection lines, and
#   the force diagnostics the same ESS/N, for several cutoffs;
# * that the reference state is never dropped, and that dropping every
#   secondary state gives the reference alone without raising;
# * that ``ess_min = 0`` (or ``None``) combines every state, the energy
#   BLUE then equalling a direct :func:`blue_combine_states` over all;
# * that an atom whose fragment lost its secondary state is averaged
#   over the reference alone;
# * that with every fragment reference-only the translational sum-rule
#   projection is skipped by default (``project_sum_rules="auto"``), the
#   averaged rows being the reference rows, while ``True`` still
#   projects;
# * that net-force errors come from their own block series, and that after
#   the projection the averaged net force is 0 +- 0 while the torque
#   keeps a finite error;
# * that the energy log always ends with the reported energy.
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# From test_pgcs_block_errors.py
#
# Printed PGCS force errors account for correlation between blocks.
#
# ``postproc_h5_pgcs`` used to form every per-state error from each
# walker's few steps inside one block, combined over blocks and walkers
# as independent; correlation between blocks was invisible to it, and
# in a production run the errors came out several times too small.
# Every printed error now comes from binning the walker-averaged
# per-block series, as :func:`OmegaQMC.utils.blue_combine_states` bins
# one state.  On a synthetic gradient file whose per-walker block
# offsets follow an AR(1) process (rho = 0.5) these tests check
#
# * that :func:`OmegaQMC.utils.binned_block_error` is the error
#   ``blue_combine_states`` gives a single state, with its bin size;
# * that the printed per-state errors (atoms, net force, torque) are
#   ``binned_block_error`` of the independently formed block series,
#   weighted for a secondary state, and agree with the analytic error
#   of the AR(1) process, which the within-block estimate misses by a
#   factor of about 2;
# * that with every fragment reference-only the averaged rows, errors
#   included, are the reference state's rows.
# ---------------------------------------------------------------------------


N_STEPS_blk, N_WALKERS_blk, N_NUC_blk, N_BLOCKS_blk = 4, 8, 2, 400
RHO = 0.5
S_NOISE = 1.0                         # per-sample white noise
S_BLOCK = S_NOISE / np.sqrt(N_STEPS_blk)  # marginal sd of the AR(1) part
LABEL = '1:E,2:Rz180'
COORDS = np.array([[0.3, -0.2, -0.7], [-0.1, 0.4, 0.7]])
MASSES = np.array([1.00782503223, 1.00782503223])


def _ar1(rng, shape):
    """AR(1) series along axis 0 with RHO and unit marginal sd."""
    x = np.empty(shape)
    x[0] = rng.normal(size=shape[1:])
    innov = np.sqrt(1.0 - RHO ** 2)
    for b in range(1, shape[0]):
        x[b] = RHO * x[b - 1] + innov * rng.normal(size=shape[1:])
    return x


def _write_file_blk(path, seed=0):
    rng = np.random.default_rng(seed)
    n = N_STEPS_blk * N_WALKERS_blk
    # per-walker block offsets, AR(1) over blocks
    offs = S_BLOCK * _ar1(rng, (N_BLOCKS_blk, N_WALKERS_blk, N_NUC_blk, 3))
    with h5py.File(path, 'w') as f:
        g = f.create_group('system')
        g['atom_symbols'] = np.bytes_('H H')
        g['atom_coords'] = COORDS
        g['units'] = np.bytes_('Bohr')
        g['atom_fragment_map'] = np.array([1, 2])
        f['grd_nn'] = np.zeros((N_NUC_blk, 3))
        for grp in ('local_energies', 'fragment_weights') + GRD_KEYS:
            f.create_group(grp)
            f[grp].create_group(LABEL)
        for b in range(N_BLOCKS_blk):
            bcs = f'{b}'
            el = rng.normal(size=(N_STEPS_blk, N_WALKERS_blk)) - 1.1
            f['local_energies'][bcs] = el
            x = (
                offs[b][None]
                + S_NOISE
                * rng.normal(size=(N_STEPS_blk, N_WALKERS_blk, N_NUC_blk, 3))
            ).reshape(n, N_NUC_blk, 3)
            f['grd_ee_en'][bcs] = x
            f['grd_ee_en'][LABEL][bcs] = x + 0.1 * rng.normal(size=x.shape)
            for grp in ('grd_ke', 'grd_logpsi'):
                f[grp][bcs] = np.zeros_like(x)
                f[grp][LABEL][bcs] = np.zeros_like(x)
            f['fragment_weights'][LABEL][bcs] = np.exp(
                0.1 * rng.normal(size=n))
            f['local_energies'][LABEL][bcs] = el.ravel()


@pytest.fixture(scope='module')
def run(tmp_path_factory):
    d = tmp_path_factory.mktemp('ar1')
    pre = str(d / 'ar1')
    _write_file_blk(pre + '.grd.h5')
    out = {}
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        for name, kw in (('auto', {}), ('ref_only', {'ess_min': 2.0})):
            log = str(d / f'{name}.log')
            res = force.postproc_h5_pgcs(prefix=pre, logfile=log, **kw)
            with open(log) as fh:
                out[name] = (res, fh.read())
    return pre + '.grd.h5', out


def _block_series(path, label):
    """Walker-averaged block series of the total gradient, (B, N, 3),
    and the within-block (old) error estimate."""
    series, sq, count = [], 0.0, 0
    with h5py.File(path, 'r') as f:
        for b in range(N_BLOCKS_blk):
            grp = f['grd_ee_en'] if label is None else f['grd_ee_en'][label]
            x = grp[f'{b}'][()].reshape(
                N_STEPS_blk, N_WALKERS_blk, N_NUC_blk, 3
            )
            if label is None:
                xw = x.mean(axis=0)
            else:
                w = f['fragment_weights'][label][f'{b}'][()].reshape(
                    N_STEPS_blk, N_WALKERS_blk)
                xw = np.einsum('sw,swnk->wnk', w, x) / w.sum(axis=0)[
                    :, None, None]
            series.append(xw.mean(axis=0))
            serr = np.asarray(do_binning_analysis_grds(x)[1])
            sq = sq + (serr ** 2).sum(axis=0)
            count += serr.shape[0]
    return np.array(series), np.sqrt(sq) / count


def _section(text, label):
    head = ('--- Reference state ---' if label is None
            else f'--- Secondary state: {label} ---')
    return text.split(head, 1)[1].split('\n--- ', 1)[0]


def _rows(text, header):
    block = text.split(header + '\n', 1)[1]
    rows = []
    for ln in block.splitlines():
        m = re.match(_ROW, ln)
        if not m:
            break
        rows.append([float(v) for v in m.groups()])
    assert len(rows) == N_NUC_blk
    return np.array(rows)


def _net(text, header):
    line = text.split(header + '\n', 1)[1].splitlines()[0]
    nums = np.array([float(v) for v in line.replace('±', ' ').split()])
    return nums[::2], nums[1::2]


def test_binned_block_error_is_single_state_blue():
    rng = np.random.default_rng(3)
    x = _ar1(rng, (300, 4))
    err, bins = binned_block_error(x)
    for m in range(x.shape[1]):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            _, e_blue, _, _, b_blue = blue_combine_states(
                {0: x[:, m]}, [0])
        assert bins[m] == b_blue
        np.testing.assert_allclose(err[m], e_blue, rtol=1e-12)
    kappa = float(do_binning_analysis(x[:, 0])[3])
    assert bins[0] == blue_bin_size(300, kappa) == int(np.ceil(2 * kappa))
    e1, b1 = binned_block_error(x[:, 0])
    assert np.ndim(e1) == 0 and e1 == err[0] and b1 == bins[0]
    # too few blocks for the requested bins, and a single block
    assert blue_bin_size(10, 20.0) == 5
    assert blue_bin_size(10, 20.0, n_states=3) == 2
    assert np.isinf(binned_block_error(np.ones((1, 2)))[0]).all()


@pytest.mark.parametrize('label', [None, LABEL])
def test_state_errors_bin_the_block_series(run, label):
    path, out = run
    text = _section(out['auto'][1], label)
    rows = _rows(text, 'Total forces (-gradients)')
    series, old = _block_series(path, label)
    err, _ = binned_block_error(series.reshape(N_BLOCKS_blk, -1))
    np.testing.assert_allclose(
        rows[:, 1::2], err.reshape(N_NUC_blk, 3), rtol=2e-5)
    np.testing.assert_allclose(
        rows[:, ::2], -series.mean(axis=0), rtol=2e-5, atol=1e-12)

    # net force and torque about the center of mass: their own series
    com = np.average(COORDS, axis=0, weights=MASSES)
    net = np.concatenate([
        series.sum(axis=1),
        np.cross(COORDS - com, series).sum(axis=1),
    ], axis=1)
    net_err = binned_block_error(net)[0].reshape(2, 3)
    np.testing.assert_allclose(
        _net(text, 'Total force on system')[1], net_err[0], rtol=2e-5)
    np.testing.assert_allclose(
        _net(text, 'Total torque (about center of mass)')[1],
        net_err[1], rtol=2e-5)

    # the AR(1) error: walker-averaged block series of variance
    # S_BLOCK^2 / W (AR) + S_NOISE^2 / (S W) (white)
    var_ar = S_BLOCK ** 2 / N_WALKERS_blk * (1 + RHO) / (1 - RHO)
    var_wn = S_NOISE ** 2 / (N_STEPS_blk * N_WALKERS_blk)
    if label is not None:
        var_wn *= 1.0 + 0.1 ** 2          # the state's extra noise
    exact = np.sqrt((var_ar + var_wn) / N_BLOCKS_blk)
    ratio = rows[:, 1::2] / exact
    assert 0.85 < ratio.mean() < 1.15
    assert np.all((ratio > 0.6) & (ratio < 1.5))
    # the old within-block estimate misses the AR(1) part
    assert np.all(old / exact < 0.6)


def test_reference_only_averaged_rows_are_the_reference_rows(run):
    _, out = run
    text = out['ref_only'][1]
    assert 'skipped (every fragment is reference-only' in text
    ref = _section(text, None)
    np.testing.assert_array_equal(
        _rows(text, 'Total forces (-gradients, averaged)'),
        _rows(ref, 'Total forces (-gradients)'))
    for h_ref, h_avg in (
        ('Total force on system', 'Total force on system (averaged)'),
        ('Total torque (about center of mass)',
         'Total torque (about center of mass, averaged)'),
    ):
        for a, b in zip(_net(ref, h_ref), _net(text, h_avg)):
            np.testing.assert_array_equal(a, b)
    forces, errs = out['ref_only'][0]
    np.testing.assert_allclose(
        np.asarray(errs), _rows(ref, 'Total forces (-gradients)')[:, 1::2],
        rtol=2e-5)


# ---------------------------------------------------------------------------
# From test_pgcs_cs_offset_error.py
#
# Error of the PGCS correlated-sampling energy offset.
#
# ``_cs_diagnostic_lines`` prints ``<w E_L(TR)>/<w> - E_V ± err`` for
# each secondary state and warns beyond ``CS_OFFSET_WARN_Z`` errors.
# The error used to treat blocks as independent and E_V as exact; it
# now bins the delta-method series of the difference of the two ratios
# (``E_V`` is the mean of the same blocks' ``E_L(R)``) with
# :func:`OmegaQMC.utils.binned_block_error`.  With weighted synthetic
# blocks whose offset and reference energy follow AR(1) processes
# (rho = 0.5) these tests check
#
# * that the printed error matches the analytic error of the offset,
#   with and without a block-level fluctuation common to E_L(TR) and
#   E_L(R), which the old error missed by a factor of about sqrt(3)
#   (too small) and counted as offset noise (too large), respectively;
# * that an exact symmetry gives a zero offset and error and no warning;
# * that a genuine offset still warns.
# ---------------------------------------------------------------------------


N_BLOCKS_off, N_SAMPLES = 400, 200
SIG_C = 0.05          # block-level offset fluctuation, AR(1)
SIG_REF = 0.1         # per-sample noise common to E_L(TR) and E_L(R)
SIG_T = 0.01          # per-sample noise of E_L(TR) alone
SIG_LOGW = 0.3


def _ar1_off(rng, n):
    x = np.empty(n)
    x[0] = rng.normal()
    for b in range(1, n):
        x[b] = RHO * x[b - 1] + np.sqrt(1 - RHO ** 2) * rng.normal()
    return x


def _blocks(seed, sig_a, shift=0.0):
    """Weighted blocks; E_L(TR) = E_L(R) + c_b + noise."""
    rng = np.random.default_rng(seed)
    a = sig_a * _ar1_off(rng, N_BLOCKS_off)       # common block fluctuation
    c = SIG_C * _ar1_off(rng, N_BLOCKS_off) + shift
    out = []
    for b in range(N_BLOCKS_off):
        w = np.exp(SIG_LOGW * rng.normal(size=N_SAMPLES))
        e_ref = -1.0 + a[b] + SIG_REF * rng.normal(size=N_SAMPLES)
        e_t = e_ref + c[b] + SIG_T * rng.normal(size=N_SAMPLES)
        out.append((w, e_t, e_ref))
    return out


def _analytic_err(blocks):
    """Error of mean_b [sum w E_t / sum w - mean E_ref] for the model:
    AR(1) part plus each block's white-noise variance."""
    white = 0.0
    for w, _, _ in blocks:
        p = w / w.sum()
        white += (SIG_REF ** 2 * np.sum((p - 1 / N_SAMPLES) ** 2)
                  + SIG_T ** 2 * np.sum(p ** 2))
    var = SIG_C ** 2 * (1 + RHO) / (1 - RHO) / N_BLOCKS_off
    return np.sqrt(var + white / N_BLOCKS_off ** 2)


def _diagnose(blocks):
    e_v = np.mean([e_ref.mean() for _, _, e_ref in blocks])
    acc = _new_cs_accumulator(True)
    for w, e_t, e_ref in blocks:
        _cs_accumulate(acc, w, e_t, e_ref, e_v)
    stats = {'n': N_BLOCKS_off * N_SAMPLES, 'mean_w': 1.0,
             'max_frac': 1e-3, 'ess': 0.9}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        lines = _cs_diagnostic_lines('1:Rz180', acc, stats)
    m = re.search(r'- E_V = (\S+) ± (\S+) Ha', lines[1])
    old = np.sqrt(
        np.sum((np.array(acc['y_blk'])
                - float(m.group(1)) * np.array(acc['w_blk'])) ** 2)
        / (N_BLOCKS_off * (N_BLOCKS_off - 1) * np.mean(acc['w_blk']) ** 2))
    return (float(m.group(1)), float(m.group(2)), old,
            [str(w.message) for w in caught])


@pytest.mark.parametrize('sig_a, old_lo, old_hi', [
    (0.0, 0.0, 0.7),       # old: autocorrelation missed, too small
    (0.5, 1.5, np.inf),    # old: E_V's fluctuation counted, too large
])
def test_offset_error_matches_analytic(sig_a, old_lo, old_hi):
    blocks = _blocks(1, sig_a)
    offset, err, old, msgs = _diagnose(blocks)
    exact = _analytic_err(blocks)
    # over seeds the binned error is 0.90 +- 0.08 of exact (BLUE's
    # ceil(2 kappa) bins are a little short); the old one 0.58 / 5.8
    assert 0.8 < err / exact < 1.15
    assert old_lo < old / exact < old_hi
    assert abs(offset) < CS_OFFSET_WARN_Z * err and not msgs


def test_exact_symmetry_zero_error_no_warning():
    rng = np.random.default_rng(2)
    blocks = []
    for b in range(50):
        e_ref = -1.0 + rng.normal(size=N_SAMPLES)
        blocks.append((np.ones(N_SAMPLES), e_ref.copy(), e_ref))
    offset, err, old, msgs = _diagnose(blocks)
    assert abs(offset) < 1e-14 and err < 1e-14 and not msgs
    assert old > 1e-3        # the old error was E_V's noise


def test_genuine_offset_warns():
    blocks = _blocks(3, 0.5, shift=0.1)
    offset, err, _, msgs = _diagnose(blocks)
    assert offset > CS_OFFSET_WARN_Z * err
    assert len(msgs) == 1 and 'standard errors from zero' in msgs[0]


# ---------------------------------------------------------------------------
# From test_pgcs_pulay_secondary.py
#
# Pulay term of PGCS secondary states in ``postproc_h5_pgcs``.
#
# A secondary state's gradient terms are evaluated at the transformed
# configuration T R, so its Pulay term must use E_L(T R), which both
# gradient writers store as ``local_energies/<label>/<blk>``; E_V stays
# the reference mean.  On a synthetic gradient file these tests check
#
# * that the printed Pulay gradients of the reference and the secondary
#   state equal 2 <(E_L - E_V) dlnpsi/dR> with E_L(R) and E_L(T R)
#   respectively, where the two energies differ;
# * that a file without ``local_energies/<label>`` warns and falls back
#   to the reference E_L(R);
# * the correlated-sampling diagnostics printed for the secondary state
#   (mean weight, Kish ESS/N, energy offset) and their warnings.
# ---------------------------------------------------------------------------


N_STEPS_pulay, N_WALKERS_pulay, N_NUC_pulay, N_BLOCKS_pulay = 4, 8, 2, 3
LABEL_pulay = '1:Rz180'


def _write_file_pulay(path, with_secondary_el=True, seed=0, exact=False,
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
        f['grd_nn'] = rng.normal(size=(N_NUC_pulay, 3))
        for grp in ('local_energies', 'fragment_weights') + GRD_KEYS:
            f.create_group(grp)
            if grp != 'local_energies' or with_secondary_el:
                f[grp].create_group(LABEL_pulay)
        for b in range(N_BLOCKS_pulay):
            bcs = f'{b}'
            el = rng.normal(size=(N_STEPS_pulay, N_WALKERS_pulay)) - 1.1
            el_t = el.ravel() + rng.normal(0.5, 0.2, size=el.size)
            if exact:
                el_t = el.ravel().copy()
            f['local_energies'][bcs] = el
            if with_secondary_el:
                f['local_energies'][LABEL_pulay][bcs] = el_t
            for grp in GRD_KEYS:
                x = rng.normal(
                    size=(N_STEPS_pulay * N_WALKERS_pulay, N_NUC_pulay, 3)
                )
                f[grp][bcs] = x
                f[grp][LABEL_pulay][bcs] = x
                if grp == 'grd_logpsi':
                    logpsi.append(x)
            f['fragment_weights'][LABEL_pulay][bcs] = (
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
        out.append(np.array(nums, dtype=float).reshape(N_NUC_pulay, 3))
    return out


def _postproc(tmp_path, **kw):
    pre = str(tmp_path / 'syn')
    data = _write_file_pulay(pre + '.grd.h5', **kw)
    log = str(tmp_path / 'syn.log')
    postproc_h5_pgcs(prefix=pre, logfile=log)
    with open(log) as fh:
        return data, fh.read()


def _run_pulay(tmp_path, **kw):
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
            if str(w.message).startswith(f'PGCS state {LABEL_pulay}')]


def test_secondary_pulay_uses_transformed_energies(tmp_path):
    (el_ref, el_sec, logpsi), (p_ref, p_sec) = _run_pulay(tmp_path)
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
    with pytest.warns(UserWarning, match=f'local_energies/{LABEL_pulay}'):
        (el_ref, _, logpsi), (p_ref, p_sec) = _run_pulay(
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
    # E_L(TR) = E_L(R) with unit weights: the difference series
    # vanishes block by block, so the offset error is zero too
    assert abs(d['offset']) < 1e-12 and d['serr'] < 1e-12
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
    assert f'local_energies/{LABEL_pulay} absent' in text


# ---------------------------------------------------------------------------
# From test_blue_combine_states.py
#
# Error of the fragment-wise BLUE combine for correlated states.
#
# PGCS states are images of the same samples, so their block series can
# be identical (e.g. a force component every operation leaves unchanged).
# The combined error must then equal the single-series error, not
# err / sqrt(K) as an independence assumption would give.  With few bins
# :func:`OmegaQMC.utils.blue_combine_states` falls back to fixed weights,
# but it propagates them through the full bin-mean covariance, so the
# combined error is still exact here.
# ---------------------------------------------------------------------------


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
