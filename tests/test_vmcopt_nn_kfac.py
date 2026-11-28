"""Molecular KFAC VMC optimiser.

Block assignment, factor shapes and chunking; agreement with the
reference FermiNet network and kfac-jax; local-energy clipping; and
the non-finite guard, which replaces walkers with NaN/inf local
energies and rejects non-finite steps.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import dataclasses
import os
import h5py

jax.config.update('jax_enable_x64', True)

from OmegaQMC.utils import Mole_custom  # noqa: E402
from OmegaQMC.vmcopt_nn_kfac import (  # noqa: E402
    _VMCOptDriverNN_KFAC,
    clip_local_energies,
    get_vmcopt_nn_func,
)
from OmegaQMC.spin_layout import to_layout  # noqa: E402
from OmegaQMC.psi.nn.adapter import make_nn_log_psi  # noqa: E402
from OmegaQMC.psi.nn.config import load_nn_config  # noqa: E402
from OmegaQMC.psi.nn.kfac_tap import leaf_path_str  # noqa: E402
import OmegaQMC.vmcopt_nn_kfac as kfac_mod  # noqa: E402


# ---------------------------------------------------------------------------
# From test_vmcopt_nn_kfac.py
#
# Unit tests for the molecular KFAC VMC optimiser.
#
# Covers the pieces of :mod:`OmegaQMC.vmcopt_nn_kfac` that are cheap
# to check without running a full optimisation: the assignment of every
# parameter to a Kronecker or a diagonal block, the shapes of the
# factor sums and state, the equivalence of the chunked (scanned)
# accumulation to a single-chunk evaluation, and the dtype contract of
# the parameter update.  The agreement with kfac-jax itself is in
# the reference tests below.
# ---------------------------------------------------------------------------


L_H2 = 1.4010


def _h2():
    return Mole_custom.from_arrays(
        charges=[1, 1],
        coords=[[0.0, 0.0, -L_H2 / 2], [0.0, 0.0, L_H2 / 2]],
        n_up=1, n_down=1,
    )


@pytest.fixture(scope="module")
def drv():
    # float64 network, so the chunk-invariance check below compares
    # summation orders rather than float32 rounding.
    return _VMCOptDriverNN_KFAC(
        _h2(), "psiformer", jax.random.key(0), nn_dtype=None,
    )


@pytest.fixture(scope="module")
def batch(drv):
    w = drv.initialize_walkers(jax.random.key(1), 8)
    e = drv.compute_batch_energy(w, drv.init_params)
    return w, e - jnp.mean(e)


def test_every_param_is_in_one_block(drv):
    """Each leaf is a Linear kernel or bias (Kronecker block) or one
    of the remaining leaves (diagonal block), and only one of them."""
    taps = drv._taps
    leaves = jax.tree.leaves(drv.init_params)
    owned = list(taps.other_idx)
    for info in taps.layers.values():
        owned.append(info['kernel'])
        if info['bias'] is not None:
            owned.append(info['bias'])
    assert sorted(owned) == list(range(len(leaves)))
    assert drv._linear_count == len(taps.layers) > 0
    n_linear = sum(leaves[i].size for info in taps.layers.values()
                   for i in (info['kernel'], info['bias'])
                   if i is not None)
    assert drv._linear_params == n_linear
    # every attention / MLP Linear of the PsiFormer acts on each of the
    # two electrons; the orbital layers on the one electron of a spin
    assert {info['n_rep'] for info in taps.layers.values()} <= {1, 2}


def test_factor_shapes(drv, batch):
    """Sums and state are ``(in [+1], in [+1])`` and ``(out, out)``,
    the gradient ``(in [+1], out)`` as the NNX kernel."""
    w, de = batch
    S_in, S_out, grad, d_sum, g_other = drv._accumulate_factors(
        drv.init_params, w[None], de[None])
    state = drv._init_factor_state()
    n_other = sum(jax.tree.leaves(drv.init_params)[i].size
                  for i in drv._taps.other_idx)
    for name, info in drv._taps.layers.items():
        k = info['in_dim'] + (info['bias'] is not None)
        assert S_in[name].shape == (k, k) == state['S_in'][name].shape
        assert (S_out[name].shape == (info['out_dim'],) * 2
                == state['S_out'][name].shape)
        assert grad[name].shape == (k, info['out_dim'])
    assert d_sum.shape == g_other.shape == state['D'].shape == (n_other,)


def _rel_max_diff(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return np.abs(a - b).max() / max(np.abs(a).max(), 1e-30)


def test_chunked_accumulation_matches_single_chunk(drv, batch):
    """The factor sums are chunk-invariant (the walker chunking is
    what decouples the walker count from memory)."""
    w, de = batch
    n = w.shape[0]
    one = drv._accumulate_factors(drv.init_params, w[None], de[None])
    two = drv._accumulate_factors(
        drv.init_params, w.reshape((2, n // 2) + w.shape[1:]),
        de.reshape(2, n // 2))
    for a, b in zip(jax.tree.leaves(one), jax.tree.leaves(two)):
        assert _rel_max_diff(a, b) < 1e-10


def test_update_preserves_pytree_and_dtypes(drv, batch):
    """The step moves the params without promoting their dtype.

    NNX builds the network in float32 while OmegaQMC enables x64, so
    a float64 KFAC step written back verbatim would silently promote
    the whole network — which costs roughly an order of magnitude in
    per-walker gradient time on a consumer GPU.
    """
    w, de = batch
    p = drv.init_params
    sums = drv._accumulate_factors(p, w[None], de[None])
    new_p, state, norm, clip, _dg, _dF = drv._kfac_apply(
        p, sums, drv._init_factor_state(), jnp.asarray(0.05),
        jnp.asarray(1.0e-3), float(w.shape[0]),
    )
    assert (jax.tree_util.tree_structure(new_p)
            == jax.tree_util.tree_structure(p))
    for old, new in zip(jax.tree.leaves(p), jax.tree.leaves(new_p)):
        assert old.dtype == new.dtype, (old.dtype, new.dtype)
        assert old.shape == new.shape
    moved = sum(float(jnp.sum(jnp.abs(a - b)))
                for a, b in zip(jax.tree.leaves(new_p), jax.tree.leaves(p)))
    assert moved > 0.0
    assert float(norm) > 0.0
    assert 0.0 < float(clip) <= 1.0
    assert float(state['weight']) == 1.0


# ---------------------------------------------------------------------------
# From test_vmcopt_nn_kfac_jax.py
#
# OmegaQMC's KFAC and the ``ferminet_deepmind`` ansatz against
# DeepMind's FermiNet and kfac-jax.
#
# ``tests/ferminet_kfac_reference.npz`` was written by
# ``tools/ferminet_kfac_reference.py`` (ferminet e9d5a60, kfac-jax 0.0.8,
# float64): parameters of a small DeepMind FermiNet, log|psi| and E_L on
# 16 walkers, and the parameters after a few kfac-jax steps on those
# walkers.  The parameters are transplanted into OmegaQMC's
# ``ferminet_deepmind`` network.  The tolerances allow for OmegaQMC's
# safe norm, ``sqrt(eps + r^2)``, which puts the self-pair distance at
# ``sqrt(eps) ~ 1.5e-8`` instead of DeepMind's exact zero.
# ---------------------------------------------------------------------------


REF = os.path.join(os.path.dirname(__file__), 'ferminet_kfac_reference.npz')


@pytest.fixture(scope='module')
def ref():
    with np.load(REF) as f:
        return {k: f[k] for k in f.files}


def _config(ref):
    hidden = ref['hidden']
    return dataclasses.replace(
        load_nn_config('ferminet_deepmind'),
        embedding_dim=int(hidden[0][0]),
        two_particle_stream_dim=int(hidden[0][1]),
        n_interactions=len(hidden), n_determinants=int(ref['ndet']),
    )


def _mol(ref, case):
    nu, nd = (int(x) for x in ref[f'{case}/nspins'])
    sym = {3.0: 'Li', 1.0: 'H'}
    mol = Mole_custom()
    mol.build(
        atom='; '.join(f"{sym[float(z)]} {x} {y} {w}" for z, (x, y, w)
                       in zip(ref['charges'], ref['atoms'])),
        unit='Bohr', basis='sto-3g', spin=nu - nd,
        charge=int(round(ref['charges'].sum())) - nu - nd, verbose=0,
    )
    return mol


def _walkers(ref, case):
    nu, nd = (int(x) for x in ref[f'{case}/nspins'])
    pos = ref[f'{case}/pos'].reshape(-1, nu + nd, 3)
    return to_layout(jnp.asarray(pos), nu, nd)


def _transplant(ref, case, k, template):
    """DeepMind parameters (step k) as an OmegaQMC State, in float64."""
    pre = f'{case}/traj{k}/'
    vals = {}
    n_layers = len(ref['hidden'])
    for i in range(n_layers):
        g = f'omni/gnn/layers/{i}'
        s = f'{pre}layers/streams/{i}'
        vals[f'{g}/subnet/layers/0/kernel'] = ref[f'{s}/single/w']
        vals[f'{g}/subnet/layers/0/bias'] = ref[f'{s}/single/b']
        if f'{s}/double/w' in ref:
            vals[f'{g}/subnet_g/layers/0/kernel'] = ref[f'{s}/double/w']
            vals[f'{g}/subnet_g/layers/0/bias'] = ref[f'{s}/double/b']
    for s, spin in enumerate(('up', 'down')):
        vals[f'omni/backflow/{spin}/nets/0/layers/0/kernel'] = \
            ref[f'{pre}orbital/{s}/w']
        vals[f'envelope/pi_{spin}'] = ref[f'{pre}envelope/{s}/pi'].T
        vals[f'envelope/zetas_{spin}'] = ref[f'{pre}envelope/{s}/sigma'].T
    used = set()

    def put(path, leaf):
        name = leaf_path_str(path)
        if name not in vals:     # the unused last two-electron layer
            return jnp.asarray(leaf, jnp.float64)
        used.add(name)
        assert vals[name].shape == leaf.shape, name
        return jnp.asarray(vals[name], jnp.float64)

    out = jax.tree_util.tree_map_with_path(put, template)
    assert used == set(vals)
    return out


@pytest.mark.parametrize('case', ['a', 'b'])
def test_ferminet_deepmind_matches_deepmind_network(ref, case):
    mol = _mol(ref, case)
    log_psi, params, _, _ = make_nn_log_psi(
        _config(ref), mol, jax.random.key(0))
    p = _transplant(ref, case, 0, params)
    walkers = _walkers(ref, case)
    nuc = jnp.asarray(ref['atoms'])
    lp = jax.vmap(log_psi, in_axes=(0, None, None))(walkers, nuc, p)
    np.testing.assert_allclose(lp, ref[f'{case}/logpsi'], rtol=0, atol=1e-6)

    drv = _VMCOptDriverNN_KFAC(mol, _config(ref), jax.random.key(0),
                               nn_dtype=None)
    e_l = drv.compute_batch_energy(walkers, p)
    np.testing.assert_allclose(e_l, ref[f'{case}/e_l'], rtol=0, atol=1e-4)


@pytest.mark.parametrize('case', ['a', 'b'])
def test_kfac_jax_steps_match_kfac_jax(ref, case):
    mol = _mol(ref, case)
    nc = float(ref[f'{case}/norm_constraint'])
    drv = get_vmcopt_nn_func(
        mol, _config(ref), jax.random.key(0), nn_dtype=None,
        lr=0.05, lr_decay=1.0e4, damping=1.0e-3,
        ema_decay=0.95, norm_constraint=None if np.isinf(nc) else nc,
    )
    walkers = _walkers(ref, case)
    params = _transplant(ref, case, 0, drv.init_params)
    state = None
    for it in range(int(ref[f'{case}/n_steps'])):
        e_l = drv.compute_batch_energy(walkers, params)
        _, de = clip_local_energies(e_l, 5.0, clip_median=False,
                                    center_at_clip=True)
        new, state, _ = drv.kfac_jax_step(params, walkers, de, state, it,
                                          chunk=8)
        target = _transplant(ref, case, it + 1, drv.init_params)
        for (path, a), b, c in zip(
                jax.tree_util.tree_flatten_with_path(new)[0],
                jax.tree.leaves(target), jax.tree.leaves(params)):
            d_dm = np.asarray(b - c)
            d_ours = np.asarray(a - c)
            scale = np.linalg.norm(d_dm)
            if scale == 0:      # the unused last two-electron layer
                assert np.linalg.norm(d_ours) == 0
                continue
            rel = np.linalg.norm(d_ours - d_dm) / scale
            assert rel < 1e-6, (it, leaf_path_str(path), rel)
        params = target


def test_tap_statistics_reproduce_parameter_gradient(ref):
    """Rows and cotangents of the taps give d log|psi| / d theta."""
    mol = _mol(ref, 'b')
    drv = _VMCOptDriverNN_KFAC(mol, _config(ref), jax.random.key(1),
                               nn_dtype=None)
    taps = drv._taps
    walkers = _walkers(ref, 'b')[:3]
    # float64 leaves, or jax.grad rounds the reference to float32
    params = jax.tree.map(lambda x: x.astype(jnp.float64), drv.init_params)
    xs, gs, g_other = jax.vmap(taps.per_walker, in_axes=(None, 0))(
        params, walkers)
    full = jax.vmap(jax.grad(
        lambda p, e: drv.log_psi(e, drv.nuc_crds, p)), in_axes=(None, 0))(
        params, walkers)
    leaves = jax.tree.leaves(full)
    for name, info in taps.layers.items():
        dW = jnp.einsum('wri,wro->wio', xs[name], gs[name])
        np.testing.assert_allclose(dW, leaves[info['kernel']],
                                   rtol=1e-8, atol=1e-10)
        if info['bias'] is not None:
            np.testing.assert_allclose(gs[name].sum(1), leaves[info['bias']],
                                       rtol=1e-8, atol=1e-10)
        # rows per walker: electrons, electron pairs or one spin
        assert xs[name].shape[1] == info['n_rep']
    for i, g in zip(taps.other_idx, g_other):
        np.testing.assert_allclose(g, leaves[i], rtol=1e-8, atol=1e-10)
    reps = {name: info['n_rep'] for name, info in taps.layers.items()}
    assert reps['omni/gnn/layers/0/subnet/layers/0'] == 5
    assert reps['omni/gnn/layers/0/subnet_g/layers/0'] == 25
    assert reps['omni/backflow/up/nets/0/layers/0'] == 3
    assert reps['omni/backflow/down/nets/0/layers/0'] == 2
    # the last layer's two-electron stream is never used
    assert 'omni/gnn/layers/2/subnet_g/layers/0' not in reps


def test_kfac_style_option(ref):
    """``kfac_style='kfac_jax'`` is still accepted (and the default);
    the removed style and its options are rejected."""
    mol = _mol(ref, 'a')
    assert _VMCOptDriverNN_KFAC(
        mol, _config(ref), jax.random.key(0),
        kfac_style='kfac_jax').kfac_style == 'kfac_jax'
    with pytest.raises(ValueError, match='removed'):
        _VMCOptDriverNN_KFAC(mol, _config(ref), jax.random.key(0),
                             kfac_style='omegaqmc')
    for kw in (dict(capture_activations=True), dict(backward='manual'),
               dict(fixed_scale=True)):
        with pytest.raises(TypeError):
            _VMCOptDriverNN_KFAC(mol, _config(ref), jax.random.key(0),
                                 **kw)


def test_kfac_jax_driver_runs(ref, tmp_path):
    """A few iterations of the full loop in float32."""
    mol = _mol(ref, 'a')
    drv = get_vmcopt_nn_func(mol, _config(ref), jax.random.key(0),
                             clip_local_energy=5.0)
    log = tmp_path / 'opt.log'
    params, data = drv(
        jax.random.key(1), num_iters=4, num_walkers=64,
        factor_chunk_size=32, num_steps_decorr=2, num_blocks_equil=2,
        num_steps_per_block=10, fname_log=str(log),
        prefix=str(tmp_path / 'nnopt'),
    )
    assert len(data['E_history']) == 4
    assert np.all(np.isfinite(data['E_history']))
    assert np.isfinite(data['energy']['mean'])
    assert 'kfac_style=kfac_jax' in log.read_text()


def test_kfac_jax_resume_is_exact(ref, tmp_path):
    """2 + 2 resumed iterations give the iterates of 4 uninterrupted
    ones: the KFAC factors, damping, walkers, step size and PRNG key
    come back from ``{prefix}.state.h5``."""
    mol = _mol(ref, 'a')
    kw = dict(num_walkers=32, factor_chunk_size=16, num_steps_decorr=2,
              num_blocks_equil=1, num_steps_per_block=5, verbose=1)

    def driver():
        return get_vmcopt_nn_func(mol, _config(ref), jax.random.key(0),
                                  clip_local_energy=5.0)

    p_full, d_full = driver()(jax.random.key(1), num_iters=4,
                              prefix=str(tmp_path / 'full'),
                              fname_log=str(tmp_path / 'full.log'), **kw)
    drv = driver()
    drv(jax.random.key(1), num_iters=2, prefix=str(tmp_path / 'split'),
        fname_log=str(tmp_path / 'split1.log'), **kw)
    assert (tmp_path / 'split.state.h5').exists()
    p_res, d_res = drv(jax.random.key(99), num_iters=2,
                       prefix=str(tmp_path / 'split'),
                       fname_log=str(tmp_path / 'split2.log'), **kw)
    log = (tmp_path / 'split2.log').read_text()
    assert 'KFAC factors and damping restored' in log
    assert 'walkers, step size and PRNG key restored' in log
    np.testing.assert_array_equal(d_res['E_history'],
                                  d_full['E_history'][2:])
    for a, b in zip(jax.tree.leaves(p_res), jax.tree.leaves(p_full)):
        np.testing.assert_array_equal(a, b)


def test_kfac_jax_resume_without_state_file(ref, tmp_path):
    """A checkpoint without a state file (older runs) still resumes:
    fresh factors, re-equilibrated walkers."""
    mol = _mol(ref, 'a')
    kw = dict(num_walkers=32, factor_chunk_size=16, num_steps_decorr=2,
              num_blocks_equil=1, num_steps_per_block=5)
    drv = get_vmcopt_nn_func(mol, _config(ref), jax.random.key(0),
                             clip_local_energy=5.0)
    pre = str(tmp_path / 'old')
    drv(jax.random.key(1), num_iters=1, prefix=pre,
        fname_log=str(tmp_path / 'a.log'), **kw)
    os.remove(pre + '.state.h5')
    _, d = drv(jax.random.key(2), num_iters=1, prefix=pre,
               fname_log=str(tmp_path / 'b.log'), **kw)
    assert np.all(np.isfinite(d['E_history']))
    log = (tmp_path / 'b.log').read_text()
    assert 'Resuming from' in log and 'start afresh' in log


# ---------------------------------------------------------------------------
# From test_vmcopt_nn_kfac_clip.py
#
# Tests for the local-energy clipping of the KFAC optimiser.
#
# :func:`OmegaQMC.vmcopt_nn_kfac.clip_local_energies` must reproduce
# FermiNet's ``loss.clip_local_values`` (window ``c +- scale * mean|E_L -
# c|``, ``c`` the batch mean or median; residuals centred at the mean of
# the clipped or of the unclipped energies), and the driver must feed
# the clipped residuals to the KFAC step only while logging unclipped
# energies.  Clipping off (the default) must leave the residuals
# untouched.
# ---------------------------------------------------------------------------


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
def _h2_clip():
    return Mole_custom.from_arrays(
        charges=[1, 1], coords=[[0, 0, -0.7], [0, 0, 0.7]],
        n_up=1, n_down=1,
    )


@pytest.fixture(scope="module")
def drv_clip():
    return get_vmcopt_nn_func(_h2_clip(), 'ferminet', jax.random.key(0))


def _run_with_outlier(drv_clip, tmp_path, monkeypatch, name):
    """Two iterations with walker 0's E_L shifted by +1000 Ha.

    Returns the E_L batches, the residuals handed to the KFAC step,
    the driver output and the log text.
    """
    orig_e, orig_acc = (
        drv_clip.compute_batch_energy,
        drv_clip._accumulate_factors,
    )
    seen = {'e': [], 'de': []}

    def with_outlier(walkers, params):
        e = orig_e(walkers, params).at[0].add(1000.0)
        seen['e'].append(np.asarray(e))
        return e

    def spy(params, walkers_r, de_r):
        seen['de'].append(np.asarray(de_r).reshape(-1))
        return orig_acc(params, walkers_r, de_r)

    monkeypatch.setattr(drv_clip, 'compute_batch_energy', with_outlier)
    monkeypatch.setattr(drv_clip, '_accumulate_factors', spy)
    log = str(tmp_path / f'{name}.log')
    out = drv_clip(jax.random.key(1), num_iters=2, num_walkers=32,
              num_blocks_equil=1, num_steps_per_block=5,
              factor_chunk_size=16, prefix=str(tmp_path / name),
              verbose=1, fname_log=log)
    with open(log) as f:
        return seen, out, f.read()


def test_driver_default_is_unclipped(drv_clip, tmp_path, monkeypatch):
    assert drv_clip.clip_local_energy == 0.0
    seen, (_, edata), log = _run_with_outlier(drv_clip, tmp_path,
                                              monkeypatch, 'off')
    for e, de in zip(seen['e'], seen['de']):
        np.testing.assert_allclose(de, e - e.mean(), atol=1e-10)
    assert edata['n_clipped_history'] == []
    assert '# clip_local_energy=0 (off)' in log


@pytest.mark.parametrize("median", [False, True])
def test_driver_clips_gradient_only(drv_clip, tmp_path, monkeypatch, median):
    monkeypatch.setattr(drv_clip, 'clip_local_energy', 5.0)
    monkeypatch.setattr(drv_clip, 'clip_median', median)
    seen, (params, edata), log = _run_with_outlier(
        drv_clip, tmp_path, monkeypatch, f'on{int(median)}')
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


# ---------------------------------------------------------------------------
# From test_vmcopt_nn_kfac_nonfinite.py
#
# Tests for the non-finite guard of the KFAC optimiser.
#
# A walker with a non-finite local energy, or a step that comes out
# non-finite, must not poison the parameters: the guard replaces such
# walkers by copies of finite ones, rejects non-finite steps, records the
# events in ``{prefix}.nonfinite.h5`` and gives up after too many
# consecutive rejections.  The faults are injected by wrapping the
# driver's compiled kernels.
# ---------------------------------------------------------------------------


def _h2_nonfinite():
    return Mole_custom.from_arrays(
        charges=[1, 1], coords=[[0, 0, -0.7], [0, 0, 0.7]],
        n_up=1, n_down=1,
    )


@pytest.fixture(scope="module")
def drv_nonfinite():
    return get_vmcopt_nn_func(_h2_nonfinite(), 'ferminet', jax.random.key(0))


def _run(drv_nonfinite, tmp_path, num_iters, name):
    prefix = str(tmp_path / name)
    log = str(tmp_path / f'{name}.log')
    out = drv_nonfinite(jax.random.key(1), num_iters=num_iters, num_walkers=32,
              num_blocks_equil=1, num_steps_per_block=5,
              factor_chunk_size=16, prefix=prefix, verbose=1,
              fname_log=log)
    with open(log) as f:
        return out, prefix, f.read()


def _all_finite(tree):
    return all(bool(jnp.all(jnp.isfinite(x)))
               for x in jax.tree.leaves(tree))


def test_nonfinite_local_energy_walkers_replaced(drv_nonfinite, tmp_path,
                                                 monkeypatch):
    orig = drv_nonfinite.compute_batch_energy
    calls = {'n': 0}

    def faulty(walkers, params):
        e = orig(walkers, params)
        calls['n'] += 1
        if calls['n'] == 2:                    # second iteration
            e = e.at[:3].set(jnp.nan).at[3].set(jnp.inf)
        return e

    monkeypatch.setattr(drv_nonfinite, 'compute_batch_energy', faulty)
    (params, edata), prefix, log = _run(drv_nonfinite, tmp_path, 3, 'eloc')
    assert _all_finite(params)
    assert np.all(np.isfinite(edata['E_history']))
    assert '4 walker(s) with a non-finite local energy replaced' in log
    with h5py.File(prefix + '.nonfinite.h5') as f:
        (key,) = list(f)
        assert f[key].attrs['kind'] == 'e_loc'
        assert f[key].attrs['iteration'] == 1
        assert f[key]['walkers'].shape == (4, 2, 3)


def test_nonfinite_step_rejected(drv_nonfinite, tmp_path, monkeypatch):
    orig = drv_nonfinite._kfac_apply
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

    monkeypatch.setattr(drv_nonfinite, '_kfac_apply', faulty)
    (params, _), prefix, log = _run(drv_nonfinite, tmp_path, 3, 'step')
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


def test_too_many_rejections_raise(drv_nonfinite, tmp_path, monkeypatch):
    orig = drv_nonfinite._kfac_apply

    def always_bad(params, *args):
        out = orig(params, *args)
        return (jax.tree.map(lambda x: x * jnp.nan, out[0]),) \
            + tuple(out[1:])

    monkeypatch.setattr(drv_nonfinite, '_kfac_apply', always_bad)
    monkeypatch.setattr(kfac_mod, '_NONFINITE_MAX_CONSECUTIVE', 3)
    with pytest.raises(FloatingPointError):
        _run(drv_nonfinite, tmp_path, 10, 'raise')


def test_mostly_nonfinite_raises(drv_nonfinite, tmp_path, monkeypatch):
    orig = drv_nonfinite.compute_batch_energy

    def faulty(walkers, params):
        e = orig(walkers, params)
        return e.at[: e.shape[0] * 3 // 4].set(jnp.nan)

    monkeypatch.setattr(drv_nonfinite, 'compute_batch_energy', faulty)
    with pytest.raises(FloatingPointError):
        _run(drv_nonfinite, tmp_path, 2, 'mostly')
