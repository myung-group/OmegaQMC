"""``kfac_style='kfac_jax'`` and the ``ferminet_deepmind`` ansatz against
DeepMind's FermiNet and kfac-jax.

``tests/ferminet_kfac_reference.npz`` was written by
``tools/ferminet_kfac_reference.py`` (ferminet e9d5a60, kfac-jax 0.0.8,
float64): parameters of a small DeepMind FermiNet, log|psi| and E_L on
16 walkers, and the parameters after a few kfac-jax steps on those
walkers.  The parameters are transplanted into OmegaQMC's
``ferminet_deepmind`` network.  The tolerances allow for OmegaQMC's
safe norm, ``sqrt(eps + r^2)``, which puts the self-pair distance at
``sqrt(eps) ~ 1.5e-8`` instead of DeepMind's exact zero.
"""
import dataclasses
import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from OmegaQMC.utils import Mole_custom
from OmegaQMC.spin_layout import to_layout
from OmegaQMC.psi.nn.adapter import make_nn_log_psi
from OmegaQMC.psi.nn.config import load_nn_config
from OmegaQMC.psi.nn.kfac_tap import leaf_path_str
from OmegaQMC.vmcopt_nn_kfac import (
    _VMCOptDriverNN_KFAC,
    clip_local_energies,
    get_vmcopt_nn_func,
)

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
        kfac_style='kfac_jax', lr=0.05, lr_decay=1.0e4, damping=1.0e-3,
        ema_decay=0.95, norm_constraint=None if np.isinf(nc) else nc,
    )
    assert drv.backward == 'tap'
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
                               nn_dtype=None, kfac_style='kfac_jax')
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


def test_kfac_jax_option_validation(ref):
    mol = _mol(ref, 'a')
    with pytest.raises(ValueError):
        _VMCOptDriverNN_KFAC(mol, _config(ref), jax.random.key(0),
                             kfac_style='kfac-jax')
    with pytest.raises(ValueError):
        _VMCOptDriverNN_KFAC(mol, _config(ref), jax.random.key(0),
                             kfac_style='kfac_jax',
                             capture_activations=True)


def test_kfac_jax_driver_runs(ref, tmp_path):
    """A few iterations of the full loop in float32."""
    mol = _mol(ref, 'a')
    drv = get_vmcopt_nn_func(mol, _config(ref), jax.random.key(0),
                             kfac_style='kfac_jax', clip_local_energy=5.0)
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
    assert 'backward=tap' in log.read_text()
