"""Tests for HF pretraining of molecular NN wavefunctions.

:mod:`OmegaQMC.pretrain_nn` follows FermiNet's pretraining (UHF
target, block-diagonal for a full determinant, Adam, one Metropolis
move per iteration on the network's |psi|^2).  These tests check

* the neutral-atom spin configurations and FermiNet's initial
  electron assignment;
* that the ``.orbitals`` accessor of the adapter reproduces
  ``log|psi|`` (so the pretrained matrices are the determinant's);
* the target construction against PySCF;
* that the loss falls on H2 and LiH;
* the KFAC driver hook: pretraining on a fresh start, none on resume.
"""
import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from OmegaQMC.utils import Mole_custom  # noqa: E402
from OmegaQMC.psi.nn.adapter import make_nn_log_psi  # noqa: E402
from OmegaQMC.pretrain_nn import (  # noqa: E402
    HartreeFockTarget,
    _target_matrix,
    atomic_spin_config,
    init_electrons,
    pretrain_hf,
)


def _h2():
    return Mole_custom.from_arrays(
        charges=[1, 1], coords=[[0, 0, -0.7], [0, 0, 0.7]],
        n_up=1, n_down=1,
    )


def _lih():
    return Mole_custom.from_arrays(
        charges=[3, 1], coords=[[0, 0, 0], [0.1, 0, 3.0]],
        n_up=2, n_down=2,
    )


def _n2():
    return Mole_custom.from_arrays(
        charges=[7, 7], coords=[[0, 0, -1.0], [0, 0, 1.0]],
        n_up=7, n_down=7,
    )


def _grouped(x):
    return jnp.concatenate([x[..., ::2, :], x[..., 1::2, :]], axis=-2)


@pytest.mark.parametrize("z,expected", [
    (1, (1, 0)), (2, (1, 1)), (3, (2, 1)), (6, (4, 2)), (7, (5, 2)),
    (8, (5, 3)), (10, (5, 5)), (24, (15, 9)), (29, (15, 14)),
])
def test_atomic_spin_config(z, expected):
    assert atomic_spin_config(z) == expected


@pytest.mark.parametrize("mol_fn,per_atom", [
    (_h2, [(1, 0), (0, 1)]),
    (_n2, [(5, 2), (2, 5)]),
])
def test_init_electrons_assignment(mol_fn, per_atom):
    mol = mol_fn()
    x = init_electrons(mol, 3, jax.random.key(0), init_width=0.0)
    g = np.asarray(_grouped(x))[0]
    nuc = np.asarray(mol.coords)
    atom_of = np.argmin(
        np.linalg.norm(g[:, None] - nuc[None], axis=-1), axis=1,
    )
    nu = mol.n_up
    counts = sorted(
        (int(np.sum(atom_of[:nu] == a)), int(np.sum(atom_of[nu:] == a)))
        for a in range(len(nuc))
    )
    assert counts == sorted(per_atom)
    # Gaussian noise of the requested width
    xs = init_electrons(mol, 4000, jax.random.key(1), init_width=1.0)
    d = np.asarray(xs - init_electrons(mol, 1, jax.random.key(2),
                                       init_width=0.0))
    assert abs(d.std() - 1.0) < 0.02


@pytest.mark.parametrize("config", ['ferminet', 'psiformer'])
@pytest.mark.parametrize("mol_fn", [_h2, _lih])
def test_orbitals_reproduce_log_psi(config, mol_fn):
    mol = mol_fn()
    nuc = jnp.asarray(mol.coords, dtype=jnp.float64)
    log_psi, params, _, _ = make_nn_log_psi(
        config, mol, jax.random.key(0), compute_dtype=None,
    )
    assert log_psi.full_determinant
    x = init_electrons(mol, 5, jax.random.key(3))
    for b in range(5):
        up, dn, rg = log_psi.orbitals(x[b], nuc, params)
        np.testing.assert_allclose(rg, _grouped(x[b]))
        dets = jnp.linalg.det(jnp.concatenate([up, dn], axis=-2))
        lp = jnp.log(jnp.abs(jnp.sum(dets)))
        if config == 'ferminet':
            np.testing.assert_allclose(
                lp, log_psi(x[b], nuc, params), rtol=1e-10,
            )


def test_hf_target_matches_pyscf():
    mol = _lih()
    tgt = HartreeFockTarget(mol)
    assert tgt.converged
    x = np.asarray(_grouped(init_electrons(mol, 4, jax.random.key(0))))
    t_up, t_dn = tgt(x)
    assert t_up.shape == (4, 2, 2) and t_dn.shape == (4, 2, 2)
    ao = tgt.mol.eval_gto('GTOval_sph', x.reshape(-1, 3)).reshape(4, 4, -1)
    np.testing.assert_allclose(t_up, ao[:, :2] @ tgt.mf.mo_coeff[0][:, :2])
    np.testing.assert_allclose(t_dn, ao[:, 2:] @ tgt.mf.mo_coeff[1][:, :2])
    full = np.asarray(_target_matrix(jnp.asarray(t_up),
                                     jnp.asarray(t_dn), True))
    assert full.shape == (4, 4, 4)
    np.testing.assert_allclose(full[:, :2, :2], t_up)
    np.testing.assert_allclose(full[:, 2:, 2:], t_dn)
    assert np.all(full[:, :2, 2:] == 0) and np.all(full[:, 2:, :2] == 0)


@pytest.mark.parametrize("mol_fn", [_h2, _lih])
def test_pretraining_reduces_loss(mol_fn):
    mol = mol_fn()
    log_psi, params, _, _ = make_nn_log_psi(
        'ferminet', mol, jax.random.key(0), compute_dtype='float32',
    )
    p, w, info = pretrain_hf(
        log_psi, mol, params, jax.random.key(1),
        iterations=200, num_walkers=64, verbose=0,
    )
    loss = info['loss']
    assert np.all(np.isfinite(loss))
    assert loss[-20:].mean() < 0.3 * loss[:5].mean()
    assert 0.3 < info['acceptance'] < 1.0
    assert w.shape == (64, mol.n_up + mol.n_down, 3)
    for a, b in zip(jax.tree.leaves(params), jax.tree.leaves(p)):
        assert a.dtype == b.dtype and a.shape == b.shape


def test_kfac_pretrain_hook(tmp_path):
    from OmegaQMC.vmcopt_nn_kfac import get_vmcopt_nn_func
    mol = _h2()
    prefix = str(tmp_path / 'nnopt')
    opt = get_vmcopt_nn_func(mol, 'ferminet', jax.random.key(0))
    out = opt(jax.random.key(1), num_iters=2, num_walkers=32,
              num_blocks_equil=1, num_steps_per_block=5,
              pretrain_iters=5, prefix=prefix, verbose=0,
              fname_log=str(tmp_path / 'log1.txt'))
    assert out[0] is not None
    assert os.path.isfile(prefix + '.pretrain.h5')
    assert os.path.isfile(prefix + '.chk.h5')
    # Resume: no second pretraining.
    os.remove(prefix + '.pretrain.h5')
    out = opt(jax.random.key(2), num_iters=1, num_walkers=32,
              num_blocks_equil=1, num_steps_per_block=5,
              pretrain_iters=5, prefix=prefix, verbose=1,
              fname_log=str(tmp_path / 'log2.txt'))
    assert out[0] is not None
    assert not os.path.isfile(prefix + '.pretrain.h5')
    with open(tmp_path / 'log2.txt') as f:
        txt = f.read()
    assert 'HF pretraining' not in txt and 'Resuming' in txt
