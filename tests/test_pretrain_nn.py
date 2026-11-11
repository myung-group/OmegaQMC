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
* the KFAC driver hook: pretraining on a fresh start, none on resume;
* the optional HF-density sampling (``hf_fraction``, Pfau et al. 2020):
  ``hf_fraction=0`` reproduces the frozen pre-option loop bit for bit,
  the HF chain samples ``|det phi_up|^2 |det phi_down|^2`` (one-electron
  moments against quadrature), and both acceptance rates are sane.
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
    hf_density_state,
    hf_metropolis_step,
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


def _h_atom():
    return Mole_custom.from_arrays(
        charges=[1], coords=[[0, 0, 0]], n_up=1, n_down=0,
    )


def _li_atom():
    return Mole_custom.from_arrays(
        charges=[3], coords=[[0, 0, 0]], n_up=2, n_down=1,
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


# ---------------------------------------------------------------------
# HF-density sampling (hf_fraction, Pfau et al. 2020)
# ---------------------------------------------------------------------
def _load_frozen():
    """The pretraining loop before ``hf_fraction`` (tests/...frozen.py)."""
    import importlib.util
    import pathlib
    import sys
    name = "_pretrain_hf_frozen"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            name, pathlib.Path(__file__).parent / "pretrain_hf_frozen.py",
        )
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
    return sys.modules[name].pretrain_hf


def _strip_times(text):
    import re
    return [re.sub(r"\s+[0-9.]+ s$", "", ln)
            for ln in text.splitlines() if ln.startswith("# pretrain ")]


@pytest.mark.parametrize("mol_fn", [_h2, _lih])
def test_hf_fraction_zero_is_frozen_loop(mol_fn, monkeypatch):
    """hf_fraction=0 (and the default) is the old loop, bit for bit.

    All three runs share one UHF solution: separate ``scf.UHF`` runs
    of the same molecule are not reproducible bit for bit (on the GH200
    nodes LiH and N2 energies differ by up to 3e-14 Ha between runs and
    the MO coefficient matrices by more), which would hide what is
    tested here.
    """
    import io
    import sys
    import OmegaQMC.pretrain_nn as live
    mol = mol_fn()
    log_psi, params, _, _ = make_nn_log_psi(
        'ferminet', mol, jax.random.key(0), compute_dtype='float32',
    )
    frozen = _load_frozen()
    shared = HartreeFockTarget(mol)
    for mod in (live, sys.modules[frozen.__module__]):
        monkeypatch.setattr(mod, 'HartreeFockTarget',
                            lambda *a, **k: shared)
    kw = dict(iterations=12, num_walkers=32, verbose=1, log_every=4)
    runs = []
    for fn, extra in ((frozen, {}),
                      (pretrain_hf, {}),
                      (pretrain_hf, {'hf_fraction': 0.0})):
        buf = io.StringIO()
        runs.append(fn(log_psi, mol, params, jax.random.key(7),
                       fout=buf, **kw, **extra) + (buf.getvalue(),))
    p_ref, w_ref, info_ref, log_ref = runs[0]
    for p, w, info, log in runs[1:]:
        np.testing.assert_array_equal(info['loss'], info_ref['loss'])
        assert info['acceptance'] == info_ref['acceptance']
        assert info['n_hf'] == 0 and np.isnan(info['acceptance_hf'])
        np.testing.assert_array_equal(w, w_ref)
        for a, b in zip(jax.tree.leaves(p), jax.tree.leaves(p_ref)):
            assert a.dtype == b.dtype
            np.testing.assert_array_equal(a, b)
        assert _strip_times(log) == _strip_times(log_ref)
        assert len(_strip_times(log)) == 4


@pytest.mark.parametrize("mol_fn", [_lih, _li_atom, _h_atom])
def test_hf_chain_samples_hf_density(mol_fn):
    """The HF chain's one-electron moments match quadrature of p_HF.

    For orthonormal orbitals the one-electron marginal of
    ``|det phi(r_1..r_n)|^2`` is ``sum_i |phi_i(r)|^2 / n``; its moments
    are integrated on a PySCF molecular grid.  The walkers are
    independent chains, so the standard error is the walker spread.
    The chains start near the nuclei (:func:`init_electrons`); the
    diffuse Li 2s electron and LiH's bond need about 1000 steps of
    width 0.3 bohr to equilibrate (after 300, Li's <r^2> was 7% short).
    """
    from pyscf.dft import gen_grid
    from OmegaQMC.spin_layout import to_grouped

    mol = mol_fn()
    tgt = HartreeFockTarget(mol)
    assert tgt.converged
    nw = 2000
    x = np.asarray(init_electrons(mol, nw, jax.random.key(0)))
    state = hf_density_state(tgt, x)
    key = jax.random.key(1)
    accs = []
    for i in range(3000):
        x, state, a = hf_metropolis_step(
            tgt, x, state, jax.random.fold_in(key, i), 0.3,
        )
        accs.append(a)
    # (c) a sane acceptance for this width, and a consistent cache
    assert 0.2 < np.mean(accs[1000:]) < 0.95
    t_up, t_dn, log_p = hf_density_state(tgt, x)
    np.testing.assert_allclose(state[0], t_up, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(state[1], t_dn, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(state[2], log_p, rtol=1e-12)

    grids = gen_grid.Grids(tgt.mol)
    grids.level = 5
    grids.build()
    ao = tgt.mol.eval_gto('GTOval_sph', grids.coords)
    g = np.asarray(to_grouped(x, mol.n_up, mol.n_down))
    moments = {
        'z': lambda r: r[..., 2],
        'x': lambda r: r[..., 0],
        'r2': lambda r: np.sum(r ** 2, axis=-1),
    }
    nu = mol.n_up
    for c, sl in ((tgt.c_up, slice(0, nu)), (tgt.c_down, slice(nu, None))):
        n = c.shape[1]
        if n == 0:
            continue
        rho = np.sum((ao @ c) ** 2, axis=1)
        assert abs(np.dot(grids.weights, rho) - n) < 1e-5 * n
        for name, f in moments.items():
            exact = np.dot(grids.weights, rho * f(grids.coords)) / n
            per_walker = f(g[:, sl]).mean(axis=1)
            mean = per_walker.mean()
            err = per_walker.std() / np.sqrt(nw)
            assert abs(mean - exact) < 4.0 * err, (name, n, mean, exact, err)


def test_pretrain_hf_fraction():
    """hf_fraction=0.5 and 1: walker order, acceptances, loss, logging."""
    import io
    mol = _lih()
    log_psi, params, _, _ = make_nn_log_psi(
        'ferminet', mol, jax.random.key(0), compute_dtype='float32',
    )
    # Walker order: the |psi|^2 set first, the HF set last, both taken
    # from the same initial walkers.
    _, w0, info0 = pretrain_hf(
        log_psi, mol, params, jax.random.key(1), iterations=0,
        num_walkers=64, hf_fraction=0.5, verbose=0,
    )
    k_init = jax.random.split(jax.random.key(1))[1]
    np.testing.assert_array_equal(w0, init_electrons(mol, 64, k_init))
    assert info0['n_hf'] == 32

    buf = io.StringIO()
    p, w, info = pretrain_hf(
        log_psi, mol, params, jax.random.key(1), iterations=200,
        num_walkers=64, hf_fraction=0.5, verbose=1, log_every=50,
        fout=buf,
    )
    loss = info['loss']
    assert info['n_hf'] == 32
    assert np.all(np.isfinite(loss))
    assert loss[-20:].mean() < 0.3 * loss[:5].mean()
    assert 0.3 < info['acceptance'] < 1.0
    assert 0.3 < info['acceptance_hf'] < 1.0
    assert w.shape == (64, mol.n_up + mol.n_down, 3)
    assert np.all(np.isfinite(np.asarray(w)))
    for a, b in zip(jax.tree.leaves(params), jax.tree.leaves(p)):
        assert a.dtype == b.dtype and a.shape == b.shape
    txt = buf.getvalue()
    assert 'HF-density sampling' in txt
    assert len([ln for ln in txt.splitlines() if 'acc_hf' in ln]) == 5

    # All walkers on the HF density: no |psi|^2 chain at all.
    _, w1, info1 = pretrain_hf(
        log_psi, mol, params, jax.random.key(2), iterations=10,
        num_walkers=16, hf_fraction=1.0, verbose=0,
    )
    assert info1['n_hf'] == 16 and np.isnan(info1['acceptance'])
    assert 0.3 < info1['acceptance_hf'] <= 1.0
    assert w1.shape == (16, mol.n_up + mol.n_down, 3)
    assert np.all(np.isfinite(info1['loss']))

    with pytest.raises(ValueError):
        pretrain_hf(log_psi, mol, params, jax.random.key(3),
                    iterations=1, num_walkers=8, hf_fraction=1.5)


@pytest.mark.parametrize("reinit", [None, False])
def test_kfac_pretrain_hf_fraction(tmp_path, reinit):
    """Driver kwargs: HF fraction passed through; walkers reinitialised
    by default when it is positive."""
    from OmegaQMC.vmcopt_nn_kfac import get_vmcopt_nn_func
    mol = _h2()
    opt = get_vmcopt_nn_func(mol, 'ferminet', jax.random.key(0))
    log = tmp_path / 'log.txt'
    out = opt(jax.random.key(1), num_iters=2, num_walkers=32,
              num_blocks_equil=1, num_steps_per_block=5,
              pretrain_iters=5, pretrain_hf_fraction=0.5,
              pretrain_reinit_walkers=reinit,
              prefix=str(tmp_path / 'nnopt'), verbose=1,
              fname_log=str(log))
    assert out[0] is not None
    assert np.all(np.isfinite(out[1]['E_history']))
    txt = log.read_text()
    assert 'acc_hf' in txt and 'HF-density acceptance' in txt
    assert ('pretrained walkers discarded' in txt) == (reinit is None)
