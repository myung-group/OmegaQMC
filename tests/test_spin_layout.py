"""OmegaQMC's electron layout with more than one unpaired electron.

Walkers alternate spin-up and spin-down electrons and put the
``|n_up - n_down|`` unpaired electrons last
(:mod:`OmegaQMC.spin_layout`).  These tests check

* the layout indices, and that plain interleaving (``n_up - n_down``
  of 0 or 1) is reproduced exactly, including the strided-slice
  grouping used before;
* that NN wavefunctions of high-spin systems (C, N and O atoms, the NH
  triplet) group the layout correctly: exchanging two electrons of the
  same spin flips the sign and keeps ``|psi|``, exchanging opposite
  spins does not;
* the forward-Laplacian path (FermiNet and PsiFormer, Li and N atoms)
  and the hand-written PsiFormer backward pass against autodiff;
* FermiNet pretraining (initial electrons, a few Adam steps) and a few
  KFAC iterations for high-spin systems;
* the compressed-sensing helpers that read NN walkers.
"""
import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from OmegaQMC.spin_layout import (  # noqa: E402
    grouping_permutation, is_plain_interleaved, layout_permutation,
    spin_indices, spin_of_position, to_grouped, to_layout,
)
from OmegaQMC.utils import Mole_custom  # noqa: E402

SEED = 1104


# ---------------------------------------------------------------------
# Layout indices
# ---------------------------------------------------------------------
@pytest.mark.parametrize("n_up,n_down,up,down", [
    (0, 0, [], []),
    (1, 0, [0], []),
    (1, 1, [0], [1]),
    (2, 1, [0, 2], [1]),
    (3, 2, [0, 2, 4], [1, 3]),
    (2, 0, [0, 1], []),
    (4, 2, [0, 2, 4, 5], [1, 3]),
    (5, 2, [0, 2, 4, 5, 6], [1, 3]),
    (5, 3, [0, 2, 4, 6, 7], [1, 3, 5]),
    (6, 3, [0, 2, 4, 6, 7, 8], [1, 3, 5]),
    (1, 2, [0], [1, 2]),
    (2, 5, [0, 2], [1, 3, 4, 5, 6]),
])
def test_spin_indices(n_up, n_down, up, down):
    u, d = spin_indices(n_up, n_down)
    np.testing.assert_array_equal(u, up)
    np.testing.assert_array_equal(d, down)
    s = spin_of_position(n_up, n_down)
    assert list(np.nonzero(s == 0)[0]) == up
    assert list(np.nonzero(s == 1)[0]) == down
    perm = grouping_permutation(n_up, n_down)
    np.testing.assert_array_equal(perm, up + down)
    np.testing.assert_array_equal(
        perm[layout_permutation(n_up, n_down)], np.arange(n_up + n_down))


@pytest.mark.parametrize("n_up,n_down", [(1, 1), (2, 1), (5, 5), (7, 6)])
def test_plain_interleaving_unchanged(n_up, n_down):
    assert is_plain_interleaved(n_up, n_down)
    n_e = n_up + n_down
    np.testing.assert_array_equal(
        grouping_permutation(n_up, n_down),
        np.concatenate([np.arange(0, n_e, 2), np.arange(1, n_e, 2)]))
    x = jnp.asarray(np.random.default_rng(0).normal(size=(4, n_e, 3)))
    old = jnp.concatenate([x[:, ::2], x[:, 1::2]], axis=1)
    np.testing.assert_array_equal(to_grouped(x, n_up, n_down), old)
    np.testing.assert_array_equal(to_grouped(x[0], n_up, n_down), old[0])


@pytest.mark.parametrize("n_up,n_down", [(4, 2), (5, 2), (5, 3), (2, 5),
                                         (3, 0)])
def test_grouping_round_trip(n_up, n_down):
    assert not is_plain_interleaved(n_up, n_down)
    n_e = n_up + n_down
    x = jnp.asarray(np.random.default_rng(1).normal(size=(2, 3, n_e, 3)))
    g = to_grouped(x, n_up, n_down)
    up, down = spin_indices(n_up, n_down)
    np.testing.assert_array_equal(g[..., :n_up, :], x[..., up, :])
    np.testing.assert_array_equal(g[..., n_up:, :], x[..., down, :])
    np.testing.assert_array_equal(to_layout(g, n_up, n_down), x)
    with pytest.raises(ValueError):
        to_grouped(x[..., :-1, :], n_up, n_down)


# ---------------------------------------------------------------------
# NN wavefunctions of high-spin systems
# ---------------------------------------------------------------------
def _atom(z, n_up, n_down):
    return Mole_custom.from_arrays(charges=[z], coords=[[0.0, 0.0, 0.0]],
                                   n_up=n_up, n_down=n_down)


def _nh():
    """NH triplet (5, 3)."""
    return Mole_custom.from_arrays(
        charges=[7, 1], coords=[[0.0, 0.0, 0.0], [0.0, 0.3, 1.96]],
        n_up=5, n_down=3,
    )


MOLS = {
    'C': lambda: _atom(6, 4, 2),
    'N': lambda: _atom(7, 5, 2),
    'O': lambda: _atom(8, 5, 3),
    'NH': _nh,
}


def _small(name):
    from OmegaQMC.psi.nn.config import load_nn_config
    return dataclasses.replace(
        load_nn_config(name), embedding_dim=32, two_particle_stream_dim=8,
        n_determinants=2, n_interactions=2,
    )


def _walkers(mol, n, scale=1.2, seed=SEED):
    rng = np.random.default_rng(seed)
    nuc = np.asarray(mol.atom_coords())
    n_e = mol.n_up + mol.n_down
    at = rng.integers(0, len(nuc), size=(n, n_e))
    return jnp.asarray(nuc[at] + scale * rng.normal(size=(n, n_e, 3)))


def _swap(x, i, j):
    idx = np.arange(x.shape[-2])
    idx[i], idx[j] = j, i
    return x[..., idx, :]


@pytest.mark.parametrize("config", ['ferminet', 'psiformer'])
@pytest.mark.parametrize("mol_name", list(MOLS))
def test_exchange_symmetry(mol_name, config):
    from OmegaQMC.psi.nn.adapter import make_nn_log_psi
    mol = MOLS[mol_name]()
    log_psi, params, _, _ = make_nn_log_psi(
        _small(config), mol, jax.random.key(SEED), compute_dtype=None)
    nuc = jnp.asarray(mol.atom_coords())
    signed = jax.vmap(lambda r: log_psi.signed(r, nuc, params))
    x = _walkers(mol, 6)
    s0, l0 = signed(x)
    assert bool(jnp.all(jnp.isfinite(l0)))
    up, down = spin_indices(mol.n_up, mol.n_down)
    # same spin: a paired and an unpaired spin-up electron, two
    # unpaired ones, and two spin-down electrons
    pairs_same = [(up[0], up[-1]), (up[-2], up[-1]), (down[0], down[-1])]
    for i, j in pairs_same:
        s1, l1 = signed(_swap(x, i, j))
        np.testing.assert_allclose(l1, l0, rtol=0, atol=1e-9)
        np.testing.assert_array_equal(s1, -s0)
    # opposite spins: |psi| changes
    for i, j in [(up[0], down[0]), (up[-1], down[-1])]:
        _, l1 = signed(_swap(x, i, j))
        assert float(jnp.max(jnp.abs(l1 - l0))) > 1e-6


@pytest.mark.parametrize("config", ['ferminet', 'psiformer'])
@pytest.mark.parametrize("mol_name", ['Li', 'N'])
def test_forward_laplacian_open_shell(mol_name, config):
    """Forward-Laplacian (VGL) path vs linearize for the Li (2, 1) and
    N (5, 2) atoms.  FermiNet's per-spin edge blocks differ in size
    whenever n_up != n_down, which the VGL edge update once rejected."""
    from OmegaQMC.psi.nn.adapter import make_nn_log_psi
    from OmegaQMC.psi.nn.physics import laplacian
    mol = _atom(3, 2, 1) if mol_name == 'Li' else MOLS['N']()
    log_psi, params, _, lap_grad = make_nn_log_psi(
        _small(config), mol, jax.random.key(SEED), compute_dtype=None)
    assert lap_grad.use_vgl
    nuc = jnp.asarray(mol.atom_coords())
    n_e = mol.n_up + mol.n_down
    for r in _walkers(mol, 3):
        lap, grad = lap_grad(r, nuc, params)
        lap_ref, grad_ref = laplacian(
            lambda f: log_psi(f.reshape(n_e, 3), nuc, params))(r.reshape(-1))
        np.testing.assert_allclose(lap, lap_ref, rtol=1e-8, atol=1e-8)
        # the VGL gradient is in spin-grouped electron order
        np.testing.assert_allclose(
            grad.reshape(n_e, 3),
            to_grouped(grad_ref.reshape(n_e, 3), mol.n_up, mol.n_down),
            rtol=1e-8, atol=1e-8)


def test_backward_psiformer_open_shell():
    from OmegaQMC.psi.nn.adapter import cast_floats, make_nn_log_psi
    from OmegaQMC.psi.nn.backward_psiformer import make_psiformer_backward
    mol = MOLS['N']()
    cfg = _small('psiformer')
    key = jax.random.key(SEED)
    log_psi, params, _, _ = make_nn_log_psi(cfg, mol, key,
                                            compute_dtype=None)
    params = cast_floats(params, jnp.float64)
    bwd, _ = make_psiformer_backward(cfg, mol, key, compute_dtype=None)
    nuc = jnp.asarray(mol.atom_coords())
    x = _walkers(mol, 4)
    lp, grads, _ = jax.vmap(lambda r: bwd(r, nuc, params))(x)
    lp_ref = jax.vmap(lambda r: log_psi(r, nuc, params))(x)
    g_ref = jax.vmap(
        lambda r: jax.grad(lambda p: log_psi(r, nuc, p))(params))(x)
    np.testing.assert_allclose(lp, lp_ref, rtol=0, atol=1e-10)
    for a, b in zip(jax.tree.leaves(grads), jax.tree.leaves(g_ref)):
        scale = float(jnp.max(jnp.abs(b)))
        assert float(jnp.max(jnp.abs(a - b))) <= 1e-10 * max(scale, 1e-300)


# ---------------------------------------------------------------------
# Pretraining and optimisation
# ---------------------------------------------------------------------
def test_init_electrons_high_spin():
    """NH (5, 3): N keeps (5, 2), H is flipped to (0, 1)."""
    from OmegaQMC.pretrain_nn import init_electrons
    mol = _nh()
    x = init_electrons(mol, 2, jax.random.key(0), init_width=0.0)
    g = np.asarray(to_grouped(x, mol.n_up, mol.n_down))[0]
    nuc = np.asarray(mol.coords)
    atom_of = np.argmin(
        np.linalg.norm(g[:, None] - nuc[None], axis=-1), axis=1)
    assert list(atom_of[:5]) == [0] * 5
    assert sorted(atom_of[5:]) == [0, 0, 1]


@pytest.mark.parametrize("mol_name", ['N', 'NH'])
def test_pretrain_high_spin(mol_name):
    from OmegaQMC.psi.nn.adapter import make_nn_log_psi
    from OmegaQMC.pretrain_nn import pretrain_hf
    mol = MOLS[mol_name]()
    log_psi, params, _, _ = make_nn_log_psi(
        _small('ferminet'), mol, jax.random.key(SEED))
    _, walkers, info = pretrain_hf(
        log_psi, mol, params, jax.random.key(1), iterations=30,
        num_walkers=64, verbose=0)
    assert walkers.shape == (64, mol.n_up + mol.n_down, 3)
    assert info['hf_converged']
    assert np.all(np.isfinite(info['loss']))
    assert info['loss'][-5:].mean() < info['loss'][:5].mean()


def test_kfac_high_spin(tmp_path):
    from OmegaQMC.vmcopt_nn_kfac import get_vmcopt_nn_func
    mol = MOLS['N']()
    drv = get_vmcopt_nn_func(mol, _small('ferminet'), jax.random.key(0))
    params, edata = drv(
        jax.random.key(1), num_iters=3, num_walkers=32,
        num_blocks_equil=2, num_steps_per_block=5, factor_chunk_size=16,
        pretrain_iters=5, prefix=str(tmp_path / 'n'), verbose=0)
    assert np.all(np.isfinite(edata['E_history']))
    assert all(bool(jnp.all(jnp.isfinite(v)))
               for v in jax.tree.leaves(params))


# ---------------------------------------------------------------------
# Compressed-sensing readers of NN walkers
# ---------------------------------------------------------------------
def test_cs_interleaved_helpers_high_spin():
    from OmegaQMC.cs.estimators import _interleaved_to_grouped_indices
    from OmegaQMC.cs.jastrow_extract import kato_cusp_jastrow
    n_up, n_down = 5, 2
    np.testing.assert_array_equal(
        _interleaved_to_grouped_indices(n_up, n_down),
        [0, 2, 4, 5, 6, 1, 3])
    w = np.asarray(_walkers(_atom(7, n_up, n_down), 5))
    perm = grouping_permutation(n_up, n_down)
    np.testing.assert_allclose(
        kato_cusp_jastrow(w, n_up, n_down, layout='interleaved'),
        kato_cusp_jastrow(w[:, perm], n_up, n_down, layout='grouped'),
        rtol=1e-14)
