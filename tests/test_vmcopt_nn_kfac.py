"""Unit tests for the molecular KFAC VMC optimiser.

Covers the pieces of :mod:`OmegaQMC.vmcopt_nn_kfac` that are cheap
to check without running a full optimisation: the assignment of every
parameter to a Kronecker or a diagonal block, the shapes of the
factor sums and state, the equivalence of the chunked (scanned)
accumulation to a single-chunk evaluation, and the dtype contract of
the parameter update.  The agreement with kfac-jax itself is in
``tests/test_vmcopt_nn_kfac_jax.py``.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from OmegaQMC.utils import Mole_custom
from OmegaQMC.vmcopt_nn_kfac import _VMCOptDriverNN_KFAC


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
