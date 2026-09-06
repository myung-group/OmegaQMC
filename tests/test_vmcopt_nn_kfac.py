"""Unit tests for the molecular KFAC VMC optimiser.

Covers the pieces of :mod:`OmegaQMC.vmcopt_nn_kfac` that are cheap
to check without running a full optimisation: the parameter
classification, the Kronecker-factor shapes for both extraction
paths, the equivalence of the chunked (scanned) accumulation to a
single-chunk evaluation, and the dtype contract of the parameter
update.

The full training loop is exercised by an actual H2 run rather
than as a unit test (it takes minutes of GPU compute).
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
    return _VMCOptDriverNN_KFAC(
        _h2(), "psiformer", jax.random.key(0),
    )


@pytest.fixture(scope="module")
def batch(drv):
    w = drv.initialize_walkers(jax.random.key(1), 8)
    e = drv.compute_batch_energy(w, drv.init_params)
    return w, e - jnp.mean(e)


def test_param_classification_covers_all_params(drv):
    """Every parameter is either a Linear kernel or generic."""
    assert drv._linear_count > 0
    assert (
        drv._linear_params + sum(drv._generic_sizes)
        == drv.n_params
    )


def test_factor_shapes_match_kernel_layout(drv, batch):
    """A_out is (out, out), G_in is (in, in), dW_loss is (in, out).

    NNX stores a Linear kernel as ``(in, out)``, so the factor that
    left-multiplies the step must be the ``(in, in)`` one.
    """
    w, de = batch
    A, G, dW, cnt, gen = drv._factor_chunk(
        drv.init_params, w, de, {},
    )
    for layer, (in_, out_) in drv._kernel_shapes.items():
        assert A[layer].shape == (out_, out_), layer
        assert G[layer].shape == (in_, in_), layer
        assert dW[layer].shape == (in_, out_), layer
    assert gen.shape == (w.shape[0], sum(drv._generic_sizes))


def test_factor_state_matches_factor_shapes(drv, batch):
    """The identity-initialised EMA state is conformable."""
    w, de = batch
    A, G, _dW, _cnt, _gen = drv._factor_chunk(
        drv.init_params, w, de, {},
    )
    A_state, G_state = drv._init_factor_state()
    for layer in A:
        assert A_state[layer].shape == A[layer].shape
        assert G_state[layer].shape == G[layer].shape


def _rel_max_diff(a, b):
    """Max elementwise difference relative to the array's scale.

    The Kronecker factors are accumulated in float32 (the dtype the
    Linear kernels are built in), so individual near-zero entries
    carry absolute round-off of order ``eps * ||A||``.  Comparing
    against the matrix scale rather than elementwise is the
    meaningful check here.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return np.abs(a - b).max() / max(np.abs(a).max(), 1e-30)


def test_chunked_accumulation_matches_single_chunk(drv, batch):
    """The scanned accumulation reproduces one unchunked pass.

    Guards the walker-chunking that decouples the walker count from
    the parameter count: factor *sums* must be chunk-invariant.
    """
    w, de = batch
    n = w.shape[0]
    A1, G1, dW1, cnt1, gen1 = drv._factor_chunk(
        drv.init_params, w, de, {},
    )
    A2, G2, dW2, cnt2, gen2 = drv._accumulate_factors(
        drv.init_params,
        w.reshape((2, n // 2) + w.shape[1:]),
        de.reshape(2, n // 2),
    )
    tol = 1.0e-5          # a few float32 eps
    for layer in A1:
        assert _rel_max_diff(A1[layer], A2[layer]) < tol, layer
        assert _rel_max_diff(G1[layer], G2[layer]) < tol, layer
        assert _rel_max_diff(dW1[layer], dW2[layer]) < tol, layer
        assert float(cnt1[layer]) == float(cnt2[layer])
    assert _rel_max_diff(gen1, gen2) < tol


def test_update_preserves_pytree_and_dtypes(drv, batch):
    """The step moves the params without promoting their dtype.

    NNX builds the network in float32 while OmegaQMC enables x64, so
    a float64 KFAC step written back verbatim would silently promote
    the whole network — which costs roughly an order of magnitude in
    per-walker gradient time on a consumer GPU.
    """
    w, de = batch
    p = drv.init_params
    A, G, dW, cnt, gen = drv._factor_chunk(p, w, de, {})
    A_hat = {k: A[k] / cnt[k] for k in A}
    G_hat = {k: G[k] / cnt[k] for k in G}
    dW_hat = {k: dW[k] / float(w.shape[0]) for k in dW}
    A_state, G_state = drv._init_factor_state()
    scale = {k: 1.0 for k in A}

    new_p, _nA, _nG, norm, clip, _dg, _dF = drv._kfac_apply(
        p, A_hat, G_hat, dW_hat, gen, de, A_state, G_state,
        jnp.asarray(0.05), jnp.asarray(1.0e-3), scale,
    )

    assert (
        jax.tree_util.tree_structure(new_p)
        == jax.tree_util.tree_structure(p)
    )
    for old, new in zip(jax.tree.leaves(p), jax.tree.leaves(new_p)):
        assert old.dtype == new.dtype, (old.dtype, new.dtype)
        assert old.shape == new.shape
    moved = sum(
        float(jnp.sum(jnp.abs(a - b)))
        for a, b in zip(jax.tree.leaves(new_p), jax.tree.leaves(p))
    )
    assert moved > 0.0
    assert float(norm) > 0.0
    assert 0.0 < float(clip) <= 1.0


def test_capture_path_shapes(batch):
    """The exact per-electron path yields the same factor shapes."""
    d = _VMCOptDriverNN_KFAC(
        _h2(), "psiformer", jax.random.key(0),
        capture_activations=True,
    )
    w = d.initialize_walkers(jax.random.key(1), 4)
    e = d.compute_batch_energy(w, d.init_params)
    de = e - jnp.mean(e)
    cap = {
        k: v for k, v in d._capture_for(d.init_params, w).items()
        if k in d._layers
    }
    assert cap, "no Linear inputs were captured"
    A, G, dW, cnt, _gen = d._factor_chunk(d.init_params, w, de, cap)
    for layer in cap:
        in_, out_ = d._kernel_shapes[layer]
        assert A[layer].shape == (out_, out_), layer
        assert G[layer].shape == (in_, in_), layer
        # Population is (walkers x electrons), not just walkers.
        assert float(cnt[layer]) >= w.shape[0]
