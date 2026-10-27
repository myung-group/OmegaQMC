"""Nuclear cusp factor (``cusp_nuclei: true``).

On a randomly initialised LiH PsiFormer in float64:

* the factor adds ``+sum Z / (alpha (1 + alpha r))`` to ``log|psi|``,
  so its slope at each nucleus is ``-Z`` (Kato), and ``cusp_alpha``
  reaches it;
* the forward-Laplacian path (``lap_grad.use_vgl``) gives the same
  gradient and Laplacian as autodiff;
* the hand-written backward pass gives the same ``log|psi|``,
  parameter gradients (including a trainable ``nuc_alpha``) and
  ``d log|psi| / dR`` as autodiff, and KFAC picks it by default.
"""
import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from OmegaQMC.utils import Mole_custom
from OmegaQMC.psi.nn.adapter import make_nn_log_psi, cast_floats
from OmegaQMC.psi.nn.backward_psiformer import make_psiformer_backward
from OmegaQMC.psi.nn.config import load_nn_config
from OmegaQMC.vmcopt_nn_kfac import get_vmcopt_nn_func

SEED = 930
RTOL = 1e-10
CHARGES = [3.0, 1.0]


def _lih():
    return Mole_custom.from_arrays(
        charges=CHARGES, coords=[[0.0, 0.0, 0.0], [0.0, 0.0, 3.0]],
        n_up=2, n_down=2,
    )


def _config(nuclei=True, alpha=0.7, trainable=False):
    return dataclasses.replace(
        load_nn_config('psiformer'), cusp_nuclei=nuclei,
        cusp_alpha=alpha, cusp_trainable_alpha=trainable,
    )


def _walkers(n=8, seed=931):
    rng = np.random.default_rng(seed)
    return jnp.asarray(rng.normal(size=(n, 4, 3)) * 1.5)


def _rel(a, b):
    return float(jnp.max(jnp.abs(a - b))) / max(
        float(jnp.max(jnp.abs(b))), 1e-300)


def test_factor_value_and_kato_slope():
    mol = _lih()
    key = jax.random.key(SEED)
    nuc = jnp.asarray(mol.atom_coords())
    alpha = 0.7
    lp_on, params, _, _ = make_nn_log_psi(_config(True, alpha), mol, key)
    lp_off, _, _, _ = make_nn_log_psi(_config(False, alpha), mol, key)
    params = cast_floats(params, jnp.float64)
    x = _walkers()
    diff = jax.vmap(lambda r: lp_on(r, nuc, params)
                    - lp_off(r, nuc, params))(x)
    d = jnp.linalg.norm(x[:, :, None, :] - nuc[None, None], axis=-1)
    Z = jnp.asarray(CHARGES)
    ref = jnp.sum(Z / (alpha * (1.0 + alpha * d)), axis=(1, 2))
    assert _rel(diff, ref) < RTOL

    # slope of the factor along a ray leaving each nucleus: -Z from
    # that nucleus plus the smooth tail of the other one
    n = jnp.array([0.6, 0.0, 0.8])
    for A, z in enumerate(CHARGES):
        def f(t):
            r = x[0].at[0].set(nuc[A] + t * n)
            return lp_on(r, nuc, params) - lp_off(r, nuc, params)
        B = 1 - A
        dB = nuc[A] - nuc[B]
        tail = -CHARGES[B] / (1.0 + alpha * jnp.linalg.norm(dB)) ** 2 * (
            n @ dB / jnp.linalg.norm(dB))
        assert abs(float(jax.grad(f)(1e-9) - tail) + z) < 1e-6


@pytest.mark.parametrize('trainable', [False, True])
def test_vgl_matches_autodiff(trainable):
    mol = _lih()
    key = jax.random.key(SEED)
    cfg = _config(trainable=trainable)
    log_psi, params, _, lap_grad = make_nn_log_psi(cfg, mol, key)
    assert lap_grad.use_vgl
    params = cast_floats(params, jnp.float64)
    nuc = jnp.asarray(mol.atom_coords())
    for r in _walkers(4):
        lap, grad = lap_grad(r, nuc, params)
        f = lambda v: log_psi(v.reshape(4, 3), nuc, params)
        v = r.reshape(-1)
        g_ref = jax.grad(f)(v)
        lap_ref = jnp.trace(jax.hessian(f)(v))
        # lap_grad works in the network's grouped spin order
        g_ref = jnp.concatenate(
            [g_ref.reshape(4, 3)[::2], g_ref.reshape(4, 3)[1::2]],
        ).reshape(-1)
        assert _rel(grad, g_ref) < 1e-9
        assert abs(float(lap - lap_ref)) < 1e-9 * abs(float(lap_ref))


@pytest.mark.parametrize('trainable', [False, True])
def test_backward_matches_autodiff(trainable):
    mol = _lih()
    key = jax.random.key(SEED)
    cfg = _config(trainable=trainable)
    log_psi, params, _, _ = make_nn_log_psi(cfg, mol, key)
    params = cast_floats(params, jnp.float64)
    bwd, _ = make_psiformer_backward(cfg, mol, key)
    nuc = jnp.asarray(mol.atom_coords())
    x = _walkers()
    lp, grads, _ = jax.vmap(lambda r: bwd(r, nuc, params))(x)
    lp_ref = jax.vmap(lambda r: log_psi(r, nuc, params))(x)
    g_ref = jax.vmap(
        lambda r: jax.grad(lambda p: log_psi(r, nuc, p))(params))(x)
    assert _rel(lp, lp_ref) < RTOL
    for a, b in zip(jax.tree.leaves(grads), jax.tree.leaves(g_ref)):
        if float(jnp.max(jnp.abs(b))) == 0.0:
            assert float(jnp.max(jnp.abs(a))) == 0.0
        else:
            assert _rel(a, b) < RTOL
    if trainable:
        paths = [jax.tree_util.keystr(p) for p, _ in
                 jax.tree_util.tree_flatten_with_path(params)[0]]
        assert any('nuc_alpha' in p for p in paths)

    _, dR = jax.vmap(lambda r: bwd.nuclear_grad(r, nuc, params))(x)
    dR_ref = jax.vmap(
        lambda r: jax.grad(log_psi, argnums=1)(r, nuc, params))(x)
    assert _rel(dR, dR_ref) < RTOL


def test_kfac_picks_manual_backward():
    mol = _lih()
    key = jax.random.key(SEED)
    assert get_vmcopt_nn_func(
        mol, _config(), key).backward == 'manual'
