"""Hand-written PsiFormer backward pass against automatic differentiation.

:func:`OmegaQMC.psi.nn.backward_psiformer.make_psiformer_backward`
returns ``log|psi|``, its parameter gradient and, per Linear layer,
the per-electron inputs ``a`` and output cotangents ``g`` without
using JAX autodiff.  Autodiff is used here only as the reference,
in float64 on a randomly initialised LiH PsiFormer (four electrons,
so same-spin cusp pairs and multi-electron attention are covered):

* ``log|psi|`` and every parameter gradient match ``jax.grad``;
* each walker's per-electron ``g`` matches the cotangent recovered
  from the autodiff per-walker kernel gradient ``M = a^T g`` by an
  undamped solve, ``g = (a a^T)^-1 a M``;
* the KFAC driver with ``backward='manual'`` builds the same energy
  gradient and generic-leaf gradients as the autodiff driver.
* ``nuclear_grad`` gives ``d log|psi| / dR`` as ``jax.grad`` does,
  and the ZVZB force components built with it are unchanged.
"""
import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from OmegaQMC.utils import Mole_custom
from OmegaQMC.psi.nn.adapter import make_nn_log_psi, cast_floats
from OmegaQMC.psi.nn.backward_psiformer import (
    _path_str, make_psiformer_backward,
)
from OmegaQMC.vmcopt_nn_kfac import get_vmcopt_nn_func

SEED = 930
RTOL = 1e-10


def _lih():
    return Mole_custom.from_arrays(
        charges=[3, 1], coords=[[0.0, 0.0, 0.0], [0.0, 0.0, 3.0]],
        n_up=2, n_down=2,
    )


def _pieces(mol):
    key = jax.random.key(SEED)
    log_psi, params, _gd, _lg = make_nn_log_psi(
        'psiformer', mol, key, compute_dtype=None,
    )
    params = cast_floats(params, jnp.float64)
    bwd, names = make_psiformer_backward(
        'psiformer', mol, key, compute_dtype=None,
    )
    nuc = jnp.asarray(mol.atom_coords())
    rng = np.random.default_rng(931)
    walkers = jnp.asarray(rng.normal(size=(8, 4, 3)) * 1.5)
    return log_psi, params, bwd, names, nuc, walkers


def test_backward_matches_autodiff():
    mol = _lih()
    log_psi, params, bwd, names, nuc, walkers = _pieces(mol)
    lp, grads, caps = jax.vmap(lambda r: bwd(r, nuc, params))(walkers)
    lp_ref = jax.vmap(lambda r: log_psi(r, nuc, params))(walkers)
    g_ref = jax.vmap(
        lambda r: jax.grad(lambda p: log_psi(r, nuc, p))(params),
    )(walkers)
    assert float(jnp.max(jnp.abs(lp - lp_ref))) < 1e-10
    worst = 0.0
    for a, b in zip(jax.tree.leaves(grads), jax.tree.leaves(g_ref)):
        scale = float(jnp.max(jnp.abs(b)))
        if scale == 0.0:
            assert float(jnp.max(jnp.abs(a))) == 0.0
            continue
        worst = max(worst, float(jnp.max(jnp.abs(a - b))) / scale)
    print(f"  worst relative gradient error: {worst:.2e}")
    assert worst < RTOL

    # Per-electron cotangents: undamped recovery from M = a^T g.
    g_by_path = {
        _path_str(pth): leaf
        for pth, leaf in jax.tree_util.tree_flatten_with_path(g_ref)[0]
    }
    for name in names:
        a, g = caps[name]
        M = g_by_path[name + '/kernel']
        aat = jnp.einsum('wei,wfi->wef', a, a)
        g_rec = jnp.linalg.solve(aat, jnp.einsum('wei,wio->weo', a, M))
        err = (jnp.max(jnp.abs(g - g_rec), axis=(1, 2))
               / jnp.max(jnp.abs(g), axis=(1, 2)))
        tol = 1e3 * jnp.finfo(jnp.float64).eps * jnp.linalg.cond(aat)
        assert bool(jnp.all(err < tol)), (name, float(err.max()))


def test_kfac_manual_gradient_matches_autodiff():
    mol = _lih()
    key = jax.random.key(SEED)
    drv = {
        m: get_vmcopt_nn_func(mol, 'psiformer', key, nn_dtype=None, **kw)
        for m, kw in (('manual', dict(backward='manual')),
                      ('autodiff', {}))
    }
    params = cast_floats(drv['manual'].init_params, jnp.float64)
    rng = np.random.default_rng(932)
    walkers = jnp.asarray(rng.normal(size=(2, 6, 4, 3)) * 1.5)
    de = jnp.asarray(rng.normal(size=(2, 6)))
    out = {m: d._accumulate_factors(params, walkers, de)
           for m, d in drv.items()}
    for layer, dW in out['manual'][2].items():
        ref = out['autodiff'][2][layer]
        scale = float(jnp.max(jnp.abs(ref)))
        err = float(jnp.max(jnp.abs(dW - ref)))
        assert err <= RTOL * max(scale, 1e-300), (layer, err, scale)
    gen, gen_ref = out['manual'][4], out['autodiff'][4]
    assert float(jnp.max(jnp.abs(gen - gen_ref))) <= RTOL * float(
        jnp.max(jnp.abs(gen_ref)))


def test_kfac_backward_auto_resolution():
    """``backward='auto'`` (the default) picks the hand-written pass
    for the PsiFormer and falls back to autodiff otherwise."""
    mol = _lih()
    key = jax.random.key(SEED)
    assert get_vmcopt_nn_func(
        mol, 'psiformer', key).backward == 'manual'
    assert get_vmcopt_nn_func(
        mol, 'ferminet', key).backward == 'autodiff'
    assert get_vmcopt_nn_func(
        mol, 'psiformer', key,
        capture_activations=True).backward == 'autodiff'
    try:
        get_vmcopt_nn_func(mol, 'ferminet', key, backward='manual')
    except NotImplementedError:
        pass
    else:
        raise AssertionError("backward='manual' accepted a FermiNet")


def test_nuclear_grad_and_zvzb():
    """``nuclear_grad`` gives ``d log|psi| / dR`` as ``jax.grad``
    does, and the ZVZB force components built with it are unchanged.
    """
    from OmegaQMC.observables.force import vmc_nn_gradients_zvzb
    from OmegaQMC.vmc_nn import get_vmc_nn_func

    mol = _lih()
    log_psi, params, bwd, _names, nuc, walkers = _pieces(mol)
    lp, dR = jax.vmap(
        lambda r: bwd.nuclear_grad(r, nuc, params))(walkers)
    dR_ref = jax.vmap(
        lambda r: jax.grad(log_psi, argnums=1)(r, nuc, params),
    )(walkers)
    scale = float(jnp.max(jnp.abs(dR_ref)))
    assert float(jnp.max(jnp.abs(dR - dR_ref))) < RTOL * scale

    _, _, _, lap_grad = make_nn_log_psi(
        'psiformer', mol, jax.random.key(SEED), compute_dtype=None,
    )
    charges = jnp.asarray([3.0, 1.0])
    out = [
        vmc_nn_gradients_zvzb(
            log_psi, nuc, charges, 4, params, lap_grad=lap_grad,
            nuc_grad=ng,
        )(walkers[:3])
        for ng in (bwd.nuclear_grad, None)
    ]
    for a, b in zip(*out):
        assert float(jnp.max(jnp.abs(a - b))) <= RTOL * float(
            jnp.max(jnp.abs(b)))

    # The VMC driver picks the hand-written pass for a PsiFormer.
    key = jax.random.key(SEED)
    assert get_vmc_nn_func(mol, 'psiformer', key,
                           prefix='/tmp/bwd_test').backward == 'manual'
    assert get_vmc_nn_func(mol, 'ferminet', key,
                           prefix='/tmp/bwd_test').backward == 'autodiff'


if __name__ == '__main__':
    test_backward_matches_autodiff()
    test_kfac_manual_gradient_matches_autodiff()
    test_kfac_backward_auto_resolution()
    test_nuclear_grad_and_zvzb()
    print('OK')
