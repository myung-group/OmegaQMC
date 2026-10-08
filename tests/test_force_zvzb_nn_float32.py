"""Regression: float32 ZVZB kinetic term near a determinant node.

The PsiFormer is a sum of 16 full determinants.  Each has its own
nodal surface, while their sum (here, H2) has none, so ordinary
walkers come close to a node of one determinant.  There
:func:`~OmegaQMC.psi.nn.forward_lap.slogdet_vgl` returns
``grad log|det_d|`` and ``lap log|det_d|`` of order ``1/det_d`` and
``1/det_d^2``, and the multi-determinant aggregation recovers the
finite derivatives of ``psi`` only by cancellation.  With the Slater
stage in float32 (as the network is by default) the ZVZB kinetic
term ``grd_ke``, a nuclear derivative of the Laplacian, came out
wrong by 1e2-1e5 Ha/bohr on such walkers; the Slater stage is now
evaluated in float64.

This test moves a walker of a randomly initialised H2 PsiFormer
onto ``|det_0| / max_d |det_d| = 1e-6`` and compares ``grd_ke``
from the float32 network with the float64 one.
"""
import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

jax.config.update("jax_enable_x64", True)

from OmegaQMC.utils import Mole_custom
from OmegaQMC.psi.nn.adapter import make_nn_log_psi
from OmegaQMC.psi.nn.build import build_nn_wf
from OmegaQMC.psi.nn.config import load_nn_config
from OmegaQMC.psi.nn.types import PhysicalConfiguration
from OmegaQMC.observables.force import vmc_nn_gradients_zvzb

SEED = 920
DET_RATIO = 1e-6
# Tolerance relative to max(1, |grd_ke|): the float32 network itself
# limits the agreement to ~1e-3 relative.  The comparison runs at
# 'highest' matmul precision, since a GPU's default TF32 matmuls add
# ~2e-2 on their own; this test is about the Slater-stage
# cancellation (errors of ~1e5 without the float64 stage).
RTOL = 1e-2


def _h2_mol():
    L = 1.4010
    return Mole_custom.from_arrays(
        charges=[1, 1],
        coords=[[0.0, 0.0, -L / 2], [0.0, 0.0, L / 2]],
        n_up=1, n_down=1,
    )


def _det_fn(mol):
    """Per-determinant values ``det_d(r)`` of the float64 model.

    *r* is ``(nelec, 3)`` in OmegaQMC's interleaved spin order;
    the model is built with the same key as ``make_nn_log_psi``,
    so it carries the same parameters.
    """
    model = build_nn_wf(
        load_nn_config('psiformer'), mol, nnx.Rngs(jax.random.key(SEED)),
    )
    nuc = jnp.asarray(mol.coords)

    def dets(r):
        r_grouped = jnp.concatenate([r[::2], r[1::2]], axis=0)
        orb_up, orb_dn = model(
            PhysicalConfiguration(R=nuc, r=r_grouped,
                                  mol_idx=jnp.array(0)),
            return_mos=True,
        )
        return jnp.linalg.det(
            jnp.concatenate([orb_up, orb_dn], axis=-2),
        )
    return dets


def _walker_near_node(dets, r0, d=0, ratio=DET_RATIO, n_iter=30):
    """Newton-solve ``det_d(r) = ratio * max|det|`` from *r0*."""
    def f(r):
        v = dets(r)
        target = ratio * jnp.max(jnp.abs(v)) * jnp.sign(v[d])
        return v[d] - jax.lax.stop_gradient(target)

    r = r0
    for _ in range(n_iter):
        val, g = jax.value_and_grad(f)(r)
        r = r - val * g / jnp.sum(g * g)
    return r


def test_h2_psiformer_zvzb_float32_near_det_node():
    mol = _h2_mol()
    nuc = jnp.asarray(mol.coords)
    charges = jnp.asarray([1.0, 1.0])
    dets = _det_fn(mol)

    rng = np.random.default_rng(921)
    walkers = []
    for d in range(3):
        r0 = jnp.asarray(rng.normal(size=(2, 3)))
        r = _walker_near_node(dets, r0, d=d)
        v = np.abs(np.asarray(dets(r)))
        ratio = v[d] / v.max()
        print(f"  det {d}: |det|/max = {ratio:.2e}")
        assert 0.1 * DET_RATIO < ratio < 10 * DET_RATIO, ratio
        walkers.append(r)
    batch = jnp.stack(walkers)

    grd_ke = {}
    for dt in ('float32', None):
        log_psi, params, _gd, lap_grad = make_nn_log_psi(
            'psiformer', mol, jax.random.key(SEED), compute_dtype=dt,
        )
        assert lap_grad.use_vgl
        fn = vmc_nn_gradients_zvzb(
            log_psi, nuc, charges, 2, params, lap_grad=lap_grad,
        )
        with jax.default_matmul_precision('highest'):
            grd_ke[dt] = np.asarray(fn(batch)[1])

    ref = grd_ke[None]
    err = np.abs(grd_ke['float32'] - ref) / np.maximum(1.0, np.abs(ref))
    print(f"  grd_ke float64: {ref.reshape(len(walkers), -1)}")
    print(f"  max relative error of float32: {err.max():.2e}")
    assert err.max() < RTOL, err.max()


if __name__ == '__main__':
    test_h2_psiformer_zvzb_float32_near_det_node()
    print('OK')
