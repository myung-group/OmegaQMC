"""Walker sharding across GPUs must compile and agree with one device.

Regression for a JAX/XLA GPU export bug ("Slice index count does not
match argument rank: 2 vs 1", "'mhlo.while' op can't be translated to
XLA HLO"): a 1-D inner product (``jnp.dot``, ``@``, ``tensordot``,
``einsum('a,a->')``) with a jit-captured constant operand fails
MHLO->HLO translation on GPU once the batch is sharded across devices.
``OmegaQMC.psi.shell.evaluate_cusp_s`` used ``jnp.dot(coeff, terms)``,
which broke every multi-GPU VMC run with cusp-corrected GTOs while
single-GPU (and multi-CPU-device) runs were fine.

The multi-GPU tests are skipped unless at least two GPUs are visible::

    CUDA_VISIBLE_DEVICES=0,2 pytest tests/test_multi_gpu_sharding.py

Running multi-GPU jobs also needs NCCL (``nvidia-nccl-cu12``).
"""
import numpy as np
import pytest

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec

jax.config.update("jax_enable_x64", True)

from OmegaQMC import generate_molecular_orbitals, get_vmc_gto_func  # noqa: E402
from OmegaQMC.gpu_memory import gpu_devices  # noqa: E402
from OmegaQMC.psi.shell import evaluate_cusp_s  # noqa: E402
from OmegaQMC.utils import _make_sharding  # noqa: E402
from OmegaQMC.vmc_gto import _initialize_walkers  # noqa: E402

requires_multi_gpu = pytest.mark.skipif(
    len(gpu_devices()) < 2,
    reason="needs >= 2 visible GPUs (e.g. CUDA_VISIBLE_DEVICES=0,2)")

WATER = ("O   0.000  0.000  0.000  1\n"
         "H   0.757  0.586  0.000  1\n"
         "H  -0.757  0.586  0.000  1\n")
PARAMS_JASTROW = {
    "J1_bspline": {
        "H": jnp.array([0.15016627, 0.11768066, 0.08354908, 0.04696105,
                        0.03430486, 0.02465762, 0.01576562, 0.01332253,
                        0.01274155, 0.00509593]),
        "O": jnp.array([2.06865133, 1.37137248, 0.826893554, 0.476564834,
                        0.274566239, 0.166319001, 0.0879788546,
                        0.0375977801, 0.0132505978, 0.00108272455]),
    },
    "J2_bspline": {
        "like": jnp.array([-0.30676863, -0.18234352, -0.09703378,
                           -0.04603801, -0.02529026, -0.0145683,
                           -0.00835278, -0.00397529, -0.00154836,
                           -0.00065004]),
        "unlike": jnp.array([-0.41269148, -0.2569143, -0.12660155,
                             -0.07305743, -0.03998659, -0.02509969,
                             -0.0149743, -0.00787847, -0.00361378,
                             -0.00117828]),
    },
}


def _cusp_reference(r, rc, Z, rad_s, q0, coeff):
    """Cusp-corrected s orbital written with an explicit ``np.dot``."""
    s = r / rc
    b = np.where(r < rc, 1.0 - 10.0 * s**3 + 15.0 * s**4 - 6.0 * s**5, 0.0)
    slater = q0 * np.exp(-Z * r)
    terms = np.concatenate([[(1.0 - b) * rad_s, b * slater],
                            b * slater * s ** np.arange(2, 8)])
    return np.dot(coeff, terms)


def test_evaluate_cusp_s_matches_dot_product_form():
    rng = np.random.default_rng(0)
    coeff = rng.normal(size=8)
    for r in (0.0, 0.03, 0.11, 0.2, 0.5):      # inside and outside rc
        args = (r, 0.25, 8.0, 0.7, 1.3, coeff)
        np.testing.assert_allclose(float(evaluate_cusp_s(*args)),
                                   _cusp_reference(*args),
                                   rtol=1e-13, atol=1e-15)


@requires_multi_gpu
def test_cusp_kernel_with_captured_constants_shards():
    """Minimal repro: failed to compile before the fix."""
    rng = np.random.default_rng(1)
    n_nuc, n_elec = 15, 20
    n_walkers = 4 * len(jax.devices())
    nuc = jnp.asarray(rng.random((n_nuc, 3)))
    rc = jnp.asarray(0.05 + 0.1 * rng.random(n_nuc))   # captured constants
    Z = jnp.asarray(np.array([8.0, 1.0, 1.0] * 5))
    q0 = jnp.asarray(rng.random(n_nuc))
    coeff = jnp.asarray(rng.random((n_nuc, 8)))
    alpha = jnp.asarray([10.0, 1.0, 0.1])
    norm = jnp.asarray([1.0, 0.5, 0.2])

    def per_electron(x, iat):
        d2 = jnp.sum((x - nuc[iat]) ** 2)
        rad = jnp.sum(jnp.exp(-alpha * d2) * norm)
        return evaluate_cusp_s(jnp.sqrt(d2), rc[iat], Z[iat], rad,
                               q0[iat], coeff[iat])

    kernel = jax.jit(lambda w: jax.vmap(jax.vmap(
        lambda x: per_electron(x, 0)))(w))
    walkers = jnp.asarray(rng.random((n_walkers, n_elec, 3)))
    mesh = Mesh(np.array(jax.devices()), ("w",))
    sharded = jax.device_put(
        walkers, NamedSharding(mesh, PartitionSpec("w", None, None)))
    np.testing.assert_allclose(kernel(sharded), kernel(walkers),
                               rtol=1e-12, atol=1e-14)


@requires_multi_gpu
@pytest.mark.slow
def test_driver_kernels_sharded_match_single_device(tmp_path):
    mf = generate_molecular_orbitals(WATER, units="Ang", basis="6-31G")
    drv = get_vmc_gto_func(mf, PARAMS_JASTROW, cusp_scheme="Quady2025",
                           gr_scheme="scheme1",
                           prefix=str(tmp_path / "h2o"), mo_relax=False)
    n_walkers = 4 * len(jax.devices())
    key = jax.random.key(7)
    walkers = _initialize_walkers(key, n_walkers, drv.nelec, drv.Z_charges,
                                  drv.nuc_crds, drv.mol_charge)
    ws, wks = _make_sharding(n_walkers)
    sharded = jax.device_put(walkers, ws)

    for kernel in (drv._log_psi_batch, drv._local_energy_batch):
        np.testing.assert_allclose(kernel(sharded), kernel(walkers),
                                   rtol=1e-10, atol=1e-10)

    # Force kernel on the replicated batch used by save_gto_gradients.
    replicated = jax.device_put(walkers,
                                NamedSharding(ws.mesh, PartitionSpec()))
    for got, ref in zip(drv.vmc_gradient_batch(replicated),
                        drv.vmc_gradient_batch(walkers)):
        np.testing.assert_allclose(got, ref, rtol=1e-9, atol=1e-9)

    # The scans that first failed to compile: basin thermalization and
    # production (Metropolis moves + local energies).
    _, w_basin, accept = drv._thermalize_basins(key, sharded, n_walkers,
                                                wks, 3)
    assert w_basin.shape == walkers.shape and np.isfinite(float(accept))
    production_step = drv._make_production_step(n_walkers, wks, 1)
    _, (_, e_ee, e_en, e_ke, _) = jax.lax.scan(
        production_step, (key, w_basin, 0.1), jnp.arange(2))
    assert np.all(np.isfinite(np.asarray(e_ee + e_en + e_ke)))
