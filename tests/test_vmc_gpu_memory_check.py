"""Pre-flight GPU memory check of the GTO VMC driver.

``_VMCDriverGTO._plan_gpu_memory`` compiles the run's kernels at their
real shapes, estimates the per-device peak from XLA's scratch-buffer
sizes plus the long-lived arrays, lowers the force-gradient batch when
it does not fit, stops (or warns about) runs whose walkers do not fit,
and hands the compiled force kernels back so the slow compilation is
not repeated.

The tests drive it on H2O / 6-31G with a *patched* allocator budget
and a fixed (preallocated-pool) peak model, so every branch is
exercised deterministically on any GPU.  The argument handling is also
checked on CPU, where the planning itself is a no-op.
"""
import numpy as np
import pytest

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

import OmegaQMC.vmc_gto as vmc_gto  # noqa: E402
from OmegaQMC import generate_molecular_orbitals, get_vmc_gto_func  # noqa: E402
from OmegaQMC.gpu_memory import (  # noqa: E402
    GiB, SAFETY_FRACTION, AllocatorConfig, MemoryPlan, gpu_devices,
)
from OmegaQMC.utils import _make_sharding  # noqa: E402

requires_gpu = pytest.mark.skipif(not gpu_devices(), reason="needs a GPU")

BATCH = 32                         # gradient batch handed to the planner
WALKERS_PER_DEVICE = 2             # keeps the force kernel dominant

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


@pytest.fixture(scope="module")
def driver(tmp_path_factory):
    mf = generate_molecular_orbitals(WATER, units="Ang", basis="6-31G")
    prefix = str(tmp_path_factory.mktemp("memcheck") / "h2o")
    return get_vmc_gto_func(mf, PARAMS_JASTROW, cusp_scheme="Quady2025",
                            gr_scheme="scheme1", prefix=prefix,
                            mo_relax=False)


def _run_plan(drv, monkeypatch, *, check, budget, compute_gradients=True,
              record=None):
    """Call ``_plan_gpu_memory`` as ``__call__`` does, with a fixed budget.

    ``record`` collects ``(fn, stats)`` for every kernel compiled and
    the :class:`MemoryPlan` objects built.
    """
    monkeypatch.setattr(vmc_gto, "allocator_config",
                        lambda: AllocatorConfig("preallocate", None))
    monkeypatch.setattr(vmc_gto, "device_budget_bytes",
                        lambda gpu, config: int(budget))
    if record is not None:
        compile_stats = vmc_gto.compile_with_memory_stats

        def recording_compile(fn, *args):
            compiled, stats = compile_stats(fn, *args)
            record["kernels"].append((fn, stats))
            return compiled, stats

        class RecordingPlan(MemoryPlan):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                record["plans"].append(self)

        monkeypatch.setattr(vmc_gto, "compile_with_memory_stats",
                            recording_compile)
        monkeypatch.setattr(vmc_gto, "MemoryPlan", RecordingPlan)

    n_walkers = WALKERS_PER_DEVICE * len(jax.devices())
    key = jax.random.key(3)
    walkers = vmc_gto._initialize_walkers(key, n_walkers, drv.nelec,
                                          drv.Z_charges, drv.nuc_crds,
                                          drv.mol_charge)
    ws, wks = _make_sharding(n_walkers)
    if ws is not None:
        walkers = jax.device_put(walkers, ws)
    return drv._plan_gpu_memory(
        check=check, rng_key=key, walkers=walkers, walkers_sharding=ws,
        num_walkers=n_walkers, num_steps_per_block=4, step_size=0.1,
        basin_step=drv._make_basin_step(n_walkers, wks),
        equilibration_step=None,
        production_step=drv._make_production_step(n_walkers, wks, 1),
        compute_gradients=compute_gradients, batch_size=BATCH)


@pytest.fixture(scope="module")
def roomy_plan(driver):
    """Plan with an ample budget; returns kernels and measured sizes."""
    record = {"kernels": [], "plans": []}
    with pytest.MonkeyPatch.context() as mp:
        batch, kernels = _run_plan(driver, mp, check="error",
                                   budget=1000 * GiB, record=record)
    grad_fns = (driver.vmc_gradient_batch, driver._log_psi_batch,
                driver._local_energy_batch)
    grad_temp = next(s.temp_size_in_bytes for fn, s in record["kernels"]
                     if fn is driver.vmc_gradient_batch)
    walker_temp = max(s.temp_size_in_bytes for fn, s in record["kernels"]
                      if not any(fn is g for g in grad_fns))
    persistent = sum(e.nbytes for e in record["plans"][-1].persistent)
    return dict(batch=batch, kernels=kernels, grad_temp=grad_temp,
                walker_temp=walker_temp, persistent=persistent)


# ---------------------------------------------------------------------
# Argument handling (CPU and GPU)
# ---------------------------------------------------------------------

def test_invalid_check_value_is_rejected(driver):
    with pytest.raises(ValueError, match="gpu_memory_check"):
        driver(jax.random.key(0), num_walkers=2, gpu_memory_check="maybe")


def test_check_off_leaves_batch_and_kernels_alone(driver, monkeypatch):
    batch, kernels = _run_plan(driver, monkeypatch, check="off", budget=1)
    assert batch == BATCH
    assert kernels["vmc_gradient_batch"] is driver.vmc_gradient_batch
    assert kernels["local_energy_batch"] is driver._local_energy_batch


# ---------------------------------------------------------------------
# Planning (GPU)
# ---------------------------------------------------------------------

@requires_gpu
@pytest.mark.slow
def test_fitting_plan_keeps_batch_and_reuses_compiled_kernels(driver,
                                                              roomy_plan):
    assert roomy_plan["batch"] == BATCH
    fast = roomy_plan["kernels"]["vmc_gradient_batch"]
    assert fast is not driver.vmc_gradient_batch
    rng = np.random.default_rng(5)
    # Full batch -> compiled executable; short final batch -> jit path.
    for b in (BATCH, BATCH - 3):
        x = jnp.asarray(0.8 * rng.normal(size=(b, driver.nelec, 3)))
        for got, ref in zip(fast(x), driver.vmc_gradient_batch(x)):
            np.testing.assert_allclose(got, ref, rtol=1e-10, atol=1e-10)


@requires_gpu
@pytest.mark.slow
def test_gradient_batch_is_lowered_to_fit(driver, roomy_plan, monkeypatch):
    grad_temp = roomy_plan["grad_temp"]
    walker_temp = roomy_plan["walker_temp"]
    if grad_temp < 4 * walker_temp:
        pytest.skip("force kernel does not dominate on this device")
    # Room for the walker phase and about half the force kernel.
    budget = (roomy_plan["persistent"]
              + max(walker_temp, grad_temp // 2)) / SAFETY_FRACTION
    batch, _ = _run_plan(driver, monkeypatch, check="warn", budget=budget)
    assert 1 <= batch < BATCH


@requires_gpu
@pytest.mark.slow
def test_run_that_cannot_fit_is_stopped(driver, monkeypatch):
    with pytest.raises(RuntimeError, match="exceeds the usable"):
        _run_plan(driver, monkeypatch, check="error", budget=1,
                  compute_gradients=False)


@requires_gpu
@pytest.mark.slow
def test_run_that_cannot_fit_warns_when_asked(driver, monkeypatch):
    with pytest.warns(UserWarning, match="exceeds the usable"):
        batch, _ = _run_plan(driver, monkeypatch, check="warn", budget=1,
                             compute_gradients=False)
    assert batch == BATCH
