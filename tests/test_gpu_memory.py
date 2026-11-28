"""GPU memory model and the pre-flight memory check of the GTO VMC driver.

The memory model (memory_analysis byte fields, allocator settings,
budget, peak and maximum-scale models) and the driver's pre-flight
check, which lowers the gradient batch, warns or stops before a run
would exceed GPU memory.  Some tests need a GPU.
"""
import os
import re
import subprocess
import sys
from types import SimpleNamespace
import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update('jax_enable_x64', True)

from OmegaQMC import (  # noqa: E402
    gpu_memory as gm,
    generate_molecular_orbitals,
    get_vmc_gto_func,
)
from OmegaQMC.gpu_memory import (  # noqa: E402
    GiB,
    MiB,
    AllocatorConfig,
    GpuInfo,
    MemoryPlan,
    SAFETY_FRACTION,
    gpu_devices,
)
from OmegaQMC.utils import _make_sharding  # noqa: E402
import OmegaQMC.vmc_gto as vmc_gto  # noqa: E402


# ---------------------------------------------------------------------------
# From test_gpu_memory.py
#
# Unit tests for :mod:`OmegaQMC.gpu_memory`.
#
# The pre-flight device-memory check of the GTO VMC driver rests on a
# few facts that were verified on an A100 node (JAX 0.8) and are pinned
# here so that a JAX/XLA upgrade changing them fails loudly:
#
# * ``Compiled.memory_analysis()`` exposes ``temp_size_in_bytes``.  An
#   earlier estimator read ``temp_size`` instead, hit ``AttributeError``
#   inside a bare ``except`` and silently reported 0.5 MB/walker.
# * The XLA BFC pool is capped at ``MEM_FRACTION`` x (total - reserved)
#   device memory -- also with ``XLA_PYTHON_CLIENT_PREALLOCATE=false``, and
#   regardless of other processes on the GPU (a growing pool can still
#   run out first when they leave less than the cap free).
# * JAX device ids are logical: ``CudaDevice(id=0)`` under
#   ``CUDA_VISIBLE_DEVICES=2`` is physical GPU 2.
#
# The peak-model tests use the scratch sizes measured for 5 H2O /
# aug-cc-pVTZ, where 1000 walkers ran out of memory with a growing pool
# but fit a preallocated one, and 500 walkers fit both.
#
# All tests except the last run without a GPU.
# ---------------------------------------------------------------------------


_ALLOC_ENV = ("XLA_PYTHON_CLIENT_PREALLOCATE",
              "XLA_PYTHON_CLIENT_MEM_FRACTION",
              "XLA_PYTHON_CLIENT_ALLOCATOR", "TF_GPU_ALLOCATOR")


def _stats(gib):
    return SimpleNamespace(temp_size_in_bytes=int(gib * GiB))


# ---------------------------------------------------------------------
# Compile-time statistics and executable reuse
# ---------------------------------------------------------------------
def test_memory_analysis_exposes_byte_fields():
    x = jnp.linspace(0.0, 1.0, 64)
    compiled, stats = gm.compile_with_memory_stats(
        lambda v: jnp.sin(v) @ jnp.cos(v), x)
    assert compiled is not None and stats is not None
    for name in ("temp_size_in_bytes", "alias_size_in_bytes",
                 "argument_size_in_bytes", "output_size_in_bytes"):
        assert isinstance(getattr(stats, name), int), name
    np.testing.assert_allclose(compiled(x), jnp.sin(x) @ jnp.cos(x))


def test_reuse_compiled_dispatches_on_shape():
    fn = jax.jit(lambda w: jnp.sum(w ** 2, axis=(1, 2)))
    example = jnp.ones((4, 5, 3))
    compiled, _ = gm.compile_with_memory_stats(fn, example)
    fallback_shapes = []

    def fallback(w):
        fallback_shapes.append(w.shape)
        return fn(w)

    call = gm.reuse_compiled(fallback, compiled, example)
    full = jnp.arange(60.0).reshape(4, 5, 3)
    short = full[:3]
    np.testing.assert_allclose(call(full), fn(full))
    assert fallback_shapes == []                 # compiled executable
    np.testing.assert_allclose(call(short), fn(short))
    assert fallback_shapes == [(3, 5, 3)]        # short final batch
    assert gm.reuse_compiled(fallback, None, example) is fallback


# ---------------------------------------------------------------------
# Allocator configuration, device mapping and budget
# ---------------------------------------------------------------------
@pytest.mark.parametrize("env, mode, frac", [
    ({}, "preallocate", None),
    ({"XLA_PYTHON_CLIENT_MEM_FRACTION": "0.9"}, "preallocate", 0.9),
    ({"XLA_PYTHON_CLIENT_PREALLOCATE": "false"}, "growth", None),
    ({"XLA_PYTHON_CLIENT_PREALLOCATE": "False",
      "XLA_PYTHON_CLIENT_MEM_FRACTION": ".5"}, "growth", 0.5),
    ({"XLA_PYTHON_CLIENT_ALLOCATOR": "platform"}, "platform", None),
    ({"TF_GPU_ALLOCATOR": "cuda_malloc_async"}, "platform", None),
])
def test_allocator_config_from_environment(monkeypatch, env, mode, frac):
    for name in _ALLOC_ENV:
        monkeypatch.delenv(name, raising=False)
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    config = gm.allocator_config()
    assert config.mode == mode
    assert config.mem_fraction == frac


_RESERVED = 765 * MiB                  # A100 80GB driver reservation
_CUDA_TOTAL = 80 * GiB - _RESERVED
_GPUS = [GpuInfo(0, "GPU-7207eccc-aaaa", 80 * GiB, 0, _RESERVED),
         GpuInfo(1, "GPU-7d5a3dc6-bbbb", 80 * GiB, 0, _RESERVED),
         GpuInfo(2, "GPU-3bd52ba8-cccc", 80 * GiB, 0, _RESERVED)]


@pytest.mark.parametrize("visible, logical_id, physical_index", [
    (None, 1, 1),              # unset: logical == physical
    ("2", 0, 2),               # the id=0 -> GPU 0 mix-up this guards
    ("2,0", 1, 0),
    ("GPU-3bd52ba8", 0, 2),    # UUID prefix
    ("2", 1, None),            # beyond the visible list
])
def test_physical_gpu_mapping(monkeypatch, visible, logical_id,
                              physical_index):
    if visible is None:
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    else:
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", visible)
    gpu = gm.physical_gpu(SimpleNamespace(id=logical_id), _GPUS)
    assert (gpu.index if gpu is not None else None) == physical_index


@pytest.mark.parametrize("config, others, expected", [
    # The pool cap ignores other processes on the GPU ...
    (AllocatorConfig("preallocate", None), 50 * GiB, 0.75 * _CUDA_TOTAL),
    (AllocatorConfig("growth", None), 10 * GiB, 0.75 * _CUDA_TOTAL),
    (AllocatorConfig("growth", 0.5), 10 * GiB, 0.5 * _CUDA_TOTAL),
    # ... but a growing pool cannot outgrow what they leave free.
    (AllocatorConfig("growth", None), 50 * GiB,
     _CUDA_TOTAL - 50 * GiB - gm.CUDA_CONTEXT_BYTES),
    (AllocatorConfig("platform", None), 10 * GiB,
     _CUDA_TOTAL - 10 * GiB - gm.CUDA_CONTEXT_BYTES
     - gm.DRIVER_RESERVE_BYTES),
])
def test_device_budget_formula(monkeypatch, config, others, expected):
    monkeypatch.setattr(gm, "other_process_bytes", lambda gpu: others)
    assert gm.device_budget_bytes(_GPUS[0], config) == int(expected)


# ---------------------------------------------------------------------
# Peak models and capacity
# ---------------------------------------------------------------------
def _h2o5_plan(mode, walker_scale=1.0, budget=59.44 * GiB):
    """Measured per-device scratch sizes, 5 H2O at 1000 walkers."""
    plan = MemoryPlan(AllocatorConfig(mode, None), int(budget))
    for name, gib, group in (("basin step", 2.79, "walkers"),
                             ("equilibration step", 2.79, "walkers"),
                             ("production step", 13.27, "walkers"),
                             ("vmc_gradient_batch", 40.60, "batch"),
                             ("log_psi_batch", 0.14, "batch"),
                             ("local_energy_batch", 0.68, "batch")):
        scale = walker_scale if group == "walkers" else 1.0
        plan.add_kernel(name, _stats(gib * scale), group)
    plan.add_persistent("arrays", 0.08 * GiB * walker_scale, "walkers")
    return plan


def test_preallocated_peak_is_largest_scratch_buffer():
    plan = _h2o5_plan("preallocate")
    assert plan.peak_bytes() == pytest.approx((40.60 + 0.08) * GiB, rel=1e-9)
    assert plan.fits()


def test_growth_peak_chains_larger_scratch_buffers():
    # Regions: basin 2.79, production 13.27, gradient 40.60 (the
    # equal-sized equilibration buffer reuses the basin region).
    plan = _h2o5_plan("growth")
    assert plan.peak_bytes() == pytest.approx(
        (2.79 + 13.27 + 40.60 + 0.08) * GiB, rel=1e-9)
    assert not plan.fits()                     # observed: OOM
    assert _h2o5_plan("growth", walker_scale=0.5).fits()   # 500 walkers


def test_max_scale_solves_capacity():
    plan = _h2o5_plan("growth")
    s = plan.max_scale("walkers")
    assert 0.0 < s < 1.0
    assert plan.fits({"walkers": s})
    assert not plan.fits({"walkers": s * 1.001})
    starved = _h2o5_plan("growth", budget=30 * GiB)   # gradient alone too big
    assert starved.max_scale("walkers") == 0.0
    assert "estimated peak" in plan.report("plan")


# ---------------------------------------------------------------------
# Budget versus XLA's own pool cap (GPU only)
# ---------------------------------------------------------------------
_CAP_PROBE = """
import jax.numpy as jnp
from OmegaQMC.gpu_memory import (gpu_devices, physical_gpu, allocator_config,
                                 pool_cap_bytes, device_budget_bytes)
float(jnp.ones(4).sum())          # creates the GPU client, logs its cap
gpu = physical_gpu(gpu_devices()[0])
config = allocator_config()
print("MODEL_CAP", pool_cap_bytes(gpu, config))
print("MODEL_BUDGET", device_budget_bytes(gpu, config))
"""


@pytest.mark.skipif(not gm.gpu_devices(), reason="needs a GPU")
@pytest.mark.parametrize("alloc_env", [
    {"XLA_PYTHON_CLIENT_PREALLOCATE": "false"},
    {"XLA_PYTHON_CLIENT_MEM_FRACTION": "0.3"},
])
def test_pool_cap_matches_xla_log(alloc_env):
    """Holds on a shared GPU too: XLA's cap ignores other processes."""
    gpu = gm.physical_gpu(gm.gpu_devices()[0])
    assert gpu is not None
    env = {k: v for k, v in os.environ.items() if k not in _ALLOC_ENV}
    env.update(alloc_env, TF_CPP_MIN_LOG_LEVEL="0",
               CUDA_VISIBLE_DEVICES=str(gpu.index))
    out = subprocess.run([sys.executable, "-c", _CAP_PROBE], env=env,
                         capture_output=True, text=True, timeout=600)
    text = out.stdout + out.stderr
    xla = re.search(r"(?:will use up to|allocating) (\d+) bytes on device "
                    r"\d+ for BFCAllocator", text)
    cap = re.search(r"MODEL_CAP (\d+)", text)
    budget = re.search(r"MODEL_BUDGET (\d+)", text)
    assert xla and cap and budget, text[-2000:]
    # nvidia-smi reports whole MiB, so allow a few MiB of rounding.
    assert abs(int(cap.group(1)) - int(xla.group(1))) < 8 * MiB
    assert int(budget.group(1)) <= int(cap.group(1))


# ---------------------------------------------------------------------------
# From test_vmc_gpu_memory_check.py
#
# Pre-flight GPU memory check of the GTO VMC driver.
#
# ``_VMCDriverGTO._plan_gpu_memory`` compiles the run's kernels at their
# real shapes, estimates the per-device peak from XLA's scratch-buffer
# sizes plus the long-lived arrays, lowers the force-gradient batch when
# it does not fit, stops (or warns about) runs whose walkers do not fit,
# and hands the compiled force kernels back so the slow compilation is
# not repeated.
#
# The tests drive it on H2O / 6-31G with a *patched* allocator budget
# and a fixed (preallocated-pool) peak model, so every branch is
# exercised deterministically on any GPU.  The argument handling is also
# checked on CPU, where the planning itself is a no-op.
# ---------------------------------------------------------------------------


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
