"""Unit tests for :mod:`OmegaQMC.gpu_memory`.

The pre-flight device-memory check of the GTO VMC driver rests on a
few facts that were verified on an A100 node (JAX 0.8) and are pinned
here so that a JAX/XLA upgrade changing them fails loudly:

* ``Compiled.memory_analysis()`` exposes ``temp_size_in_bytes``.  An
  earlier estimator read ``temp_size`` instead, hit ``AttributeError``
  inside a bare ``except`` and silently reported 0.5 MB/walker.
* The XLA BFC pool is capped at ``MEM_FRACTION`` x (total - reserved)
  device memory -- also with ``XLA_PYTHON_CLIENT_PREALLOCATE=false``, and
  regardless of other processes on the GPU (a growing pool can still
  run out first when they leave less than the cap free).
* JAX device ids are logical: ``CudaDevice(id=0)`` under
  ``CUDA_VISIBLE_DEVICES=2`` is physical GPU 2.

The peak-model tests use the scratch sizes measured for 5 H2O /
aug-cc-pVTZ, where 1000 walkers ran out of memory with a growing pool
but fit a preallocated one, and 500 walkers fit both.

All tests except the last run without a GPU.
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

from OmegaQMC import gpu_memory as gm
from OmegaQMC.gpu_memory import (
    GiB, MiB, AllocatorConfig, GpuInfo, MemoryPlan,
)

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
