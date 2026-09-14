"""GPU memory planning for batched JAX kernels.

Estimates the peak device memory of a run from XLA's ahead-of-time
buffer assignment instead of a per-walker rule of thumb.  For every
compiled kernel a driver is about to execute, at the exact shapes and
shardings it will be called with,
``jax.jit(f).lower(*args).compile().memory_analysis()`` reports
``temp_size_in_bytes`` -- the single scratch buffer the XLA GPU runtime
requests for that program.  (For the 5-H2O aug-cc-pVTZ force kernel at
batch 50 this matched the failed out-of-memory allocation to the byte.)

A :class:`MemoryPlan` combines those scratch buffers, in execution
order, with the long-lived arrays a driver holds on the device, and
compares the result with a per-device budget derived from the JAX
allocator settings.  The XLA BFC pool is capped at
``XLA_PYTHON_CLIENT_MEM_FRACTION`` (default 0.75) of the device's total
memory as CUDA sees it (``nvidia-smi`` total minus reserved) -- with or
without preallocation, and regardless of other processes on the GPU
(XLA logs "XLA backend will use up to N bytes"):

* ``XLA_PYTHON_CLIENT_PREALLOCATE`` unset/true (default): the capped
  pool is reserved at start-up and freed blocks coalesce, so the peak
  is the persistent arrays plus the largest scratch buffer.
* ``XLA_PYTHON_CLIENT_PREALLOCATE=false``: the pool grows up to the
  same cap -- or until the GPU is physically full, which on a shared
  GPU can come first -- in separate regions that never merge while any small
  array still lives in them.  A scratch buffer larger than every
  existing region forces a new one, so the peak is the persistent
  arrays plus the *chain* of successively larger scratch buffers.
* ``XLA_PYTHON_CLIENT_ALLOCATOR=platform`` (or
  ``TF_GPU_ALLOCATOR=cuda_malloc_async``): no pool and no cap; the
  peak is as in the preallocated case, against the free memory less a
  driver reserve.

The memory left outside the pool serves the CUDA context, library
handles and CUDA-graph command buffers, which are not modelled; a
``MEM_FRACTION`` close to 1 can starve them.  Other processes are
sampled when the plan is made, not when this process started, so a
job that grows on a shared GPU afterwards is not accounted for.
"""
import os
import subprocess
from dataclasses import dataclass, field

import jax

MiB = 2 ** 20
GiB = 2 ** 30

DEFAULT_MEM_FRACTION = 0.75        # XLA pool cap (either PREALLOCATE mode)
CUDA_CONTEXT_BYTES = 0.5 * GiB     # this process's context, outside the pool
DRIVER_RESERVE_BYTES = 2 * GiB     # handles, CUDA graphs (platform mode)
SAFETY_FRACTION = 0.95             # headroom for rounding/fragmentation


# ---------------------------------------------------------------------
# Device discovery
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class GpuInfo:
    """One physical GPU as reported by ``nvidia-smi``."""
    index: int
    uuid: str
    total_bytes: int
    used_bytes: int
    reserved_bytes: int = 0    # driver/firmware reservation, not usable


def _nvidia_smi_rows(args):
    """Run ``nvidia-smi`` with CSV output; ``None`` if unavailable."""
    try:
        result = subprocess.run(
            ['nvidia-smi', *args, '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return [[c.strip() for c in line.split(',')]
            for line in result.stdout.strip().splitlines() if line.strip()]


def query_gpus():
    """List physical GPUs (all of them, ignoring CUDA_VISIBLE_DEVICES)."""
    fields = 'index,uuid,memory.total,memory.used'
    rows = _nvidia_smi_rows([f'--query-gpu={fields},memory.reserved'])
    if rows is None:      # driver without the memory.reserved field
        rows = _nvidia_smi_rows([f'--query-gpu={fields}'])
    if rows is None:
        return None

    def mib(value):
        try:
            return int(float(value) * MiB)
        except ValueError:   # "[N/A]"
            return 0
    return [GpuInfo(int(r[0]), r[1], mib(r[2]), mib(r[3]),
                    mib(r[4]) if len(r) > 4 else 0) for r in rows]


def physical_gpu(device, gpus=None):
    """Map a JAX CUDA device to its physical ``nvidia-smi`` GPU.

    JAX numbers visible devices from 0, whereas ``nvidia-smi`` lists
    every GPU on the node, so ``CudaDevice(id=0)`` under
    ``CUDA_VISIBLE_DEVICES=2`` is physical GPU 2.  Entries of
    ``CUDA_VISIBLE_DEVICES`` may be indices or (prefixes of) UUIDs.
    Assumes ``CUDA_DEVICE_ORDER`` enumerates in PCI-bus order, which is
    what ``nvidia-smi`` uses (and CUDA's default on identical GPUs).
    """
    gpus = query_gpus() if gpus is None else gpus
    if not gpus:
        return None
    visible = os.environ.get('CUDA_VISIBLE_DEVICES')
    if visible is None:
        key = str(device.id)
    else:
        entries = [e.strip() for e in visible.split(',') if e.strip()]
        if device.id >= len(entries):
            return None
        key = entries[device.id]
    for g in gpus:
        if key == str(g.index) or (key.startswith('GPU-')
                                   and g.uuid.startswith(key)):
            return g
    return None


def other_process_bytes(gpu):
    """Device memory held by processes other than this one."""
    rows = _nvidia_smi_rows(
        ['--query-compute-apps=gpu_uuid,pid,used_memory'])
    if rows is None:
        return 0
    own = os.getpid()
    total = 0
    for uuid, pid, used in rows:
        if uuid == gpu.uuid and int(pid) != own:
            try:
                total += int(float(used) * MiB)
            except ValueError:   # "[N/A]"
                pass
    return total


def gpu_devices():
    """JAX devices on the GPU platform (empty on CPU-only runs)."""
    return [d for d in jax.devices() if d.platform == 'gpu']


# ---------------------------------------------------------------------
# Allocator configuration and budget
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class AllocatorConfig:
    """How JAX obtains device memory in this process."""
    mode: str                  # 'preallocate' | 'growth' | 'platform'
    mem_fraction: float | None

    @property
    def description(self):
        frac = (f", MEM_FRACTION={self.mem_fraction:g}"
                if self.mem_fraction is not None else "")
        return {
            'preallocate': 'preallocated pool',
            'growth': 'growing pool (PREALLOCATE=false)',
            'platform': 'platform allocator',
        }[self.mode] + frac


def allocator_config():
    """Read the JAX/XLA GPU allocator settings from the environment."""
    frac = os.environ.get('XLA_PYTHON_CLIENT_MEM_FRACTION')
    frac = float(frac) if frac else None
    allocator = os.environ.get('XLA_PYTHON_CLIENT_ALLOCATOR', '').lower()
    tf_alloc = os.environ.get('TF_GPU_ALLOCATOR', '').lower()
    prealloc = os.environ.get('XLA_PYTHON_CLIENT_PREALLOCATE', 'true')
    if allocator == 'platform' or tf_alloc == 'cuda_malloc_async':
        mode = 'platform'
    elif prealloc.strip().lower() in ('false', '0', 'no'):
        mode = 'growth'
    else:
        mode = 'preallocate'
    return AllocatorConfig(mode, frac)


def pool_cap_bytes(gpu, config):
    """XLA's BFC pool cap on *gpu*; ``None`` for the platform allocator.

    ``MEM_FRACTION × (total − reserved)``: XLA sizes the pool from the
    memory CUDA reports as the device total, not from what is free, so
    other processes on the GPU do not lower it.
    """
    if config.mode == 'platform':
        return None
    frac = (config.mem_fraction if config.mem_fraction is not None
            else DEFAULT_MEM_FRACTION)
    return int(frac * (gpu.total_bytes - gpu.reserved_bytes))


def device_budget_bytes(gpu, config):
    """Bytes this process's allocator can actually use on *gpu*.

    * preallocated pool: the pool cap (reserved at start-up);
    * growing pool: the cap or the memory other processes leave free,
      whichever is smaller;
    * platform allocator: the free memory less a driver reserve.
    """
    available = (gpu.total_bytes - gpu.reserved_bytes
                 - other_process_bytes(gpu) - CUDA_CONTEXT_BYTES)
    if config.mode == 'platform':
        budget = available - DRIVER_RESERVE_BYTES
    elif config.mode == 'preallocate':
        budget = pool_cap_bytes(gpu, config)
    else:
        budget = min(pool_cap_bytes(gpu, config), available)
    return int(max(budget, 0))


# ---------------------------------------------------------------------
# Compile-time memory statistics
# ---------------------------------------------------------------------

def compile_with_memory_stats(fn, *args):
    """Ahead-of-time compile *fn* for *args* and read its buffer sizes.

    Returns ``(compiled, stats)``.  ``compiled`` is a callable
    ``jax.stages.Compiled`` that runs the same XLA executable for
    arguments of the same shapes, dtypes and pytree structure, so the
    compilation can be reused instead of being repeated by ``jax.jit``.
    Returns ``(None, None)`` where AOT analysis is unavailable.
    """
    try:
        compiled = jax.jit(fn).lower(*args).compile()
        return compiled, compiled.memory_analysis()
    except Exception:  # noqa: BLE001 -- backend-dependent API
        return None, None


def reuse_compiled(fn, compiled, example):
    """Dispatch calls matching *example*'s shape to *compiled*.

    *fn* and *compiled* take a single array argument; any other shape
    (e.g. a short final batch) falls through to *fn*.
    """
    if compiled is None:
        return fn
    shape, dtype = example.shape, example.dtype

    def call(x):
        if x.shape == shape and x.dtype == dtype:
            return compiled(x)
        return fn(x)
    return call


# ---------------------------------------------------------------------
# Plan
# ---------------------------------------------------------------------

@dataclass
class _Entry:
    name: str
    nbytes: int
    group: str | None     # scaling group, e.g. 'walkers', 'batch'


@dataclass
class MemoryPlan:
    """Per-device peak-memory model of a run.

    Register scratch buffers with :meth:`add_kernel` in the order the
    kernels first execute, and long-lived device arrays with
    :meth:`add_persistent`.  Entries tagged with a scaling ``group``
    are assumed to grow linearly with that group's size (walkers per
    device, gradient batch), which :meth:`max_scale` uses to solve for
    capacity.
    """
    config: AllocatorConfig
    budget_bytes: int
    kernels: list = field(default_factory=list)
    persistent: list = field(default_factory=list)

    def add_kernel(self, name, stats, group=None):
        nbytes = 0 if stats is None else int(stats.temp_size_in_bytes)
        self.kernels.append(_Entry(name, nbytes, group))

    def add_persistent(self, name, nbytes, group=None):
        self.persistent.append(_Entry(name, int(nbytes), group))

    def _scaled(self, entries, scales):
        return [e.nbytes * scales.get(e.group, 1.0) for e in entries]

    def peak_bytes(self, scales=None):
        """Estimated peak with optional per-group linear scale factors."""
        scales = scales or {}
        persistent = sum(self._scaled(self.persistent, scales))
        temps = self._scaled(self.kernels, scales)
        if not temps:
            return persistent
        if self.config.mode == 'growth':
            regions = []
            for t in temps:
                if not regions or t > max(regions):
                    regions.append(t)
            return persistent + sum(regions)
        return persistent + max(temps)

    @property
    def usable_bytes(self):
        return SAFETY_FRACTION * self.budget_bytes

    def fits(self, scales=None):
        return self.peak_bytes(scales) <= self.usable_bytes

    def max_scale(self, group, hi=1024.0, iters=60):
        """Largest scale of *group* that still fits (0 if none does)."""
        if not self.fits({group: 0.0}):
            return 0.0
        lo = 0.0
        if self.fits({group: hi}):
            return hi
        for _ in range(iters):
            mid = 0.5 * (lo + hi)
            if self.fits({group: mid}):
                lo = mid
            else:
                hi = mid
        return lo

    def report(self, header):
        """Human-readable table of the plan."""
        lines = [f"ℹ️\t{header}"]
        for e in self.kernels:
            lines.append(f"\t  scratch  {e.name:<34}{e.nbytes / GiB:>9.2f} GiB")
        for e in self.persistent:
            lines.append(f"\t  arrays   {e.name:<34}{e.nbytes / GiB:>9.2f} GiB")
        peak = self.peak_bytes()
        lines.append(
            f"\t  estimated peak {peak / GiB:.2f} GiB of "
            f"{self.budget_bytes / GiB:.2f} GiB budget "
            f"({100 * peak / max(self.budget_bytes, 1):.0f}%; "
            f"{self.config.description})")
        return "\n".join(lines)
