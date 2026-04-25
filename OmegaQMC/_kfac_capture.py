"""JAX-traceable per-Linear input capture for KFAC.

The trick: during model construction, replace every ``nnx.Linear``
instance with a ``CapturingLinear`` (a subclass that adds an
``nnx.Intermediate`` slot for its input).  The model's ``__call__``
is unchanged because ``CapturingLinear.__call__`` writes to the
intermediate slot before delegating to ``nnx.Linear.__call__``.

After building, ``nnx.split(model, nnx.Param, nnx.Intermediate)``
separates the trainable parameters from the captured-input slots.
A ``vmap``ed forward pass then yields a batched intermediate state
where each ``CapturingLinear``'s slot holds a ``(W, *input_shape)``
array — exactly the per-(walker, electron) inputs that FermiNet's
``RepeatedDenseBlock`` consumes.

Why this is faster than the eager monkey-patch path:
  * Single ``jax.vmap`` over walkers compiled once via JIT.
  * No Python-side dispatch per walker.
  * Captures flow through standard NNX state machinery, so they
    survive JIT and vmap by design.

Usage:

.. code-block:: python

    with use_capturing_linears():
        model = build_heg_psiformer_wf(config, rngs)
    # `model` is structurally identical to the non-capturing build,
    # but every Linear is now a CapturingLinear.

    graphdef, params, inters = nnx.split(model, nnx.Param, nnx.Intermediate)

    def apply(params, inters, walker):
        m = nnx.merge(graphdef, params, inters)
        psi = m(walker)
        _, _, new_inters = nnx.split(m, nnx.Param, nnx.Intermediate)
        return psi.log, new_inters

    # Batched forward + captures, JIT-compiled:
    log_psi_W, inters_W = jax.jit(jax.vmap(
        apply, in_axes=(None, None, 0),
    ))(params, inters, walkers)
    # ``inters_W`` is a state pytree where each CapturingLinear's slot
    # holds an array of shape (W, *input_shape).
"""
from __future__ import annotations

import contextlib
from typing import Iterator

import jax
import jax.numpy as jnp
from flax import nnx


class CapturingLinear(nnx.Linear):
    """``nnx.Linear`` that records its input on each call.

    The captured input is stored in ``self.captured_input`` (an
    ``nnx.Intermediate``).  After a forward pass, read out via
    ``nnx.split(model, nnx.Param, nnx.Intermediate)``.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Sentinel zero — replaced on first call.  Concrete shape
        # is unknown at init time (depends on call-site x shape).
        self.captured_input = nnx.Intermediate(
            jnp.zeros((), dtype=jnp.float64),
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        # Write x to the intermediate slot.  This is JAX-traceable:
        # under ``jax.jit``/``jax.vmap`` the slot's value becomes a
        # batched tracer that materialises after compilation.
        self.captured_input = nnx.Intermediate(x)
        return super().__call__(x)


@contextlib.contextmanager
def use_capturing_linears() -> Iterator[None]:
    """Inside this context, every ``nnx.Linear(...)`` constructor
    returns a ``CapturingLinear`` instead.

    Built models retain ``CapturingLinear`` instances after the
    context exits — only the constructor lookup is patched.

    Reentrant-safe: nested contexts don't double-patch.  Restored
    on exit even if the caller raises.
    """
    sentinel_attr = '_omegaqmc_kfac_orig_linear'
    if not hasattr(nnx, sentinel_attr):
        setattr(nnx, sentinel_attr, nnx.Linear)
        nnx.Linear = CapturingLinear
        try:
            yield
        finally:
            nnx.Linear = getattr(nnx, sentinel_attr)
            delattr(nnx, sentinel_attr)
    else:
        # Already in a capturing context; just pass through.
        yield


def find_capturing_linear_paths(model: nnx.Module) -> dict:
    """Walk the live model and return ``{layer_path: nnx.Linear}``
    for every ``CapturingLinear``.  Path strings match the format
    used in ``vmcopt_nn_heg_kfac._classify_params`` so the
    captured-intermediate paths align with the kernel-param paths.
    """
    out: dict = {}

    def _r(obj, prefix: str):
        if isinstance(obj, CapturingLinear):
            out[prefix.rstrip('/')] = obj
            return
        if isinstance(obj, nnx.Module):
            for k, v in obj.__dict__.items():
                if k.startswith('_'):
                    continue
                _r(v, prefix + str(k) + '/')
        elif isinstance(obj, (list, tuple)):
            for i, v in enumerate(obj):
                _r(v, prefix + str(i) + '/')
        elif isinstance(obj, dict):
            for k, v in obj.items():
                _r(v, prefix + str(k) + '/')

    _r(model, '')
    return out


def captures_to_path_dict(intermediates_state) -> dict:
    """Convert the ``nnx.Intermediate`` state pytree into a flat
    ``{layer_path → captured_array}`` dict.

    The state has structure ``{module_path → {captured_input: …}}``
    where ``module_path`` matches the model's module tree.  We
    string-join keys with ``/`` to match the
    ``_classify_params``/``_discover_linear_ids`` path convention.
    """
    leaves = jax.tree_util.tree_flatten_with_path(intermediates_state)[0]
    out: dict = {}
    for path, leaf in leaves:
        # Strip the trailing ('captured_input', '.value') tail.
        keys = []
        for p in path:
            k = p.key if hasattr(p, 'key') else p
            if isinstance(k, int):
                keys.append(str(k))
            else:
                keys.append(k)
        # The last useful key is two before the end:
        #   ... / <layer_name> / 'captured_input' / '.value'
        path_str = '/'.join(keys[:-2])
        out[path_str] = leaf
    return out
