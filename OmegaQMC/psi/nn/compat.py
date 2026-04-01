"""JAX and Flax version compatibility utilities.

Provides version-safe wrappers for APIs that changed
between JAX 0.6--0.9 and Flax 0.10--0.12+.

JAX compatibility
-----------------
All JAX APIs used in this package (``jax.tree.map``,
``jax.tree_util.register_pytree_node``, ``jax.Array``,
``jax.nn.silu``, ``jax.lax.stop_gradient``, etc.) are
stable across JAX 0.6 through 0.9.2.  No wrappers are
currently needed for the JAX side.

Flax NNX compatibility
----------------------
``flax.nnx`` moved from ``flax.experimental.nnx``
(Flax 0.8--0.9) to ``flax.nnx`` (Flax 0.10+).  Since
JAX >= 0.6 requires Flax >= 0.10, the top-level import
path is always available for in-range versions.

``Variable.value`` was deprecated in Flax 0.12.1 in
favour of ``Variable.get_value()`` /
``Variable[...]``.  Use :func:`param_value` as a
cross-version accessor.
"""

import jax

# ---- Flax NNX import ----
try:
    from flax import nnx                   # Flax >= 0.10
except ImportError:
    try:
        from flax.experimental import nnx  # Flax 0.8--0.9
    except ImportError:
        nnx = None  # Flax not installed


# ---- Parameter value access ----------------------------

def param_value(param):
    """Extract the raw array from an NNX Variable.

    Uses ``get_value()`` (Flax >= 0.12.1) when
    available, falling back to ``.value``
    (Flax 0.10--0.12.0).

    Args:
        param: An ``nnx.Param`` (or any NNX
            ``Variable``) instance.

    Returns:
        The underlying ``jax.Array``.
    """
    if hasattr(param, 'get_value'):
        return param.get_value()
    return param.value


# ---- Pytree registration ------------------------------

def register_pytree(cls):
    """Decorator for JAX pytree registration.

    Compatible with JAX 0.6 through 0.9.  The
    decorated class must define:

    * ``tree_flatten(self) -> (children, aux_data)``
    * ``tree_unflatten(cls, aux, children) -> cls``
    """
    jax.tree_util.register_pytree_node(
        cls,
        lambda obj: obj.tree_flatten(),
        lambda aux, children: cls.tree_unflatten(
            aux, children,
        ),
    )
    return cls
