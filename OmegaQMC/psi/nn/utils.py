"""Utility functions for NN trial wavefunctions.

Ported from ``deepqmc/utils.py`` — only the subset used
by the wavefunction and GNN modules.
"""

import jax.numpy as jnp


def norm(rs, safe=False, axis=-1):
    """Compute vector norm, optionally with safe sqrt.

    Args:
        rs: Input array.
        safe: If True, add machine epsilon inside sqrt
            to avoid zero-gradient at the origin.
        axis: Axis along which to compute the norm.

    Returns:
        Norm of *rs* along *axis*.
    """
    if safe:
        eps = jnp.finfo(rs.dtype).eps
        return jnp.sqrt(eps + (rs * rs).sum(axis=axis))
    return jnp.linalg.norm(rs, axis=axis)


def triu_flat(x):
    """Extract strict upper-triangular elements.

    Args:
        x: Square matrix ``(..., n, n)``.

    Returns:
        Flat array of ``n*(n-1)/2`` elements.
    """
    i, j = jnp.triu_indices(x.shape[-1], 1)
    return x[..., i, j]


def flatten(x, start_axis=0):
    """Flatten dimensions from *start_axis* onward.

    Args:
        x: Input array.
        start_axis: First axis to collapse.

    Returns:
        Array with ``x.shape[:start_axis]`` prepended
        to a single flattened trailing dimension.
    """
    return x.reshape(*x.shape[:start_axis], -1)


def unflatten(x, axis, shape):
    """Reshape a single axis into *shape*.

    Args:
        x: Input array.
        axis: Axis to unflatten (negative indexing
            supported).
        shape: Target shape for the unflattened axis.

    Returns:
        Array with *axis* expanded into *shape*.
    """
    if axis < 0:
        axis = len(x.shape) + axis
    begin = x.shape[:axis]
    end = x.shape[axis + 1:]
    return x.reshape(*begin, *shape, *end)
