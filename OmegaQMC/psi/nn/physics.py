"""Physics utilities for NN trial wavefunctions.

Ported from ``deepqmc/physics.py`` — pairwise distances.
The O(N) Laplacian helper lives in :mod:`OmegaQMC.utils`
as ``laplacian_linearize`` and is re-exported here as
``laplacian`` for backward compatibility.
"""

import jax
import jax.numpy as jnp

from ...utils import laplacian_linearize as laplacian
from .utils import norm

__all__ = [
    "pairwise_diffs",
    "pairwise_self_distance",
    "laplacian",
]


def pairwise_diffs(
    coords1: jax.Array, coords2: jax.Array,
) -> jax.Array:
    """Pairwise differences with appended squared distance.

    Args:
        coords1: shape ``(n1, 3)``.
        coords2: shape ``(n2, 3)``.

    Returns:
        Array of shape ``(n1, n2, 4)`` where the last
        element along axis -1 is the squared distance.
    """
    diffs = (
        coords1[..., :, None, :]
        - coords2[..., None, :, :]
    )
    return jnp.concatenate(
        [diffs, (diffs**2).sum(axis=-1, keepdims=True)],
        axis=-1,
    )


def pairwise_self_distance(
    coords: jax.Array, full: bool = False,
) -> jax.Array:
    """Pairwise distances within a single set of points.

    Args:
        coords: shape ``(n, 3)``.
        full: If True return the full ``(n, n)`` matrix;
            otherwise return the upper-triangular flat
            vector of length ``n*(n-1)/2``.

    Returns:
        Distance array.
    """
    i, j = jnp.triu_indices(coords.shape[-2], k=1)
    diffs = (
        coords[..., :, None, :]
        - coords[..., None, :, :]
    )
    dists = norm(diffs[..., i, j, :], safe=True, axis=-1)
    if full:
        dists = (
            jnp.zeros(diffs.shape[:-1])
            .at[..., i, j].set(dists)
            .at[..., j, i].set(dists)
        )
    return dists
