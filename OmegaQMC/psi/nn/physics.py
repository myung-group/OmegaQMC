"""Physics utilities for NN trial wavefunctions.

Ported from ``deepqmc/physics.py`` — pairwise distances and
the efficient O(N) Laplacian via ``jax.linearize``.
"""

from collections.abc import Callable

import jax
import jax.numpy as jnp

from .utils import norm


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


def laplacian(
    f: Callable[[jax.Array], jax.Array],
) -> Callable[
    [jax.Array], tuple[jax.Array, jax.Array]
]:
    """O(N) Laplacian via ``jax.linearize`` + ``fori_loop``.

    Given a scalar function *f* of a flat coordinate vector,
    returns a function that computes ``(nabla^2 f, grad f)``.

    This is more efficient than the full Hessian approach
    ``jax.hessian`` which scales as O(N^2).

    Args:
        f: Scalar function of a 1-D coordinate array.

    Returns:
        Function ``(x) -> (laplacian, gradient)``.
    """
    def lap(x: jax.Array) -> tuple[jax.Array, jax.Array]:
        n_coord = len(x)
        grad_f = jax.grad(f)
        df, grad_f_jvp = jax.linearize(grad_f, x)
        eye = jnp.eye(n_coord)
        d2f = (
            lambda i, val: val + grad_f_jvp(eye[i])[i]
        )
        d2f_sum = jax.lax.fori_loop(
            0, n_coord, d2f, 0.0,
        )
        return d2f_sum, df
    return lap
