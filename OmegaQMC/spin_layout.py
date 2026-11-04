"""OmegaQMC's electron layout for walkers, and its grouping by spin.

Walkers store the electrons of one configuration as an ``(n_e, 3)``
array in OmegaQMC's *layout*: the first ``2 * min(n_up, n_down)``
electrons alternate spin up, spin down, spin up, ...; the remaining
``|n_up - n_down|`` electrons, at the end, all have the majority spin.
For ``n_up == n_down`` and ``n_up == n_down + 1`` this is plain
interleaving (spin up at the even positions, spin down at the odd ones),
the convention OmegaQMC has always used; the trailing block extends it
to any number of unpaired electrons, e.g. the N atom (5, 2) is laid
out ``u d u d u u u``.

The networks see the electrons *grouped* by spin, spin up first.
:func:`to_grouped` maps a layout array to that order and
:func:`to_layout` maps it back.  For plain interleaving
:func:`to_grouped` keeps the original strided-slice implementation, so
existing systems compile to the same program as before.
"""

from __future__ import annotations

import numpy as np
import jax.numpy as jnp


def is_plain_interleaved(n_up: int, n_down: int) -> bool:
    """Whether the layout is plain interleaving (``x[::2]`` = spin up)."""
    return int(n_up) - int(n_down) in (0, 1)


def spin_indices(n_up: int, n_down: int):
    """Layout positions of the spin-up and of the spin-down electrons.

    Returns:
        ``(up_idx, down_idx)``: sorted int arrays of lengths ``n_up``
        and ``n_down``.
    """
    n_up, n_down = int(n_up), int(n_down)
    if n_up < 0 or n_down < 0:
        raise ValueError(f"negative electron count ({n_up}, {n_down})")
    n_pair = min(n_up, n_down)
    paired = np.arange(2 * n_pair)
    tail = np.arange(2 * n_pair, n_up + n_down)
    up = paired[0::2]
    down = paired[1::2]
    if n_up > n_down:
        up = np.concatenate([up, tail])
    else:
        down = np.concatenate([down, tail])
    return up.astype(np.int64), down.astype(np.int64)


def spin_of_position(n_up: int, n_down: int) -> np.ndarray:
    """Spin of every layout position: 0 for up, 1 for down."""
    up, down = spin_indices(n_up, n_down)
    s = np.empty(int(n_up) + int(n_down), dtype=np.int64)
    s[up] = 0
    s[down] = 1
    return s


def grouping_permutation(n_up: int, n_down: int) -> np.ndarray:
    """``perm`` with ``x_grouped = x[perm]`` (spin-up electrons first)."""
    up, down = spin_indices(n_up, n_down)
    return np.concatenate([up, down])


def layout_permutation(n_up: int, n_down: int) -> np.ndarray:
    """Inverse of :func:`grouping_permutation`: ``x = x_grouped[perm]``."""
    return np.argsort(grouping_permutation(n_up, n_down))


def to_grouped(x, n_up: int, n_down: int, axis: int = -2):
    """Reorder electrons from the layout to spin-grouped order.

    Args:
        x: Array whose *axis* indexes the ``n_up + n_down``
            electrons in OmegaQMC's layout (default: the second-last
            axis, as in ``(..., n_e, 3)``).
        n_up, n_down: Electron counts per spin.
        axis: Electron axis.

    Returns:
        *x* with spin-up electrons first along *axis*.
    """
    n_e = int(n_up) + int(n_down)
    if x.shape[axis] != n_e:
        raise ValueError(
            f"electron axis has {x.shape[axis]} entries, expected"
            f" n_up + n_down = {n_e}"
        )
    if is_plain_interleaved(n_up, n_down):
        ax = axis % x.ndim
        sl_up = (slice(None),) * ax + (slice(0, None, 2),)
        sl_dn = (slice(None),) * ax + (slice(1, None, 2),)
        return jnp.concatenate([x[sl_up], x[sl_dn]], axis=ax)
    return jnp.take(x, jnp.asarray(grouping_permutation(n_up, n_down)),
                    axis=axis)


def to_layout(x_grouped, n_up: int, n_down: int, axis: int = -2):
    """Inverse of :func:`to_grouped`."""
    return jnp.take(x_grouped,
                    jnp.asarray(layout_permutation(n_up, n_down)),
                    axis=axis)
