"""Periodic primitives for HEG-style neural-network wavefunctions.

Provides:
  * :class:`PeriodicLattice` — a JAX-pytree lattice descriptor
    (primitive vectors, metric tensor, volume).
  * Position helpers — :func:`wrap_to_cell`, :func:`fractional_coords`,
    :func:`minimum_image_diff`.
  * Periodic pairwise features — :func:`periodic_sincos_features`
    and :func:`periodic_norm_sq`, following Cassella et al.
    (PRL 130, 036401, 2023; arXiv:2202.05183).

The periodic norm is smooth on the torus and reduces (up to the
2π scale factor) to the Euclidean norm at short range, preserving
the analytic structure needed for electron-electron cusps.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp


class PeriodicLattice(NamedTuple):
    """Bravais lattice descriptor.

    Column-major convention: the columns of ``A`` are the primitive
    vectors ``a_i`` so that ``r = A @ s`` with ``s`` the fractional
    coordinates.  The metric tensor is ``S = A^T A``, i.e. the
    Gram matrix of the primitive vectors.

    Attributes:
        A: Primitive vectors as columns, shape ``(3, 3)``.
        A_inv: Precomputed ``A^{-1}``.
        metric: Metric tensor ``S = A^T A``.
        volume: Cell volume ``|det A|``.
    """

    A: jax.Array
    A_inv: jax.Array
    metric: jax.Array
    volume: jax.Array


def make_cubic_lattice(L: float) -> PeriodicLattice:
    """Build a cubic :class:`PeriodicLattice` of side length *L*."""
    A = jnp.eye(3) * L
    A_inv = jnp.eye(3) / L
    metric = jnp.eye(3) * (L * L)
    volume = jnp.asarray(L ** 3)
    return PeriodicLattice(A=A, A_inv=A_inv, metric=metric, volume=volume)


def make_lattice(A: jax.Array) -> PeriodicLattice:
    """Build a :class:`PeriodicLattice` from a primitive-vector matrix.

    Args:
        A: Matrix whose *columns* are the primitive vectors,
            shape ``(3, 3)``.
    """
    A = jnp.asarray(A, dtype=jnp.float64)
    A_inv = jnp.linalg.inv(A)
    metric = A.T @ A
    volume = jnp.abs(jnp.linalg.det(A))
    return PeriodicLattice(A=A, A_inv=A_inv, metric=metric, volume=volume)


# ---------------------------------------------------------------------
# Position helpers
# ---------------------------------------------------------------------

def fractional_coords(r: jax.Array, lattice: PeriodicLattice) -> jax.Array:
    """Map Cartesian positions to fractional coordinates.

    Args:
        r: ``(..., 3)`` Cartesian positions.
        lattice: :class:`PeriodicLattice`.

    Returns:
        ``(..., 3)`` array with ``s = A^{-1} r``.
    """
    return r @ lattice.A_inv.T


def wrap_to_cell(r: jax.Array, lattice: PeriodicLattice) -> jax.Array:
    """Wrap Cartesian positions into the primitive cell ``s ∈ [0, 1)``.

    Used for MCMC walker wrapping; the NN wavefunction itself is
    translationally invariant under simulation-cell shifts, but
    keeping walkers in a single image avoids floating-point drift.
    """
    s = fractional_coords(r, lattice)
    s_wrapped = s - jnp.floor(s)
    return s_wrapped @ lattice.A.T


def minimum_image_diff(
    diff: jax.Array, lattice: PeriodicLattice,
) -> jax.Array:
    """Fractional minimum-image convention for pairwise differences.

    Wraps ``diff`` so its fractional coordinates lie in
    ``s ∈ [-0.5, 0.5)``.  For *orthorhombic* lattices this is the
    Cartesian-shortest image; for general triclinic lattices the
    Cartesian-shortest image may lie in a different integer cell
    (one of the 27 neighbours of the Wigner–Seitz cell).  For the
    cubic and orthorhombic HEG simulation cells used in this
    project the two conventions coincide.

    Args:
        diff: ``(..., 3)`` Cartesian differences.
        lattice: :class:`PeriodicLattice`.

    Returns:
        ``(..., 3)`` Cartesian diffs with fractional coords in
        ``[-0.5, 0.5)``.
    """
    s = fractional_coords(diff, lattice)
    s_min = s - jnp.round(s)
    return s_min @ lattice.A.T


# ---------------------------------------------------------------------
# Periodic pairwise features (Cassella et al. 2023)
# ---------------------------------------------------------------------

def periodic_sincos_features(
    diff: jax.Array, lattice: PeriodicLattice,
) -> jax.Array:
    """Smooth, periodic pairwise feature ``(sin 2π s, cos 2π s)``.

    Args:
        diff: ``(..., 3)`` Cartesian differences.
        lattice: :class:`PeriodicLattice`.

    Returns:
        ``(..., 6)`` feature array ``[sin(2π s₁), sin(2π s₂),
        sin(2π s₃), cos(2π s₁), cos(2π s₂), cos(2π s₃)]``.
    """
    s = fractional_coords(diff, lattice)
    two_pi_s = 2.0 * jnp.pi * s
    return jnp.concatenate(
        [jnp.sin(two_pi_s), jnp.cos(two_pi_s)], axis=-1,
    )


_INV_FOUR_PI_SQ = 1.0 / (4.0 * jnp.pi * jnp.pi)
_INV_TWO_PI = 1.0 / (2.0 * jnp.pi)


def periodic_norm_sq(
    diff: jax.Array, lattice: PeriodicLattice,
) -> jax.Array:
    """Squared smooth periodic norm in **Bohr² units**.

    Implements the Cassella–Sutterud formula with the 1/(2π)²
    normalisation used by FermiNet (pbc/feature_layer.py) — i.e.
    the output is in Bohr², matching Euclidean distances at
    short range rather than ``(2π)² |r|²``.  This lets the Kato
    cusp coefficients (which are defined in Bohr units) be used
    directly with the smooth norm, without the ``2π`` rescale
    that previously forced us to fall back to the minimum-image
    Euclidean distance for the cusp.

    .. math::

        \\|s\\|_p^2 = \\frac{1}{(2\\pi)^2} \\sum_{ij} \\left[
            (1-\\cos 2\\pi s_i) S_{ij} (1-\\cos 2\\pi s_j)
          + \\sin(2\\pi s_i) S_{ij} \\sin(2\\pi s_j) \\right]

    with metric tensor ``S_{ij} = a_i · a_j``.  Smooth on the
    torus (no minimum-image kink) and reduces to ``|diff|²``
    (Bohr²) as ``diff → 0``.

    Args:
        diff: ``(..., 3)`` Cartesian differences.
        lattice: :class:`PeriodicLattice`.

    Returns:
        ``(...,)`` array of squared periodic norms in Bohr².
    """
    s = fractional_coords(diff, lattice)
    two_pi_s = 2.0 * jnp.pi * s
    sin_s = jnp.sin(two_pi_s)
    one_minus_cos = 1.0 - jnp.cos(two_pi_s)
    S = lattice.metric

    # (1-cos)^T S (1-cos)
    t1 = jnp.einsum(
        '...i,ij,...j->...', one_minus_cos, S, one_minus_cos,
    )
    # sin^T S sin
    t2 = jnp.einsum('...i,ij,...j->...', sin_s, S, sin_s)
    return _INV_FOUR_PI_SQ * (t1 + t2)


def periodic_norm(
    diff: jax.Array, lattice: PeriodicLattice, *, safe: bool = True,
) -> jax.Array:
    """Smooth periodic norm in **Bohr units**, matching FermiNet.

    Equal to ``sqrt(periodic_norm_sq)``; short-range behaviour is
    ``|diff|`` (Bohr), *not* the ``2π |diff|`` of the un-normalised
    Cassella formula.  The Bohr-unit convention lets the Kato cusp
    (which assumes Bohr-scaled ``r``) consume this distance
    directly, removing the need for a separate minimum-image
    Euclidean helper for the cusp.

    Args:
        diff: ``(..., 3)`` Cartesian differences.
        lattice: :class:`PeriodicLattice`.
        safe: If True, add machine epsilon inside sqrt to avoid
            a zero-gradient singularity at the origin.
    """
    n2 = periodic_norm_sq(diff, lattice)
    if safe:
        eps = jnp.finfo(n2.dtype).eps
        return jnp.sqrt(n2 + eps)
    return jnp.sqrt(n2)


def periodic_pairwise_diffs(
    r1: jax.Array, r2: jax.Array, lattice: PeriodicLattice,
) -> jax.Array:
    """Periodic analogue of :func:`..physics.pairwise_diffs`.

    Returns a feature tensor shaped like the molecular
    ``pairwise_diffs``: the last dimension concatenates the
    minimum-image difference vector with the squared periodic
    norm, giving ``(n1, n2, 4)``.  This keeps the downstream
    :class:`ExponentialEnvelopes` / GNN shapes identical so a
    periodic ansatz can be slotted in by swapping only the
    feature producer.

    Args:
        r1: ``(n1, 3)``.
        r2: ``(n2, 3)``.
        lattice: :class:`PeriodicLattice`.

    Returns:
        ``(n1, n2, 4)`` array.
    """
    diff = r1[:, None, :] - r2[None, :, :]
    diff_mi = minimum_image_diff(diff, lattice)
    n2 = periodic_norm_sq(diff, lattice)
    return jnp.concatenate(
        [diff_mi, n2[..., None]], axis=-1,
    )
