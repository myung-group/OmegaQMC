"""On-the-fly observable estimators for HEG QMC sampling.

These are intensive (per-walker) estimators that can be averaged over
the MCMC trajectory to give physical observables of the wavefunction.
The implementations are dimension-agnostic (work for 2D and 3D HEG)
and JIT-friendly (pure JAX functions).

Observables provided
--------------------

* :func:`structure_factor` — ``S(k) = (1/N) <|sum_i exp(i k.r_i)|^2>``;
  Bragg peaks reveal positional order (Wigner crystal Bravais lattice).

* :func:`spin_structure_factor` —
  ``S_spin(k) = (1/N) <|sum_i sigma_i exp(i k.r_i)|^2>`` with
  ``sigma_i = +-1`` for up/down; reveals magnetic order
  (antiferromagnetic Wigner crystal, ferromagnetic instability).

* :func:`pair_correlation` — ``g(r)`` accumulated as a histogram of
  inter-electron distances, binned in radial bins of width ``dr``.
  Resolves the exchange-correlation hole.

* :func:`momentum_distribution` — ``n(k) = <a_k^dag a_k>`` at the
  occupied k-vectors of the Fermi sea; gives the renormalisation
  factor ``Z`` (jump at k_F) and complements
  Holzmann 2009 cross-checks.

The cumulative running averages are accumulated by callers (the VMC
driver) which can maintain a small dataclass of moments per
observable; helper :class:`StructureFactorAccumulator` does this
bookkeeping for the structure factor case (the most useful for
phase-boundary detection).

For triangular Wigner-crystal detection use
:func:`reciprocal_lattice_vectors_triangular` to get the relevant
``k_BZ`` vectors.
"""

from typing import NamedTuple, Optional

import numpy as np
import jax
import jax.numpy as jnp


# ---------------------------------------------------------------------
# k-space grid helpers
# ---------------------------------------------------------------------

def reciprocal_grid_2d(L: float, n_max: int) -> jax.Array:
    """Cartesian k-vectors on the simulation-cell reciprocal lattice.

    Returns all 2D ``(k_x, k_y)`` with integer indices in
    ``[-n_max, n_max]^2`` excluding ``(0, 0)`` — useful as a default
    set for ``S(k)`` accumulation around the first Brillouin zone.
    """
    rng = np.arange(-n_max, n_max + 1)
    nx, ny = np.meshgrid(rng, rng, indexing='ij')
    n_ints = np.stack([nx.ravel(), ny.ravel()], axis=1)
    mask = np.any(n_ints != 0, axis=1)
    n_ints = n_ints[mask]
    dk = 2.0 * np.pi / L
    return jnp.asarray(n_ints * dk, dtype=jnp.float64)


def reciprocal_lattice_vectors_triangular(rs: float, n_shell: int = 1):
    """First-shell triangular-WC reciprocal lattice vectors.

    For the 2D Wigner crystal at density ``n = 1/(pi rs^2)`` on a
    triangular Bravais lattice with primitive vectors
    ``a1 = a (1, 0), a2 = a (1/2, sqrt(3)/2)`` and lattice constant
    ``a = sqrt(2 / (sqrt(3) n))``, the reciprocal primitives are
    ``b1 = (4 pi/(a sqrt(3))) (sqrt(3)/2, -1/2)`` and
    ``b2 = (4 pi/(a sqrt(3))) (0, 1)``.

    Returns the first ``6 * n_shell`` reciprocal-lattice vectors —
    the locations where Bragg peaks appear in ``S(k)`` if the
    crystal phase is realised.
    """
    n = 1.0 / (np.pi * rs ** 2)
    a = np.sqrt(2.0 / (np.sqrt(3.0) * n))
    b_mag = 4.0 * np.pi / (a * np.sqrt(3.0))
    # First shell: 6 vectors at angles 0, 60, 120, ..., 300.
    angles = np.arange(6 * n_shell) * (np.pi / 3.0) / n_shell
    return np.stack(
        [b_mag * np.cos(angles), b_mag * np.sin(angles)], axis=1,
    )


# ---------------------------------------------------------------------
# Structure factor S(k)
# ---------------------------------------------------------------------

def structure_factor(r: jax.Array, k_grid: jax.Array) -> jax.Array:
    """Per-configuration structure factor on a fixed k-grid.

    Args:
        r: ``(n_elec, dim)`` electron positions.
        k_grid: ``(n_k, dim)`` Cartesian k-vectors.

    Returns:
        ``(n_k,)`` structure factor estimates ``|sum_i exp(i k.r_i)|^2 / N``.
    """
    n_elec = r.shape[0]
    # (n_k, n_elec) phases.
    phase = r @ k_grid.T  # (n_elec, n_k)
    re = jnp.sum(jnp.cos(phase), axis=0)  # (n_k,)
    im = jnp.sum(jnp.sin(phase), axis=0)
    return (re ** 2 + im ** 2) / n_elec


def spin_structure_factor(
    r: jax.Array, k_grid: jax.Array, n_up: int,
) -> jax.Array:
    """Per-configuration spin structure factor.

    With ``sigma_i = +1`` for the first ``n_up`` electrons (spin up)
    and ``-1`` for the remaining (spin down).  Antiferromagnetic
    order shows a peak at the magnetic ordering vector.
    """
    n_elec = r.shape[0]
    spin = jnp.concatenate([
        jnp.ones(n_up), -jnp.ones(n_elec - n_up),
    ])
    phase = r @ k_grid.T  # (n_elec, n_k)
    re = jnp.sum(spin[:, None] * jnp.cos(phase), axis=0)
    im = jnp.sum(spin[:, None] * jnp.sin(phase), axis=0)
    return (re ** 2 + im ** 2) / n_elec


# ---------------------------------------------------------------------
# Pair correlation function g(r)
# ---------------------------------------------------------------------

def pair_correlation(
    r: jax.Array, n_up: int, lattice_diff_fn, r_max: float,
    n_bins: int = 50, channel: str = 'all',
) -> jax.Array:
    """Per-configuration unnormalised pair-correlation histogram.

    Args:
        r: ``(n_elec, dim)`` positions.
        n_up: Number of up-spin electrons (rows ``[0:n_up]``).
        lattice_diff_fn: ``(diff -> minimum-image diff)`` for the
            simulation cell.
        r_max: Maximum pair distance to include in the histogram.
        n_bins: Number of radial bins.
        channel: ``'all'``, ``'same'``, or ``'anti'`` — which spin
            pairs to count.

    Returns:
        ``(n_bins,)`` histogram of pair distances (integer counts).
        The caller normalises by total samples and the radial
        ``2 pi r dr`` (2D) or ``4 pi r^2 dr`` (3D) volume element.
    """
    n_elec = r.shape[0]
    # All ordered pair differences (i, j) with i < j.
    i, j = jnp.triu_indices(n_elec, k=1)
    diff = r[i] - r[j]
    diff_mi = lattice_diff_fn(diff)
    dist = jnp.linalg.norm(diff_mi, axis=-1)

    if channel == 'same':
        same_mask = ((i < n_up) & (j < n_up)) | ((i >= n_up) & (j >= n_up))
        weight = jnp.where(same_mask, 1.0, 0.0)
    elif channel == 'anti':
        anti_mask = (i < n_up) ^ (j < n_up)  # XOR: one up, one down
        weight = jnp.where(anti_mask, 1.0, 0.0)
    elif channel == 'all':
        weight = jnp.ones_like(dist)
    else:
        raise ValueError(f"channel must be 'all', 'same', or 'anti'")

    # Histogram via index_add over jnp.digitize.
    bin_idx = jnp.clip(
        (dist / r_max * n_bins).astype(jnp.int32), 0, n_bins - 1,
    )
    hist = jnp.zeros(n_bins).at[bin_idx].add(weight)
    return hist


# ---------------------------------------------------------------------
# Momentum distribution n(k)
# ---------------------------------------------------------------------

def momentum_distribution(
    r: jax.Array, k_orbitals: jax.Array,
) -> jax.Array:
    """Single-particle momentum distribution.

    For a Slater-determinant trial wavefunction the natural-orbital
    occupation matrix is diagonal in the plane-wave basis to leading
    order; the diagonal elements are estimated via
    ``n(k) ~ (1/N) sum_i |<phi_k | psi_i>|^2`` which evaluated on a
    walker reduces to the per-electron occupation of the k-orbital.
    For a non-interacting Fermi sea the result is 1 inside the
    Fermi surface and 0 outside; for the interacting state the jump
    at the Fermi surface is the renormalisation factor ``Z``.

    This estimator is *approximate* for non-product trial states but
    sufficient for the cross-check vs Holzmann 2009.

    Args:
        r: ``(n_elec, dim)`` positions.
        k_orbitals: ``(n_orb, dim)`` k-vectors of the orbital basis.

    Returns:
        ``(n_orb,)`` occupations.
    """
    # |<phi_k | psi_i>|^2 ~ |sum over walker electrons of plane-wave
    # phases|^2 averaged.
    phase = r @ k_orbitals.T  # (n_elec, n_orb)
    re = jnp.sum(jnp.cos(phase), axis=0)
    im = jnp.sum(jnp.sin(phase), axis=0)
    n_elec = r.shape[0]
    return (re ** 2 + im ** 2) / n_elec ** 2


# ---------------------------------------------------------------------
# Cumulative-average accumulators
# ---------------------------------------------------------------------

class StructureFactorAccumulator(NamedTuple):
    """Running first-moment accumulator for ``S(k)``.

    Use ``initial(...)`` to construct, then call ``update(state, sk)``
    after each block to incorporate the latest per-block average.
    Final estimate is ``state.s_sum / state.count``.
    """

    s_sum: jax.Array  # (n_k,) — sum of S(k) per accepted walker block
    count: int


def sf_accumulator_initial(n_k: int) -> StructureFactorAccumulator:
    return StructureFactorAccumulator(
        s_sum=jnp.zeros(n_k),
        count=0,
    )


def sf_accumulator_update(
    state: StructureFactorAccumulator, sk_block: jax.Array,
) -> StructureFactorAccumulator:
    """Add one per-walker-block estimate ``sk_block`` (shape ``(n_k,)``)."""
    return StructureFactorAccumulator(
        s_sum=state.s_sum + sk_block,
        count=state.count + 1,
    )


def sf_accumulator_mean(
    state: StructureFactorAccumulator,
) -> jax.Array:
    return state.s_sum / max(1, state.count)
