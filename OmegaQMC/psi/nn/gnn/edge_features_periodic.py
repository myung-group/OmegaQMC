"""Periodic edge features for HEG-style neural-network wavefunctions.

These classes are drop-in replacements for the molecular
:mod:`~.edge_features` versions, but aware of the simulation-cell
periodic boundary conditions.  Each class exposes the same
``(d) -> feature`` interface and ``__len__`` as the molecular
counterpart, so an :class:`~.electron_gnn.ElectronGNN` constructed
with periodic edge features is otherwise identical to the molecular
GNN — no other GNN-side changes are required.

The three feature types provided match Cassella et al. 2023
(arXiv:2202.05183) and the generalisations in Pescia et al. 2023
(WAPNet):

* :class:`PeriodicDifferenceEdgeFeature` — minimum-image 3-vector.
  Direct analogue of :class:`DifferenceEdgeFeature` with a kink at
  the Wigner-Seitz face (acceptable for cubic cells, where the
  face is far from typical short-range attention support).

* :class:`PeriodicSinCosFeature` — smooth 6-vector
  ``(sin 2π s_x, sin 2π s_y, sin 2π s_z, cos 2π s_x, cos 2π s_y,
  cos 2π s_z)`` with ``s = A⁻¹ d``.  Smooth across the cell
  boundary; preferred input for the attention layers.

* :class:`PeriodicDistancePowerEdgeFeature` — powers of the
  Cassella smooth periodic norm ``‖s‖_p`` (see
  :func:`~..periodic.periodic_norm`).  Used as a short-range
  distance feature; the smooth norm reduces to ``2π |r|`` at
  short range so electron-electron cusps remain representable
  downstream.
"""

from typing import Optional, Sequence

import jax
import jax.numpy as jnp

from ..periodic import (
    PeriodicLattice,
    minimum_image_diff,
    periodic_norm,
    periodic_sincos_features,
)
from ..utils import norm


class PeriodicDifferenceEdgeFeature:
    """Minimum-image wrapped 3-vector difference feature.

    Args:
        lattice: :class:`PeriodicLattice` describing the simulation
            cell.
        log_rescale: If True, rescale the output by ``log(1 + r) / r``
            where ``r`` is the minimum-image Euclidean distance.
    """

    def __init__(
        self,
        *,
        lattice: PeriodicLattice,
        log_rescale: bool = False,
    ):
        self.lattice = lattice
        self.log_rescale = log_rescale

    def __call__(self, d: jax.Array) -> jax.Array:
        d_mi = minimum_image_diff(d, self.lattice)
        if self.log_rescale:
            r = norm(d_mi, safe=True)
            d_mi = d_mi * (jnp.log1p(r) / r)[..., None]
        return d_mi

    def __len__(self) -> int:
        return 3


class PeriodicSinCosFeature:
    """Smooth 6-dimensional ``(sin 2π s, cos 2π s)`` feature.

    Args:
        lattice: :class:`PeriodicLattice`.
    """

    def __init__(self, *, lattice: PeriodicLattice):
        self.lattice = lattice

    def __call__(self, d: jax.Array) -> jax.Array:
        return periodic_sincos_features(d, self.lattice)

    def __len__(self) -> int:
        return 6


class PeriodicDistancePowerEdgeFeature:
    """Powers of the Cassella smooth periodic distance.

    Uses the smooth periodic norm (see
    :func:`~..periodic.periodic_norm`) rather than the minimum-image
    Euclidean distance, so there is no gradient kink at the cell
    face.  At short range ``‖s‖_p ≈ 2π |r|``, so positive powers
    behave like ``|r|^p`` up to the ``(2π)^p`` scale factor that
    the downstream MLP can absorb.

    Args:
        lattice: :class:`PeriodicLattice`.
        powers: List of exponents (e.g. ``[1]``).
        eps: Regulariser for negative powers.
        log_rescale: If True, rescale by ``log(1 + r_p) / r_p``
            (``r_p`` = smooth periodic norm).
    """

    def __init__(
        self,
        *,
        lattice: PeriodicLattice,
        powers: Sequence,
        eps: Optional[float] = None,
        log_rescale: bool = False,
    ):
        if any(p < 0 for p in powers):
            assert eps is not None, (
                "Negative powers require `eps` to avoid division "
                "by zero at the origin."
            )
        self.lattice = lattice
        self.powers = jnp.asarray(powers)
        self.eps = float(eps) if eps is not None else 0.0
        self.log_rescale = log_rescale

    def __call__(self, d: jax.Array) -> jax.Array:
        r = periodic_norm(d, self.lattice, safe=True)
        pw = jnp.where(
            self.powers > 0,
            r[..., None] ** self.powers,
            1.0 / (r[..., None] ** (-self.powers) + self.eps),
        )
        if self.log_rescale:
            pw = pw * (jnp.log1p(r) / r)[..., None]
        return pw

    def __len__(self) -> int:
        return len(self.powers)


class PeriodicCombinedEdgeFeature:
    """Concatenation of multiple periodic edge features.

    Mirrors :class:`..edge_features.CombinedEdgeFeature` so a
    periodic GNN can mix several feature types (e.g. sin/cos +
    power-of-distance).

    Args:
        features: List of periodic edge feature objects.
    """

    def __init__(self, *, features):
        self.features = features

    def __call__(self, d: jax.Array) -> jax.Array:
        return jnp.concatenate([f(d) for f in self.features], axis=-1)

    def __len__(self) -> int:
        return sum(len(f) for f in self.features)
