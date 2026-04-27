"""Dimension-dispatch wrappers for Ewald sums.

Routes ``build_ewald_tables`` and ``ewald_pair_energy`` calls to the
3D (:mod:`OmegaQMC.observables.ewald`) or 2D
(:mod:`OmegaQMC.observables.ewald_2d`) implementation based on a
``dim`` argument.  Lets the VMC drivers and run scripts stay
dimension-agnostic without conditionals at every call site.

The 2D and 3D ``EwaldTables`` types are distinct NamedTuples but
expose compatible attributes (``L``, ``eta``, ``madelung``,
``R_vecs``, ``G_vecs``, etc.), so downstream consumers that only
read these attributes work with either variant.
"""

from typing import Optional

from . import ewald as _ewald_3d
from . import ewald_2d as _ewald_2d


def build_ewald_tables_dim(
    L: float,
    *,
    dim: int = 3,
    eta: Optional[float] = None,
    n_real: int = 3,
    n_recip: int = 6,
):
    """Build Ewald tables for the requested spatial dimension.

    Args:
        L: Cubic (3D) or square (2D) cell side length.
        dim: 3 (default) or 2.
        eta, n_real, n_recip: as in the per-dim builders.
    """
    if dim == 3:
        return _ewald_3d.build_ewald_tables(
            L, eta=eta, n_real=n_real, n_recip=n_recip,
        )
    if dim == 2:
        return _ewald_2d.build_ewald_2d_tables(
            L, eta=eta, n_real=n_real, n_recip=n_recip,
        )
    raise ValueError(f"dim must be 2 or 3, got {dim}")


def ewald_pair_energy_dim(r, tables, *, dim: Optional[int] = None):
    """Dispatch ``ewald_pair_energy`` based on ``r``'s spatial dim
    (or an explicit ``dim`` argument)."""
    if dim is None:
        dim = int(r.shape[-1])
    if dim == 3:
        return _ewald_3d.ewald_pair_energy(r, tables)
    if dim == 2:
        return _ewald_2d.ewald_2d_pair_energy(r, tables)
    raise ValueError(f"dim must be 2 or 3, got {dim}")
