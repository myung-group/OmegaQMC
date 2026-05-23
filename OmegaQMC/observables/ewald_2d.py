"""2D Ewald summation for quasi-2D Coulomb interactions in a periodic cell.

For electrons confined to a 2D plane (z=0) interacting via the full 3D
``1/|r|`` Coulomb potential.  The 2D Ewald formula (Parry 1975, Heyes
1981) splits the bare ``1/|r_par|`` lattice sum at z=0 into a real-space
part with the same ``erfc(eta|r+R|)/|r+R|`` structure as 3D Ewald, and a
reciprocal-space part with a *qualitatively different* functional form
than 3D — the reciprocal weights are ``(2 pi/A) · erfc(G/(2 eta)) / G``
rather than the 3D ``(4 pi/V) · exp(-G^2/(4 eta^2)) / G^2``.

The reciprocal weight comes from the 2D Fourier transform of the
long-range piece ``erf(eta r)/r`` evaluated at z=0.  Using
Gradshteyn-Ryzhik 6.616.1, ``int_0^inf J_0(qr) erf(eta r) dr =
erf(q/(2 eta))/q``, so

    FT_2D[ erf(eta r)/r ]   = (2 pi / q) · erf(q/(2 eta))
    FT_2D[ erfc(eta r)/r ]  = (2 pi / q) · erfc(q/(2 eta))

and the Ewald sum is

    v_Ew(r) = sum_R   erfc(eta |r+R|) / |r+R|
            + (2 pi/A) sum_{G != 0} erfc(G/(2 eta))/G · cos(G·r)
            - 2 sqrt(pi) / (A eta).

The constant ``-2 sqrt(pi) / (A eta)`` is the finite ``G -> 0`` limit
of the reciprocal weight after the divergent ``1/G`` has been removed
by the neutralising background charge (the 2D analog of the 3D
``-pi/(V eta^2)``).

The Madelung self-energy ``v_M = lim_{r -> 0} [v_Ew(r) - 1/|r|]`` is the
finite self-interaction of one point charge with its periodic copies in
their neutralising background.  Following the 3D convention of
:mod:`OmegaQMC.observables.ewald` we store
``tables.madelung = (1/2) * v_M`` so that the total Madelung
contribution to a configuration of ``N`` electrons is
``N * tables.madelung``.

For benchmark Bravais lattices at density ``n = 1/(pi r_s^2)``,
Bonsall & Maradudin (1977) give the per-electron crystal energy

* triangular: ``epsilon_M = -1.106103 / r_s``
* square    : ``epsilon_M = -1.100244 / r_s``

both in Hartree.  These are the validation targets for
:func:`compute_madelung_2d_reference`.

References:
    D.E. Parry, Surface Sci. 49, 433 (1975).
    D.M. Heyes, Phys. Rev. B 23, 1755 (1981).
    D.M. Bonsall & A.A. Maradudin, Phys. Rev. B 15, 1959 (1977).
"""

from typing import NamedTuple

import numpy as np
import jax
import jax.numpy as jnp
from jax.scipy.special import erfc


# ---------------------------------------------------------------------
# Precomputed grids
# ---------------------------------------------------------------------

class EwaldTables2D(NamedTuple):
    """Precomputed real- and reciprocal-space grids for 2D Ewald sum.

    Attributes:
        L: Square simulation-cell side length.
        eta: Ewald splitting parameter.
        R_vecs: Real-space lattice vectors excluding the origin,
            shape ``(N_real, 2)``.
        G_vecs: Reciprocal-space vectors excluding ``G=0``, shape
            ``(N_recip, 2)``.
        G_norm: Magnitudes ``|G|`` of ``G_vecs``, shape ``(N_recip,)``.
        recip_weight: ``(2 pi/A) * erfc(G/(2 eta)) / G`` — Fourier
            weight for the reciprocal-space sum.
        bg_const: ``-2 sqrt(pi) / (A * eta)`` — constant ``G -> 0``
            background term remaining after the neutralising-background
            cancellation of the divergent ``1/G`` mode.
        madelung: Per-electron Madelung energy ``(1/2) * v_M`` for the
            simulation-cell Bravais lattice.
    """

    L: float
    eta: float
    R_vecs: jax.Array
    G_vecs: jax.Array
    G_norm: jax.Array
    recip_weight: jax.Array
    bg_const: float
    madelung: float


def _square_lattice_vectors(L: float, nmax: int, include_origin: bool):
    rng = np.arange(-nmax, nmax + 1)
    nx, ny = np.meshgrid(rng, rng, indexing='ij')
    n = np.stack([nx.ravel(), ny.ravel()], axis=1)
    if not include_origin:
        mask = np.any(n != 0, axis=1)
        n = n[mask]
    return (L * n).astype(np.float64)


def _rectangular_lattice_vectors(
    L_x: float, L_y: float, nmax: int, include_origin: bool,
):
    """Same as _square_lattice_vectors but with per-axis scaling."""
    rng = np.arange(-nmax, nmax + 1)
    nx, ny = np.meshgrid(rng, rng, indexing='ij')
    n = np.stack([nx.ravel(), ny.ravel()], axis=1)
    if not include_origin:
        mask = np.any(n != 0, axis=1)
        n = n[mask]
    scaled = n.astype(np.float64) * np.asarray([L_x, L_y])
    return scaled


def build_ewald_2d_tables(
    L,
    *,
    eta: float = None,
    n_real: int = 4,
    n_recip: int = 8,
) -> EwaldTables2D:
    """Precompute 2D Ewald real/reciprocal tables.

    Args:
        L: Cell dimensions.  Scalar → square cell of side ``L``.
            2-tuple/sequence ``(L_x, L_y)`` → rectangular cell.
        eta: Ewald splitting parameter.  Default ``2.8/sqrt(area)`` —
            the 2D analog of the CASINO/FermiNet 3D recommendation
            ``eta = 2.8/V^{1/3}`` (Cassella 2022).
        n_real: Real-space cutoff in lattice-vector units.
        n_recip: Reciprocal-space cutoff in integer-G units.

    Returns:
        :class:`EwaldTables2D`.
    """
    if isinstance(L, (tuple, list, np.ndarray, jnp.ndarray)):
        L_x, L_y = float(L[0]), float(L[1])
    else:
        L_x = L_y = float(L)
    area = L_x * L_y
    L_eff = float(np.sqrt(area))    # for label + default eta
    if eta is None:
        eta = 2.8 / L_eff

    R_vecs = _rectangular_lattice_vectors(
        L_x, L_y, n_real, include_origin=False,
    )
    G_ints = _square_lattice_vectors(1.0, n_recip, include_origin=False)
    G_vecs = 2.0 * np.pi * G_ints / np.asarray([L_x, L_y])
    G_norm = np.sqrt(np.sum(G_vecs ** 2, axis=-1))

    from scipy.special import erfc as _erfc_np
    recip_weight = (2.0 * np.pi / area) * _erfc_np(G_norm / (2.0 * eta)) / G_norm
    bg_const = -2.0 * np.sqrt(np.pi) / (area * eta)

    # Madelung constant: v_M = lim_{r->0} [v_Ew(r) - 1/r]
    #                       = sum_{R != 0} erfc(eta|R|)/|R|
    #                       + (2 pi/A) sum_{G != 0} erfc(G/(2 eta))/G
    #                       - 2 eta / sqrt(pi)        [self-energy]
    #                       + bg_const,                [G -> 0 background]
    # weighted by 1/2 to express it as a per-electron energy (so the
    # total Madelung correction to a configuration of N electrons is
    # N * tables.madelung — same convention as 3D ewald.py).
    R_norm = np.linalg.norm(R_vecs, axis=-1)
    e_real = float(np.sum(_erfc_np(eta * R_norm) / R_norm))
    e_recip = float(np.sum(recip_weight))
    e_self = -2.0 * eta / np.sqrt(np.pi)
    e_bg = bg_const
    madelung = 0.5 * (e_real + e_recip + e_self + e_bg)

    return EwaldTables2D(
        L=L_eff,
        eta=float(eta),
        R_vecs=jnp.asarray(R_vecs),
        G_vecs=jnp.asarray(G_vecs),
        G_norm=jnp.asarray(G_norm),
        recip_weight=jnp.asarray(recip_weight),
        bg_const=float(bg_const),
        madelung=float(madelung),
    )


# ---------------------------------------------------------------------
# Pair potential
# ---------------------------------------------------------------------

def ewald_2d_pair_potential(
    diff: jax.Array, tables: EwaldTables2D,
) -> jax.Array:
    """Periodic 2D Ewald pair potential ``v_Ew(r_i - r_j)``.

    Args:
        diff: Pairwise Cartesian differences ``(..., 2)``.
        tables: Precomputed :class:`EwaldTables2D`.

    Returns:
        Scalar pair energy per input diff, shape ``(...,)``.
    """
    eta = tables.eta

    # Real-space contribution: sum over lattice images R != 0.
    shifted = diff[..., None, :] + tables.R_vecs  # (..., N_real, 2)
    r = jnp.sqrt(jnp.sum(shifted ** 2, axis=-1) + 1e-300)
    real_sum = jnp.sum(erfc(eta * r) / r, axis=-1)

    # Origin image (R=0).  Use the same machine-eps regularization as
    # 3D ewald.py for cusp consistency: the 1/r regularization in the
    # short-range piece must match the regularization the wavefunction
    # cusp uses for its Laplacian, so the divergences cancel at
    # coincidence (Kato cancellation).
    eps = jnp.finfo(diff.dtype).eps
    r0 = jnp.sqrt(jnp.sum(diff ** 2, axis=-1) + eps)
    real_origin = erfc(eta * r0) / r0
    real = real_sum + real_origin

    # Reciprocal-space contribution.
    Gr = diff @ tables.G_vecs.T  # (..., N_recip)
    recip = jnp.sum(tables.recip_weight * jnp.cos(Gr), axis=-1)

    return real + recip + tables.bg_const


def ewald_2d_pair_energy(
    r: jax.Array, tables: EwaldTables2D,
) -> jax.Array:
    """Electron-electron Coulomb energy via 2D Ewald.

    Args:
        r: Electron positions ``(n_elec, 2)``.
        tables: Precomputed :class:`EwaldTables2D`.

    Returns:
        Scalar energy including the ``N * v_M`` background.
    """
    n_elec = r.shape[-2]
    if n_elec < 2:
        return jnp.asarray(n_elec * tables.madelung)
    i, j = jnp.triu_indices(n_elec, k=1)
    diff = r[..., i, :] - r[..., j, :]
    pair = ewald_2d_pair_potential(diff, tables)
    return jnp.sum(pair, axis=-1) + n_elec * tables.madelung


# ---------------------------------------------------------------------
# Analytic Madelung references for benchmark Bravais lattices
# ---------------------------------------------------------------------

def compute_madelung_2d_reference(
    rs: float, lattice: str = 'square',
) -> float:
    """Per-electron Madelung energy for the infinite 2D Wigner crystal.

    From Bonsall & Maradudin (1977) at density ``n = 1/(pi r_s^2)``:

    * triangular Bravais lattice: ``epsilon_M = -1.106103 / r_s``
    * square Bravais lattice    : ``epsilon_M = -1.100244 / r_s``

    Both in Hartree per electron.  These are *physical* per-electron
    Madelung energies — directly comparable to ``tables.madelung`` from
    a single-electron-per-cell setup with the corresponding cell shape.

    Args:
        rs: Wigner-Seitz radius (Bohr radii).
        lattice: One of ``'square'`` or ``'triangular'``.

    Returns:
        Madelung energy per electron in Hartree.
    """
    coefficients = {
        'square': 1.100244,
        'triangular': 1.106103,
    }
    if lattice not in coefficients:
        raise ValueError(
            f"Unknown 2D Bravais lattice '{lattice}'. "
            f"Valid: {list(coefficients.keys())}.",
        )
    return -coefficients[lattice] / rs
