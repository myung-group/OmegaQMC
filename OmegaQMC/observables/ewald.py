"""3D Ewald summation for Coulomb interactions in a periodic cell.

Computes the lattice-summed pair potential and the Madelung
self-energy needed for HEG Monte Carlo energies:

.. math::

    V_{\\text{e-e}}(\\{r_i\\}) = \\tfrac{1}{2}
        \\sum_{i\\neq j} v_{\\text{Ew}}(r_i - r_j)
        + \\tfrac{N}{2}\\, v_M.

``v_{Ew}(r)`` splits the bare ``1/|r|`` Coulomb interaction into a
short-ranged real-space part that falls off as ``erfc(η r)/r`` and
a long-ranged reciprocal-space part that converges rapidly once
``G²`` exceeds ``4 η²``:

.. math::

    v_{\\text{Ew}}(\\mathbf{r}) = \\sum_{\\mathbf{R}}
        \\frac{\\mathrm{erfc}(\\eta |\\mathbf{r}+\\mathbf{R}|)}
             {|\\mathbf{r}+\\mathbf{R}|}
      + \\frac{4\\pi}{V} \\sum_{\\mathbf{G}\\neq 0}
             \\frac{e^{-G^2/(4\\eta^2)}}{G^2}
             \\cos(\\mathbf{G}\\cdot\\mathbf{r})
      - \\frac{\\pi}{V\\eta^2}.

The Madelung constant ``v_M = lim_{r→0} [v_Ew(r) - 1/|r|]`` is the
finite self-interaction of a point charge with its own neutralising
background.  We reuse :func:`~OmegaQMC.afqmc_3deg.compute_madelung_3d`
where possible; a standalone implementation is included here so the
HEG VMC driver has no circular imports.
"""

from typing import NamedTuple

import numpy as np
import jax
import jax.numpy as jnp
from jax.scipy.special import erfc


# ---------------------------------------------------------------------
# Precomputed grids
# ---------------------------------------------------------------------

class EwaldTables(NamedTuple):
    """Precomputed real- and reciprocal-space grids for 3D Ewald sum.

    Attributes:
        L: Cubic simulation-cell side length.
        eta: Ewald splitting parameter.
        R_vecs: Real-space lattice vectors excluding the origin,
            shape ``(N_real, 3)``.
        G_vecs: Reciprocal-space vectors excluding ``G=0``, shape
            ``(N_recip, 3)``.
        G_sq: Squared magnitudes of ``G_vecs``.
        recip_weight: ``exp(-G²/(4η²)) / G²``, the Fourier weight
            for the reciprocal-space sum.
        bg_const: ``-π / (V η²)``, the constant background term.
        madelung: Madelung energy per electron for the cell.
    """

    L: float
    eta: float
    R_vecs: jax.Array
    G_vecs: jax.Array
    G_sq: jax.Array
    recip_weight: jax.Array
    bg_const: float
    madelung: float


def _cubic_lattice_vectors(L: float, nmax: int, include_origin: bool):
    rng = np.arange(-nmax, nmax + 1)
    nx, ny, nz = np.meshgrid(rng, rng, rng, indexing='ij')
    n = np.stack([nx.ravel(), ny.ravel(), nz.ravel()], axis=1)
    if not include_origin:
        mask = np.any(n != 0, axis=1)
        n = n[mask]
    return (L * n).astype(np.float64)


def build_ewald_tables(
    L: float,
    *,
    eta: float = None,
    n_real: int = 3,
    n_recip: int = 6,
) -> EwaldTables:
    """Precompute Ewald real/reciprocal tables for a cubic cell.

    Args:
        L: Cubic simulation-cell side length.
        eta: Ewald splitting parameter ``η`` such that
            ``γ = η² = (2.8 / V^{1/3})²`` by default — the CASINO /
            FermiNet recommendation (Cassella et al. 2022 §B.1,
            CASINO manual, FermiNet ``pbc/hamiltonian.py``).  This
            is ~1.58× the previous ``√π / L`` default, giving a
            narrower real-space Gaussian and faster real-space
            convergence.  With the asymmetric default cutoffs
            ``(n_real=3, n_recip=6)`` both sums are well converged
            at N=14, rs=2–10.  Passing ``eta`` explicitly overrides
            this default.
        n_real: Real-space cutoff in lattice-vector units
            (shells with ``|n_i| ≤ n_real`` are summed).
        n_recip: Reciprocal-space cutoff in integer-G units.

    Returns:
        :class:`EwaldTables`.
    """
    if eta is None:
        # CASINO / FermiNet γ = (2.8 / V^{1/3})² → η = 2.8 / V^{1/3}.
        # For a cubic cell V^{1/3} = L exactly.
        eta = 2.8 / L
    volume = L ** 3

    R_vecs = _cubic_lattice_vectors(L, n_real, include_origin=False)
    G_ints = _cubic_lattice_vectors(1.0, n_recip, include_origin=False)
    G_vecs = 2.0 * np.pi * G_ints / L
    G_sq = np.sum(G_vecs ** 2, axis=-1)
    recip_weight = (4.0 * np.pi / volume) * np.exp(
        -G_sq / (4.0 * eta ** 2),
    ) / G_sq

    # Madelung constant: v_M = Σ_{R≠0} erfc(η|R|)/|R|
    #                         + (4π/V) Σ_{G≠0} exp(-G²/(4η²))/G²
    #                         - π/(V η²) - 2η/√π,
    # weighted by 1/2 to express it as an energy per electron.
    from scipy.special import erfc as _erfc_np
    R_norm = np.linalg.norm(R_vecs, axis=-1)
    e_real = float(np.sum(_erfc_np(eta * R_norm) / R_norm))
    e_recip = float(np.sum(recip_weight))
    e_self = -2.0 * eta / np.sqrt(np.pi)
    e_bg = -np.pi / (volume * eta ** 2)
    madelung = 0.5 * (e_real + e_recip + e_self + e_bg)

    return EwaldTables(
        L=float(L),
        eta=float(eta),
        R_vecs=jnp.asarray(R_vecs),
        G_vecs=jnp.asarray(G_vecs),
        G_sq=jnp.asarray(G_sq),
        recip_weight=jnp.asarray(recip_weight),
        bg_const=float(-np.pi / (volume * eta ** 2)),
        madelung=float(madelung),
    )


# ---------------------------------------------------------------------
# Pair potential
# ---------------------------------------------------------------------

def ewald_pair_potential(
    diff: jax.Array, tables: EwaldTables,
) -> jax.Array:
    """Periodic Ewald pair potential ``v_Ew(r_i - r_j)``.

    Args:
        diff: Pairwise Cartesian differences ``(..., 3)``.
        tables: Precomputed :class:`EwaldTables`.

    Returns:
        Scalar pair energy per input diff, shape ``(...,)``.
    """
    eta = tables.eta

    # Real-space contribution: sum over lattice images R.
    shifted = diff[..., None, :] + tables.R_vecs  # (..., N_real, 3)
    r = jnp.sqrt(jnp.sum(shifted ** 2, axis=-1) + 1e-300)
    real_sum = jnp.sum(erfc(eta * r) / r, axis=-1)
    # Origin image (R=0).  Apply the same machine-eps floor that
    # :func:`~OmegaQMC.psi.nn.heg_psiformer._pair_distances_mi_full`
    # uses for the Kato cusp.  This keeps the 1/r regularization
    # *consistent* between the Coulomb potential and the cusp's
    # Laplacian so they cancel at coincidence — the Kato
    # cancellation that makes the local energy finite.
    # Previously ``r0`` had no eps floor and the coincidence term
    # was masked to 0 via ``jnp.where(r0>0, ...)``, while the
    # cusp still saw a regularized ``sqrt(eps)`` distance.  That
    # mismatch produced a spurious ~−1/√eps kinetic contribution
    # at near-coincident walkers and showed up as ``⟨E⟩ < E_DMC``
    # (variationally invalid) whenever the cusp was enabled.
    eps = jnp.finfo(diff.dtype).eps
    r0 = jnp.sqrt(jnp.sum(diff ** 2, axis=-1) + eps)
    real_origin = erfc(eta * r0) / r0
    real = real_sum + real_origin

    # Reciprocal-space contribution.
    Gr = diff @ tables.G_vecs.T  # (..., N_recip)
    recip = jnp.sum(tables.recip_weight * jnp.cos(Gr), axis=-1)

    return real + recip + tables.bg_const


def ewald_pair_energy(
    r: jax.Array, tables: EwaldTables,
) -> jax.Array:
    """Electron-electron Coulomb energy via Ewald.

    The per-electron Madelung convention of
    :func:`~OmegaQMC.afqmc_3deg.compute_madelung_3d` is used: the
    ``tables.madelung`` value already absorbs the ``1/2`` factor
    from the standard Fraser/Foulkes/Rajagopal formula, so the
    total Madelung contribution is ``N · v_M`` (one copy per
    electron for the self-image energy in the neutralising
    background).

    Args:
        r: Electron positions ``(n_elec, 3)``.
        tables: Precomputed :class:`EwaldTables`.

    Returns:
        Scalar energy including the ``N · v_M`` background.
    """
    n_elec = r.shape[-2]
    if n_elec < 2:
        return jnp.asarray(n_elec * tables.madelung)
    i, j = jnp.triu_indices(n_elec, k=1)
    diff = r[..., i, :] - r[..., j, :]
    pair = ewald_pair_potential(diff, tables)
    return jnp.sum(pair, axis=-1) + n_elec * tables.madelung
