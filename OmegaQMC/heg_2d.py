"""2D HEG system builder, Hartree-Fock reference, and Madelung utilities.

The 2D analog of the relevant parts of :mod:`OmegaQMC.afqmc_3deg`.  The
2D HEG lives on a square simulation cell of side
``L = sqrt(pi * N) * r_s`` (from the density relation
``pi * r_s^2 = A / N``), with electrons on a discrete plane-wave grid
``k = 2 pi n / L``, ``n in Z^2``.

Standard closed-shell counts per spin, filling shells of ``|n|^2``:

    1, 5, 9, 13, 21, 25, 29, 37, 45, 57, 61, 69, 81, 89, ...

Standard total ``N`` for unpolarized closed shell (twice the per-spin
count): 2, 10, 18, 26, 42, 50, 58, 74, 90, 114, ...

The standard benchmark size for direct comparison with Attaccalite et al.
(PRL 88, 256601, 2002) is ``N = 58`` unpolarized.

Hartree-Fock thermodynamic-limit closed forms (Stern 1973).  The
average kinetic energy of a 2D Fermi sea is ``<k^2/2>_FS = k_F^2/4``
(half of the Fermi energy ``e_F = k_F^2/2``):

* unpolarized: ``epsilon_HF / N = 1/(2 r_s^2) - 4 sqrt(2) / (3 pi r_s)``
* polarized  : ``epsilon_HF / N = 1/r_s^2     - 8 / (3 pi r_s)``

both in Hartree per electron.  Finite-cell HF energies include a
Madelung self-image correction and approach the TD limit with
``O(1/sqrt(N))`` finite-size corrections.
"""

from typing import Optional

import numpy as np

from OmegaQMC.observables.ewald_2d import (
    EwaldTables2D,
    build_ewald_2d_tables,
)


# ---------------------------------------------------------------------
# Closed-shell k-grid for 2D Fermi sea
# ---------------------------------------------------------------------

# Per-spin closed-shell counts in 2D, filling shells of |n|^2.
CLOSED_SHELL_PER_SPIN_2D = (
    1, 5, 9, 13, 21, 25, 29, 37, 45, 57,
    61, 69, 81, 89, 97, 101, 109, 113, 121,
)


def closed_shell_n_per_spin(n_target: int) -> int:
    """Smallest tabulated closed-shell per-spin count ``>= n_target``."""
    for n in CLOSED_SHELL_PER_SPIN_2D:
        if n >= n_target:
            return n
    raise ValueError(
        f"n_target={n_target} exceeds tabulated closed shells "
        f"{CLOSED_SHELL_PER_SPIN_2D}",
    )


def is_closed_shell_n_per_spin(n: int) -> bool:
    return n in CLOSED_SHELL_PER_SPIN_2D


def generate_2d_fermi_sea(n_per_spin: int):
    """Return the closed-shell 2D Fermi-sea integer k-grid.

    Args:
        n_per_spin: Per-spin electron count; must be a closed-shell value.

    Returns:
        ``(n_per_spin, 2)`` integer array of (n_x, n_y), sorted by
        ``|n|^2`` then lexicographically — these are the integer
        k-vector indices that fill the Fermi sea.
    """
    if not is_closed_shell_n_per_spin(n_per_spin):
        raise ValueError(
            f"n_per_spin={n_per_spin} is not a closed-shell count. "
            f"Valid: {CLOSED_SHELL_PER_SPIN_2D}",
        )
    nmax = int(np.ceil(np.sqrt(n_per_spin))) + 4
    rng = np.arange(-nmax, nmax + 1)
    nx, ny = np.meshgrid(rng, rng, indexing='ij')
    grid = np.stack([nx.ravel(), ny.ravel()], axis=1)
    n_sq = np.sum(grid ** 2, axis=1)
    order = np.lexsort((grid[:, 1], grid[:, 0], n_sq))
    grid_sorted = grid[order]
    return grid_sorted[:n_per_spin]


# ---------------------------------------------------------------------
# 2D HEG system builder
# ---------------------------------------------------------------------

def build_2deg_system(
    rs: float,
    N_elec: int,
    polarization: str = 'unpolarized',
    kappa: Optional[np.ndarray] = None,
) -> dict:
    """Build a 2D HEG system at density ``n = 1/(pi r_s^2)``.

    The square simulation cell has side
    ``L = sqrt(pi * N) * r_s`` (from ``pi r_s^2 = A/N``).

    Args:
        rs: Wigner-Seitz radius (Bohr).
        N_elec: Total electron count.  Must form a closed shell in
            the chosen polarization (i.e., per-spin count must be in
            :data:`CLOSED_SHELL_PER_SPIN_2D`).
        polarization: ``'unpolarized'`` (``N_up = N_dn = N/2``) or
            ``'polarized'`` (``N_up = N``, ``N_dn = 0``).
        kappa: Twist-angle in fractional coordinates ``[-1/2, 1/2)^2``
            shifting all k-vectors as ``k = (n + kappa) * dk``.
            Default ``None`` (Gamma-point).

    Returns:
        Dict containing ``rs``, ``N_elec``, ``nup``, ``ndown``, ``L``,
        ``area``, ``dk``, ``kvecs`` (per-spin Fermi sea, shape
        ``(nup, 2)``), ``grid_int``, ``kappa``, ``polarization``,
        ``dim``.
    """
    if polarization == 'unpolarized':
        if N_elec % 2 != 0:
            raise ValueError("N_elec must be even for unpolarized")
        nup = N_elec // 2
        ndown = N_elec // 2
    elif polarization == 'polarized':
        nup = N_elec
        ndown = 0
    else:
        raise ValueError(f"Unknown polarization: {polarization}")
    if not is_closed_shell_n_per_spin(nup):
        raise ValueError(
            f"Per-spin count nup={nup} is not closed-shell. "
            f"Valid: {CLOSED_SHELL_PER_SPIN_2D}",
        )

    L = np.sqrt(np.pi * N_elec) * rs
    area = L ** 2
    dk = 2.0 * np.pi / L

    grid_int = generate_2d_fermi_sea(nup)
    if kappa is not None:
        kappa_arr = np.asarray(kappa, dtype=np.float64)
        if kappa_arr.shape != (2,):
            raise ValueError(f"kappa must have shape (2,), got {kappa_arr.shape}")
        kvecs = (grid_int.astype(np.float64) + kappa_arr[None, :]) * dk
    else:
        kappa_arr = None
        kvecs = grid_int.astype(np.float64) * dk

    return {
        'rs': float(rs),
        'N_elec': int(N_elec),
        'nup': int(nup),
        'ndown': int(ndown),
        'L': float(L),
        'area': float(area),
        'dk': float(dk),
        'kvecs': kvecs,
        'grid_int': grid_int,
        'kappa': kappa_arr,
        'polarization': polarization,
        'dim': 2,
    }


# ---------------------------------------------------------------------
# Hartree-Fock reference: thermodynamic limit (closed-form)
# ---------------------------------------------------------------------

def hf_kinetic_energy_2d_td(
    rs: float, polarization: str = 'unpolarized',
) -> float:
    """2D HEG kinetic energy per electron in the thermodynamic limit.

    The average kinetic energy of a 2D Fermi sea is ``<k^2/2>_FS =
    k_F^2 / 4`` per electron in each spin channel (half the Fermi
    energy ``e_F = k_F^2/2``), giving

    * unpolarized: ``k_F = sqrt(2)/r_s``  ⇒  ``T/N = 1/(2 r_s^2)``
    * polarized  : ``k_F = 2/r_s``        ⇒  ``T/N = 1/r_s^2``
    """
    if polarization == 'unpolarized':
        return 0.5 / rs ** 2
    if polarization == 'polarized':
        return 1.0 / rs ** 2
    raise ValueError(f"Unknown polarization: {polarization}")


def hf_exchange_energy_2d_td(
    rs: float, polarization: str = 'unpolarized',
) -> float:
    """2D HEG Hartree-Fock exchange per electron in the TD limit.

    Per single-spin Fermi sea: ``epsilon_x_sigma = -(4/(3 pi)) k_F``
    (Hartree).  Population-weighted total (Stern 1973):

    * unpolarized: ``epsilon_x = -4 sqrt(2) / (3 pi r_s) ~ -0.6004/r_s``
    * polarized  : ``epsilon_x = -8 / (3 pi r_s) ~ -0.8488/r_s``
    """
    if polarization == 'unpolarized':
        return -4.0 * np.sqrt(2.0) / (3.0 * np.pi * rs)
    if polarization == 'polarized':
        return -8.0 / (3.0 * np.pi * rs)
    raise ValueError(f"Unknown polarization: {polarization}")


def hf_energy_2d_td(
    rs: float, polarization: str = 'unpolarized',
) -> float:
    """2D HEG Hartree-Fock energy per electron in the TD limit."""
    return (
        hf_kinetic_energy_2d_td(rs, polarization)
        + hf_exchange_energy_2d_td(rs, polarization)
    )


# ---------------------------------------------------------------------
# Hartree-Fock reference: finite N (the baseline our PsiFormer must match)
# ---------------------------------------------------------------------

def generate_halton_twists_2d(n_twists: int) -> np.ndarray:
    """Generate quasi-random 2D twist angles from a Halton sequence.

    Used for twist-averaged boundary conditions (TABC) in 2D HEG VMC.
    The Halton low-discrepancy sequence produces twist points that
    integrate the Brillouin zone with O(1/N_tw) error rather than
    the O(1/sqrt(N_tw)) of plain Monte Carlo.

    Args:
        n_twists: Number of 2D twist angles to generate.

    Returns:
        ``(n_twists, 2)`` array of twist angles in ``[-0.5, 0.5)^2``.
    """
    from scipy.stats.qmc import Halton
    sampler = Halton(d=2, scramble=False)
    return sampler.random(n=n_twists) - 0.5


def hf_energy_2d_finite(
    system: dict,
    ewald_tables: Optional[EwaldTables2D] = None,
) -> dict:
    """Finite-cell Hartree-Fock energy of a 2D HEG closed-shell Fermi sea.

    Computed as

        T   = sum_{k_occ, sigma}                           |k|^2 / 2
        V_x = -(2 pi / A) sum_sigma sum_{i<j in sigma}     1 / |k_i - k_j|
        E_M = N * tables.madelung

    The exchange uses the periodic-cell plane-wave Coulomb matrix
    element ``2 pi / (A |q|)`` for ``q != 0`` (the ``q = 0`` mode is
    killed by the neutralising background); the Madelung absorbs each
    electron's interaction with its own periodic images and the
    background.

    Args:
        system: Dict from :func:`build_2deg_system`.
        ewald_tables: Optional precomputed :class:`EwaldTables2D` for
            the Madelung term.  Built fresh from ``system['L']`` if
            ``None``.

    Returns:
        Dict with per-electron energies (Hartree):
        ``kinetic``, ``exchange``, ``madelung``, ``total``.
    """
    L = system['L']
    area = system['area']
    N_elec = system['N_elec']
    nup = system['nup']
    ndown = system['ndown']
    kvecs = system['kvecs']  # (nup, 2)

    if ewald_tables is None:
        ewald_tables = build_ewald_2d_tables(L)

    # Kinetic: each spin contributes sum_{k_occ} |k|^2/2.  In our
    # build_2deg_system, both spins (in unpolarised case) occupy the
    # same Fermi sea, so just multiply by the spin multiplicity.
    T_per_spin = 0.5 * float(np.sum(np.linalg.norm(kvecs, axis=-1) ** 2))
    spin_multiplicity = 1 if ndown == 0 else 2
    T_total = spin_multiplicity * T_per_spin

    # Exchange: Coulomb pair matrix element 2 pi/(A |k_i - k_j|) for
    # same-spin pairs with k_i != k_j (closed-shell Fermi sea has
    # k_i != k_j by construction for i != j).
    def _exchange_per_spin(kv: np.ndarray) -> float:
        n = len(kv)
        if n < 2:
            return 0.0
        i, j = np.triu_indices(n, k=1)
        diff = kv[i] - kv[j]
        q = np.linalg.norm(diff, axis=-1)
        return -float(np.sum(2.0 * np.pi / (area * q)))

    Vx_total = spin_multiplicity * _exchange_per_spin(kvecs)

    # Madelung self-image, one copy per electron.
    E_M_total = N_elec * ewald_tables.madelung

    E_total = T_total + Vx_total + E_M_total

    return {
        'kinetic': T_total / N_elec,
        'exchange': Vx_total / N_elec,
        'madelung': float(ewald_tables.madelung),
        'total': E_total / N_elec,
    }
