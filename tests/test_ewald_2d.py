"""Unit tests for 2D Ewald pair potential and Madelung constant.

Validates the Parry/Heyes 2D Ewald implementation against:

* Bonsall-Maradudin (1977) analytic Madelung constants for the
  square 2D Wigner crystal.
* Independence of the Ewald splitting parameter eta.
* Periodicity, reflection symmetry, and convergence to the bare
  ``1/r`` Coulomb at separations small compared to the cell.
"""

import numpy as np
import jax
import jax.numpy as jnp
import pytest

from OmegaQMC.observables.ewald_2d import (
    build_ewald_2d_tables,
    compute_madelung_2d_reference,
    ewald_2d_pair_energy,
    ewald_2d_pair_potential,
)


# ---------------------------------------------------------------------
# Madelung
# ---------------------------------------------------------------------

@pytest.mark.parametrize('L', [1.0, 2.0, 4.0, 7.5, 12.3])
def test_madelung_matches_bonsall_maradudin_square(L):
    """Single charge in square cell of side L, density n=1/L^2,
    rs = L/sqrt(pi).  Bonsall-Maradudin: epsilon_M = -1.100244 / rs.
    Must match the Ewald tables to ~1e-6."""
    tab = build_ewald_2d_tables(L, n_real=6, n_recip=12)
    rs = L / np.sqrt(np.pi)
    expected = compute_madelung_2d_reference(rs, 'square')
    assert abs(tab.madelung - expected) < 1e-5


def test_madelung_eta_independence():
    """Madelung must be independent of the Ewald splitting eta once
    both real- and reciprocal-space sums are converged."""
    L = 4.0
    madelungs = []
    for eta_factor in [0.7, 1.0, 1.3, 1.6]:
        eta = eta_factor * 2.8 / L
        tab = build_ewald_2d_tables(
            L, eta=eta, n_real=8, n_recip=16,
        )
        madelungs.append(tab.madelung)
    spread = max(madelungs) - min(madelungs)
    assert spread < 1e-8


def test_madelung_scales_as_inverse_L():
    """For square cells: epsilon_M(L) = -1.100244 * sqrt(pi) / L.
    Verify the 1/L scaling at the level of single-precision tolerance."""
    Ls = np.array([1.0, 2.0, 5.0, 10.0])
    madelungs = np.array([
        build_ewald_2d_tables(L, n_real=6, n_recip=12).madelung
        for L in Ls
    ])
    expected = -1.100244 * np.sqrt(np.pi) / Ls
    np.testing.assert_allclose(madelungs, expected, atol=1e-5)


# ---------------------------------------------------------------------
# Pair-potential symmetries
# ---------------------------------------------------------------------

def test_pair_potential_lattice_periodic():
    """Periodicity ``v(r + L hat_x) == v(r)`` to ~1e-6 with the
    default cutoffs."""
    L = 4.0
    tab = build_ewald_2d_tables(L, n_real=5, n_recip=12)
    rng = np.random.default_rng(0)
    r = jnp.asarray(rng.normal(size=(10, 2)) * 0.6)
    v0 = ewald_2d_pair_potential(r, tab)
    shift = L * jnp.asarray([1.0, 0.0])
    v_shift = ewald_2d_pair_potential(r + shift, tab)
    np.testing.assert_allclose(v0, v_shift, atol=1e-6)


def test_pair_potential_reflection():
    """``v(r) == v(-r)`` (Coulomb is reflection-symmetric)."""
    L = 4.0
    tab = build_ewald_2d_tables(L, n_real=4, n_recip=8)
    rng = np.random.default_rng(1)
    r = jnp.asarray(rng.normal(size=(8, 2)) * 0.6)
    np.testing.assert_allclose(
        ewald_2d_pair_potential(r, tab),
        ewald_2d_pair_potential(-r, tab),
        atol=1e-10,
    )


def test_pair_potential_short_range_limit():
    """At separations ``r << L`` the leading singularity ``1/r``
    is recovered; ``v(r) - 1/r`` approaches the *full* Madelung
    constant ``v_M = 2 * tables.madelung``."""
    L = 6.0
    tab = build_ewald_2d_tables(L, n_real=6, n_recip=12)
    r_small = jnp.asarray([1e-3, 0.0])
    v = float(ewald_2d_pair_potential(r_small, tab))
    bare = 1.0 / 1e-3
    # Note the factor of 2: tables.madelung = (1/2) v_M.
    assert abs((v - bare) - 2.0 * tab.madelung) < 1e-2


# ---------------------------------------------------------------------
# Pair energy
# ---------------------------------------------------------------------

def test_pair_energy_shape_and_grad():
    """Energy is a scalar and is differentiable wrt positions."""
    L = 4.0
    tab = build_ewald_2d_tables(L, n_real=4, n_recip=8)
    rng = np.random.default_rng(2)
    r = jnp.asarray(rng.uniform(0, L, size=(14, 2)))
    e = ewald_2d_pair_energy(r, tab)
    assert e.shape == ()
    g = jax.grad(lambda x: ewald_2d_pair_energy(x, tab))(r)
    assert jnp.all(jnp.isfinite(g))


def test_single_electron_is_madelung():
    """A single electron in its neutralising background pays the
    full per-electron Madelung self-energy, no pair contribution."""
    L = 3.0
    tab = build_ewald_2d_tables(L, n_real=4, n_recip=8)
    r = jnp.zeros((1, 2))
    e = ewald_2d_pair_energy(r, tab)
    assert abs(float(e) - tab.madelung) < 1e-12


def test_two_electron_perfect_lattice_recovers_madelung():
    """Two electrons placed at the two perfectly-symmetric WC sites
    of a square cell ((0,0) and (L/2, L/2)) form a square WC at
    half the cell density.  Per-electron energy must equal
    Bonsall-Maradudin epsilon_M for that effective rs.

    The 2-electron-per-cell square superlattice has primitive cell
    of area L^2/2, so the effective per-electron rs satisfies
    pi rs^2 = L^2/2, i.e. rs = L / sqrt(2 pi).  The supercell
    Ewald energy includes both electrons' self-Madelung in their
    shared neutralising background — this should equal
    2 * epsilon_M_BM(rs_eff)."""
    L = 4.0
    tab = build_ewald_2d_tables(L, n_real=8, n_recip=16)
    r = jnp.asarray([[0.0, 0.0], [L / 2, L / 2]])
    e = float(ewald_2d_pair_energy(r, tab))
    rs_eff = L / np.sqrt(2.0 * np.pi)
    e_per_elec_expected = compute_madelung_2d_reference(rs_eff, 'square')
    e_expected = 2.0 * e_per_elec_expected
    # Slightly looser tolerance: this also tests the pair-potential
    # at finite separation, not just the Madelung table.
    assert abs(e - e_expected) < 1e-4


# ---------------------------------------------------------------------
# Cross-check against direct lattice sum (small-cell sanity)
# ---------------------------------------------------------------------

def _direct_lattice_madelung(L: float, R_cutoff_factor: float = 80.0):
    """Brute-force per-electron Madelung for square 2D lattice.

    Uses the regularised lattice sum
        epsilon_M = (1/2) lim_{R_max -> inf} [ sum_{0 < |R| < R_max} 1/|R|
                                               - 2 pi n R_max ]
    with n = 1/L^2 and circular cutoff radius R_max = R_cutoff_factor * L.
    Convergence is slow (~ 1/R_max) so this is only useful as an
    order-of-magnitude check."""
    R_max = R_cutoff_factor * L
    n_max = int(np.ceil(R_max / L)) + 1
    rng = np.arange(-n_max, n_max + 1)
    nx, ny = np.meshgrid(rng, rng, indexing='ij')
    R = L * np.stack([nx.ravel(), ny.ravel()], axis=1)
    R_norm = np.linalg.norm(R, axis=-1)
    mask = (R_norm > 1e-12) & (R_norm < R_max)
    direct_sum = float(np.sum(1.0 / R_norm[mask]))
    bg = 2.0 * np.pi * (1.0 / L ** 2) * R_max
    return 0.5 * (direct_sum - bg)


def test_madelung_against_direct_lattice_sum():
    """Order-of-magnitude cross-check: a brute-force lattice sum
    over a circular truncation, with a leading uniform-background
    subtraction, agrees with the Ewald-summed Madelung at the level
    of the residual circular-truncation shape factor.  Convergence
    is intrinsically slow in 2D — this test only verifies the
    Ewald result is within ~5% of the brute-force answer."""
    L = 1.0
    tab = build_ewald_2d_tables(L, n_real=6, n_recip=12)
    direct = _direct_lattice_madelung(L, R_cutoff_factor=400.0)
    rel_err = abs(tab.madelung - direct) / abs(tab.madelung)
    assert rel_err < 0.05


# ---------------------------------------------------------------------
# Wigner-Seitz scaling
# ---------------------------------------------------------------------

@pytest.mark.parametrize('rs', [1.0, 2.0, 5.0, 10.0, 30.0])
def test_madelung_matches_bm_at_physical_rs(rs):
    """Build the square Ewald cell at physical rs (one electron per
    cell, density n=1/(pi rs^2), so L = rs * sqrt(pi)) and verify
    the resulting Madelung matches BM's -1.100244/rs to ~1e-6."""
    L = rs * np.sqrt(np.pi)
    tab = build_ewald_2d_tables(L, n_real=6, n_recip=12)
    expected = compute_madelung_2d_reference(rs, 'square')
    assert abs(tab.madelung - expected) < 1e-5
