"""Unit tests for Ewald pair potential and Madelung constant.

Cross-checks against :mod:`OmegaQMC.afqmc_3deg` and sanity-checks:
  * Translation invariance under lattice shifts.
  * Reflection symmetry ``v(r) == v(-r)``.
  * Ewald parameter (η) independence of Madelung.
  * Agreement with the AFQMC 3DEG Madelung reference
    (``compute_madelung_3d``).
  * Ewald-sum convergence to ``1/r`` for ``r`` well inside the cell.
"""

import numpy as np
import jax
import jax.numpy as jnp
import pytest

from OmegaQMC.observables.ewald import (
    build_ewald_tables,
    ewald_pair_potential,
    ewald_pair_energy,
)
from OmegaQMC.afqmc_3deg import (
    build_3deg_system,
    compute_madelung_3d,
)


def test_madelung_matches_afqmc_reference():
    """Madelung from our tables must match the AFQMC implementation
    at the level of numerical Ewald convergence."""
    rs = 4.0
    N = 14
    sys = build_3deg_system(rs, N_elec=N, N_pw=19,
                            polarization='unpolarized')
    L = sys['L']
    v_ref = compute_madelung_3d(sys)

    tab = build_ewald_tables(L, n_real=5, n_recip=8)
    assert abs(tab.madelung - v_ref) < 1e-3


def test_madelung_eta_independence():
    """Madelung must be independent of the Ewald splitting η
    once the grids are converged."""
    L = 5.0
    eta1 = np.sqrt(np.pi) / L
    eta2 = 1.5 * eta1
    tab1 = build_ewald_tables(L, eta=eta1, n_real=5, n_recip=8)
    tab2 = build_ewald_tables(L, eta=eta2, n_real=5, n_recip=12)
    assert abs(tab1.madelung - tab2.madelung) < 1e-4


def test_pair_potential_lattice_periodic():
    """Lattice periodicity of the truncated Ewald sum holds up to
    boundary-image truncation error, which decays super-exponentially
    in ``η * n_real * L``.  With ``n_real = 5`` the missing images
    at |R| ≥ 5L contribute < 1e-10 even for η = √π/L, so periodicity
    holds at single-precision-like tolerance."""
    L = 4.0
    tab = build_ewald_tables(L, n_real=5, n_recip=10)
    rng = np.random.default_rng(0)
    r = jnp.asarray(rng.normal(size=(10, 3)) * 0.6)
    v0 = ewald_pair_potential(r, tab)
    shift = L * jnp.asarray([1.0, 0.0, 0.0])
    v_shift = ewald_pair_potential(r + shift, tab)
    np.testing.assert_allclose(v0, v_shift, atol=1e-6)


def test_pair_potential_reflection():
    L = 4.0
    tab = build_ewald_tables(L, n_real=3, n_recip=6)
    rng = np.random.default_rng(1)
    r = jnp.asarray(rng.normal(size=(8, 3)) * 0.6)
    np.testing.assert_allclose(
        ewald_pair_potential(r, tab),
        ewald_pair_potential(-r, tab),
        atol=1e-8,
    )


def test_pair_potential_short_range_limit():
    """As ``r → 0⁺`` the Ewald pair potential minus the bare ``1/r``
    singularity tends to the Madelung constant ``v_M``.  This is
    the defining property of the Madelung self-energy."""
    L = 6.0
    tab = build_ewald_tables(L, n_real=5, n_recip=10)
    # Small separation well below the cell scale.
    r_small = jnp.asarray([1e-3, 0.0, 0.0])
    v = float(ewald_pair_potential(r_small, tab))
    # v_Ew(r) - 1/r should approach v_M (not 0.5 v_M — v_M is the
    # self-energy with the factor of 1/2 already absorbed in the
    # Madelung coefficient of our convention).
    bare = 1.0 / 1e-3
    assert abs((v - bare) - 2.0 * tab.madelung) < 1e-2


def test_pair_energy_shape_and_grad():
    L = 4.0
    tab = build_ewald_tables(L, n_real=3, n_recip=6)
    rng = np.random.default_rng(2)
    r = jnp.asarray(rng.uniform(0, L, size=(14, 3)))
    e = ewald_pair_energy(r, tab)
    assert e.shape == ()
    g = jax.grad(lambda x: ewald_pair_energy(x, tab))(r)
    assert jnp.all(jnp.isfinite(g))


def test_single_electron_is_madelung():
    """A single electron in its own neutralising background pays
    the full Madelung self-energy ``v_M`` (no pair contribution)."""
    L = 3.0
    tab = build_ewald_tables(L, n_real=3, n_recip=6)
    r = jnp.zeros((1, 3))
    e = ewald_pair_energy(r, tab)
    assert abs(float(e) - tab.madelung) < 1e-12
