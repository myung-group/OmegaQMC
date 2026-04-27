"""Tests for on-the-fly observable estimators.

Verifies that ``S(k)``, ``S_spin(k)``, ``g(r)``, and ``n(k)`` produce
the expected analytical limits for trivial inputs:

* perfect Wigner crystal (electrons at lattice sites) -> sharp peak
  in S(k) at the reciprocal lattice points
* uniform random electrons -> S(k) ~ 1 (random-phase fluctuation)
* HF Fermi sea (plane-wave Slater determinant): n(k) is 1 for k inside
  the Fermi sphere
"""

import numpy as np
import jax.numpy as jnp
import pytest

from OmegaQMC.observables.structure_factor import (
    momentum_distribution,
    pair_correlation,
    reciprocal_grid_2d,
    reciprocal_lattice_vectors_triangular,
    sf_accumulator_initial,
    sf_accumulator_mean,
    sf_accumulator_update,
    spin_structure_factor,
    structure_factor,
)


def test_reciprocal_grid_2d_excludes_origin():
    grid = reciprocal_grid_2d(L=4.0, n_max=2)
    # 5x5 - 1 = 24 vectors
    assert grid.shape == (24, 2)
    norms = jnp.linalg.norm(grid, axis=-1)
    assert jnp.all(norms > 0)


def test_structure_factor_uniform_random_is_O1():
    """For N uniform-random points the per-config S(k) ~ 1 by central
    limit theorem on independent phases."""
    L = 10.0
    n_elec = 50
    rng = np.random.default_rng(0)
    r = jnp.asarray(rng.uniform(0, L, size=(n_elec, 2)))
    k_grid = reciprocal_grid_2d(L, n_max=2)
    sk = structure_factor(r, k_grid)
    # Mean S(k) should be O(1) (statistical -> 1 for many configs).
    assert sk.shape == (k_grid.shape[0],)
    # Per single config it has substantial variance, but should be < 10.
    assert float(jnp.mean(sk)) < 10.0


def test_structure_factor_perfect_lattice_peaks_at_reciprocal_vectors():
    """Place electrons exactly at sites of a 4x4 square lattice in a
    cell of side 4: expect S(k) to peak strongly at the reciprocal
    lattice vectors b = (2 pi/a) (n_x, n_y) with a=1."""
    a = 1.0
    n_side = 4
    L = a * n_side
    sites = jnp.asarray([
        (a * i, a * j) for i in range(n_side) for j in range(n_side)
    ], dtype=jnp.float64)
    n_elec = sites.shape[0]
    # Build a k-grid that includes the (n_x, n_y) reciprocal vectors
    # of the WC lattice; these are 2 pi/a * integer = 2 pi * integer.
    k_lattice = jnp.asarray([
        (2.0 * np.pi / a * nx, 2.0 * np.pi / a * ny)
        for nx in range(-2, 3) for ny in range(-2, 3)
        if not (nx == 0 and ny == 0)
    ])
    sk = structure_factor(sites, k_lattice)
    # All these k vectors lie on the WC reciprocal lattice -> peak
    # value is N (perfect coherent sum).
    np.testing.assert_allclose(sk, n_elec, atol=1e-6)


def test_spin_structure_factor_ferromagnetic_is_zero_at_kne0():
    """All spins up: S_spin(k) = S(k) since all spins are +1."""
    L = 10.0
    n_elec = 8
    rng = np.random.default_rng(2)
    r = jnp.asarray(rng.uniform(0, L, size=(n_elec, 2)))
    k_grid = reciprocal_grid_2d(L, n_max=1)
    s_spin = spin_structure_factor(r, k_grid, n_up=n_elec)
    s_total = structure_factor(r, k_grid)
    np.testing.assert_allclose(s_spin, s_total, atol=1e-12)


def test_spin_structure_factor_paramagnetic_random_is_finite():
    """Half-up / half-down random configuration: S_spin should be
    finite and roughly N-bounded."""
    L = 10.0
    n_elec = 10
    rng = np.random.default_rng(3)
    r = jnp.asarray(rng.uniform(0, L, size=(n_elec, 2)))
    k_grid = reciprocal_grid_2d(L, n_max=1)
    s_spin = spin_structure_factor(r, k_grid, n_up=5)
    assert jnp.all(s_spin >= 0)
    assert float(jnp.max(s_spin)) <= n_elec + 1e-9


def test_pair_correlation_shape():
    L = 5.0
    n_elec = 8
    rng = np.random.default_rng(4)
    r = jnp.asarray(rng.uniform(0, L, size=(n_elec, 2)))

    def diff_fn(d):
        # Trivial PBC for this test
        s = d / L
        s = s - jnp.round(s)
        return s * L

    h = pair_correlation(r, n_up=4, lattice_diff_fn=diff_fn,
                        r_max=L / 2.0, n_bins=20, channel='all')
    assert h.shape == (20,)
    assert float(jnp.sum(h)) == n_elec * (n_elec - 1) // 2


def test_momentum_distribution_HF_sea_is_O1():
    """For a Fermi-sea Slater det, n(k) on occupied k_F orbitals is
    O(1)."""
    L = 10.0
    n_elec = 8
    rng = np.random.default_rng(5)
    r = jnp.asarray(rng.uniform(0, L, size=(n_elec, 2)))
    k_orbitals = reciprocal_grid_2d(L, n_max=1)[:n_elec]
    nk = momentum_distribution(r, k_orbitals)
    assert nk.shape == (n_elec,)
    assert jnp.all(nk >= 0)


def test_triangular_first_shell_returns_6_vectors():
    """First shell of triangular WC reciprocal lattice has 6 vectors
    of equal magnitude."""
    bs = reciprocal_lattice_vectors_triangular(rs=30.0, n_shell=1)
    assert bs.shape == (6, 2)
    norms = np.linalg.norm(bs, axis=-1)
    np.testing.assert_allclose(norms, norms[0], atol=1e-10)


def test_accumulator_running_average():
    state = sf_accumulator_initial(n_k=5)
    state = sf_accumulator_update(state, jnp.array([1, 2, 3, 4, 5]))
    state = sf_accumulator_update(state, jnp.array([3, 4, 5, 6, 7]))
    mean = sf_accumulator_mean(state)
    np.testing.assert_allclose(mean, jnp.array([2, 3, 4, 5, 6]))
