"""Tests for the Wigner-crystal Gaussian envelope (Phase 2)."""

import numpy as np
import jax
import jax.numpy as jnp
import pytest

from OmegaQMC.psi.nn.env_localized_2d import (
    GaussianLocalizedEnvelope2D,
    commensurate_triangular_supercell,
    triangular_lattice_sites,
)
from OmegaQMC.psi.nn.types import PhysicalConfiguration


def test_triangular_lattice_sites_shape():
    sites = triangular_lattice_sites(rs=30.0, n_sites=18)
    assert sites.shape == (18, 2)


def test_triangular_lattice_density():
    """Density of placed sites should equal n = 1/(pi rs^2) at large
    super-cell sizes."""
    rs = 30.0
    n_sites = 100
    sites = triangular_lattice_sites(rs, n_sites)
    # Bounding rectangle area:
    a = np.sqrt(2.0 / (np.sqrt(3.0) / (np.pi * rs ** 2)))
    M = int(np.ceil(np.sqrt(n_sites)))
    # M^2 sites in an M*a x (M * sqrt(3)/2 * a + M * a) parallelogram
    # — too coarse a check; instead verify the nearest-neighbour
    # spacing matches the analytic value within 1%.
    diffs = sites[:, None, :] - sites[None, :, :]
    d = np.linalg.norm(diffs, axis=-1)
    np.fill_diagonal(d, np.inf)
    nn = float(np.min(d))
    assert abs(nn - a) / a < 0.01


def test_envelope_2d_shape_unpolarized():
    rs = 30.0
    n_up = 9
    n_dn = 9
    L1, L2 = commensurate_triangular_supercell(rs, n_up + n_dn)
    L = float(np.linalg.norm(L1))  # use one side for square-cell init
    env = GaussianLocalizedEnvelope2D(
        n_up=n_up, n_down=n_dn, n_det=2, rs=rs, L=L,
        sigma_init=0.25, spin_pattern='neel',
    )
    rng = np.random.default_rng(0)
    pc = PhysicalConfiguration(
        R=jnp.zeros((0, 2)),
        r=jnp.asarray(rng.uniform(0, L, size=(n_up + n_dn, 2))),
        mol_idx=jnp.asarray(0),
    )
    orb = env(pc)
    assert orb.shape == (2, n_up + n_dn, n_up + n_dn)


def test_envelope_2d_evaluated_at_sites_is_block_diagonal_dominant():
    """Place electrons exactly at the lattice sites: the resulting
    Slater orbital matrix should have large diagonal entries (each
    electron sits on its own Gaussian) and tiny off-diagonal entries
    (other electrons are at distant sites)."""
    rs = 30.0
    n_up, n_dn = 4, 4
    L = float(triangular_lattice_sites(rs, n_up + n_dn)[:, 0].max() * 4)
    env = GaussianLocalizedEnvelope2D(
        n_up=n_up, n_down=n_dn, n_det=1, rs=rs, L=L,
        sigma_init=0.1, spin_pattern='neel',
    )
    # Place electrons exactly at the per-spin sites.
    sites_up = np.asarray(env.sites_up)[0]   # (n_up, 2)
    sites_dn = np.asarray(env.sites_dn)[0]   # (n_dn, 2)
    r = jnp.concatenate([
        jnp.asarray(sites_up), jnp.asarray(sites_dn),
    ], axis=0)
    pc = PhysicalConfiguration(
        R=jnp.zeros((0, 2)), r=r, mol_idx=jnp.asarray(0),
    )
    orb = env(pc)  # (1, n_elec, n_up + n_dn)
    orb_up_block = np.asarray(orb[0, :n_up, :n_up])
    # Diagonal entries should be ~1 (Gaussian at zero distance).
    diag = np.diag(orb_up_block)
    np.testing.assert_allclose(diag, 1.0, atol=1e-6)
    # Off-diagonal should be much smaller (well-separated sites).
    off_max = float(np.max(np.abs(orb_up_block - np.diag(diag))))
    assert off_max < 0.5


def test_envelope_2d_grad_finite():
    rs = 30.0
    n_up, n_dn = 4, 4
    L = float(triangular_lattice_sites(rs, n_up + n_dn)[:, 0].max() * 4)
    env = GaussianLocalizedEnvelope2D(
        n_up=n_up, n_down=n_dn, n_det=1, rs=rs, L=L,
    )
    rng = np.random.default_rng(1)
    r = jnp.asarray(rng.uniform(0, L, size=(n_up + n_dn, 2)))

    def f(r):
        pc = PhysicalConfiguration(
            R=jnp.zeros((0, 2)), r=r, mol_idx=jnp.asarray(0),
        )
        orb = env(pc)
        return jnp.sum(orb)

    g = jax.grad(f)(r)
    assert g.shape == r.shape
    assert jnp.all(jnp.isfinite(g))


def test_crystal_init_walkers_2d_shape_and_locality():
    """Crystal-aware walker init: walkers sit near triangular sites
    with Gaussian noise of size ~ sigma_init * a_NN."""
    import jax
    from OmegaQMC.psi.nn.env_localized_2d import (
        crystal_init_walkers_2d, triangular_lattice_sites,
    )
    rs = 30.0
    n_up, n_down = 9, 9
    L = float(np.sqrt(np.pi * 18) * rs)
    walkers = crystal_init_walkers_2d(
        jax.random.key(0), num_walkers=8,
        n_up=n_up, n_down=n_down, L=L,
        sigma_init=0.10, spin_pattern='neel',
    )
    assert walkers.shape == (8, 18, 2)
    # All walker positions should be within [0, L)^2 after wrapping
    assert jnp.all((walkers >= 0) & (walkers < L))
    # Average distance from each walker to the nearest triangular site
    # should be O(sigma_init * a_NN).
    sites = triangular_lattice_sites(rs, 18) % L
    a_nn = float(np.sqrt(2 * np.pi / np.sqrt(3)) * rs)
    walker_flat = np.asarray(walkers).reshape(-1, 2)  # (8*18, 2)
    # Minimum-image distance from each walker to each site:
    diffs = walker_flat[:, None, :] - sites[None, :, :]
    diffs_mi = diffs - L * np.round(diffs / L)
    dists = np.linalg.norm(diffs_mi, axis=-1)
    nn_dist = np.min(dists, axis=-1)
    median_nn_dist = float(np.median(nn_dist))
    # Walkers should sit close to sites: median NN distance <
    # ~2 * (sigma_init * a_NN) per spin assignment.
    assert median_nn_dist < 2.0 * 0.10 * a_nn, (
        f"median walker->site distance {median_nn_dist:.2f} Bohr "
        f"vs noise scale {0.10 * a_nn:.2f} Bohr — walkers are not "
        "near sites."
    )


def test_crystal_init_walkers_2d_invalid_pattern_raises():
    import jax
    from OmegaQMC.psi.nn.env_localized_2d import crystal_init_walkers_2d
    with pytest.raises(ValueError):
        crystal_init_walkers_2d(
            jax.random.key(0), 4, n_up=4, n_down=4, L=10.0,
            spin_pattern='banana',
        )


def test_neel_assigns_alternating_sites():
    """In Neel pattern, up sites and down sites should interleave."""
    rs = 30.0
    n_up, n_dn = 4, 4
    L = 30.0
    env = GaussianLocalizedEnvelope2D(
        n_up=n_up, n_down=n_dn, n_det=1, rs=rs, L=L,
        spin_pattern='neel',
    )
    all_sites = triangular_lattice_sites(rs, n_up + n_dn)
    expected_up = all_sites[0::2][:n_up]
    expected_dn = all_sites[1::2][:n_dn]
    np.testing.assert_allclose(
        np.asarray(env.sites_up)[0], expected_up % L, atol=1e-10,
    )
    np.testing.assert_allclose(
        np.asarray(env.sites_dn)[0], expected_dn % L, atol=1e-10,
    )
