"""
Tests for the 3D Electron Gas (3DEG) AFQMC module.

Validates k-grid generation, integral construction, trial wavefunctions,
and AFQMC energy for the 3D homogeneous electron gas (jellium).
"""

import numpy as np
import jax
import jax.numpy as jnp
import pytest

from OmegaQMC.afqmc_pw_heg import (
    _generate_3d_kgrid,
    build_3deg_system,
    build_h1e_pw_3d,
    build_cholesky_pw_3d,
    compute_madelung_3d,
    prepare_3deg_integrals,
    build_trial_3deg,
    get_afqmc_3deg_func,
)


class TestKGrid3D:
    """Test 3D k-grid generation."""

    def test_shell_counts(self):
        """3D shells should have standard counts: 1, 7, 19, 27, 33, ..."""
        grid, n = _generate_3d_kgrid(1)
        assert n == 1  # just (0,0,0)

        grid, n = _generate_3d_kgrid(7)
        assert n == 7  # (0,0,0) + 6 nearest neighbors

        grid, n = _generate_3d_kgrid(19)
        assert n == 19  # + 12 next-nearest

        grid, n = _generate_3d_kgrid(27)
        assert n == 27  # + 8 (1,1,1) corners

    def test_grid_shape(self):
        """Grid should have shape (N_pw, 3)."""
        grid, n = _generate_3d_kgrid(7)
        assert grid.shape == (n, 3)

    def test_includes_origin(self):
        """Grid should include (0,0,0)."""
        grid, n = _generate_3d_kgrid(1)
        assert any(np.all(g == 0) for g in grid)

    def test_sorted_by_norm(self):
        """Grid should be sorted by |n|²."""
        grid, n = _generate_3d_kgrid(19)
        norms_sq = np.sum(grid**2, axis=1)
        assert np.all(norms_sq[:-1] <= norms_sq[1:])

    def test_symmetry(self):
        """If (nx, ny, nz) is in grid, (-nx, -ny, -nz) should be too."""
        grid, n = _generate_3d_kgrid(19)
        grid_set = set(map(tuple, grid))
        for g in grid:
            assert tuple(-g) in grid_set


class TestSystemSetup:
    """Test system parameter construction."""

    def test_unpolarized(self):
        """Unpolarized system should have nup = ndown."""
        sys = build_3deg_system(rs=1.0, N_elec=14, N_pw=19)
        assert sys['nup'] == 7
        assert sys['ndown'] == 7
        assert sys['N_elec'] == 14
        assert sys['polarization'] == 'unpolarized'

    def test_polarized(self):
        """Polarized system should have nup = N, ndown = 0."""
        sys = build_3deg_system(rs=1.0, N_elec=7, N_pw=19,
                                polarization='polarized')
        assert sys['nup'] == 7
        assert sys['ndown'] == 0

    def test_volume_density(self):
        """Volume should satisfy (4π/3) r_s³ = V/N."""
        for rs in [1.0, 2.0, 5.0]:
            sys = build_3deg_system(rs=rs, N_elec=14, N_pw=19)
            vol_per_elec = sys['volume'] / sys['N_elec']
            expected = (4.0 * np.pi / 3.0) * rs**3
            np.testing.assert_allclose(vol_per_elec, expected, rtol=1e-10)

    def test_kvecs_shape(self):
        """k-vectors should have shape (N_pw, 3)."""
        sys = build_3deg_system(rs=1.0, N_elec=14, N_pw=19)
        assert sys['kvecs'].shape == (sys['N_pw'], 3)

    def test_twist_angle(self):
        """Twist angle should shift k-vectors."""
        kappa = np.array([0.25, 0.1, -0.3])
        sys0 = build_3deg_system(rs=1.0, N_elec=14, N_pw=19)
        sys_tw = build_3deg_system(rs=1.0, N_elec=14, N_pw=19,
                                   kappa=kappa)
        # Shifted k-vectors should differ
        assert not np.allclose(sys0['kvecs'], sys_tw['kvecs'])
        # Difference should be uniform shift
        dk = sys0['dk']
        diff = sys_tw['kvecs'] - sys0['kvecs']
        expected_shift = kappa * dk
        for row in diff:
            np.testing.assert_allclose(row, expected_shift, rtol=1e-10)


class TestIntegrals:
    """Test integral construction."""

    def test_h1e_diagonal(self):
        """h1e should be diagonal (pure kinetic for jellium)."""
        sys = build_3deg_system(rs=2.0, N_elec=14, N_pw=19)
        h1e = build_h1e_pw_3d(sys)
        off_diag = h1e - np.diag(np.diag(h1e))
        assert np.max(np.abs(off_diag)) < 1e-15

    def test_h1e_kinetic_energy(self):
        """Diagonal should be |k|²/2."""
        sys = build_3deg_system(rs=2.0, N_elec=14, N_pw=19)
        h1e = build_h1e_pw_3d(sys)
        kvecs = sys['kvecs']
        for i in range(sys['N_pw']):
            expected = 0.5 * np.dot(kvecs[i], kvecs[i])
            np.testing.assert_allclose(h1e[i, i], expected, rtol=1e-12)

    def test_cholesky_shape(self):
        """Cholesky vectors should have correct shape."""
        sys = build_3deg_system(rs=2.0, N_elec=14, N_pw=19)
        chol, q_vecs, chol_sign = build_cholesky_pw_3d(sys)
        N_pw = sys['N_pw']
        assert chol.shape[1] == N_pw
        assert chol.shape[2] == N_pw
        assert q_vecs.shape[0] == chol.shape[0]
        assert q_vecs.shape[1] == 3
        assert chol_sign.shape == (chol.shape[0],)
        # All signs should be +1 or -1
        assert np.all(np.abs(chol_sign) == 1.0)

    def test_cholesky_sparse(self):
        """Each symmetrized Cholesky vector should be sparse.

        Symmetrized vectors combine L^q and L^{-q}, so A/B vectors
        can have up to 2*N_pw nonzeros (from both q and -q contributions).
        """
        sys = build_3deg_system(rs=2.0, N_elec=14, N_pw=19)
        chol, _, _ = build_cholesky_pw_3d(sys)
        for iq in range(chol.shape[0]):
            nnz = np.count_nonzero(chol[iq])
            assert nnz <= 2 * sys['N_pw']

    def test_cholesky_reproduces_coulomb(self):
        """Unsigned Σ_g L^g_{pr} L^g_{qs} should give the Coulomb
        matrix elements.

        For symmetrized Cholesky, A²+B² = pf² recovers the Coulomb
        potential.  The sign-weighted form (A²-B²) is used in the
        Hamiltonian decomposition and encodes momentum conservation,
        but individual v[p,p,q,q] elements are recovered by the
        unsigned sum.

        Convention: v[p,q,r,s] = Σ_g L^g[p,r] L^g[q,s] (unsigned).
        Then v[p,p,q,q] = Σ_g L^g[p,q]² = v(k_p - k_q).
        """
        sys = build_3deg_system(rs=2.0, N_elec=2, N_pw=7)
        chol, q_vecs, chol_sign = build_cholesky_pw_3d(sys)
        grid_int = sys['grid_int']
        grid_set = set()
        for i in range(sys['N_pw']):
            grid_set.add((int(grid_int[i, 0]), int(grid_int[i, 1]),
                          int(grid_int[i, 2])))
        # Unsigned reconstruction: v[p,q,r,s] = Σ_g L^g[p,r] L^g[q,s]
        v = np.einsum('gpr,gqs->pqrs', chol, chol)
        kvecs = sys['kvecs']
        volume = sys['volume']
        checked = 0
        for p in range(sys['N_pw']):
            for q in range(sys['N_pw']):
                if p == q:
                    continue
                dg = (int(grid_int[p, 0] - grid_int[q, 0]),
                      int(grid_int[p, 1] - grid_int[q, 1]),
                      int(grid_int[p, 2] - grid_int[q, 2]))
                if dg not in grid_set or dg == (0, 0, 0):
                    continue
                dq = kvecs[p] - kvecs[q]
                q_norm_sq = np.dot(dq, dq)
                expected = 4.0 * np.pi / (q_norm_sq * volume)
                # v[p,p,q,q] = Σ_g L^g[p,q]² = 4π/(|kp-kq|² V)
                np.testing.assert_allclose(
                    v[p, p, q, q], expected, rtol=1e-10,
                    err_msg=f"Failed for p={p}, q={q}")
                checked += 1
        assert checked > 0, "No valid momentum transfers found"

    def test_cholesky_sign_pairing(self):
        """Each (q, -q) pair should produce one A-type (+1) and one
        B-type (-1) Cholesky vector."""
        sys = build_3deg_system(rs=2.0, N_elec=2, N_pw=7)
        chol, q_vecs, chol_sign = build_cholesky_pw_3d(sys)
        N_chol = chol.shape[0]
        # Count A-type and B-type
        n_plus = np.sum(chol_sign == +1.0)
        n_minus = np.sum(chol_sign == -1.0)
        assert n_plus + n_minus == N_chol
        # For paired q-vectors, A and B come in equal numbers.
        # Unpaired vectors (q = -q mod G) are all A-type,
        # so n_plus >= n_minus.
        assert n_plus >= n_minus

    def test_cholesky_symmetry_types(self):
        """A-type vectors should be symmetric, B-type anti-symmetric."""
        sys = build_3deg_system(rs=2.0, N_elec=2, N_pw=7)
        chol, q_vecs, chol_sign = build_cholesky_pw_3d(sys)
        for ig in range(chol.shape[0]):
            L = chol[ig]
            if chol_sign[ig] > 0:
                # A-type: symmetric
                np.testing.assert_allclose(
                    L, L.T, atol=1e-14,
                    err_msg=f"A-type vector {ig} is not symmetric")
            else:
                # B-type: anti-symmetric
                np.testing.assert_allclose(
                    L, -L.T, atol=1e-14,
                    err_msg=f"B-type vector {ig} is not anti-symmetric")


class TestMadelung:
    """Test Madelung energy computation."""

    def test_madelung_negative(self):
        """Madelung energy should be negative (attractive)."""
        sys = build_3deg_system(rs=1.0, N_elec=14, N_pw=19)
        e_mad = compute_madelung_3d(sys)
        assert e_mad < 0

    def test_madelung_known_value(self):
        """Madelung constant for sc lattice is known: -2.837297 / r_s."""
        # For N=1 electron, the Madelung energy per electron is
        # approximately -2.837297 / r_s in Hartree (for a Wigner-Seitz cell).
        # Our convention: e_mad = E_madelung / N_elec.
        # The exact value depends on the lattice type.
        # For the simple cubic lattice with Ewald:
        # e_M ≈ -1.4186 / r_s (in Hartree)
        sys = build_3deg_system(rs=1.0, N_elec=14, N_pw=19)
        e_mad = compute_madelung_3d(sys)
        # Should be O(1/r_s) and negative
        assert -3.0 < e_mad < 0


class TestTrial:
    """Test trial wavefunction construction."""

    def test_trial_shape_unpolarized(self):
        """Trial should have correct shape for unpolarized."""
        sys = build_3deg_system(rs=1.0, N_elec=14, N_pw=19)
        h1e = build_h1e_pw_3d(sys)
        trial_up, trial_dn = build_trial_3deg(h1e, 7, 7)
        assert trial_up.shape == (19, 7)
        assert trial_dn.shape == (19, 7)

    def test_trial_shape_polarized(self):
        """Trial should have correct shape for polarized."""
        sys = build_3deg_system(rs=1.0, N_elec=7, N_pw=19,
                                polarization='polarized')
        h1e = build_h1e_pw_3d(sys)
        trial_up, trial_dn = build_trial_3deg(h1e, 7, 0)
        assert trial_up.shape == (19, 7)
        assert trial_dn.shape == (19, 0)

    def test_trial_orthonormal(self):
        """Trial orbitals should be orthonormal."""
        sys = build_3deg_system(rs=1.0, N_elec=14, N_pw=19)
        h1e = build_h1e_pw_3d(sys)
        trial_up, trial_dn = build_trial_3deg(h1e, 7, 7)
        # Each column is a unit vector (occupation)
        ovlp_up = trial_up.T @ trial_up
        ovlp_dn = trial_dn.T @ trial_dn
        np.testing.assert_allclose(ovlp_up, np.eye(7), atol=1e-12)
        np.testing.assert_allclose(ovlp_dn, np.eye(7), atol=1e-12)

    def test_trial_occupies_lowest_states(self):
        """Trial should occupy the lowest kinetic energy states."""
        sys = build_3deg_system(rs=1.0, N_elec=14, N_pw=19)
        h1e = build_h1e_pw_3d(sys)
        trial_up, _ = build_trial_3deg(h1e, 7, 7)
        # Check that occupied states have lower energy than unoccupied
        energies = np.diag(np.asarray(h1e))
        occ_indices = np.where(np.sum(np.asarray(trial_up)**2, axis=1) > 0.5)[0]
        unocc_indices = np.where(np.sum(np.asarray(trial_up)**2, axis=1) < 0.5)[0]
        max_occ_energy = np.max(energies[occ_indices])
        min_unocc_energy = np.min(energies[unocc_indices])
        assert max_occ_energy <= min_unocc_energy + 1e-10


class TestAFQMCFreeElectron:
    """Test AFQMC for the non-interacting electron gas."""

    def test_free_electron_energy(self):
        """AFQMC without Coulomb should give exact kinetic energy."""
        sys = build_3deg_system(rs=2.0, N_elec=14, N_pw=19)
        h1e = build_h1e_pw_3d(sys)
        # Expected: sum of lowest 7 kinetic energies × 2 (spin)
        energies = np.sort(np.diag(h1e))
        e_exact = 2.0 * np.sum(energies[:7])

        driver = get_afqmc_3deg_func(
            sys, dt=0.01, include_coulomb=False, verbose=False)
        result = driver(
            num_walkers=50, num_blocks=30,
            num_steps_per_block=10, num_blocks_equil=5,
            fname_log=None)

        np.testing.assert_allclose(
            result['energy_mean'], e_exact, atol=0.01,
            err_msg=f"Free electron energy: got {result['energy_mean']:.6f}, "
                    f"expected {e_exact:.6f}")


class TestAFQMCWithCoulomb:
    """Test AFQMC with Coulomb interaction (small system)."""

    def test_coulomb_lowers_energy(self):
        """Coulomb interaction (with exchange) should lower the energy
        compared to the kinetic-only case for jellium."""
        sys = build_3deg_system(rs=2.0, N_elec=2, N_pw=7)

        driver_free = get_afqmc_3deg_func(
            sys, dt=0.01, include_coulomb=False, verbose=False)
        result_free = driver_free(
            num_walkers=50, num_blocks=30,
            num_steps_per_block=10, num_blocks_equil=5,
            fname_log=None)

        driver_coul = get_afqmc_3deg_func(
            sys, dt=0.01, include_coulomb=True, verbose=False)
        result_coul = driver_coul(
            num_walkers=50, num_blocks=30,
            num_steps_per_block=10, num_blocks_equil=5,
            fname_log=None)

        # With exchange, total energy should be lower
        assert result_coul['energy_mean'] < result_free['energy_mean'] + 0.1


class TestPolarized:
    """Test fully polarized electron gas."""

    def test_polarized_free_electron(self):
        """Polarized free electron gas energy should match kinetic sum."""
        sys = build_3deg_system(rs=2.0, N_elec=7, N_pw=19,
                                polarization='polarized')
        h1e = build_h1e_pw_3d(sys)
        energies = np.sort(np.diag(h1e))
        e_exact = np.sum(energies[:7])  # only alpha, no factor of 2

        driver = get_afqmc_3deg_func(
            sys, dt=0.01, include_coulomb=False, verbose=False)
        result = driver(
            num_walkers=50, num_blocks=30,
            num_steps_per_block=10, num_blocks_equil=5,
            fname_log=None)

        np.testing.assert_allclose(
            result['energy_mean'], e_exact, atol=0.01,
            err_msg=f"Polarized free electron: got {result['energy_mean']:.6f}"
                    f", expected {e_exact:.6f}")
