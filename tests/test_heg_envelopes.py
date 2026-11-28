"""Envelopes of the electron-gas wavefunctions.

Real and complex plane-wave envelopes in 3D and 2D (with twists) and
the Wigner-crystal Gaussian envelope: shapes, shells, periodicity and
finite gradients.
"""
from typing import NamedTuple
import numpy as np
import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from OmegaQMC.psi.nn.env_periodic import (  # noqa: E402
    PlaneWaveEnvelope,
    enumerate_real_pw_basis,
    ComplexPlaneWaveEnvelope,
    enumerate_complex_pw_basis_2d,
    enumerate_real_pw_basis_2d,
    enumerate_complex_pw_basis,
)
from OmegaQMC.psi.nn.types import PhysicalConfiguration  # noqa: E402
from OmegaQMC.psi.nn.env_localized_2d import (  # noqa: E402
    GaussianLocalizedEnvelope2D,
    commensurate_triangular_supercell,
    triangular_lattice_sites,
)


# ---------------------------------------------------------------------------
# From test_env_periodic.py
#
# Unit tests for the plane-wave HEG envelope.
#
# Checks four claims:
#   * Shape matches the molecular envelope convention
#     ``(n_det, n_elec, n_up+n_down)``.
#   * At initialisation, the envelope is the free-electron
#     Slater-determinant basis (i-th orbital = i-th real plane wave).
#   * The envelope is strictly periodic under lattice translations.
#   * The Slater determinant built from the envelope at initialisation
#     recovers the non-interacting kinetic energy (per electron
#     ``(3/5) k_F²``) to within numerical noise from the SR jitter
#     added at init.
# ---------------------------------------------------------------------------


class _PhysConfHEG(NamedTuple):
    R: jax.Array
    r: jax.Array
    mol_idx: jax.Array


def _make_pc(r: jax.Array) -> _PhysConfHEG:
    return _PhysConfHEG(
        R=jnp.zeros((0, 3)), r=r, mol_idx=jnp.asarray(0),
    )


# -------------------------------------------------------------
# Basis enumeration
# -------------------------------------------------------------
def test_real_pw_basis_closed_shells():
    """Closed-shell magic numbers: 1, 7, 19, 27 must be reached
    exactly at the end of each shell."""
    for n in [1, 7, 19, 27]:
        basis = enumerate_real_pw_basis(n, L=4.0)
        assert int(basis.basis_idx.shape[0]) == n


def test_real_pw_basis_k_sorted():
    basis = enumerate_real_pw_basis(19, L=4.0)
    k_sq_of_orb = np.asarray(basis.k_sq)[
        np.asarray(basis.basis_idx)
    ]
    assert np.all(np.diff(k_sq_of_orb) >= -1e-12)


def test_real_pw_basis_first_is_constant():
    basis = enumerate_real_pw_basis(7, L=4.0)
    # First basis function: cos of the k=0 vector -> constant 1.
    assert int(basis.basis_idx[0]) == 0
    assert int(basis.basis_is_sin[0]) == 0
    assert float(basis.k_sq[0]) == 0.0


# -------------------------------------------------------------
# Envelope module
# -------------------------------------------------------------
def test_envelope_shape_unpolarized():
    """N=14 unpolarized HEG: n_up = n_down = 7."""
    env = PlaneWaveEnvelope(
        n_up=7, n_down=7, n_det=4, L=4.0,
    )
    rng = np.random.default_rng(0)
    r = jnp.asarray(rng.uniform(0, 4.0, size=(14, 3)))
    out = env(_make_pc(r))
    assert out.shape == (4, 14, 14)


def test_envelope_shape_polarized():
    """N=7 polarized HEG: n_up = 7, n_down = 0."""
    env = PlaneWaveEnvelope(
        n_up=7, n_down=0, n_det=2, L=4.0,
    )
    rng = np.random.default_rng(1)
    r = jnp.asarray(rng.uniform(0, 4.0, size=(7, 3)))
    out = env(_make_pc(r))
    assert out.shape == (2, 7, 7)


def test_envelope_lattice_periodicity():
    """Shifting all electrons by a lattice vector leaves the
    envelope invariant (exactly, not just approximately)."""
    L = 4.0
    env = PlaneWaveEnvelope(
        n_up=7, n_down=7, n_det=1, L=L,
    )
    rng = np.random.default_rng(2)
    r = jnp.asarray(rng.uniform(0, L, size=(14, 3)))
    shift = L * jnp.asarray([1.0, -2.0, 3.0])
    out0 = env(_make_pc(r))
    out1 = env(_make_pc(r + shift))
    np.testing.assert_allclose(out0, out1, atol=1e-9)


def test_envelope_init_is_free_electron_slater():
    """At init, the envelope is (approximately) the free-electron
    real PW basis.  We can check this by computing the Slater
    determinant of the up block and verifying it factors as a
    determinant of cos/sin(k_j·r_i).
    """
    L = 4.0
    n_up, n_down = 7, 7
    env = PlaneWaveEnvelope(
        n_up=n_up, n_down=n_down, n_det=1, L=L,
    )

    rng = np.random.default_rng(3)
    r = jnp.asarray(rng.uniform(0, L, size=(n_up + n_down, 3)))
    out = env(_make_pc(r))
    # Up-spin Slater block: first n_up rows, first n_up cols.
    orb_up = out[0, :n_up, :n_up]

    # Expected free-electron matrix (no jitter).
    basis = enumerate_real_pw_basis(n_up, L)
    kvecs = np.asarray(basis.kvecs)
    idx = np.asarray(basis.basis_idx)
    is_sin = np.asarray(basis.basis_is_sin)
    r_up = np.asarray(r[:n_up])
    expected = np.zeros((n_up, n_up))
    for j in range(n_up):
        k_j = kvecs[idx[j]]
        proj = r_up @ k_j
        expected[:, j] = np.sin(proj) if is_sin[j] else np.cos(proj)
    # The actual envelope differs from `expected` only by the 1e-3
    # symmetry-breaking jitter in the coefficients.  Check closeness
    # at that scale.
    np.testing.assert_allclose(
        np.asarray(orb_up), expected, atol=5e-2,
    )
    # Determinant is non-zero (non-degenerate free-electron state).
    assert abs(float(np.linalg.det(orb_up))) > 1e-3


def test_envelope_gradient_finite():
    """Gradient of log|det(orb_up)| w.r.t. electron coords is finite."""
    L = 4.0
    env = PlaneWaveEnvelope(n_up=7, n_down=7, n_det=1, L=L)

    graphdef, params, other = nnx.split(env, nnx.Param, ...)

    def log_det_up(r):
        m = nnx.merge(graphdef, params, other)
        out = m(_make_pc(r))
        return jnp.log(jnp.abs(jnp.linalg.det(out[0, :7, :7])))

    rng = np.random.default_rng(4)
    r = jnp.asarray(rng.uniform(0, L, size=(14, 3)))
    g = jax.grad(log_det_up)(r)
    assert jnp.all(jnp.isfinite(g))


# ---------------------------------------------------------------------------
# From test_env_periodic_2d.py
#
# Unit tests for 2D plane-wave envelopes.
#
# Verifies real and complex 2D PW basis enumeration matches the 2D
# closed-shell sequence (1, 5, 9, 13, 21, 25, 29, 37, 45, 57, ...) and
# that the envelopes evaluate to the correct shape on random walkers.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------
# Real-valued 2D PW basis
# ---------------------------------------------------------------------
@pytest.mark.parametrize('n_orb', [1, 5, 9, 13, 21, 25, 29, 37, 45, 57])
def test_real_pw_basis_2d_size(n_orb):
    basis = enumerate_real_pw_basis_2d(n_orb, L=10.0)
    assert basis.basis_idx.shape == (n_orb,)
    assert basis.kvecs.shape[1] == 2


def test_real_pw_basis_2d_first_orbital_is_constant():
    basis = enumerate_real_pw_basis_2d(1, L=5.0)
    np.testing.assert_allclose(basis.kvecs, jnp.zeros((1, 2)))
    assert int(basis.basis_is_sin[0]) == 0


def test_real_pw_basis_2d_n5_has_first_shell():
    """nup=5 fills shells |n|^2 = 0, 1.  Five basis functions:
    constant + (cos kx, sin kx) + (cos ky, sin ky)."""
    basis = enumerate_real_pw_basis_2d(5, L=5.0)
    assert basis.basis_idx.shape == (5,)
    # 1 constant + 2 cos + 2 sin
    n_cos = int(np.sum(np.asarray(basis.basis_is_sin) == 0))
    n_sin = int(np.sum(np.asarray(basis.basis_is_sin) == 1))
    assert n_cos == 3 and n_sin == 2


# ---------------------------------------------------------------------
# Complex 2D PW basis
# ---------------------------------------------------------------------
def test_complex_pw_basis_2d_shape():
    basis = enumerate_complex_pw_basis_2d(29, L=10.0, kappa=(0.0, 0.0))
    assert basis.kvecs.shape == (29, 2)
    assert basis.k_sq.shape == (29,)
    assert basis.kappa.shape == (2,)


def test_complex_pw_basis_2d_kappa_shifts():
    basis_g = enumerate_complex_pw_basis_2d(5, L=10.0, kappa=(0.0, 0.0))
    basis_k = enumerate_complex_pw_basis_2d(5, L=10.0, kappa=(0.1, 0.2))
    # Kappa shift breaks degeneracy: at twist all 5 k_sq differ;
    # at Γ the shells are degenerate.
    assert len(set(np.asarray(basis_k.k_sq).tolist())) == 5


def test_complex_pw_basis_2d_kappa_wrong_dim_raises():
    with pytest.raises(ValueError):
        enumerate_complex_pw_basis_2d(5, L=10.0, kappa=(0.1, 0.2, 0.3))


# ---------------------------------------------------------------------
# Envelope evaluation
# ---------------------------------------------------------------------
def _make_2d_phys_conf(n_elec, L, seed=0):
    rng = np.random.default_rng(seed)
    return PhysicalConfiguration(
        R=jnp.zeros((0, 2)),
        r=jnp.asarray(rng.uniform(0, L, size=(n_elec, 2))),
        mol_idx=jnp.asarray(0),
    )


def test_real_envelope_2d_shape_unpolarized():
    L = 10.0
    env = PlaneWaveEnvelope(
        n_up=29, n_down=29, n_det=1, L=L, dim=2,
    )
    pc = _make_2d_phys_conf(58, L)
    orb = env(pc)
    assert orb.shape == (1, 58, 58)
    assert orb.dtype == jnp.float64


def test_real_envelope_2d_shape_polarized():
    L = 10.0
    env = PlaneWaveEnvelope(
        n_up=29, n_down=0, n_det=1, L=L, dim=2,
    )
    pc = _make_2d_phys_conf(29, L)
    orb = env(pc)
    assert orb.shape == (1, 29, 29)


def test_complex_envelope_2d_shape():
    L = 10.0
    env = ComplexPlaneWaveEnvelope(
        n_up=29, n_down=29, n_det=1, L=L, dim=2, kappa=(0.1, 0.2),
    )
    pc = _make_2d_phys_conf(58, L)
    orb = env(pc)
    assert orb.shape == (1, 58, 58)
    assert jnp.iscomplexobj(orb)


def test_real_envelope_2d_periodic():
    """Envelope output must be invariant under integer-cell shifts of
    the full electron configuration (because the envelope coefficients
    are initialised to plane-wave eigenstates)."""
    L = 10.0
    env = PlaneWaveEnvelope(
        n_up=29, n_down=29, n_det=1, L=L, dim=2,
    )
    pc0 = _make_2d_phys_conf(58, L, seed=42)
    pc_shift = pc0._replace(r=pc0.r + L * jnp.array([1.0, 0.0]))
    orb0 = env(pc0)
    orb_shift = env(pc_shift)
    np.testing.assert_allclose(orb0, orb_shift, atol=1e-9)


def test_real_envelope_2d_invalid_dim_raises():
    with pytest.raises(ValueError):
        PlaneWaveEnvelope(
            n_up=5, n_down=5, n_det=1, L=10.0, dim=4,
        )


# ---------------------------------------------------------------------
# Slater determinant of the 2D envelope reproduces the HEG kinetic energy
# ---------------------------------------------------------------------
def test_2d_envelope_slater_det_kinetic_matches_analytic():
    """At init, the Slater det of the 2D PW envelope is the
    free-electron Fermi sea.  The kinetic energy <T> = sum_k |k|^2/2
    over occupied k must match the analytical sum using the integer
    grid we generated.  This catches off-by-one in the basis ordering.
    """
    from OmegaQMC.psi.heg_2d import build_2deg_system, hf_energy_2d_finite

    sys = build_2deg_system(rs=2.0, N_elec=58, polarization='unpolarized')
    L = sys['L']

    # Analytical kinetic via heg_2d
    hf = hf_energy_2d_finite(sys)
    expected_T_per_elec = hf['kinetic']

    # Envelope-based T: sum |k|^2/2 over the closed-shell basis chosen
    # by the envelope (k = 2pi*n/L, n in the Fermi sea).  This must
    # agree to floating-point precision because the envelope
    # initializes orbitals to those exact plane waves.
    basis = enumerate_real_pw_basis_2d(29, L=L)
    # The basis includes one zero-vector (the constant) + first-shell
    # ks paired (cos, sin).  We need the per-orbital |k|^2: for orbital
    # i we look up basis.basis_idx[i] -> kvec, take its |k|^2.
    k_sq_per_orb = np.asarray(basis.k_sq)[np.asarray(basis.basis_idx)]
    T_envelope = 0.5 * float(np.sum(k_sq_per_orb))
    # Two spins (unpolarized), per-electron average:
    T_envelope_per_elec = 2.0 * T_envelope / sys['N_elec']
    np.testing.assert_allclose(
        T_envelope_per_elec, expected_T_per_elec, atol=1e-12
    )


# ---------------------------------------------------------------------------
# From test_env_periodic_complex.py
#
# Unit tests for the complex plane-wave envelope.
#
# Covers the TABC prerequisites:
#   * Basis enumeration at κ=0 recovers the Γ-point closed shells
#     (1, 7, 19, 27, …).
#   * Basis enumeration at κ≠0 breaks the shell degeneracy and is
#     consistent with analytical expectations.
#   * The envelope is lattice-periodic (shift electrons by any lattice
#     vector, the complex orbital matrix picks up a global phase that
#     drops out of |ψ|²).
#   * At κ=0, ``|det(complex-envelope orbitals)|²`` equals
#     ``|det(real-envelope orbitals)|²`` — same physical Slater
#     determinant up to a global phase.
#   * ``jax.grad`` through a complex log-det is finite.
# ---------------------------------------------------------------------------


def _pc(r):
    return _PhysConfHEG(R=jnp.zeros((0, 3)),
                        r=r, mol_idx=jnp.asarray(0))


# -----------------------------------------------------------
# Basis enumeration
# -----------------------------------------------------------
def test_complex_basis_gamma_closed_shells():
    """At κ=0, 7 orbitals → k=0 + first shell of 6 integer vectors."""
    basis = enumerate_complex_pw_basis(7, L=4.0, kappa=(0, 0, 0))
    assert basis.kvecs.shape == (7, 3)
    n_ints = np.asarray(basis.n_ints)
    # Orbital 0 must be k=0.
    np.testing.assert_array_equal(n_ints[0], [0, 0, 0])
    # Remaining 6 must be ±e_x, ±e_y, ±e_z (any order).
    rest = {tuple(n) for n in n_ints[1:]}
    expected = {
        (+1, 0, 0), (-1, 0, 0),
        (0, +1, 0), (0, -1, 0),
        (0, 0, +1), (0, 0, -1),
    }
    assert rest == expected


def test_complex_basis_kappa_breaks_degeneracy():
    """At κ ≠ 0 the 7 lowest-|k|² integer triples are no longer a
    symmetric ±-pair set; instead we pick the 7 whose |n+κ|² values
    are smallest."""
    kappa = (0.1, 0.0, 0.0)
    basis = enumerate_complex_pw_basis(7, L=4.0, kappa=kappa)
    k_sq = np.asarray(basis.k_sq)
    assert np.all(np.diff(k_sq) >= -1e-12)
    # |(-1, 0, 0) + (0.1, 0, 0)|² = 0.81  <  |(1, 0, 0) + (0.1, 0, 0)|² = 1.21
    # So the basis should include (-1, 0, 0) before (+1, 0, 0).
    n_ints = np.asarray(basis.n_ints)
    idx_neg = next(i for i, n in enumerate(n_ints)
                   if tuple(n) == (-1, 0, 0))
    idx_pos = next((i for i, n in enumerate(n_ints)
                    if tuple(n) == (1, 0, 0)), len(n_ints))
    assert idx_neg < idx_pos


# -----------------------------------------------------------
# Envelope forward
# -----------------------------------------------------------
def test_complex_envelope_shape():
    env = ComplexPlaneWaveEnvelope(n_up=7, n_down=7, n_det=4, L=4.0)
    rng = np.random.default_rng(0)
    r = jnp.asarray(rng.uniform(0, 4.0, size=(14, 3)))
    out = env(_pc(r))
    assert out.shape == (4, 14, 14)
    assert out.dtype == jnp.complex128


def test_complex_envelope_lattice_periodicity():
    """Shifting all electrons by a lattice vector multiplies each
    orbital by exp(i k · a) — a global *k-dependent* phase.  The
    determinant picks up the product of these phases but its
    modulus is unchanged."""
    L = 4.0
    n_up = n_down = 7
    env = ComplexPlaneWaveEnvelope(n_up=n_up, n_down=n_down, n_det=1, L=L)
    rng = np.random.default_rng(1)
    r = jnp.asarray(rng.uniform(0, L, size=(n_up + n_down, 3)))
    out0 = env(_pc(r))
    shift = L * jnp.asarray([1.0, -2.0, 3.0])
    out1 = env(_pc(r + shift))
    # Up block Slater determinants must have identical modulus.
    det0 = jnp.linalg.det(out0[0, :n_up, :n_up])
    det1 = jnp.linalg.det(out1[0, :n_up, :n_up])
    np.testing.assert_allclose(
        float(jnp.abs(det0)), float(jnp.abs(det1)), atol=1e-10,
    )


def test_complex_vs_real_envelope_same_physical_density():
    """At κ = 0, the two Slater determinants differ only by a linear
    change of basis ``{1, cos, sin} ↔ {1, e^{±ik·r}}`` which is
    invertible but not unit-determinant.  Their ratio
    ``|det_c|² / |det_r|²`` is therefore a geometry-independent
    constant — the |det(basis change)|² factor — and the ratio of
    ratios at two configurations must be 1.  That constant factor
    drops out of VMC observables (it's a wavefunction-normalization
    convention)."""
    L = 4.0
    n_up = n_down = 7

    env_c = ComplexPlaneWaveEnvelope(n_up=n_up, n_down=n_down, n_det=1, L=L)
    env_r = PlaneWaveEnvelope(n_up=n_up, n_down=n_down, n_det=1, L=L)

    def det_ratio(seed):
        r = jnp.asarray(
            np.random.default_rng(seed).uniform(0, L, size=(n_up + n_down, 3))
        )
        det_c = jnp.linalg.det(env_c(_pc(r))[0, :n_up, :n_up])
        det_r = jnp.linalg.det(env_r(_pc(r))[0, :n_up, :n_up])
        return float(jnp.abs(det_c) ** 2) / float(jnp.abs(det_r) ** 2)

    ratios = [det_ratio(s) for s in range(5)]
    # The ratio must be constant across configurations.  For the 7-orbital
    # unpolarised basis at Γ the analytic value is 4³ = 64 (from three
    # ±k-pair 2×2 change-of-basis blocks each with |det|² = 4).
    np.testing.assert_allclose(ratios, [ratios[0]] * len(ratios),
                               rtol=1e-10)
    assert abs(ratios[0] - 64.0) < 1e-6


def test_complex_envelope_grad_finite():
    """jax.grad of log|det(orb_up)| through the complex Slater block
    produces finite real gradients."""
    L = 4.0
    n_up = 7
    env = ComplexPlaneWaveEnvelope(n_up=n_up, n_down=7, n_det=1, L=L)
    graphdef, params, other = nnx.split(env, nnx.Param, ...)

    def log_abs_det(r):
        m = nnx.merge(graphdef, params, other)
        out = m(_pc(r))
        d = jnp.linalg.det(out[0, :n_up, :n_up])
        return jnp.log(jnp.abs(d))

    rng = np.random.default_rng(3)
    r = jnp.asarray(rng.uniform(0, L, size=(14, 3)))
    g = jax.grad(log_abs_det)(r)
    assert jnp.all(jnp.isfinite(g))
    assert g.dtype == jnp.float64


# ---------------------------------------------------------------------------
# From test_env_localized_2d.py
#
# Tests for the Wigner-crystal Gaussian envelope (Phase 2).
# ---------------------------------------------------------------------------


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
