"""
Tests for OmegaQMC.cs.transcorrelated (Phase-0 TC two-sided decode).

Unit tests pin the single-knob estimator algebra, the diagnostics, and the
biorthogonal 1-RDM reduction. The non-trivial correctness test constructs a
controlled trial Psi_NN = Phi_det * e^{J} on H2/STO-3G with an *in-basis*
Phi_det and a known pairwise Jastrow, then verifies that the tau=+1 decode
recovers Phi_det's determinant coefficients (cusp removed) while the tau=0
Hermitian decode does not -- the controlled analogue of the leakage-collapse
that gates the TC program.
"""

import itertools

import numpy as np
import pytest
from pyscf import gto, scf

from OmegaQMC.cs.estimators import (
    evaluate_orbitals_on_walkers,
    f_I_matrix,
)
from OmegaQMC.cs.properties import compute_1rdm
from OmegaQMC.cs.transcorrelated import (
    biorthogonal_1rdm,
    biorthogonal_overlap,
    decode_single,
    decode_two_sided,
    f_I_matrix_tc,
    leakage_mass,
)
from OmegaQMC.cs.jastrow_extract import jastrow_from_logamps

from .conftest import metropolis_sample as _metropolis


# --------------------------------------------------------------------------
# Pure-algebra unit tests (no sampling)
# --------------------------------------------------------------------------

def _random_inputs(seed=0, n_det=6, K_s=40, n_orb=3, n_alpha=1, n_beta=1):
    rng = np.random.default_rng(seed)
    K = K_s
    N = n_alpha + n_beta
    orb_vals = rng.normal(size=(K, N, n_orb))
    psi_vals = rng.normal(size=(K,)) + 1.5  # keep away from 0
    occs = list(itertools.combinations(range(n_orb), n_alpha))
    cand = [(a, b) for a in occs for b in occs][:n_det]
    return orb_vals, cand, psi_vals, n_alpha, n_beta


def test_tau0_matches_hermitian_exactly():
    orb_vals, cand, psi_vals, na, nb = _random_inputs()
    J = np.random.default_rng(1).normal(size=psi_vals.shape)
    f0 = f_I_matrix_tc(orb_vals, cand, psi_vals, J, 0.0, na, nb)
    f_ref = f_I_matrix(orb_vals, cand, psi_vals, na, nb)
    np.testing.assert_allclose(f0, f_ref, rtol=0, atol=0)


def test_tau_applies_jastrow_weight():
    orb_vals, cand, psi_vals, na, nb = _random_inputs()
    J = np.random.default_rng(2).normal(size=psi_vals.shape)
    f_ref = f_I_matrix(orb_vals, cand, psi_vals, na, nb)
    for tau in (+1.0, -1.0, 0.5):
        f = f_I_matrix_tc(orb_vals, cand, psi_vals, J, tau, na, nb)
        np.testing.assert_allclose(
            f, f_ref * np.exp(-tau * J)[None, :], rtol=1e-12, atol=1e-12
        )


def test_jastrow_length_mismatch_raises():
    orb_vals, cand, psi_vals, na, nb = _random_inputs()
    with pytest.raises(ValueError):
        f_I_matrix_tc(orb_vals, cand, psi_vals, np.ones(3), 1.0, na, nb)


def test_biorthogonal_overlap_bounds():
    c = np.array([0.8, -0.5, 0.1])
    assert biorthogonal_overlap(c, c) == pytest.approx(1.0)
    assert biorthogonal_overlap(c, -c) == pytest.approx(-1.0)
    assert biorthogonal_overlap(np.array([1.0, 0.0]),
                                np.array([0.0, 1.0])) == pytest.approx(0.0)
    assert biorthogonal_overlap(np.zeros(3), c) == 0.0


def test_leakage_mass():
    c_ref = np.array([0.95, 0.30, 1e-5, -2e-5])
    c_hat = np.array([0.90, 0.40, 0.10, -0.08])  # mass on the two tail dets
    # tail = indices 2,3 (|c_ref|<1e-3): 0.10^2 + 0.08^2
    assert leakage_mass(c_hat, c_ref, 1e-3) == pytest.approx(0.0164)
    # raising the threshold catches index 1 too
    assert leakage_mass(c_hat, c_ref, 0.5) == pytest.approx(
        0.40 ** 2 + 0.10 ** 2 + 0.08 ** 2
    )


def test_biorthogonal_1rdm_reduces_to_single_vector():
    """For c_L == c_R the biorthogonal 1-RDM equals the standard 1-RDM."""
    n_orb, na, nb = 3, 1, 1
    occs = list(itertools.combinations(range(n_orb), na))
    cand = [(a, b) for a in occs for b in occs]  # full 3x3 = 9 dets
    rng = np.random.default_rng(7)
    c = rng.normal(size=len(cand))
    c /= np.linalg.norm(c)
    gamma_std = compute_1rdm(c, cand, n_orb, (na, nb))
    gamma_bi = biorthogonal_1rdm(c, c, cand, n_orb, (na, nb))
    np.testing.assert_allclose(gamma_bi, gamma_std, rtol=1e-10, atol=1e-10)


def test_jastrow_from_logamps():
    full = np.array([1.0, 2.0, -0.5])
    slat = np.array([0.4, 1.5, -1.0])
    np.testing.assert_allclose(
        jastrow_from_logamps(full, slat), full - slat
    )
    with pytest.raises(ValueError):
        jastrow_from_logamps(full, slat[:2])


def test_biorthogonal_1rdm_near_defective_raises():
    n_orb, na, nb = 2, 1, 1
    cand = [((0,), (0,)), ((1,), (1,))]
    c_L = np.array([1.0, 0.0])
    c_R = np.array([0.0, 1.0])  # c_L . c_R = 0
    with pytest.raises(ValueError):
        biorthogonal_1rdm(c_L, c_R, cand, n_orb, (na, nb))


# --------------------------------------------------------------------------
# Sampling-based correctness: controlled Psi_NN = Phi_det * e^{J}
# --------------------------------------------------------------------------

@pytest.fixture(scope="module")
def h2_basis():
    mol = gto.M(atom="H 0 0 0; H 0 0 1.4", basis="sto-3g",
                unit="Bohr", verbose=0)
    mf = scf.RHF(mol).run()
    return mol, np.asarray(mf.mo_coeff)


def _jastrow(walkers):
    """Bounded smooth pairwise log-Jastrow J(R) = -0.4 exp(-|r12|^2/2)."""
    w = np.asarray(walkers)
    r12 = w[:, 0, :] - w[:, 1, :]
    r12sq = np.sum(r12 ** 2, axis=1)
    return -0.4 * np.exp(-0.5 * r12sq)


def _build_trial(mol, mo_coeff, c_det):
    """Return psi_fn(R) = Phi_det(R) * exp(J(R)) for a 2-config in-basis
    Phi_det = c_det[0] D_(0,0) + c_det[1] D_(1,1)."""
    def psi_fn(R):
        orb = evaluate_orbitals_on_walkers(mol, np.asarray(R), mo_coeff)
        orb_a = orb[:, 0, :]
        orb_b = orb[:, 1, :]
        phi = (c_det[0] * orb_a[:, 0] * orb_b[:, 0]
               + c_det[1] * orb_a[:, 1] * orb_b[:, 1])
        return phi * np.exp(_jastrow(R))
    return psi_fn


def test_two_sided_decode_recovers_right_vector(h2_basis):
    mol, mo_coeff = h2_basis
    c_det = np.array([0.9, -0.4])  # Phi_det coefficients (in-basis)
    psi_fn = _build_trial(mol, mo_coeff, c_det)

    walkers, psi_vals, acc = _metropolis(
        psi_fn, n_walkers=6000, n_electrons=2, n_steps=40, seed=3,
    )
    assert 0.1 < acc < 0.95
    jastrow_vals = _jastrow(walkers)
    orb_vals = evaluate_orbitals_on_walkers(mol, walkers, mo_coeff)

    # Full 2-orbital 1a1b candidate set; reference det index 0 = (0,0).
    cand = [((0,), (0,)), ((1,), (1,)), ((0,), (1,)), ((1,), (0,))]
    ref_sign = 1.0

    out = decode_two_sided(
        orb_vals, cand, psi_vals, jastrow_vals,
        n_alpha=1, n_beta=1, reference_sign=ref_sign, use_lasso=False,
    )

    c_R = out["c_R"]
    target = np.array([c_det[0], c_det[1], 0.0, 0.0])
    target = target / np.linalg.norm(target)
    # tau=+1 recovers Phi_det (cusp removed) within MC noise.
    np.testing.assert_allclose(c_R, target, atol=0.05)
    # The Hermitian decode sees the e^{J}-dressed trial and differs on the
    # leading amplitude (controlled analogue of basis leakage).
    assert abs(out["c_herm"][0] - target[0]) > abs(c_R[0] - target[0])
    # Biorthogonal overlap is well-conditioned.
    assert abs(out["rho"]) > 0.5


def test_jzero_all_branches_agree(h2_basis):
    """With J == 0 the three decodes coincide and rho == 1."""
    mol, mo_coeff = h2_basis
    c_det = np.array([0.85, 0.527])

    def psi_fn(R):
        orb = evaluate_orbitals_on_walkers(mol, np.asarray(R), mo_coeff)
        orb_a, orb_b = orb[:, 0, :], orb[:, 1, :]
        return (c_det[0] * orb_a[:, 0] * orb_b[:, 0]
                + c_det[1] * orb_a[:, 1] * orb_b[:, 1])

    walkers, psi_vals, _ = _metropolis(
        psi_fn, n_walkers=3000, n_electrons=2, n_steps=30, seed=5,
    )
    orb_vals = evaluate_orbitals_on_walkers(mol, walkers, mo_coeff)
    cand = [((0,), (0,)), ((1,), (1,)), ((0,), (1,)), ((1,), (0,))]
    Jzero = np.zeros(walkers.shape[0])

    out = decode_two_sided(
        orb_vals, cand, psi_vals, Jzero,
        n_alpha=1, n_beta=1, reference_sign=1.0, use_lasso=False,
    )
    np.testing.assert_allclose(out["c_R"], out["c_herm"], atol=1e-12)
    np.testing.assert_allclose(out["c_L"], out["c_herm"], atol=1e-12)
    assert out["rho"] == pytest.approx(1.0, abs=1e-10)
