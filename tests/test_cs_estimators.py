"""
Tests for OmegaQMC.cs.estimators.

Smoke tests for the f_I matrix and recovery primitives. The non-trivial
correctness check uses Metropolis sampling from a small FCI wavefunction
(H2/STO-3G) and verifies that the empirical sample mean of f_I converges
to the FCI coefficients c_I.
"""

import math

import numpy as np
import pytest
from pyscf import gto

from OmegaQMC.cs.estimators import (
    _interleaved_to_grouped_indices,
    _normalization,
    estimate_ci,
    evaluate_ci_wavefunction,
    evaluate_orbitals_on_walkers,
    f_I_matrix,
    lambda_cv,
    normalize_and_align,
    recovery_metrics,
    soft_threshold,
)
from OmegaQMC.cs.reference import compute_fci_reference

from .conftest import metropolis_sample as _metropolis


@pytest.fixture(scope="module")
def h2_sto3g_equilibrium():
    """H2 at 1.4 Bohr in STO-3G, with PySCF Mole attached."""
    mol = gto.M(atom="H 0 0 0; H 0 0 1.4", basis="sto-3g",
                unit="Bohr", verbose=0)
    ref = compute_fci_reference(mol, n_alpha=1, n_beta=1)
    ref["mol"] = mol
    return ref


@pytest.fixture(scope="module")
def h2_sto3g_stretched():
    """H2 at 2.5 Bohr in STO-3G; significant multireference character."""
    mol = gto.M(atom="H 0 0 0; H 0 0 2.5", basis="sto-3g",
                unit="Bohr", verbose=0)
    ref = compute_fci_reference(mol, n_alpha=1, n_beta=1)
    ref["mol"] = mol
    return ref


def test_normalization_values():
    assert _normalization(1, 1) == 1.0
    assert abs(_normalization(2, 2) - 0.5) < 1e-14
    assert abs(_normalization(3, 3) - 1.0 / math.sqrt(36.0)) < 1e-14


def test_soft_threshold():
    x = np.array([1.0, -2.0, 0.1, -0.05, 0.0])
    np.testing.assert_allclose(soft_threshold(x, 0.5), [0.5, -1.5, 0.0, 0.0, 0.0])


def test_normalize_and_align_unit_norm():
    x = np.array([0.7, -0.3, 0.05])
    c_norm, proj_mass = normalize_and_align(x, reference_sign=+1.0)
    assert abs(float(np.sum(c_norm ** 2)) - 1.0) < 1e-12
    assert proj_mass == pytest.approx(float(np.sum(x ** 2)))
    assert c_norm[0] > 0


def test_normalize_and_align_flips_when_reference_opposite():
    x = np.array([-0.7, 0.3, -0.05])
    c_norm, _ = normalize_and_align(x, reference_sign=+1.0)
    assert c_norm[0] > 0
    # The relative structure is preserved (signs of non-reference flip too)
    assert c_norm[1] < 0
    assert c_norm[2] > 0


def test_normalize_and_align_no_flip_when_reference_matches():
    x = np.array([0.7, -0.3, 0.05])
    c_norm_pos, _ = normalize_and_align(x, reference_sign=+1.0)
    c_norm_neg, _ = normalize_and_align(-x, reference_sign=-1.0)
    np.testing.assert_allclose(c_norm_pos, -c_norm_neg, atol=1e-12)


def test_normalize_and_align_handles_zero_input():
    x = np.zeros(3)
    c_norm, proj_mass = normalize_and_align(x, reference_sign=+1.0)
    assert proj_mass == 0.0
    np.testing.assert_array_equal(c_norm, x)


def test_normalize_and_align_handles_zero_reference_sign():
    """When c_true[0] is zero, sign alignment is undefined; pass through."""
    x = np.array([-0.7, 0.3, 0.05])
    c_norm, _ = normalize_and_align(x, reference_sign=0.0)
    assert c_norm[0] < 0


def test_interleaved_to_grouped_balanced():
    """H4 (2+2): interleaved [a,b,a,b] -> grouped [a,a,b,b]."""
    perm = _interleaved_to_grouped_indices(2, 2)
    np.testing.assert_array_equal(perm, [0, 2, 1, 3])


def test_interleaved_to_grouped_unbalanced():
    """Li (2+1): interleaved [a,b,a] -> grouped [a,a,b]."""
    perm = _interleaved_to_grouped_indices(2, 1)
    np.testing.assert_array_equal(perm, [0, 2, 1])


def test_interleaved_to_grouped_h2_is_identity():
    """H2 (1+1): permutation reduces to identity."""
    perm = _interleaved_to_grouped_indices(1, 1)
    np.testing.assert_array_equal(perm, [0, 1])


@pytest.mark.slow
def test_evaluate_orbitals_interleaved_permutes_correctly():
    """Build deterministic walkers, verify interleaved-convention output
    matches grouped-convention output with electrons swapped."""
    from pyscf import gto
    mol = gto.M(atom="H 0 0 0; H 0 0 1.0; H 0 0 2.0; H 0 0 3.0",
                basis="sto-3g", unit="Bohr", verbose=0)
    no_coeff = np.eye(mol.nao)
    rng = np.random.default_rng(7)
    walkers_interleaved = rng.normal(size=(5, 4, 3))
    walkers_grouped_equivalent = walkers_interleaved[:, [0, 2, 1, 3], :]
    orb_int = evaluate_orbitals_on_walkers(
        mol, walkers_interleaved, no_coeff,
        convention="interleaved", n_alpha=2, n_beta=2,
    )
    orb_grp = evaluate_orbitals_on_walkers(
        mol, walkers_grouped_equivalent, no_coeff, convention="grouped",
    )
    np.testing.assert_allclose(orb_int, orb_grp, atol=1e-12)


@pytest.mark.slow
def test_evaluate_orbitals_interleaved_requires_spin_counts():
    from pyscf import gto
    mol = gto.M(atom="H 0 0 0; H 0 0 1.0", basis="sto-3g",
                unit="Bohr", verbose=0)
    walkers = np.zeros((3, 2, 3))
    no_coeff = np.eye(mol.nao)
    with pytest.raises(ValueError, match="n_alpha and n_beta"):
        evaluate_orbitals_on_walkers(
            mol, walkers, no_coeff, convention="interleaved",
        )


def test_recovery_metrics_basic():
    c_hat = np.array([1.0, 0.0, 0.05])
    c_true = np.array([1.0, 0.0, 0.06])
    m = recovery_metrics(c_hat, c_true, eta=1e-2)
    assert abs(m["L_inf_err"] - 0.01) < 1e-12
    assert m["L_2_err"] > 0
    assert m["support_err"] == 0.0  # both flag index 2 as in-support


def test_recovery_metrics_support_difference():
    c_hat = np.array([1.0, 0.5, 0.0])
    c_true = np.array([1.0, 0.0, 0.5])
    m = recovery_metrics(c_hat, c_true, eta=1e-2)
    assert m["support_err"] == pytest.approx(2.0 / 3.0)


@pytest.mark.slow
def test_orbital_evaluation_shape(h2_sto3g_equilibrium):
    mol = h2_sto3g_equilibrium["mol"]
    no_coeff = h2_sto3g_equilibrium["no_coeff_ao"]
    walkers = np.random.default_rng(0).normal(scale=1.0, size=(7, 2, 3))
    orb = evaluate_orbitals_on_walkers(mol, walkers, no_coeff)
    assert orb.shape == (7, 2, no_coeff.shape[1])
    assert np.all(np.isfinite(orb))


@pytest.mark.slow
def test_evaluate_ci_wavefunction_finite(h2_sto3g_equilibrium):
    ref = h2_sto3g_equilibrium
    candidate = ref["candidate_set"]
    coeffs = np.array([ref["ci_dict"][k] for k in candidate])
    walkers = np.random.default_rng(1).normal(scale=1.0, size=(11, 2, 3))
    orb = evaluate_orbitals_on_walkers(ref["mol"], walkers, ref["no_coeff_ao"])
    psi = evaluate_ci_wavefunction(orb, candidate, coeffs, 1, 1)
    assert psi.shape == (11,)
    assert np.all(np.isfinite(psi))
    assert float(np.max(np.abs(psi))) > 0


@pytest.mark.slow
def test_f_I_reference_is_one_under_single_det_psi(h2_sto3g_equilibrium):
    """For Psi = D_ref alone (c_ref = 1), f_ref(R) = 1 identically."""
    ref = h2_sto3g_equilibrium
    ref_det = ref["reference_det"]
    walkers = np.random.default_rng(2).normal(scale=1.2, size=(20, 2, 3))
    orb = evaluate_orbitals_on_walkers(ref["mol"], walkers, ref["no_coeff_ao"])
    psi = evaluate_ci_wavefunction(orb, [ref_det], np.array([1.0]), 1, 1)
    fI = f_I_matrix(orb, [ref_det], psi, 1, 1, use_jit=False)
    assert fI.shape == (1, 20)
    np.testing.assert_allclose(fI[0], 1.0, atol=1e-10)


@pytest.mark.slow
def test_f_I_orthogonality_under_single_det_psi(h2_sto3g_equilibrium):
    """Under Psi = D_ref, MC mean of f_I for I != ref converges to 0."""
    ref = h2_sto3g_equilibrium
    ref_det = ref["reference_det"]
    no_coeff = ref["no_coeff_ao"]
    mol = ref["mol"]

    def psi_fn(R):
        orb = evaluate_orbitals_on_walkers(mol, R, no_coeff)
        return evaluate_ci_wavefunction(orb, [ref_det], np.array([1.0]), 1, 1)

    walkers, psi, _ = _metropolis(
        psi_fn, n_walkers=200, n_electrons=2,
        n_steps=200, burnin=100, seed=3,
    )
    orb = evaluate_orbitals_on_walkers(mol, walkers, no_coeff)
    fI = f_I_matrix(orb, ref["candidate_set"], psi, 1, 1, use_jit=False)
    means = fI.mean(axis=1)
    ref_idx = ref["candidate_set"].index(ref_det)
    assert abs(means[ref_idx] - 1.0) < 1e-10
    for i, det in enumerate(ref["candidate_set"]):
        if det != ref_det:
            assert abs(means[i]) < 0.1, (
                f"f_{det} empirical mean = {means[i]} (should be ~0)"
            )


@pytest.mark.slow
def test_recover_fci_h2_stretched(h2_sto3g_stretched):
    """End-to-end: Psi from FCI coefficients -> Metropolis -> recover c_FCI."""
    ref = h2_sto3g_stretched
    mol = ref["mol"]
    candidate = ref["candidate_set"]
    coeffs = np.array([ref["ci_dict"][k] for k in candidate])
    no_coeff = ref["no_coeff_ao"]

    def psi_fn(R):
        orb = evaluate_orbitals_on_walkers(mol, R, no_coeff)
        return evaluate_ci_wavefunction(orb, candidate, coeffs, 1, 1)

    walkers, psi, accept_rate = _metropolis(
        psi_fn, n_walkers=400, n_electrons=2,
        n_steps=400, burnin=200, seed=4,
    )
    assert 0.2 < accept_rate < 0.95, f"acceptance rate {accept_rate}"

    orb = evaluate_orbitals_on_walkers(mol, walkers, no_coeff)
    fI = f_I_matrix(orb, candidate, psi, 1, 1, use_jit=False)
    c_hat = fI.mean(axis=1)

    for i, (ch, ct) in enumerate(zip(c_hat, coeffs)):
        assert abs(ch - ct) < 0.05, (
            f"coef {i} ({candidate[i]}): hat={ch:.4f} true={ct:.4f}"
        )


@pytest.mark.slow
def test_estimate_ci_preserves_reference_and_thresholds_others():
    f_I = np.array([
        [1.0, 1.0, 1.0, 1.0],   # reference, large mean
        [0.05, 0.06, 0.04, 0.05],  # small, should be thresholded
        [0.3, 0.3, 0.3, 0.3],   # mid-size, partially thresholded
    ])
    c_hat = estimate_ci(f_I, lam=0.1, reference_idx=0)
    assert c_hat[0] == pytest.approx(1.0)
    assert c_hat[1] == 0.0
    assert c_hat[2] == pytest.approx(0.2)


@pytest.mark.slow
def test_lambda_cv_picks_finite_lambda():
    rng = np.random.default_rng(0)
    n_det = 20
    K_s = 200
    c_true = np.zeros(n_det)
    c_true[0] = 1.0
    c_true[1] = 0.3
    c_true[2] = -0.2
    f_I = c_true[:, None] + 0.1 * rng.standard_normal((n_det, K_s))
    lambdas = np.logspace(-3, 0, 12)
    best, losses = lambda_cv(f_I, lambdas, reference_idx=0,
                             n_folds=5, rng_seed=7)
    assert lambdas[0] <= best <= lambdas[-1]
    assert losses.shape == (12,)
    assert np.all(np.isfinite(losses))


@pytest.mark.slow
def test_f_I_matrix_jit_matches_eager(h2_sto3g_equilibrium):
    ref = h2_sto3g_equilibrium
    candidate = ref["candidate_set"]
    coeffs = np.array([ref["ci_dict"][k] for k in candidate])
    no_coeff = ref["no_coeff_ao"]
    walkers = np.random.default_rng(5).normal(scale=1.2, size=(15, 2, 3))
    orb = evaluate_orbitals_on_walkers(ref["mol"], walkers, no_coeff)
    psi = evaluate_ci_wavefunction(orb, candidate, coeffs, 1, 1)
    eager = f_I_matrix(orb, candidate, psi, 1, 1, use_jit=False)
    jitted = f_I_matrix(orb, candidate, psi, 1, 1, use_jit=True)
    np.testing.assert_allclose(eager, jitted, atol=1e-10)
