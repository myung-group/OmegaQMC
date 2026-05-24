"""
Tests for OmegaQMC.cs.properties — RDMs and properties from c_hat.

The pivotal verification is that feeding the PySCF FCI vector itself
into our pipeline reproduces PySCF's reference RDM to floating-point
precision. Beyond that we test the simple invariants (trace, NO
ordering, symmetry-required vanishing dipoles).
"""

import numpy as np
import pytest
from pyscf import fci, gto, scf

from OmegaQMC.cs.properties import (
    compute_1rdm,
    effective_unpaired_electrons,
    electric_dipole,
    electric_quadrupole,
    multireference_diagnostics,
    natural_occupations_from_rdm,
    report_properties,
    reshape_chat_to_pyscf_matrix,
)
from OmegaQMC.cs.reference import compute_fci_reference


@pytest.fixture(scope="module")
def h2_stretched():
    mol = gto.M(atom="H 0 0 0; H 0 0 2.5", basis="cc-pvdz",
                unit="Bohr", verbose=0)
    ref = compute_fci_reference(mol, n_alpha=1, n_beta=1)
    ref["mol"] = mol
    return ref


@pytest.fixture(scope="module")
def lih_eq():
    mol = gto.M(atom="Li 0 0 0; H 0 0 3.04", basis="sto-3g",
                unit="Bohr", verbose=0)
    ref = compute_fci_reference(mol, n_alpha=2, n_beta=2)
    ref["mol"] = mol
    return ref


@pytest.mark.slow
def test_chat_to_pyscf_matrix_roundtrip(h2_stretched):
    """Reshape preserves coefficient values at the right indices."""
    ref = h2_stretched
    candidate = ref["candidate_set"]
    c_fci = np.array([ref["ci_dict"][k] for k in candidate])
    matrix = reshape_chat_to_pyscf_matrix(
        c_fci, candidate, ref["n_orb"], 1, 1,
    )
    # Norm should equal ||c_FCI||
    assert abs(float(np.sum(matrix ** 2)) - float(np.sum(c_fci ** 2))) < 1e-12


@pytest.mark.slow
def test_1rdm_matches_pyscf_for_c_fci(h2_stretched):
    """Feeding the FCI vector itself must reproduce PySCF's reference 1-RDM."""
    ref = h2_stretched
    mol = ref["mol"]
    candidate = ref["candidate_set"]
    c_fci = np.array([ref["ci_dict"][k] for k in candidate])
    gamma_us = compute_1rdm(c_fci, candidate, ref["n_orb"], ref["nelec"])
    cisolver = fci.FCI(mol, ref["no_coeff_ao"])
    _, ci_no = cisolver.kernel()
    gamma_pyscf = cisolver.make_rdm1(ci_no, ref["n_orb"], ref["nelec"])
    # Sign-align: PySCF may pick opposite global sign on the eigenvector
    err = min(
        float(np.max(np.abs(gamma_us - gamma_pyscf))),
        float(np.max(np.abs(gamma_us + gamma_pyscf))),
    )
    assert err < 1e-10, (
        f"1-RDM mismatch ||delta||_inf = {err} between our pipeline and PySCF"
    )


@pytest.mark.slow
def test_trace_of_1rdm_equals_nelec(h2_stretched, lih_eq):
    for ref in (h2_stretched, lih_eq):
        candidate = ref["candidate_set"]
        c_fci = np.array([ref["ci_dict"][k] for k in candidate])
        gamma = compute_1rdm(c_fci, candidate, ref["n_orb"], ref["nelec"])
        n_elec = ref["nelec"][0] + ref["nelec"][1]
        assert abs(float(np.trace(gamma)) - n_elec) < 1e-10


@pytest.mark.slow
def test_natural_occupations_descending_and_bounded(h2_stretched):
    ref = h2_stretched
    candidate = ref["candidate_set"]
    c_fci = np.array([ref["ci_dict"][k] for k in candidate])
    gamma = compute_1rdm(c_fci, candidate, ref["n_orb"], ref["nelec"])
    occ = natural_occupations_from_rdm(gamma)
    assert np.all(np.diff(occ) <= 1e-10)
    assert occ.min() > -1e-8
    assert occ.max() < 2.0 + 1e-8


@pytest.mark.slow
def test_dipole_vanishes_for_h2(h2_stretched):
    """H2 is centrosymmetric; the FCI dipole moment is identically zero."""
    ref = h2_stretched
    mol = ref["mol"]
    c_fci = np.array([ref["ci_dict"][k] for k in ref["candidate_set"]])
    gamma = compute_1rdm(c_fci, ref["candidate_set"], ref["n_orb"], ref["nelec"])
    dipole = electric_dipole(mol, ref["no_coeff_ao"], gamma)
    assert dipole["mu_magnitude_debye"] < 1e-6


@pytest.mark.slow
def test_lih_dipole_nonzero(lih_eq):
    """LiH has a substantial dipole moment (~5-6 D experimental)."""
    ref = lih_eq
    mol = ref["mol"]
    c_fci = np.array([ref["ci_dict"][k] for k in ref["candidate_set"]])
    gamma = compute_1rdm(c_fci, ref["candidate_set"], ref["n_orb"], ref["nelec"])
    dipole = electric_dipole(mol, ref["no_coeff_ao"], gamma)
    # Should be O(several) Debye; loose check for non-pathological output
    assert 1.0 < dipole["mu_magnitude_debye"] < 10.0


@pytest.mark.slow
def test_effective_unpaired_closed_shell_small():
    """For a perfectly closed-shell single-reference, eff. unpaired = 0."""
    occ = np.array([2.0, 2.0, 0.0, 0.0])
    assert effective_unpaired_electrons(occ) < 1e-12


@pytest.mark.slow
def test_effective_unpaired_singlet_diradical():
    """Half-filled NOs give eff. unpaired = 1 per radical pair."""
    occ = np.array([1.0, 1.0])  # two singly-occupied NOs
    # N_unp = 1*(2-1)/2 + 1*(2-1)/2 = 1.0
    assert abs(effective_unpaired_electrons(occ) - 1.0) < 1e-12


@pytest.mark.slow
def test_multireference_diagnostics_reference_is_largest(h2_stretched):
    ref = h2_stretched
    c_fci = np.array([ref["ci_dict"][k] for k in ref["candidate_set"]])
    diag = multireference_diagnostics(c_fci, ref["candidate_set"], ref["nelec"])
    abs_c = np.abs(c_fci)
    assert diag["C0"] == c_fci[int(np.argmax(abs_c))]
    assert diag["C0_sq"] == diag["C0"] ** 2


@pytest.mark.slow
def test_report_properties_complete_fields(h2_stretched):
    """The report dict should have all the fields the writer needs."""
    ref = h2_stretched
    c_fci = np.array([ref["ci_dict"][k] for k in ref["candidate_set"]])
    rep = report_properties(c_fci, ref, ref["mol"])
    for key in ("gamma", "occupations", "dipole",
                "quadrupole", "diagnostics", "trace_gamma"):
        assert key in rep
    for key in ("mu_au", "mu_debye", "mu_magnitude_debye"):
        assert key in rep["dipole"]
    for key in ("C0", "C0_sq", "reference_index",
                "effective_unpaired"):
        assert key in rep["diagnostics"]
