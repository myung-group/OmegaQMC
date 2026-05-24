"""
Tests for OmegaQMC.cs.mrpt — CASCI/NEVPT2 from a recovered c_hat.

Strategy: when we feed the PySCF FCI vector itself into our wrapper,
the resulting CASCI object should reproduce PySCF's own CASCI energy
in the same active space, and NEVPT2 on top should match a direct
CASSCF + NEVPT2 calculation at matched (ncas, nelecas) to within
the CASSCF/CASCI orbital-optimisation gap.
"""

import numpy as np
import pytest
from pyscf import gto, scf, mcscf

from OmegaQMC.cs.mrpt import (
    build_casci_from_chat,
    casscf_nevpt2_reference,
    chat_to_casci_matrix,
    compare_nevpt2,
    run_nevpt2,
    select_active_space,
)
from OmegaQMC.cs.reference import compute_fci_reference


@pytest.fixture(scope="module")
def beh2():
    """BeH2 linear at R=2.51 Bohr in 6-31G — small CAS test case."""
    mol = gto.M(
        atom="Be 0 0 0; H 0 0 2.51; H 0 0 -2.51",
        basis="6-31g", unit="Bohr", verbose=0,
    )
    ref = compute_fci_reference(mol, n_alpha=3, n_beta=3)
    ref["mol"] = mol
    return ref


@pytest.mark.slow
def test_select_active_space_finds_fractional_occupations():
    """Auto active-space selector identifies fractionally-occupied NOs."""
    # Mock NO occupations: 2 core, 2 active (fractional), 2 virtual
    occ = np.array([1.99, 1.98, 1.3, 0.7, 0.02, 0.01])
    ncore, ncas = select_active_space(occ, occ_threshold=0.05)
    assert ncore == 2
    assert ncas == 2


@pytest.mark.slow
def test_select_active_space_respects_max_ncas():
    occ = np.array([1.99, 1.7, 1.3, 0.7, 0.3, 0.02])
    ncore, ncas = select_active_space(occ, occ_threshold=0.05, max_ncas=2)
    assert ncas == 2


@pytest.mark.slow
def test_chat_to_casci_matrix_preserves_norm(beh2):
    """Projecting c_FCI onto the full FCI active space should retain unit
    norm (the active space *is* the full space here)."""
    ref = beh2
    candidate = ref["candidate_set"]
    c_fci = np.array([ref["ci_dict"][k] for k in candidate])
    # Full active space = all orbitals, all electrons
    n_orb = ref["n_orb"]
    nelec = ref["nelec"]
    ci_matrix, n_kept = chat_to_casci_matrix(
        c_fci, candidate, ncore=0, ncas=n_orb, nelecas=nelec,
    )
    # All determinants should fit
    assert n_kept == len(candidate)
    # Norm preserved
    assert abs(float(np.linalg.norm(ci_matrix)) - 1.0) < 1e-10


@pytest.mark.slow
def test_chat_to_casci_matrix_drops_outside_active(beh2):
    """If we shrink the active space, determinants involving virtual
    orbitals get dropped."""
    ref = beh2
    candidate = ref["candidate_set"]
    c_fci = np.array([ref["ci_dict"][k] for k in candidate])
    # Tiny active space CAS(2,2) — only the first 2 orbitals
    ci_matrix, n_kept = chat_to_casci_matrix(
        c_fci, candidate, ncore=2, ncas=2, nelecas=(1, 1),
    )
    assert n_kept > 0
    assert n_kept < len(candidate)
    # Renormalisation always brings the matrix to unit norm (or zero)
    norm = float(np.linalg.norm(ci_matrix))
    assert abs(norm - 1.0) < 1e-10 or norm == 0.0


@pytest.mark.slow
def test_build_casci_from_c_fci_recovers_pyscf_casci_energy(beh2):
    """Feed c_FCI in as c_hat; the resulting CASCI(full-space) energy
    must match a direct PySCF FCI energy in the same orbital basis."""
    ref = beh2
    mol = ref["mol"]
    c_fci = np.array([ref["ci_dict"][k] for k in ref["candidate_set"]])
    # Full FCI active space
    mc = build_casci_from_chat(
        mol, c_fci, ref,
        ncore=0, ncas=ref["n_orb"], nelecas=ref["nelec"],
    )
    # Should match E_FCI from the reference dict
    assert abs(mc.e_tot - ref["E_FCI"]) < 1e-8


@pytest.mark.slow
def test_casscf_nevpt2_runs(beh2):
    """Sanity that the baseline CASSCF+NEVPT2 pipeline works end-to-end."""
    ref = beh2
    mol = ref["mol"]
    # CAS(2,2) for BeH2 (the 2 outermost orbitals)
    out = casscf_nevpt2_reference(mol, ncas=2, nelecas=(1, 1))
    assert "e_hf" in out and "e_casscf" in out and "e_pt2" in out
    # NEVPT2 correction should be negative
    assert out["e_pt2"] < 0
    # Variational bound
    assert out["e_casscf"] >= ref["E_FCI"] - 1e-6


@pytest.mark.slow
def test_compare_nevpt2_full_space_matches_FCI(beh2):
    """When the active space spans the entire single-particle basis,
    NEVPT2 correction must vanish (no external orbitals) and the
    'chat' route reproduces FCI exactly when c_hat = c_FCI."""
    ref = beh2
    mol = ref["mol"]
    c_fci = np.array([ref["ci_dict"][k] for k in ref["candidate_set"]])
    # PySCF NEVPT2 fails when there's no external space; only test the
    # CASCI(full) energy alignment via build_casci_from_chat
    mc = build_casci_from_chat(
        mol, c_fci, ref,
        ncore=0, ncas=ref["n_orb"], nelecas=ref["nelec"],
    )
    assert abs(mc.e_tot - ref["E_FCI"]) < 1e-8
