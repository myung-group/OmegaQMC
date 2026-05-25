"""Tests for the active-space CASCI reference (large-basis support
for the CS-recovery pipeline)."""

import math
import numpy as np
import pytest

from OmegaQMC.utils import Mole_custom
from OmegaQMC.cs.reference import (
    compute_casci_reference, compute_fci_reference,
)


def _build_h2o(basis):
    r = 0.957
    theta = math.radians(104.5 / 2.0)
    hx = r * math.sin(theta)
    hz = r * math.cos(theta)
    mol = Mole_custom()
    mol.build(
        atom=[("O", [0, 0, 0]),
              ("H", [hx, 0, hz]),
              ("H", [-hx, 0, hz])],
        basis=basis, spin=0, charge=0, unit="Angstrom", verbose=0,
    )
    return mol


@pytest.mark.slow
def test_casci_full_active_matches_fci_sto3g():
    """For H2O/STO-3G, CASCI(7, (5,5)) covering the entire valence
    space must match full FCI energy to machine precision."""
    mol = _build_h2o("sto-3g")
    fci_ref = compute_fci_reference(
        mol, n_alpha=5, n_beta=5, candidate_tol=0.0,
    )
    casci_ref = compute_casci_reference(
        mol, ncas=7, nelecas=(5, 5), ncore=0, candidate_tol=0.0,
    )
    assert abs(casci_ref["E_CASCI"] - fci_ref["E_FCI"]) < 1e-8
    # n_orb same (= mol.nao); reference det = HF dets
    assert casci_ref["n_orb"] == fci_ref["n_orb"]
    # Spin-summed natural-orbital occupations should match (modulo
    # virtuals which CASCI sets to zero by construction)
    assert np.allclose(
        casci_ref["occ_no"][:5], fci_ref["occ_no"][:5], atol=1e-6,
    )


@pytest.mark.slow
def test_casci_h2o_ccpvdz_runs_and_indexes_full_orbitals():
    """The H2O/cc-pVDZ CAS(8,8) reference runs and produces a CI dict
    whose determinant tuples address orbitals in [0, n_orb)."""
    mol = _build_h2o("cc-pvdz")
    ref = compute_casci_reference(
        mol, ncas=8, nelecas=(4, 4), candidate_tol=1e-4,
    )
    assert ref["n_orb"] == mol.nao
    assert ref["ncore"] == 1
    assert ref["ncas"] == 8
    assert ref["nelecas_active"] == (4, 4)
    assert ref["nelec"] == (5, 5)  # ncore + active = 1 + 4 = 5 per spin
    # All determinant indices in [0, mol.nao)
    for (occ_a, occ_b) in ref["candidate_set"]:
        assert all(0 <= o < mol.nao for o in occ_a), occ_a
        assert all(0 <= o < mol.nao for o in occ_b), occ_b
        assert len(occ_a) == 5 and len(occ_b) == 5
    # Reference det = HF in NO basis = (0,1,2,3,4)x(0,1,2,3,4)
    assert ref["reference_det"] == (
        (0, 1, 2, 3, 4), (0, 1, 2, 3, 4),
    )
    # ncore + n_active = (n_total - 2*ncore)/2 + ncore => the per-spin
    # electron count matches nelec[0]
    # Sum of natural-orbital occupations = total electron count
    assert abs(float(np.sum(ref["occ_no"])) - 10.0) < 1e-6


@pytest.mark.slow
def test_casci_psi_evaluator_finite_on_random_walkers():
    """evaluate_ci_wavefunction must run on the CASCI candidate set
    (which uses full-orbital indexing with frozen core) without errors."""
    from OmegaQMC.cs.estimators import (
        evaluate_orbitals_on_walkers, evaluate_ci_wavefunction,
    )

    mol = _build_h2o("cc-pvdz")
    ref = compute_casci_reference(
        mol, ncas=8, nelecas=(4, 4), candidate_tol=1e-4,
    )
    rng = np.random.default_rng(0)
    walkers = rng.normal(size=(8, 10, 3)) * 0.5
    orb = evaluate_orbitals_on_walkers(mol, walkers, ref["no_coeff_ao"])
    c = np.array([ref["ci_dict"][k] for k in ref["candidate_set"]])
    psi = evaluate_ci_wavefunction(
        orb, ref["candidate_set"], c,
        ref["nelec"][0], ref["nelec"][1],
    )
    assert psi.shape == (8,)
    assert np.all(np.isfinite(psi))
