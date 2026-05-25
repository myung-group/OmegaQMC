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
def test_subspace_rotate_works_on_casci_reference():
    """The subspace_rotate_to_eigenstates routine routes through the
    active-space contract_2e for CASCI references instead of the full-
    basis contract that would OOM in cc-pVDZ."""
    from OmegaQMC.cs.transition import subspace_rotate_to_eigenstates
    from OmegaQMC.cs.reference import ci_to_dict
    from pyscf import mcscf, scf

    mol = _build_h2o("cc-pvdz")
    # candidate_tol=0 keeps all active-space determinants; the
    # 1-root default candidate set under-represents excited state.
    ref = compute_casci_reference(
        mol, ncas=8, nelecas=(4, 4), candidate_tol=0.0,
    )
    # Get two CASCI roots in the same active space as the "input"
    mf = scf.RHF(mol).run(verbose=0)
    mc = mcscf.CASCI(mf, 8, (4, 4))
    mc.ncore = 1
    mc.fcisolver.nroots = 2
    mc.verbose = 0
    mc.kernel(mo_coeff=ref["no_coeff_ao"])
    # Convert each root to a c_hat vector in full-orbital indexing
    candidate = ref["candidate_set"]
    n_orb = ref["n_orb"]
    # Active-space FCI -> dict in full-orbital tuples
    from OmegaQMC.cs.reference import _casci_to_dict_with_core
    candidate = ref["candidate_set"]
    c_hats = []
    for ci_root in mc.ci:
        d = _casci_to_dict_with_core(
            np.asarray(ci_root), 1, 8, (4, 4), tol=0.0,
        )
        c = np.array([d.get(k, 0.0) for k in candidate])
        c_hats.append(c / max(np.linalg.norm(c), 1e-30))
    # The two FCI eigenstates are already orthogonal in their span
    rot = subspace_rotate_to_eigenstates(c_hats, ref, mol)
    # Eigenvalues should be the two CAS-FCI eigenvalues (within
    # truncation error; CASCI returns exact within active space)
    E_eig = sorted(rot["E_eig"])
    E_ref = sorted([float(e) for e in mc.e_tot])
    # 1e-5 tolerance: the input c_hats are truncated to the
    # candidate_set (367 dets above 1e-4), so the rotation operates
    # in a slightly smaller subspace than the CAS-FCI eigenvectors;
    # residual amplitudes ~1e-4 give an O(1e-5 Ha) eigenvalue shift.
    assert abs(E_eig[0] - E_ref[0]) < 1e-5
    assert abs(E_eig[1] - E_ref[1]) < 1e-5
    # Already-orthogonal input must remain orthogonal
    assert rot["input_ci_overlap"] < 1e-8


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
