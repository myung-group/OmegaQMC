"""Reference wavefunctions of the compressed-sensing workflow.

FCI reference, natural orbitals and candidate sets, and the
active-space CASCI reference for large bases.
"""
import numpy as np
import pytest
import math

from OmegaQMC.cs.reference import (  # noqa: E402
    K_eff,
    K_eff_table,
    build_h4_square,
    compute_fci_reference,
    max_c_corr,
    compute_casci_reference,
)
from OmegaQMC.utils import Mole_custom  # noqa: E402


# ---------------------------------------------------------------------------
# From test_cs_reference.py
#
# Tests for the FCI reference, natural orbital extractor, and candidate set
# used as ground truth in the H4-square compressed-sensing experiment.
#
# These tests use PySCF; they are marked ``slow`` because each FCI call costs
# a few hundred milliseconds.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def h4_eq_sto3g():
    return compute_fci_reference(
        build_h4_square(R=1.0, basis="sto-3g"), n_alpha=2, n_beta=2,
    )


@pytest.fixture(scope="module")
def h4_stretched_sto3g():
    return compute_fci_reference(
        build_h4_square(R=2.5, basis="sto-3g"), n_alpha=2, n_beta=2,
    )


@pytest.mark.slow
def test_h4_sto3g_variational(h4_eq_sto3g):
    assert h4_eq_sto3g["E_HF"] >= h4_eq_sto3g["E_FCI"] - 1e-10


@pytest.mark.slow
def test_h4_natural_orbital_trace(h4_eq_sto3g):
    occ = h4_eq_sto3g["occ_no"]
    assert abs(float(np.sum(occ)) - 4.0) < 1e-6


@pytest.mark.slow
def test_h4_natural_orbital_order(h4_eq_sto3g):
    occ = h4_eq_sto3g["occ_no"]
    assert np.all(np.diff(occ) <= 1e-10)


@pytest.mark.slow
def test_h4_natural_orbital_bounded(h4_eq_sto3g):
    occ = h4_eq_sto3g["occ_no"]
    assert occ.min() > -1e-8
    assert occ.max() < 2.0 + 1e-8


@pytest.mark.slow
def test_h4_ci_normalized(h4_eq_sto3g):
    norm = sum(c * c for c in h4_eq_sto3g["ci_dict"].values())
    assert abs(norm - 1.0) < 1e-6


@pytest.mark.slow
def test_h4_reference_is_largest_coefficient(h4_eq_sto3g):
    ci = h4_eq_sto3g["ci_dict"]
    ref = h4_eq_sto3g["reference_det"]
    c_ref = abs(ci[ref])
    for k, v in ci.items():
        assert abs(v) <= c_ref + 1e-10


@pytest.mark.slow
def test_h4_K_eff_monotone_in_eta(h4_stretched_sto3g):
    ci = h4_stretched_sto3g["ci_dict"]
    K1 = K_eff(ci, 1e-1)
    K2 = K_eff(ci, 1e-2)
    K3 = K_eff(ci, 1e-4)
    assert K3 >= K2 >= K1


@pytest.mark.slow
def test_h4_K_eff_table_keys(h4_eq_sto3g):
    table = K_eff_table(h4_eq_sto3g["ci_dict"])
    assert set(table.keys()) == {1e-2, 1e-3, 1e-4}
    for v in table.values():
        assert isinstance(v, int)
        assert v >= 0


@pytest.mark.slow
def test_h4_stretched_more_correlated_than_equilibrium(
    h4_eq_sto3g, h4_stretched_sto3g
):
    c_eq = abs(h4_eq_sto3g["ci_dict"][h4_eq_sto3g["reference_det"]])
    c_st = abs(
        h4_stretched_sto3g["ci_dict"][h4_stretched_sto3g["reference_det"]]
    )
    assert c_st < c_eq, (
        f"stretched reference weight {c_st} should be smaller than "
        f"equilibrium {c_eq} (more multireference character)"
    )


@pytest.mark.slow
def test_h4_candidate_set_starts_with_reference(h4_eq_sto3g):
    sset = h4_eq_sto3g["candidate_set"]
    assert sset[0] == h4_eq_sto3g["reference_det"]


@pytest.mark.slow
def test_h4_candidate_set_descending_by_abs(h4_eq_sto3g):
    sset = h4_eq_sto3g["candidate_set"]
    ci = h4_eq_sto3g["ci_dict"]
    coeffs = [abs(ci[k]) for k in sset]
    assert all(coeffs[i] >= coeffs[i + 1] - 1e-12
               for i in range(len(coeffs) - 1))


@pytest.mark.slow
def test_h4_max_c_corr_below_reference(h4_eq_sto3g):
    ci = h4_eq_sto3g["ci_dict"]
    ref = h4_eq_sto3g["reference_det"]
    assert max_c_corr(ci, ref) <= abs(ci[ref])


@pytest.mark.slow
def test_h4_ci_dict_filter_tol(h4_eq_sto3g):
    """candidate_tol postcondition: every retained coefficient exceeds tol,
    and the retained subset captures most of the CI mass."""
    ci = h4_eq_sto3g["ci_dict"]
    for c in ci.values():
        assert abs(c) > 1e-10
    filtered = {k: v for k, v in ci.items() if abs(v) > 1e-3}
    kept_mass = sum(v * v for v in filtered.values())
    assert kept_mass > 0.99


@pytest.mark.slow
def test_h4_string_to_occ_count():
    mol = build_h4_square(R=1.0, basis="sto-3g")
    ref = compute_fci_reference(mol)
    n_alpha, n_beta = ref["nelec"]
    for occ_a, occ_b in ref["ci_dict"].keys():
        assert len(occ_a) == n_alpha
        assert len(occ_b) == n_beta
        assert len(set(occ_a)) == n_alpha
        assert len(set(occ_b)) == n_beta


# ---------------------------------------------------------------------------
# From test_cs_casci_reference.py
#
# Tests for the active-space CASCI reference (large-basis support
# for the CS-recovery pipeline).
# ---------------------------------------------------------------------------


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
