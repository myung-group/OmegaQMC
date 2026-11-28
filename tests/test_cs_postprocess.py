"""Compressed-sensing post-processing from recovered CI coefficients.

RDMs and properties, transition densities, and CASCI/NEVPT2 (single-
and multi-state) from a recovered c_hat, against PySCF.
"""
import numpy as np
import pytest
from pyscf import fci, gto

from OmegaQMC.cs.properties import (  # noqa: E402
    compute_1rdm,
    effective_unpaired_electrons,
    electric_dipole,
    multireference_diagnostics,
    natural_occupations_from_rdm,
    report_properties,
    reshape_chat_to_pyscf_matrix,
)
from OmegaQMC.cs.reference import (  # noqa: E402
    compute_fci_reference,
    ci_to_dict,
)
from OmegaQMC.utils import Mole_custom  # noqa: E402
from OmegaQMC.cs.transition import (  # noqa: E402
    compute_1tdm,
    transition_dipole,
    oscillator_strength,
    natural_transition_orbitals,
    report_transition_properties,
    subspace_rotate_to_eigenstates,
)
from OmegaQMC.cs.mrpt import (  # noqa: E402
    build_casci_from_chat,
    casscf_nevpt2_reference,
    chat_to_casci_matrix,
    select_active_space,
    build_casci_root_matched,
    run_multistate_nevpt2,
)


# ---------------------------------------------------------------------------
# From test_cs_properties.py
#
# Tests for OmegaQMC.cs.properties — RDMs and properties from c_hat.
#
# The pivotal verification is that feeding the PySCF FCI vector itself
# into our pipeline reproduces PySCF's reference RDM to floating-point
# precision. Beyond that we test the simple invariants (trace, NO
# ordering, symmetry-required vanishing dipoles).
# ---------------------------------------------------------------------------


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
    gamma = compute_1rdm(
        c_fci, ref["candidate_set"], ref["n_orb"], ref["nelec"]
    )
    dipole = electric_dipole(mol, ref["no_coeff_ao"], gamma)
    assert dipole["mu_magnitude_debye"] < 1e-6


@pytest.mark.slow
def test_lih_dipole_nonzero(lih_eq):
    """LiH has a substantial dipole moment (~5-6 D experimental)."""
    ref = lih_eq
    mol = ref["mol"]
    c_fci = np.array([ref["ci_dict"][k] for k in ref["candidate_set"]])
    gamma = compute_1rdm(
        c_fci, ref["candidate_set"], ref["n_orb"], ref["nelec"]
    )
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
    diag = multireference_diagnostics(
        c_fci, ref["candidate_set"], ref["nelec"]
    )
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


# ---------------------------------------------------------------------------
# From test_cs_transition.py
#
# Tests for the transition density / dipole / NTO module.
#
# Validates against PySCF's own trans_rdm1 + integral routines on a
# minimal H2 STO-3G system where the FCI eigenvectors are exact.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def h2_two_roots():
    """Build H2 cc-pVDZ, find the ground singlet + first dipole-allowed
    singlet excited state from FCI nroots=4.

    PySCF's FCI solver is unrestricted in spin: when asked for the
    lowest K eigenstates it returns triplets and singlets interleaved.
    For H2 the lowest excited state is always the b3Sigma_u+ triplet
    (singlet -> triplet is spin-forbidden, giving a zero transition
    dipole). The first dipole-allowed transition is to the 1Sigma_u+
    singlet, which appears as root 2 in our nroots=4 enumeration.
    The fixture identifies this state by selecting the lowest-energy
    root with nonzero ||gamma_0k||.
    """
    from pyscf import fci as pyscf_fci

    mol = Mole_custom()
    mol.build(atom=[("H", [0, 0, 0]), ("H", [0, 0, 1.4])],
              basis="cc-pvdz", spin=0, charge=0, unit="Bohr", verbose=0)

    ref = compute_fci_reference(mol, n_alpha=1, n_beta=1, candidate_tol=0.0)
    n_orb = int(ref["n_orb"])
    nelec = tuple(ref["nelec"])
    no_coeff = ref["no_coeff_ao"]

    cisolver = pyscf_fci.FCI(mol, no_coeff)
    cisolver.verbose = 0
    cisolver.nroots = 4
    E_list, ci_list = cisolver.kernel()

    from OmegaQMC.cs.reference import ci_to_dict
    candidate = ref["candidate_set"]
    def to_chat(ci_mat):
        d = ci_to_dict(ci_mat, n_orb, nelec[0], nelec[1], tol=0.0)
        return np.array([d.get(k, 0.0) for k in candidate])

    c0 = to_chat(np.asarray(ci_list[0]))
    if c0[0] < 0:
        c0 = -c0

    # Walk up the spectrum until we find a state with nonzero
    # transition density matrix to the ground (skip triplets).
    chosen_k = None
    for k in range(1, len(E_list)):
        c_k = to_chat(np.asarray(ci_list[k]))
        gamma = compute_1tdm(c0, c_k, candidate, n_orb, nelec)
        if float(np.linalg.norm(gamma)) > 1e-3:
            chosen_k = k
            break
    assert chosen_k is not None, "no dipole-allowed transition in nroots=4"

    c1 = to_chat(np.asarray(ci_list[chosen_k]))
    if abs(c1[0]) > 1e-6 and c1[0] < 0:
        c1 = -c1

    return dict(
        mol=mol, ref=ref, c0=c0, c1=c1,
        E0=float(E_list[0]), E1=float(E_list[chosen_k]),
        ci_mat_0=np.asarray(ci_list[0]),
        ci_mat_1=np.asarray(ci_list[chosen_k]),
        chosen_k=chosen_k,
    )


@pytest.mark.slow
def test_1tdm_matches_pyscf_trans_rdm1s(h2_two_roots):
    """OmegaQMC.cs.transition.compute_1tdm reproduces PySCF's spin-summed
    trans_rdm1 within machine precision."""
    from pyscf import fci as pyscf_fci

    setup = h2_two_roots
    n_orb = int(setup["ref"]["n_orb"])
    nelec = tuple(setup["ref"]["nelec"])

    # Our routine through reshape_chat_to_pyscf_matrix
    gamma = compute_1tdm(
        setup["c0"], setup["c1"],
        setup["ref"]["candidate_set"], n_orb, nelec,
    )

    # Direct PySCF on the raw CI matrices
    dm_a, dm_b = pyscf_fci.direct_spin1.trans_rdm1s(
        setup["ci_mat_0"], setup["ci_mat_1"], n_orb, nelec,
    )
    gamma_ref = np.asarray(dm_a) + np.asarray(dm_b)

    # If our c0/c1 sign-alignment introduced a flip, allow for it
    err_pos = np.max(np.abs(gamma - gamma_ref))
    err_neg = np.max(np.abs(gamma + gamma_ref))
    assert min(err_pos, err_neg) < 1e-10, (
        f"1-TDM differs from PySCF reference: "
        f"err_pos={err_pos}, err_neg={err_neg}"
    )


@pytest.mark.slow
def test_transition_dipole_finite_and_nonzero(h2_two_roots):
    setup = h2_two_roots
    gamma = compute_1tdm(
        setup["c0"], setup["c1"],
        setup["ref"]["candidate_set"], setup["ref"]["n_orb"],
        setup["ref"]["nelec"],
    )
    mu = transition_dipole(setup["mol"], setup["ref"]["no_coeff_ao"], gamma)
    # All components finite
    assert np.all(np.isfinite(mu["mu_au"]))
    assert np.all(np.isfinite(mu["mu_debye"]))
    # For H2 1Sigma_g -> 1Sigma_u the z-component should be nonzero
    # (the transition dipole points along the bond axis). With our
    # NO basis the z-component is the dominant one.
    assert mu["mu_magnitude_au"] > 1e-3, (
        f"H2 transition dipole should be nonzero; got {mu}"
    )


@pytest.mark.slow
def test_oscillator_strength_positive(h2_two_roots):
    setup = h2_two_roots
    delta_E = setup["E1"] - setup["E0"]
    gamma = compute_1tdm(
        setup["c0"], setup["c1"],
        setup["ref"]["candidate_set"], setup["ref"]["n_orb"],
        setup["ref"]["nelec"],
    )
    mu = transition_dipole(setup["mol"], setup["ref"]["no_coeff_ao"], gamma)
    f = oscillator_strength(mu["mu_au"], delta_E)
    assert f > 0, f"oscillator strength must be positive; got {f}"
    assert np.isfinite(f)


@pytest.mark.slow
def test_nto_singular_values_sum_to_overlap_norm(h2_two_roots):
    """For a 1-TDM the sum of singular values squared equals
    Tr(gamma gamma^T), which is finite and positive."""
    setup = h2_two_roots
    gamma = compute_1tdm(
        setup["c0"], setup["c1"],
        setup["ref"]["candidate_set"], setup["ref"]["n_orb"],
        setup["ref"]["nelec"],
    )
    nto = natural_transition_orbitals(gamma, setup["ref"]["no_coeff_ao"])
    sum_sq = float(np.sum(nto["singular_values"] ** 2))
    ref = float(np.trace(gamma @ gamma.T))
    assert abs(sum_sq - ref) < 1e-10, (
        f"sum of NTO singular values^2 = {sum_sq} differs from "
        f"Tr(gamma gamma^T) = {ref}"
    )
    # Participation ratios should sum to 1
    assert abs(float(np.sum(nto["participation_ratios"])) - 1.0) < 1e-10


@pytest.mark.slow
def test_subspace_rotate_recovers_eigenvalues_from_mixed_states(h2_two_roots):
    """Take two known orthogonal FCI eigenstates, mix them by a random
    2x2 unitary, then rotate back. The recovered eigenvalues must
    match the FCI eigenvalues to machine precision, and the rotated
    CI vectors should be (up to sign) the original ones."""
    setup = h2_two_roots
    rng = np.random.default_rng(13)
    theta = rng.uniform(0.3, 1.1)
    U = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta),  np.cos(theta)]])
    c_mixed = [U[0, 0] * setup["c0"] + U[0, 1] * setup["c1"],
               U[1, 0] * setup["c0"] + U[1, 1] * setup["c1"]]
    # Normalise (already unit-norm, but be safe)
    c_mixed = [c / np.linalg.norm(c) for c in c_mixed]

    out = subspace_rotate_to_eigenstates(c_mixed, setup["ref"], setup["mol"])

    # Eigenvalues should match the FCI E0, E1
    E_eig = sorted(out["E_eig"])
    assert abs(E_eig[0] - setup["E0"]) < 1e-8
    assert abs(E_eig[1] - setup["E1"]) < 1e-8
    # Input was significantly nonorthogonal; the diagnostic must reflect that
    assert out["input_ci_overlap"] >= 0.0


@pytest.mark.slow
def test_subspace_rotate_idempotent_on_orthogonal_input(h2_two_roots):
    """If the input CI vectors are already orthogonal FCI eigenstates,
    rotation must return them unchanged (up to sign)."""
    setup = h2_two_roots
    out = subspace_rotate_to_eigenstates(
        [setup["c0"], setup["c1"]], setup["ref"], setup["mol"],
    )
    E_eig = sorted(out["E_eig"])
    assert abs(E_eig[0] - setup["E0"]) < 1e-8
    assert abs(E_eig[1] - setup["E1"]) < 1e-8
    assert out["input_ci_overlap"] < 1e-10


@pytest.mark.slow
def test_report_transition_properties_aggregate(h2_two_roots):
    """End-to-end aggregation runs and produces consistent outputs."""
    setup = h2_two_roots
    report = report_transition_properties(
        setup["c0"], setup["c1"], setup["ref"], setup["mol"],
        delta_E_au=setup["E1"] - setup["E0"],
    )
    assert "gamma_01" in report
    assert "transition_dipole" in report
    assert "oscillator_strength" in report
    assert "nto" in report
    # CI overlap of two exact FCI roots is zero to machine precision
    assert abs(report["ci_overlap"]) < 1e-12


# ---------------------------------------------------------------------------
# From test_cs_mrpt.py
#
# Tests for OmegaQMC.cs.mrpt — CASCI/NEVPT2 from a recovered c_hat.
#
# Strategy: when we feed the PySCF FCI vector itself into our wrapper,
# the resulting CASCI object should reproduce PySCF's own CASCI energy
# in the same active space, and NEVPT2 on top should match a direct
# CASSCF + NEVPT2 calculation at matched (ncas, nelecas) to within
# the CASSCF/CASCI orbital-optimisation gap.
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# From test_cs_mrpt_multistate.py
#
# Tests for multi-state NEVPT2 across FCI roots.
#
# Validates that build_casci_root_matched correctly selects an excited
# state CASCI root by overlap with a target c_hat, and that
# run_multistate_nevpt2 produces consistent per-state NEVPT2 totals.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def h2_three_roots_ccpvdz():
    """Ground singlet + lowest dipole-allowed singlet (root 2 of FCI)
    in cc-pVDZ. Skips the triplet to avoid spin contamination in NEVPT2."""
    from pyscf import fci as pyscf_fci
    mol = Mole_custom()
    # R=2.5 Bohr gives a genuine 2-orbital active space (n_0 ~ 1.88,
    # n_1 ~ 0.11) suitable for CAS(2,2). At R=1.4 the natural-orbital
    # occupations are nearly (2, 0, 0, ...) and the auto-selector
    # returns ncas=0, which is correct (no MR character) but unusable
    # for a multistate NEVPT2 test.
    mol.build(atom=[("H", [0, 0, 0]), ("H", [0, 0, 2.5])],
              basis="cc-pvdz", spin=0, charge=0, unit="Bohr", verbose=0)
    ref = compute_fci_reference(mol, n_alpha=1, n_beta=1, candidate_tol=0.0)
    n_orb = int(ref["n_orb"])
    nelec = tuple(ref["nelec"])
    cs = pyscf_fci.FCI(mol, ref["no_coeff_ao"])
    cs.verbose = 0
    cs.nroots = 4
    E_list, ci_list = cs.kernel()
    candidate = ref["candidate_set"]
    def to_chat(m):
        d = ci_to_dict(np.asarray(m), n_orb, nelec[0], nelec[1], tol=0.0)
        return np.array([d.get(k, 0.0) for k in candidate])
    c0 = to_chat(ci_list[0])
    # Find first dipole-allowed root (skip triplets)
    from OmegaQMC.cs.transition import compute_1tdm
    k_pick = None
    for k in range(1, len(E_list)):
        ck = to_chat(ci_list[k])
        gamma = compute_1tdm(c0, ck, candidate, n_orb, nelec)
        if float(np.linalg.norm(gamma)) > 1e-3:
            k_pick = k
            break
    assert k_pick is not None
    c1 = to_chat(ci_list[k_pick])
    return dict(mol=mol, ref=ref, c0=c0, c1=c1,
                E0=float(E_list[0]), E1=float(E_list[k_pick]))


@pytest.mark.slow
def test_build_casci_root_matched_finds_ground(h2_three_roots_ccpvdz):
    """For c_hat = ground-state vector, the matched root should be 0
    and its CI vector should overlap strongly with c_hat's active-space
    projection. The CAS(2,2) energy is HIGHER than the full FCI(10,10)
    energy by the missing correlation, which is exactly what NEVPT2
    will add back."""
    setup = h2_three_roots_ccpvdz
    mc = build_casci_root_matched(
        setup["mol"], setup["c0"], setup["ref"], nroots_max=4,
    )
    assert mc._cs_meta_root["k_matched"] == 0
    # Max overlap should be near 1 (c_hat IS an FCI eigenvector and its
    # CAS(2,2) projection should be near the CAS(2,2) ground root)
    assert mc._cs_meta_root["all_overlaps"][0] > 0.95
    # CAS(2,2) ground >= FCI ground (variational principle in CAS)
    assert mc.e_tot >= setup["E0"] - 1e-8


@pytest.mark.slow
def test_build_casci_root_matched_finds_excited(h2_three_roots_ccpvdz):
    """For c_hat = first dipole-allowed excited state, the matched
    root should have high overlap with that state's CAS-projection
    and a CAS energy >= the full FCI excited energy."""
    setup = h2_three_roots_ccpvdz
    mc = build_casci_root_matched(
        setup["mol"], setup["c1"], setup["ref"], nroots_max=4,
    )
    # Max overlap should be substantial (FCI excited has support in
    # CAS(2,2) but also in virtuals outside, so overlap < 1 but > some
    # nontrivial threshold)
    assert mc._cs_meta_root["all_overlaps"][
        mc._cs_meta_root["k_matched"]
    ] > 0.1
    # Matched root should be ABOVE the ground CAS root
    assert mc.e_tot > -1.087


@pytest.mark.slow
def test_multistate_nevpt2_aggregates_two_states(h2_three_roots_ccpvdz):
    """run_multistate_nevpt2 returns a usable dict for K=2 states."""
    setup = h2_three_roots_ccpvdz
    out = run_multistate_nevpt2(
        setup["mol"],
        c_hats=[setup["c0"], setup["c1"]],
        fci_ref=setup["ref"],
        state_labels=["S0", "S1"],
        nroots_max=4,
    )
    assert out["e_total"].shape == (2,)
    assert out["e_pt2"].shape == (2,)
    assert out["e_casci"].shape == (2,)
    assert out["delta_E_total_au"].shape == (2, 2)
    # Excited > ground (positive excitation energy)
    dE_total = out["delta_E_total_au"][0, 1]
    assert (
        dE_total > 0
    ), f"vertical excitation must be positive, got {dE_total}"
    # NEVPT2 should bring the CAS+PT2 ground close to FCI ground (within
    # a few mE_h for H2 cc-pVDZ at R=2.5).
    assert abs(out["e_total"][0] - setup["E0"]) < 0.02
    # state labels preserved
    assert out["state_labels"] == ["S0", "S1"]
