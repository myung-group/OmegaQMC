"""Tests for the transition density / dipole / NTO module.

Validates against PySCF's own trans_rdm1 + integral routines on a
minimal H2 STO-3G system where the FCI eigenvectors are exact.
"""

import numpy as np
import pytest

from OmegaQMC.utils import Mole_custom
from OmegaQMC.cs.reference import compute_fci_reference
from OmegaQMC.cs.properties import reshape_chat_to_pyscf_matrix
from OmegaQMC.cs.transition import (
    compute_1tdm,
    transition_dipole,
    oscillator_strength,
    natural_transition_orbitals,
    report_transition_properties,
)


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
