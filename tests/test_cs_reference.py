"""
Tests for the FCI reference, natural orbital extractor, and candidate set
used as ground truth in the H4-square compressed-sensing experiment.

These tests use PySCF; they are marked ``slow`` because each FCI call costs
a few hundred milliseconds.
"""

import numpy as np
import pytest

from OmegaQMC.cs.reference import (
    K_eff,
    K_eff_table,
    build_h4_square,
    ci_to_dict,
    compute_fci_reference,
    max_c_corr,
    ordered_candidate_set,
    reference_determinant,
)


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
    c_st = abs(h4_stretched_sto3g["ci_dict"][h4_stretched_sto3g["reference_det"]])
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
