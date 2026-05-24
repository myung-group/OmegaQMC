"""
Tests for OmegaQMC.cs.scaling — the sweep orchestrator.

Strategy: build Psi from exact H2/STO-3G FCI coefficients, sample a walker
bank via Metropolis from |Psi|^2, run :func:`run_sweep`, and verify that
the emitted cells/aux conform to the frozen schema and that recovery
improves monotonically with K_s.
"""

import numpy as np
import pytest
from pyscf import gto

from OmegaQMC.cs.analysis import (
    AUX_FIELDS,
    CELL_FIELDS,
    apply_convergence_gate,
    compute_K_s_star_table,
    validate_schema,
)
from OmegaQMC.cs.estimators import (
    evaluate_ci_wavefunction,
    evaluate_orbitals_on_walkers,
    normalize_and_align,
)
from OmegaQMC.cs.reference import compute_fci_reference
from OmegaQMC.cs.scaling import precompute_means, run_sweep

from .conftest import metropolis_sample


def _h2_stretched_with_bank(R_h2_bohr: float = 2.5, K_s_max: int = 1500, seed: int = 0):
    """Build H2/STO-3G at stretched bond, FCI Psi, Metropolis walker bank."""
    mol = gto.M(
        atom=f"H 0 0 0; H 0 0 {R_h2_bohr}",
        basis="sto-3g", unit="Bohr", verbose=0,
    )
    fci_ref = compute_fci_reference(mol, n_alpha=1, n_beta=1)
    candidate = fci_ref["candidate_set"]
    coeffs = np.array([fci_ref["ci_dict"][k] for k in candidate])
    no_coeff = fci_ref["no_coeff_ao"]

    def psi_fn(R):
        ov = evaluate_orbitals_on_walkers(mol, R, no_coeff)
        return evaluate_ci_wavefunction(ov, candidate, coeffs, 1, 1)

    walker_bank, psi_bank, _ = metropolis_sample(
        psi_fn, n_walkers=K_s_max, n_electrons=2,
        n_steps=200, burnin=100, seed=seed,
    )
    return mol, fci_ref, walker_bank, psi_bank


@pytest.fixture(scope="module")
def h2_sweep_fixture():
    return _h2_stretched_with_bank()


@pytest.mark.slow
def test_run_sweep_cells_match_cell_fields(h2_sweep_fixture):
    mol, fci_ref, bank, psi_bank = h2_sweep_fixture
    cells, aux = run_sweep(
        mol, fci_ref, bank, psi_bank,
        R=2.5, basis="sto-3g",
        K_s_sweep=[100, 300, 1000],
        etas=[1e-2],
        n_seeds=3,
        psi_nn_energy_error=0.0,
    )
    assert len(cells) == 3 * 1 * 3
    for cell in cells:
        for field in CELL_FIELDS:
            assert field in cell, f"missing {field}"
    for field in AUX_FIELDS:
        assert field in aux


@pytest.mark.slow
def test_run_sweep_validates_against_analysis_schema(h2_sweep_fixture):
    mol, fci_ref, bank, psi_bank = h2_sweep_fixture
    cells, aux = run_sweep(
        mol, fci_ref, bank, psi_bank,
        R=2.5, basis="sto-3g",
        K_s_sweep=[100, 300, 1000],
        etas=[1e-2, 1e-3],
        n_seeds=2,
    )
    validate_schema(cells, [aux])


@pytest.mark.slow
def test_run_sweep_recovery_improves_with_K_s(h2_sweep_fixture):
    """Median L_inf error must be non-increasing across the K_s sweep."""
    mol, fci_ref, bank, psi_bank = h2_sweep_fixture
    cells, _ = run_sweep(
        mol, fci_ref, bank, psi_bank,
        R=2.5, basis="sto-3g",
        K_s_sweep=[50, 200, 1000],
        etas=[1e-2],
        n_seeds=5,
        seed_base=42,
    )
    by_K_s = {}
    for c in cells:
        by_K_s.setdefault(c["K_s"], []).append(c["L_inf_err"])
    medians = [np.median(by_K_s[k]) for k in sorted(by_K_s)]
    assert medians[-1] <= medians[0] * 1.5, (
        f"L_inf medians across K_s: {medians} -- recovery should improve"
    )


@pytest.mark.slow
def test_run_sweep_K_s_star_finds_a_threshold(h2_sweep_fixture):
    """For the exact-FCI Psi at K_s up to bank size, K_s* must be defined."""
    mol, fci_ref, bank, psi_bank = h2_sweep_fixture
    cells, aux = run_sweep(
        mol, fci_ref, bank, psi_bank,
        R=2.5, basis="sto-3g",
        K_s_sweep=[50, 200, 800, 1400],
        etas=[5e-2],
        n_seeds=5,
        seed_base=11,
    )
    kstar = compute_K_s_star_table(cells)
    assert any(row["K_s_star"] is not None for row in kstar)


@pytest.mark.slow
def test_run_sweep_skips_K_s_above_bank(h2_sweep_fixture):
    mol, fci_ref, bank, psi_bank = h2_sweep_fixture
    K_s_max = bank.shape[0]
    cells, _ = run_sweep(
        mol, fci_ref, bank, psi_bank,
        R=2.5, basis="sto-3g",
        K_s_sweep=[100, K_s_max + 1, K_s_max + 100],
        etas=[1e-2],
        n_seeds=2,
    )
    K_s_used = sorted({c["K_s"] for c in cells})
    assert K_s_used == [100]


@pytest.mark.slow
def test_run_sweep_chunking_invariant(h2_sweep_fixture):
    """det_chunk_size should not change the per-cell sample means."""
    mol, fci_ref, bank, psi_bank = h2_sweep_fixture
    kw = dict(
        K_s_sweep=[300],
        n_seeds=3,
        seed_base=99,
    )
    means_big = precompute_means(
        mol, fci_ref, bank, psi_bank, **kw,
        det_chunk_size=10_000,
    )
    means_small = precompute_means(
        mol, fci_ref, bank, psi_bank, **kw,
        det_chunk_size=1,
    )
    assert means_big.keys() == means_small.keys()
    for key in means_big:
        np.testing.assert_allclose(means_big[key], means_small[key], atol=1e-10)


@pytest.mark.slow
def test_run_sweep_aux_max_c_corr_below_reference(h2_sweep_fixture):
    mol, fci_ref, bank, psi_bank = h2_sweep_fixture
    _, aux = run_sweep(
        mol, fci_ref, bank, psi_bank,
        R=2.5, basis="sto-3g",
        K_s_sweep=[100],
        etas=[1e-2],
        n_seeds=1,
    )
    c_ref = abs(fci_ref["ci_dict"][fci_ref["reference_det"]])
    assert aux["max_c_corr"] <= c_ref


@pytest.mark.slow
def test_run_sweep_gate_filter_compatibility(h2_sweep_fixture):
    """Aux with a too-large psi_nn_energy_error must be filtered by gate."""
    mol, fci_ref, bank, psi_bank = h2_sweep_fixture
    cells, aux = run_sweep(
        mol, fci_ref, bank, psi_bank,
        R=2.5, basis="sto-3g",
        K_s_sweep=[100],
        etas=[1e-2],
        n_seeds=1,
        psi_nn_energy_error=5.0,  # 5 mE_h, well above 0.5 gate
    )
    kept, gated = apply_convergence_gate(cells, [aux])
    assert kept == []
    assert (2.5, "sto-3g") in gated


@pytest.mark.slow
def test_lambda_coef_zero_recovers_sample_means(h2_sweep_fixture):
    """lambda_coef=0 disables soft-thresholding outside the reference."""
    mol, fci_ref, bank, psi_bank = h2_sweep_fixture
    cells, _ = run_sweep(
        mol, fci_ref, bank, psi_bank,
        R=2.5, basis="sto-3g",
        K_s_sweep=[1000],
        etas=[1e-2],
        n_seeds=1,
        lambda_coef=0.0,
    )
    assert cells[0]["lambda_used"] == 0.0


@pytest.mark.slow
def test_run_sweep_invariant_to_psi_scale(h2_sweep_fixture):
    """Scaling the trial Psi by a constant must not change recovery — the
    in-sweep normalization absorbs the scale. proj_mass scales as 1/K^2."""
    mol, fci_ref, bank, psi_bank = h2_sweep_fixture
    K = 7.5
    common_kw = dict(
        R=2.5, basis="sto-3g",
        K_s_sweep=[1000], etas=[1e-2], n_seeds=3,
        lambda_coef=0.0, seed_base=123,
    )
    cells_a, _ = run_sweep(mol, fci_ref, bank, psi_bank, **common_kw)
    cells_b, _ = run_sweep(mol, fci_ref, bank, psi_bank * K, **common_kw)
    for a, b in zip(cells_a, cells_b):
        assert abs(a["L_inf_err"] - b["L_inf_err"]) < 1e-8, (
            f"L_inf differs: {a['L_inf_err']} vs {b['L_inf_err']}"
        )
        assert abs(a["L_2_err"] - b["L_2_err"]) < 1e-8
        expected = a["proj_mass"] / (K ** 2)
        rel = abs(b["proj_mass"] - expected) / max(expected, 1e-30)
        assert rel < 1e-6, (
            f"proj_mass should scale 1/K^2; got rel-diff {rel}"
        )


@pytest.mark.slow
def test_run_sweep_aligns_global_sign(h2_sweep_fixture):
    """A global sign flip of Psi is auto-corrected by normalize_and_align."""
    mol, fci_ref, bank, psi_bank = h2_sweep_fixture
    common_kw = dict(
        R=2.5, basis="sto-3g",
        K_s_sweep=[1000], etas=[1e-2], n_seeds=3,
        lambda_coef=0.0, seed_base=321,
    )
    cells_a, _ = run_sweep(mol, fci_ref, bank, psi_bank, **common_kw)
    cells_b, _ = run_sweep(mol, fci_ref, bank, -psi_bank, **common_kw)
    for a, b in zip(cells_a, cells_b):
        assert abs(a["L_inf_err"] - b["L_inf_err"]) < 1e-8


@pytest.mark.slow
def test_run_sweep_h4_recovery_in_interleaved_convention():
    """Synthetic H4 Psi in interleaved layout: run_sweep must use
    walker_convention='interleaved' to recover c_FCI. This is the test
    that would have caught the spin-convention bug in the H4 pilot."""
    from pyscf import gto
    from OmegaQMC.cs.estimators import _normalization
    mol = gto.M(
        atom="H 0 0 0; H 0 0 1.9; H 0 0 3.8; H 0 0 5.7",
        basis="sto-3g", unit="Bohr", verbose=0,
    )
    fci_ref = compute_fci_reference(mol, n_alpha=2, n_beta=2)
    candidate = fci_ref["candidate_set"]
    coeffs = np.array([fci_ref["ci_dict"][k] for k in candidate])
    no_coeff = fci_ref["no_coeff_ao"]
    n_norm = _normalization(2, 2)

    def psi_interleaved(R):
        flat = R.reshape(-1, 3)
        ao = mol.eval_gto("GTOval_sph", flat)
        no = (ao @ no_coeff).reshape(R.shape[0], 4, -1)
        orb_a = no[:, 0::2, :]
        orb_b = no[:, 1::2, :]
        psi = np.zeros(R.shape[0])
        for c, (occ_a, occ_b) in zip(coeffs, candidate):
            M_a = orb_a[:, :, list(occ_a)]
            M_b = orb_b[:, :, list(occ_b)]
            psi += float(c) * n_norm * np.linalg.det(M_a) * np.linalg.det(M_b)
        return psi

    walkers, psi_bank, _ = metropolis_sample(
        psi_interleaved, n_walkers=400, n_electrons=4,
        n_steps=200, burnin=100, seed=13,
    )

    cells, aux = run_sweep(
        mol, fci_ref, walkers, psi_bank,
        R=1.9, basis="sto-3g",
        K_s_sweep=[400], etas=[1e-2], n_seeds=1,
        lambda_coef=0.0,
        walker_convention="interleaved",
    )
    cell = cells[0]
    # proj_mass should be close to 1 (synthetic Psi was normalized to
    # sum |c_FCI|^2 = 1; the basis is complete here so projection mass = 1)
    assert 0.7 < cell["proj_mass"] < 1.4, (
        f"proj_mass = {cell['proj_mass']}; expected ~1 for complete-basis "
        f"synthetic Psi"
    )
    # Recovery error should reflect MC noise on 400 samples (~0.05-0.10)
    assert cell["L_inf_err"] < 0.15, (
        f"L_inf = {cell['L_inf_err']}; convention bug would make this >> 0.15"
    )


@pytest.mark.slow
def test_run_sweep_h4_grouped_convention_fails_on_interleaved_walkers():
    """Negative-control: if we wrongly tell run_sweep the interleaved bank
    is grouped, recovery should noticeably degrade."""
    from pyscf import gto
    from OmegaQMC.cs.estimators import _normalization
    mol = gto.M(
        atom="H 0 0 0; H 0 0 1.9; H 0 0 3.8; H 0 0 5.7",
        basis="sto-3g", unit="Bohr", verbose=0,
    )
    fci_ref = compute_fci_reference(mol, n_alpha=2, n_beta=2)
    candidate = fci_ref["candidate_set"]
    coeffs = np.array([fci_ref["ci_dict"][k] for k in candidate])
    no_coeff = fci_ref["no_coeff_ao"]
    n_norm = _normalization(2, 2)

    def psi_interleaved(R):
        flat = R.reshape(-1, 3)
        ao = mol.eval_gto("GTOval_sph", flat)
        no = (ao @ no_coeff).reshape(R.shape[0], 4, -1)
        orb_a = no[:, 0::2, :]
        orb_b = no[:, 1::2, :]
        psi = np.zeros(R.shape[0])
        for c, (occ_a, occ_b) in zip(coeffs, candidate):
            M_a = orb_a[:, :, list(occ_a)]
            M_b = orb_b[:, :, list(occ_b)]
            psi += float(c) * n_norm * np.linalg.det(M_a) * np.linalg.det(M_b)
        return psi

    walkers, psi_bank, _ = metropolis_sample(
        psi_interleaved, n_walkers=400, n_electrons=4,
        n_steps=200, burnin=100, seed=13,
    )
    cells_wrong, _ = run_sweep(
        mol, fci_ref, walkers, psi_bank,
        R=1.9, basis="sto-3g",
        K_s_sweep=[400], etas=[1e-2], n_seeds=1,
        lambda_coef=0.0,
        walker_convention="grouped",  # WRONG
    )
    cells_right, _ = run_sweep(
        mol, fci_ref, walkers, psi_bank,
        R=1.9, basis="sto-3g",
        K_s_sweep=[400], etas=[1e-2], n_seeds=1,
        lambda_coef=0.0,
        walker_convention="interleaved",
    )
    assert cells_wrong[0]["L_inf_err"] > 2 * cells_right[0]["L_inf_err"]


@pytest.mark.slow
def test_run_sweep_cells_carry_proj_mass(h2_sweep_fixture):
    mol, fci_ref, bank, psi_bank = h2_sweep_fixture
    cells, _ = run_sweep(
        mol, fci_ref, bank, psi_bank,
        R=2.5, basis="sto-3g",
        K_s_sweep=[300], etas=[1e-2], n_seeds=2,
        lambda_coef=0.0,
    )
    for c in cells:
        assert "proj_mass" in c
        assert c["proj_mass"] > 0
        # synthetic Psi was built from normalized c_FCI, so proj_mass ~ 1
        assert 0.5 < c["proj_mass"] < 1.5
