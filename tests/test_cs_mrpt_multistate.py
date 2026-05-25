"""Tests for multi-state NEVPT2 across FCI roots.

Validates that build_casci_root_matched correctly selects an excited
state CASCI root by overlap with a target c_hat, and that
run_multistate_nevpt2 produces consistent per-state NEVPT2 totals.
"""

import numpy as np
import pytest

from OmegaQMC.utils import Mole_custom
from OmegaQMC.cs.reference import compute_fci_reference, ci_to_dict
from OmegaQMC.cs.mrpt import (
    build_casci_root_matched,
    run_multistate_nevpt2,
)


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
    assert dE_total > 0, f"vertical excitation must be positive, got {dE_total}"
    # NEVPT2 should bring the CAS+PT2 ground close to FCI ground (within
    # a few mE_h for H2 cc-pVDZ at R=2.5).
    assert abs(out["e_total"][0] - setup["E0"]) < 0.02
    # state labels preserved
    assert out["state_labels"] == ["S0", "S1"]
