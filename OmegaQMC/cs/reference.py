"""
FCI reference, natural orbitals, and candidate set for the H4-square
compressed-sensing scaling experiment.

This module is the data-collection side's source of ground truth. It computes:

- the FCI ground-state energy and CI vector,
- natural orbitals from the FCI 1-RDM,
- the FCI CI vector re-expressed in the natural-orbital basis,
- the candidate set ordered with the reference determinant at index 0,
- summary statistics consumed by :mod:`OmegaQMC.cs.analysis`.

Stable PySCF API only (gto, scf.RHF, fci.FCI, fci.cistring).
"""

from typing import Mapping, Optional, Sequence

import numpy as np

from pyscf import gto, scf, fci
from pyscf.fci import cistring


def build_h4_square(R: float, basis: str = "cc-pVDZ", unit: str = "Angstrom"):
    """H4 with the four atoms at the corners of a square of side ``R``."""
    atoms = (
        f"H  0.0  0.0  0.0\n"
        f"H  {R}  0.0  0.0\n"
        f"H  {R}  {R}  0.0\n"
        f"H  0.0  {R}  0.0\n"
    )
    return gto.M(atom=atoms, basis=basis, unit=unit, symmetry=False, verbose=0)


def run_rhf(mol):
    """RHF; returns ``(E_HF, mo_coeff)``."""
    mf = scf.RHF(mol)
    mf.verbose = 0
    mf.run()
    return float(mf.e_tot), np.asarray(mf.mo_coeff)


def run_fci(mol, mo_coeff, force_singlet: bool = True):
    """FCI in the basis spanned by ``mo_coeff``.

    Returns ``(E_FCI, ci_matrix, cisolver)``. The CI matrix is indexed by
    (alpha-string-index, beta-string-index).

    For closed-shell molecules (equal alpha/beta count) PySCF's auto
    ``fci.FCI`` can dispatch to ``direct_spin1`` whose Ms=0 sector
    contains BOTH singlets and triplets. Davidson then converges to the
    Ms=0 lowest root regardless of S², which for near-degenerate systems
    (e.g. H4 square at stretched geometry) may be the wrong root and
    differs between orbital bases. With ``force_singlet=True`` (default)
    we explicitly use ``direct_spin0`` which spans only singlet wave
    functions — guarantees a stable, basis-independent singlet root.
    """
    if force_singlet and mol.spin == 0:
        from pyscf.fci import direct_spin0
        cisolver = direct_spin0.FCI(mol)
        cisolver.mo_coeff = mo_coeff
    else:
        cisolver = fci.FCI(mol, mo_coeff)
    cisolver.verbose = 0
    if force_singlet and mol.spin == 0:
        from pyscf import ao2mo
        h1 = mo_coeff.T @ scf.hf.get_hcore(mol) @ mo_coeff
        norb = mo_coeff.shape[1]
        eri = ao2mo.kernel(mol, mo_coeff, compact=False).reshape(
            norb, norb, norb, norb,
        )
        nelec = mol.nelec
        E_elec, ci = cisolver.kernel(h1, eri, norb, nelec)
        E = float(E_elec + mol.energy_nuc())
    else:
        E, ci = cisolver.kernel()
        E = float(E)
    return E, np.asarray(ci), cisolver


def natural_orbitals(cisolver, ci, mo_coeff, nelec):
    """Diagonalize the spin-summed 1-RDM produced by ``cisolver``.

    Returns ``(occ_numbers_desc, no_coeff_in_ao_basis)``.
    """
    norb = mo_coeff.shape[1]
    dm1 = cisolver.make_rdm1(ci, norb, nelec)
    dm1 = 0.5 * (dm1 + dm1.T)
    occ, U = np.linalg.eigh(dm1)
    order = np.argsort(-occ)
    occ = occ[order]
    U = U[:, order]
    no_coeff = np.asarray(mo_coeff) @ U
    return occ, no_coeff


def _string_to_occ(s: int, norb: int):
    return tuple(k for k in range(norb) if (s >> k) & 1)


def ci_to_dict(
    ci: np.ndarray,
    norb: int,
    n_alpha: int,
    n_beta: int,
    tol: float = 0.0,
) -> dict:
    """Materialize the CI matrix as ``{(occ_alpha, occ_beta): coeff}``.

    Determinants with ``|coeff| <= tol`` are dropped.
    """
    strings_a = cistring.make_strings(range(norb), n_alpha)
    strings_b = cistring.make_strings(range(norb), n_beta)
    out: dict = {}
    for i, sa in enumerate(strings_a):
        occ_a = _string_to_occ(int(sa), norb)
        row = ci[i]
        for j, sb in enumerate(strings_b):
            c = float(row[j])
            if abs(c) > tol:
                occ_b = _string_to_occ(int(sb), norb)
                out[(occ_a, occ_b)] = c
    return out


def reference_determinant(ci_dict: Mapping):
    """The determinant with the largest |coefficient|."""
    return max(ci_dict, key=lambda k: abs(ci_dict[k]))


def ordered_candidate_set(ci_dict: Mapping, ref_det) -> list:
    """List of (occ_a, occ_b) tuples; reference at index 0, then descending |c|."""
    others = sorted(
        (k for k in ci_dict if k != ref_det),
        key=lambda k: -abs(ci_dict[k]),
    )
    return [ref_det] + others


def K_eff(ci_dict: Mapping, eta: float) -> int:
    """Number of CI coefficients strictly above ``eta``."""
    return sum(1 for v in ci_dict.values() if abs(v) > eta)


def max_c_corr(ci_dict: Mapping, ref_det) -> float:
    """Largest |coefficient| over determinants other than the reference."""
    return max((abs(v) for k, v in ci_dict.items() if k != ref_det), default=0.0)


def K_eff_table(ci_dict: Mapping, etas: Sequence[float] = (1e-2, 1e-3, 1e-4)) -> dict:
    return {float(eta): K_eff(ci_dict, eta) for eta in etas}


def _casci_to_dict_with_core(
    ci_active: np.ndarray,
    ncore: int,
    ncas: int,
    nelecas: tuple,
    tol: float = 0.0,
) -> dict:
    """Materialize a CASCI CI matrix as ``{(occ_alpha, occ_beta): coeff}``
    using *full-orbital* occupations including the frozen core.

    The CASCI solver indexes determinants by occupations within the
    active space ``[0, ncas)``. To make the result a drop-in replacement
    for ``ci_to_dict`` (so downstream evaluators in
    :mod:`OmegaQMC.cs.estimators` work unchanged), we shift the active
    indices by ``ncore`` and prepend ``(0, 1, ..., ncore - 1)`` for the
    frozen-core occupation. This way the resulting determinant tuple
    addresses orbitals in the full ``ncore + ncas + nvirt`` space and
    can be passed straight to
    :func:`OmegaQMC.cs.estimators.evaluate_ci_wavefunction` without
    knowing it came from a CASCI.
    """
    n_alpha_act, n_beta_act = nelecas
    strings_a = cistring.make_strings(range(ncas), n_alpha_act)
    strings_b = cistring.make_strings(range(ncas), n_beta_act)
    core_tuple = tuple(range(ncore))
    out: dict = {}
    for i, sa in enumerate(strings_a):
        occ_a_act = _string_to_occ(int(sa), ncas)
        full_occ_a = core_tuple + tuple(o + ncore for o in occ_a_act)
        row = ci_active[i]
        for j, sb in enumerate(strings_b):
            c = float(row[j])
            if abs(c) > tol:
                occ_b_act = _string_to_occ(int(sb), ncas)
                full_occ_b = core_tuple + tuple(o + ncore
                                                 for o in occ_b_act)
                out[(full_occ_a, full_occ_b)] = c
    return out


def compute_casci_reference(
    mol,
    ncas: int,
    nelecas: tuple,
    ncore: int = None,
    candidate_tol: float = 1e-10,
    casci_natorb: bool = True,
) -> dict:
    """Active-space CASCI reference for the CS-recovery pipeline.

    Drop-in replacement for :func:`compute_fci_reference` when the full
    FCI is intractable (large basis sets). Runs RHF, then CASCI(ncas,
    nelecas) at the HF orbitals; optionally rotates the active
    orbitals into the CASCI 1-RDM natural-orbital basis and re-runs
    CASCI once for consistency. Returns the same dict schema as
    :func:`compute_fci_reference` with two extra metadata keys
    (``ncore``, ``ncas``, ``nelecas_active``) and ``n_orb`` set to the
    full AO count so the orbital evaluator sees all orbitals.

    The CI dict uses full-orbital occupations including the frozen
    core, so downstream estimators (``evaluate_ci_wavefunction``,
    ``f_I_matrix``) work unchanged.

    Args:
        mol: PySCF molecule.
        ncas: number of active orbitals.
        nelecas: ``(n_alpha_active, n_beta_active)`` electrons in active
            space.
        ncore: number of doubly-occupied frozen-core orbitals; inferred
            from ``(n_total - sum(nelecas)) // 2`` if not given.
        candidate_tol: drop CI coefficients below this magnitude.
        casci_natorb: if True, rotate to CASCI 1-RDM natural orbitals
            and re-diagonalise (gives a more concentrated CI vector,
            matching what compute_fci_reference does in NO basis).
    """
    from pyscf import mcscf, scf
    from pyscf import fci as pyscf_fci

    n_total = int(mol.nelectron)
    if ncore is None:
        if (n_total - sum(nelecas)) % 2 != 0:
            raise ValueError(
                f"cannot infer ncore from n_total={n_total} and "
                f"nelecas={nelecas} (mismatch)"
            )
        ncore = (n_total - sum(nelecas)) // 2

    E_HF, mo_coeff_hf = run_rhf(mol)

    # First CASCI pass at HF orbitals
    mf = scf.RHF(mol).run(verbose=0)
    mc = mcscf.CASCI(mf, ncas, nelecas)
    mc.ncore = ncore
    mc.verbose = 0
    mc.kernel(mo_coeff=mo_coeff_hf)
    E_casci_hf = float(mc.e_tot)
    ci_active_hf = np.asarray(mc.ci)

    # Optionally rotate the active block to CASCI-natural orbitals
    if casci_natorb:
        # Spin-summed active-space 1-RDM
        dm1 = mc.fcisolver.make_rdm1(ci_active_hf, ncas, nelecas)
        dm1 = 0.5 * (dm1 + dm1.T)
        occ_act, U = np.linalg.eigh(dm1)
        order = np.argsort(-occ_act)
        occ_act = occ_act[order]
        U = U[:, order]

        # Compose: keep ncore frozen, rotate active by U, leave virtuals
        no_coeff = np.array(mo_coeff_hf, copy=True)
        active_mos = mo_coeff_hf[:, ncore:ncore + ncas]
        no_coeff[:, ncore:ncore + ncas] = active_mos @ U

        # Re-run CASCI in the natural-orbital active basis
        mc2 = mcscf.CASCI(mf, ncas, nelecas)
        mc2.ncore = ncore
        mc2.verbose = 0
        mc2.kernel(mo_coeff=no_coeff)
        E_casci_no = float(mc2.e_tot)
        ci_active = np.asarray(mc2.ci)
        no_coeff_ao = no_coeff
    else:
        # Construct NO-like coefficients: only the active block diag-
        # onalised; everything else from HF
        occ_act = np.array([
            *([2.0] * ncore),
            *(np.linalg.eigvalsh(
                mc.fcisolver.make_rdm1(ci_active_hf, ncas, nelecas),
            )[::-1]),
        ])
        no_coeff_ao = np.array(mo_coeff_hf, copy=True)
        E_casci_no = E_casci_hf
        ci_active = ci_active_hf

    # Full natural-orbital occupations (core = 2, active eigenvalues, virt = 0)
    occ_no_full = np.array(
        [2.0] * ncore
        + list(np.asarray(occ_act))
        + [0.0] * (mol.nao - ncore - ncas),
    )

    nelec_total = (ncore + nelecas[0], ncore + nelecas[1])
    ci_dict = _casci_to_dict_with_core(
        ci_active, ncore, ncas, nelecas, tol=candidate_tol,
    )
    if not ci_dict:
        raise RuntimeError(
            "CASCI vector vanished under candidate_tol filter"
        )
    ref = reference_determinant(ci_dict)

    return dict(
        E_HF=float(E_HF),
        E_FCI=float(E_casci_no),  # CASCI energy in the "FCI slot"
        E_CASCI=float(E_casci_no),
        mo_coeff_hf=mo_coeff_hf,
        no_coeff_ao=no_coeff_ao,
        occ_no=occ_no_full,
        ci_dict=ci_dict,
        reference_det=ref,
        candidate_set=ordered_candidate_set(ci_dict, ref),
        n_orb=int(mol.nao),
        nelec=nelec_total,
        ncore=int(ncore),
        ncas=int(ncas),
        nelecas_active=tuple(nelecas),
    )


def compute_fci_reference(
    mol,
    n_alpha: Optional[int] = None,
    n_beta: Optional[int] = None,
    candidate_tol: float = 1e-10,
    energy_consistency_tol: float = 1e-6,
) -> dict:
    """End-to-end ground-truth pipeline.

    Steps: RHF, FCI in HF basis, natural orbitals from the FCI 1-RDM,
    FCI in the NO basis, candidate-set materialization. Verifies that the
    two FCI calculations agree on energy.
    """
    if n_alpha is None or n_beta is None:
        ne = int(mol.nelectron)
        if ne % 2 != 0 and (n_alpha is None and n_beta is None):
            raise ValueError(
                "odd electron count; pass n_alpha and n_beta explicitly"
            )
        n_alpha = ne // 2
        n_beta = ne // 2
    nelec = (int(n_alpha), int(n_beta))

    E_HF, mo_coeff = run_rhf(mol)
    E_FCI_hf, ci_hf, cisolver_hf = run_fci(mol, mo_coeff)
    occ_no, no_coeff = natural_orbitals(cisolver_hf, ci_hf, mo_coeff, nelec)
    E_FCI_no, ci_no, _ = run_fci(mol, no_coeff)

    if abs(E_FCI_hf - E_FCI_no) > energy_consistency_tol:
        raise RuntimeError(
            f"FCI energy inconsistent across bases: "
            f"HF={E_FCI_hf:.10f}, NO={E_FCI_no:.10f}"
        )

    norb = int(no_coeff.shape[1])
    ci_dict = ci_to_dict(ci_no, norb, nelec[0], nelec[1], tol=candidate_tol)
    if not ci_dict:
        raise RuntimeError("FCI vector vanished under candidate_tol filter")
    ref = reference_determinant(ci_dict)

    return dict(
        E_HF=float(E_HF),
        E_FCI=float(E_FCI_no),
        mo_coeff_hf=mo_coeff,
        no_coeff_ao=no_coeff,
        occ_no=occ_no,
        ci_dict=ci_dict,
        reference_det=ref,
        candidate_set=ordered_candidate_set(ci_dict, ref),
        n_orb=norb,
        nelec=nelec,
    )
