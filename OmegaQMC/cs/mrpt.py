"""
Multireference perturbation theory from a CS-recovered CI vector.

Takes a normalized ``c_hat`` in the natural-orbital basis (output of
:func:`OmegaQMC.cs.estimators.normalize_and_align`) and constructs a
PySCF CASCI object whose ``mo_coeff`` are the natural orbitals,
whose ``ci`` is the active-space-projected ``c_hat``, and whose
``e_tot`` is recomputed analytically. NEVPT2 (and CASPT2 via
external interface) can then be run on top of this object without
any CASSCF orbital optimisation step.

The active space is selected automatically from the natural-orbital
occupations: orbitals with occupations in ``(occ_threshold,
2-occ_threshold)`` define the active space; the remaining occupied
orbitals form the core and the remaining virtuals are excluded.

Pipeline:
  c_hat (NO basis) ----+----> active-space CI matrix
                       |        \\
  natural orbitals ----+----> CASCI(ncas, nelecas) instance
                                   |
                                   v
                          NEVPT2 (or external CASPT2/MRCI)
"""

from typing import Mapping, Sequence, Tuple

import numpy as np


def select_active_space(
    occupations: np.ndarray,
    occ_threshold: float = 0.1,
    max_ncas: int = None,
) -> Tuple[int, int]:
    """Identify (ncore, ncas) from natural-orbital occupations.

    The activity measure for orbital p is
    ``activity(p) = min(n_p, 2 - n_p)`` --- distance from the nearest
    trivial occupancy (0 or 2). Orbitals with ``activity > occ_threshold``
    are considered active. Among the remainder, those with
    ``n_p > 2 - occ_threshold`` are doubly-occupied core; those with
    ``n_p < occ_threshold`` are virtual.

    The default ``occ_threshold = 0.1`` matches common practice in
    selected-CI active-space identification.

    Returns ``(ncore, ncas)``. The caller is expected to derive
    ``nelecas`` from total electron count and ``ncore``.
    """
    occ = np.asarray(occupations)
    activity = np.minimum(occ, 2.0 - occ)
    active_mask = activity > occ_threshold
    core_mask = (~active_mask) & (occ > 2.0 - occ_threshold)
    ncore = int(np.sum(core_mask))
    ncas = int(np.sum(active_mask))
    if max_ncas is not None and ncas > max_ncas:
        active_idx = np.where(active_mask)[0]
        keep = active_idx[np.argsort(-activity[active_idx])[:max_ncas]]
        new_active = np.zeros_like(active_mask)
        new_active[keep] = True
        ncas = int(np.sum(new_active))
    return ncore, ncas


def chat_to_casci_matrix(
    c_hat: np.ndarray,
    candidate_set: Sequence,
    ncore: int,
    ncas: int,
    nelecas: Tuple[int, int],
) -> np.ndarray:
    """Project ``c_hat`` (full NO-basis CI vector) onto the active space
    and renormalise.

    Determinants outside the active space (i.e., with any occupied
    orbital index outside ``[ncore, ncore + ncas)`` or any unoccupied
    core orbital) are dropped. The surviving coefficients are placed
    at their PySCF active-space string indices and the resulting
    matrix is renormalised to unit Frobenius norm.
    """
    from pyscf.fci import cistring
    strings_a = cistring.make_strings(range(ncas), nelecas[0])
    strings_b = cistring.make_strings(range(ncas), nelecas[1])
    a_idx = {tuple(k for k in range(ncas) if (int(s) >> k) & 1): i
             for i, s in enumerate(strings_a)}
    b_idx = {tuple(k for k in range(ncas) if (int(s) >> k) & 1): j
             for j, s in enumerate(strings_b)}

    core_set = set(range(ncore))
    active_set = set(range(ncore, ncore + ncas))
    n_alpha_total = ncore + nelecas[0]
    n_beta_total = ncore + nelecas[1]

    ci_matrix = np.zeros((len(strings_a), len(strings_b)), dtype=float)
    n_kept = 0
    for c, (occ_a, occ_b) in zip(np.asarray(c_hat), candidate_set):
        occ_a_set = set(occ_a)
        occ_b_set = set(occ_b)
        if not core_set.issubset(occ_a_set):
            continue
        if not core_set.issubset(occ_b_set):
            continue
        active_a = occ_a_set - core_set
        active_b = occ_b_set - core_set
        if not active_a.issubset(active_set):
            continue
        if not active_b.issubset(active_set):
            continue
        if len(active_a) != nelecas[0] or len(active_b) != nelecas[1]:
            continue
        rel_a = tuple(sorted(o - ncore for o in active_a))
        rel_b = tuple(sorted(o - ncore for o in active_b))
        if rel_a not in a_idx or rel_b not in b_idx:
            continue
        ci_matrix[a_idx[rel_a], b_idx[rel_b]] = float(c)
        n_kept += 1

    norm = float(np.linalg.norm(ci_matrix))
    if norm > 0:
        ci_matrix /= norm
    return ci_matrix, n_kept


def build_casci_from_chat(
    mol,
    c_hat: np.ndarray,
    fci_ref: Mapping,
    ncore: int = None,
    ncas: int = None,
    nelecas: Tuple[int, int] = None,
    occ_threshold: float = 0.1,
    max_ncas: int = None,
    diagonalize: bool = True,
):
    """Construct a PySCF CASCI object whose orbitals are the natural
    orbitals from ``fci_ref`` (with the active space auto-identified
    from the recovered ``c_hat``'s 1-RDM) and whose ``ci`` is either
    (a) the variational CASCI eigenvector in that active space
    (``diagonalize=True``, default) or (b) the active-space projection
    of ``c_hat`` directly (``diagonalize=False``).

    Auto-selection uses the activity measure ``min(n_p, 2 - n_p)``
    applied to the eigenvalues of the c_hat-derived 1-RDM. Orbitals
    with activity above ``occ_threshold`` form the active space.

    The variational CASCI form (a) is required for downstream methods
    that assume a variational reference (NEVPT2, CASPT2 first-order
    interacting space, etc.). The projection form (b) is informational
    only: it shows what energy the raw recovered c_hat would have in
    the chosen active space, without further diagonalisation.

    The NN contribution in form (a) is the active-space identification:
    no CASSCF orbital-optimisation step was performed, and the active
    orbitals are read directly off the c_hat 1-RDM diagonalisation.
    """
    from pyscf import mcscf, scf
    from pyscf import fci as pyscf_fci

    mf = scf.RHF(mol).run(verbose=0)
    no_coeff = np.asarray(fci_ref["no_coeff_ao"])

    n_total = sum(fci_ref["nelec"])
    if ncore is None or ncas is None or nelecas is None:
        from OmegaQMC.cs.properties import compute_1rdm, natural_occupations_from_rdm
        gamma = compute_1rdm(c_hat, fci_ref["candidate_set"],
                             fci_ref["n_orb"], fci_ref["nelec"])
        no_occ = natural_occupations_from_rdm(gamma)
        ncore_auto, ncas_auto = select_active_space(
            no_occ, occ_threshold=occ_threshold, max_ncas=max_ncas,
        )
        if ncore is None and nelecas is None:
            ncore = ncore_auto
        if ncas is None:
            ncas = ncas_auto
        if nelecas is None:
            n_active_e = n_total - 2 * ncore
            nelecas = (n_active_e // 2 + n_active_e % 2,
                       n_active_e // 2)
        if ncore is None:
            # nelecas given but ncore not: derive ncore from the constraint
            # 2*ncore + sum(nelecas) == n_total
            ncore = (n_total - sum(nelecas)) // 2
    # Final sanity check
    if 2 * ncore + sum(nelecas) != n_total:
        raise ValueError(
            f"electron-count mismatch: 2*ncore({ncore}) + sum(nelecas)"
            f"({sum(nelecas)}) = {2*ncore + sum(nelecas)} != "
            f"n_total({n_total})"
        )

    mc = mcscf.CASCI(mf, ncas, nelecas)
    mc.ncore = ncore
    mc.mo_coeff = no_coeff

    ci_matrix, n_kept = chat_to_casci_matrix(
        c_hat, fci_ref["candidate_set"], ncore, ncas, nelecas,
    )

    if diagonalize:
        # Diagonalise the active-space Hamiltonian to get the variational
        # CASCI eigenvector at the NN-derived orbitals. This is the
        # principled reference for downstream NEVPT2/CASPT2.
        mc.verbose = 0
        mc.kernel(mo_coeff=no_coeff)
        ci_used = mc.ci
        e_used = float(mc.e_tot)
    else:
        # Use the raw c_hat projection. NEVPT2 from this reference may
        # give a positive (theoretically wrong-signed) PT2 correction
        # because the reference is not variational.
        mc.ci = ci_matrix
        h1, ecore = mc.get_h1eff()
        h2 = mc.get_h2eff()
        e_ci_active = pyscf_fci.direct_spin1.energy(
            h1, h2, ci_matrix, ncas, nelecas,
        )
        mc.e_tot = float(e_ci_active) + float(ecore)
        ci_used = ci_matrix
        e_used = mc.e_tot

    mc._cs_meta = dict(
        n_det_kept_in_active=n_kept,
        ncore=ncore, ncas=ncas, nelecas=nelecas,
        diagonalized=bool(diagonalize),
        e_chat_projection=mc.e_tot if not diagonalize else None,
    )
    return mc


def run_nevpt2(mc) -> dict:
    """Run NEVPT2 on a CASCI object and return total energies.

    Returns a dict with ``e_casci``, ``e_pt2``, ``e_total``.
    """
    from pyscf import mrpt
    nev = mrpt.NEVPT(mc)
    e_pt2 = float(nev.kernel())
    return dict(
        e_casci=float(mc.e_tot),
        e_pt2=e_pt2,
        e_total=float(mc.e_tot) + e_pt2,
    )


def build_casci_root_matched(
    mol,
    c_hat: np.ndarray,
    fci_ref: Mapping,
    ncore: int = None,
    ncas: int = None,
    nelecas: Tuple[int, int] = None,
    occ_threshold: float = 0.1,
    max_ncas: int = None,
    nroots_max: int = 8,
):
    """Variational CASCI reference matched to an excited-state ``c_hat``.

    Like :func:`build_casci_from_chat` with ``diagonalize=True``, but
    instead of returning the lowest CASCI root (which would be the
    ground state regardless of ``c_hat``), this routine

      1. Diagonalises the active-space Hamiltonian for the lowest
         ``nroots_max`` roots,
      2. Computes the overlap of each root's CI vector with the
         active-space projection of ``c_hat``,
      3. Selects the root with the largest overlap (typically the
         excited state that ``c_hat`` represents),
      4. Sets ``mc.ci``, ``mc.e_tot``, and ``mc.fcisolver.nroots = 1``
         to the matched root, so a subsequent NEVPT2 call uses the
         correct reference.

    This is the principled way to do NEVPT2 on top of a Pfau-NES (or
    any other) excited-state CI vector: the post-NES PT2 correction
    is computed at the *variational* CASCI eigenvector closest to the
    NN-derived state, not at the lowest root by default.

    Returns the matched ``mc`` (CASCI object) plus a small
    ``_cs_meta_root`` attribute on it with the overlap and root index.
    """
    from pyscf import mcscf, scf
    from pyscf import fci as pyscf_fci
    from OmegaQMC.cs.properties import (
        compute_1rdm, natural_occupations_from_rdm,
    )

    mf = scf.RHF(mol).run(verbose=0)
    no_coeff = np.asarray(fci_ref["no_coeff_ao"])

    n_total = sum(fci_ref["nelec"])
    if ncore is None or ncas is None or nelecas is None:
        gamma = compute_1rdm(c_hat, fci_ref["candidate_set"],
                             fci_ref["n_orb"], fci_ref["nelec"])
        no_occ = natural_occupations_from_rdm(gamma)
        ncore_auto, ncas_auto = select_active_space(
            no_occ, occ_threshold=occ_threshold, max_ncas=max_ncas,
        )
        if ncore is None and nelecas is None:
            ncore = ncore_auto
        if ncas is None:
            ncas = ncas_auto
        if nelecas is None:
            n_active_e = n_total - 2 * ncore
            nelecas = (n_active_e // 2 + n_active_e % 2,
                       n_active_e // 2)
        if ncore is None:
            ncore = (n_total - sum(nelecas)) // 2
    if 2 * ncore + sum(nelecas) != n_total:
        raise ValueError(
            f"electron-count mismatch: 2*ncore({ncore}) + "
            f"sum(nelecas)({sum(nelecas)}) = {2*ncore + sum(nelecas)}"
            f" != n_total({n_total})"
        )

    # Project c_hat onto the active space (for the overlap reference)
    ci_matrix_ref, _ = chat_to_casci_matrix(
        c_hat, fci_ref["candidate_set"], ncore, ncas, nelecas,
    )
    # Total CAS det count for this (ncas, nelecas) — required for nroots cap
    from pyscf.fci import cistring
    n_dets_alpha = cistring.num_strings(ncas, nelecas[0])
    n_dets_beta = cistring.num_strings(ncas, nelecas[1])
    n_dets_total = n_dets_alpha * n_dets_beta
    nroots = min(nroots_max, n_dets_total)

    mc = mcscf.CASCI(mf, ncas, nelecas)
    mc.ncore = ncore
    mc.mo_coeff = no_coeff
    mc.fcisolver.nroots = nroots
    mc.verbose = 0
    mc.kernel(mo_coeff=no_coeff)

    # mc.ci is now a list (or tuple) of nroots CI matrices; select by
    # maximum overlap with c_hat's active-space projection
    overlaps = []
    for ci_k in mc.ci:
        ovl = float(np.sum(np.asarray(ci_k) * ci_matrix_ref))
        overlaps.append(abs(ovl))
    overlaps = np.asarray(overlaps)
    k_best = int(np.argmax(overlaps))
    if overlaps[k_best] < 1e-3:
        raise RuntimeError(
            f"no CASCI root has appreciable overlap with c_hat "
            f"(best overlap = {overlaps[k_best]:.3e} at root {k_best}); "
            f"check ncore/ncas or the c_hat input"
        )

    # Collapse mc to the matched root so NEVPT2 uses it
    mc.ci = mc.ci[k_best]
    mc.e_tot = float(mc.e_tot[k_best])
    mc.fcisolver.nroots = 1
    mc._cs_meta_root = dict(
        k_matched=k_best,
        all_root_energies=[float(e) for e in mc.e_cas]
            if hasattr(mc, "e_cas") and len(np.atleast_1d(mc.e_cas)) > 1
            else None,
        all_overlaps=overlaps.tolist(),
        ncore=ncore, ncas=ncas, nelecas=nelecas,
    )
    return mc


def run_multistate_nevpt2(
    mol,
    c_hats: Sequence[np.ndarray],
    fci_ref: Mapping,
    state_labels: Sequence[str] = None,
    ncore: int = None,
    ncas: int = None,
    nelecas: Tuple[int, int] = None,
    occ_threshold: float = 0.1,
    nroots_max: int = 8,
) -> dict:
    """State-specific NEVPT2 across K Pfau-NES (or other) reference CI vectors.

    For each ``c_hat^(k)`` in ``c_hats``:
      1. Build a variational CASCI reference matched to that state via
         :func:`build_casci_root_matched`,
      2. Run NEVPT2 to obtain ``E_pt2^(k)``,
      3. Aggregate the total energies and vertical excitation energies.

    Returns a dict with per-state results and the matrix of vertical
    excitation energies ``Delta_E[i, j] = E_total^(j) - E_total^(i)``
    in atomic units, suitable for direct insertion in a paper table.

    The CASCI active space is held fixed across states (taken from
    the ground-state c_hat^(0) unless overridden) so the NEVPT2
    corrections are computed in a consistent reference frame and the
    excitation energies are physically meaningful.
    """
    if state_labels is None:
        state_labels = [f"state_{k}" for k in range(len(c_hats))]
    if len(state_labels) != len(c_hats):
        raise ValueError(
            f"state_labels has {len(state_labels)} entries, "
            f"expected {len(c_hats)}"
        )

    # If active space not specified, derive from c_hat^(0) so all states
    # share the same active space.
    if ncore is None or ncas is None or nelecas is None:
        from OmegaQMC.cs.properties import (
            compute_1rdm, natural_occupations_from_rdm,
        )
        gamma_0 = compute_1rdm(
            c_hats[0], fci_ref["candidate_set"],
            fci_ref["n_orb"], fci_ref["nelec"],
        )
        no_occ = natural_occupations_from_rdm(gamma_0)
        ncore_auto, ncas_auto = select_active_space(
            no_occ, occ_threshold=occ_threshold,
        )
        n_total = sum(fci_ref["nelec"])
        if ncore is None:
            ncore = ncore_auto
        if ncas is None:
            ncas = ncas_auto
        if nelecas is None:
            n_active_e = n_total - 2 * ncore
            nelecas = (n_active_e // 2 + n_active_e % 2,
                       n_active_e // 2)

    per_state = []
    for k, (c_hat, label) in enumerate(zip(c_hats, state_labels)):
        mc = build_casci_root_matched(
            mol, c_hat, fci_ref,
            ncore=ncore, ncas=ncas, nelecas=nelecas,
            occ_threshold=occ_threshold, nroots_max=nroots_max,
        )
        res = run_nevpt2(mc)
        res["label"] = label
        res["root_index"] = mc._cs_meta_root["k_matched"]
        res["max_overlap"] = float(np.max(mc._cs_meta_root["all_overlaps"]))
        per_state.append(res)

    e_totals = np.array([r["e_total"] for r in per_state])
    e_casci = np.array([r["e_casci"] for r in per_state])
    # Vertical excitation matrices (au)
    delta_E_total = e_totals[None, :] - e_totals[:, None]
    delta_E_casci = e_casci[None, :] - e_casci[:, None]

    return dict(
        per_state=per_state,
        e_casci=e_casci,
        e_pt2=np.array([r["e_pt2"] for r in per_state]),
        e_total=e_totals,
        delta_E_casci_au=delta_E_casci,
        delta_E_total_au=delta_E_total,
        ncore=ncore, ncas=ncas, nelecas=nelecas,
        state_labels=list(state_labels),
    )


def casscf_nevpt2_reference(
    mol,
    ncas: int,
    nelecas: Tuple[int, int],
    init_mo_coeff: np.ndarray = None,
    verbose: int = 0,
) -> dict:
    """CASSCF + NEVPT2 baseline using PySCF's standard pipeline.

    Returns a dict with ``e_hf``, ``e_casscf``, ``e_pt2``, ``e_total``.
    """
    from pyscf import scf, mcscf, mrpt
    mf = scf.RHF(mol).run(verbose=0)
    mc = mcscf.CASSCF(mf, ncas, nelecas)
    if init_mo_coeff is not None:
        mc.mo_coeff = init_mo_coeff
    mc.verbose = verbose
    mc.kernel()
    e_pt2 = float(mrpt.NEVPT(mc).kernel())
    return dict(
        e_hf=float(mf.e_tot),
        e_casscf=float(mc.e_tot),
        e_pt2=e_pt2,
        e_total=float(mc.e_tot) + e_pt2,
        mc_casscf=mc,
    )


def chat_only_nevpt2(
    mol,
    c_hat: np.ndarray,
    fci_ref: Mapping,
    ncas: int = None,
    nelecas: Tuple[int, int] = None,
    occ_threshold: float = 0.02,
    max_ncas: int = None,
) -> dict:
    """ĉ-derived CASCI + NEVPT2 without the CASSCF baseline.

    Same first two steps as :func:`compare_nevpt2` (build CASCI from
    ĉ in the NN-NO frame; run NEVPT2 on it) but skips the matched
    CASSCF reference computation. Used when PySCF's CASSCF kernel is
    unstable or expensive at the target active space (e.g. H8/cc-pVDZ
    at n_orb=40, where the CASSCF baseline segfaults on the GH200
    build).

    Returns the same ``chat`` sub-dict shape as ``compare_nevpt2`` so
    downstream consumers can branch on key presence.
    """
    mc_chat = build_casci_from_chat(
        mol, c_hat, fci_ref,
        ncas=ncas, nelecas=nelecas,
        occ_threshold=occ_threshold, max_ncas=max_ncas,
    )
    chat_run = run_nevpt2(mc_chat)
    return dict(
        chat=dict(
            **chat_run,
            ncore=mc_chat.ncore, ncas=mc_chat.ncas, nelecas=mc_chat.nelecas,
            n_det_kept_in_active=mc_chat._cs_meta["n_det_kept_in_active"],
        ),
    )


def compare_nevpt2(
    mol,
    c_hat: np.ndarray,
    fci_ref: Mapping,
    ncas: int = None,
    nelecas: Tuple[int, int] = None,
    occ_threshold: float = 0.02,
    max_ncas: int = None,
) -> dict:
    """Run NEVPT2 from both the CS-derived reference and the CASSCF
    reference at matched (ncas, nelecas); report side-by-side totals.
    """
    mc_chat = build_casci_from_chat(
        mol, c_hat, fci_ref,
        ncas=ncas, nelecas=nelecas,
        occ_threshold=occ_threshold, max_ncas=max_ncas,
    )
    chat_run = run_nevpt2(mc_chat)
    ncas_used = mc_chat.ncas
    nelecas_used = mc_chat.nelecas
    casscf_run = casscf_nevpt2_reference(mol, ncas_used, nelecas_used)
    return dict(
        chat=dict(
            **chat_run,
            ncore=mc_chat.ncore, ncas=mc_chat.ncas, nelecas=mc_chat.nelecas,
            n_det_kept_in_active=mc_chat._cs_meta["n_det_kept_in_active"],
        ),
        casscf=dict(
            e_hf=casscf_run["e_hf"], e_casscf=casscf_run["e_casscf"],
            e_pt2=casscf_run["e_pt2"], e_total=casscf_run["e_total"],
            ncas=ncas_used, nelecas=nelecas_used,
        ),
        delta_total=chat_run["e_total"] - casscf_run["e_total"],
        delta_pt2=chat_run["e_pt2"] - casscf_run["e_pt2"],
        delta_reference=chat_run["e_casci"] - casscf_run["e_casscf"],
    )
