"""
QED-DMRG -> factorized QED-AFQMC trial converter.

Mirrors ``OmegaQMC.addons.qed_casscf.to_afqmc_trial`` byte-for-byte in schema
and convention, but the electronic CI and the photon-traced 1-RDM are sourced
from a QED-DMRG / QED-EDGA expansion instead of a PySCF QED-CASSCF result.

Factorized = electronic MSD  ⊗  single-Gaussian photon  f(q; s, q0).
Per-determinant photon Fock info (the multi-det QED-AFQMC branch) is
deliberately not used here; this is the DMRG -> AFQMC pipeline baseline.

Convention (must match qed_casscf.py for the comparison to be meaningful):

  - lam        = ||coupling_vec||
  - epsilon    = coupling_vec / lam              (unit polarization)
  - dip_ao     = lam * einsum('k,kpq->pq', epsilon, mol.intor('int1e_r', 3))
  - dip_mo     = mo_coeff.T @ dip_ao @ mo_coeff
  - dipole     = einsum('pq,pq->', dip_mo, dm1)
  - q0         = -dipole / sqrt(omega)

  dm1 is the photon-traced (core+active) electronic 1-RDM in the SAME MO
  basis as ``mo_coeff``. We reconstruct it from the EDGA expansion by:

    1. Grouping EDGA dets by photon Fock number n.
    2. For each n, populating a dense (na, nb) active CI array.
    3. Building casdm1[n] = direct_spin1.make_rdm1(ci_n, ncas, nelec_act)
       and summing over n.
    4. Embedding the active block into the (norb, norb) MO-basis dm1
       (core diagonal = 2.0, virtual = 0).

  The electronic CI for the trial is taken from the m=0 (displaced-vacuum)
  sector only and renormalised within that sector. With the coherent-state
  photon basis ``<n_ph>`` is small, so the m=0 sector carries ~all of the
  electronic weight — same trial shape qed_casscf.py uses.

Inputs needed (apples-to-apples with the comparison CASSCF trial):

  - ``mo_coeff``: the SAME MO basis used to build the DMRG FCIDUMP. EDGA
    dets are active-space indices in this basis.
  - ``omega``, ``coupling_vec``: the cavity parameters used by both the
    DMRG run and the AFQMC run.
  - ``n_core``, ``ncas``, ``nelecas``: active-space layout matching EDGA.

Output dict keys (identical to to_afqmc_trial):

  ci_coeffs, occ_up, occ_dn, ndet, mo_coeff, q0, cavity=True
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------- #
#  EDGA file parser (electronic CI per photon sector)                    #
# ---------------------------------------------------------------------- #

def _parse_edga_file(path):
    """Parse ``edga_result.dat`` produced by ``qed_edga``.

    Returns (dets, coeffs, meta) where:
      dets   : list of dict with keys ``alpha`` (list[int], active-space
               0-based), ``beta``, ``nphot``.
      coeffs : list[float] of CI coefficients.
      meta   : dict from the leading ``#`` header line.
    """
    dets, coeffs, meta = [], [], {}
    with open(path, 'r') as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            if line.startswith('#'):
                for tok in line.split():
                    if '=' in tok:
                        k, v = tok.split('=', 1)
                        meta[k] = v
                continue
            parts = line.split('|')
            if len(parts) < 3:
                continue
            head = parts[0].split()
            if len(head) < 3:
                continue
            coeff = float(head[0])
            nphot = int(head[1])
            alpha = [int(x) for x in parts[1].split()]
            beta = [int(x) for x in parts[2].split()]
            coeffs.append(coeff)
            dets.append({'alpha': alpha, 'beta': beta, 'nphot': nphot})
    return dets, coeffs, meta


# ---------------------------------------------------------------------- #
#  Photon-traced 1-RDM from EDGA expansion                                #
# ---------------------------------------------------------------------- #

def _photon_traced_casdm1_from_edga(dets, coeffs, ncas, nelecas):
    """Reconstruct the photon-traced active 1-RDM from an EDGA expansion.

    Operates per photon Fock sector:
        casdm1 = sum_n  make_rdm1(ci_n, ncas, nelecas)
    where ci_n[ia, ib] = c_{IJ,n} from EDGA dets with nphot == n.

    Args:
        dets   : output of ``_parse_edga_file``.
        coeffs : output of ``_parse_edga_file``.
        ncas   : number of active orbitals.
        nelecas: (na, nb) tuple.

    Returns:
        casdm1 : np.ndarray, shape (ncas, ncas).
        ci_by_n: dict {n_ph: ndarray(na, nb)} — needed for the m=0 trial.
        diagnostics: dict with sum_c2 (total expansion weight),
                     sum_c2_m0 (m=0 sector weight),
                     n_sectors, observed_n_ph_max.
    """
    from pyscf.fci import cistring, direct_spin1

    nelecas_a, nelecas_b = (int(n) for n in nelecas)
    occslst_a = cistring.gen_occslst(range(ncas), nelecas_a)
    occslst_b = cistring.gen_occslst(range(ncas), nelecas_b)
    na, nb = len(occslst_a), len(occslst_b)

    idx_a = {tuple(int(x) for x in occ): i
             for i, occ in enumerate(np.asarray(occslst_a).tolist())}
    idx_b = {tuple(int(x) for x in occ): i
             for i, occ in enumerate(np.asarray(occslst_b).tolist())}

    ci_by_n = {}
    for det, coeff in zip(dets, coeffs):
        n = int(det['nphot'])
        a_key = tuple(sorted(int(x) for x in det['alpha']))
        b_key = tuple(sorted(int(x) for x in det['beta']))
        if len(a_key) != nelecas_a or len(b_key) != nelecas_b:
            raise ValueError(
                f"EDGA det has nalpha={len(a_key)}, nbeta={len(b_key)}, "
                f"but expected ({nelecas_a}, {nelecas_b}). Check ncas / "
                f"nelecas vs. the DMRG FCIDUMP.")
        if a_key not in idx_a or b_key not in idx_b:
            raise ValueError(
                f"EDGA det occupation {a_key}/{b_key} not in active-space "
                f"strings (ncas={ncas}). Are the EDGA indices already in "
                f"the active basis (0..ncas-1)?")
        ia = idx_a[a_key]
        ib = idx_b[b_key]
        if n not in ci_by_n:
            ci_by_n[n] = np.zeros((na, nb), dtype=np.float64)
        ci_by_n[n][ia, ib] = coeff

    casdm1 = np.zeros((ncas, ncas), dtype=np.float64)
    for n, ci_n in ci_by_n.items():
        d1, _ = direct_spin1.make_rdm12(ci_n, ncas, (nelecas_a, nelecas_b))
        casdm1 += d1

    sum_c2 = float(sum((c ** 2) for c in coeffs))
    sum_c2_m0 = float(np.sum(ci_by_n.get(0, np.zeros(1)) ** 2))
    diag = {
        'sum_c2': sum_c2,
        'sum_c2_m0': sum_c2_m0,
        'n_sectors': len(ci_by_n),
        'observed_n_ph_max': max(ci_by_n) if ci_by_n else 0,
        'na': na, 'nb': nb,
    }
    return casdm1, ci_by_n, diag


# ---------------------------------------------------------------------- #
#  Main API: build the factorized trial dict                              #
# ---------------------------------------------------------------------- #

def build_factorized_trial_from_qed_dmrg(
        mol, mo_coeff,
        omega, coupling_vec,
        n_core, ncas, nelecas,
        edga_dets, edga_coeffs,
        coeff_threshold=1e-4,
        max_det=None,
        verbose=False,
):
    """Build a factorized QED-AFQMC trial from EDGA-QED expansion.

    Schema, sign conventions and normalisation are identical to
    ``OmegaQMC.addons.qed_casscf.to_afqmc_trial``. The only difference is
    the source of the CI vector and the 1-RDM.

    Args:
        mol         : PySCF Mole (for dipole AO integrals).
        mo_coeff    : (nao, nmo) np.ndarray. Same MO basis as the DMRG /
                      EDGA run (the FCIDUMP orbital basis).
        omega       : photon frequency (Ha).
        coupling_vec: light-matter coupling vector (3,). lam = ||vec||,
                      epsilon = vec/lam, dip_ao = lam * eps . r_AO.
        n_core, ncas, nelecas: active-space layout used by EDGA.
        edga_dets   : list of {'alpha','beta','nphot'} from
                      ``_parse_edga_file`` (or ``load_qed_edga_result``
                      from qed_edga2omegaqmc).
        edga_coeffs : list[float], CI coefficients aligned with edga_dets.
        coeff_threshold: drop |c_I| below this from the m=0 trial.
        max_det     : optional upper bound on the number of dets kept
                      after sorting by |c_I| (descending). None = no cap.
                      Note: q0 / dipole are STILL computed from the full
                      EDGA expansion (all photon sectors + all dets);
                      max_det only truncates the m=0 trial CI list.
        verbose     : print diagnostics.

    Returns:
        dict with the same keys ``to_afqmc_trial`` produces, plus a nested
        ``_diagnostics`` block for the comparison report.
    """
    mo_coeff = np.asarray(mo_coeff, dtype=np.float64)
    nao = mo_coeff.shape[0]
    norb = mo_coeff.shape[1]
    nelecas_a, nelecas_b = (int(n) for n in nelecas)

    # --- 1. Photon-traced active 1-RDM from EDGA ----------------------- #
    casdm1, ci_by_n, edga_diag = _photon_traced_casdm1_from_edga(
        edga_dets, edga_coeffs, ncas, (nelecas_a, nelecas_b))
    na, nb = edga_diag['na'], edga_diag['nb']

    # --- 2. Full (core + active) 1-RDM in the MO basis ----------------- #
    dm1 = np.zeros((norb, norb), dtype=np.float64)
    for i in range(n_core):
        dm1[i, i] = 2.0
    dm1[n_core:n_core + ncas, n_core:n_core + ncas] = casdm1

    # --- 3. Lambda-scaled dipole AO -> MO (mirror qed_casscf.py) ------- #
    coupling_vec_arr = np.asarray(coupling_vec, dtype=np.float64)
    lam = float(np.linalg.norm(coupling_vec_arr))
    if lam > 1e-15:
        epsilon = coupling_vec_arr / lam
    else:
        epsilon = np.array([0.0, 0.0, 1.0])
        lam = 0.0
    dip_ao = lam * np.einsum('k,kpq->pq', epsilon,
                             mol.intor('int1e_r', comp=3))
    dip_mo = mo_coeff.T @ dip_ao @ mo_coeff
    dipole = float(np.einsum('pq,pq->', dip_mo, dm1))
    q0 = -dipole / np.sqrt(omega) if omega > 0 else 0.0

    # --- 4. Electronic CI from m=0 sector, renormalised ---------------- #
    ci0 = ci_by_n.get(0, np.zeros((na, nb), dtype=np.float64))
    norm0 = float(np.linalg.norm(ci0))
    if norm0 > 0.0:
        ci0_unit = ci0 / norm0
    else:
        raise ValueError(
            "EDGA expansion contains no nphot=0 determinants; the "
            "factorized trial requires the displaced-vacuum photon "
            "sector to define the electronic CI.")

    # --- 5. Enumerate dets above threshold, sort by |c_I| descending --- #
    from pyscf.fci import cistring
    occslst_a = cistring.gen_occslst(range(ncas), nelecas_a)
    occslst_b = cistring.gen_occslst(range(ncas), nelecas_b)
    core = list(range(n_core))

    coeffs, occ_up_list, occ_dn_list = [], [], []
    for ia in range(na):
        for ib in range(nb):
            c = float(ci0_unit[ia, ib])
            if abs(c) > coeff_threshold:
                coeffs.append(c)
                occ_up_list.append(
                    core + [n_core + int(j) for j in occslst_a[ia]])
                occ_dn_list.append(
                    core + [n_core + int(j) for j in occslst_b[ib]])

    if not coeffs:
        raise ValueError(
            f"No m=0 determinants pass coeff_threshold={coeff_threshold}. "
            f"Lower the threshold or check the EDGA expansion.")

    coeffs_arr = np.array(coeffs, dtype=np.float64)
    occ_up = np.array(occ_up_list, dtype=np.int32)
    occ_dn = np.array(occ_dn_list, dtype=np.int32)
    order = np.argsort(-np.abs(coeffs_arr))
    coeffs_arr = coeffs_arr[order]
    occ_up = occ_up[order]
    occ_dn = occ_dn[order]

    if max_det is not None and len(coeffs_arr) > int(max_det):
        n_keep = int(max_det)
        coeffs_arr = coeffs_arr[:n_keep]
        occ_up = occ_up[:n_keep]
        occ_dn = occ_dn[:n_keep]

    trial = {
        'ci_coeffs': coeffs_arr,
        'occ_up': occ_up,
        'occ_dn': occ_dn,
        'ndet': int(len(coeffs_arr)),
        'mo_coeff': mo_coeff,
        'q0': float(q0),
        'cavity': True,
    }

    # Diagnostics (kept separate; trial dict consumers ignore unknown keys
    # but we keep it under a single sentinel name for cleanliness).
    trial['_diagnostics'] = {
        'dipole': float(dipole),
        'q0': float(q0),
        'lam': float(lam),
        'omega': float(omega),
        'edga_sum_c2': edga_diag['sum_c2'],
        'edga_sum_c2_m0': edga_diag['sum_c2_m0'],
        'edga_m0_norm': float(norm0),
        'edga_n_sectors': edga_diag['n_sectors'],
        'edga_observed_n_ph_max': edga_diag['observed_n_ph_max'],
        'm0_trial_ndet': int(len(coeffs_arr)),
        'm0_trial_norm_after_threshold':
            float(np.linalg.norm(coeffs_arr)),
    }

    if verbose:
        d = trial['_diagnostics']
        print("=" * 62)
        print("DMRG-sourced FACTORIZED QED-AFQMC trial")
        print("=" * 62)
        print(f"  lam = {d['lam']:.6f}, omega = {d['omega']:.6f}")
        print(f"  EDGA: sum(c^2)        = {d['edga_sum_c2']:.10f}  "
              f"(<= 1; 1 - x = truncation loss)")
        print(f"  EDGA: m=0 sector wt   = {d['edga_sum_c2_m0']:.10f}  "
              f"(close to total if <n_ph> << 1)")
        print(f"  EDGA: n_ph sectors    = {d['edga_n_sectors']} "
              f"(max observed n_ph = {d['edga_observed_n_ph_max']})")
        print(f"  <D> (photon-traced)   = {d['dipole']:+.10e}")
        print(f"  q0 = -<D>/sqrt(Omega) = {d['q0']:+.10e}")
        print(f"  m=0 trial ndet (post-threshold) = {d['m0_trial_ndet']}")
        print(f"  m=0 trial coeff norm  = "
              f"{d['m0_trial_norm_after_threshold']:.10f}  "
              f"(1 - x = threshold loss)")

    return trial


def build_factorized_trial_from_qed_dmrg_file(
        mol, mo_coeff, omega, coupling_vec,
        n_core, ncas, nelecas,
        edga_path,
        coeff_threshold=1e-4,
        max_det=None,
        verbose=False,
):
    """Thin wrapper: load ``edga_result.dat`` and call the main builder."""
    dets, coeffs, meta = _parse_edga_file(edga_path)
    if verbose:
        print(f"[qed_dmrg2omegaqmc_factorized] loaded {len(dets)} dets "
              f"from {edga_path}")
        if meta:
            kv = ', '.join(f"{k}={v}" for k, v in meta.items())
            print(f"  EDGA header: {kv}")
    trial = build_factorized_trial_from_qed_dmrg(
        mol=mol, mo_coeff=mo_coeff,
        omega=omega, coupling_vec=coupling_vec,
        n_core=n_core, ncas=ncas, nelecas=nelecas,
        edga_dets=dets, edga_coeffs=coeffs,
        coeff_threshold=coeff_threshold,
        max_det=max_det,
        verbose=verbose,
    )
    trial['_diagnostics']['edga_meta'] = meta
    return trial


# ---------------------------------------------------------------------- #
#  Helper: side-by-side compare against to_afqmc_trial                   #
# ---------------------------------------------------------------------- #

def compare_trials(trial_dmrg, trial_casscf, atol_q0=1e-4, atol_ci=1e-4):
    """Compare a DMRG-sourced factorized trial to a CASSCF-sourced one.

    Returns a dict with per-field max abs differences. q0 / mo_coeff /
    leading-CI agreement is the apples-to-apples sanity gate. ``ndet``
    can legitimately differ because EDGA truncates and pyscf CASSCF does
    not; we focus on the dominant-determinant overlap and q0 agreement.
    """
    report = {}
    report['q0_dmrg'] = float(trial_dmrg['q0'])
    report['q0_casscf'] = float(trial_casscf['q0'])
    report['q0_diff'] = float(trial_dmrg['q0'] - trial_casscf['q0'])
    report['q0_match'] = abs(report['q0_diff']) < atol_q0

    mo_d = np.asarray(trial_dmrg['mo_coeff'])
    mo_c = np.asarray(trial_casscf['mo_coeff'])
    if mo_d.shape != mo_c.shape:
        report['mo_shape_match'] = False
        report['mo_max_abs_diff'] = float('inf')
    else:
        report['mo_shape_match'] = True
        # The two MO bases can differ by sign/phase per column; align then
        # take the max diff. (If the bases truly diverge, all bets are off.)
        signs = np.sign(np.sum(mo_d * mo_c, axis=0))
        signs[signs == 0] = 1.0
        diff = mo_d - mo_c * signs[None, :]
        report['mo_max_abs_diff'] = float(np.max(np.abs(diff)))

    # Dominant determinant: occupation strings should match.
    occ_d = (tuple(trial_dmrg['occ_up'][0].tolist()),
             tuple(trial_dmrg['occ_dn'][0].tolist()))
    occ_c = (tuple(trial_casscf['occ_up'][0].tolist()),
             tuple(trial_casscf['occ_dn'][0].tolist()))
    report['dom_det_match'] = (occ_d == occ_c)
    report['ndet_dmrg'] = int(trial_dmrg['ndet'])
    report['ndet_casscf'] = int(trial_casscf['ndet'])

    # Common-det CI overlap. Build occ -> coeff maps and project.
    def _ci_map(t):
        m = {}
        for c, oa, ob in zip(np.asarray(t['ci_coeffs']),
                             np.asarray(t['occ_up']),
                             np.asarray(t['occ_dn'])):
            m[(tuple(int(x) for x in oa),
               tuple(int(x) for x in ob))] = float(c)
        return m
    md = _ci_map(trial_dmrg)
    mc = _ci_map(trial_casscf)
    common = set(md.keys()) & set(mc.keys())
    if common:
        # Account for global sign freedom between the two trials.
        sign = 1.0
        # Use the dominant key (largest |c| in either) to fix the sign.
        def _key_score(k):
            return abs(md.get(k, 0.0)) + abs(mc.get(k, 0.0))
        k0 = max(common, key=_key_score)
        if md[k0] * mc[k0] < 0:
            sign = -1.0
        diffs = np.array([abs(md[k] * sign - mc[k]) for k in common])
        report['ci_common_ndet'] = len(common)
        report['ci_max_abs_diff'] = float(diffs.max())
        report['ci_match'] = report['ci_max_abs_diff'] < atol_ci
    else:
        report['ci_common_ndet'] = 0
        report['ci_max_abs_diff'] = float('inf')
        report['ci_match'] = False

    report['overall_pass'] = (
        report['q0_match']
        and report['mo_shape_match']
        and report['mo_max_abs_diff'] < 1e-6
        and report['dom_det_match']
        and report['ci_match']
    )
    return report


if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print("Usage: python qed_dmrg2omegaqmc_factorized.py "
              "<edga_result.dat> [--verbose]")
        sys.exit(1)
    # Lightweight inspection: parse the file, print sector breakdown.
    dets, coeffs, meta = _parse_edga_file(sys.argv[1])
    by_n = {}
    for d, c in zip(dets, coeffs):
        by_n.setdefault(d['nphot'], []).append(c)
    print(f"  Loaded {len(dets)} EDGA dets; header: {meta}")
    print(f"  sum(c^2) total = "
          f"{sum(c**2 for c in coeffs):.10f}")
    for n in sorted(by_n):
        cs = np.array(by_n[n])
        print(f"    n_ph={n}: ndet={len(cs)}, "
              f"sum(c^2)={float(np.sum(cs**2)):.6e}")

