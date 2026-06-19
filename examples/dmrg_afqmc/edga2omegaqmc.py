"""
Convert EDGA output (edga_result.dat) to OmegaQMC trial wavefunction dict.

EDGA output format (edga_result.dat):
  # EDGA: state=0 completeness=0.9950000000 n_det=142 gen=5001
  -0.884632783511037  11111114444444  |7 8 9 10 11 12 13  |7 8 9 10 11 12 13
   0.123456789012345  11121114344444  |2 7 8 9 10 12 13   |7 8 9 10 11 12 13

Occupation encoding: 1=empty, 2=alpha, 3=beta, 4=double
Alpha orbitals: positions where occ is 2 or 4
Beta  orbitals: positions where occ is 3 or 4

OmegaQMC trial dict format (consumed by _AFQMCDriver):
  {
      'ci_coeffs': jnp.array, shape (ndet,),
      'occ_up':    jnp.array int, shape (ndet, nup),   # MO indices occupied by alpha
      'occ_dn':    jnp.array int, shape (ndet, ndown),  # MO indices occupied by beta
      'ndet':      int,
      'mo_coeff':  np.array, shape (nao, nmo),          # from PySCF mf/mc
  }

IMPORTANT: OmegaQMC works in MO basis — the occ_up/occ_dn indices refer to
MO indices (columns of mo_coeff).  EDGA operates in "chemical ordering" of
DMRG orbitals.  If the DMRG MOs match the PySCF MOs exactly (same ordering,
same basis), the indices can be used directly.  Otherwise a mapping is needed.

Usage:
    # As a module
    from edga2omegaqmc import load_edga_result, make_omegaqmc_trial

    dets, coeffs, meta = load_edga_result("edga_result.dat")
    trial = make_omegaqmc_trial(coeffs, dets, mo_coeff)

    # Then pass to OmegaQMC:
    driver = get_afqmc_func(mf, dt=0.005, trial=trial)

    # As a standalone script (prints summary)
    python edga2omegaqmc.py edga_result.dat
"""

import sys
import numpy as np


def load_edga_result(filename, coeff_threshold=0.0):
    """Parse edga_result.dat.

    Args:
        filename: Path to edga_result.dat.
        coeff_threshold: Discard determinants with |c| below this.
            Default 0.0 (keep all).

    Returns:
        dets: list of dict, each with keys 'occ', 'alpha', 'beta'.
            'occ': str like '11111114444444'
            'alpha': list of int (MO indices with alpha electron)
            'beta':  list of int (MO indices with beta electron)
        coeffs: list of float (CI coefficients).
        meta: dict with header info ('state', 'completeness', 'n_det', 'gen').
    """
    dets = []
    coeffs = []
    meta = {}

    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # Parse header
            if line.startswith('#'):
                for token in line.split():
                    if '=' in token:
                        key, val = token.split('=', 1)
                        meta[key] = val
                continue

            # Parse data line:
            #   coeff  occ_string  |alpha_orbs  |beta_orbs
            parts = line.split('|')
            if len(parts) < 3:
                continue

            # Part before first '|': "coeff  occ_string"
            head = parts[0].split()
            coeff = float(head[0])
            occ_str = head[1]

            # Alpha orbitals (between first and second '|')
            alpha = [int(x) for x in parts[1].split()]
            # Beta orbitals (after second '|')
            beta = [int(x) for x in parts[2].split()]

            if abs(coeff) >= coeff_threshold:
                coeffs.append(coeff)
                dets.append({
                    'occ': occ_str,
                    'alpha': alpha,
                    'beta': beta,
                })

    return dets, coeffs, meta


def make_omegaqmc_trial(coeffs, dets, mo_coeff, n_core=0, index_map=None, meta=None):
    """Build OmegaQMC trial dict from EDGA output.

    Args:
        coeffs: list/array of CI coefficients, length ndet.
        dets: list of dict with 'alpha' and 'beta' keys
            (MO index lists), from load_edga_result().
        mo_coeff: np.array, shape (nao, nmo).
            MO coefficient matrix from PySCF (mf.mo_coeff or mc.mo_coeff).
        n_core: number of core orbitals (always doubly occupied, below active space).
            When n_core > 0, EDGA active-space indices are shifted by n_core
            and core orbitals [0, 1, ..., n_core-1] are prepended to every
            determinant's occ_up and occ_dn.
            Example: naphthalene CAS(10,10) with 68 electrons → n_core = 29.
        index_map: optional dict or array mapping EDGA orbital index
            to PySCF MO index.  If provided, used instead of the n_core shift.
            Ignored when EDGA output has indexing=active (indices are already
            active orbital indices and only need n_core shift).
            Example: {0: 2, 1: 0, ...} means EDGA orbital 0 = PySCF MO 2.
        meta: optional dict from load_edga_result(). When meta contains
            'indexing'='active', index_map is ignored and only n_core shift
            is applied (EDGA already output active orbital indices).

    Returns:
        dict compatible with OmegaQMC's get_afqmc_func(trial=...).
    """
    # Auto-detect: if EDGA output is already in active orbital indexing,
    # skip index_map and just apply n_core shift
    if meta and meta.get('indexing') == 'active':
        if index_map is not None:
            print("  [edga2omegaqmc] indexing=active detected in header, ignoring index_map")
        index_map = None
    try:
        import jax.numpy as jnp
    except ImportError:
        jnp = np  # fallback for testing without JAX

    ndet = len(coeffs)
    nup_active = len(dets[0]['alpha'])
    ndn_active = len(dets[0]['beta'])
    nup = n_core + nup_active
    ndn = n_core + ndn_active

    core_orbs = list(range(n_core))  # [0, 1, ..., n_core-1]

    occ_up = np.zeros((ndet, nup), dtype=np.int32)
    occ_dn = np.zeros((ndet, ndn), dtype=np.int32)

    for i, det in enumerate(dets):
        alpha = det['alpha']
        beta = det['beta']

        assert len(alpha) == nup_active, \
            f"Det {i}: expected {nup_active} alpha electrons, got {len(alpha)}"
        assert len(beta) == ndn_active, \
            f"Det {i}: expected {ndn_active} beta electrons, got {len(beta)}"

        if index_map is not None:
            alpha = [index_map[a] for a in alpha]
            beta = [index_map[b] for b in beta]
        elif n_core > 0:
            alpha = [a + n_core for a in alpha]
            beta = [b + n_core for b in beta]

        occ_up[i] = sorted(core_orbs + alpha)
        occ_dn[i] = sorted(core_orbs + beta)

    trial = {
        'ci_coeffs': jnp.array(np.array(coeffs, dtype=np.float64)),
        'occ_up': jnp.array(occ_up),
        'occ_dn': jnp.array(occ_dn),
        'ndet': ndet,
        'mo_coeff': np.asarray(mo_coeff),
    }

    return trial


# ---- Standalone: print summary ----
if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python edga2omegaqmc.py <edga_result.dat> [coeff_threshold] [--n_core N]")
        print("  Parses EDGA output and prints summary for OmegaQMC integration.")
        print("  --n_core N : number of core orbitals (shifts active indices by N)")
        sys.exit(1)

    fname = sys.argv[1]
    threshold = 0.0
    n_core = 0

    # Parse positional and keyword arguments
    i = 2
    while i < len(sys.argv):
        if sys.argv[i] == '--n_core':
            n_core = int(sys.argv[i + 1])
            i += 2
        else:
            threshold = float(sys.argv[i])
            i += 1

    dets, coeffs, meta = load_edga_result(fname, coeff_threshold=threshold)
    coeffs_arr = np.array(coeffs)

    print("=" * 60)
    print("EDGA -> OmegaQMC conversion summary")
    print("=" * 60)
    print(f"  File: {fname}")
    for k, v in meta.items():
        print(f"  {k} = {v}")
    print(f"  Determinants loaded: {len(dets)}")
    if threshold > 0:
        print(f"  Threshold applied: |c| >= {threshold}")

    if len(dets) == 0:
        print("  No determinants found!")
        sys.exit(1)

    nup_active = len(dets[0]['alpha'])
    ndn_active = len(dets[0]['beta'])
    n_orbs = len(dets[0]['occ'])
    sum_c2 = np.sum(coeffs_arr ** 2)

    print(f"  n_orbs (active) = {n_orbs}")
    print(f"  nup_active = {nup_active}, ndn_active = {ndn_active}")
    if n_core > 0:
        print(f"  n_core = {n_core}")
        print(f"  nup_total = {n_core + nup_active}, ndn_total = {n_core + ndn_active}")
    print(f"  sum(c^2) = {sum_c2:.10f}")
    print(f"  max|c| = {np.max(np.abs(coeffs_arr)):.10f}")
    print()

    # Print top determinants
    idx_sorted = np.argsort(-np.abs(coeffs_arr))
    n_show = min(10, len(dets))
    print(f"  Top {n_show} determinants:")
    print(f"  {'Rank':>4}  {'Coefficient':>15}  {'c^2':>12}  "
          f"alpha_MOs  beta_MOs")
    print(f"  {'-' * 70}")
    for rank, i in enumerate(idx_sorted[:n_show]):
        c = coeffs_arr[i]
        print(f"  {rank+1:4d}  {c:15.10f}  {c*c:12.8f}  "
              f"{dets[i]['alpha']}  {dets[i]['beta']}")

    nup_total = n_core + nup_active
    ndn_total = n_core + ndn_active

    print()
    print("  OmegaQMC trial dict keys needed:")
    print(f"    'ci_coeffs': shape ({len(dets)},)")
    print(f"    'occ_up':    shape ({len(dets)}, {nup_total})  int32")
    print(f"    'occ_dn':    shape ({len(dets)}, {ndn_total})  int32")
    print(f"    'ndet':      {len(dets)}")
    print(f"    'mo_coeff':  shape (nao, nmo)  -- from PySCF")
    if n_core > 0:
        print(f"  Core orbitals [0..{n_core-1}] prepended to every determinant.")
        print(f"  Active indices shifted by {n_core}.")
    print()
    print("  NOTE: mo_coeff must come from PySCF (mf.mo_coeff or mc.mo_coeff).")
    print("  If DMRG orbital ordering differs from PySCF, provide index_map")
    print("  to make_omegaqmc_trial().")


