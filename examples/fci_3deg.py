#!/usr/bin/env python3
"""
Compare AFQMC vs exact FCI for the 3D homogeneous electron gas.

Builds the full many-body Hamiltonian in the Slater determinant basis
and diagonalizes exactly, then runs AFQMC on the same system.

This serves as a definitive correctness test for the AFQMC implementation.

Usage:
    python scripts/fci_3deg.py [--N_elec 2] [--N_pw 7] [--rs 1.0]

Defaults: N=2, N_pw=7, rs=1.0 (tiny system where FCI is tractable).
"""

import argparse
import time
import bisect
import numpy as np
from itertools import combinations
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh

from OmegaQMC.afqmc_pw_heg import (
    build_3deg_system,
    prepare_3deg_integrals,
    build_trial_3deg,
    get_afqmc_3deg_func,
    _generate_3d_kgrid,
)


# ===================================================================
# Closed-shell detection
# ===================================================================

# 3D HEG shell-closing PW counts: 1, 7, 19, 27, 33, 57, 81, 93, 123, ...
# Closed-shell electron counts (unpolarized) = 2 * shell_count
# Closed-shell electron counts (polarized)   = 1 * shell_count
_3D_SHELL_COUNTS = [1, 7, 19, 27, 33, 57, 81, 93, 123, 147, 171,
                    179, 203, 251, 257, 305, 341, 365, 389]


def check_closed_shell(N_elec, N_pw, polarization):
    """Check if the electron count gives a closed-shell configuration.

    Returns (is_closed, message).
    """
    if polarization == 'unpolarized':
        if N_elec % 2 != 0:
            return False, "Odd N_elec with unpolarized: inherently open-shell"
        n_per_spin = N_elec // 2
    else:
        n_per_spin = N_elec

    # Get actual shell structure
    grid, N_pw_actual = _generate_3d_kgrid(N_pw)
    norms_sq = grid[:, 0]**2 + grid[:, 1]**2 + grid[:, 2]**2
    unique_norms = np.unique(norms_sq)

    # Check if n_per_spin fills complete shells
    count = 0
    for norm_sq in unique_norms:
        shell_size = np.sum(norms_sq == norm_sq)
        count += shell_size
        if count == n_per_spin:
            return True, f"Closed shell: {n_per_spin} orbitals/spin fill complete shells"
        if count > n_per_spin:
            n_in_shell = n_per_spin - (count - shell_size)
            return False, (
                f"OPEN SHELL: {n_per_spin} orbitals/spin partially fills "
                f"a {shell_size}-fold degenerate shell "
                f"({n_in_shell}/{shell_size} occupied). "
                f"Single-determinant trial will break symmetry.\n"
                f"  Nearby closed-shell N_elec (unpolarized): "
                f"{', '.join(str(2*s) for s in _3D_SHELL_COUNTS if abs(2*s - N_elec) < N_elec)}"
            )

    return False, f"n_per_spin={n_per_spin} exceeds N_pw={N_pw_actual}"


# ===================================================================
# FCI: exact diagonalization in the Slater determinant basis
# ===================================================================

def reconstruct_eri_from_cholesky(chol, chol_sign):
    """Reconstruct the full ERI tensor from symmetrized Cholesky vectors.

    (pq|rs) = Sigma_g sign[g] * L^g_{pq} * L^g_{rs}   [chemist notation]

    Args:
        chol: (N_chol, N_orb, N_orb) Cholesky vectors.
        chol_sign: (N_chol,) signs (+1 or -1).

    Returns:
        eri: (N_orb, N_orb, N_orb, N_orb) in chemist notation.
    """
    return np.einsum('g,gpq,grs->pqrs', chol_sign, chol, chol)


def _occ_diff(occ_bra, occ_ket):
    """Find orbitals that differ between bra and ket.

    Returns (in_ket_not_bra, in_bra_not_ket) as sorted lists.
    """
    set_bra = set(occ_bra)
    set_ket = set(occ_ket)
    return sorted(set_ket - set_bra), sorted(set_bra - set_ket)


def _compute_sign(occ, p, q):
    """Sign of the single excitation a^dag_q a_p on |occ>.

    = (-1)^(number of occupied orbitals strictly between p and q).
    """
    lo, hi = min(p, q), max(p, q)
    n_between = sum(1 for k in occ if lo < k < hi)
    return (-1) ** n_between


def _sign_double_aa(occ, p1, p2, q1, q2):
    """Sign for double same-spin excitation a^dag_q2 a^dag_q1 a_p1 a_p2 on |occ>.

    Decomposed into four sequential fermion operations.
    """
    lst = sorted(occ)

    # Remove p2
    pos_p2 = lst.index(p2)
    sign = (-1) ** pos_p2
    lst.pop(pos_p2)

    # Remove p1
    pos_p1 = lst.index(p1)
    sign *= (-1) ** pos_p1
    lst.pop(pos_p1)

    # Insert q1
    pos_q1 = bisect.bisect_left(lst, q1)
    sign *= (-1) ** pos_q1
    lst.insert(pos_q1, q1)

    # Insert q2
    pos_q2 = bisect.bisect_left(lst, q2)
    sign *= (-1) ** pos_q2

    return sign


def _diagonal_element(occ_a, occ_b, h1e, eri):
    """Diagonal Hamiltonian matrix element <D|H|D>."""
    val = 0.0

    # One-body
    for i in occ_a:
        val += h1e[i, i]
    for i in occ_b:
        val += h1e[i, i]

    # Two-body: alpha-alpha
    for i1 in range(len(occ_a)):
        for i2 in range(i1 + 1, len(occ_a)):
            i, j = occ_a[i1], occ_a[i2]
            val += eri[i, i, j, j] - eri[i, j, j, i]

    # Two-body: beta-beta
    for i1 in range(len(occ_b)):
        for i2 in range(i1 + 1, len(occ_b)):
            i, j = occ_b[i1], occ_b[i2]
            val += eri[i, i, j, j] - eri[i, j, j, i]

    # Two-body: alpha-beta (Coulomb only)
    for i in occ_a:
        for j in occ_b:
            val += eri[i, i, j, j]

    return val


def _single_excitation_element(p, q, occ_a, occ_b, h1e, eri, spin):
    """Unsigned matrix element for a single excitation p -> q.

    h_{qp} + sum_j [(qp|jj) - (qj|jp)] (same spin)
           + sum_j  (qp|jj)             (opposite spin)
    """
    val = h1e[q, p]
    same_occ = occ_a if spin == 'alpha' else occ_b
    opp_occ = occ_b if spin == 'alpha' else occ_a

    for j in same_occ:
        if j != p:
            val += eri[q, p, j, j] - eri[q, j, j, p]
    for j in opp_occ:
        val += eri[q, p, j, j]

    return val


def slater_condon(occ_a_bra, occ_b_bra, occ_a_ket, occ_b_ket, h1e, eri):
    """Compute <bra|H|ket> using Slater-Condon rules.

    ERIs in chemist notation: (pq|rs).
    """
    diff_a = _occ_diff(occ_a_bra, occ_a_ket)
    diff_b = _occ_diff(occ_b_bra, occ_b_ket)
    n_diff_a = len(diff_a[0])
    n_diff_b = len(diff_b[0])
    total_diff = n_diff_a + n_diff_b

    if total_diff > 2:
        return 0.0

    if total_diff == 0:
        return _diagonal_element(occ_a_ket, occ_b_ket, h1e, eri)

    if total_diff == 1:
        if n_diff_a == 1:
            p, q = diff_a[0][0], diff_a[1][0]
            sign = _compute_sign(occ_a_ket, p, q)
            return sign * _single_excitation_element(
                p, q, occ_a_ket, occ_b_ket, h1e, eri, 'alpha')
        else:
            p, q = diff_b[0][0], diff_b[1][0]
            sign = _compute_sign(occ_b_ket, p, q)
            return sign * _single_excitation_element(
                p, q, occ_a_ket, occ_b_ket, h1e, eri, 'beta')

    # total_diff == 2
    if n_diff_a == 2:
        p1, p2 = diff_a[0]
        q1, q2 = diff_a[1]
        sign = _sign_double_aa(occ_a_ket, p1, p2, q1, q2)
        return sign * (eri[q1, p1, q2, p2] - eri[q1, p2, q2, p1])

    if n_diff_b == 2:
        p1, p2 = diff_b[0]
        q1, q2 = diff_b[1]
        sign = _sign_double_aa(occ_b_ket, p1, p2, q1, q2)
        return sign * (eri[q1, p1, q2, p2] - eri[q1, p2, q2, p1])

    # n_diff_a == 1 and n_diff_b == 1: alpha-beta double
    pa, qa = diff_a[0][0], diff_a[1][0]
    pb, qb = diff_b[0][0], diff_b[1][0]
    sign_a = _compute_sign(occ_a_ket, pa, qa)
    sign_b = _compute_sign(occ_b_ket, pb, qb)
    return sign_a * sign_b * eri[qa, pa, qb, pb]


def build_fci_hamiltonian(h1e, eri, nup, ndown, N_orb):
    """Build the full CI Hamiltonian matrix.

    Args:
        h1e:   (N_orb, N_orb) one-body integrals.
        eri:   (N_orb, N_orb, N_orb, N_orb) ERIs in chemist notation.
        nup, ndown: electron counts per spin.
        N_orb: number of spatial orbitals.

    Returns:
        H_dense: dense Hamiltonian matrix.
        dets_a, dets_b: lists of alpha/beta determinants.
    """
    orbs = list(range(N_orb))
    dets_a = list(combinations(orbs, nup))
    dets_b = list(combinations(orbs, ndown)) if ndown > 0 else [()]

    Na, Nb = len(dets_a), len(dets_b)
    Ntot = Na * Nb
    print(f"  FCI dimension: {Na} x {Nb} = {Ntot} determinants")

    if Ntot > 50000:
        print(f"  WARNING: FCI space is large ({Ntot}). This will be slow.")

    H_dense = np.zeros((Ntot, Ntot))

    t0 = time.time()
    for ia_bra in range(Na):
        for ib_bra in range(Nb):
            I = ia_bra * Nb + ib_bra
            for ia_ket in range(Na):
                for ib_ket in range(Nb):
                    J = ia_ket * Nb + ib_ket
                    if J < I:
                        continue
                    mel = slater_condon(
                        dets_a[ia_bra], dets_b[ib_bra],
                        dets_a[ia_ket], dets_b[ib_ket],
                        h1e, eri)
                    if abs(mel) > 1e-15:
                        H_dense[I, J] = mel
                        H_dense[J, I] = mel

    t1 = time.time()
    asym = np.max(np.abs(H_dense - H_dense.T))
    print(f"  Hamiltonian built in {t1 - t0:.2f} s  "
          f"(hermiticity: {asym:.2e})")

    return H_dense, dets_a, dets_b


def run_fci(h1e, eri, nup, ndown, N_orb, enuc=0.0, n_roots=1):
    """Run exact FCI and return ground state energy.

    Args:
        h1e, eri: integrals in chemist notation.
        nup, ndown: electron counts.
        N_orb: number of spatial orbitals.
        enuc: nuclear/background energy.
        n_roots: number of eigenvalues.

    Returns:
        energies: array of lowest eigenvalues (including enuc).
    """
    H, dets_a, dets_b = build_fci_hamiltonian(h1e, eri, nup, ndown, N_orb)
    Ntot = H.shape[0]

    print(f"  Diagonalizing ({Ntot} x {Ntot})...")
    t0 = time.time()

    if Ntot <= 3000:
        eigenvalues = np.linalg.eigvalsh(H)[:n_roots]
    else:
        H_sp = csr_matrix(H)
        eigenvalues = eigsh(H_sp, k=min(n_roots, Ntot - 1),
                            which='SA', return_eigenvectors=False)
        eigenvalues = np.sort(eigenvalues)

    print(f"  Diagonalization done in {time.time() - t0:.2f} s")
    return eigenvalues + enuc


def compute_hf_energy(h1e, eri, nup, ndown, enuc=0.0):
    """Compute HF energy from density matrices."""
    trial_up, trial_dn = build_trial_3deg(h1e, nup, ndown)
    Pa = np.array(trial_up) @ np.array(trial_up).T
    Pb = (np.array(trial_dn) @ np.array(trial_dn).T
          if ndown > 0 else np.zeros_like(Pa))
    P = Pa + Pb

    e_1b = np.trace(h1e @ P)
    e_J = 0.5 * np.einsum('pq,rs,pqrs->', P, P, eri)
    e_Ka = 0.5 * np.einsum('ps,rq,pqrs->', Pa, Pa, eri)
    e_Kb = 0.5 * np.einsum('ps,rq,pqrs->', Pb, Pb, eri)
    return e_1b + e_J - e_Ka - e_Kb + enuc


# ===================================================================
# Main comparison
# ===================================================================

def run_comparison(rs, N_elec, N_pw, polarization, dt, num_walkers,
                   num_blocks, verbose=True):
    """Run AFQMC vs FCI comparison for one set of parameters.

    Returns dict with FCI and AFQMC correlation energies.
    """
    system = build_3deg_system(rs, N_elec, N_pw, polarization)
    N_pw_actual = system['N_pw']
    nup, ndown = system['nup'], system['ndown']

    if verbose:
        print(f"\n  System: N={N_elec}, N_pw={N_pw_actual}, r_s={rs}, "
              f"pol={polarization}")
        is_closed, shell_msg = check_closed_shell(N_elec, N_pw, polarization)
        if not is_closed:
            print(f"  *** WARNING: {shell_msg}")

    # Integrals
    integrals = prepare_3deg_integrals(system)
    h1e = np.array(integrals['h1e'])
    chol = np.array(integrals['chol'])
    chol_sign = np.array(integrals['chol_sign'])
    enuc = integrals['e_madelung'] * N_elec

    # ERI
    eri = reconstruct_eri_from_cholesky(chol, chol_sign)

    # FCI
    fci_energies = run_fci(h1e, eri, nup, ndown, N_pw_actual,
                           enuc=enuc, n_roots=1)
    e_fci = fci_energies[0]

    # HF
    e_hf = compute_hf_energy(h1e, eri, nup, ndown, enuc)

    # AFQMC
    driver = get_afqmc_3deg_func(
        system, dt=dt, include_coulomb=True, verbose=verbose)
    result = driver(
        num_walkers=num_walkers, num_blocks=num_blocks,
        num_steps_per_block=25, num_blocks_equil=20)

    e_afqmc = result['energy_mean']
    e_afqmc_err = result['energy_err']
    e_trial = driver.e_trial

    ec_fci = e_fci - e_hf
    ec_afqmc = e_afqmc - e_trial

    return {
        'rs': rs, 'N_elec': N_elec, 'N_pw': N_pw_actual,
        'e_fci': e_fci, 'e_hf': e_hf, 'e_afqmc': e_afqmc,
        'e_afqmc_err': e_afqmc_err, 'e_trial': e_trial,
        'ec_fci': ec_fci, 'ec_afqmc': ec_afqmc,
    }


def main():
    parser = argparse.ArgumentParser(
        description="AFQMC vs FCI comparison for 3D electron gas")
    parser.add_argument('--N_elec', type=int, default=2,
                        help='Number of electrons (default: 2)')
    parser.add_argument('--N_pw', type=int, default=7,
                        help='Number of plane waves (default: 7)')
    parser.add_argument('--rs', type=float, default=1.0,
                        help='Wigner-Seitz radius (default: 1.0)')
    parser.add_argument('--polarization', type=str, default='unpolarized',
                        choices=['unpolarized', 'polarized'])
    parser.add_argument('--num_walkers', type=int, default=500)
    parser.add_argument('--num_blocks', type=int, default=200)
    parser.add_argument('--dt', type=float, default=0.005)
    parser.add_argument('--scan', action='store_true',
                        help='Run r_s scan from 0.5 to 10')
    args = parser.parse_args()

    N_elec = args.N_elec
    N_pw = args.N_pw
    rs = args.rs
    polarization = args.polarization

    print("=" * 70)
    print("  AFQMC vs Exact FCI: 3D Homogeneous Electron Gas")
    print("=" * 70)
    print(f"  N_elec       = {N_elec}")
    print(f"  N_pw         = {N_pw}")
    print(f"  r_s          = {rs}")
    print(f"  polarization = {polarization}")
    print(f"  num_walkers  = {args.num_walkers}")
    print(f"  num_blocks   = {args.num_blocks}")
    print(f"  dt           = {args.dt}")

    # --- Build system and integrals ---
    system = build_3deg_system(rs, N_elec, N_pw, polarization)
    N_pw_actual = system['N_pw']
    nup, ndown = system['nup'], system['ndown']
    print(f"  N_pw (actual) = {N_pw_actual}")
    print(f"  nup={nup}, ndown={ndown}")

    # Check closed-shell
    is_closed, shell_msg = check_closed_shell(N_elec, N_pw, polarization)
    if is_closed:
        print(f"  ** {shell_msg}")
    else:
        print(f"\n  *** WARNING: {shell_msg}")
        print(f"  *** AFQMC with single-determinant trial will have large "
              f"phaseless bias!\n")

    integrals = prepare_3deg_integrals(system)
    h1e = np.array(integrals['h1e'])
    chol = np.array(integrals['chol'])
    chol_sign = np.array(integrals['chol_sign'])
    e_madelung = integrals['e_madelung']
    enuc = e_madelung * N_elec

    print(f"  N_chol       = {len(chol)}")
    print(f"  e_madelung   = {e_madelung:.8f} Ha/elec")
    print(f"  E_nuc        = {enuc:.8f} Ha")

    # --- ERI reconstruction ---
    print("\n--- Reconstructing ERI tensor ---")
    eri = reconstruct_eri_from_cholesky(chol, chol_sign)
    sym_err = np.max(np.abs(eri - eri.transpose(2, 3, 0, 1)))
    print(f"  ERI shape: {eri.shape}")
    print(f"  (pq|rs) vs (rs|pq) symmetry: {sym_err:.2e}")

    # --- FCI ---
    print("\n--- Running FCI ---")
    t_fci_start = time.time()
    fci_energies = run_fci(h1e, eri, nup, ndown, N_pw_actual, enuc=enuc,
                           n_roots=min(5, 100))
    t_fci = time.time() - t_fci_start

    e_fci = fci_energies[0]
    print(f"\n  FCI ground state = {e_fci:.10f} Ha = {e_fci*2:.10f} Ry")
    print(f"  FCI E/N          = {e_fci/N_elec:.10f} Ha/elec")
    if len(fci_energies) > 1:
        print(f"  FCI gap (E1-E0)  = {fci_energies[1]-fci_energies[0]:.8f} Ha")

    # --- HF energy ---
    e_hf = compute_hf_energy(h1e, eri, nup, ndown, enuc)
    e_corr_fci = e_fci - e_hf

    print(f"\n  HF energy        = {e_hf:.10f} Ha")
    print(f"  FCI corr energy  = {e_corr_fci:.10f} Ha = {e_corr_fci*2:.10f} Ry")
    print(f"  FCI E_c/N        = {e_corr_fci/N_elec*2:.10f} Ry/elec")

    # --- AFQMC ---
    print("\n--- Running AFQMC ---")
    t_afqmc_start = time.time()
    driver = get_afqmc_3deg_func(
        system, dt=args.dt, include_coulomb=True, verbose=True)
    result = driver(
        num_walkers=args.num_walkers, num_blocks=args.num_blocks,
        num_steps_per_block=25, num_blocks_equil=20)
    t_afqmc = time.time() - t_afqmc_start

    e_afqmc = result['energy_mean']
    e_afqmc_err = result['energy_err']
    e_trial = driver.e_trial
    e_corr_afqmc = e_afqmc - e_trial

    # --- Comparison table ---
    print("\n" + "=" * 70)
    print("  COMPARISON: AFQMC vs FCI")
    print("=" * 70)
    print(f"  {'':30s} {'AFQMC':>16s} {'FCI':>16s}")
    print(f"  {'-'*62}")
    print(f"  {'Total energy (Ha)':30s} {e_afqmc:16.10f} {e_fci:16.10f}")
    print(f"  {'Total energy (Ry)':30s} {e_afqmc*2:16.10f} {e_fci*2:16.10f}")
    print(f"  {'HF/Trial energy (Ha)':30s} {e_trial:16.10f} {e_hf:16.10f}")
    print(f"  {'Correlation energy (Ha)':30s} "
          f"{e_corr_afqmc:16.10f} {e_corr_fci:16.10f}")
    print(f"  {'Correlation energy (Ry)':30s} "
          f"{e_corr_afqmc*2:16.10f} {e_corr_fci*2:16.10f}")
    print(f"  {'E_c/N (Ry/elec)':30s} "
          f"{e_corr_afqmc/N_elec*2:16.10f} {e_corr_fci/N_elec*2:16.10f}")
    print(f"  {'-'*62}")

    if abs(e_corr_fci) > 1e-10:
        ratio = e_corr_afqmc / e_corr_fci
        diff_ha = e_afqmc - e_fci
        diff_sigma = (abs(diff_ha) / e_afqmc_err
                      if e_afqmc_err > 0 else float('inf'))
        print(f"  {'AFQMC/FCI ratio':30s} {ratio:16.6f}")
        print(f"  {'E_AFQMC - E_FCI (Ha)':30s} {diff_ha:16.10f}")
        print(f"  {'|diff| / sigma':30s} {diff_sigma:16.2f}")
        print(f"\n  AFQMC captures {ratio*100:.2f}% of FCI correlation energy")

    print(f"\n  FCI runtime:   {t_fci:.1f} s")
    print(f"  AFQMC runtime: {t_afqmc:.1f} s")

    # --- r_s scan ---
    if args.scan:
        print("\n" + "=" * 70)
        print("  SCAN: AFQMC vs FCI across r_s values")
        print("=" * 70)
        rs_list = [0.5, 1.0, 2.0, 5.0, 10.0]
        print(f"  {'r_s':>6s} {'E_c FCI (Ry)':>14s} {'E_c AFQMC (Ry)':>16s} "
              f"{'Ratio':>8s} {'Err (Ry)':>10s}")
        print(f"  {'-'*58}")

        for rs_val in rs_list:
            r = run_comparison(
                rs_val, N_elec, N_pw, polarization,
                args.dt, args.num_walkers, args.num_blocks, verbose=False)

            ec_fci_ry = r['ec_fci'] / N_elec * 2
            ec_afqmc_ry = r['ec_afqmc'] / N_elec * 2
            ec_err_ry = r['e_afqmc_err'] / N_elec * 2
            ratio = (ec_afqmc_ry / ec_fci_ry
                     if abs(ec_fci_ry) > 1e-12 else 0)

            print(f"  {rs_val:6.1f} {ec_fci_ry:14.8f} {ec_afqmc_ry:16.8f} "
                  f"{ratio:8.4f} {ec_err_ry:10.8f}")

    print("\nDone.")


if __name__ == '__main__':
    main()
