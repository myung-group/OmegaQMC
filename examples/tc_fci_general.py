"""
General-N transcorrelated FCI foundation (step 3, scaling beyond 2 electrons).

Spin-orbital determinant FCI that accepts general (possibly non-symmetric)
1- and 2-body integrals, so the non-Hermitian TC corrections can be layered in.
This module establishes and verifies the general-N machinery (Gate G1: bare
FCI == PySCF FCI); the spin-dependent 2-body TC corrections (cusp 1/2 for
opposite spin, 1/4 for same spin) and the three-body (grad u)^2 term build on
this scaffold next.

  python examples/tc_fci_general.py --atom "H 0 0 0; H 0 0 1; H 0 0 2; H 0 0 3" \
      --unit Angstrom --basis sto-3g --na 2 --nb 2
"""

import argparse
from itertools import combinations

import numpy as np
import scipy.linalg
from pyscf import gto, scf, fci, ao2mo


def enumerate_dets(n_orb, na, nb):
    """Determinants as (occ_alpha tuple, occ_beta tuple)."""
    A = list(combinations(range(n_orb), na))
    B = list(combinations(range(n_orb), nb))
    return [(a, b) for a in A for b in B]


def enumerate_cas_dets(n_core, n_act, n_act_a, n_act_b):
    """CASCI determinants in the core+active subspace: orbitals 0..n_core-1
    doubly occupied (frozen), n_act_a/n_act_b active electrons distributed
    over the active orbitals n_core..n_core+n_act-1. Virtuals (outside the
    subspace) are absent. Feeding these to build_fci_matrix gives frozen-core
    CASCI automatically (core sits in every determinant)."""
    core = tuple(range(n_core))
    act = range(n_core, n_core + n_act)
    A = list(combinations(act, n_act_a))
    B = list(combinations(act, n_act_b))
    return [(tuple(sorted(core + a)), tuple(sorted(core + b)))
            for a in A for b in B]


def _excite(occ, i, a):
    """Apply a_a^dag a_i to an occupation tuple; return (sign, new_occ) or
    (0, None). i must be occupied, a must be empty (unless a==i)."""
    if i not in occ:
        return 0, None
    if a in occ and a != i:
        return 0, None
    lst = list(occ)
    pos_i = lst.index(i)
    # sign from annihilating i: (-1)^(# occupied before i)
    sign = (-1) ** pos_i
    lst.pop(pos_i)
    # insert a in sorted order; sign from creation
    pos_a = 0
    while pos_a < len(lst) and lst[pos_a] < a:
        pos_a += 1
    sign *= (-1) ** pos_a
    lst.insert(pos_a, a)
    return sign, tuple(lst)


def apply_op(occ, ann, cre):
    """Apply (creations cre, left) (annihilations ann, right) to |occ>,
    operators acting right-to-left in the given list order. Returns
    (sign, new_occ) or (0, None). Handles all index coincidences correctly."""
    lst = list(occ)
    sign = 1
    for o in ann:
        if o not in lst:
            return 0, None
        pos = lst.index(o)
        sign *= (-1) ** pos
        lst.pop(pos)
    for o in cre:
        if o in lst:
            return 0, None
        pos = 0
        while pos < len(lst) and lst[pos] < o:
            pos += 1
        sign *= (-1) ** pos
        lst.insert(pos, o)
    return sign, tuple(lst)


def build_fci_matrix(h1, g_ab, n_orb, dets, g_same=None):
    """H matrix over `dets` via spin-orbital Slater-Condon, accepting separate
    opposite-spin (`g_ab`) and same-spin (`g_same`) 2-body tensors
    (physicist <pq|rs>). For the bare/spin-free Hamiltonian pass
    g_same=None (defaults to g_ab). The tensors may be non-symmetric (the
    non-Hermitian TC drift term), so H is in general non-Hermitian.
    """
    if g_same is None:
        g_same = g_ab
    idx = {d: I for I, d in enumerate(dets)}
    ndet = len(dets)
    H = np.zeros((ndet, ndet))

    for J, (oa, ob) in enumerate(dets):
        # --- 1-body: sum_{pq} h_pq (a_p^dag a_q) on alpha and beta ---
        for (occ, other, is_a) in ((oa, ob, True), (ob, oa, False)):
            for q in occ:
                for p in range(n_orb):
                    s, new = _excite(occ, q, p)
                    if s == 0:
                        continue
                    det = (new, ob) if is_a else (oa, new)
                    I = idx.get(det)
                    if I is not None:
                        H[I, J] += s * h1[p, q]
        # --- 2-body: 1/2 sum <pq|rs> a_p^dag a_q^dag a_s a_r ---
        # Split by spin channels: aa, bb (same spin) and ab/ba (opposite).
        # Opposite spin alpha-beta (no exchange)
        for r in oa:
            for s_ in ob:
                for p in range(n_orb):
                    s1, na_new = _excite(oa, r, p)
                    if s1 == 0:
                        continue
                    for q in range(n_orb):
                        s2, nb_new = _excite(ob, s_, q)
                        if s2 == 0:
                            continue
                        I = idx.get((na_new, nb_new))
                        if I is None:
                            continue
                        H[I, J] += s1 * s2 * g_ab[p, q, r, s_]
        # Same spin (a_p^dag a_q^dag a_s a_r over ordered pairs p<q, r<s,
        # antisymmetrised integral) via correct right-to-left operator action.
        for (occ, fixed, is_a) in ((oa, ob, True), (ob, oa, False)):
            for ri in range(len(occ)):
                for si in range(ri + 1, len(occ)):
                    r, s_ = occ[ri], occ[si]
                    for p in range(n_orb):
                        for q in range(p + 1, n_orb):
                            sgn, new = apply_op(occ, [r, s_], [q, p])
                            if sgn == 0:
                                continue
                            det = (new, fixed) if is_a else (fixed, new)
                            I = idx.get(det)
                            if I is None:
                                continue
                            val = g_same[p, q, r, s_] - g_same[p, q, s_, r]
                            H[I, J] += sgn * val
    return H


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--atom", default="H 0 0 0; H 0 0 1; H 0 0 2; H 0 0 3")
    p.add_argument("--unit", default="Angstrom")
    p.add_argument("--basis", default="sto-3g")
    p.add_argument("--na", type=int, required=True)
    p.add_argument("--nb", type=int, required=True)
    args = p.parse_args()

    mol = gto.M(atom=args.atom, unit=args.unit, basis=args.basis, verbose=0)
    mf = scf.RHF(mol).run()
    C = np.asarray(mf.mo_coeff); n = C.shape[1]
    e_nuc = mol.energy_nuc()
    h1 = C.T @ mf.get_hcore() @ C
    eri_phys = np.transpose(ao2mo.restore(1, ao2mo.kernel(mol, C), n),
                            (0, 2, 1, 3))

    dets = enumerate_dets(n, args.na, args.nb)
    H = build_fci_matrix(h1, eri_phys, n, dets)
    w = scipy.linalg.eigvalsh(H)
    e_mine = float(w[0]) + e_nuc

    e_pyscf, _ = fci.FCI(mol, mf.mo_coeff).kernel(nelec=(args.na, args.nb))
    print(f"system: {n} orbitals, ({args.na},{args.nb}) e-, "
          f"|dets|={len(dets)}")
    print(f"[G1] general-N det-FCI {e_mine:.8f}  vs PySCF {e_pyscf:.8f}  "
          f"diff {abs(e_mine - e_pyscf)*1e6:.2f} uE_h  "
          f"{'PASS' if abs(e_mine - e_pyscf) < 1e-6 else 'FAIL'}")


if __name__ == "__main__":
    main()
