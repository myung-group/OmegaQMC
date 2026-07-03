"""Derive QED-CCSD residual equations with density-fitted (3-index) integrals.

Uses the improved Wick (/Users/willow/Python/wick) with ``two_e_df`` so the
generated equations contain only the 3-index DF factor B (no nmo^4 tensor).

Hamiltonian (coherent-state / QED-HF basis, single cavity mode):

    H = f_pq p+ q                          (QED-HF Fock, DSE folded into B)
      + 1/2 sum_x B_{x,pr} B_{x,qs} p+ q+ s r
      + w b+ b
      + d_pq (b+ + b) p+ q                 (bilinear coupling, d = -dip)

Cluster operator (QED-CCSD-22, full):

    T = t1 + t2 + s1 + s2 + u11 + u21 + u12 + u22

Naming matches OmegaQMC/addons/qed_ccsd.py:
    t1 = t1_10, t2 = t2_20, s1 = t1_01, s2 = t2_02,
    u11 = t2_11, u21 = t2_21, u12 = t2_12, u22 = t2_22.

Usage:  python derive_qed_ccsd_df.py <target>
        target in {energy, t1_10, t2_20, t1_01, t2_02,
                   t2_11, t2_21, t2_12, t2_22}
"""
import sys
import time
from fractions import Fraction

from wick.index import Idx
from wick.operator import BOperator, FOperator, Tensor, Sigma, TensorSym
from wick.expression import Term, Expression, AExpression
from wick.wick import apply_wick
from wick.convenience import (
    one_e, two_e_df, two_p, ep11,
    E1, E2, P1, P2, EPS1, EPS2,
    braE1, braE2, braP1, braP2, braP1E1, braP2E1,
    commute)

O = ["occ"]
V = ["vir"]
NM = ["nm"]


# --------------------------------------------------------------------------
# Cluster operators missing from wick.convenience: boson (x doubled boson)
# coupled to a fermionic double excitation.
# --------------------------------------------------------------------------
def EP1E2(name):
    """u_{x,abij} b+_x a+ b+ j i   (one boson x double excitation)."""
    sym = TensorSym(
        [(0, 1, 2, 3, 4), (0, 2, 1, 3, 4), (0, 1, 2, 4, 3), (0, 2, 1, 4, 3)],
        [1, -1, -1, 1])
    x = Idx(0, "nm", fermion=False)
    i = Idx(0, "occ")
    a = Idx(0, "vir")
    j = Idx(1, "occ")
    b = Idx(1, "vir")
    sums = [Sigma(x), Sigma(i), Sigma(a), Sigma(j), Sigma(b)]
    tensors = [Tensor([x, a, b, i, j], name, sym=sym)]
    operators = [BOperator(x, True),
                 FOperator(a, True), FOperator(b, True),
                 FOperator(j, False), FOperator(i, False)]
    return Expression([Term(Fraction(1, 4), sums, tensors, operators, [])])


def EP2E2(name):
    """1/2 u_{xy,abij} b+_x b+_y a+ b+ j i  (two bosons x double excitation)."""
    fperm = [((0, 1), 1), ((1, 0), 1)]          # boson exchange, symmetric
    aperm = [((2, 3, 4, 5), 1), ((3, 2, 4, 5), -1),
             ((2, 3, 5, 4), -1), ((3, 2, 5, 4), 1)]
    perms, signs = [], []
    for bp, bs in fperm:
        for fp, fs in aperm:
            perms.append(bp + fp)
            signs.append(bs * fs)
    sym = TensorSym(perms, signs)
    x = Idx(0, "nm", fermion=False)
    y = Idx(1, "nm", fermion=False)
    i = Idx(0, "occ")
    a = Idx(0, "vir")
    j = Idx(1, "occ")
    b = Idx(1, "vir")
    sums = [Sigma(x), Sigma(y), Sigma(i), Sigma(a), Sigma(j), Sigma(b)]
    tensors = [Tensor([x, y, a, b, i, j], name, sym=sym)]
    operators = [BOperator(x, True), BOperator(y, True),
                 FOperator(a, True), FOperator(b, True),
                 FOperator(j, False), FOperator(i, False)]
    return Expression([Term(Fraction(1, 8), sums, tensors, operators, [])])


def braP1E2():
    """<0| b_x  i+ j+ b a  — projection for the u21 residual."""
    x = Idx(0, "nm", fermion=False)
    i = Idx(0, "occ")
    a = Idx(0, "vir")
    j = Idx(1, "occ")
    b = Idx(1, "vir")
    operators = [BOperator(x, False),
                 FOperator(i, True), FOperator(j, True),
                 FOperator(b, False), FOperator(a, False)]
    tensors = [Tensor([x, a, b, i, j], "")]
    return Expression([Term(1, [], tensors, operators, [])])


def braP2E2():
    """<0| b_x b_y  i+ j+ b a  — projection for the u22 residual."""
    x = Idx(0, "nm", fermion=False)
    y = Idx(1, "nm", fermion=False)
    i = Idx(0, "occ")
    a = Idx(0, "vir")
    j = Idx(1, "occ")
    b = Idx(1, "vir")
    operators = [BOperator(x, False), BOperator(y, False),
                 FOperator(i, True), FOperator(j, True),
                 FOperator(b, False), FOperator(a, False)]
    tensors = [Tensor([x, y, a, b, i, j], "")]
    return Expression([Term(1, [], tensors, operators, [])])


def main():
    target = sys.argv[1]
    t0 = time.time()

    # ---- Hamiltonian ----
    H1 = one_e("f", ["occ", "vir"], norder=True)
    H2 = two_e_df("B", ["occ", "vir"], norder=True)
    Hp = two_p("w")
    Hep = ep11("d", ["occ", "vir"], NM, norder=True)
    H = H1 + H2 + Hp + Hep

    # ---- Cluster operator ----
    T1 = E1("t1", O, V)
    T2 = E2("t2", O, V)
    S1 = P1("s1", NM)
    S2 = P2("s2", NM)
    U11 = EPS1("u11", NM, O, V)
    U12 = EPS2("u12", NM, O, V)
    U21 = EP1E2("u21")
    U22 = EP2E2("u22")
    T = T1 + T2 + S1 + S2 + U11 + U12 + U21 + U22

    # ---- BCH expansion ----
    # Quadratic: full. Cubic: only H2 (4 fermion lines) and Hep (3 lines)
    # survive three nested commutators; f and w b+b have too few lines.
    # Quartic: only H2 with four single-fermion-pair pieces (t1, u11, u12).
    HT = commute(H, T)
    HTT = commute(HT, T)
    print(f"# HTT done ({time.time()-t0:.1f} s)", file=sys.stderr)
    H3 = H2 + Hep
    HTTT = commute(commute(commute(H3, T), T), T)
    print(f"# HTTT done ({time.time()-t0:.1f} s)", file=sys.stderr)
    Tq = T1 + U11 + U12
    HTTTT = commute(commute(commute(commute(H2, Tq), Tq), Tq), Tq)
    print(f"# HTTTT done ({time.time()-t0:.1f} s)", file=sys.stderr)

    Hbar = (H + HT + Fraction(1, 2) * HTT + Fraction(1, 6) * HTTT
            + Fraction(1, 24) * HTTTT)

    bras = {
        "energy": None,
        "t1_10": braE1("occ", "vir"),
        "t2_20": braE2("occ", "vir", "occ", "vir"),
        "t1_01": braP1("nm"),
        "t2_02": braP2("nm"),
        "t2_11": braP1E1("nm", "occ", "vir"),
        "t2_21": braP1E2(),
        "t2_12": braP2E1("nm", "nm", "occ", "vir"),
        "t2_22": braP2E2(),
    }
    bra = bras[target]
    S = Hbar if bra is None else bra * Hbar
    out = apply_wick(S)
    out.resolve()
    final = AExpression(Ex=out)
    print(f"# wick done, {len(final.terms)} terms ({time.time()-t0:.1f} s)",
          file=sys.stderr)
    print(final._print_einsum(f"res_{target}", optimize=True))


if __name__ == "__main__":
    main()
