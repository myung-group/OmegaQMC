"""
General-N transcorrelated FCI, 2-body terms (step 3a).

Extends the verified general-N FCI (tc_fci_general) with the spin-dependent
2-body TC correction: opposite-spin pairs use cusp a=1/2, same-spin pairs
a=1/4 (Kato). The genuinely new N>2 piece -- the three-body (grad u)^2
cross-term -- is NOT yet included here; this isolates the 2-body contribution
so the 3-body increment can be measured against it next.

Gates: G1 (bare det-FCI == PySCF FCI), G2 (u=0 -> bare), G3 (real eigenvalue).

  python examples/run_tc_fci_nbody.py \
      --atom "H 0 0 0; H 0 0 1; H 0 0 2; H 0 0 3" --unit Angstrom \
      --basis 6-31g --na 2 --nb 2 --u-b 0.5 --grid-level 1
"""

import argparse
import numpy as np
from pyscf import gto, scf, fci, ao2mo

from tc_fci_general import enumerate_dets, build_fci_matrix
from tc_fci_2e import (
    u_profiles, compute_tc_corrections, _mo_on_grid, ground_state,
)


def cbs_estimate(atom, unit):
    e = {}
    for bas in ("cc-pvdz", "cc-pvtz"):
        m = gto.M(atom=atom, unit=unit, basis=bas, verbose=0)
        mf = scf.RHF(m).run(verbose=0)
        e[bas] = fci.FCI(m, mf.mo_coeff).kernel()[0]
    # 2-pt X^-3 extrapolation (DZ=2, TZ=3)
    e_cbs = (27.0 * e["cc-pvtz"] - 8.0 * e["cc-pvdz"]) / 19.0
    return e["cc-pvdz"], e["cc-pvtz"], e_cbs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--atom", default="H 0 0 0; H 0 0 1; H 0 0 2; H 0 0 3")
    p.add_argument("--unit", default="Angstrom")
    p.add_argument("--basis", default="6-31g")
    p.add_argument("--na", type=int, required=True)
    p.add_argument("--nb", type=int, required=True)
    p.add_argument("--u-b", type=float, default=0.5)
    p.add_argument("--grid-level", type=int, default=1)
    p.add_argument("--cbs", action="store_true", help="also compute CBS ref")
    args = p.parse_args()

    mol = gto.M(atom=args.atom, unit=args.unit, basis=args.basis, verbose=0)
    mf = scf.RHF(mol).run()
    C = np.asarray(mf.mo_coeff); n = C.shape[1]
    e_nuc = mol.energy_nuc()
    h1 = C.T @ mf.get_hcore() @ C
    g_phys = np.transpose(ao2mo.restore(1, ao2mo.kernel(mol, C), n),
                          (0, 2, 1, 3))
    dets = enumerate_dets(n, args.na, args.nb)
    e_fci_pyscf = fci.FCI(mol, mf.mo_coeff).kernel(nelec=(args.na, args.nb))[0]
    print(f"system: {n} orb ({args.na},{args.nb}) e-, |dets|={len(dets)}, "
          f"basis={args.basis}")

    # [G1] bare det-FCI.
    Hb = build_fci_matrix(h1, g_phys, n, dets)
    e_bare, _ = ground_state(Hb, e_nuc)
    print(f"[G1] bare det-FCI {e_bare:.8f} vs PySCF {e_fci_pyscf:.8f}  "
          f"{'PASS' if abs(e_bare-e_fci_pyscf)<1e-6 else 'FAIL'}")

    grid = _mo_on_grid(mol, C, args.grid_level)

    # [G2] u=0 -> bare.
    z = u_profiles("cusp", 0.0, args.u_b)
    dV0 = compute_tc_corrections(mol, C, z, args.grid_level, grid=grid)
    H0 = build_fci_matrix(h1, g_phys + dV0, n, dets, g_same=g_phys + dV0)
    e0, _ = ground_state(H0, e_nuc)
    print(f"[G2] u=0 TC-FCI {e0:.8f} vs bare {e_bare:.8f}  "
          f"{'PASS' if abs(e0-e_bare)<1e-6 else 'FAIL'}")

    # 2-body TC: opposite-spin cusp a=1/2, same-spin a=1/4.
    dV_opp = compute_tc_corrections(
        mol, C, u_profiles("cusp", 0.5, args.u_b), args.grid_level, grid=grid)
    dV_same = compute_tc_corrections(
        mol, C, u_profiles("cusp", 0.25, args.u_b), args.grid_level, grid=grid)
    Htc = build_fci_matrix(h1, g_phys + dV_opp, n, dets,
                           g_same=g_phys + dV_same)
    e_tc, imag = ground_state(Htc, e_nuc)
    print(f"[G3] TC(2-body) imag {imag:.1e}  "
          f"{'PASS' if imag<1e-6 else 'WARN'}")

    print(f"\n=== 2-body TC-FCI (cusp a=1/2 | 1/4, b={args.u_b}, "
          f"grid L{args.grid_level}) ===")
    print(f"  FCI({args.basis})        = {e_bare:.6f}  (bare ceiling)")
    print(f"  2-body TC-FCI({args.basis}) = {e_tc:.6f}")
    print(f"  recovered vs ceiling    = {(e_tc-e_bare)*1e3:+.3f} mE_h")
    if args.cbs:
        e_dz, e_tz, e_cbs = cbs_estimate(args.atom, args.unit)
        print(f"  FCI(cc-pVDZ)={e_dz:.6f}  FCI(cc-pVTZ)={e_tz:.6f}  "
              f"CBS~={e_cbs:.6f}")
        print(f"  2-body TC-FCI - CBS     = {(e_tc-e_cbs)*1e3:+.3f} mE_h "
              f"(3-body term, added next, should close more)")


if __name__ == "__main__":
    main()
