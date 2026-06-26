"""
Full TC-FCI with 2-body + 3-body terms (step 3 complete) on a small system.

  python examples/run_tc_3body.py \
      --atom "H 0 0 0; H 0 0 1; H 0 0 2; H 0 0 3" --unit Angstrom \
      --basis sto-3g --na 2 --nb 2 --u-b 0.8 --grid-level 1
"""

import argparse
import numpy as np
from pyscf import gto, scf, fci, ao2mo

from tc_fci_general import enumerate_dets, build_fci_matrix
from tc_fci_2e import u_profiles, compute_tc_corrections, _mo_on_grid, ground_state
from tc_3body import build_3body_fci


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--atom", default="H 0 0 0; H 0 0 1; H 0 0 2; H 0 0 3")
    p.add_argument("--unit", default="Angstrom")
    p.add_argument("--basis", default="sto-3g")
    p.add_argument("--na", type=int, required=True)
    p.add_argument("--nb", type=int, required=True)
    p.add_argument("--u-b", type=float, default=0.8)
    p.add_argument("--grid-level", type=int, default=1)
    args = p.parse_args()

    mol = gto.M(atom=args.atom, unit=args.unit, basis=args.basis, verbose=0)
    mf = scf.RHF(mol).run()
    C = np.asarray(mf.mo_coeff); n = C.shape[1]
    e_nuc = mol.energy_nuc()
    h1 = C.T @ mf.get_hcore() @ C
    g_phys = np.transpose(ao2mo.restore(1, ao2mo.kernel(mol, C), n), (0, 2, 1, 3))
    dets = enumerate_dets(n, args.na, args.nb)
    e_fci = fci.FCI(mol, mf.mo_coeff).kernel(nelec=(args.na, args.nb))[0]
    grid = _mo_on_grid(mol, C, args.grid_level)
    print(f"system {n} orb ({args.na},{args.nb})e-, |dets|={len(dets)}, "
          f"basis={args.basis}, b={args.u_b}")

    # 2-body TC (spin-resolved cusp).
    dV_opp = compute_tc_corrections(mol, C, u_profiles("cusp", 0.5, args.u_b),
                                    args.grid_level, grid=grid)
    dV_same = compute_tc_corrections(mol, C, u_profiles("cusp", 0.25, args.u_b),
                                     args.grid_level, grid=grid)
    H2 = build_fci_matrix(h1, g_phys + dV_opp, n, dets, g_same=g_phys + dV_same)
    e_2, _ = ground_state(H2, e_nuc)

    # 3-body term.
    H3 = build_3body_fci(mol, C, args.u_b, dets, grid)
    herm_err = float(np.max(np.abs(H3 - H3.T)))
    print(f"[3b-Herm] max|H3 - H3^T| = {herm_err:.2e}  "
          f"{'PASS' if herm_err < 1e-9 else 'FAIL'}  (real 3-body op => symmetric)")

    Htot = H2 + H3
    e_tot, imag = ground_state(Htot, e_nuc)
    print(f"[G3] total spectrum imag = {imag:.1e}  "
          f"{'PASS' if imag < 1e-6 else 'WARN'}")

    print(f"\n=== TC-FCI 2-body vs 2+3-body ({args.basis}) ===")
    print(f"  bare FCI            = {e_fci:.6f}")
    print(f"  2-body TC-FCI       = {e_2:.6f}   ({(e_2-e_fci)*1e3:+.2f} mE_h)")
    print(f"  2+3-body TC-FCI     = {e_tot:.6f}   ({(e_tot-e_fci)*1e3:+.2f} mE_h)")
    print(f"  3-body contribution = {(e_tot-e_2)*1e3:+.3f} mE_h")


if __name__ == "__main__":
    main()
