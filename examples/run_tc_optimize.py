"""
Optimize the transcorrelation factor for 2-electron TC-FCI.

Scans the cusp Pade factor u(r) = a r/(1+b r) and for each (a,b) reports:
  * TC-FCI(basis) energy (full non-Hermitian diagonalisation),
  * distance to the CBS limit (TZ/QZ X^-3 extrapolation), and
  * the Haupt et al. TC-reference variance
        sigma2_ref = sum_{i != HF} |<D_i|H~|D_HF>|^2,
    the CBS-free criterion for the TC-optimal Jastrow.

Goal: how close to CBS can a tuned cusp factor drive TC-FCI in a small basis,
and does minimizing sigma2_ref pick (near) the same factor.

  python examples/run_tc_optimize.py --atom "H 0 0 0; H 0 0 1.4" \
      --unit Angstrom --basis cc-pvdz --grid-level 1
"""

import argparse
import numpy as np
from pyscf import gto, scf, fci, ao2mo

from tc_fci_2e import (
    build_2e_hamiltonian, ground_state, u_profiles,
    compute_tc_corrections, _mo_on_grid,
)


def fci_energy(atom, unit, basis):
    mol = gto.M(atom=atom, unit=unit, basis=basis, verbose=0)
    mf = scf.RHF(mol).run(verbose=0)
    e, _ = fci.FCI(mol, mf.mo_coeff).kernel()
    return e


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--atom", default="H 0 0 0; H 0 0 1.4")
    p.add_argument("--unit", default="Angstrom")
    p.add_argument("--basis", default="cc-pvdz")
    p.add_argument("--grid-level", type=int, default=1)
    p.add_argument("--a-list", default="0.3,0.4,0.5,0.6")
    p.add_argument("--b-list", default="0.2,0.3,0.5,0.8,1.2,2.0")
    args = p.parse_args()

    a_list = [float(x) for x in args.a_list.split(",")]
    b_list = [float(x) for x in args.b_list.split(",")]

    # CBS reference from TZ/QZ X^-3 extrapolation of the total FCI energy.
    e_dz = fci_energy(args.atom, args.unit, args.basis)
    e_tz = fci_energy(args.atom, args.unit, "cc-pvtz")
    e_qz = fci_energy(args.atom, args.unit, "cc-pvqz")
    e_cbs = (64.0 * e_qz - 27.0 * e_tz) / 37.0
    print(f"FCI({args.basis})={e_dz:.6f}  FCI(TZ)={e_tz:.6f}  "
          f"FCI(QZ)={e_qz:.6f}  CBS(X^-3)={e_cbs:.6f}")
    print(f"beyond-{args.basis} correlation to recover: "
          f"{(e_cbs - e_dz)*1e3:.2f} mE_h\n")

    # Bare integrals + grid, computed once.
    mol = gto.M(atom=args.atom, unit=args.unit, basis=args.basis, verbose=0)
    mf = scf.RHF(mol).run(verbose=0)
    C = np.asarray(mf.mo_coeff)
    n = C.shape[1]
    e_nuc = mol.energy_nuc()
    h1 = C.T @ mf.get_hcore() @ C
    eri = ao2mo.restore(1, ao2mo.kernel(mol, C), n)
    g_phys = np.transpose(eri, (0, 2, 1, 3))
    grid = _mo_on_grid(mol, C, args.grid_level)
    ref_idx = 0  # HF determinant ((0,),(0,)) is index 0

    print(f"{'a':>5} {'b':>5} {'TC-FCI':>12} {'err_CBS/mEh':>12} "
          f"{'sigma2_ref':>12}")
    rows = []
    for a in a_list:
        for b in b_list:
            up = u_profiles("cusp", a, b)
            dV = compute_tc_corrections(mol, C, up, args.grid_level, grid=grid)
            Htc = build_2e_hamiltonian(h1, g_phys + dV, n)
            e_tc, imag = ground_state(Htc, e_nuc)
            col = Htc[:, ref_idx].copy()
            col[ref_idx] = 0.0
            sigma2 = float(np.sum(col ** 2))
            err = (e_tc - e_cbs) * 1e3
            rows.append((a, b, e_tc, err, sigma2, imag))
            flag = "*" if imag > 1e-6 else " "
            print(f"{a:5.2f} {b:5.2f} {e_tc:12.6f} {err:12.3f} "
                  f"{sigma2:12.5f}{flag}")

    # Best factor by closeness to CBS (from above) and by Haupt sigma2_ref.
    valid = [r for r in rows if r[5] < 1e-6]
    best_cbs = min(valid, key=lambda r: abs(r[3]))
    best_sig = min(valid, key=lambda r: r[4])
    print(f"\nclosest to CBS:  a={best_cbs[0]}, b={best_cbs[1]}  "
          f"TC-FCI={best_cbs[2]:.6f}  err={best_cbs[3]:+.3f} mE_h "
          f"(recovered {(best_cbs[2]-e_dz)*1e3:+.2f} of "
          f"{(e_cbs-e_dz)*1e3:.2f} mE_h)")
    print(f"min sigma2_ref:  a={best_sig[0]}, b={best_sig[1]}  "
          f"TC-FCI={best_sig[2]:.6f}  err={best_sig[3]:+.3f} mE_h "
          f"(sigma2={best_sig[4]:.5f})")


if __name__ == "__main__":
    main()
