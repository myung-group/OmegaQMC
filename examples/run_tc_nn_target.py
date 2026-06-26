"""
Step 2: NN-informed transcorrelation factor selection.

TC is non-variational, so factor selection cannot be "minimise the energy".
For H2 we showed a tuned cusp factor reaches CBS, but picking it needs the CBS
target -- which is unknown in general. The NN supplies it: a converged NN-VMC
energy is essentially CBS (real-space, basis-free). So we select the Kato-cusp
range b (a fixed at the physical 1/2) such that

    TC-FCI(small basis; a=1/2, b) = E_NN,

a 1-D root find. The result is a deterministic, CI-structured wavefunction at
the NN's (near-CBS) accuracy, obtained in the small basis -- the NN tells TC
where to stop, replacing the CBS extrapolation you otherwise could not do for
larger systems.

  python examples/run_tc_nn_target.py --atom "H 0 0 0; H 0 0 1.4" \
      --unit Angstrom --basis cc-pvdz --e-nn -1.082177 --grid-level 2
"""

import argparse
import numpy as np
from pyscf import gto, scf, fci, ao2mo

from tc_fci_2e import (
    build_2e_hamiltonian, ground_state, u_profiles,
    compute_tc_corrections, _mo_on_grid,
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--atom", default="H 0 0 0; H 0 0 1.4")
    p.add_argument("--unit", default="Angstrom")
    p.add_argument("--basis", default="cc-pvdz")
    p.add_argument("--e-nn", type=float, required=True,
                   help="converged NN-VMC energy (the near-CBS target)")
    p.add_argument("--a", type=float, default=0.5, help="Kato cusp slope")
    p.add_argument("--grid-level", type=int, default=2)
    p.add_argument("--b-lo", type=float, default=0.15)
    p.add_argument("--b-hi", type=float, default=0.6)
    args = p.parse_args()

    mol = gto.M(atom=args.atom, unit=args.unit, basis=args.basis, verbose=0)
    mf = scf.RHF(mol).run(verbose=0)
    C = np.asarray(mf.mo_coeff); n = C.shape[1]
    e_nuc = mol.energy_nuc()
    h1 = C.T @ mf.get_hcore() @ C
    g_phys = np.transpose(ao2mo.restore(1, ao2mo.kernel(mol, C), n), (0, 2, 1, 3))
    e_fci = fci.FCI(mol, mf.mo_coeff).kernel()[0]
    grid = _mo_on_grid(mol, C, args.grid_level)

    def tc_energy(b):
        up = u_profiles("cusp", args.a, b)
        dV = compute_tc_corrections(mol, C, up, args.grid_level, grid=grid)
        H = build_2e_hamiltonian(h1, g_phys + dV, n)
        e, imag = ground_state(H, e_nuc)
        return e

    print(f"FCI({args.basis}) = {e_fci:.6f}   E_NN (target) = {args.e_nn:.6f}")
    print(f"beyond-basis gap NN-FCI = {(args.e_nn - e_fci)*1e3:.2f} mE_h")
    print(f"root-finding b (a={args.a}) so TC-FCI = E_NN ...")

    # Bisection: TC-FCI decreases as b decreases (stronger factor).
    lo, hi = args.b_lo, args.b_hi
    e_lo, e_hi = tc_energy(lo), tc_energy(hi)
    print(f"  b={lo:.3f} -> {e_lo:.6f}   b={hi:.3f} -> {e_hi:.6f}")
    if not (e_lo <= args.e_nn <= e_hi):
        print("  WARNING: target not bracketed; widen --b-lo/--b-hi")
    b = None
    for _ in range(30):
        mid = 0.5 * (lo + hi)
        em = tc_energy(mid)
        if abs(em - args.e_nn) < 2e-5:
            b = mid; break
        if em < args.e_nn:   # too low -> increase b (weaker)
            lo = mid
        else:
            hi = mid
        b = mid
    e_b = tc_energy(b)
    print(f"\n=== NN-informed TC factor ===")
    print(f"  selected b* = {b:.4f}  (a = {args.a}, Kato cusp)")
    print(f"  TC-FCI(cc-pVDZ; a,b*) = {e_b:.6f}")
    print(f"  E_NN                  = {args.e_nn:.6f}")
    print(f"  match                 = {(e_b - args.e_nn)*1e6:+.1f} uE_h")
    print(f"  recovered beyond-basis = {(e_b - e_fci)*1e3:+.2f} mE_h "
          f"of {(args.e_nn - e_fci)*1e3:.2f} mE_h "
          f"-> deterministic CI-structured state at NN/near-CBS accuracy")


if __name__ == "__main__":
    main()
