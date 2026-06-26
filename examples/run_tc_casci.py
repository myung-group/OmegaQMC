"""
TC-CASCI: transcorrelated CI in a valence active space (step 3, scalable).

Restricts the general-N TC-FCI to a core+active subspace: core orbitals frozen
doubly-occupied, CI over the active space. This is the route to systems whose
full TC-FCI is intractable (BeH2/cc-pVDZ = 4M dets). 2-body TC only -- justified
by the ANOVA distillation showing J_NN's 3-body content is ~0.04%.

Gates: G1 (bare CASCI == PySCF mcscf.CASCI), G2 (u=0 -> bare), G3 (real).

  python examples/run_tc_casci.py --atom "Be 0 0 0; H 0 0 2.5133; H 0 0 -2.5133" \
      --unit Bohr --basis cc-pvdz --ncas 8 --nelecas 3 3 --u-b 0.3 --grid-level 1
"""

import argparse
import numpy as np
from pyscf import gto, scf, ao2mo, mcscf

from tc_fci_general import enumerate_cas_dets, build_fci_matrix
from tc_fci_2e import u_profiles, compute_tc_corrections, _mo_on_grid, ground_state
from distilled_u import make_uprime


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--atom", required=True)
    p.add_argument("--unit", default="Bohr")
    p.add_argument("--basis", default="cc-pvdz")
    p.add_argument("--ncas", type=int, required=True)
    p.add_argument("--nelecas", type=int, nargs=2, required=True,
                   metavar=("NA", "NB"))
    p.add_argument("--u-b", type=float, default=0.3)
    p.add_argument("--distilled-u", default=None,
                   help="npz of NN-distilled u coeffs (c_opp,c_same); cusp "
                        "clamped to Kato. Overrides the analytic --u-b cusp.")
    p.add_argument("--nn-energy", type=float, default=None,
                   help="if set (with --distilled-u): root-find the tail-damp "
                        "lambda so TC-CASCI = E_NN (refinement #1).")
    p.add_argument("--lam", type=float, default=None,
                   help="fixed tail-damp lambda (transferability check); "
                        "overrides the root-find.")
    p.add_argument("--grid-level", type=int, default=1)
    args = p.parse_args()

    na_act, nb_act = args.nelecas
    mol = gto.M(atom=args.atom, unit=args.unit, basis=args.basis, verbose=0)
    mf = scf.RHF(mol).run()
    C = np.asarray(mf.mo_coeff)
    n_elec = mol.nelectron
    n_core = (n_elec - na_act - nb_act) // 2
    n_sub = n_core + args.ncas
    e_nuc = mol.energy_nuc()
    print(f"system: {mol.nao} AOs, {n_elec}e-, CAS({args.ncas},"
          f"({na_act},{nb_act})), n_core={n_core}, subspace={n_sub} orb")

    # Work in the core+active MO subspace.
    Csub = C[:, :n_sub]
    h1 = Csub.T @ mf.get_hcore() @ Csub
    g = np.transpose(ao2mo.restore(1, ao2mo.kernel(mol, Csub), n_sub),
                     (0, 2, 1, 3))
    dets = enumerate_cas_dets(n_core, args.ncas, na_act, nb_act)
    print(f"  |CAS dets| = {len(dets)}")

    # [G1] bare CASCI vs PySCF.
    e_bare, _ = ground_state(build_fci_matrix(h1, g, n_sub, dets), e_nuc)
    e_pyscf = mcscf.CASCI(mf, args.ncas, (na_act, nb_act)).kernel()[0]
    print(f"[G1] bare CASCI {e_bare:.8f} vs PySCF {e_pyscf:.8f}  "
          f"diff {abs(e_bare-e_pyscf)*1e6:.2f} uE_h  "
          f"{'PASS' if abs(e_bare-e_pyscf)<1e-6 else 'FAIL'}")

    grid = _mo_on_grid(mol, Csub, args.grid_level)

    # [G2] u=0 -> bare.
    dV0 = compute_tc_corrections(mol, Csub, u_profiles("cusp", 0.0, args.u_b),
                                 args.grid_level, grid=grid)
    e0, _ = ground_state(build_fci_matrix(h1, g + dV0, n_sub, dets,
                                          g_same=g + dV0), e_nuc)
    print(f"[G2] u=0 TC-CASCI {e0:.8f} vs bare {e_bare:.8f}  "
          f"{'PASS' if abs(e0-e_bare)<1e-6 else 'FAIL'}")

    # 2-body TC-CASCI (spin-resolved). Distilled-u (cusp-clamped) or analytic.
    def tc_energy(lam=1.0):
        if args.distilled_u:
            du = np.load(args.distilled_u)
            uo = make_uprime(du["c_opp"], 0.5, lam)
            us = make_uprime(du["c_same"], 0.25, lam)
        else:
            uo = u_profiles("cusp", 0.5, args.u_b)
            us = u_profiles("cusp", 0.25, args.u_b)
        dVo = compute_tc_corrections(mol, Csub, uo, args.grid_level, grid=grid)
        dVs = compute_tc_corrections(mol, Csub, us, args.grid_level, grid=grid)
        return ground_state(build_fci_matrix(h1, g + dVo, n_sub, dets,
                                             g_same=g + dVs), e_nuc)

    if args.lam is not None and args.distilled_u:
        # Transferability check: fixed lambda from another CAS.
        lam = args.lam
        e_tc, imag = tc_energy(lam)
        tag = f"NN-distilled u, FIXED lambda={lam:.4f} (transfer)"
    elif args.nn_energy is not None and args.distilled_u:
        # Refinement #1: damp the distilled tail to match E_NN.
        print(f"\n[#1] root-finding tail-damp lambda so TC-CASCI = E_NN "
              f"= {args.nn_energy:.6f}")
        lo, hi = 0.0, 1.0
        e_lo, _ = tc_energy(lo); e_hi, _ = tc_energy(hi)
        print(f"    lam=0 (bare cusp) -> {e_lo:.6f};  lam=1 (raw distilled) "
              f"-> {e_hi:.6f}")
        lam = None
        if not (min(e_lo, e_hi) <= args.nn_energy <= max(e_lo, e_hi)):
            print("    WARNING: E_NN not bracketed by lam in [0,1]")
        for _ in range(30):
            mid = 0.5 * (lo + hi)
            em, _ = tc_energy(mid)
            if abs(em - args.nn_energy) < 2e-5:
                lam = mid; break
            # energy decreases as lam grows (more tail -> more correlation)
            if em > args.nn_energy:
                lo = mid
            else:
                hi = mid
            lam = mid
        e_tc, imag = tc_energy(lam)
        tag = (f"NN-distilled u, tail-damped lambda*={lam:.4f} "
               f"(E_NN-calibrated)")
    else:
        lam = 1.0
        e_tc, imag = tc_energy(lam)
        tag = (f"NN-distilled u (raw tail)" if args.distilled_u
               else f"analytic cusp b={args.u_b}")
    print(f"[G3] TC-CASCI imag {imag:.1e}  {'PASS' if imag<1e-6 else 'WARN'}")

    print(f"\n=== TC-CASCI ({tag}, L{args.grid_level}) ===")
    print(f"  E_HF              = {mf.e_tot:.6f}")
    print(f"  bare CASCI        = {e_bare:.6f}")
    print(f"  2-body TC-CASCI   = {e_tc:.6f}   "
          f"({(e_tc-e_bare)*1e3:+.3f} mE_h vs bare CASCI)")
    if args.nn_energy is not None:
        print(f"  E_NN (target)     = {args.nn_energy:.6f}   "
              f"(match {(e_tc-args.nn_energy)*1e6:+.1f} uE_h)")


if __name__ == "__main__":
    main()
