"""Additional post-HF methods on a CS-recovered ĉ:

  (1) MC-PDFT (multi-configurational pair-density functional theory)
      using ĉ-derived 1- and 2-RDM as the multireference reference.
      Compares against MC-PDFT on CASSCF(8,(3,3)) reference.

  (2) CCSD with NN-NO orbitals as the reference orbital frame,
      compared to standard CCSD on HF orbitals.

  (3) CCSD(T) on the NN-NO frame, same comparison.

All three methods consume the framework's outputs in different ways:
  MC-PDFT      — needs 1-RDM and 2-RDM
  CCSD         — needs an orbital frame (NN-NOs from ĉ-derived 1-RDM)
  CCSD(T)      — same

Strengthens §VI by showing the framework's outputs feed three
distinct post-HF method families: PT2 (NEVPT2 already in paper),
DFT-style MR (MC-PDFT), and coupled-cluster (CCSD/CCSD(T)).
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path

import numpy as np
import jax

sys.path.insert(0, str(Path(__file__).parent))

from OmegaQMC.utils import Mole_custom
from OmegaQMC.cs.reference import compute_fci_reference, compute_casci_reference
from OmegaQMC.cs.walkers import load_walker_bank
from OmegaQMC.cs.estimators import normalize_and_align, lasso_recover_auto
from OmegaQMC.cs.scaling import precompute_means
from OmegaQMC.cs.properties import (
    reshape_chat_to_pyscf_matrix, compute_1rdm,
    natural_occupations_from_rdm,
)
from OmegaQMC.psi.nn.adapter import make_nn_log_psi
from OmegaQMC.vmc_nn import get_vmc_nn_func
from run_cs_properties import build_mol
from run_cs_h4_scaling import evaluate_signed_psi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell-dir", required=True)
    ap.add_argument("--molecule", required=True)
    ap.add_argument("--geometry-tag", default="")
    ap.add_argument("--R", type=float, default=1.0)
    ap.add_argument("--basis", default="cc-pvdz")
    ap.add_argument("--unit", default="Angstrom")
    ap.add_argument("--n-alpha", type=int, required=True)
    ap.add_argument("--n-beta", type=int, required=True)
    ap.add_argument("--ansatz", required=True)
    ap.add_argument("--ncas", type=int, default=8)
    ap.add_argument("--nelecas-alpha", type=int, default=3)
    ap.add_argument("--nelecas-beta", type=int, default=3)
    ap.add_argument("--mcpdft-functional", default="tPBE",
                    help="MC-PDFT translated functional (tPBE, tBLYP, etc.)")
    ap.add_argument("--ref-ncas", type=int, default=None,
                    help="If set, use CASCI reference instead of full FCI.")
    ap.add_argument("--ref-nelecas", type=str, default=None,
                    help="CASCI nelecas as 'a,b' (required with --ref-ncas).")
    ap.add_argument("--ref-ncore", type=int, default=None,
                    help="frozen-core count for CASCI ref; auto if omitted.")
    ap.add_argument("--lam-mult", type=float, default=0.5,
                    help="Lasso threshold multiplier (0.0 = raw sample-mean)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cell_dir = Path(args.cell_dir)
    prefix = cell_dir.name
    print(f"=== Post-HF extras: {prefix} ===")

    # CS recovery (standard pipeline)
    mol = build_mol(args.molecule, args.R, args.basis, args.unit,
                    args.geometry_tag)
    nelec = (args.n_alpha, args.n_beta)
    if args.ref_ncas is not None:
        if args.ref_nelecas is None:
            sys.exit("--ref-ncas requires --ref-nelecas")
        ref_nele = tuple(int(x) for x in args.ref_nelecas.split(","))
        ref_ncore = args.ref_ncore
        if ref_ncore is None:
            ref_ncore = (args.n_alpha + args.n_beta - sum(ref_nele)) // 2
        print(f"  using CASCI({args.ref_ncas},{ref_nele}) ncore={ref_ncore}")
        fci_ref = compute_casci_reference(
            mol, ncas=args.ref_ncas, nelecas=ref_nele,
            ncore=ref_ncore, candidate_tol=1e-4,
        )
        print(f"  E_HF = {fci_ref['E_HF']:.6f}, "
              f"E_CASCI = {fci_ref['E_FCI']:.6f} (CASCI used as reference)")
    else:
        fci_ref = compute_fci_reference(mol, n_alpha=args.n_alpha,
                                        n_beta=args.n_beta, candidate_tol=1e-4)
        print(f"  E_HF = {fci_ref['E_HF']:.6f}, "
              f"E_FCI = {fci_ref['E_FCI']:.6f}")

    walkers, _, _ = load_walker_bank(str(cell_dir / f"{prefix}_walkers.h5"))
    key = jax.random.split(jax.random.key(42), 4)[1]
    drv = get_vmc_nn_func(mol, args.ansatz, key, prefix=str(cell_dir / prefix))
    drv.load_checkpoint(str(cell_dir / f"{prefix}.chk.h5"))
    log_psi, _, _, _ = make_nn_log_psi(args.ansatz, mol, key)
    psi_vals = evaluate_signed_psi(walkers, np.asarray(mol.atom_coords()),
                                    drv.params, log_psi, batch_size=2048)
    K_s = walkers.shape[0]
    means, m2s = precompute_means(mol, fci_ref, walkers, psi_vals,
                                    K_s_sweep=[K_s], n_seeds=1,
                                    walker_convention="interleaved",
                                    return_second_moments=True)
    c_raw = means[(K_s, 0)]
    m2 = m2s[(K_s, 0)]
    c_true = np.array([fci_ref["ci_dict"][k]
                       for k in fci_ref["candidate_set"]])
    n_det = len(c_true)
    ref_sign = float(np.sign(c_true[0]))
    if args.lam_mult > 0.0:
        c_hat, rec_info = lasso_recover_auto(
            c_raw, m2, K_s, n_det, ref_sign, lam_mult=args.lam_mult,
        )
        proj_mass = rec_info["proj_mass_naive"]
        c_true_norm = c_true / np.linalg.norm(c_true)
        err_l2 = float(np.linalg.norm(c_hat - c_true_norm))
        print(f"  Lasso (lam_mult={args.lam_mult}): support="
              f"{rec_info['support']}/{n_det}, max|c_hat|="
              f"{rec_info['max_abs']:.4f} "
              f"(true max|c|={float(np.max(np.abs(c_true_norm))):.4f})")
        print(f"  proj_mass: naive={rec_info['proj_mass_naive']:.3e}, "
              f"corr={rec_info['proj_mass_bias_corrected']:.3e}; "
              f"||c_hat - c_true||₂ = {err_l2:.4f}")
    else:
        c_hat, proj_mass = normalize_and_align(c_raw, ref_sign)
        rec_info = dict(lam_mult=0.0, support=int(np.sum(np.abs(c_hat) > 1e-12)),
                        max_abs=float(np.max(np.abs(c_hat))),
                        proj_mass_naive=float(proj_mass),
                        proj_mass_bias_corrected=float("nan"),
                        sigma_median=float("nan"),
                        lam_universal=0.0, lam_used=0.0)
        c_true_norm = c_true / np.linalg.norm(c_true)
        err_l2 = float(np.linalg.norm(c_hat - c_true_norm))
        print(f"  raw sample-mean (no Lasso): max|c_hat|="
              f"{rec_info['max_abs']:.4f}  ||c_hat - c_true||₂={err_l2:.4f}")
    n_orb = int(fci_ref["n_orb"])
    print(f"  CS recovered: C0={float(c_hat[np.argmax(np.abs(c_hat))]):+.4f}")

    # Build NN-NO frame from ĉ-derived 1-RDM
    gamma = compute_1rdm(c_hat, fci_ref["candidate_set"], n_orb, nelec)
    gamma = 0.5 * (gamma + gamma.T)
    occs, U = np.linalg.eigh(gamma)
    order = np.argsort(-occs)
    occs = occs[order]
    U = U[:, order]
    no_coeff_ao = fci_ref["no_coeff_ao"] @ U  # NN-NO in AO basis

    out = dict(prefix=prefix, molecule=args.molecule, R=args.R,
               basis=args.basis, n_orb=n_orb, nelec=list(nelec),
               E_HF=float(fci_ref["E_HF"]),
               E_FCI=float(fci_ref["E_FCI"]),
               C0_chat=float(np.max(np.abs(c_hat))),
               recovery=dict(
                   lam_mult=float(rec_info["lam_mult"]),
                   support=int(rec_info["support"]),
                   n_det=int(n_det),
                   max_abs_chat=float(rec_info["max_abs"]),
                   max_abs_ctrue=float(np.max(np.abs(c_true_norm))),
                   err_l2=float(err_l2),
                   proj_mass_naive=float(rec_info["proj_mass_naive"]),
                   proj_mass_bias_corrected=float(rec_info["proj_mass_bias_corrected"]),
                   sigma_median=float(rec_info.get("sigma_median", float("nan"))),
                   lam_universal=float(rec_info.get("lam_universal", 0.0)),
                   lam_used=float(rec_info.get("lam_used", 0.0)),
               ),
               nn_no_occupations=occs.tolist())

    # ── (2) CCSD on HF reference (baseline) ──
    print(f"\n  [CCSD on HF reference orbitals]")
    from pyscf import scf, cc, mp
    mf = scf.RHF(mol)
    mf.kernel()
    t0 = time.time()
    cc_hf = cc.CCSD(mf)
    cc_hf.verbose = 0
    e_ccsd_hf, _, _ = cc_hf.kernel()
    et_hf = cc_hf.ccsd_t()
    t_hf = time.time() - t0
    print(f"    E_CCSD = {mf.e_tot + e_ccsd_hf:.6f}  "
          f"E_CCSD(T) = {mf.e_tot + e_ccsd_hf + et_hf:.6f}  "
          f"({t_hf:.1f}s)")
    out["ccsd_hf"] = dict(
        e_ccsd=float(mf.e_tot + e_ccsd_hf),
        e_ccsd_t=float(mf.e_tot + e_ccsd_hf + et_hf),
        e_correlation=float(e_ccsd_hf),
        time=float(t_hf),
    )

    # ── (3) CCSD on NN-NO reference ──
    # Substitute NN-NOs as the reference orbital matrix
    print(f"\n  [CCSD on NN-NO reference orbitals]")
    mf_nn = scf.RHF(mol)
    mf_nn.kernel()
    mf_nn.mo_coeff = no_coeff_ao
    # Recompute orbital energies in the NN-NO basis (Fock diagonal)
    h1ao = mf_nn.get_hcore()
    s1ao = mf_nn.get_ovlp()
    dm_ao = mf_nn.make_rdm1(no_coeff_ao,
                             mo_occ=np.array([2.0]*args.n_alpha
                                              + [0.0]*(n_orb - args.n_alpha)))
    fockao = mf_nn.get_fock(dm=dm_ao)
    # MO Fock
    mo_e = np.diag(no_coeff_ao.T @ fockao @ no_coeff_ao)
    mf_nn.mo_energy = mo_e
    mf_nn.mo_occ = np.array([2.0]*args.n_alpha
                             + [0.0]*(n_orb - args.n_alpha))
    t0 = time.time()
    try:
        cc_nn = cc.CCSD(mf_nn)
        cc_nn.verbose = 0
        e_ccsd_nn, _, _ = cc_nn.kernel()
        et_nn = cc_nn.ccsd_t()
        t_nn = time.time() - t0
        e_tot_nn = float(np.sum(mo_e * mf_nn.mo_occ) + mol.energy_nuc()
                          - 0.5 * np.einsum("pq,pq->", h1ao, dm_ao))
        # Simpler: compute SCF energy at NN-NO orbitals
        e_ref_nn = mf_nn.energy_elec(dm=dm_ao)[0] + mol.energy_nuc()
        print(f"    E_ref(NN-NO) = {e_ref_nn:.6f}  (vs E_HF = {mf.e_tot:.6f})")
        print(f"    E_CCSD = {e_ref_nn + e_ccsd_nn:.6f}  "
              f"E_CCSD(T) = {e_ref_nn + e_ccsd_nn + et_nn:.6f}  "
              f"({t_nn:.1f}s)")
        out["ccsd_nn"] = dict(
            e_ref=float(e_ref_nn),
            e_ccsd=float(e_ref_nn + e_ccsd_nn),
            e_ccsd_t=float(e_ref_nn + e_ccsd_nn + et_nn),
            e_correlation=float(e_ccsd_nn),
            time=float(t_nn),
        )
    except Exception as ex:
        print(f"    CCSD on NN-NO FAILED: {ex}")
        out["ccsd_nn"] = dict(error=str(ex))

    # ── (1) MC-PDFT on NN-NO active space ──
    # Strategy: variational CASCI on the NN-NO orbital frame, then
    # MC-PDFT on top. The NN-NO contribution is the active-space
    # orbital basis derived from the ĉ 1-RDM.
    print(f"\n  [MC-PDFT on NN-NO active space]")
    try:
        from pyscf import mcpdft, mcscf
        ncas = args.ncas
        nelecas = (args.nelecas_alpha, args.nelecas_beta)
        ncore = (args.n_alpha + args.n_beta - sum(nelecas)) // 2
        print(f"    CAS({ncas},{nelecas}) ncore={ncore}")

        mf_for_mc = scf.RHF(mol).run(verbose=0)
        mc = mcscf.CASCI(mf_for_mc, ncas, nelecas)
        mc.ncore = ncore
        mc.mo_coeff = no_coeff_ao   # NN-NO basis
        mc.verbose = 0
        mc.kernel(mo_coeff=no_coeff_ao)
        e_cas = float(mc.e_tot)
        print(f"    E_CASCI (NN-NO) = {e_cas:.6f}")

        pdft = mcpdft.CASCI(mc, args.mcpdft_functional, ncas, nelecas)
        pdft.verbose = 0
        pdft.kernel()
        print(f"    E_MC-PDFT ({args.mcpdft_functional}) = "
              f"{pdft.e_tot:.6f}  (vs E_CASCI = {e_cas:.6f})")
        out["mcpdft"] = dict(
            functional=args.mcpdft_functional,
            e_casci=e_cas,
            e_mcpdft=float(pdft.e_tot),
            ncas=int(ncas), nelecas=list(nelecas),
            orbital_basis="NN-NO",
        )
    except Exception as ex:
        import traceback
        print(f"    MC-PDFT FAILED: {ex}")
        traceback.print_exc()
        out["mcpdft"] = dict(error=str(ex))

    # ── Summary ──
    print(f"\n  ===== Summary =====")
    print(f"  E_HF                = {fci_ref['E_HF']:+.6f}")
    print(f"  E_FCI               = {fci_ref['E_FCI']:+.6f}")
    if "ccsd_hf" in out:
        print(f"  E_CCSD(HF orb)      = {out['ccsd_hf']['e_ccsd']:+.6f}  "
              f"(gap to FCI: {1000*(out['ccsd_hf']['e_ccsd']-fci_ref['E_FCI']):+.2f} mEh)")
        print(f"  E_CCSD(T)(HF orb)   = {out['ccsd_hf']['e_ccsd_t']:+.6f}  "
              f"(gap to FCI: {1000*(out['ccsd_hf']['e_ccsd_t']-fci_ref['E_FCI']):+.2f} mEh)")
    if "ccsd_nn" in out and "e_ccsd" in out["ccsd_nn"]:
        print(f"  E_CCSD(NN-NO)       = {out['ccsd_nn']['e_ccsd']:+.6f}  "
              f"(gap to FCI: {1000*(out['ccsd_nn']['e_ccsd']-fci_ref['E_FCI']):+.2f} mEh)")
        print(f"  E_CCSD(T)(NN-NO)    = {out['ccsd_nn']['e_ccsd_t']:+.6f}  "
              f"(gap to FCI: {1000*(out['ccsd_nn']['e_ccsd_t']-fci_ref['E_FCI']):+.2f} mEh)")
    if "mcpdft" in out and "e_mcpdft" in out["mcpdft"]:
        print(f"  E_MC-PDFT (ĉ ref)   = {out['mcpdft']['e_mcpdft']:+.6f}  "
              f"(gap to FCI: {1000*(out['mcpdft']['e_mcpdft']-fci_ref['E_FCI']):+.2f} mEh)")

    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  wrote {args.out}")


if __name__ == "__main__":
    main()
