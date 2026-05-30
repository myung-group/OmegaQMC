"""NEVPT2 from a CS-recovered CI vector — head-to-head against CASSCF baseline.

Given a trained NN-VMC + walker bank, recovers ``c_hat`` in the natural-
orbital basis, builds a CASCI(ncas, nelecas) reference whose ``ci`` is
the active-space projection of ``c_hat``, runs NEVPT2 on it, and
compares against a fully-converged CASSCF+NEVPT2 calculation at
matched ``(ncas, nelecas)``. Output is a publication-ready table for
the paper's §VI.A demonstration.

Usage:
    python examples/run_cs_nevpt2.py --cell-dir cs_pilot_results/beh2_... \
        --molecule beh2 --R 1.33 --basis cc-pvdz --unit Angstrom \
        --ansatz OmegaQMC/psi/nn/conf/ferminet_jastrow.yaml \
        --n-alpha 3 --n-beta 3 --ncas 2 --nelecas 2,2

If ``--ncas`` is omitted, the active space is auto-selected from
NO occupations.
"""

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")

import numpy as np
import jax

from OmegaQMC.utils import Mole_custom
from OmegaQMC.cs.reference import compute_fci_reference, compute_casci_reference
from OmegaQMC.cs.walkers import load_walker_bank
from OmegaQMC.cs.estimators import (
    evaluate_orbitals_on_walkers, f_I_matrix,
    normalize_and_align, normalize_and_align_bias_corrected,
    lasso_recover_auto,
)
from OmegaQMC.cs.scaling import precompute_means
from OmegaQMC.cs.mrpt import compare_nevpt2
from OmegaQMC.psi.nn.adapter import make_nn_log_psi
from OmegaQMC.vmc_nn import get_vmc_nn_func

sys.path.insert(0, str(Path(__file__).parent))
from run_cs_h4_scaling import evaluate_signed_psi  # noqa: E402
from run_cs_properties import build_mol  # noqa: E402


def parse_nelecas(s):
    if s is None:
        return None
    a, b = s.split(",")
    return (int(a), int(b))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cell-dir", required=True)
    p.add_argument("--molecule", required=True,
                   choices=["h2", "h4", "lih", "beh2", "h2o", "n2", "c2",
                            "h6", "h8"])
    p.add_argument("--geometry-tag", default="")
    p.add_argument("--R", type=float, default=1.0)
    p.add_argument("--basis", default="cc-pvdz")
    p.add_argument("--unit", default="Angstrom", choices=["Bohr", "Angstrom"])
    p.add_argument("--ansatz", required=True)
    p.add_argument("--n-alpha", type=int, required=True)
    p.add_argument("--n-beta", type=int, required=True)
    p.add_argument("--ref-ncas", type=int, default=None,
                   help="If set, use CASCI(ref_ncas, ref_nelecas) as the "
                        "FCI reference instead of full FCI. Required when "
                        "full FCI is intractable (e.g. C2/cc-pVDZ).")
    p.add_argument("--ref-nelecas", type=parse_nelecas, default=None,
                   help="electron counts for the CASCI reference, "
                        "e.g. '4,4'.")
    p.add_argument("--ref-ncore", type=int, default=None,
                   help="frozen-core count for the CASCI reference; "
                        "auto-derived if omitted.")
    p.add_argument("--ncas", type=int, default=None,
                   help="if omitted, auto-select from NO occupations")
    p.add_argument("--nelecas", type=parse_nelecas, default=None,
                   help="format: 'n_a,n_b' (default: inferred)")
    p.add_argument("--occ-threshold", type=float, default=0.1)
    p.add_argument("--max-ncas", type=int, default=None)
    p.add_argument("--psi-batch", type=int, default=2048)
    p.add_argument("--det-chunk", type=int, default=200)
    p.add_argument("--candidate-tol", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--lam-mult", type=float, default=0.5,
                   help="Lasso threshold multiplier: lam = lam_mult * "
                        "sigma_median * sqrt(2 log n_det). 0.0 disables "
                        "thresholding (raw sample-mean recovery). "
                        "Default 0.5 (empirically optimal for MR systems).")
    p.add_argument("--out-json", default=None)
    args = p.parse_args()

    cell_dir = Path(args.cell_dir)
    prefix = cell_dir.name
    walkers_path = cell_dir / f"{prefix}_walkers.h5"
    checkpoint = cell_dir / f"{prefix}.chk.h5"
    if not walkers_path.exists() or not checkpoint.exists():
        sys.exit(f"missing files in {cell_dir}")

    print(f"=== NEVPT2 from CS — {prefix} ===")
    mol = build_mol(args.molecule, args.R, args.basis, args.unit,
                    args.geometry_tag)
    if args.ref_ncas is not None:
        if args.ref_nelecas is None:
            sys.exit("--ref-ncas requires --ref-nelecas")
        ncore = args.ref_ncore
        if ncore is None:
            ncore = (args.n_alpha + args.n_beta
                     - sum(args.ref_nelecas)) // 2
        print(f"  using CASCI({args.ref_ncas},{tuple(args.ref_nelecas)}) "
              f"reference; ncore={ncore}")
        fci_ref = compute_casci_reference(
            mol, ncas=args.ref_ncas,
            nelecas=tuple(args.ref_nelecas),
            ncore=ncore,
            candidate_tol=args.candidate_tol,
        )
    else:
        fci_ref = compute_fci_reference(
            mol, n_alpha=args.n_alpha, n_beta=args.n_beta,
            candidate_tol=args.candidate_tol,
        )
    print(f"  mol: {mol.nao} AOs, {mol.nelec} electrons")
    print(f"  E_HF  = {fci_ref['E_HF']: .6f} Ha")
    print(f"  E_FCI = {fci_ref['E_FCI']: .6f} Ha")

    walkers, _, _ = load_walker_bank(str(walkers_path))
    init_key = jax.random.split(jax.random.key(args.seed), 4)[1]
    driver = get_vmc_nn_func(mol, args.ansatz, init_key, prefix=str(cell_dir / prefix))
    driver.load_checkpoint(str(checkpoint))
    log_psi, _, _, _ = make_nn_log_psi(args.ansatz, mol, init_key)
    psi_vals = evaluate_signed_psi(walkers, np.asarray(mol.atom_coords()),
                                    driver.params, log_psi,
                                    batch_size=args.psi_batch)
    K_s_max = walkers.shape[0]
    means, m2s = precompute_means(
        mol, fci_ref, walkers, psi_vals,
        K_s_sweep=[K_s_max], n_seeds=1,
        det_chunk_size=args.det_chunk,
        walker_convention="interleaved",
        return_second_moments=True,
    )
    c_raw = means[(K_s_max, 0)]
    m2 = m2s[(K_s_max, 0)]
    c_true = np.array([fci_ref["ci_dict"][k] for k in fci_ref["candidate_set"]])
    n_det = len(c_true)
    ref_sign = float(np.sign(c_true[0]))
    if args.lam_mult > 0.0:
        c_hat, rec_info = lasso_recover_auto(
            c_raw, m2, K_s_max, n_det, ref_sign,
            lam_mult=args.lam_mult,
        )
        print(f"  K_s = {K_s_max},  n_det = {n_det}")
        print(f"  proj_mass: naive={rec_info['proj_mass_naive']:.3e}, "
              f"bias-corrected={rec_info['proj_mass_bias_corrected']:.3e}")
        print(f"  Lasso: sigma_median={rec_info['sigma_median']:.3e}, "
              f"lam_universal={rec_info['lam_universal']:.3e}, "
              f"lam_used={rec_info['lam_used']:.3e} "
              f"(mult={rec_info['lam_mult']:.2f})")
        print(f"  recovered support = {rec_info['support']}/{n_det}, "
              f"max|c_hat|={rec_info['max_abs']:.4f}")
        # c_true diagnostics for comparison
        c_true_norm = c_true / np.linalg.norm(c_true)
        err_l2 = float(np.linalg.norm(c_hat - c_true_norm))
        print(f"  c_true: max|c|={np.max(np.abs(c_true_norm)):.4f}, "
              f"support(>1e-4)={int(np.sum(np.abs(c_true_norm) > 1e-4))}")
        print(f"  recovery error ||c_hat - c_true||_2 = {err_l2:.4f}")
    else:
        c_hat, proj_mass = normalize_and_align(c_raw, ref_sign)
        rec_info = dict(proj_mass_naive=float(proj_mass),
                        proj_mass_bias_corrected=float("nan"),
                        sigma_median=float("nan"),
                        lam_universal=0.0, lam_used=0.0, lam_mult=0.0,
                        support=int(np.sum(np.abs(c_hat) > 1e-12)),
                        max_abs=float(np.max(np.abs(c_hat))),
                        renorm_after_threshold=1.0)
        c_true_norm = c_true / np.linalg.norm(c_true)
        err_l2 = float(np.linalg.norm(c_hat - c_true_norm))
        print(f"  K_s = {K_s_max} (raw sample-mean, no Lasso)")
        print(f"  proj_mass = {proj_mass:.3e}, max|c_hat| = {rec_info['max_abs']:.4f}")
        print(f"  recovery error ||c_hat - c_true||_2 = {err_l2:.4f}")
    proj_mass = rec_info["proj_mass_naive"]

    print(f"\n[NEVPT2 comparison]")
    cmp = compare_nevpt2(
        mol, c_hat, fci_ref,
        ncas=args.ncas, nelecas=args.nelecas,
        occ_threshold=args.occ_threshold, max_ncas=args.max_ncas,
    )
    ch, cs = cmp["chat"], cmp["casscf"]
    print(f"\n  active space: CAS({ch['ncas']}, {ch['nelecas']})  "
          f"core orbitals = {ch['ncore']}")
    print(f"  determinants of c_hat kept in active space: {ch['n_det_kept_in_active']}")

    print(f"\n  {'quantity':<25} {'chat reference':>16} {'CASSCF reference':>18} {'diff':>14}")
    print("-" * 75)
    print(f"  {'E (reference)':<25} {ch['e_casci']:>16.6f} {cs['e_casscf']:>18.6f}"
          f" {cmp['delta_reference']:>+14.6f}")
    print(f"  {'E_NEVPT2 (correction)':<25} {ch['e_pt2']:>16.6f} {cs['e_pt2']:>18.6f}"
          f" {cmp['delta_pt2']:>+14.6f}")
    print(f"  {'E_total':<25} {ch['e_total']:>16.6f} {cs['e_total']:>18.6f}"
          f" {cmp['delta_total']:>+14.6f}")
    print(f"\n  E_FCI(full basis) = {fci_ref['E_FCI']:.6f} Ha")
    print(f"  Gap (chat NEVPT2  vs FCI) = {(ch['e_total'] - fci_ref['E_FCI'])*1000:+.3f} mE_h")
    print(f"  Gap (CASSCF NEVPT2 vs FCI) = {(cs['e_total'] - fci_ref['E_FCI'])*1000:+.3f} mE_h")

    if args.out_json:
        out = dict(
            prefix=prefix, molecule=args.molecule,
            geometry_tag=args.geometry_tag, R=args.R, basis=args.basis,
            K_s=int(K_s_max), proj_mass=float(proj_mass),
            recovery=dict(
                lam_mult=float(rec_info["lam_mult"]),
                lam_universal=float(rec_info["lam_universal"]),
                lam_used=float(rec_info["lam_used"]),
                sigma_median=float(rec_info["sigma_median"]),
                support=int(rec_info["support"]),
                n_det=int(n_det),
                max_abs_chat=float(rec_info["max_abs"]),
                max_abs_ctrue=float(np.max(np.abs(c_true_norm))),
                err_l2=float(err_l2),
                proj_mass_naive=float(rec_info["proj_mass_naive"]),
                proj_mass_bias_corrected=float(rec_info["proj_mass_bias_corrected"]),
            ),
            E_HF=float(fci_ref["E_HF"]),
            E_FCI=float(fci_ref["E_FCI"]),
            ncas=int(ch["ncas"]), nelecas=list(ch["nelecas"]),
            ncore=int(ch["ncore"]),
            n_det_kept_in_active=int(ch["n_det_kept_in_active"]),
            chat={k: float(v) if isinstance(v, (int, float, np.floating))
                  else v
                  for k, v in ch.items()
                  if k not in ("nelecas",)},
            casscf={k: float(v) if isinstance(v, (int, float, np.floating))
                    else v
                    for k, v in cs.items()
                    if k not in ("nelecas",)},
            delta_total=float(cmp["delta_total"]),
            delta_pt2=float(cmp["delta_pt2"]),
            delta_reference=float(cmp["delta_reference"]),
        )
        with open(args.out_json, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nJSON -> {args.out_json}")


if __name__ == "__main__":
    main()
