"""
Gate-0 on converged paper banks: TC two-sided decode reusing the exact
decoder-sweep setup (build_mol + evaluate_signed_psi) so geometry, basis,
ansatz, and checkpoint loading match training bit-for-bit.

For a chosen system it loads the converged FermiNet+Jastrow bank+checkpoint,
reconstructs signed Psi, retrofit-extracts J(R) (jastrow-off sibling),
runs the tau = 0 / +1 / -1 decode, and prints the Gate-0 report:
leakage M_spurious(0) -> M_spurious(+1) and the biorthogonal overlap.

    python examples/run_cs_tc_gate0.py --system beh2
    python examples/run_cs_tc_gate0.py --system c2
"""

import argparse
import os
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import jax

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # sibling imports

from run_cs_properties import build_mol
from run_cs_h4_scaling import evaluate_signed_psi

from OmegaQMC.cs.reference import compute_fci_reference
from OmegaQMC.cs.walkers import load_walker_bank
from OmegaQMC.cs.estimators import evaluate_orbitals_on_walkers
from OmegaQMC.cs.transcorrelated import decode_two_sided, print_gate0_report
from OmegaQMC.cs.jastrow_extract import (
    nn_jastrow_on_walkers,
    kato_cusp_jastrow,
)
from OmegaQMC.psi.nn.adapter import make_nn_log_psi
from OmegaQMC.vmc_nn import get_vmc_nn_func

ANSATZ = "OmegaQMC/psi/nn/conf/ferminet_jastrow.yaml"

SYSTEMS = {
    "h2": dict(geometry="h2", R=1.4, basis="cc-pvdz", unit="Angstrom",
               tag="", n_alpha=1, n_beta=1,
               cell_dir="cs_h2_tight/h2_ferminet_jastrow_R1p400_cc-pvdz"),
    "h4_linear": dict(geometry="h4", R=1.0, basis="cc-pvdz", unit="Angstrom",
                      tag="linear", n_alpha=2, n_beta=2,
                      cell_dir="cs_h4_results/"
                               "h4_linear_ferminet_jastrow_R1p000_cc-pvdz"),
    "h6": dict(geometry="h6", R=1.0, basis="cc-pvdz", unit="Angstrom",
               tag="", n_alpha=3, n_beta=3,
               cell_dir="cs_h6_converged/h6_ferminet_jastrow_R1p000_cc-pvdz"),
    "beh2": dict(geometry="beh2", R=1.33, basis="cc-pvdz", unit="Angstrom",
                 tag="", n_alpha=3, n_beta=3,
                 cell_dir="cs_beh2_tight/beh2_ferminet_jastrow_R1p330_cc-pvdz"),
    "c2": dict(geometry="c2", R=1.243, basis="cc-pvdz", unit="Angstrom",
               tag="", n_alpha=6, n_beta=6,
               cell_dir="cs_c2_converged/c2_ferminet_jastrow_R1p243_cc-pvdz"),
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--system", required=True, choices=list(SYSTEMS))
    p.add_argument("--j-source", default="nn", choices=["nn", "cusp"],
                   help="TC factor: 'nn' = full learned Jastrow (Strategy A); "
                        "'cusp' = analytic Kato cusp-only factor (Strategy B)")
    p.add_argument("--cusp-b", type=float, default=1.0,
                   help="Pade damping b for the cusp-only factor")
    p.add_argument("--ansatz", default=ANSATZ)
    p.add_argument("--candidate-tol", type=float, default=1e-4)
    p.add_argument("--full-pool", action="store_true",
                   help="decode over the FULL determinant enumeration so the "
                        "high-virtual leakage determinants are counted (the "
                        "correct leakage metric; default uses the filtered "
                        "FCI candidate set)")
    p.add_argument("--leak-thresh", type=float, default=1e-3)
    p.add_argument("--psi-batch", type=int, default=2048)
    p.add_argument("--top-k", type=int, default=10)
    args = p.parse_args()

    s = SYSTEMS[args.system]
    na, nb = s["n_alpha"], s["n_beta"]
    print(f"== TC Gate-0 :: {args.system} :: {jax.default_backend()} "
          f"{jax.devices()} ==")

    mol = build_mol(s["geometry"], s["R"], s["basis"], s["unit"], s["tag"])
    n_orb = int(mol.nao)
    print(f"[1] mol: {n_orb} AOs ({na},{nb}) e-, basis={s['basis']}")
    fci = compute_fci_reference(mol, n_alpha=na, n_beta=nb,
                                candidate_tol=args.candidate_tol)
    if args.full_pool:
        occ_a = list(combinations(range(n_orb), na))
        occ_b = list(combinations(range(n_orb), nb))
        pool = [(a, b) for a in occ_a for b in occ_b]
        ptag = "FULL enumeration"
    else:
        pool = fci["candidate_set"]
        ptag = f"filtered (tol={args.candidate_tol})"
    c_fci = np.array([fci["ci_dict"].get(I, 0.0) for I in pool])
    c_fci = c_fci / np.linalg.norm(c_fci)
    print(f"    |pool|={len(pool)} [{ptag}]  E_HF={fci['E_HF']:.6f}  "
          f"E_FCI={fci['E_FCI']:.6f}")

    cell = Path(s["cell_dir"])
    walkers, _, _ = load_walker_bank(str(cell / f"{cell.name}_walkers.h5"))
    key = jax.random.split(jax.random.key(42), 4)[1]
    drv = get_vmc_nn_func(mol, args.ansatz, key, prefix=str(cell / cell.name))
    drv.load_checkpoint(str(cell / f"{cell.name}.chk.h5"))
    log_psi, _, _, _ = make_nn_log_psi(args.ansatz, mol, key)
    nuc = np.asarray(mol.atom_coords())
    psi = evaluate_signed_psi(walkers, nuc, drv.params, log_psi,
                              batch_size=args.psi_batch)
    print(f"[2] bank K_s={walkers.shape[0]}  sign mean "
          f"{np.mean(np.sign(psi)):+.3f}")

    if args.j_source == "cusp":
        print(f"[3] analytic Kato cusp-only factor (Strategy B, b={args.cusp_b})")
        J = kato_cusp_jastrow(walkers, na, nb, b=args.cusp_b,
                              layout="interleaved")
    else:
        print("[3] full learned Jastrow, retrofit jastrow-off (Strategy A)")
        J = nn_jastrow_on_walkers(args.ansatz, mol, key, drv.params, walkers,
                                  nuc, batch_size=args.psi_batch)
    print(f"    J range [{J.min():+.4f}, {J.max():+.4f}]  mean {J.mean():+.4f}"
          f"  e^-J dyn.range {np.exp(J.max()-J.min()):.1f}x")

    orb = evaluate_orbitals_on_walkers(mol, walkers, fci["no_coeff_ao"],
                                       convention="interleaved",
                                       n_alpha=na, n_beta=nb)
    ref_sign = float(np.sign(c_fci[0])) if c_fci[0] != 0 else 1.0
    print("[4] two-sided TC decode (tau = 0, +1, -1)")
    out = decode_two_sided(orb, pool, psi, J, n_alpha=na, n_beta=nb,
                           reference_sign=ref_sign, use_lasso=False)
    print_gate0_report(pool, c_fci, out, leak_thresh=args.leak_thresh,
                       top_k=args.top_k)


if __name__ == "__main__":
    main()
