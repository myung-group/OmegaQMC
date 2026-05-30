"""HCI warm-start with CS-recovered ĉ as the initial CI vector.

Compares iteration count to converge a selected-CI calculation when
initialized from:
  (a) the HF determinant (standard)
  (b) the CS-recovered ĉ from a trained NN-VMC checkpoint
on the same molecular cell.

Demonstrates the framework's enabling claim: ĉ from a trained NN
accelerates a downstream post-HF method.

Test target: BeH2/cc-pVDZ converged checkpoint.
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import jax

sys.path.insert(0, str(Path(__file__).parent))

from OmegaQMC.utils import Mole_custom
from OmegaQMC.cs.reference import compute_fci_reference
from OmegaQMC.cs.walkers import load_walker_bank
from OmegaQMC.cs.estimators import (
    evaluate_orbitals_on_walkers, f_I_matrix, normalize_and_align,
)
from OmegaQMC.cs.scaling import precompute_means
from OmegaQMC.psi.nn.adapter import make_nn_log_psi
from OmegaQMC.vmc_nn import get_vmc_nn_func

from run_cs_properties import build_mol
from run_cs_h4_scaling import evaluate_signed_psi


def hci_iter_to_converge(mol, mo_coeff, n_alpha, n_beta, initial_ci=None,
                         tol=1e-8, max_iter=200):
    """Run selected_ci and count iterations to convergence.

    Uses PySCF's selected_ci.SelectedCI. Returns dict with E, n_iter, time.
    """
    from pyscf.fci import selected_ci, direct_spin1
    sci = selected_ci.SCI(mol)
    sci.conv_tol = tol
    sci.max_cycle = max_iter
    n_orb = mo_coeff.shape[1]
    nelec = (n_alpha, n_beta)
    t0 = time.time()
    try:
        if initial_ci is not None:
            e, c = sci.kernel(
                mol.intor("int1e_kin") + mol.intor("int1e_nuc"),
                mol.intor("int2e", aosym="s4"),
                n_orb, nelec, ci0=initial_ci,
            )
        else:
            e, c = sci.kernel(
                mol.intor("int1e_kin") + mol.intor("int1e_nuc"),
                mol.intor("int2e", aosym="s4"),
                n_orb, nelec,
            )
        t = time.time() - t0
        n_iter = getattr(sci, "converged_iter", -1)
    except Exception as ex:
        return dict(E=float("nan"), n_iter=-1, time=time.time() - t0,
                    error=str(ex))
    return dict(E=float(e), n_iter=int(n_iter) if n_iter > 0 else -1,
                time=float(t),
                n_dets_final=int(c.size if hasattr(c, "size") else 0))


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
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cell_dir = Path(args.cell_dir)
    prefix = cell_dir.name
    print(f"=== HCI warm-start study: {prefix} ===")

    mol = build_mol(args.molecule, args.R, args.basis, args.unit,
                    args.geometry_tag)
    fci_ref = compute_fci_reference(mol, n_alpha=args.n_alpha,
                                    n_beta=args.n_beta, candidate_tol=1e-4)
    print(f"  E_FCI = {fci_ref['E_FCI']:.6f} Ha; "
          f"|S| = {len(fci_ref['candidate_set'])}")

    # CS recovery
    walkers, _, _ = load_walker_bank(str(cell_dir / f"{prefix}_walkers.h5"))
    key = jax.random.split(jax.random.key(42), 4)[1]
    drv = get_vmc_nn_func(mol, args.ansatz, key,
                          prefix=str(cell_dir / prefix))
    drv.load_checkpoint(str(cell_dir / f"{prefix}.chk.h5"))
    log_psi, _, _, _ = make_nn_log_psi(args.ansatz, mol, key)
    psi_vals = evaluate_signed_psi(walkers, np.asarray(mol.atom_coords()),
                                    drv.params, log_psi, batch_size=2048)
    K_s = walkers.shape[0]
    means, _ = precompute_means(mol, fci_ref, walkers, psi_vals,
                                 K_s_sweep=[K_s], n_seeds=1,
                                 walker_convention="interleaved",
                                 return_second_moments=True)
    c_raw = means[(K_s, 0)]
    c_true = np.array([fci_ref["ci_dict"][k] for k in fci_ref["candidate_set"]])
    c_hat, proj_mass = normalize_and_align(c_raw, float(np.sign(c_true[0])))
    print(f"  CS recovered: C0={c_hat[0]:+.4f}, ||ĉ||₂={np.linalg.norm(c_hat):.4f}")

    no_coeff = fci_ref["no_coeff_ao"]
    n_orb = int(fci_ref["n_orb"])

    # Build initial CI vector in PySCF (alpha-string × beta-string) format
    from OmegaQMC.cs.properties import reshape_chat_to_pyscf_matrix
    init_ci_hat = reshape_chat_to_pyscf_matrix(
        c_hat, fci_ref["candidate_set"], n_orb, args.n_alpha, args.n_beta,
    )
    print(f"  initial CI matrix shape: {init_ci_hat.shape}, "
          f"||C||={np.linalg.norm(init_ci_hat):.4f}")

    # HCI runs
    print("\n  [HCI from HF init]")
    hci_hf = hci_iter_to_converge(
        mol, no_coeff, args.n_alpha, args.n_beta, initial_ci=None,
    )
    print(f"    E = {hci_hf['E']:.6f}, n_iter = {hci_hf['n_iter']}, "
          f"time = {hci_hf['time']:.2f}s")

    print("\n  [HCI from ĉ-warm-start]")
    hci_ch = hci_iter_to_converge(
        mol, no_coeff, args.n_alpha, args.n_beta, initial_ci=init_ci_hat,
    )
    print(f"    E = {hci_ch['E']:.6f}, n_iter = {hci_ch['n_iter']}, "
          f"time = {hci_ch['time']:.2f}s")

    out = dict(
        prefix=prefix, molecule=args.molecule, R=args.R, basis=args.basis,
        E_FCI=float(fci_ref["E_FCI"]),
        hf_init=hci_hf, chat_init=hci_ch,
        C0_chat=float(c_hat[0]),
    )
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  wrote {args.out}")


if __name__ == "__main__":
    main()
