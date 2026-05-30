"""S² + 2-RDM-derived properties from the CS-recovered CI vector.

Extends §VIII (RDMs + chemical interpretation) with three additional
diagnostics computable from the 1- and 2-RDM contractions of ĉ:

  ⟨S²⟩            spin contamination
  ⟨S_z⟩           z-spin
  pair density    n_2(r1, r2) at a few representative geometries
  on-top density  n_2(r, r) at the nuclei (for chemists' KS-DFT analogy)

The 2-RDM is obtained via PySCF's direct_spin1.make_rdm12 on the
recovered ĉ projected into the candidate basis. ⟨S²⟩ is computed via
the standard CI-vector contraction (fci.spin_op.spin_square0).

Test target: converged BeH2/cc-pVDZ checkpoint.
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import jax

sys.path.insert(0, str(Path(__file__).parent))

from OmegaQMC.utils import Mole_custom
from OmegaQMC.cs.reference import compute_fci_reference
from OmegaQMC.cs.walkers import load_walker_bank
from OmegaQMC.cs.estimators import normalize_and_align
from OmegaQMC.cs.scaling import precompute_means
from OmegaQMC.cs.properties import reshape_chat_to_pyscf_matrix
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
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cell_dir = Path(args.cell_dir)
    prefix = cell_dir.name
    print(f"=== S² + 2-RDM properties: {prefix} ===")

    mol = build_mol(args.molecule, args.R, args.basis, args.unit,
                    args.geometry_tag)
    fci_ref = compute_fci_reference(mol, n_alpha=args.n_alpha,
                                    n_beta=args.n_beta, candidate_tol=1e-4)
    n_orb = int(fci_ref["n_orb"])
    nelec = (args.n_alpha, args.n_beta)

    # CS recovery (standard pipeline)
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
    c_hat, _ = normalize_and_align(c_raw, float(np.sign(c_true[0])))

    # Reshape to PySCF (alpha-string, beta-string) matrix
    ci_chat = reshape_chat_to_pyscf_matrix(
        c_hat, fci_ref["candidate_set"], n_orb, *nelec
    )
    ci_fci = reshape_chat_to_pyscf_matrix(
        c_true, fci_ref["candidate_set"], n_orb, *nelec
    )

    # S² and S_z via PySCF spin_op
    from pyscf.fci.spin_op import spin_square0
    s2_chat, mult_chat = spin_square0(ci_chat, n_orb, nelec)
    s2_fci, mult_fci = spin_square0(ci_fci, n_orb, nelec)
    print(f"  ⟨S²⟩  : chat = {s2_chat:.6f}  vs FCI = {s2_fci:.6f}")
    print(f"  spin mult (2S+1):  chat = {mult_chat:.4f}  vs FCI = {mult_fci:.4f}")

    # 1-RDM and 2-RDM contractions
    from pyscf.fci import direct_spin1
    gamma_chat, Gamma_chat = direct_spin1.make_rdm12(ci_chat, n_orb, nelec)
    gamma_fci, Gamma_fci = direct_spin1.make_rdm12(ci_fci, n_orb, nelec)

    # On-top pair density at each nucleus
    no_coeff = fci_ref["no_coeff_ao"]
    nuclei = np.asarray(mol.atom_coords())  # (n_nuc, 3)
    ao_vals_nuc = mol.eval_gto("GTOval_sph", nuclei)  # (n_nuc, n_AO)
    orb_vals_nuc = ao_vals_nuc @ no_coeff             # (n_nuc, n_orb)
    n_top_chat = []
    n_top_fci = []
    for k in range(len(nuclei)):
        # n_2(r, r) = Σ_{pqrs} Γ_{pqrs} η_p(r) η_q(r) η_r(r) η_s(r)
        # Here we use the standard convention: Γ_pqrs = Σ_{IJ} c_I c_J
        # <D_I | a†_p a†_r a_s a_q | D_J> giving a 4-index tensor.
        v = orb_vals_nuc[k]
        # On-top density: n_2(r,r) = Σ G[p,q,r,s] η_p(r)η_q(r)η_r(r)η_s(r)
        n_top_chat.append(float(np.einsum(
            "pqrs,p,q,r,s->", Gamma_chat, v, v, v, v
        )))
        n_top_fci.append(float(np.einsum(
            "pqrs,p,q,r,s->", Gamma_fci, v, v, v, v
        )))
    print(f"  on-top n_2(r=nucleus) chat: {n_top_chat}")
    print(f"  on-top n_2(r=nucleus) FCI : {n_top_fci}")

    # Tr(Γ) sanity check: should be N_e * (N_e - 1) / 2 (spin-summed
    # PySCF convention is N(N-1) without the half).
    tr_G_chat = float(np.einsum("pqpq->", Gamma_chat))
    tr_G_fci = float(np.einsum("pqpq->", Gamma_fci))
    N = args.n_alpha + args.n_beta
    print(f"  Tr(Γ_2)/2: chat={tr_G_chat/2:.4f}  FCI={tr_G_fci/2:.4f}  "
          f"expected = N(N-1)/2 = {N*(N-1)/2}")

    out = dict(
        prefix=prefix, molecule=args.molecule, R=args.R, basis=args.basis,
        n_orb=n_orb, nelec=list(nelec),
        s2_chat=float(s2_chat), s2_fci=float(s2_fci),
        mult_chat=float(mult_chat), mult_fci=float(mult_fci),
        n_top_chat=n_top_chat, n_top_fci=n_top_fci,
        tr_Gamma_chat=tr_G_chat, tr_Gamma_fci=tr_G_fci,
    )
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  wrote {args.out}")


if __name__ == "__main__":
    main()
