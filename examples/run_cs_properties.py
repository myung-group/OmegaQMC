"""Properties from a CS-recovered CI vector — head-to-head against FCI.

Reads an existing CS pilot cell (trained NN-VMC checkpoint + walker
bank), recovers ``c_hat`` via the standard pipeline, and prints a
publication-ready table comparing ``c_hat``-derived 1-RDM properties
against PySCF FCI in the same natural-orbital basis. Output is
suitable for direct inclusion as a paper table.

Usage:
    python examples/run_cs_properties.py --cell-dir cs_h2_validation/h2_R2p500_cc-pvdz \
        --geometry h2 --R 2.5 --basis cc-pvdz --ansatz examples/inputs/psiformer_small.yaml \
        --n-alpha 1 --n-beta 1 --unit Bohr

For the H4 linear cell from the methods paper:
    python examples/run_cs_properties.py --cell-dir cs_h4_results/h4_linear_ferminet_jastrow_R1p000_cc-pvdz \
        --geometry h4 --R 1.0 --basis cc-pvdz \
        --ansatz OmegaQMC/psi/nn/conf/ferminet_jastrow.yaml \
        --n-alpha 2 --n-beta 2 --geometry-tag linear --unit Angstrom
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
from OmegaQMC.cs.reference import compute_fci_reference
from OmegaQMC.cs.walkers import load_walker_bank
from OmegaQMC.cs.estimators import (
    evaluate_orbitals_on_walkers, f_I_matrix,
    normalize_and_align, normalize_and_align_bias_corrected,
)
from OmegaQMC.cs.scaling import precompute_means
from OmegaQMC.cs.properties import report_properties, compare_to_fci
from OmegaQMC.psi.nn.adapter import make_nn_log_psi
from OmegaQMC.vmc_nn import get_vmc_nn_func

sys.path.insert(0, str(Path(__file__).parent))
from run_cs_h4_scaling import evaluate_signed_psi  # noqa: E402


def build_mol(geometry: str, R: float, basis: str, unit: str,
              geometry_tag: str = "") -> Mole_custom:
    if geometry == "h2":
        atoms = [("H", [0, 0, 0]), ("H", [0, 0, R])]
    elif geometry == "h4" and geometry_tag == "linear":
        atoms = [("H", [0, 0, k * R]) for k in range(4)]
    elif geometry == "h4" and geometry_tag == "square":
        atoms = [
            ("H", [0, 0, 0]),
            ("H", [R, 0, 0]),
            ("H", [R, R, 0]),
            ("H", [0, R, 0]),
        ]
    elif geometry == "lih":
        atoms = [("Li", [0, 0, 0]), ("H", [0, 0, R])]
    elif geometry == "beh2":
        atoms = [("Be", [0, 0, 0]),
                 ("H", [0, 0, +R]), ("H", [0, 0, -R])]
    elif geometry == "h2o":
        atoms = [("O", [0.000, 0.000, 0.000]),
                 ("H", [0.758, 0.586, 0.000]),
                 ("H", [-0.758, 0.586, 0.000])]
    elif geometry == "n2":
        atoms = [("N", [0, 0, 0]), ("N", [0, 0, R])]
    else:
        raise ValueError(f"unknown geometry '{geometry}' / tag '{geometry_tag}'")
    mol = Mole_custom()
    mol.build(atom=atoms, basis=basis, spin=0, charge=0,
              unit=unit, verbose=0)
    return mol


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cell-dir", required=True,
                   help="Cell directory containing the checkpoint + walker bank")
    p.add_argument("--geometry", required=True,
                   choices=["h2", "h4", "lih", "beh2", "h2o", "n2"])
    p.add_argument("--geometry-tag", default="",
                   help="extra geometry qualifier (e.g. 'linear' or 'square' for H4)")
    p.add_argument("--R", type=float, default=1.0)
    p.add_argument("--basis", default="cc-pvdz")
    p.add_argument("--unit", default="Bohr", choices=["Bohr", "Angstrom"])
    p.add_argument("--ansatz", required=True)
    p.add_argument("--n-alpha", type=int, required=True)
    p.add_argument("--n-beta", type=int, required=True)
    p.add_argument("--psi-batch", type=int, default=2048)
    p.add_argument("--det-chunk", type=int, default=200,
                   help="determinant-chunk size for streaming f_I (memory bound)")
    p.add_argument("--candidate-tol", type=float, default=1e-4,
                   help="drop FCI determinants with |c_FCI| <= this "
                        "from the candidate set; default 1e-4 filters out "
                        "noise-dominated coefficients in large bases")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-json", default=None,
                   help="Optional path for a JSON dump of all numbers")
    args = p.parse_args()

    cell_dir = Path(args.cell_dir)
    prefix = cell_dir.name
    walkers_path = cell_dir / f"{prefix}_walkers.h5"
    checkpoint = cell_dir / f"{prefix}.chk.h5"
    if not walkers_path.exists() or not checkpoint.exists():
        sys.exit(f"missing checkpoint or walker bank in {cell_dir}")

    print(f"=== CS properties — {prefix} ===")
    mol = build_mol(args.geometry, args.R, args.basis, args.unit,
                    args.geometry_tag)
    print(f"  mol: {mol.nao} AOs, {mol.nelec} electrons, basis={args.basis}")

    fci_ref = compute_fci_reference(
        mol, n_alpha=args.n_alpha, n_beta=args.n_beta,
        candidate_tol=args.candidate_tol,
    )
    print(f"  candidate_tol = {args.candidate_tol:.0e}  "
          f"|candidate set| = {len(fci_ref['candidate_set'])}")
    print(f"  E_HF  = {fci_ref['E_HF']: .6f} Ha")
    print(f"  E_FCI = {fci_ref['E_FCI']: .6f} Ha")

    walkers, _, _ = load_walker_bank(str(walkers_path))
    init_key = jax.random.split(jax.random.key(args.seed), 4)[1]
    driver = get_vmc_nn_func(mol, args.ansatz, init_key, prefix=str(cell_dir / prefix))
    driver.load_checkpoint(str(checkpoint))
    log_psi, _, _, _ = make_nn_log_psi(args.ansatz, mol, init_key)
    psi_vals = evaluate_signed_psi(
        walkers, np.asarray(mol.atom_coords()),
        driver.params, log_psi, batch_size=args.psi_batch,
    )
    # Stream f_I in determinant chunks to bound memory at L * K_s_max
    # instead of n_det * K_s_max. Also accumulate second moments so we
    # can subtract the per-coefficient variance bias from proj_mass.
    # The bias scales as n_det/(Z K_s) and dominates the signal at
    # large basis (e.g. cc-pVDZ for atoms with valence shells).
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
    c_true = np.array([fci_ref["ci_dict"][k]
                       for k in fci_ref["candidate_set"]])
    from OmegaQMC.cs.estimators import bias_corrected_proj_mass
    c_hat, proj_mass = normalize_and_align(c_raw, float(np.sign(c_true[0])))
    proj_mass_corrected = bias_corrected_proj_mass(c_raw, m2, K_s_max)

    chat_rep = report_properties(c_hat, fci_ref, mol)
    cmp = compare_to_fci(chat_rep, fci_ref, mol)
    ch, fc, dl = cmp["chat"], cmp["fci"], cmp["delta"]

    print(f"\n  K_s = {K_s_max}")
    print(f"  proj_mass (naive, used)    = {proj_mass:.4e}")
    print(f"  proj_mass (bias-corrected) = {proj_mass_corrected:.4e}  "
          f"(bias = {proj_mass - proj_mass_corrected:+.4e})")
    print(f"\n{'property':<38} {'c_hat':>14} {'c_FCI':>14} {'diff':>14}")
    print("-" * 82)
    n_show = min(5, len(ch["occupations"]))
    for k in range(n_show):
        print(f"  {'NO occ #' + str(k+1):<36} "
              f"{ch['occupations'][k]:>14.6f} {fc['occupations'][k]:>14.6f} "
              f"{ch['occupations'][k]-fc['occupations'][k]:>+14.4e}")
    print(f"  {'Tr(gamma) [N_elec]':<36} "
          f"{ch['trace_gamma']:>14.6f} {fc['trace_gamma']:>14.6f} "
          f"{dl['trace_gamma_diff']:>+14.4e}")
    print(f"  {'C0 (HF amplitude)':<36} "
          f"{ch['diagnostics']['C0']:>+14.6f} "
          f"{fc['diagnostics']['C0']:>+14.6f} "
          f"{dl['C0_diff']:>+14.4e}")
    print(f"  {'C0^2':<36} "
          f"{ch['diagnostics']['C0_sq']:>14.6f} "
          f"{fc['diagnostics']['C0_sq']:>14.6f}")
    print(f"  {'Eff. unpaired (Head-Gordon)':<36} "
          f"{ch['diagnostics']['effective_unpaired']:>14.6f} "
          f"{fc['diagnostics']['effective_unpaired']:>14.6f} "
          f"{dl['effective_unpaired_diff']:>+14.4e}")
    print(f"  {'Dipole |mu| (Debye)':<36} "
          f"{ch['dipole']['mu_magnitude_debye']:>14.6f} "
          f"{fc['dipole']['mu_magnitude_debye']:>14.6f}")
    for ax in range(3):
        print(f"  {'Dipole mu_' + 'xyz'[ax] + ' (Debye)':<36} "
              f"{ch['dipole']['mu_debye'][ax]:>+14.6f} "
              f"{fc['dipole']['mu_debye'][ax]:>+14.6f}")
    for ax in range(3):
        comp = "xx yy zz".split()[ax]
        print(f"  {'Quadrupole Q_' + comp + ' (au)':<36} "
              f"{ch['quadrupole'][ax,ax]:>+14.6f} "
              f"{fc['quadrupole'][ax,ax]:>+14.6f}")

    if args.out_json:
        out = dict(
            prefix=prefix, geometry=args.geometry,
            geometry_tag=args.geometry_tag,
            R=args.R, basis=args.basis,
            n_alpha=args.n_alpha, n_beta=args.n_beta,
            E_HF=fci_ref["E_HF"], E_FCI=fci_ref["E_FCI"],
            K_s=int(K_s_max),
            proj_mass=float(proj_mass),
            proj_mass_bias_corrected=float(proj_mass_corrected),
            chat=dict(
                occupations=ch["occupations"].tolist(),
                trace_gamma=ch["trace_gamma"],
                C0=ch["diagnostics"]["C0"],
                C0_sq=ch["diagnostics"]["C0_sq"],
                effective_unpaired=ch["diagnostics"]["effective_unpaired"],
                mu_debye=ch["dipole"]["mu_debye"].tolist(),
                mu_magnitude_debye=ch["dipole"]["mu_magnitude_debye"],
                quadrupole_diag_au=np.diag(ch["quadrupole"]).tolist(),
            ),
            fci=dict(
                occupations=fc["occupations"].tolist(),
                trace_gamma=fc["trace_gamma"],
                C0=fc["diagnostics"]["C0"],
                C0_sq=fc["diagnostics"]["C0_sq"],
                effective_unpaired=fc["diagnostics"]["effective_unpaired"],
                mu_debye=fc["dipole"]["mu_debye"].tolist(),
                mu_magnitude_debye=fc["dipole"]["mu_magnitude_debye"],
                quadrupole_diag_au=np.diag(fc["quadrupole"]).tolist(),
            ),
        )
        with open(args.out_json, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nJSON dump -> {args.out_json}")


if __name__ == "__main__":
    main()
