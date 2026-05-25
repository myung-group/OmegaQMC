"""H2 Pfau-NES + CS-recovery + spectroscopy: the full §VI.B pipeline.

Demonstrates the novelty stack that compressed sensing brings on top
of Pfau et al.'s 2024 determinantal NES-VMC:

  Pfau-NES K=2  ->  CS recovery of c^(0), c^(1)  ->  three downstream
                                                     applications:
  (1) Transition density matrix gamma^(01) and transition dipole
      mu_{01} + oscillator strength f_{01} (length gauge).
  (2) Natural Transition Orbitals: SVD of gamma^(01) -> hole/particle
      orbital pairs giving the one-electron picture of the excitation.
  (3) State-specific NEVPT2 on top of c^(0) and c^(1), recovering the
      dynamic correlation outside the CS-recovery basis. Reports the
      CAS+PT2 vertical excitation energy for direct comparison to FCI.

The Pfau-NES training step itself is reused from
:mod:`OmegaQMC.vmcopt_nn_pfau`. If a trained checkpoint already
exists in the output directory, training is skipped.

Output: a printed summary suitable for direct insertion in the paper,
plus a JSON dump of all numerical results for downstream plotting.

Usage:
    python examples/run_h2_pfau_nes_spectroscopy.py \\
        --R 2.5 --basis cc-pvdz \\
        --pfau-iters 500 --init-from-ground
"""

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")

import jax
import numpy as np

from OmegaQMC.utils import Mole_custom
from OmegaQMC.vmc_nn import get_vmc_nn_func
from OmegaQMC.vmcopt_nn_pfau import get_vmcopt_nn_pfau_k2_func
from OmegaQMC.cs.reference import compute_fci_reference
from OmegaQMC.cs.walkers import load_walker_bank
from OmegaQMC.cs.estimators import (
    evaluate_orbitals_on_walkers, f_I_matrix, normalize_and_align,
)
from OmegaQMC.cs.transition import (
    report_transition_properties, print_transition_summary,
)
from OmegaQMC.cs.mrpt import run_multistate_nevpt2
from OmegaQMC.psi.nn.adapter import make_nn_log_psi

sys.path.insert(0, str(Path(__file__).parent))
from run_cs_h4_scaling import evaluate_signed_psi  # noqa: E402


def build_h2(R_bohr, basis):
    mol = Mole_custom()
    mol.build(
        atom=[("H", [0, 0, 0]), ("H", [0, 0, R_bohr])],
        basis=basis, spin=0, charge=0, unit="Bohr", verbose=0,
    )
    return mol


def recover_c_hat(walker_bank_path, ckpt_path, mol, fci_ref, ansatz,
                  key_seed):
    walkers, _, _ = load_walker_bank(walker_bank_path)
    init_key = jax.random.split(jax.random.key(key_seed), 4)[1]
    driver = get_vmc_nn_func(mol, ansatz, init_key,
                              prefix=ckpt_path.replace(".chk.h5", ""))
    driver.load_checkpoint(ckpt_path)
    log_psi, _, _, _ = make_nn_log_psi(ansatz, mol, init_key)
    psi_vals = evaluate_signed_psi(walkers, np.asarray(mol.atom_coords()),
                                    driver.params, log_psi)
    n_alpha, n_beta = fci_ref["nelec"]
    orb = evaluate_orbitals_on_walkers(
        mol, walkers, fci_ref["no_coeff_ao"],
        convention="interleaved", n_alpha=n_alpha, n_beta=n_beta,
    )
    f_I = f_I_matrix(orb, fci_ref["candidate_set"], psi_vals,
                     n_alpha=n_alpha, n_beta=n_beta)
    c_raw = f_I.mean(axis=1)
    c_true = np.array([fci_ref["ci_dict"][k]
                       for k in fci_ref["candidate_set"]])
    c_hat, proj_mass = normalize_and_align(
        c_raw, float(np.sign(c_true[0])),
    )
    return c_hat, proj_mass


def estimate_vmc_energy(walker_bank_path, ckpt_path, mol, ansatz, key_seed):
    """Cheap walker-based energy estimate for a sampled state."""
    from OmegaQMC.vmcopt_nn_iradam import _VMCOptDriverNN_IRAdam
    walkers, _, _ = load_walker_bank(walker_bank_path)
    init_key = jax.random.split(jax.random.key(key_seed), 4)[1]
    driver_iradam = _VMCOptDriverNN_IRAdam(mol, ansatz, init_key)
    init_params = driver_iradam.init_params
    from OmegaQMC.psi.nn.checkpoint import load_nn_checkpoint
    params, _ = load_nn_checkpoint(ckpt_path, init_params)
    e_list = []
    for si in range(0, walkers.shape[0], 256):
        ei = min(si + 256, walkers.shape[0])
        e_list.append(driver_iradam.compute_batch_energy(
            walkers[si:ei], params,
        ))
    energies = np.concatenate([np.asarray(e) for e in e_list])
    return float(np.mean(energies)), float(np.std(energies))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--R", type=float, default=2.5)
    p.add_argument("--basis", default="cc-pvdz")
    p.add_argument("--ansatz",
                   default=str(Path(__file__).parent / "inputs"
                               / "psiformer_small.yaml"))
    p.add_argument("--out-dir", default="cs_h2_pfau_spec_results")
    p.add_argument("--gs-source-dir", default="cs_h2_nesvmc_results")
    p.add_argument("--seed", type=int, default=77)
    p.add_argument("--pfau-iters", type=int, default=500)
    p.add_argument("--num-walkers", type=int, default=256)
    p.add_argument("--num-sample-blocks", type=int, default=2)
    p.add_argument("--num-epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--mc-timestep", type=float, default=0.1)
    p.add_argument("--sample-blocks", type=int, default=50)
    p.add_argument("--sample-walkers", type=int, default=256)
    p.add_argument("--candidate-tol", type=float, default=1e-6)
    p.add_argument("--init-from-ground", action="store_true")
    p.add_argument("--skip-train", action="store_true",
                   help="reuse existing Pfau-NES checkpoints if present")
    args = p.parse_args()

    R_str = f"R{args.R:.3f}".replace(".", "p")
    basis_tag = args.basis.replace("*", "s")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = out_dir / f"h2_pfau_{R_str}_{basis_tag}"
    chk_1 = Path(f"{prefix}_1.chk.h5")
    chk_2 = Path(f"{prefix}_2.chk.h5")
    bank_1 = out_dir / (prefix.name + "_1_walkers.h5")
    bank_2 = out_dir / (prefix.name + "_2_walkers.h5")

    chk_gs_ref = None
    if args.init_from_ground:
        src_dir = Path(args.gs_source_dir)
        chk_gs_ref = src_dir / f"h2_gs_{R_str}_{basis_tag}.chk.h5"
        if not chk_gs_ref.exists():
            sys.exit(f"missing GS checkpoint {chk_gs_ref}; "
                     f"run examples/run_h2_nesvmc.py first or omit "
                     f"--init-from-ground")

    print(f"=== H2 Pfau-NES + CS spectroscopy at R={args.R} a0, "
          f"{args.basis} ===\n")
    mol = build_h2(args.R, args.basis)
    fci_ref = compute_fci_reference(
        mol, n_alpha=1, n_beta=1,
        candidate_tol=args.candidate_tol,
    )
    n_det = len(fci_ref["candidate_set"])
    print(f"  mol: {mol.nao} AOs, |candidate set| = {n_det}")
    print(f"  E_FCI (ground, full {mol.nao}-orb) = "
          f"{fci_ref['E_FCI']:.6f} Ha")

    # --- 1. Train Pfau-NES K=2 ---
    if not (chk_1.exists() and chk_2.exists()) and not args.skip_train:
        print(f"\n[Step 1] Training Pfau-NES K=2 "
              f"({args.pfau_iters} iters)")
        init_key = jax.random.key(args.seed)
        driver = get_vmcopt_nn_pfau_k2_func(
            mol, args.ansatz, init_key,
            init_from_ground_checkpoint=(
                str(chk_gs_ref) if chk_gs_ref else None
            ),
        )
        rng_opt = jax.random.split(init_key, 4)[3]
        driver(
            rng_opt,
            num_iters=args.pfau_iters,
            num_walkers=args.num_walkers,
            num_sample_blocks=args.num_sample_blocks,
            num_epochs=args.num_epochs,
            batch_size=args.batch_size,
            mc_timestep=args.mc_timestep,
            lr=args.lr,
            prefix=str(prefix),
            verbose=1,
        )
    else:
        print(f"\n[Step 1] Reusing Pfau-NES checkpoints")

    # --- 2. Sample walker banks from each state ---
    for k, (chk, bank) in enumerate([(chk_1, bank_1), (chk_2, bank_2)], 1):
        if not bank.exists():
            print(f"\n[Step 2.{k}] Sampling Psi_{k} walker bank")
            rng = jax.random.key(args.seed + k)
            init_key, smp_key = jax.random.split(rng)
            drv = get_vmc_nn_func(mol, args.ansatz, init_key,
                                   prefix=f"{prefix}_{k}")
            drv.load_checkpoint(str(chk))
            drv(
                smp_key,
                num_walkers=args.sample_walkers,
                num_steps_per_block=20,
                num_blocks=args.sample_blocks,
                num_blocks_equil=5,
                mc_timestep=args.mc_timestep,
                compute_gradients=False,
                dump_walkers_path=str(bank),
                verbose=1,
            )
        else:
            print(f"\n[Step 2.{k}] Reusing Psi_{k} bank {bank.name}")

    # --- 3. CS recovery of both c vectors + per-state energy ---
    print(f"\n[Step 3] CS recovery + per-state energy")
    c_hat_1, pm_1 = recover_c_hat(
        str(bank_1), str(chk_1), mol, fci_ref, args.ansatz,
        key_seed=args.seed + 10,
    )
    c_hat_2, pm_2 = recover_c_hat(
        str(bank_2), str(chk_2), mol, fci_ref, args.ansatz,
        key_seed=args.seed + 20,
    )
    E_1, sig_1 = estimate_vmc_energy(
        str(bank_1), str(chk_1), mol, args.ansatz, args.seed + 30,
    )
    E_2, sig_2 = estimate_vmc_energy(
        str(bank_2), str(chk_2), mol, args.ansatz, args.seed + 40,
    )
    # Order by energy: state with lower E becomes the "ground" for the
    # downstream transition analysis
    if E_2 < E_1:
        c_g, c_e = c_hat_2, c_hat_1
        E_g, sig_g, pm_g = E_2, sig_2, pm_2
        E_e, sig_e, pm_e = E_1, sig_1, pm_1
        which = "Psi_2 is ground, Psi_1 is excited"
    else:
        c_g, c_e = c_hat_1, c_hat_2
        E_g, sig_g, pm_g = E_1, sig_1, pm_1
        E_e, sig_e, pm_e = E_2, sig_2, pm_2
        which = "Psi_1 is ground, Psi_2 is excited"
    print(f"  {which}")
    print(f"  E_ground   = {E_g:.6f} +/- {sig_g/np.sqrt(256):.5f} Ha, "
          f"proj_mass = {pm_g:.3e}")
    print(f"  E_excited  = {E_e:.6f} +/- {sig_e/np.sqrt(256):.5f} Ha, "
          f"proj_mass = {pm_e:.3e}")
    print(f"  c_ground (top 3 by |c|):   "
          f"{[(str(fci_ref['candidate_set'][i]), f'{c_g[i]:+.4f}')
                for i in np.argsort(-np.abs(c_g))[:3]]}")
    print(f"  c_excited (top 3 by |c|):  "
          f"{[(str(fci_ref['candidate_set'][i]), f'{c_e[i]:+.4f}')
                for i in np.argsort(-np.abs(c_e))[:3]]}")
    print(f"  <c_g | c_e> = {float(np.dot(c_g, c_e)):+.5f}")

    # --- 4. Transition properties (Pillar 1 + 2) ---
    print(f"\n[Step 4] Transition properties + NTO analysis")
    delta_E = E_e - E_g
    trans = report_transition_properties(
        c_g, c_e, fci_ref, mol, delta_E_au=delta_E,
    )
    print_transition_summary(trans, label="ground -> excited")
    # Dominant NTO pair
    nto = trans["nto"]
    print(f"  Dominant NTO pair (weight {nto['participation_ratios'][0]*100:.1f}%):")
    print(f"    hole NTO     (AO coeffs, first 3): "
          f"{nto['hole_nto_ao'][:3, 0].round(4).tolist()}")
    print(f"    particle NTO (AO coeffs, first 3): "
          f"{nto['particle_nto_ao'][:3, 0].round(4).tolist()}")

    # --- 5. State-specific NEVPT2 (Pillar 3) ---
    print(f"\n[Step 5] State-specific NEVPT2 on Pfau-NES references")
    try:
        nevpt2 = run_multistate_nevpt2(
            mol, c_hats=[c_g, c_e], fci_ref=fci_ref,
            state_labels=["ground", "excited"],
            nroots_max=4,
        )
        print(f"  Active space: CAS({nevpt2['ncas']},{nevpt2['nelecas']}), "
              f"ncore = {nevpt2['ncore']}")
        for k, res in enumerate(nevpt2["per_state"]):
            print(f"  {res['label']:>10s}: "
                  f"E_CASCI = {res['e_casci']:.6f}, "
                  f"E_PT2 = {res['e_pt2']:+.6f}, "
                  f"E_total = {res['e_total']:.6f} Ha, "
                  f"match-overlap = {res['max_overlap']:.3f}")
        dE_casci_eV = nevpt2["delta_E_casci_au"][0, 1] * 27.2114
        dE_total_eV = nevpt2["delta_E_total_au"][0, 1] * 27.2114
        print(f"  vertical excitation (CASCI):    "
              f"{dE_casci_eV:.4f} eV ({nevpt2['delta_E_casci_au'][0,1]*1000:.2f} mE_h)")
        print(f"  vertical excitation (CAS+PT2):  "
              f"{dE_total_eV:.4f} eV ({nevpt2['delta_E_total_au'][0,1]*1000:.2f} mE_h)")
    except Exception as exc:
        nevpt2 = None
        import traceback
        print(f"  NEVPT2 step failed: {type(exc).__name__}: {exc}")
        traceback.print_exc()

    # --- 6. Save numerical results ---
    summary = dict(
        R_bohr=float(args.R),
        basis=args.basis,
        E_FCI_ground_full=float(fci_ref["E_FCI"]),
        E_g=float(E_g), E_e=float(E_e),
        proj_mass_g=float(pm_g), proj_mass_e=float(pm_e),
        ci_overlap=float(np.dot(c_g, c_e)),
        c_g=c_g.tolist(), c_e=c_e.tolist(),
        candidate_set=[[list(a), list(b)] for (a, b) in fci_ref["candidate_set"]],
        delta_E_au=float(delta_E),
        transition_dipole_au=trans["transition_dipole"]["mu_au"].tolist(),
        transition_dipole_debye=trans["transition_dipole"]["mu_debye"].tolist(),
        oscillator_strength=trans["oscillator_strength"],
        nto_singular_values=trans["nto"]["singular_values"].tolist(),
        nto_participation_ratios=trans["nto"]["participation_ratios"].tolist(),
    )
    if nevpt2 is not None:
        summary["nevpt2"] = dict(
            ncore=int(nevpt2["ncore"]),
            ncas=int(nevpt2["ncas"]),
            nelecas=list(nevpt2["nelecas"]),
            e_casci=nevpt2["e_casci"].tolist(),
            e_pt2=nevpt2["e_pt2"].tolist(),
            e_total=nevpt2["e_total"].tolist(),
            delta_E_total_au=nevpt2["delta_E_total_au"].tolist(),
            per_state=[
                dict(label=r["label"], root_index=int(r["root_index"]),
                     max_overlap=float(r["max_overlap"]))
                for r in nevpt2["per_state"]
            ],
        )

    json_path = out_dir / f"h2_pfau_spec_{R_str}_{basis_tag}.json"
    with open(json_path, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"\n[done] Numerical summary written to {json_path}")


if __name__ == "__main__":
    main()
