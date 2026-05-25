"""H2O Pfau-NES K-general spectroscopy: consumes K pre-trained checkpoints.

Workflow:
  1. Sample each of the K trained states (walker bank per state)
  2. CS-recover each c_hat^(k) in the natural-orbital basis
  3. Subspace-rotate the K-dim recovered span to extract eigenstates
  4. For each (ground, excited_k) pair, compute the transition dipole
     and oscillator strength after rotation
  5. Select the dipole-allowed excited state with maximum oscillator
     strength as the ``target'' transition for the paper Table
  6. Run state-specific NEVPT2 on the (ground, target) pair to add
     dynamic correlation outside the CAS

This is the post-processing pipeline that turns Pfau-NES K>=2
training output into spectroscopically usable quantities, without
re-running the (expensive) joint-walker training.
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")

import jax
import numpy as np

from OmegaQMC.utils import Mole_custom
from OmegaQMC.vmc_nn import get_vmc_nn_func
from OmegaQMC.cs.reference import (
    compute_fci_reference, compute_casci_reference,
)
from OmegaQMC.cs.walkers import load_walker_bank
from OmegaQMC.cs.estimators import (
    evaluate_orbitals_on_walkers, f_I_matrix, normalize_and_align,
)
from OmegaQMC.cs.transition import (
    subspace_rotate_to_eigenstates,
    report_transition_properties,
    print_transition_summary,
)
from OmegaQMC.cs.mrpt import run_multistate_nevpt2
from OmegaQMC.psi.nn.adapter import make_nn_log_psi

sys.path.insert(0, str(Path(__file__).parent))
from run_cs_h4_scaling import evaluate_signed_psi  # noqa: E402


def build_h2o(basis):
    r = 0.957
    theta = math.radians(104.5 / 2.0)
    h_x = r * math.sin(theta)
    h_z = r * math.cos(theta)
    mol = Mole_custom()
    mol.build(
        atom=[("O", [0.0, 0.0, 0.0]),
              ("H", [h_x, 0.0, h_z]),
              ("H", [-h_x, 0.0, h_z])],
        basis=basis, spin=0, charge=0, unit="Angstrom", verbose=0,
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
    from OmegaQMC.vmcopt_nn_iradam import _VMCOptDriverNN_IRAdam
    from OmegaQMC.psi.nn.checkpoint import load_nn_checkpoint
    walkers, _, _ = load_walker_bank(walker_bank_path)
    init_key = jax.random.split(jax.random.key(key_seed), 4)[1]
    driver_iradam = _VMCOptDriverNN_IRAdam(mol, ansatz, init_key)
    init_params = driver_iradam.init_params
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
    p.add_argument("--basis", default="cc-pvdz")
    p.add_argument("--ansatz",
                   default=str(Path(__file__).parent / "inputs"
                               / "psiformer_small.yaml"))
    p.add_argument("--K", type=int, default=4)
    p.add_argument("--checkpoint-dir", default="cs_h2o_pfau_k4_results")
    p.add_argument("--out-dir", default="cs_h2o_pfau_k4_results")
    p.add_argument("--seed", type=int, default=77)
    p.add_argument("--sample-blocks", type=int, default=50)
    p.add_argument("--sample-walkers", type=int, default=256)
    p.add_argument("--mc-timestep", type=float, default=0.05)
    p.add_argument("--candidate-tol", type=float, default=1e-4)
    p.add_argument("--cas-ncas", type=int, default=8)
    p.add_argument("--cas-nelecas", type=str, default="4,4")
    p.add_argument("--nevpt2-occ-threshold", type=float, default=0.02)
    args = p.parse_args()

    basis_tag = args.basis.replace("*", "s")
    chk_dir = Path(args.checkpoint_dir)
    chk_paths = [chk_dir / f"h2o_pfau_k{args.K}_{basis_tag}_{k+1}.chk.h5"
                 for k in range(args.K)]
    for p_chk in chk_paths:
        if not p_chk.exists():
            sys.exit(f"missing K={args.K} checkpoint {p_chk}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    bank_paths = [out_dir / f"h2o_pfau_k{args.K}_{basis_tag}_{k+1}_walkers.h5"
                  for k in range(args.K)]

    print(f"=== H2O Pfau-NES K={args.K} spectroscopy, {args.basis} ===\n")
    mol = build_h2o(args.basis)
    print(f"  mol: {mol.nao} AOs, nelec={mol.nelec}")

    nelecas_parts = args.cas_nelecas.split(",")
    nelecas = (int(nelecas_parts[0]), int(nelecas_parts[1]))
    print(f"  Using CASCI({args.cas_ncas},{nelecas}) reference")
    fci_ref = compute_casci_reference(
        mol, ncas=args.cas_ncas, nelecas=nelecas,
        candidate_tol=args.candidate_tol,
    )
    n_det = len(fci_ref["candidate_set"])
    print(f"  |candidate set| = {n_det}")
    print(f"  E_CASCI(ground) = {fci_ref['E_CASCI']:.6f} Ha")

    # Sample walker bank from each state
    for k in range(args.K):
        if not bank_paths[k].exists():
            print(f"\n[Step 1.{k+1}] Sampling Psi_{k+1} walker bank")
            rng = jax.random.key(args.seed + k * 100)
            init_key, smp_key = jax.random.split(rng)
            drv = get_vmc_nn_func(
                mol, args.ansatz, init_key,
                prefix=str(chk_paths[k]).replace(".chk.h5", ""),
            )
            drv.load_checkpoint(str(chk_paths[k]))
            drv(
                smp_key,
                num_walkers=args.sample_walkers,
                num_steps_per_block=20,
                num_blocks=args.sample_blocks,
                num_blocks_equil=5,
                mc_timestep=args.mc_timestep,
                compute_gradients=False,
                dump_walkers_path=str(bank_paths[k]),
                verbose=1,
            )

    # CS recovery of each state + energy
    print(f"\n[Step 2] CS recovery + per-state VMC energies")
    c_hats = []
    proj_masses = []
    E_states = []
    for k in range(args.K):
        c_hat, pm = recover_c_hat(
            str(bank_paths[k]), str(chk_paths[k]),
            mol, fci_ref, args.ansatz, args.seed + k * 1000,
        )
        E, sig = estimate_vmc_energy(
            str(bank_paths[k]), str(chk_paths[k]),
            mol, args.ansatz, args.seed + k * 2000,
        )
        c_hats.append(c_hat)
        proj_masses.append(pm)
        E_states.append(E)
        print(f"  State {k+1}: E = {E:.6f} +/- {sig/np.sqrt(256):.5f} Ha, "
              f"proj_mass = {pm:.3e}")

    # Pairwise CI overlaps
    print(f"\n[Step 3] Pairwise CI overlaps (pre-rotation)")
    for i in range(args.K):
        for j in range(i + 1, args.K):
            ovl = float(np.dot(c_hats[i], c_hats[j]))
            print(f"  <c_{i+1} | c_{j+1}> = {ovl:+.4f}")

    # K-state subspace rotation -> eigenstates
    print(f"\n[Step 4] Subspace rotation to extract K={args.K} eigenstates")
    rot = subspace_rotate_to_eigenstates(c_hats, fci_ref, mol)
    c_eig = rot["c_eig"]
    E_eig = rot["E_eig"]
    print(f"  input max-overlap diagnostic = {rot['input_ci_overlap']:.4f}")
    for k in range(args.K):
        print(f"  Eigenstate {k+1}: E = {float(E_eig[k]):.6f} Ha")

    # Transition properties for each (ground, excited_k) pair
    print(f"\n[Step 5] Transition properties from rotated ground to each excited")
    transitions = []
    for k in range(1, args.K):
        delta_E = float(E_eig[k] - E_eig[0])
        rep = report_transition_properties(
            c_eig[0], c_eig[k], fci_ref, mol, delta_E_au=delta_E,
        )
        rep["state_idx"] = k
        print_transition_summary(
            rep, label=f"ground -> eigenstate {k+1}",
        )
        transitions.append(rep)

    # Pick the dipole-allowed excited state with max oscillator strength
    print(f"\n[Step 6] Selected transition (max oscillator strength)")
    target = max(transitions, key=lambda r: r["oscillator_strength"])
    k_target = target["state_idx"]
    print(f"  Eigenstate {k_target + 1} selected, f = "
          f"{target['oscillator_strength']:.6f}, "
          f"dE = {target['delta_E_au'] * 27.2114:.4f} eV")

    # NEVPT2 on (ground, target)
    print(f"\n[Step 7] NEVPT2 on (eigenstate 1, eigenstate {k_target + 1})")
    try:
        nevpt2 = run_multistate_nevpt2(
            mol, c_hats=[c_eig[0], c_eig[k_target]], fci_ref=fci_ref,
            state_labels=["ground", "excited"],
            ncas=args.cas_ncas, nelecas=nelecas,
            occ_threshold=args.nevpt2_occ_threshold,
            nroots_max=8,
        )
        for k, res in enumerate(nevpt2["per_state"]):
            print(f"  {res['label']:>10s}: "
                  f"E_CASCI = {res['e_casci']:.6f}, "
                  f"E_PT2 = {res['e_pt2']:+.6f}, "
                  f"E_total = {res['e_total']:.6f} Ha, "
                  f"match-overlap = {res['max_overlap']:.3f}")
        dE_casci_eV = nevpt2["delta_E_casci_au"][0, 1] * 27.2114
        dE_total_eV = nevpt2["delta_E_total_au"][0, 1] * 27.2114
        print(f"  vertical excitation (CASCI):    {dE_casci_eV:.4f} eV")
        print(f"  vertical excitation (CAS+PT2):  {dE_total_eV:.4f} eV")
    except Exception as exc:
        nevpt2 = None
        import traceback
        print(f"  NEVPT2 step failed: {type(exc).__name__}: {exc}")
        traceback.print_exc()

    # Save summary
    summary = dict(
        molecule="H2O",
        basis=args.basis,
        K=args.K,
        E_CASCI_ground=float(fci_ref["E_CASCI"]),
        E_states_sampled=[float(e) for e in E_states],
        proj_masses=[float(m) for m in proj_masses],
        c_hats=[c.tolist() for c in c_hats],
        c_eig=[c.tolist() for c in c_eig],
        E_eig_rotated=[float(e) for e in E_eig],
        input_ci_overlap_max=float(rot["input_ci_overlap"]),
        target_state_idx=int(k_target),
        target_oscillator=float(target["oscillator_strength"]),
        target_delta_E_au=float(target["delta_E_au"]),
        target_delta_E_eV=float(target["delta_E_au"] * 27.2114),
        target_transition_dipole_debye=target["transition_dipole"]["mu_debye"].tolist(),
        target_nto_weights=target["nto"]["participation_ratios"].tolist(),
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
            delta_E_total_eV=float(
                nevpt2["delta_E_total_au"][0, 1] * 27.2114,
            ),
        )

    json_path = out_dir / f"h2o_pfau_k{args.K}_spec_{basis_tag}.json"
    with open(json_path, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"\n[done] Summary written to {json_path}")


if __name__ == "__main__":
    main()
