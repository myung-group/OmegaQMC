"""YAML driver for cavity-QED VMC on the periodic 2D/3D HEG.

Velocity-gauge Pauli-Fierz, single q=0 cavity mode, complex_psi default.
Runs SR optimization then optionally a fixed-parameter eval block.

Usage:
    python scripts/run_qed_vmc_heg.py inputs/2dheg_qed/<config>.yaml

YAML structure (see inputs/2dheg_qed/smoke_lambda0.yaml):

    project: <run name>
    seed: 0

    system:
        dim: 2
        rs: 10.0
        N: 26
        polarization: unpolarized

    cavity:
        omega: 0.1
        lambda: 0.0
        polarization: [1.0, 0.0]   # in-plane unit vector for 2D
        nph_max: 4

    ansatz:
        embedding_dim: 64
        n_interactions: 2
        two_particle_stream_dim: 16
        n_det: 4
        full_determinant: true
        use_cusp: false
        use_backflow: false

    optimize:
        complex_psi: true
        iters: 50
        walkers: 256
        lr: 0.05
        damping: 1.0e-3
        cg_maxiter: 100
        max_param_change: 0.5
        jac_batch_size: 64
        mc_timestep: 0.1
        equil_blocks: 5
        steps_per_equil_block: 30
        decorr_steps: 5

    eval:
        enabled: true
        walkers: 256
        blocks: 30
        equil_blocks: 5
        steps_per_block: 30
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import yaml

jax.config.update("jax_enable_x64", True)

from OmegaQMC.psi.nn.heg_wf import HEGPsiFormerConfig
from OmegaQMC.qed_vmcopt_nn_heg_sr import get_qed_vmcopt_nn_heg_sr_func
from OmegaQMC.qed_vmc_nn_heg import get_qed_vmc_nn_heg_func


def _build_heg_config(system: dict, ansatz: dict) -> HEGPsiFormerConfig:
    rs = float(system["rs"])
    N = int(system["N"])
    dim = int(system.get("dim", 2))
    pol = str(system.get("polarization", "unpolarized")).lower()
    if pol == "unpolarized":
        n_up = n_down = N // 2
    elif pol == "polarized":
        n_up = N
        n_down = 0
    else:
        raise ValueError(f"polarization must be 'unpolarized' or 'polarized', got {pol!r}")
    if dim == 2:
        L = float(np.sqrt(np.pi * N) * rs)
    elif dim == 3:
        L = float((4.0 * np.pi / 3.0 * N) ** (1.0 / 3.0) * rs)
    else:
        raise ValueError(f"dim must be 2 or 3, got {dim}")
    return HEGPsiFormerConfig(
        n_up=n_up, n_down=n_down, dim=dim, L=L,
        n_det=int(ansatz.get("n_det", 4)),
        full_determinant=bool(ansatz.get("full_determinant", True)),
        embedding_dim=int(ansatz.get("embedding_dim", 64)),
        n_interactions=int(ansatz.get("n_interactions", 2)),
        two_particle_stream_dim=int(ansatz.get("two_particle_stream_dim", 16)),
        use_cusp=bool(ansatz.get("use_cusp", False)),
        use_backflow=bool(ansatz.get("use_backflow", False)),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("yaml_path")
    args = ap.parse_args()

    with open(args.yaml_path, "r") as fh:
        cfg = yaml.safe_load(fh)

    project = cfg.get("project", Path(args.yaml_path).stem)
    seed = int(cfg.get("seed", 0))
    rng = jax.random.PRNGKey(seed)

    system = cfg["system"]
    cavity = cfg["cavity"]
    ansatz = cfg.get("ansatz", {})
    optimize = cfg.get("optimize", {})
    eval_cfg = cfg.get("eval", {})

    heg_config = _build_heg_config(system, ansatz)
    omega = float(cavity["omega"])
    lam = float(cavity["lambda"])
    pol = np.asarray(cavity.get("polarization",
                                 [1.0, 0.0] if heg_config.dim == 2 else [1.0, 0.0, 0.0]),
                     dtype=np.float64)
    pol = pol / max(np.linalg.norm(pol), 1e-30)
    coupling_vec = jnp.asarray(lam * pol)
    nph_max = int(cavity.get("nph_max", 4))
    complex_psi = bool(optimize.get("complex_psi", True))

    print("=" * 70, flush=True)
    print(f"Cavity-QED HEG VMC: project={project}", flush=True)
    print(f"  System: dim={heg_config.dim}  rs={system['rs']}  N={system['N']}  "
          f"L={heg_config.L:.3f} Bohr  n_up={heg_config.n_up}  n_down={heg_config.n_down}",
          flush=True)
    print(f"  Cavity: ω={omega} Ha  λ={lam}  ε={pol.tolist()}  nph_max={nph_max}",
          flush=True)
    print(f"  Trial:  complex_psi={complex_psi}  emb={heg_config.embedding_dim}  "
          f"layers={heg_config.n_interactions}  n_det={heg_config.n_det}",
          flush=True)
    print("=" * 70, flush=True)

    run_dir = Path("runs") / project
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.yaml", "w") as fh:
        yaml.safe_dump(cfg, fh, sort_keys=False)

    # ---- SR training ----
    rng, key_opt = jax.random.split(rng)
    iters = int(optimize.get("iters", 50))
    if iters > 0:
        opt = get_qed_vmcopt_nn_heg_sr_func(
            heg_config, init_key=key_opt,
            omega=omega, coupling_vec=coupling_vec,
            nph_max=nph_max, complex_psi=complex_psi,
        )
        rng, key_run = jax.random.split(rng)
        print("\n[1/2] SR-VMC training", flush=True)
        params_final, history = opt(
            key_run,
            num_iters=iters,
            num_walkers=int(optimize.get("walkers", 256)),
            num_steps_per_block=int(optimize.get("steps_per_equil_block", 30)),
            num_blocks_equil=int(optimize.get("equil_blocks", 5)),
            num_steps_decorr=int(optimize.get("decorr_steps", 5)),
            mc_timestep=float(optimize.get("mc_timestep", 0.1)),
            lr=float(optimize.get("lr", 0.05)),
            damping=float(optimize.get("damping", 1e-3)),
            cg_maxiter=int(optimize.get("cg_maxiter", 100)),
            max_param_change=float(optimize.get("max_param_change", 0.5)),
            jac_batch_size=int(optimize.get("jac_batch_size", 64)),
            verbose=int(optimize.get("verbose", 1)),
        )
        np.savez(
            run_dir / "history.npz",
            energies=np.asarray(history["energies"]),
            energy_serrs=np.asarray(history["energy_serrs"]),
            n_photon=np.asarray(history["n_photon"]),
            param_change_max=np.asarray(history["param_change_max"]),
        )
    else:
        # No-train path (eval only): build driver directly.
        params_final = None
        opt = None

    # ---- Eval ----
    summary = {"project": project, "omega": omega, "lambda": lam}
    if eval_cfg.get("enabled", True):
        rng, key_eval_init = jax.random.split(rng)
        drv_eval = get_qed_vmc_nn_heg_func(
            heg_config, init_key=key_eval_init,
            omega=omega, coupling_vec=coupling_vec,
            nph_max=nph_max, complex_psi=complex_psi,
        )
        if params_final is not None:
            drv_eval.params = params_final
        rng, key_eval = jax.random.split(rng)
        print("\n[2/2] Eval block", flush=True)
        eval_res = drv_eval(
            key_eval,
            num_walkers=int(eval_cfg.get("walkers", 256)),
            num_steps_per_block=int(eval_cfg.get("steps_per_block", 30)),
            num_blocks=int(eval_cfg.get("blocks", 30)),
            num_blocks_equil=int(eval_cfg.get("equil_blocks", 5)),
            mc_timestep=float(eval_cfg.get("mc_timestep",
                                            optimize.get("mc_timestep", 0.1))),
            verbose=int(eval_cfg.get("verbose", 1)),
        )
        summary.update({
            "E_mean": eval_res["E_mean"],
            "E_serr": eval_res["E_serr"],
            "E_per_elec": eval_res["E_per_elec"],
            "n_photon_mean": eval_res["n_photon_mean"],
            "acceptance_r": eval_res["acceptance_r"],
            "acceptance_n": eval_res["acceptance_n"],
        })
        print(f"\n  E/N      = {eval_res['E_per_elec']:+.6f} Ha "
              f"(serr {eval_res['E_serr'] / heg_config.n_up / 2:.6f})", flush=True)
        print(f"  <n_phot> = {eval_res['n_photon_mean']:.4f}", flush=True)
        print(f"  acc r/n  = {eval_res['acceptance_r']:.2f} / "
              f"{eval_res['acceptance_n']:.2f}", flush=True)

    with open(run_dir / "summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"\n[done] artifacts in {run_dir}", flush=True)


if __name__ == "__main__":
    main()
