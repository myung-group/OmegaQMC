"""Minimal runner for Level 5 cavity-QED HEG (position-rep photon).

Loads a YAML config, builds the L5 optimizer, runs SR training, then
non-gradient evaluation.  Single-device only (no pmap).
"""
import json
import math
import os
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import yaml

jax.config.update("jax_enable_x64", True)

from OmegaQMC.psi.nn.heg_wf import HEGPsiFormerConfig
from OmegaQMC.qed_vmcopt_nn_heg_l5 import _QEDL5Optimizer


def _get(cfg, key, default=None):
    """Nested dict lookup with dot notation: 'cavity.omega'."""
    parts = key.split(".")
    d = cfg
    for p in parts:
        if not isinstance(d, dict) or p not in d:
            return default
        d = d[p]
    return d


def fmt_time(s):
    if s < 60:
        return f"{s:.1f}s"
    m, s = divmod(s, 60)
    if m < 60:
        return f"{int(m)}m {s:.0f}s"
    h, m = divmod(m, 60)
    return f"{int(h)}h {int(m)}m"


def main():
    if len(sys.argv) != 2:
        print("Usage: run_qed_l5_heg.py <config.yaml>")
        sys.exit(1)

    cfg_path = sys.argv[1]
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    project = cfg.get("project", "l5_run")
    run_dir = Path("runs") / project
    run_dir.mkdir(parents=True, exist_ok=True)

    # Copy config to run dir for reproducibility
    with open(run_dir / "config.yaml", "w") as f:
        yaml.dump(cfg, f, sort_keys=False)

    # ---- System ----
    n_up = int(_get(cfg, "system.n_up"))
    n_down = int(_get(cfg, "system.n_down"))
    rs = float(_get(cfg, "system.rs"))
    dim = int(_get(cfg, "system.dim", 2))
    N = n_up + n_down
    if dim == 2:
        # 2D unpolarised: cell area = π·rs²·N
        L = rs * math.sqrt(math.pi * N)
    else:
        # 3D: (4π/3)·rs³·N = L³
        L = (4.0 / 3.0 * math.pi * (rs ** 3) * N) ** (1.0 / 3.0)

    # ---- Cavity ----
    omega = float(_get(cfg, "cavity.omega", 0.1))
    coupling_lambda = float(_get(cfg, "cavity.lambda", 0.0))
    coupling_polarization = _get(cfg, "cavity.polarization", None)
    K_max = int(_get(cfg, "cavity.K_max", 5))
    phase_mlp_hidden = tuple(
        _get(cfg, "cavity.phase_mlp_hidden", [64, 64])
    )
    mag_mlp_hidden = tuple(
        _get(cfg, "cavity.mag_mlp_hidden", [64, 64])
    )
    activation = str(_get(cfg, "cavity.activation", "tanh"))

    # ---- Ansatz ----
    backbone = str(_get(cfg, "ansatz.backbone", "ferminet"))
    embedding_dim = int(_get(cfg, "ansatz.embedding_dim", 128))
    n_interactions = int(_get(cfg, "ansatz.n_interactions", 3))
    two_particle_stream_dim = int(
        _get(cfg, "ansatz.two_particle_stream_dim", 16)
    )
    n_det = int(_get(cfg, "ansatz.n_det", 8))
    full_determinant = bool(_get(cfg, "ansatz.full_determinant", True))
    use_backflow = bool(_get(cfg, "ansatz.use_backflow", True))
    use_cusp = bool(_get(cfg, "ansatz.use_cusp", True))
    use_smith_deep_jastrow = bool(
        _get(cfg, "ansatz.use_smith_deep_jastrow", True)
    )
    n_virt_pw = int(_get(cfg, "ansatz.n_virt_pw", 0))
    use_ghost_atom = bool(_get(cfg, "ansatz.use_ghost_atom", False))

    config_ansatz = HEGPsiFormerConfig(
        n_up=n_up, n_down=n_down, L=L, dim=dim,
        backbone=backbone,
        embedding_dim=embedding_dim,
        n_interactions=n_interactions,
        two_particle_stream_dim=two_particle_stream_dim,
        n_det=n_det,
        full_determinant=full_determinant,
        use_backflow=use_backflow,
        use_cusp=use_cusp,
        n_virt_pw=n_virt_pw,
        use_ghost_atom=use_ghost_atom,
    )

    # ---- Optimization ----
    lr = float(_get(cfg, "optimize.lr", 0.5))
    damping = float(_get(cfg, "optimize.sr_damping", 1e-3))
    n_cg = int(_get(cfg, "optimize.sr_n_cg", 20))
    iters = int(_get(cfg, "optimize.iters", 500))
    walkers = int(_get(cfg, "optimize.walkers", 1024))
    mcmc_decorr_steps = int(_get(cfg, "optimize.mcmc_decorr_steps", 20))
    mc_timestep_R = float(_get(cfg, "optimize.mc_timestep_R", 0.1))
    mc_timestep_qc = float(_get(cfg, "optimize.mc_timestep_qc", 0.5))
    lr_schedule = str(_get(cfg, "optimize.lr_schedule", "cosine"))
    lr_min = float(_get(cfg, "optimize.lr_min", 1e-5))
    lr_T_max = _get(cfg, "optimize.lr_T_max", iters)
    spring_mu = float(_get(cfg, "optimize.spring_mu", 0.0))
    spring_norm_clip = float(_get(cfg, "optimize.spring_norm_clip", 0.0))
    use_smw_sr = bool(_get(cfg, "optimize.use_smw_sr", True))
    use_fused_step = bool(_get(cfg, "optimize.use_fused_step", True))
    freeze_mlps = bool(_get(cfg, "optimize.freeze_mlps", False))

    # ---- Ewald ----
    ewald_n_real = int(_get(cfg, "ewald.n_real", 3))
    ewald_n_recip = int(_get(cfg, "ewald.n_recip", 6))

    # ---- Eval ----
    eval_walkers = int(_get(cfg, "eval.walkers", 512))
    eval_blocks = int(_get(cfg, "eval.blocks", 50))
    eval_equil_blocks = int(_get(cfg, "eval.equil_blocks", 10))
    eval_steps_per_block = int(_get(cfg, "eval.steps_per_block", 20))

    # ---- Seed ----
    seed = int(_get(cfg, "seed", abs(hash(project)) & 0xFFFFFFFF))
    init_key, train_key, eval_key = jax.random.split(
        jax.random.key(seed), 3,
    )

    # ---- Banner ----
    sep = "=" * 70
    print(sep)
    print(f"L5 cavity-QED HEG run: project={project}  dim={dim}")
    print(f"  rs={rs}  N={N} (n_up={n_up}, n_down={n_down})")
    print(f"  Cell L={L:.4f} Bohr")
    print(f"  Ω={omega}  λ={coupling_lambda}  "
          f"Ω_eff=√(Ω²+Nλ²)={math.sqrt(omega**2 + N*coupling_lambda**2):.4f}")
    print(f"  Run dir: {run_dir}")
    print(f"  Seed: {seed}")
    print(sep)
    print()

    # ---- Build optimizer ----
    print("[1/3] Building optimizer ...")
    t0 = time.time()
    opt = _QEDL5Optimizer(
        config_ansatz, init_key,
        lr=lr, damping=damping, n_cg=n_cg,
        ewald_n_real=ewald_n_real,
        ewald_n_recip=ewald_n_recip,
        ofname_chkpt=str(run_dir / f"{project}.chk.h5"),
        lr_schedule=lr_schedule,
        lr_min=lr_min,
        lr_T_max=lr_T_max,
        spring_mu=spring_mu,
        spring_norm_clip=spring_norm_clip,
        use_smw_sr=use_smw_sr,
        use_fused_step=use_fused_step,
        freeze_mlps=freeze_mlps,
        omega=omega,
        coupling_lambda=coupling_lambda,
        coupling_polarization=coupling_polarization,
        K_max=K_max,
        phase_mlp_hidden=phase_mlp_hidden,
        mag_mlp_hidden=mag_mlp_hidden,
        activation=activation,
    )
    print(f"  built in {fmt_time(time.time() - t0)}")
    print(f"  Params: total={opt.n_params}  "
          f"(elec={opt.l5['n_electronic']}, "
          f"mag_mlp={opt.l5['n_mag_mlp']}, "
          f"phase_mlp={opt.l5['n_phase_mlp']}, "
          f"n_K={opt.l5['n_K']})")
    print()

    # ---- Training ----
    print(f"[2/3] SR-VMC training: {iters} iters × {walkers} walkers")
    t0 = time.time()
    train_result = opt(
        train_key,
        num_iters=iters,
        num_walkers=walkers,
        mcmc_decorr_steps=mcmc_decorr_steps,
        num_equil_steps=400,
        mc_timestep_R=mc_timestep_R,
        mc_timestep_qc=mc_timestep_qc,
        fname_log=str(run_dir / "train.log"),
        verbose=1,
    )
    train_time = time.time() - t0
    e_final = train_result.get("E_final_ha")
    print(f"  Training E/N (last): {e_final:+.8e} Ha")
    print(f"  Training time: {fmt_time(train_time)}")
    print()

    # ---- Evaluation ----
    if eval_blocks <= 0:
        print("[3/3] Evaluation skipped (eval.blocks <= 0)")
        summary = {
            "project": project, "seed": seed,
            "system": {
                "n_up": n_up, "n_down": n_down, "rs": rs, "dim": dim,
                "L": L, "N": N,
            },
            "cavity": {
                "omega": omega, "lambda": coupling_lambda,
                "omega_eff": math.sqrt(omega**2 + N*coupling_lambda**2),
                "K_max": K_max,
            },
            "params": {
                "total": opt.n_params,
                "electronic": opt.l5["n_electronic"],
                "mag_mlp": opt.l5["n_mag_mlp"],
                "phase_mlp": opt.l5["n_phase_mlp"],
                "n_K": opt.l5["n_K"],
            },
            "training": {
                "iters": iters, "walkers": walkers,
                "E_final_ha": e_final, "wall_time_s": train_time,
            },
        }
        with open(run_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        print(f"[write] {run_dir}/summary.json")
        return
    print(f"[3/3] Evaluation: {eval_blocks} blocks × "
          f"{eval_steps_per_block} steps × {eval_walkers} walkers")
    t0 = time.time()
    eval_result = opt.evaluate(
        eval_key,
        num_walkers=eval_walkers,
        num_blocks=eval_blocks,
        num_blocks_equil=eval_equil_blocks,
        num_steps_per_block=eval_steps_per_block,
        mc_timestep_R=mc_timestep_R,
        mc_timestep_qc=mc_timestep_qc,
        fname_log=str(run_dir / "train.log"),   # append to training log
        verbose=1,
    )
    eval_time = time.time() - t0
    print(f"  Eval time: {fmt_time(eval_time)}")
    print()

    # ---- Summary ----
    print(sep)
    print("SUMMARY")
    print(sep)
    e_per = eval_result["E_per_elec_ha"]
    e_serr_per = eval_result["E_serr_per_e_ha"]
    im_per = eval_result["Im_per_e_ha"]
    im_serr = eval_result["Im_serr_per_e_ha"]
    qc_sq = eval_result["qc_sq_mean"]
    expected_qc_sq_HO = 1.0 / (2.0 * omega)
    print(f"  E_QED / N            = {e_per:+.6e} ± {e_serr_per:.2e} Ha")
    print(f"                       = {e_per * 2:+.6e} ± "
          f"{e_serr_per * 2:.2e} Ry")
    print(f"  ⟨Im E_loc⟩ / N       = {im_per:+.4e} ± {im_serr:.2e} Ha   "
          f"(Hermiticity check)")
    print(f"  ⟨q_c²⟩               = {qc_sq:.4e}  "
          f"(HO ground state ≈ {expected_qc_sq_HO:.4e})")
    print(sep)

    summary = {
        "project": project,
        "seed": seed,
        "system": {
            "n_up": n_up, "n_down": n_down, "rs": rs, "dim": dim, "L": L,
            "N": N,
        },
        "cavity": {
            "omega": omega, "lambda": coupling_lambda,
            "omega_eff": math.sqrt(omega**2 + N*coupling_lambda**2),
            "K_max": K_max,
        },
        "params": {
            "total": opt.n_params,
            "electronic": opt.l5["n_electronic"],
            "mag_mlp": opt.l5["n_mag_mlp"],
            "phase_mlp": opt.l5["n_phase_mlp"],
            "n_K": opt.l5["n_K"],
        },
        "result": {
            "E_per_elec_ha": e_per,
            "E_serr_per_e_ha": e_serr_per,
            "Im_per_e_ha": im_per,
            "Im_serr_per_e_ha": im_serr,
            "qc_sq_mean": qc_sq,
            "qc_sq_HO_expected": expected_qc_sq_HO,
        },
        "wall_time_s": {
            "training": train_time, "eval": eval_time,
        },
    }
    with open(run_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[write] {run_dir}/summary.json")


if __name__ == "__main__":
    main()
