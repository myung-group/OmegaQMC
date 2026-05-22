"""Minimal runner for Level 8 cavity-QED HEG (Fock-basis photon).

Loads a YAML config, builds the L8 Fock-basis optimizer, runs SR
training, then a non-gradient evaluation.  Single-device only.

Pattern mirrors scripts/run_qed_l5_heg.py — see that file for the L5/L7
counterpart.  L8-specific YAML fields live under cavity.N_max and have
no q_c MCMC step.
"""
import json
import math
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import yaml

jax.config.update("jax_enable_x64", True)

from OmegaQMC.psi.nn.heg_wf import HEGPsiFormerConfig
from OmegaQMC.qed_vmcopt_nn_heg_fock import _QEDFockOptimizer


def _get(cfg, key, default=None):
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
        print("Usage: run_qed_fock_heg.py <config.yaml>")
        sys.exit(1)

    cfg_path = sys.argv[1]
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    project = cfg.get("project", "l8_run")
    run_dir = Path("runs") / project
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.yaml", "w") as f:
        yaml.dump(cfg, f, sort_keys=False)

    # ---- System ----
    n_up = int(_get(cfg, "system.n_up", 9))
    n_down = int(_get(cfg, "system.n_down", 9))
    rs = float(_get(cfg, "system.rs", 1.5958))
    dim = int(_get(cfg, "system.dim", 2))
    include_vee = bool(_get(cfg, "system.include_vee", True))
    N = n_up + n_down
    L = rs * math.sqrt(math.pi * N)
    L_y_arg = _get(cfg, "system.L_y", None)

    # ---- v_ext ----
    v_ext_amp = float(_get(cfg, "v_ext.amp", 0.0))
    v_ext_a = _get(cfg, "v_ext.a", None)
    if v_ext_a is not None:
        v_ext_a = float(v_ext_a)

    # ---- Cavity + L8 architecture ----
    omega = float(_get(cfg, "cavity.omega", 0.1))
    lam = float(_get(cfg, "cavity.lambda", 0.0))
    polarization = _get(cfg, "cavity.polarization", [1.0, 0.0])
    coupling_op = str(_get(cfg, "cavity.coupling_op", "P"))
    N_max = int(_get(cfg, "cavity.N_max", 6))
    K_max = int(_get(cfg, "cavity.K_max", 5))
    mag_mlp_hidden = tuple(_get(cfg, "cavity.mag_mlp_hidden", [64, 64]))
    phase_mlp_hidden = tuple(_get(cfg, "cavity.phase_mlp_hidden", [64, 64]))
    activation = str(_get(cfg, "cavity.activation", "tanh"))
    offset_floor = float(_get(cfg, "cavity.offset_floor", -50.0))

    # ---- Electronic ansatz ----
    backbone = str(_get(cfg, "ansatz.backbone", "ferminet"))
    embedding_dim = int(_get(cfg, "ansatz.embedding_dim", 64))
    n_interactions = int(_get(cfg, "ansatz.n_interactions", 3))
    two_particle_stream_dim = int(
        _get(cfg, "ansatz.two_particle_stream_dim", 16)
    )
    n_det = int(_get(cfg, "ansatz.n_det", 1))
    full_determinant = bool(_get(cfg, "ansatz.full_determinant", True))
    use_backflow = bool(_get(cfg, "ansatz.use_backflow", True))
    use_cusp = bool(_get(cfg, "ansatz.use_cusp", True))
    use_smith_deep_jastrow = bool(
        _get(cfg, "ansatz.use_smith_deep_jastrow", True)
    )
    n_virt_pw = int(_get(cfg, "ansatz.n_virt_pw", 0))
    use_ghost_atom = bool(_get(cfg, "ansatz.use_ghost_atom", True))
    use_deep_jastrow = bool(_get(cfg, "ansatz.use_deep_jastrow", False))
    envelope_type = str(_get(cfg, "ansatz.envelope_type", "plane_wave"))

    config_ansatz = HEGPsiFormerConfig(
        n_up=n_up, n_down=n_down, L=L, L_y=L_y_arg, dim=dim,
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
        use_deep_jastrow=use_deep_jastrow,
        use_smith_deep_jastrow=use_smith_deep_jastrow,
        envelope_type=envelope_type,
    )

    # ---- Optimization ----
    lr = float(_get(cfg, "optimize.lr", 0.005))
    damping = float(_get(cfg, "optimize.sr_damping", 1e-3))
    n_cg = int(_get(cfg, "optimize.sr_n_cg", 20))
    iters = int(_get(cfg, "optimize.iters", 500))
    walkers = int(_get(cfg, "optimize.walkers", 1024))
    mcmc_decorr_steps = int(_get(cfg, "optimize.mcmc_decorr_steps", 15))
    mc_timestep_R = float(_get(cfg, "optimize.mc_timestep_R", 0.1))
    equil_steps = int(_get(cfg, "optimize.equil_steps", 50))
    lr_schedule = str(_get(cfg, "optimize.lr_schedule", "cosine"))
    lr_min = float(_get(cfg, "optimize.lr_min", 1e-5))
    lr_T_max = _get(cfg, "optimize.lr_T_max", iters)
    spring_mu = float(_get(cfg, "optimize.spring_mu", 0.0))
    spring_norm_clip = float(_get(cfg, "optimize.spring_norm_clip", 1e10))
    save_every = int(_get(cfg, "optimize.save_every", 0))
    use_fused_step = bool(_get(cfg, "optimize.use_fused_step", True))

    # ---- Ewald ----
    ewald_n_real = int(_get(cfg, "ewald.n_real", 3))
    ewald_n_recip = int(_get(cfg, "ewald.n_recip", 6))

    # ---- Eval ----
    eval_walkers = int(_get(cfg, "eval.walkers", 1024))
    eval_blocks = int(_get(cfg, "eval.blocks", 50))
    eval_equil_blocks = int(_get(cfg, "eval.equil_blocks", 5))
    eval_steps_per_block = int(_get(cfg, "eval.steps_per_block", 10))

    # ---- Seed ----
    seed = int(_get(cfg, "seed", abs(hash(project)) & 0xFFFFFFFF))
    init_key, train_key, eval_key = jax.random.split(
        jax.random.key(seed), 3,
    )

    # ---- Banner ----
    print("=" * 70)
    print(f"L8 cavity-QED HEG (Fock basis): project={project}  dim={dim}")
    print(f"  rs={rs}  N={N} (n_up={n_up}, n_down={n_down})")
    print(f"  Cell L={L:.4f} Bohr")
    print(f"  Ω={omega}  λ={lam}  Ω_eff=√(Ω²+Nλ²)={math.sqrt(omega**2+N*lam**2):.4f}")
    print(f"  N_max={N_max}  (Fock truncation)")
    print(f"  K_max={K_max}  mag/phase MLPs={mag_mlp_hidden}")
    print(f"  Run dir: {run_dir}")
    print(f"  Seed: {seed}")
    print("=" * 70)
    print()

    # ---- Build optimizer ----
    print("[1/3] Building Fock optimizer ...")
    t0 = time.time()
    chkpt_path = run_dir / f"{project}.chk.npz"
    opt = _QEDFockOptimizer(
        config_ansatz, init_key,
        lr=lr, damping=damping, n_cg=n_cg,
        ewald_n_real=ewald_n_real,
        ewald_n_recip=ewald_n_recip,
        ofname_chkpt=str(chkpt_path),
        lr_schedule=lr_schedule,
        lr_min=lr_min, lr_T_max=lr_T_max,
        spring_mu=spring_mu,
        spring_norm_clip=spring_norm_clip,
        omega=omega, coupling_lambda=lam,
        coupling_polarization=polarization,
        coupling_op=coupling_op,
        v_ext_amp=v_ext_amp, v_ext_a=v_ext_a,
        include_vee=include_vee,
        N_max=N_max, K_max=K_max,
        mag_mlp_hidden=mag_mlp_hidden,
        phase_mlp_hidden=phase_mlp_hidden,
        activation=activation,
        offset_floor=offset_floor,
    )
    print(f"  built in {fmt_time(time.time() - t0)}")
    print(f"  Params: total={opt.n_params}  "
          f"(elec={opt.fock['n_electronic']}, "
          f"mag_mlp={opt.fock['n_mag_mlp']}, "
          f"phase_mlp={opt.fock['n_phase_mlp']}, "
          f"N_max={opt.N_max}, n_K={opt.fock['n_K']})")
    print()

    # ---- Training ----
    if iters <= 0:
        print("[2/3] Training skipped (iters=0)")
        params_flat = opt.params_flat
        R_walkers = None
    else:
        mode = "fused JIT" if use_fused_step else "Python-orchestrated"
        print(f"[2/3] Training: {iters} iters × {walkers} walkers ({mode})")
        t0 = time.time()
        log_path = run_dir / "train.log"
        train_method = opt.train_fused if use_fused_step else opt.train
        with open(log_path, "w") as log_file:
            params_flat, R_walkers = train_method(
                train_key,
                num_walkers=walkers,
                n_iters=iters,
                mcmc_decorr_steps=mcmc_decorr_steps,
                mc_timestep_R=mc_timestep_R,
                equil_steps=equil_steps,
                save_every=save_every,
                verbose=1,
                chkpt_path=str(chkpt_path),
                log_file=log_file,
            )
        print(f"  training time: {fmt_time(time.time() - t0)}")
    print()

    # ---- Evaluation ----
    if eval_blocks <= 0:
        print("[3/3] Evaluation skipped (eval.blocks=0)")
        result = None
    else:
        print(f"[3/3] Evaluation: {eval_blocks} blocks × "
              f"{eval_steps_per_block} steps × {eval_walkers} walkers")
        t0 = time.time()
        result = opt.evaluate(
            eval_key,
            num_walkers=eval_walkers,
            num_blocks=eval_blocks,
            steps_per_block=eval_steps_per_block,
            equil_blocks=eval_equil_blocks,
            mc_timestep_R=mc_timestep_R,
            params_flat=params_flat,
            verbose=1,
        )
        print(f"  Eval time: {fmt_time(time.time() - t0)}")
        print()
        print("=" * 70)
        print("SUMMARY")
        print("=" * 70)
        print(f"  E / N        = {result['E_per_e_ha']:+.6e} ± "
              f"{result['E_per_e_sem']:.2e} Ha")
        print(f"  Im / N       = {result['Im_per_e_ha']:+.4e} Ha   "
              f"(Hermiticity check)")
        print("=" * 70)

    # ---- Summary JSON ----
    out_summary = {
        "project": project,
        "seed": seed,
        "system": {
            "n_up": n_up, "n_down": n_down, "rs": rs, "N": N,
            "L": L, "dim": dim, "include_vee": include_vee,
        },
        "cavity": {
            "omega": omega, "lambda": lam,
            "polarization": list(polarization),
            "coupling_op": coupling_op, "N_max": N_max,
            "K_max": K_max,
            "mag_mlp_hidden": list(mag_mlp_hidden),
            "phase_mlp_hidden": list(phase_mlp_hidden),
        },
        "params": {
            "total": opt.n_params,
            "electronic": opt.fock["n_electronic"],
            "mag_mlp": opt.fock["n_mag_mlp"],
            "phase_mlp": opt.fock["n_phase_mlp"],
            "n_K": opt.fock["n_K"],
        },
        "result": result,
    }
    with open(run_dir / "summary.json", "w") as f:
        json.dump(out_summary, f, indent=2, default=float)
    print(f"[write] {run_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
