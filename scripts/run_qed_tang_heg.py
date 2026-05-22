"""Runner for Level 8 V2 (Tang-architecture) cavity-QED HEG.

Loads a YAML config, builds the L8 V2 Tang optimizer, runs SR
training, then a non-gradient evaluation.  Mirror of
scripts/run_qed_fock_heg.py with the Tang trial.  N_max and
phase_mlp_hidden are the L8-V2-specific cavity fields.
"""
import json, math, sys, time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import yaml

jax.config.update("jax_enable_x64", True)

from OmegaQMC.psi.nn.heg_wf import HEGPsiFormerConfig
from OmegaQMC.qed_vmcopt_nn_heg_tang import _QEDTangOptimizer


def _get(cfg, key, default=None):
    parts = key.split(".")
    d = cfg
    for p in parts:
        if not isinstance(d, dict) or p not in d:
            return default
        d = d[p]
    return d


def fmt_time(s):
    if s < 60: return f"{s:.1f}s"
    m, s = divmod(s, 60)
    if m < 60: return f"{int(m)}m {s:.0f}s"
    h, m = divmod(m, 60)
    return f"{int(h)}h {int(m)}m"


def main():
    if len(sys.argv) != 2:
        print("Usage: run_qed_tang_heg.py <config.yaml>"); sys.exit(1)
    cfg_path = sys.argv[1]
    with open(cfg_path) as f: cfg = yaml.safe_load(f)

    project = cfg.get("project", "l8v2_run")
    run_dir = Path("runs") / project
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.yaml", "w") as f:
        yaml.dump(cfg, f, sort_keys=False)

    # System
    n_up = int(_get(cfg, "system.n_up", 9))
    n_down = int(_get(cfg, "system.n_down", 9))
    rs = float(_get(cfg, "system.rs", 1.5958))
    dim = int(_get(cfg, "system.dim", 2))
    include_vee = bool(_get(cfg, "system.include_vee", True))
    N = n_up + n_down
    L = rs * math.sqrt(math.pi * N)
    L_y_arg = _get(cfg, "system.L_y", None)

    # v_ext
    v_ext_amp = float(_get(cfg, "v_ext.amp", 0.0))
    v_ext_a = _get(cfg, "v_ext.a", None)
    if v_ext_a is not None: v_ext_a = float(v_ext_a)

    # Cavity + L8 V2 architecture
    omega = float(_get(cfg, "cavity.omega", 0.1))
    lam = float(_get(cfg, "cavity.lambda", 0.0))
    polarization = _get(cfg, "cavity.polarization", [1.0, 0.0])
    coupling_op = str(_get(cfg, "cavity.coupling_op", "P"))
    N_max = int(_get(cfg, "cavity.N_max", 4))
    phase_mlp_hidden = tuple(_get(cfg, "cavity.phase_mlp_hidden", [64, 64]))
    offset_floor = float(_get(cfg, "cavity.offset_floor", -50.0))

    # Ansatz (V11-style)
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
        use_backflow=use_backflow, use_cusp=use_cusp,
        n_virt_pw=n_virt_pw, use_ghost_atom=use_ghost_atom,
        use_deep_jastrow=use_deep_jastrow,
        use_smith_deep_jastrow=use_smith_deep_jastrow,
        envelope_type=envelope_type,
    )

    # Optimize
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
    load_chkpt = _get(cfg, "optimize.load_chkpt", None)
    kick_phase_on_warmstart = bool(
        _get(cfg, "optimize.kick_phase_on_warmstart", False)
    )

    # Ewald + eval
    ewald_n_real = int(_get(cfg, "ewald.n_real", 3))
    ewald_n_recip = int(_get(cfg, "ewald.n_recip", 6))
    eval_walkers = int(_get(cfg, "eval.walkers", 1024))
    eval_blocks = int(_get(cfg, "eval.blocks", 50))
    eval_equil_blocks = int(_get(cfg, "eval.equil_blocks", 5))
    eval_steps_per_block = int(_get(cfg, "eval.steps_per_block", 10))

    seed = int(_get(cfg, "seed", abs(hash(project)) & 0xFFFFFFFF))
    init_key, train_key, eval_key = jax.random.split(jax.random.key(seed), 3)

    print("=" * 70)
    print(f"L8 V2 Tang cavity-QED HEG: project={project}")
    print(f"  rs={rs}  N={N}  L={L:.4f} Bohr  dim={dim}")
    print(f"  Ω={omega}  λ={lam}  Ω_eff={math.sqrt(omega**2+N*lam**2):.4f}")
    print(f"  N_max={N_max}  phase_mlp_hidden={phase_mlp_hidden}")
    print(f"  Run dir: {run_dir}  Seed: {seed}")
    print("=" * 70 + "\n")

    print("[1/3] Building Tang optimizer ...")
    t0 = time.time()
    chkpt_path = run_dir / f"{project}.chk.npz"
    opt = _QEDTangOptimizer(
        config_ansatz, init_key,
        lr=lr, damping=damping, n_cg=n_cg,
        ewald_n_real=ewald_n_real, ewald_n_recip=ewald_n_recip,
        ofname_chkpt=str(chkpt_path),
        lr_schedule=lr_schedule, lr_min=lr_min, lr_T_max=lr_T_max,
        spring_mu=spring_mu, spring_norm_clip=spring_norm_clip,
        omega=omega, coupling_lambda=lam,
        coupling_polarization=polarization, coupling_op=coupling_op,
        v_ext_amp=v_ext_amp, v_ext_a=v_ext_a,
        include_vee=include_vee,
        N_max=N_max, phase_mlp_hidden=phase_mlp_hidden,
        offset_floor=offset_floor,
    )
    print(f"  built in {fmt_time(time.time() - t0)}")
    print(f"  Params: total={opt.n_params}, N_max={opt.N_max}")

    if load_chkpt is not None:
        chkpt_p = Path(load_chkpt)
        if not chkpt_p.exists():
            raise FileNotFoundError(f"optimize.load_chkpt not found: {chkpt_p}")
        d = np.load(chkpt_p, allow_pickle=True)
        p_loaded = jnp.asarray(d["params_flat"])
        if p_loaded.shape[0] != opt.n_params:
            raise ValueError(
                f"chkpt n_params {p_loaded.shape[0]} != "
                f"reconstructed {opt.n_params}"
            )
        if kick_phase_on_warmstart:
            # Override phase_mlp last-layer kernel with the fresh
            # Xavier-init values, leaving all other params from chkpt.
            # Designed to escape the vacuum local-min seen in earlier
            # runs where phase_mlp ended up at ~zero.
            from jax.flatten_util import ravel_pytree
            loaded_pytree = opt.unravel(p_loaded)
            init_pytree = opt.tang["init_params_pytree"]
            # NNX state doesn't support negative indexing — use explicit
            # last-layer index (= number of hidden layers).
            last_idx = len(phase_mlp_hidden)
            old_last = loaded_pytree["phase_mlp"]["layers"][last_idx]["kernel"]
            new_last = init_pytree["phase_mlp"]["layers"][last_idx]["kernel"]
            loaded_pytree["phase_mlp"]["layers"][last_idx]["kernel"] = new_last
            p_loaded, _ = ravel_pytree(loaded_pytree)
            print(f"  [phase-kick] phase_mlp layer[{last_idx}].kernel overridden: "
                  f"chkpt |W|_max={float(jnp.abs(old_last).max()):.3e} → "
                  f"fresh |W|_max={float(jnp.abs(new_last).max()):.3e}")
        opt.params_flat = p_loaded
        e_loaded = float(d["E_final_ha"]) if "E_final_ha" in d.files else None
        e_str = f"E/N={e_loaded*1000:+.4f} mHa/e" if e_loaded is not None else "?"
        print(f"  [warmstart] loaded {p_loaded.shape[0]} params from {chkpt_p}  "
              f"(epoch={int(d.get('n_iters_trained', 0))}, {e_str})")
    print()

    if iters > 0:
        mode = "fused JIT" if use_fused_step else "Python-orchestrated"
        print(f"[2/3] Training: {iters} iters × {walkers} walkers ({mode})")
        t0 = time.time()
        log_path = run_dir / "train.log"
        train_method = opt.train_fused if use_fused_step else opt.train
        with open(log_path, "w") as log_file:
            params_flat, R_walkers = train_method(
                train_key,
                num_walkers=walkers, n_iters=iters,
                mcmc_decorr_steps=mcmc_decorr_steps,
                mc_timestep_R=mc_timestep_R,
                equil_steps=equil_steps,
                save_every=save_every,
                verbose=1,
                chkpt_path=str(chkpt_path),
                log_file=log_file,
            )
        print(f"  training time: {fmt_time(time.time() - t0)}\n")
    else:
        print("[2/3] Training skipped (iters=0)")
        params_flat = opt.params_flat

    if eval_blocks > 0:
        print(f"[3/3] Eval: {eval_blocks} blocks × {eval_steps_per_block} × "
              f"{eval_walkers} walkers")
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
        print(f"  eval time: {fmt_time(time.time() - t0)}")
        print("\n" + "=" * 70 + "\nSUMMARY\n" + "=" * 70)
        print(f"  E / N        = {result['E_per_e_ha']:+.6e} ± "
              f"{result['E_per_e_sem']:.2e} Ha")
        print(f"  Im / N       = {result['Im_per_e_ha']:+.4e} Ha")
        print("=" * 70)
    else:
        result = None
        print("[3/3] Eval skipped")

    summary = {
        "project": project, "seed": seed,
        "system": {"n_up": n_up, "n_down": n_down, "rs": rs, "N": N,
                   "L": L, "dim": dim, "include_vee": include_vee},
        "cavity": {"omega": omega, "lambda": lam,
                   "polarization": list(polarization),
                   "coupling_op": coupling_op, "N_max": N_max,
                   "phase_mlp_hidden": list(phase_mlp_hidden)},
        "params": {"total": opt.n_params, "N_max": opt.N_max},
        "result": result,
    }
    with open(run_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(f"[write] {run_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
