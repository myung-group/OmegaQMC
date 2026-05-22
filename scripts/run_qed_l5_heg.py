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
    cell_shape = str(_get(cfg, "system.cell_shape", "square"))
    N = n_up + n_down
    L_y = None
    if dim == 2:
        if cell_shape == "rectangular_triangular":
            # Natural centered-rectangular cell for N=2·M² triangular
            # WC: L_x = M·a, L_y = M·a·√3.  Total area = π·rs²·N.
            M = int(round(math.sqrt(N / 2)))
            if 2 * M * M != N:
                raise ValueError(
                    f"system.cell_shape=rectangular_triangular requires "
                    f"N=2·M² in {{2, 8, 18, 32, 50, 72, 98, ...}}; "
                    f"got N={N}",
                )
            n_density = 1.0 / (math.pi * rs ** 2)
            a = math.sqrt(2.0 / (math.sqrt(3.0) * n_density))
            L = M * a
            L_y = M * a * math.sqrt(3.0)
        else:
            # Square cell: cell area = π·rs²·N
            L = rs * math.sqrt(math.pi * N)
    else:
        # 3D: (4π/3)·rs³·N = L³
        L = (4.0 / 3.0 * math.pi * (rs ** 3) * N) ** (1.0 / 3.0)

    # ---- Cavity ----
    omega = float(_get(cfg, "cavity.omega", 0.1))
    coupling_lambda = float(_get(cfg, "cavity.lambda", 0.0))
    coupling_polarization = _get(cfg, "cavity.polarization", None)
    coupling_op = str(_get(cfg, "cavity.coupling_op", "P"))
    # Weber-style external cosine potential for TI breaking.
    v_ext_amp = float(_get(cfg, "v_ext.amp", 0.0))
    v_ext_a = _get(cfg, "v_ext.a", None)
    include_vee = bool(_get(cfg, "system.include_vee", True))
    K_max = int(_get(cfg, "cavity.K_max", 5))
    phase_mlp_hidden = tuple(
        _get(cfg, "cavity.phase_mlp_hidden", [64, 64])
    )
    mag_mlp_hidden = tuple(
        _get(cfg, "cavity.mag_mlp_hidden", [64, 64])
    )
    activation = str(_get(cfg, "cavity.activation", "tanh"))
    # L6: matter-dependent photon shift and width (default off → L5)
    use_matter_photon_shift = bool(
        _get(cfg, "cavity.use_matter_photon_shift", False)
    )
    use_matter_photon_width = bool(
        _get(cfg, "cavity.use_matter_photon_width", False)
    )
    # L7: matter-dependent photon momentum shift P₀(R)·q_c.  For
    # coupling_op="P" this is the Lang-Firsov mean-field channel.
    use_matter_photon_pshift = bool(
        _get(cfg, "cavity.use_matter_photon_pshift", False)
    )
    q0_mlp_hidden = tuple(
        _get(cfg, "cavity.q0_mlp_hidden", [32, 32])
    )
    s_mlp_hidden = tuple(
        _get(cfg, "cavity.s_mlp_hidden", [32, 32])
    )
    p0_mlp_hidden = tuple(
        _get(cfg, "cavity.p0_mlp_hidden", [32, 32])
    )

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
    use_deep_jastrow = bool(_get(cfg, "ansatz.use_deep_jastrow", False))
    use_pair_jastrow = bool(_get(cfg, "ansatz.use_pair_jastrow", False))
    envelope_type = str(_get(cfg, "ansatz.envelope_type", "plane_wave"))
    crystal_sigma_init = float(
        _get(cfg, "ansatz.crystal_sigma_init", 0.25)
    )
    crystal_spin_pattern = str(
        _get(cfg, "ansatz.crystal_spin_pattern", "neel")
    )
    crystal_det_jitter = float(
        _get(cfg, "ansatz.crystal_det_jitter", 0.0)
    )
    crystal_lattice_type = str(
        _get(cfg, "ansatz.crystal_lattice_type", "triangular")
    )
    crystal_anisotropic_sigma = bool(
        _get(cfg, "ansatz.crystal_anisotropic_sigma", False)
    )
    crystal_site_offset = float(
        _get(cfg, "ansatz.crystal_site_offset", 0.5)
    )

    config_ansatz = HEGPsiFormerConfig(
        n_up=n_up, n_down=n_down, L=L, L_y=L_y, dim=dim,
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
        use_pair_jastrow=use_pair_jastrow,
        envelope_type=envelope_type,
        crystal_sigma_init=crystal_sigma_init,
        crystal_spin_pattern=crystal_spin_pattern,
        crystal_det_jitter=crystal_det_jitter,
        crystal_lattice_type=crystal_lattice_type,
        crystal_anisotropic_sigma=crystal_anisotropic_sigma,
        crystal_site_offset=crystal_site_offset,
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
    save_every = int(_get(cfg, "optimize.save_every", 0))

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
    if L_y is not None:
        print(f"  Cell rectangular  L_x={L:.4f}, L_y={L_y:.4f} Bohr  "
              f"(aspect {L_y/L:.4f})")
    else:
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
        coupling_op=coupling_op,
        v_ext_amp=v_ext_amp,
        v_ext_a=v_ext_a,
        include_vee=include_vee,
        K_max=K_max,
        phase_mlp_hidden=phase_mlp_hidden,
        mag_mlp_hidden=mag_mlp_hidden,
        activation=activation,
        use_matter_photon_shift=use_matter_photon_shift,
        use_matter_photon_width=use_matter_photon_width,
        use_matter_photon_pshift=use_matter_photon_pshift,
        q0_mlp_hidden=q0_mlp_hidden,
        s_mlp_hidden=s_mlp_hidden,
        p0_mlp_hidden=p0_mlp_hidden,
    )
    print(f"  built in {fmt_time(time.time() - t0)}")
    n_q0 = opt.l5.get("n_q0_mlp", 0)
    n_s = opt.l5.get("n_s_mlp", 0)
    n_p0 = opt.l5.get("n_p0_mlp", 0)
    l67_str = ""
    if n_q0 > 0 or n_s > 0 or n_p0 > 0:
        l67_str = f", q0_mlp={n_q0}, s_mlp={n_s}, p0_mlp={n_p0}"
    print(f"  Params: total={opt.n_params}  "
          f"(elec={opt.l5['n_electronic']}, "
          f"mag_mlp={opt.l5['n_mag_mlp']}, "
          f"phase_mlp={opt.l5['n_phase_mlp']}{l67_str}, "
          f"n_K={opt.l5['n_K']})")
    print()

    # ---- Optional: resume from a saved checkpoint ----
    load_chkpt = _get(cfg, "optimize.load_chkpt", None)
    skip_training = bool(_get(cfg, "optimize.skip_training", False))
    chkpt_walkers = None
    if load_chkpt:
        chkpt_path = Path(load_chkpt)
        if not chkpt_path.is_absolute():
            chkpt_path = Path.cwd() / chkpt_path
        if not chkpt_path.is_file():
            raise FileNotFoundError(
                f"optimize.load_chkpt: {chkpt_path} not found"
            )
        print(f"[resume] Loading params from {chkpt_path}")
        data = np.load(chkpt_path)
        opt.params_flat = jnp.asarray(data["params_flat"])
        chkpt_walkers = {
            "R": jnp.asarray(data["R"]),
            "q_c": jnp.asarray(data["q_c"]),
            "R_step_size": jnp.asarray(data["R_step_size"]),
            "qc_step_size": jnp.asarray(data["qc_step_size"]),
        }
        e_chkpt = float(data["E_final_ha"]) if "E_final_ha" in data.files else 0.0
        n_chkpt = int(data["n_iters_trained"]) if "n_iters_trained" in data.files else 0
        print(f"  source: epoch={n_chkpt}, energy={e_chkpt:+.6f} Ha")

    # ---- Training ----
    if skip_training:
        print("[2/3] Training skipped (optimize.skip_training=true)")
        train_result = {
            "E_final_ha": e_chkpt if load_chkpt else None,
            "final_walkers": chkpt_walkers,
        }
        train_time = 0.0
        e_final = train_result["E_final_ha"]
    else:
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
            save_every=save_every,
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
        init_walkers=train_result.get("final_walkers"),  # warm start
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
    lz_per = eval_result.get("Lz_per_e", 0.0)
    lz_per_serr = eval_result.get("Lz_per_e_serr", 0.0)
    lz_sq_per = eval_result.get("Lz_sq_per_e", 0.0)
    lz_sq_per_serr = eval_result.get("Lz_sq_per_e_serr", 0.0)
    expected_qc_sq_HO = 1.0 / (2.0 * omega)
    print(f"  E_QED / N            = {e_per:+.6e} ± {e_serr_per:.2e} Ha")
    print(f"                       = {e_per * 2:+.6e} ± "
          f"{e_serr_per * 2:.2e} Ry")
    print(f"  ⟨Im E_loc⟩ / N       = {im_per:+.4e} ± {im_serr:.2e} Ha   "
          f"(Hermiticity check)")
    print(f"  ⟨q_c²⟩               = {qc_sq:.4e}  "
          f"(HO ground state ≈ {expected_qc_sq_HO:.4e})")
    print(f"  ⟨L_z⟩ / N            = {lz_per:+.4e} ± {lz_per_serr:.2e}   "
          f"(chirality order parameter)")
    print(f"  ⟨L_z²⟩ / N           = {lz_sq_per:+.4e} ± {lz_sq_per_serr:.2e}   "
          f"(angular-momentum variance)")
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
            "coupling_op": coupling_op,
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
            "Lz_per_e": lz_per,
            "Lz_per_e_serr": lz_per_serr,
            "Lz_sq_per_e": lz_sq_per,
            "Lz_sq_per_e_serr": lz_sq_per_serr,
        },
        "wall_time_s": {
            "training": train_time, "eval": eval_time,
        },
    }

    # ---- Observables (S(k), optional) ----
    obs_cfg = cfg.get("observables") or {}
    if obs_cfg.get("enabled", False):
        print()
        print("[obs] Accumulating S(k) on L5 |Ψ|² walkers ...")
        from OmegaQMC.observables.structure_factor import (
            reciprocal_grid_2d,
            reciprocal_lattice_vectors_triangular,
            structure_factor,
        )
        # k-grids
        k_grids = {}
        if obs_cfg.get("triangular_shells", 0) > 0:
            n_shell = int(obs_cfg["triangular_shells"])
            kt = reciprocal_lattice_vectors_triangular(
                rs=rs, n_shell=n_shell,
            )
            k_grids["triangular_bragg"] = jnp.asarray(kt)
        if obs_cfg.get("cartesian_n_max", None) is not None:
            n_max_k = int(obs_cfg["cartesian_n_max"])
            k_grids["cartesian_grid"] = reciprocal_grid_2d(L, n_max=n_max_k)

        if k_grids:
            n_walkers_obs = int(obs_cfg.get("n_walkers", eval_walkers))
            n_sample_obs = int(obs_cfg.get("sample_steps", 200))
            decorr_obs = int(obs_cfg.get("decorr_steps", 5))

            # Warm-start S(k) MCMC from eval's final walker state
            # (already converged on trained ansatz).
            warm = eval_result.get("final_walkers")
            if warm is None:
                warm = train_result.get("final_walkers")
            R_obs = jnp.asarray(warm["R"])
            q_c_obs = jnp.asarray(warm["q_c"])
            step_R_obs = jnp.asarray(
                warm["R_step_size"], dtype=jnp.float64,
            )
            step_q_obs = jnp.asarray(
                warm["qc_step_size"], dtype=jnp.float64,
            )
            # Tile/truncate walker count if needed
            if R_obs.shape[0] != n_walkers_obs:
                if R_obs.shape[0] < n_walkers_obs:
                    reps = (
                        (n_walkers_obs + R_obs.shape[0] - 1)
                        // R_obs.shape[0]
                    )
                    R_obs = jnp.tile(
                        R_obs, (reps,) + (1,) * (R_obs.ndim - 1),
                    )[:n_walkers_obs]
                    q_c_obs = jnp.tile(q_c_obs, (reps,))[:n_walkers_obs]
                else:
                    R_obs = R_obs[:n_walkers_obs]
                    q_c_obs = q_c_obs[:n_walkers_obs]

            # MCMC-only step (no eloc/Ewald — halves per-step cost
            # vs the fused eval step).
            mcmc_step = opt._build_mcmc_only_step(n_walkers_obs)
            params_flat_obs = jnp.asarray(opt.params_flat)
            rng_obs = jax.random.fold_in(eval_key, 999)
            carry_obs = (
                rng_obs, R_obs, q_c_obs,
                step_R_obs, step_q_obs, params_flat_obs,
            )

            sk_eval_fns = {
                name: jax.jit(jax.vmap(
                    lambda r, kg=kg: structure_factor(r, kg),
                ))
                for name, kg in k_grids.items()
            }
            sk_blocks = {name: [] for name in k_grids}
            # 2D density-map accumulation (configurable bin count)
            do_density = bool(obs_cfg.get("density_2d", True))
            n_bins = int(obs_cfg.get("density_n_bins", 64))
            if do_density and dim == 2:
                density_hist = np.zeros((n_bins, n_bins), dtype=np.float64)
                density_n_samples = 0
            sk_t0 = time.time()
            for _ in range(n_sample_obs):
                # decorr_obs MCMC steps then a measurement
                carry_obs, _ = jax.lax.scan(
                    mcmc_step, carry_obs, None, length=decorr_obs,
                )
                R_obs = carry_obs[1]
                for name in k_grids:
                    sk_per_walker = sk_eval_fns[name](R_obs)
                    sk_blocks[name].append(
                        np.asarray(jnp.mean(sk_per_walker, axis=0))
                    )
                if do_density and dim == 2:
                    L_x_bin = L
                    L_y_bin = L_y if L_y is not None else L
                    pts = np.asarray(R_obs).reshape(-1, 2)
                    pts = np.mod(pts, np.asarray([L_x_bin, L_y_bin]))
                    h2d, _, _ = np.histogram2d(
                        pts[:, 0], pts[:, 1],
                        bins=n_bins,
                        range=[[0.0, L_x_bin], [0.0, L_y_bin]],
                    )
                    density_hist += h2d
                    density_n_samples += pts.shape[0]
            sk_time = time.time() - sk_t0
            print(f"  S(k) MCMC + measure time: {fmt_time(sk_time)}")

            # ---- 2D density map plot ----
            if do_density and dim == 2 and density_n_samples > 0:
                L_x_bin = L
                L_y_bin = L_y if L_y is not None else L
                bin_area = (L_x_bin / n_bins) * (L_y_bin / n_bins)
                # ⟨ρ(r)⟩ in units of electrons/Bohr² (avg over the
                # n_sample_obs × n_walkers_obs walker configurations).
                n_configs = n_sample_obs * n_walkers_obs
                avg_count_per_bin = density_hist / n_configs
                rho = avg_count_per_bin / bin_area
                rho_avg = float(N) / (L_x_bin * L_y_bin)   # mean density
                try:
                    import matplotlib
                    matplotlib.use("Agg")
                    import matplotlib.pyplot as plt
                    fig, ax = plt.subplots(figsize=(6, 6))
                    im = ax.imshow(
                        rho.T / rho_avg,
                        origin="lower",
                        extent=[0.0, L_x_bin, 0.0, L_y_bin],
                        cmap="viridis",
                        aspect="equal",
                    )
                    ax.set_xlabel("x (Bohr)")
                    ax.set_ylabel("y (Bohr)")
                    ax.set_title(
                        f"⟨ρ(r)⟩/ρ_avg — N={N}, rs={rs}, "
                        f"λ={coupling_lambda}"
                    )
                    plt.colorbar(im, ax=ax, label="ρ(r) / ρ_avg")
                    plt.tight_layout()
                    out_png = run_dir / "density_2d.png"
                    plt.savefig(out_png, dpi=150)
                    plt.close(fig)
                    print(f"  density map saved: {out_png}")
                except ImportError:
                    print(
                        "  (matplotlib not available — skipping density plot)"
                    )
                # Save raw histogram as npz for later re-plotting
                np.savez(
                    run_dir / "density_2d.npz",
                    rho_over_rho_avg=rho / rho_avg,
                    n_bins=n_bins,
                    L=L_x_bin,
                    L_y=L_y_bin,
                    n_samples=n_sample_obs,
                )

            summary["observables"] = {}
            # Append S(k) results to train.log too (single log per run).
            log_path = run_dir / "train.log"
            with open(log_path, "a") as flog:
                flog.write(
                    f"\n# === S(k) accumulation ({n_sample_obs} samples × "
                    f"{decorr_obs} decorr × {n_walkers_obs} walkers, "
                    f"{fmt_time(sk_time)}) ===\n"
                )
                for name, blocks in sk_blocks.items():
                    arr = np.stack(blocks)
                    mean_sk = np.mean(arr, axis=0)
                    serr_sk = np.std(arr, axis=0) / np.sqrt(arr.shape[0])
                    summary["observables"][f"sk_{name}"] = {
                        "k_vectors": np.asarray(k_grids[name]).tolist(),
                        "mean": mean_sk.tolist(),
                        "serr": serr_sk.tolist(),
                    }
                    # Print top peaks (stdout + train.log)
                    idx = np.argsort(-mean_sk)[:6]
                    header = (
                        f"  [{name}] top S(k) peaks (mean +- serr):"
                    )
                    print(header)
                    flog.write(header + "\n")
                    for i in idx:
                        k = np.asarray(k_grids[name])[i]
                        line = (
                            f"    k=({k[0]:+.4f}, {k[1]:+.4f})  "
                            f"|k|={np.linalg.norm(k):.4f}  "
                            f"S(k) = {mean_sk[i]:.3f} +- {serr_sk[i]:.3f}"
                        )
                        print(line)
                        flog.write(line + "\n")

    with open(run_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[write] {run_dir}/summary.json")


if __name__ == "__main__":
    main()
