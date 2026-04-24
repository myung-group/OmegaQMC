"""HEG VMC driver — YAML-configured, self-contained per-project runs.

Usage:
    python scripts/run_heg_psiformer.py input.yaml

The input file fully specifies the run.  Given ``project: <name>`` in the
YAML, the script creates ``runs/<name>/`` (relative to CWD) and writes

    runs/<name>/
        config.yaml     (verbatim copy of the input)
        train.log       (full stdout — pretrain + opt + eval traces)
        <name>.chk.h5   (trained wavefunction parameters)
        summary.json    (key results: energies, correlation fraction)

No command-line flags besides the YAML path; edit the YAML to change
the run.  See ``input_heg.yaml`` for a documented template.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
os.environ.setdefault('XLA_PYTHON_CLIENT_ALLOCATOR', 'platform')

import numpy as np
import jax
import yaml

from OmegaQMC.afqmc_3deg import (
    build_3deg_system,
    get_afqmc_3deg_func,
    pz_correlation_energy,
)
from OmegaQMC.psi.nn.heg_wf import HEGConfig, HEGPsiFormerConfig
from OmegaQMC.vmcopt_nn_heg import get_vmcopt_nn_heg_func
from OmegaQMC.vmcopt_nn_heg_sr import get_vmcopt_nn_heg_sr_func
from OmegaQMC.pretrain_heg import pretrain_heg_psiformer
from OmegaQMC.vmc_nn_heg import (
    get_vmc_nn_heg_func,
    run_twist_averaged_heg,
)


# ---------------------------------------------------------------------
# Stdout duplication: everything printed goes to terminal AND train.log
# ---------------------------------------------------------------------

class _Tee:
    def __init__(self, *streams):
        self._streams = streams

    def write(self, data):
        for s in self._streams:
            s.write(data)
            s.flush()

    def flush(self):
        for s in self._streams:
            s.flush()


# ---------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------

def _get(d, key, default=None):
    """Dict-get with dotted key lookup: _get(cfg, 'ansatz.n_det', 4)."""
    cur = d
    for part in key.split('.'):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def _build_psiformer_config(cfg, n_up, n_down, L):
    a = cfg.get('ansatz', {})
    jas_act = a.get('jas_activation', 'tanh')
    return HEGPsiFormerConfig(
        n_up=n_up, n_down=n_down, L=L,
        n_det=int(a.get('n_det', 1)),
        full_determinant=bool(a.get('full_determinant', False)),
        embedding_dim=int(a.get('embedding_dim', 64)),
        n_interactions=int(a.get('layers', 2)),
        two_particle_stream_dim=int(a.get('two_particle_dim', 16)),
        n_attention_heads=int(a.get('heads', 2)),
        use_cusp=bool(a.get('use_cusp', True)),
        use_deep_jastrow=bool(a.get('deep_jastrow', False)),
        use_pair_jastrow=bool(a.get('pair_jastrow', False)),
        jas_mlp_activation=(None if jas_act in (None, 'none')
                            else jas_act),
        jas_mlp_bias=bool(a.get('jas_bias', True)),
        jas_mlp_zero_init_last=bool(a.get('jas_zero_init_last', True)),
        n_virt_pw=int(a.get('n_virt_pw', 12)),
        det_jitter=float(a.get('det_jitter', 0.02)),
        use_ghost_atom=bool(a.get('use_ghost_atom', True)),
    )


def _build_slater_jastrow_config(cfg, n_up, n_down, L):
    a = cfg.get('ansatz', {})
    return HEGConfig(
        n_up=n_up, n_down=n_down, L=L,
        n_det=int(a.get('n_det', 1)),
        use_jastrow=bool(a.get('use_jastrow', True)),
    )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    if len(sys.argv) != 2 or sys.argv[1] in ('-h', '--help'):
        print(__doc__)
        sys.exit(0 if '-h' in sys.argv or '--help' in sys.argv else 1)

    yaml_path = Path(sys.argv[1]).resolve()
    if not yaml_path.is_file():
        print(f"error: config file not found: {yaml_path}",
              file=sys.stderr)
        sys.exit(1)

    with open(yaml_path) as f:
        cfg = yaml.safe_load(f)

    project = cfg.get('project')
    if not project:
        print("error: config must specify `project: <name>`",
              file=sys.stderr)
        sys.exit(1)

    # Per-run output directory.
    run_dir = Path('runs') / project
    run_dir.mkdir(parents=True, exist_ok=True)
    prefix = str(run_dir / project)       # chk file stem

    # Snapshot the config into the run directory for reproducibility.
    shutil.copy2(yaml_path, run_dir / 'config.yaml')

    # Tee stdout to train.log.
    log_path = run_dir / 'train.log'
    log_file = open(log_path, 'w', buffering=1)   # line-buffered
    sys.stdout = _Tee(sys.__stdout__, log_file)
    sys.stderr = _Tee(sys.__stderr__, log_file)

    try:
        result = _run(cfg, project, run_dir, prefix)
    finally:
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__
        log_file.close()

    # Persist key results.
    with open(run_dir / 'summary.json', 'w') as f:
        json.dump(result, f, indent=2)

    print(f"\n[write] {run_dir}/config.yaml")
    print(f"[write] {run_dir}/train.log")
    print(f"[write] {run_dir}/summary.json")
    print(f"[write] {run_dir}/{project}.chk.h5")


def _run(cfg, project, run_dir, prefix):
    # --- System ---
    rs = float(_get(cfg, 'system.rs', 2.0))
    N = int(_get(cfg, 'system.N', 14))
    polarization = _get(cfg, 'system.polarization', 'unpolarized')

    # --- Seed ---
    # Omitted, null, or 'random' → draw one from OS entropy so repeat
    # runs aren't bit-identical.  The chosen seed is printed and
    # persisted to summary.json so any run can still be reproduced.
    raw_seed = _get(cfg, 'seed', None)
    if raw_seed is None or (isinstance(raw_seed, str)
                            and raw_seed.lower() == 'random'):
        seed = int.from_bytes(os.urandom(4), 'little')
        seed_auto = True
    else:
        seed = int(raw_seed)
        seed_auto = False

    sys_info = build_3deg_system(
        rs, N_elec=N, N_pw=N // 2, polarization=polarization,
    )
    L = sys_info['L']
    n_up = sys_info['nup']
    n_down = sys_info['ndown']

    print("=" * 70)
    print(f"HEG VMC run: project={project}")
    print(f"  rs={rs}  N={N}  pol={polarization}")
    print(f"  Cell L={L:.4f} Bohr   V={sys_info['volume']:.4f} Bohr³")
    print(f"  n_up={n_up}  n_down={n_down}")
    print(f"  Run dir: {run_dir}")
    print(f"  Seed:    {seed}"
          f"{'  (auto: os.urandom)' if seed_auto else ''}")
    print("=" * 70)

    # --- Reference energies ---
    afqmc = get_afqmc_3deg_func(
        sys_info, dt=0.005, include_coulomb=True, verbose=False,
    )
    e_hf_ha = float(afqmc.e_trial) / N
    e_corr_pz_ha = pz_correlation_energy(rs, polarization)
    print(f"\n[ref] Finite-cell HF (AFQMC trial): "
          f"{e_hf_ha:.6f} Ha/elec = {e_hf_ha * 2:.6f} Ry/elec")
    print(f"[ref] Perdew-Zunger correlation (∞-limit): "
          f"{e_corr_pz_ha:.6f} Ha/elec = "
          f"{e_corr_pz_ha * 2:.6f} Ry/elec")

    # --- Ansatz ---
    ansatz_type = _get(cfg, 'ansatz.type', 'psiformer')
    if ansatz_type == 'psiformer':
        config = _build_psiformer_config(cfg, n_up, n_down, L)
        a = cfg['ansatz']
        print(f"  Ansatz: PsiFormer — "
              f"emb={a.get('embedding_dim', 64)}, "
              f"layers={a.get('layers', 2)}, "
              f"tp_dim={a.get('two_particle_dim', 16)}, "
              f"heads={a.get('heads', 2)}, "
              f"n_det={a.get('n_det', 1)}")
    elif ansatz_type in ('slater_jastrow', 'sj'):
        config = _build_slater_jastrow_config(cfg, n_up, n_down, L)
        print("  Ansatz: Slater-Jastrow")
    else:
        raise ValueError(f"Unknown ansatz.type: {ansatz_type!r}")

    # --- RNG keys ---
    rng = jax.random.key(seed)
    init_key, opt_key, eval_key, pretrain_key = jax.random.split(rng, 4)

    # --- Ewald ---
    ewald_n_real = int(_get(cfg, 'ewald.n_real', 3))
    ewald_n_recip = int(_get(cfg, 'ewald.n_recip', 6))

    # --- [0] Pretraining ---
    pretrained_params = None
    pretrain_iters = int(_get(cfg, 'pretrain.iters', 0))
    if pretrain_iters > 0:
        if ansatz_type != 'psiformer':
            print("[warn] pretrain.iters > 0 ignored "
                  "(pretraining is PsiFormer-only)")
        else:
            pre_walkers = int(_get(cfg, 'pretrain.walkers', 256))
            pre_lr = float(_get(cfg, 'pretrain.lr', 1e-3))
            print(f"\n[0/2] Supervised HF pre-training: "
                  f"{pretrain_iters} iters, {pre_walkers} walkers, "
                  f"lr={pre_lr}")
            pretrain_result = pretrain_heg_psiformer(
                config, init_key,
                mcmc_key=pretrain_key,
                num_iters=pretrain_iters,
                num_walkers=pre_walkers,
                lr=pre_lr, verbose=1,
            )
            pretrained_params = pretrain_result['params']
            print(f"  Pre-training MSE: "
                  f"{pretrain_result['loss_history'][0]:.4e} → "
                  f"{pretrain_result['final_loss']:.4e}")

    # --- [1] Training ---
    skip_opt = bool(_get(cfg, 'optimize.skip', False))
    opt_result = None
    trained_params = None
    if not skip_opt:
        opt_type = _get(cfg, 'optimize.type', 'sr')
        opt_lr = float(_get(cfg, 'optimize.lr', 0.05))
        opt_iters = int(_get(cfg, 'optimize.iters', 5000))
        opt_walkers = int(_get(cfg, 'optimize.walkers', 2048))
        var_weight = float(_get(cfg, 'optimize.var_weight', 0.0))
        mcmc_decorr = int(_get(cfg, 'optimize.mcmc_decorr_steps', 20))
        mc_timestep = float(_get(cfg, 'optimize.mc_timestep', 0.1))

        obj_tag = (f"L = ⟨E⟩ + {var_weight:.3g}·Var"
                   if var_weight > 0 else "⟨E⟩")
        print(f"\n[1/2] {opt_type.upper()}-VMC training: "
              f"{opt_iters} iters, {opt_walkers} walkers, "
              f"lr={opt_lr}, objective = {obj_tag}")

        if opt_type == 'sr':
            sr_damp = float(_get(cfg, 'optimize.sr_damping', 1e-3))
            sr_n_cg = int(_get(cfg, 'optimize.sr_n_cg', 30))
            opt = get_vmcopt_nn_heg_sr_func(
                config, init_key,
                lr=opt_lr, damping=sr_damp, n_cg=sr_n_cg,
                var_weight=var_weight,
                ewald_n_real=ewald_n_real,
                ewald_n_recip=ewald_n_recip,
            )
        elif opt_type == 'adam':
            opt = get_vmcopt_nn_heg_func(
                config, init_key,
                prefix=prefix, lr=opt_lr,
                var_weight=var_weight,
                ewald_n_real=ewald_n_real,
                ewald_n_recip=ewald_n_recip,
            )
        else:
            raise ValueError(f"Unknown optimize.type: {opt_type!r}")

        # Inject pre-trained weights if available.
        if pretrained_params is not None:
            if opt_type == 'sr':
                from jax.flatten_util import ravel_pytree
                opt.params_flat = ravel_pytree(pretrained_params)[0]
            else:
                opt.params = pretrained_params

        opt_result = opt(
            opt_key,
            num_iters=opt_iters,
            num_walkers=opt_walkers,
            mcmc_decorr_steps=mcmc_decorr,
            num_equil_steps=400,
            mc_timestep=mc_timestep,
            verbose=1,
        )
        trained_params = opt_result['params']
        print(f"  Final training E/N: "
              f"{opt_result['E_final_ha']:.6f} Ha/elec")
    else:
        print("\n[1/2] Skipping training (optimize.skip=true).")

    # --- [2] Evaluation ---
    eval_walkers = int(_get(cfg, 'eval.walkers', 256))
    eval_blocks = int(_get(cfg, 'eval.blocks', 40))
    eval_equil_blocks = int(_get(cfg, 'eval.equil_blocks', 20))
    steps_per_block = int(_get(cfg, 'eval.steps_per_block', 30))
    mc_timestep = float(_get(cfg, 'eval.mc_timestep',
                             _get(cfg, 'optimize.mc_timestep', 0.1)))

    print(f"\n[2/2] Evaluation VMC: "
          f"{eval_blocks} blocks × {steps_per_block} steps "
          f"× {eval_walkers} walkers")
    driver = get_vmc_nn_heg_func(
        config, init_key,
        prefix=prefix,
        ewald_n_real=ewald_n_real,
        ewald_n_recip=ewald_n_recip,
    )
    if trained_params is not None:
        driver.params = trained_params

    result = driver(
        eval_key,
        num_walkers=eval_walkers,
        num_steps_per_block=steps_per_block,
        num_blocks=eval_blocks,
        num_blocks_equil=eval_equil_blocks,
        mc_timestep=mc_timestep,
        verbose=1,
    )
    e_vmc_ha = result['E_per_elec_ha']
    e_serr_ha = result['E_serr'] / N

    # --- Summary ---
    e_corr_ha = e_vmc_ha - e_hf_ha
    recovered_frac = (100.0 * e_corr_ha / e_corr_pz_ha
                      if e_corr_pz_ha != 0 else float('nan'))

    print("\n" + "=" * 70)
    print("SUMMARY (Ha/elec unless noted)")
    print("=" * 70)
    print(f"  VMC Γ-point             = {e_vmc_ha:+.6f} ± "
          f"{e_serr_ha:.6f} Ha")
    print(f"  VMC Γ-point             = {e_vmc_ha * 2:+.6f} ± "
          f"{e_serr_ha * 2:.6f} Ry")
    print(f"  Finite-cell HF (Γ)      = {e_hf_ha:+.6f} Ha "
          f"(= {e_hf_ha * 2:+.6f} Ry)")
    print(f"  Correlation (VMC - HF)  = {e_corr_ha:+.6f} Ha")
    print(f"  Perdew-Zunger corr (∞)  = {e_corr_pz_ha:+.6f} Ha")
    print(f"  Correlation recovered   = {recovered_frac:.1f}% (vs PZ ∞)")
    print("=" * 70)

    summary = {
        'project': project,
        'seed': int(seed),
        'seed_auto': bool(seed_auto),
        'system': {'rs': rs, 'N': N, 'polarization': polarization,
                   'L': float(L), 'n_up': int(n_up),
                   'n_down': int(n_down)},
        'e_vmc_ha': float(e_vmc_ha),
        'e_vmc_serr_ha': float(e_serr_ha),
        'e_hf_ha': float(e_hf_ha),
        'e_corr_vmc_ha': float(e_corr_ha),
        'e_corr_pz_ha': float(e_corr_pz_ha),
        'recovered_pz_pct': float(recovered_frac),
    }

    # --- TABC (optional) ---
    n_twists = int(_get(cfg, 'twist.n_twists', 0))
    if n_twists > 0:
        tw_walkers = int(_get(cfg, 'twist.walkers', eval_walkers))
        tw_blocks = int(_get(cfg, 'twist.blocks', eval_blocks))
        tw_equil = int(_get(cfg, 'twist.equil_blocks',
                             eval_equil_blocks))
        print(f"\n[TABC] {n_twists} Halton twists "
              f"({tw_walkers} walkers × {tw_blocks} blocks each)")
        tabc = run_twist_averaged_heg(
            config, init_key,
            trained_params_real=trained_params,
            n_twists=n_twists,
            ewald_n_real=ewald_n_real,
            ewald_n_recip=ewald_n_recip,
            num_walkers=tw_walkers,
            num_steps_per_block=steps_per_block,
            num_blocks=tw_blocks,
            num_blocks_equil=tw_equil,
            mc_timestep=mc_timestep,
            eval_seed=seed + 1000,
            verbose=1,
        )
        e_tabc_ha = tabc['E_per_elec_ha']
        e_tabc_err_ha = tabc['E_serr_ha'] / N
        e_tabc_corr_ha = e_tabc_ha - e_hf_ha
        recovered_tabc = (100.0 * e_tabc_corr_ha / e_corr_pz_ha
                          if e_corr_pz_ha != 0 else float('nan'))

        print("\n" + "=" * 70)
        print("TABC SUMMARY (Ha/elec unless noted)")
        print("=" * 70)
        print(f"  VMC TABC                = {e_tabc_ha:+.6f} ± "
              f"{e_tabc_err_ha:.6f} Ha")
        print(f"  Correlation recovered   = {recovered_tabc:.1f}% "
              f"(vs PZ ∞)")
        print("=" * 70)

        summary['tabc'] = {
            'n_twists': n_twists,
            'e_vmc_ha': float(e_tabc_ha),
            'e_vmc_serr_ha': float(e_tabc_err_ha),
            'e_corr_ha': float(e_tabc_corr_ha),
            'recovered_pz_pct': float(recovered_tabc),
            'twist_scatter_ha': float(
                np.std(tabc['energies_per_twist']) / N
            ),
        }

    return summary


if __name__ == '__main__':
    main()
