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
import time
from pathlib import Path


def _fmt_elapsed(seconds: float) -> str:
    """Format elapsed seconds as e.g. '2d 3h 14m 7s' (drops leading zero units)."""
    secs = int(seconds)
    days, rem = divmod(secs, 86400)
    hours, rem = divmod(rem, 3600)
    minutes, sec = divmod(rem, 60)
    parts = []
    if days:
        parts.append(f"{days}d")
    if hours or parts:
        parts.append(f"{hours}h")
    if minutes or parts:
        parts.append(f"{minutes}m")
    parts.append(f"{sec}s")
    return ' '.join(parts)

os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
os.environ.setdefault('XLA_PYTHON_CLIENT_ALLOCATOR', 'platform')

import numpy as np
import jax
import yaml
from flax import nnx

from OmegaQMC.afqmc_3deg import (
    build_3deg_system,
    get_afqmc_3deg_func,
    pz_correlation_energy,
)
from OmegaQMC.heg_2d import (
    build_2deg_system,
    hf_energy_2d_finite,
    hf_energy_2d_td,
)
from OmegaQMC.psi.nn.heg_wf import HEGConfig, HEGPsiFormerConfig
from OmegaQMC.vmcopt_nn_heg import get_vmcopt_nn_heg_func
from OmegaQMC.qed_vmcopt_nn_heg_sr import get_qed_vmcopt_nn_heg_sr_func
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


_VALID_BACKBONES = ('psiformer', 'mpnqs', 'ferminet')


def _validated_backbone(value):
    """Normalise + validate the ``ansatz.backbone`` YAML field.

    A typo would silently fall through and produce an opaque
    AttributeError later in the GNN builder; raise here instead so the
    user sees the bad value next to the valid options.
    """
    bb = str(value).lower()
    if bb not in _VALID_BACKBONES:
        raise ValueError(
            f"ansatz.backbone={value!r} is not recognised; "
            f"valid choices are {_VALID_BACKBONES}.",
        )
    return bb


def _build_psiformer_config(cfg, n_up, n_down, L, dim=3):
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
        # Default False — locked Kato slope.  Trainable α was the
        # cause of a documented variational catastrophe; opt in only
        # for diagnostic studies.
        cusp_trainable_alpha=bool(a.get('cusp_trainable_alpha', False)),
        use_deep_jastrow=bool(a.get('deep_jastrow', False)),
        use_pair_jastrow=bool(a.get('pair_jastrow', False)),
        jas_mlp_activation=(None if jas_act in (None, 'none')
                            else jas_act),
        jas_mlp_bias=bool(a.get('jas_bias', True)),
        jas_mlp_zero_init_last=bool(a.get('jas_zero_init_last', True)),
        n_virt_pw=int(a.get('n_virt_pw', 12)),
        det_jitter=float(a.get('det_jitter', 0.02)),
        use_ghost_atom=bool(a.get('use_ghost_atom', True)),
        # Backflow on/off — defaults to True (FermiNet/PsiFormer recipe).
        use_backflow=bool(a.get('use_backflow', True)),
        # Backbone choice: 'psiformer' (default attention GNN), 'mpnqs'
        # (Smith 2024 / Pescia 2024 message-passing GNN), or 'ferminet'
        # (Pfau 2020 dual-stream-with-EdgeSum on sender-spin edges).
        backbone=_validated_backbone(a.get('backbone', 'psiformer')),
        mpnqs_d1=int(a.get('mpnqs_d1', 32)),
        mpnqs_d2=int(a.get('mpnqs_d2', 26)),
        mpnqs_hidden=int(a.get('mpnqs_hidden', 32)),
        mpnqs_n_layers=int(a.get('mpnqs_n_layers', 4)),
        mpnqs_use_layer_norm=bool(a.get('mpnqs_use_layer_norm', False)),
        mpnqs_layer_norm_mode=str(a.get('mpnqs_layer_norm_mode', 'post_each')),
        # Coord-transform backflow (Smith 2024).  Off by default; set
        # ``use_coord_backflow: true`` in YAML to enable.
        use_coord_backflow=bool(a.get('use_coord_backflow', False)),
        coord_bf_zero_init=bool(a.get('coord_bf_zero_init', True)),
        # Smith deep Jastrow (eqs. 20-21).  Replaces the standard
        # deep_jastrow when enabled.
        use_smith_deep_jastrow=bool(
            a.get('use_smith_deep_jastrow', False),
        ),
        smith_jastrow_hidden=int(a.get('smith_jastrow_hidden', 32)),
        smith_jastrow_n_layers=int(a.get('smith_jastrow_n_layers', 4)),
        # Envelope choice — 'plane_wave' (Fermi-sea Slater) or
        # 'crystal_gaussian' (localised Gaussians on triangular Bravais
        # lattice for the Wigner-crystal sector).
        envelope_type=str(a.get('envelope_type', 'plane_wave')),
        crystal_sigma_init=float(a.get('crystal_sigma_init', 0.25)),
        crystal_spin_pattern=str(a.get('crystal_spin_pattern', 'neel')),
        crystal_det_jitter=float(a.get('crystal_det_jitter', 0.0)),
        walker_init=str(a.get('walker_init', 'auto')),
        dim=int(dim),
    )


def _build_slater_jastrow_config(cfg, n_up, n_down, L, dim=3):
    a = cfg.get('ansatz', {})
    return HEGConfig(
        n_up=n_up, n_down=n_down, L=L,
        n_det=int(a.get('n_det', 1)),
        use_jastrow=bool(a.get('use_jastrow', True)),
        dim=int(dim),
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

    t_start = time.monotonic()
    try:
        result = _run(cfg, project, run_dir, prefix)
        elapsed = time.monotonic() - t_start
        result['elapsed_seconds'] = float(elapsed)
        print(f"\nTotal elapsed time: {_fmt_elapsed(elapsed)}  "
              f"({elapsed:.1f} s)")
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
    dim = int(_get(cfg, 'system.dim', 3))
    if dim not in (2, 3):
        raise ValueError(f"system.dim must be 2 or 3, got {dim}")

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

    if dim == 3:
        sys_info = build_3deg_system(
            rs, N_elec=N, N_pw=N // 2, polarization=polarization,
        )
        cell_label = "V"
        cell_unit = "Bohr^3"
        cell_val = sys_info['volume']
    else:  # dim == 2
        sys_info = build_2deg_system(
            rs, N_elec=N, polarization=polarization,
        )
        cell_label = "A"
        cell_unit = "Bohr^2"
        cell_val = sys_info['area']
    L = sys_info['L']
    n_up = sys_info['nup']
    n_down = sys_info['ndown']

    print("=" * 70)
    print(f"HEG VMC run: project={project}  dim={dim}")
    print(f"  rs={rs}  N={N}  pol={polarization}")
    print(f"  Cell L={L:.4f} Bohr   {cell_label}={cell_val:.4f} {cell_unit}")
    print(f"  n_up={n_up}  n_down={n_down}")
    print(f"  Run dir: {run_dir}")
    print(f"  Seed:    {seed}"
          f"{'  (auto: os.urandom)' if seed_auto else ''}")
    print("=" * 70)

    # --- Reference energies ---
    if dim == 3:
        afqmc = get_afqmc_3deg_func(
            sys_info, dt=0.005, include_coulomb=True, verbose=False,
        )
        e_hf_ha = float(afqmc.e_trial) / N
        e_corr_pz_ha = pz_correlation_energy(rs, polarization)
        print(f"\n[ref] Finite-cell HF (AFQMC trial): "
              f"{e_hf_ha:.6f} Ha/elec = {e_hf_ha * 2:.6f} Ry/elec")
        print(f"[ref] Perdew-Zunger correlation (inf-limit): "
              f"{e_corr_pz_ha:.6f} Ha/elec = "
              f"{e_corr_pz_ha * 2:.6f} Ry/elec")
    else:
        # 2D: analytical HF reference (Stern 1973 closed form for the
        # thermodynamic limit, plus the finite-N correction via 2D
        # Ewald and a discrete-k Fermi-sea sum).
        hf_2d = hf_energy_2d_finite(sys_info)
        e_hf_ha = float(hf_2d['total'])
        e_hf_td = hf_energy_2d_td(rs, polarization)
        # No PZ-style 2D correlation parametrization included by
        # default; user can compare to Attaccalite 2002 manually.
        e_corr_pz_ha = 0.0
        print(f"\n[ref] Finite-N HF (analytic, 2D): "
              f"{e_hf_ha:.6f} Ha/elec  "
              f"(TD limit: {e_hf_td:.6f}, FS = "
              f"{(e_hf_ha - e_hf_td)*1000:+.2f} mHa)")
        print(f"  T={hf_2d['kinetic']:.6f}, "
              f"V_x={hf_2d['exchange']:.6f}, "
              f"e_M={hf_2d['madelung']:.6f}  Ha/elec")

    # --- Ansatz ---
    # ``ansatz.type`` selects the *config dataclass* and the matching
    # builder pipeline.  Within ``nn_heg`` (PsiFormer/MP-NQS family),
    # the actual GNN backbone is selected by ``ansatz.backbone``
    # ('psiformer' single-stream attention vs. 'mpnqs' Pescia/Smith
    # dual-stream message passing).  ``type: psiformer`` is kept as a
    # backward-compat alias for existing YAMLs.
    ansatz_type = str(_get(cfg, 'ansatz.type', 'nn_heg')).lower()
    if ansatz_type in ('nn_heg', 'psiformer'):
        config = _build_psiformer_config(cfg, n_up, n_down, L, dim=dim)
        a = cfg['ansatz']
        backbone = _validated_backbone(a.get('backbone', 'psiformer'))
        if backbone == 'mpnqs':
            print(
                f"  Ansatz: nn_heg / MP-NQS (dim={dim}) - "
                f"d1={a.get('mpnqs_d1', 32)}, "
                f"d2={a.get('mpnqs_d2', 26)}, "
                f"hidden={a.get('mpnqs_hidden', 32)}, "
                f"T={a.get('mpnqs_n_layers', 4)}, "
                f"n_det={a.get('n_det', 1)}"
            )
        elif backbone == 'ferminet':
            print(
                f"  Ansatz: nn_heg / FermiNet (dim={dim}) - "
                f"emb={a.get('embedding_dim', 64)}, "
                f"layers={a.get('layers', 2)}, "
                f"tp_dim={a.get('two_particle_dim', 16)}, "
                f"n_det={a.get('n_det', 1)}, "
                f"full_det={a.get('full_determinant', False)}"
            )
        else:
            print(
                f"  Ansatz: nn_heg / PsiFormer (dim={dim}) - "
                f"emb={a.get('embedding_dim', 64)}, "
                f"layers={a.get('layers', 2)}, "
                f"tp_dim={a.get('two_particle_dim', 16)}, "
                f"heads={a.get('heads', 2)}, "
                f"n_det={a.get('n_det', 1)}"
            )
    elif ansatz_type in ('slater_jastrow', 'sj'):
        config = _build_slater_jastrow_config(
            cfg, n_up, n_down, L, dim=dim,
        )
        print(f"  Ansatz: Slater-Jastrow (dim={dim})")
    else:
        raise ValueError(
            f"Unknown ansatz.type: {ansatz_type!r}.  "
            f"Valid: 'nn_heg' (=PsiFormer/MP-NQS pipeline; backbone "
            f"selectable), 'psiformer' (legacy alias for 'nn_heg'), "
            f"'slater_jastrow' (or 'sj')."
        )

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
        if ansatz_type not in ('nn_heg', 'psiformer'):
            print("[warn] pretrain.iters > 0 ignored "
                  "(pretraining only available for nn_heg ansatz)")
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
            lr_schedule = str(_get(cfg, 'optimize.lr_schedule', 'auto'))
            lr_decay_T = _get(cfg, 'optimize.lr_decay_T', None)
            if lr_decay_T is not None:
                lr_decay_T = float(lr_decay_T)
            lr_min = float(_get(cfg, 'optimize.lr_min', 0.0))
            lr_T_max = _get(cfg, 'optimize.lr_T_max', None)
            if lr_T_max is not None:
                lr_T_max = int(lr_T_max)
            lr_n_restarts = int(_get(cfg, 'optimize.lr_n_restarts', 0))
            spring_mu = float(_get(cfg, 'optimize.spring_mu', 0.0))
            spring_norm_clip = _get(cfg, 'optimize.spring_norm_clip', None)
            if spring_norm_clip is not None:
                spring_norm_clip = float(spring_norm_clip)
            damping_adapt = bool(_get(
                cfg, 'optimize.damping_adapt', False,
            ))
            damping_min = float(_get(
                cfg, 'optimize.damping_min', 1e-5,
            ))
            damping_max = float(_get(
                cfg, 'optimize.damping_max', 1e-1,
            ))
            damping_factor = float(_get(
                cfg, 'optimize.damping_factor', 2.0,
            ))
            damping_lookback = int(_get(
                cfg, 'optimize.damping_lookback', 50,
            ))
            sampler = str(_get(cfg, 'optimize.sampler', 'metropolis'))
            mala_grad_clip = _get(cfg, 'optimize.mala_grad_clip', 1.0)
            if mala_grad_clip is not None:
                mala_grad_clip = float(mala_grad_clip)
            save_every = int(_get(cfg, 'optimize.save_every', 500))
            # ---- QED Step 2: cavity-mode parameters from YAML ----
            # cavity:
            #   omega: 0.1
            #   nph_max: 4
            # (lambda is a Step-4 thing; not used here.)
            omega = float(_get(cfg, 'cavity.omega', 0.0))
            nph_max = int(_get(cfg, 'cavity.nph_max', 0))
            coupling_lambda = float(_get(cfg, 'cavity.lambda', 0.0))
            coupling_polarization = _get(
                cfg, 'cavity.polarization', None,
            )
            # Step 5: Option C multi-K phase + coherent-state c_n.
            phase_K_vectors = _get(cfg, 'cavity.phase_K_vectors', None)
            phase_alpha_init = _get(cfg, 'cavity.phase_alpha_init', None)
            alpha_step_clip = float(
                _get(cfg, 'cavity.alpha_step_clip', 0.005)
            )
            coh_alpha_init = float(
                _get(cfg, 'cavity.coh_alpha_init', 0.05)
            )
            coh_alpha_step_clip = float(
                _get(cfg, 'cavity.coh_alpha_step_clip', 0.005)
            )
            coh_alpha_floor = float(
                _get(cfg, 'cavity.coh_alpha_floor', 1.0e-3)
            )
            opt = get_qed_vmcopt_nn_heg_sr_func(
                config, init_key,
                prefix=prefix,
                lr=opt_lr, damping=sr_damp, n_cg=sr_n_cg,
                var_weight=var_weight,
                ewald_n_real=ewald_n_real,
                ewald_n_recip=ewald_n_recip,
                lr_schedule=lr_schedule,
                lr_decay_T=lr_decay_T,
                lr_min=lr_min,
                lr_T_max=lr_T_max,
                lr_n_restarts=lr_n_restarts,
                spring_mu=spring_mu,
                spring_norm_clip=spring_norm_clip,
                damping_adapt=damping_adapt,
                damping_min=damping_min,
                damping_max=damping_max,
                damping_factor=damping_factor,
                damping_lookback=damping_lookback,
                sampler=sampler,
                mala_grad_clip=mala_grad_clip,
                save_every=save_every,
                omega=omega,
                nph_max=nph_max,
                coupling_lambda=coupling_lambda,
                coupling_polarization=coupling_polarization,
                phase_K_vectors=phase_K_vectors,
                phase_alpha_init=phase_alpha_init,
                alpha_step_clip=alpha_step_clip,
                coh_alpha_init=coh_alpha_init,
                coh_alpha_step_clip=coh_alpha_step_clip,
                coh_alpha_floor=coh_alpha_floor,
            )
        elif opt_type == 'adam':
            opt = get_vmcopt_nn_heg_func(
                config, init_key,
                prefix=prefix, lr=opt_lr,
                var_weight=var_weight,
                ewald_n_real=ewald_n_real,
                ewald_n_recip=ewald_n_recip,
            )
        elif opt_type == 'kfac':
            from OmegaQMC.vmcopt_nn_heg_kfac import (
                get_vmcopt_nn_heg_kfac_func,
            )
            opt = get_vmcopt_nn_heg_kfac_func(
                config, init_key,
                lr=opt_lr,
                lr_decay=_get(cfg, 'optimize.lr_decay', 1.0e4),
                damping=float(_get(cfg, 'optimize.kfac_damping', 0.1)),
                damping_adapt=bool(_get(cfg,
                                        'optimize.kfac_damping_adapt', True)),
                damping_min=float(_get(cfg, 'optimize.kfac_damping_min',
                                       1.0e-4)),
                damping_max=float(_get(cfg, 'optimize.kfac_damping_max',
                                       1.0e2)),
                damping_lookback=int(_get(cfg,
                                          'optimize.kfac_damping_lookback', 10)),
                damping_decay=float(_get(cfg,
                                        'optimize.kfac_damping_decay', 0.95)),
                damping_overshoot_threshold=float(_get(cfg,
                    'optimize.kfac_damping_overshoot_threshold', 10.0)),
                damping_overshoot_factor=float(_get(cfg,
                    'optimize.kfac_damping_overshoot_factor', 2.0)),
                ema_decay=float(_get(cfg, 'optimize.kfac_ema_decay', 0.0)),
                norm_constraint=_get(cfg,
                                     'optimize.kfac_norm_constraint', None),
                var_weight=var_weight,
                capture_activations=bool(_get(
                    cfg, 'optimize.kfac_capture_activations', False,
                )),
                fixed_scale=bool(_get(
                    cfg, 'optimize.kfac_fixed_scale', False,
                )),
                ewald_n_real=ewald_n_real,
                ewald_n_recip=ewald_n_recip,
                multi_device=bool(_get(cfg, 'optimize.multi_device', False)),
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

        # Resume from a previous checkpoint (overrides pretrain).
        # ``load_chkpt_partial: true`` enables transfer learning from
        # a checkpoint with different shapes — e.g. transferring the
        # MPNN body / deep Jastrow / coord BF (N-independent) from a
        # smaller-N run while keeping fresh init for the orbital
        # coefficients (N-dependent).
        load_chkpt = _get(cfg, 'optimize.load_chkpt', None)
        load_partial = bool(_get(
            cfg, 'optimize.load_chkpt_partial', False,
        ))
        if load_chkpt:
            from OmegaQMC.nn_checkpoint import (
                load_nn_checkpoint, load_nn_checkpoint_partial,
                load_nn_checkpoint_partial_by_path,
            )
            from jax.flatten_util import ravel_pytree
            chkpt_path = Path(load_chkpt)
            if not chkpt_path.is_absolute():
                chkpt_path = Path.cwd() / chkpt_path
            if not chkpt_path.is_file():
                raise FileNotFoundError(
                    f"optimize.load_chkpt: {chkpt_path} not found"
                )
            # Path-based partial loading: when source has a different
            # set of optional modules than target (e.g., adding
            # coord_backflow to a chkpt that didn't have it), nnx's
            # alphabetical leaf order shifts existing leaves so the
            # index-based partial loader misaligns.  Path-based load
            # avoids this by matching leaves on key path.  User
            # supplies ``load_chkpt_source_overrides`` — ansatz-field
            # overrides describing the source's config relative to
            # the current target.
            src_overrides = _get(
                cfg, 'optimize.load_chkpt_source_overrides', None,
            )
            mode_label = (
                "partial transfer (path)" if (load_partial and src_overrides)
                else ("partial transfer" if load_partial else "resume")
            )
            print(f"\n[{mode_label}] Loading params from {chkpt_path}")
            if load_partial and src_overrides:
                # Build shadow-source pytree to derive the chkpt's paths.
                shadow_cfg_dict = {**cfg}
                shadow_cfg_dict['ansatz'] = {
                    **cfg.get('ansatz', {}), **src_overrides,
                }
                shadow_config = _build_psiformer_config(
                    shadow_cfg_dict, n_up, n_down, L, dim=dim,
                )
                from OmegaQMC.psi.nn.heg_wf_module import (
                    build_heg_psiformer_wf,
                )
                shadow_model = build_heg_psiformer_wf(
                    shadow_config, nnx.Rngs(int(seed)),
                )
                shadow_state = nnx.state(shadow_model, nnx.Param)
                if opt_type == 'sr':
                    template = opt.unravel(opt.params_flat)
                    loaded, meta = load_nn_checkpoint_partial_by_path(
                        str(chkpt_path), template, shadow_state,
                    )
                else:
                    loaded, meta = load_nn_checkpoint_partial_by_path(
                        str(chkpt_path), opt.params, shadow_state,
                    )
            else:
                loader = (
                    load_nn_checkpoint_partial
                    if load_partial
                    else load_nn_checkpoint
                )
                if opt_type == 'sr':
                    template = opt.unravel(opt.params_flat)
                    loaded, meta = loader(str(chkpt_path), template)
                else:
                    loaded, meta = loader(str(chkpt_path), opt.params)

            # Optional: reset specific module(s) to zero after load.
            # Useful for transfer learning: keep MPNN body + Jastrow
            # from source, but reset coord_backflow so the optimizer
            # learns BF fresh at the new system size (avoids over-
            # shifted BF when N changes — h_i^(T) magnitudes differ
            # because aggregation is sum-not-mean).
            reset_bf = bool(_get(
                cfg, 'optimize.load_chkpt_reset_bf', False,
            ))
            if reset_bf:
                from OmegaQMC.nn_checkpoint import zero_module_leaves
                loaded = zero_module_leaves(loaded, 'coord_backflow')

            if opt_type == 'sr':
                opt.params_flat = ravel_pytree(loaded)[0]
            else:
                opt.params = loaded

            ep_meta = meta.get('epoch', '?')
            e_meta = meta.get('energy', None)
            e_str = (f"{float(e_meta):+.6f} Ha"
                     if e_meta is not None else "n/a")
            print(f"  source @ epoch={ep_meta}, energy={e_str}")

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
        e_final = opt_result.get('E_final_ha')
        if e_final is not None:
            print(f"  Final training E/N: {e_final:.6f} Ha/elec")
        else:
            print(f"  (no training iters executed — proceeding to eval "
                  f"with loaded params)")
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

    qed_eval_result = None
    if opt_type == 'sr':
        # QED-aware eval — reuses opt's MCMC + E_loc (full Pauli-Fierz
        # including ω·n_ph photon term and diamagnetic A² shift) and
        # the composite (R, n) Metropolis chain. Single source of
        # truth with training, no analytic post-hoc offsets.
        qed_eval_result = opt.evaluate(
            eval_key,
            params_flat=opt.params_flat,
            num_walkers=eval_walkers,
            num_blocks=eval_blocks,
            num_blocks_equil=eval_equil_blocks,
            num_steps_per_block=steps_per_block,
            mc_timestep=mc_timestep,
            verbose=1,
        )
        result = qed_eval_result
    else:
        # Non-QED path (Adam fallback) — bare HEG eval driver.
        driver = get_vmc_nn_heg_func(
            config, init_key, prefix=prefix,
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
    if qed_eval_result is not None:
        kin_avg = float(np.mean(qed_eval_result['E_kin_blocks'])) / N
        pot_avg = float(np.mean(qed_eval_result['E_pot_blocks'])) / N
        phot_avg = float(np.mean(qed_eval_result['E_phot_blocks'])) / N
        diamag_avg = float(np.mean(qed_eval_result['E_diamag_blocks'])) / N
        para_re_avg = float(np.mean(qed_eval_result['E_para_re_blocks'])) / N
        para_im_avg = float(np.mean(qed_eval_result['E_para_im_blocks'])) / N
        para_im_serr = float(
            np.std(qed_eval_result['E_para_im_blocks'])
            / np.sqrt(len(qed_eval_result['E_para_im_blocks']))
        ) / N
        nph_avg = float(np.mean(qed_eval_result['n_ph_blocks']))
        print(f"  VMC Γ-point (E_QED)      = {e_vmc_ha:+.6f} ± "
              f"{e_serr_ha:.6f} Ha")
        print(f"  VMC Γ-point (E_QED)      = {e_vmc_ha * 2:+.6f} ± "
              f"{e_serr_ha * 2:.6f} Ry")
        print(f"    <E_kin>/N              = {kin_avg:+.6f} Ha")
        print(f"    <E_pot>/N              = {pot_avg:+.6f} Ha")
        print(f"    <E_phot>/N             = {phot_avg:+.6f} Ha")
        print(f"    <E_diamag>/N           = {diamag_avg:+.6f} Ha")
        print(f"    <E_para_re>/N          = {para_re_avg:+.6f} Ha "
              f"  (Phase 1: must be 0 exactly)")
        print(f"    <E_para_im>/N          = {para_im_avg:+.4e} ± "
              f"{para_im_serr:.4e} Ha   "
              f"(Hermiticity: must avg to 0 within MCMC noise)")
        print(f"    <n_ph>                 = {nph_avg:.4e}")
    else:
        print(f"  VMC Γ-point             = {e_vmc_ha:+.6f} ± "
              f"{e_serr_ha:.6f} Ha")
        print(f"  VMC Γ-point             = {e_vmc_ha * 2:+.6f} ± "
              f"{e_serr_ha * 2:.6f} Ry")
    print(f"  Finite-cell HF (Γ)       = {e_hf_ha:+.6f} Ha "
          f"(= {e_hf_ha * 2:+.6f} Ry)")
    print(f"  Correlation (VMC - HF)   = {e_corr_ha:+.6f} Ha")
    print(f"  Perdew-Zunger corr (∞)   = {e_corr_pz_ha:+.6f} Ha")
    print(f"  Correlation recovered    = {recovered_frac:.1f}% (vs PZ ∞)")
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
    if qed_eval_result is not None:
        summary['cavity_eval'] = {
            'omega': float(_get(cfg, 'cavity.omega', 0.0)),
            'nph_max': int(_get(cfg, 'cavity.nph_max', 0)),
            'lambda': float(_get(cfg, 'cavity.lambda', 0.0)),
            'e_kin_per_e_ha':     float(np.mean(qed_eval_result['E_kin_blocks'])) / N,
            'e_pot_per_e_ha':     float(np.mean(qed_eval_result['E_pot_blocks'])) / N,
            'e_phot_per_e_ha':    float(np.mean(qed_eval_result['E_phot_blocks'])) / N,
            'e_diamag_per_e_ha':  float(np.mean(qed_eval_result['E_diamag_blocks'])) / N,
            'e_para_re_per_e_ha': float(np.mean(qed_eval_result['E_para_re_blocks'])) / N,
            'e_para_im_per_e_ha': float(np.mean(qed_eval_result['E_para_im_blocks'])) / N,
            'n_ph_avg':           float(np.mean(qed_eval_result['n_ph_blocks'])),
            'phase_alpha':        qed_eval_result.get('phase_alpha', []),
            'coh_alpha':          qed_eval_result.get('coh_alpha', 0.0),
        }

    # --- Observables (S(k), optional) ---
    obs_cfg = cfg.get('observables') or {}
    if obs_cfg.get('enabled', False):
        print("\n[obs] Accumulating S(k) on |psi|^2 walkers ...")
        # Build a bare-HEG driver on demand — needed only for the
        # |ψ_e|² walker chain (photonic state factors out for the
        # purely-electronic structure factor S(k)).
        if 'driver' not in locals():
            driver = get_vmc_nn_heg_func(
                config, init_key, prefix=prefix,
                ewald_n_real=ewald_n_real,
                ewald_n_recip=ewald_n_recip,
            )
            if trained_params is not None:
                driver.params = trained_params
        from OmegaQMC.observables.structure_factor import (
            reciprocal_grid_2d,
            reciprocal_lattice_vectors_triangular,
            structure_factor,
        )
        import jax.numpy as jnp

        # Build k-grids requested by the YAML.
        k_grids = {}
        if obs_cfg.get('triangular_shells', 0) > 0:
            n_shell = int(obs_cfg['triangular_shells'])
            kt = reciprocal_lattice_vectors_triangular(
                rs=rs, n_shell=n_shell,
            )
            k_grids['triangular_bragg'] = jnp.asarray(kt)
        if obs_cfg.get('cartesian_n_max', None) is not None:
            n_max_k = int(obs_cfg['cartesian_n_max'])
            k_grids['cartesian_grid'] = reciprocal_grid_2d(L, n_max=n_max_k)
        if not k_grids:
            print("[obs] No k-grids requested; skipping.")
        else:
            n_walkers_obs = int(obs_cfg.get('n_walkers', eval_walkers))
            n_equil_obs = int(obs_cfg.get('equil_steps', 200))
            n_sample_obs = int(obs_cfg.get('sample_steps', 100))
            decorr_obs = int(obs_cfg.get('decorr_steps', 5))

            rng_obs = jax.random.fold_in(eval_key, 999)
            walkers_obs = driver.initialize_walkers(
                rng_obs, n_walkers_obs,
            )
            step_size_obs = (3 * mc_timestep) ** 0.5

            # Equilibrate on |psi|^2 with trained params.
            for _ in range(n_equil_obs):
                rng_obs, sk_key = jax.random.split(rng_obs)
                keys = jax.random.split(sk_key, n_walkers_obs)
                walkers_obs, _ = driver._metropolis_move_allw(
                    keys, walkers_obs, step_size_obs, driver.params,
                )

            # Per-block S(k) averages, then mean +- serr across blocks.
            sk_blocks = {name: [] for name in k_grids}
            sk_eval_fns = {
                name: jax.jit(jax.vmap(
                    lambda r, kg=kg: structure_factor(r, kg),
                ))
                for name, kg in k_grids.items()
            }
            for _ in range(n_sample_obs):
                for _ in range(decorr_obs):
                    rng_obs, sk_key = jax.random.split(rng_obs)
                    keys = jax.random.split(sk_key, n_walkers_obs)
                    walkers_obs, _ = driver._metropolis_move_allw(
                        keys, walkers_obs, step_size_obs, driver.params,
                    )
                for name in k_grids:
                    sk_per_walker = sk_eval_fns[name](walkers_obs)
                    sk_blocks[name].append(
                        np.asarray(jnp.mean(sk_per_walker, axis=0)),
                    )

            summary['observables'] = {}
            for name, blocks in sk_blocks.items():
                arr = np.stack(blocks)            # (n_sample, n_k)
                mean = np.mean(arr, axis=0)
                serr = np.std(arr, axis=0) / np.sqrt(arr.shape[0])
                summary['observables'][f'sk_{name}'] = {
                    'k_vectors': np.asarray(k_grids[name]).tolist(),
                    'mean': mean.tolist(),
                    'serr': serr.tolist(),
                }
                # Print top-few peaks for quick eyeballing.
                idx = np.argsort(-mean)[:6]
                print(f"  [{name}] top S(k) peaks (mean +- serr):")
                for i in idx:
                    k = np.asarray(k_grids[name])[i]
                    print(
                        f"    k=({k[0]:+.4f}, {k[1]:+.4f})  "
                        f"|k|={np.linalg.norm(k):.4f}  "
                        f"S(k) = {mean[i]:.3f} +- {serr[i]:.3f}"
                    )

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
