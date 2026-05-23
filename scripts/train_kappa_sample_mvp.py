"""κ-sampling fine-tuning MVP for V11.

Loads a V11 Γ-trained chkpt, then runs short SR-VMC training rounds at
different twists κ_t (sampled uniformly).  At each round the optimizer
is rebuilt at twist=κ_t and params are transferred from the previous
round.  The complex envelope's coefficients are indexed by integer
n-vectors, so the transfer between κ's is identity for the envelope
block (and direct for non-envelope blocks).

Goal: train base coefficients to be "κ-averaged optimal" rather than
"Γ-specialized" — should reduce the +23 mHa TABC shift seen on
V11_on_g050.

Note: this MVP does NOT use the κ-aware Δ-coefficient MLPs (those
need full L5 plumbing of κ as a runtime arg, deferred to Phase 2b).
If MVP works, we know κ-sampling helps; Phase 2b adds the Δ MLP for
a finer-grained correction.

Usage:
    python scripts/train_kappa_sample_mvp.py \\
        inputs/2dheg_qed/l5_weber_fig1b_v11_on.yaml \\
        --chkpt runs/l5_weber_fig1b_v11_on/l5_weber_fig1b_v11_on.chk.npz \\
        --n-rounds 4 --steps-per-round 50 --out runs/v11_kappa_sample_mvp
"""
from __future__ import annotations
import argparse, json, math, sys, time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import yaml
from jax.flatten_util import ravel_pytree

jax.config.update("jax_enable_x64", True)
sys.path.insert(0, ".")

from OmegaQMC.psi.nn.heg_wf import HEGPsiFormerConfig, transfer_trained_params
from OmegaQMC.psi.nn.env_periodic import (
    enumerate_complex_pw_basis_2d, enumerate_real_pw_basis_2d,
)
from OmegaQMC.qed_vmcopt_nn_heg_l5 import _QEDL5Optimizer


# -- Path helpers + transfer (reused from run_qed_l5_tabc.py) ----------------

def _path_to_str(path):
    parts = []
    for p in path:
        for attr in ("key", "name", "idx"):
            v = getattr(p, attr, None)
            if v is not None: parts.append(str(v)); break
        else: parts.append(str(p))
    return "/".join(parts)


def _extract_envelope_coeffs_real(chkpt_pytree):
    """Extract V11 real-env cos/sin coefficients."""
    out = {}
    for path, leaf in jax.tree_util.tree_leaves_with_path(chkpt_pytree):
        s = _path_to_str(path)
        for k in ("cos_up", "sin_up", "cos_dn", "sin_dn"):
            if s.endswith(f"envelope/coeff_{k}/value"):
                out[k] = np.asarray(leaf)
    return out


def _extract_envelope_coeffs_complex(pytree):
    """Extract complex env coefficients (coeff_up, coeff_dn)."""
    out = {}
    for path, leaf in jax.tree_util.tree_leaves_with_path(pytree):
        s = _path_to_str(path)
        for k in ("up", "dn"):
            if s.endswith(f"envelope/coeff_{k}/value"):
                out[k] = np.asarray(leaf)
    return out


def cos_sin_to_complex_coeffs(c_cos, c_sin, real_basis, complex_basis, L, tol=1e-6):
    """V11 (cos/sin) → complex (exp) transfer (n_int matching for any κ)."""
    n_det, n_orb, n_unique = c_cos.shape
    real_kvecs = np.asarray(real_basis.kvecs)
    complex_kvecs = np.asarray(complex_basis.kvecs)
    n_complex_pw = complex_kvecs.shape[0]
    dk = 2.0 * np.pi / L
    real_n_int = np.round(real_kvecs / dk).astype(np.int32)
    complex_n_int = np.asarray(complex_basis.n_ints)
    basis_idx = np.asarray(real_basis.basis_idx)
    basis_is_sin = np.asarray(real_basis.basis_is_sin)
    active_cos = np.zeros(n_unique, dtype=bool)
    active_sin = np.zeros(n_unique, dtype=bool)
    for k_i, is_s in zip(basis_idx, basis_is_sin):
        if int(is_s): active_sin[int(k_i)] = True
        else: active_cos[int(k_i)] = True
    out = np.zeros((n_det, n_orb, n_complex_pw), dtype=np.complex128)
    for n in range(n_unique):
        if not (active_cos[n] or active_sin[n]): continue
        n_int = real_n_int[n]
        is_zero = bool(np.all(n_int == 0))
        plus_idx, minus_idx = None, None
        for m in range(n_complex_pw):
            if np.array_equal(complex_n_int[m], n_int): plus_idx = m
            elif np.array_equal(complex_n_int[m], -n_int): minus_idx = m
        if is_zero:
            if plus_idx is not None and active_cos[n]:
                out[:, :, plus_idx] += c_cos[:, :, n].astype(np.complex128)
            continue
        if plus_idx is None and minus_idx is None: continue
        if active_cos[n]:
            if plus_idx is not None: out[:, :, plus_idx] += c_cos[:, :, n] / 2.0
            if minus_idx is not None: out[:, :, minus_idx] += c_cos[:, :, n] / 2.0
        if active_sin[n]:
            if plus_idx is not None: out[:, :, plus_idx] += -1j * c_sin[:, :, n] / 2.0
            if minus_idx is not None: out[:, :, minus_idx] += +1j * c_sin[:, :, n] / 2.0
    return out


def _set_envelope_complex(pytree, c_up, c_dn):
    cu = jnp.asarray(c_up); cd = jnp.asarray(c_dn)
    def replace(path, leaf):
        s = _path_to_str(path)
        if s.endswith("envelope/coeff_up/value"): return cu
        if s.endswith("envelope/coeff_dn/value"): return cd
        return leaf
    return jax.tree_util.tree_map_with_path(replace, pytree)


def _transfer_complex_to_complex(src_pytree, dst_pytree):
    """Identity transfer between two complex-env pytrees (assumes same
    n-vector indexing, which holds since both built with same antipodal-
    closure bump).  Copies envelope coeffs + everything else."""
    src_leaves = {_path_to_str(p): l
                  for p, l in jax.tree_util.tree_leaves_with_path(src_pytree)}
    def replace(path, leaf):
        key = _path_to_str(path)
        if key in src_leaves:
            src_leaf = src_leaves[key]
            if src_leaf.shape == leaf.shape:
                return src_leaf
        return leaf
    return jax.tree_util.tree_map_with_path(replace, dst_pytree)


def _build_optimizer_at_twist(cfg_yaml, twist, init_key, lr_override=None):
    sys_c = cfg_yaml["system"]; cav = cfg_yaml["cavity"]
    an = cfg_yaml["ansatz"]; oc = cfg_yaml["optimize"]
    lr_use = float(lr_override) if lr_override is not None else oc["lr"]
    N = sys_c["n_up"] + sys_c["n_down"]
    L = sys_c["rs"] * math.sqrt(math.pi * N)
    cfg = HEGPsiFormerConfig(
        n_up=sys_c["n_up"], n_down=sys_c["n_down"], L=L, dim=2,
        backbone=an["backbone"], embedding_dim=an["embedding_dim"],
        n_interactions=an["n_interactions"],
        two_particle_stream_dim=an["two_particle_stream_dim"],
        n_det=an["n_det"], full_determinant=an["full_determinant"],
        use_backflow=an["use_backflow"], use_cusp=an["use_cusp"],
        n_virt_pw=an["n_virt_pw"], use_ghost_atom=an["use_ghost_atom"],
        use_deep_jastrow=an.get("use_deep_jastrow", False),
        use_smith_deep_jastrow=an.get("use_smith_deep_jastrow", False),
        envelope_type=an["envelope_type"],
    )
    return _QEDL5Optimizer(
        cfg, init_key,
        lr=lr_use, damping=oc["sr_damping"],
        n_cg=oc.get("sr_n_cg", 20),
        omega=cav["omega"], coupling_lambda=cav["lambda"],
        coupling_polarization=cav.get("polarization", [1.0, 0.0]),
        coupling_op=cav.get("coupling_op", "P"),
        K_max=cav["K_max"],
        phase_mlp_hidden=tuple(cav["phase_mlp_hidden"]),
        mag_mlp_hidden=tuple(cav["mag_mlp_hidden"]),
        spring_mu=oc.get("spring_mu", 0.0),
        spring_norm_clip=oc.get("spring_norm_clip", 1e10),
        use_smw_sr=oc.get("use_smw_sr", True),
        use_fused_step=oc.get("use_fused_step", True),
        use_matter_photon_shift=bool(cav.get("use_matter_photon_shift", False)),
        use_matter_photon_pshift=bool(cav.get("use_matter_photon_pshift", False)),
        q0_mlp_hidden=tuple(cav.get("q0_mlp_hidden", [32, 32])),
        p0_mlp_hidden=tuple(cav.get("p0_mlp_hidden", [32, 32])),
        v_ext_amp=float(cfg_yaml.get("v_ext", {}).get("amp", 0.0)),
        v_ext_a=cfg_yaml.get("v_ext", {}).get("a", None),
        include_vee=bool(sys_c.get("include_vee", True)),
        twist=twist,
    )


# -- Main --------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("config", type=str, help="V11 YAML config")
    p.add_argument("--chkpt", type=str, required=True,
                   help="V11 Γ-trained chkpt")
    p.add_argument("--n-rounds", type=int, default=4,
                   help="Number of distinct κ values to train at "
                        "(each gets STEPS_PER_ROUND SR iters)")
    p.add_argument("--steps-per-round", type=int, default=50)
    p.add_argument("--equil-steps", type=int, default=50,
                   help="Equilibration steps per round (kept short to "
                        "amortize across rounds)")
    p.add_argument("--walkers", type=int, default=1024)
    p.add_argument("--mcmc-decorr-steps", type=int, default=20)
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--lr-override", type=float, default=None,
                   help="override yaml's optimize.lr (recommend 0.05-0.1 "
                        "for fine-tuning to avoid κ-overfitting)")
    p.add_argument("--out", type=str, required=True,
                   help="output directory")
    args = p.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.config) as f:
        cfg_yaml = yaml.safe_load(f)

    # Load V11 chkpt
    chkpt = np.load(args.chkpt, allow_pickle=True)
    chkpt_params_flat = jnp.asarray(chkpt["params_flat"])
    print(f"[load] {args.chkpt}: {chkpt_params_flat.shape[0]} params, "
          f"E_final = {float(chkpt['E_final_ha'])*1000:+.4f} mHa/e")

    # Build reference real-Γ opt to unravel chkpt
    ref_opt = _build_optimizer_at_twist(cfg_yaml, twist=None,
                                         init_key=jax.random.key(args.seed),
                                         lr_override=args.lr_override)
    assert ref_opt.n_params == chkpt_params_flat.shape[0]
    chkpt_pytree = ref_opt.unravel(chkpt_params_flat)
    v11_real_coeffs = _extract_envelope_coeffs_real(chkpt_pytree)

    # Reference real basis (for cos/sin → complex conversion at iter 0)
    sys_c0 = cfg_yaml["system"]
    an0 = cfg_yaml["ansatz"]
    N = sys_c0["n_up"] + sys_c0["n_down"]
    L = sys_c0["rs"] * math.sqrt(math.pi * N)
    if an0.get("full_determinant", False):
        v11_pw_size = (sys_c0["n_up"] + sys_c0["n_down"]) + int(an0.get("n_virt_pw", 0))
    else:
        v11_pw_size = max(sys_c0["n_up"], sys_c0["n_down"], 1) + int(an0.get("n_virt_pw", 0))
    v11_real_basis = enumerate_real_pw_basis_2d(v11_pw_size, L)

    # Sample κ values for the training rounds (Halton-like; deterministic)
    rng = np.random.default_rng(args.seed)
    kappas = rng.uniform(-0.5, 0.5, size=(args.n_rounds, 2))
    np.save(out_dir / "training_kappas.npy", kappas)
    print(f"[training plan] {args.n_rounds} rounds × "
          f"{args.steps_per_round} SR steps/round, "
          f"κ values: {[tuple(k.round(3)) for k in kappas]}")

    current_params_flat = None      # will be set after first round transfer
    current_walkers = None

    e_history = []  # (round_idx, κ, last E_per_e)
    timestamp_start = time.time()

    for round_idx in range(args.n_rounds):
        κ = tuple(kappas[round_idx])
        t_round_start = time.time()
        print(f"\n[round {round_idx + 1}/{args.n_rounds}] κ = {κ}")

        # Fresh optimizer at this κ (rebuilds wf with twisted basis)
        opt_κ = _build_optimizer_at_twist(
            cfg_yaml, twist=κ,
            init_key=jax.random.key(args.seed + round_idx),
            lr_override=args.lr_override,
        )
        opt_pytree = opt_κ.unravel(opt_κ.params_flat)

        if current_params_flat is None:
            # First round: transfer V11 cos/sin → complex (κ-aware via n_int)
            print(f"  [first round] cos/sin → complex transfer from V11")
            # Discover envelope n_pw via tree leaves (avoids string indexing)
            n_pw_actual = None
            for path, leaf in jax.tree_util.tree_leaves_with_path(opt_pytree):
                if _path_to_str(path).endswith("envelope/coeff_up/value"):
                    n_pw_actual = leaf.shape[-1]; break
            assert n_pw_actual is not None, "envelope leaf not found"
            complex_basis_κ = enumerate_complex_pw_basis_2d(
                n_pw_actual, opt_κ.L_x, kappa=κ,
            )
            cu = cos_sin_to_complex_coeffs(
                v11_real_coeffs["cos_up"], v11_real_coeffs["sin_up"],
                v11_real_basis, complex_basis_κ, opt_κ.L_x,
            )
            cd = cos_sin_to_complex_coeffs(
                v11_real_coeffs["cos_dn"], v11_real_coeffs["sin_dn"],
                v11_real_basis, complex_basis_κ, opt_κ.L_x,
            )
            new_pytree = dict(opt_pytree)
            new_pytree["e"] = transfer_trained_params(
                chkpt_pytree["e"], opt_pytree["e"],
            )
            new_pytree = _set_envelope_complex(new_pytree, cu, cd)
            for k in ("s", "mag_mlp", "phase_mlp", "q0_mlp", "p0_mlp", "s_mlp"):
                if k in chkpt_pytree and k in new_pytree:
                    new_pytree[k] = chkpt_pytree[k]
            # Use opt's real-aware flatten (complex coeffs → re/im pair)
            new_flat, _ = opt_κ.flatten(new_pytree)
        else:
            # Subsequent rounds: identity transfer (same n-indexing)
            print(f"  [identity transfer] from previous round")
            prev_unravel = lambda flat: _build_optimizer_at_twist(
                cfg_yaml, twist=kappas[round_idx - 1],
                init_key=jax.random.key(args.seed + round_idx - 1),
                lr_override=args.lr_override,
            ).unravel(flat)
            src_pytree = prev_unravel(current_params_flat)
            new_pytree = _transfer_complex_to_complex(src_pytree, opt_pytree)
            new_flat, _ = opt_κ.flatten(new_pytree)
        assert new_flat.shape[0] == opt_κ.n_params
        opt_κ.params_flat = new_flat

        # Train K SR steps at this κ
        rng_key = jax.random.key(args.seed + round_idx + 100)
        train_result = opt_κ(
            rng_key,
            num_iters=args.steps_per_round,
            num_walkers=args.walkers,
            mcmc_decorr_steps=args.mcmc_decorr_steps,
            num_equil_steps=args.equil_steps,
            fname_log=str(out_dir / f"train_round{round_idx}.log"),
            verbose=1, save_every=0,
        )
        e_final = train_result["E_final_ha"]
        e_history.append((round_idx, κ, e_final))
        print(f"  E/N final = {e_final*1000:+.4f} mHa/e  "
              f"({time.time() - t_round_start:.1f}s)")
        current_params_flat = opt_κ.params_flat

        # Save chkpt at end of round
        chkpt_path = out_dir / f"round{round_idx}.chk.npz"
        np.savez(
            chkpt_path,
            params_flat=np.asarray(current_params_flat),
            kappa=np.asarray(κ),
            E_final_ha=e_final,
            n_iters_trained=args.steps_per_round,
        )
        print(f"  saved: {chkpt_path}")
        del opt_κ; jax.clear_caches()

    total_t = time.time() - timestamp_start
    summary = {
        "rounds": [{"idx": r, "kappa": list(k), "E_final_ha": e}
                   for r, k, e in e_history],
        "wall_time_min": total_t / 60,
        "final_chkpt": str(out_dir / f"round{args.n_rounds - 1}.chk.npz"),
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(f"\n[done] {args.n_rounds} rounds in {total_t/60:.1f} min")
    print(f"  summary: {out_dir / 'summary.json'}")
    print(f"  final chkpt: {summary['final_chkpt']}")


if __name__ == "__main__":
    main()
