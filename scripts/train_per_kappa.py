"""Per-κ fine-tuning: warm-start V11 at each Halton twist, train K SR
iters, save independent chkpt + final energy.

This is the variationally-best TABC: at each κ_h, find the optimal
wavefunction (warm-started from V11), then average ⟨E_κ⟩ over twists.

Unlike train_kappa_sample_mvp.py (which carries state across κs and
suffers κ-specialization), each κ here trains from V11 INDEPENDENTLY.

Supports a contiguous Halton-index range per process for parallel runs.

Usage:
    python scripts/train_per_kappa.py inputs/.../v11.yaml \\
        --chkpt runs/v11/v11.chk.npz \\
        --n-twists 60 --twist-start 0 --twist-end 10 \\
        --iters 100 --chunk-tag c0 \\
        --out runs/v11_perk
"""
from __future__ import annotations
import argparse, json, math, sys, time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import yaml

jax.config.update("jax_enable_x64", True)
sys.path.insert(0, ".")

from OmegaQMC.psi.nn.heg_wf import HEGPsiFormerConfig, transfer_trained_params
from OmegaQMC.psi.nn.env_periodic import (
    enumerate_complex_pw_basis_2d, enumerate_real_pw_basis_2d,
)
from OmegaQMC.qed_vmcopt_nn_heg_l5 import _QEDL5Optimizer


def _path_to_str(path):
    parts = []
    for p in path:
        for attr in ("key", "name", "idx"):
            v = getattr(p, attr, None)
            if v is not None: parts.append(str(v)); break
        else: parts.append(str(p))
    return "/".join(parts)


def _extract_envelope_coeffs_real(chkpt_pytree):
    out = {}
    for path, leaf in jax.tree_util.tree_leaves_with_path(chkpt_pytree):
        s = _path_to_str(path)
        for k in ("cos_up", "sin_up", "cos_dn", "sin_dn"):
            if s.endswith(f"envelope/coeff_{k}/value"):
                out[k] = np.asarray(leaf)
    return out


def cos_sin_to_complex_coeffs(c_cos, c_sin, real_basis, complex_basis, L, tol=1e-6):
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


def _halton_sequence(n_points, dim, skip_first=1):
    primes = [2, 3, 5, 7, 11, 13]
    def halton_1d(i, base):
        f, r = 1.0, 0.0
        while i > 0:
            f /= base; r += f * (i % base); i //= base
        return r
    out = np.zeros((n_points, dim), dtype=np.float64)
    for i in range(n_points):
        for d in range(dim):
            out[i, d] = halton_1d(i + skip_first, primes[d])
    return out - 0.5


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
        lr=lr_use, damping=oc["sr_damping"], n_cg=oc.get("sr_n_cg", 20),
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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("config", type=str, help="V11 YAML config")
    p.add_argument("--chkpt", type=str, required=True,
                   help="V11 Γ-trained chkpt (used as warm-start for every κ)")
    p.add_argument("--n-twists", type=int, default=60)
    p.add_argument("--twist-start", type=int, default=0)
    p.add_argument("--twist-end", type=int, default=None,
                   help="exclusive end; defaults to n-twists")
    p.add_argument("--iters", type=int, default=100,
                   help="SR fine-tune iters per κ")
    p.add_argument("--equil-steps", type=int, default=30)
    p.add_argument("--walkers", type=int, default=1024)
    p.add_argument("--mcmc-decorr-steps", type=int, default=20)
    p.add_argument("--lr-override", type=float, default=None,
                   help="if unset, uses YAML's optimize.lr (V11=0.005)")
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--chunk-tag", type=str, default="")
    p.add_argument("--out", type=str, required=True)
    args = p.parse_args()

    twist_end = args.twist_end if args.twist_end is not None else args.n_twists
    out_dir = Path(args.out)
    if args.chunk_tag:
        out_dir = Path(args.out + "_" + args.chunk_tag)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.config) as f:
        cfg_yaml = yaml.safe_load(f)

    chkpt = np.load(args.chkpt, allow_pickle=True)
    chkpt_params_flat = jnp.asarray(chkpt["params_flat"])
    print(f"[load] {args.chkpt}: {chkpt_params_flat.shape[0]} params, "
          f"E_final = {float(chkpt['E_final_ha'])*1000:+.4f} mHa/e")

    # Reference real-Γ opt to unravel chkpt + extract V11 cos/sin coefs
    ref_opt = _build_optimizer_at_twist(
        cfg_yaml, twist=None, init_key=jax.random.key(args.seed),
        lr_override=args.lr_override,
    )
    assert ref_opt.n_params == chkpt_params_flat.shape[0]
    chkpt_pytree = ref_opt.unravel(chkpt_params_flat)
    v11_real_coeffs = _extract_envelope_coeffs_real(chkpt_pytree)

    sys_c0 = cfg_yaml["system"]; an0 = cfg_yaml["ansatz"]
    N = sys_c0["n_up"] + sys_c0["n_down"]
    L = sys_c0["rs"] * math.sqrt(math.pi * N)
    if an0.get("full_determinant", False):
        v11_pw_size = (sys_c0["n_up"] + sys_c0["n_down"]) + int(an0.get("n_virt_pw", 0))
    else:
        v11_pw_size = max(sys_c0["n_up"], sys_c0["n_down"], 1) + int(an0.get("n_virt_pw", 0))
    v11_real_basis = enumerate_real_pw_basis_2d(v11_pw_size, L)

    twists_all = _halton_sequence(args.n_twists, 2, skip_first=1)
    twists = twists_all[args.twist_start:twist_end]
    np.save(out_dir / "twists.npy", twists)
    np.save(out_dir / "twist_indices.npy",
            np.arange(args.twist_start, twist_end))
    print(f"[chunk] twists [{args.twist_start}:{twist_end}] "
          f"of {args.n_twists}; iters/κ = {args.iters}, "
          f"lr = {args.lr_override if args.lr_override is not None else cfg_yaml['optimize']['lr']}")

    del ref_opt

    results = []
    t_start = time.time()

    for local_i, κ in enumerate(twists):
        idx_global = args.twist_start + local_i
        t0 = time.time()
        print(f"\n[twist {idx_global} ({local_i + 1}/{len(twists)})] "
              f"κ = ({float(κ[0]):+.4f}, {float(κ[1]):+.4f})")

        opt_κ = _build_optimizer_at_twist(
            cfg_yaml, twist=tuple(κ),
            init_key=jax.random.key(args.seed + idx_global),
            lr_override=args.lr_override,
        )
        opt_pytree = opt_κ.unravel(opt_κ.params_flat)

        # Discover envelope n_pw + build basis at this κ
        n_pw_actual = None
        for path, leaf in jax.tree_util.tree_leaves_with_path(opt_pytree):
            if _path_to_str(path).endswith("envelope/coeff_up/value"):
                n_pw_actual = leaf.shape[-1]; break
        assert n_pw_actual is not None
        complex_basis_κ = enumerate_complex_pw_basis_2d(
            n_pw_actual, opt_κ.L_x, kappa=tuple(κ),
        )

        # Transfer V11 cos/sin → complex at this κ (warm-start every time)
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
        new_flat, _ = opt_κ.flatten(new_pytree)
        assert new_flat.shape[0] == opt_κ.n_params
        opt_κ.params_flat = new_flat

        # SR fine-tune — verbose=1 so per-iter (every 10) prints land in
        # the train log; gives live progress visibility for a long run.
        train_result = opt_κ(
            jax.random.key(args.seed + idx_global + 100000),
            num_iters=args.iters,
            num_walkers=args.walkers,
            mcmc_decorr_steps=args.mcmc_decorr_steps,
            num_equil_steps=args.equil_steps,
            fname_log=str(out_dir / f"twist{idx_global:03d}_train.log"),
            verbose=1, save_every=0,
        )
        e_final = float(train_result["E_final_ha"])
        e_history = train_result.get("E_history", [])
        e_initial = float(e_history[0]) if len(e_history) > 0 else None

        chkpt_path = out_dir / f"twist{idx_global:03d}.chk.npz"
        np.savez(
            chkpt_path,
            params_flat=np.asarray(opt_κ.params_flat),
            kappa=np.asarray(κ),
            E_final_ha=e_final,
            E_initial_ha=e_initial if e_initial else 0.0,
            n_iters_trained=args.iters,
            twist_idx=idx_global,
        )

        dt = time.time() - t0
        improv_str = (f"  Δ = {(e_final - e_initial)*1000:+.3f} mHa/e"
                      if e_initial else "")
        print(f"  E_final/N = {e_final*1000:+.4f} mHa/e{improv_str}  ({dt:.0f}s)")
        results.append({
            "twist_idx": idx_global,
            "kappa": [float(κ[0]), float(κ[1])],
            "E_initial_ha": e_initial,
            "E_final_ha": e_final,
            "wall_time_s": dt,
        })
        del opt_κ; jax.clear_caches()

    total_t = time.time() - t_start
    with open(out_dir / "per_kappa_summary.json", "w") as f:
        json.dump({
            "chkpt_source": args.chkpt,
            "n_twists_total": args.n_twists,
            "twist_range": [args.twist_start, twist_end],
            "iters_per_kappa": args.iters,
            "lr_override": args.lr_override,
            "results": results,
            "wall_time_min": total_t / 60,
        }, f, indent=2, default=float)
    print(f"\n[done] {len(results)} twists in {total_t/60:.1f} min")
    print(f"  summary: {out_dir / 'per_kappa_summary.json'}")


if __name__ == "__main__":
    main()
