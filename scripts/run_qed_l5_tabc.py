"""Twist-averaged BC evaluation for an existing L7 (V11) chkpt.

For a trained Γ-point chkpt, loops over quasi-random twists κ and at
each one (a) builds a fresh L7 optimizer whose matter trunk has the
complex envelope at twist κ, (b) transfers all Γ-trained parameters
except the envelope into the new optimizer, (c) runs a non-gradient
eval block, (d) collects E_κ.  Averages over twists at the end.

Matches Weber et al. 2024 (arXiv:2412.19222) TABC recipe: Halton-
sequence-quasirandom κ ∈ [-1/2, 1/2]^dim, 60 twists by default.

Usage:
    python scripts/run_qed_l5_tabc.py inputs/2dheg_qed/<config>.yaml \\
        --chkpt runs/<project>/<project>.chk.npz \\
        [--n-twists 60] [--walkers 1024] [--blocks 20]
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

from OmegaQMC.psi.nn.heg_wf import (
    HEGPsiFormerConfig, transfer_trained_params,
)
from OmegaQMC.psi.nn.env_periodic import (
    enumerate_complex_pw_basis_2d, enumerate_real_pw_basis_2d,
)
from OmegaQMC.qed_vmcopt_nn_heg_l5 import _QEDL5Optimizer


def _path_to_str(path):
    parts = []
    for p in path:
        for attr in ("key", "name", "idx"):
            v = getattr(p, attr, None)
            if v is not None:
                parts.append(str(v))
                break
        else:
            parts.append(str(p))
    return "/".join(parts)


def _extract_v11_real_envelope(chkpt_pytree):
    """Extract V11's coeff_cos_up/sin_up/cos_dn/sin_dn arrays from
    the Γ-point chkpt pytree."""
    coeffs = {}
    for path, leaf in jax.tree_util.tree_leaves_with_path(chkpt_pytree):
        s = _path_to_str(path)
        for key in ("cos_up", "sin_up", "cos_dn", "sin_dn"):
            if s.endswith(f"envelope/coeff_{key}/value"):
                coeffs[key] = np.asarray(leaf)
    return coeffs


def transfer_real_to_complex_envelope_coeffs(
    c_cos, c_sin, real_basis, complex_basis, L, tol=1e-6,
):
    """Convert V11 (cos/sin) coeffs to complex-env (exp) coeffs at any κ.

    Matches by integer n-vector (n_int = round(kvec * L / (2π))), so it
    works at both κ=0 and κ≠0.  The cos/sin → exp transform is exact:

        c·cos(k·r) = (c/2) exp(+ikr) + (c/2) exp(-ikr)
        c·sin(k·r) = (-ic/2) exp(+ikr) + (+ic/2) exp(-ikr)

    Only basis entries that V11 actually trained (per basis_is_sin) are
    transferred — untrained partial-shell entries (cos-only at the
    Fermi-edge shell) are ignored to avoid leaking random init.
    """
    n_det, n_orb, n_unique = c_cos.shape
    real_kvecs = np.asarray(real_basis.kvecs)
    complex_kvecs = np.asarray(complex_basis.kvecs)
    n_complex_pw = complex_kvecs.shape[0]
    # Integer n-vectors for matching at arbitrary κ
    dk = 2.0 * np.pi / L
    real_n_int = np.round(real_kvecs / dk).astype(np.int32)
    complex_n_int = np.asarray(complex_basis.n_ints)  # already int
    # Per-k activity flags from V11's real basis
    basis_idx = np.asarray(real_basis.basis_idx)
    basis_is_sin = np.asarray(real_basis.basis_is_sin)
    active_cos = np.zeros(n_unique, dtype=bool)
    active_sin = np.zeros(n_unique, dtype=bool)
    for k_i, is_s in zip(basis_idx, basis_is_sin):
        if int(is_s): active_sin[int(k_i)] = True
        else:          active_cos[int(k_i)] = True
    out = np.zeros((n_det, n_orb, n_complex_pw), dtype=np.complex128)
    log = {"k_zero": 0, "paired": 0, "cos_only": 0, "sin_only": 0,
           "missing_antipode": 0, "unmapped": 0}
    for n in range(n_unique):
        if not (active_cos[n] or active_sin[n]):
            continue
        n_int = real_n_int[n]
        is_zero = bool(np.all(n_int == 0))
        plus_idx, minus_idx = None, None
        for m in range(n_complex_pw):
            if np.array_equal(complex_n_int[m], n_int):
                plus_idx = m
            elif np.array_equal(complex_n_int[m], -n_int):
                minus_idx = m
        if is_zero:
            if plus_idx is None:
                log["unmapped"] += 1
                continue
            if active_cos[n]:
                out[:, :, plus_idx] += c_cos[:, :, n].astype(np.complex128)
                log["k_zero"] += 1
            continue
        if plus_idx is None and minus_idx is None:
            log["unmapped"] += 1
            continue
        if plus_idx is None or minus_idx is None:
            log["missing_antipode"] += 1
        if active_cos[n]:
            if plus_idx  is not None: out[:, :, plus_idx]  += c_cos[:, :, n] / 2.0
            if minus_idx is not None: out[:, :, minus_idx] += c_cos[:, :, n] / 2.0
        if active_sin[n]:
            if plus_idx  is not None: out[:, :, plus_idx]  += -1j * c_sin[:, :, n] / 2.0
            if minus_idx is not None: out[:, :, minus_idx] += +1j * c_sin[:, :, n] / 2.0
        if   active_cos[n] and active_sin[n]: log["paired"] += 1
        elif active_cos[n]:                   log["cos_only"] += 1
        else:                                 log["sin_only"] += 1
    return out, log


def _set_complex_envelope_coeffs(pytree, coeff_up, coeff_dn):
    """Overwrite envelope/coeff_up/value and envelope/coeff_dn/value."""
    cu_j = jnp.asarray(coeff_up)
    cd_j = jnp.asarray(coeff_dn)
    def replace(path, leaf):
        s = _path_to_str(path)
        if s.endswith("envelope/coeff_up/value"): return cu_j
        if s.endswith("envelope/coeff_dn/value"): return cd_j
        return leaf
    return jax.tree_util.tree_map_with_path(replace, pytree)


# ---------------------------------------------------------------------
# HF init of the complex envelope at twist κ
# ---------------------------------------------------------------------

def hf_orbitals_2d(L, kappa, v_ext_amp, v_ext_a, n_pw, n_orbitals):
    """Diagonalize T + v_ext on the κ-shifted PW basis (2D).

    For non-interacting matter under cosine v_ext, returns the lowest
    n_orbitals HF eigenvectors as columns of a (n_pw, n_orbitals)
    complex matrix.  These are the optimal orbital coefficients at
    twist κ — they replace the identity-init of the fresh complex
    envelope so the κ=0 limit recovers V11's trained Γ wavefunction.

    Args:
      L: cell side length (square cell)
      kappa: 2-tuple of fractional twist in [-0.5, 0.5)
      v_ext_amp, v_ext_a: external potential params (V_ext = -amp·Σ_d cos(2π r_d/a))
      n_pw: number of PWs in the basis (must match envelope's n_pw)
      n_orbitals: number of orbitals to return (= n_up or n_down)
    """
    basis = enumerate_complex_pw_basis_2d(n_pw, L, kappa=tuple(kappa))
    k_sq = np.asarray(basis.k_sq)            # (n_pw,)
    n_ints = np.asarray(basis.n_ints)        # (n_pw, 2) — integer indices into 2π/L grid

    # Kinetic-energy diagonal
    H = np.diag(k_sq / 2.0).astype(np.complex128)

    # v_ext = -amp · Σ_d cos(2π r_d / a) has nonzero matrix elements
    # between PWs differing by Δn = ±(L/a)·ê_d.  We require L/a integer.
    dn_step_float = L / v_ext_a
    dn_step = int(round(dn_step_float))
    # Allow small float roundoff in L (e.g., rs·√(πN) gives L=12.0002 not 12.0
    # exactly); v_ext is meant to be lattice-periodic with period a, so we
    # round to the nearest integer step.  Real physical mismatches (>0.1%)
    # should still raise.
    if abs(dn_step_float - dn_step) > 1e-3:
        raise ValueError(
            f"L/a must be (approximately) integer; got L={L}, a={v_ext_a}, "
            f"ratio={dn_step_float}, rounded step={dn_step}"
        )
    n_lookup = {tuple(n_ints[i].tolist()): i for i in range(n_pw)}
    for i in range(n_pw):
        for d in range(2):
            for sign in (+1, -1):
                target = list(n_ints[i].tolist())
                target[d] += sign * dn_step
                j = n_lookup.get(tuple(target))
                if j is not None:
                    H[i, j] += -v_ext_amp / 2.0

    eigvals, eigvecs = np.linalg.eigh(H)
    # Return lowest n_orbitals eigenvectors (each as a column).
    return eigvecs[:, :n_orbitals]   # complex (n_pw, n_orbitals)


# ---------------------------------------------------------------------
# Halton-sequence twists in [-1/2, 1/2]^dim
# ---------------------------------------------------------------------

def _halton_sequence(n_points, dim, skip_first=1):
    """Halton low-discrepancy sequence in [0,1]^dim.  Shifted to [-1/2, 1/2]."""
    primes = [2, 3, 5, 7, 11, 13]
    def halton_1d(i, base):
        f = 1.0
        r = 0.0
        while i > 0:
            f /= base
            r += f * (i % base)
            i //= base
        return r
    out = np.zeros((n_points, dim), dtype=np.float64)
    for i in range(n_points):
        for d in range(dim):
            out[i, d] = halton_1d(i + skip_first, primes[d])
    return out - 0.5    # shift to [-0.5, 0.5]


# ---------------------------------------------------------------------
# Load + build at twist
# ---------------------------------------------------------------------

def _build_optimizer_at_twist(cfg_yaml, twist, init_key):
    """Build an L7 optimizer at twist κ, ready for eval (no training)."""
    sys_c = cfg_yaml["system"]; cav = cfg_yaml["cavity"]
    an = cfg_yaml["ansatz"]; oc = cfg_yaml["optimize"]
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
        lr=oc["lr"], damping=oc["sr_damping"], n_cg=oc.get("sr_n_cg", 20),
        omega=cav["omega"], coupling_lambda=cav["lambda"],
        coupling_polarization=cav.get("polarization", [1.0, 0.0]),
        coupling_op=cav.get("coupling_op", "P"),
        K_max=cav["K_max"],
        phase_mlp_hidden=tuple(cav["phase_mlp_hidden"]),
        mag_mlp_hidden=tuple(cav["mag_mlp_hidden"]),
        activation=cav.get("activation", "tanh"),
        use_smw_sr=oc.get("use_smw_sr", True),
        use_fused_step=oc.get("use_fused_step", True),
        spring_mu=oc.get("spring_mu", 0.0),
        spring_norm_clip=oc.get("spring_norm_clip", 1e10),
        use_matter_photon_shift=bool(cav.get("use_matter_photon_shift", False)),
        use_matter_photon_width=bool(cav.get("use_matter_photon_width", False)),
        use_matter_photon_pshift=bool(cav.get("use_matter_photon_pshift", False)),
        q0_mlp_hidden=tuple(cav.get("q0_mlp_hidden", [32, 32])),
        s_mlp_hidden=tuple(cav.get("s_mlp_hidden", [32, 32])),
        p0_mlp_hidden=tuple(cav.get("p0_mlp_hidden", [32, 32])),
        v_ext_amp=float(cfg_yaml.get("v_ext", {}).get("amp", 0.0)),
        v_ext_a=cfg_yaml.get("v_ext", {}).get("a", None),
        include_vee=bool(sys_c.get("include_vee", True)),
        twist=twist,
    )


def _transfer_chkpt_params(src_pytree, dst_pytree):
    """Move Γ-trained params into fresh-at-twist pytree.

    Matter subtree ``e``: use transfer_trained_params (skips envelope).
    All other keys (s, mag_mlp, phase_mlp, q0_mlp, p0_mlp, s_mlp):
    direct copy when present in src.
    """
    new = dict(dst_pytree)
    if "e" in src_pytree and "e" in new:
        new["e"] = transfer_trained_params(src_pytree["e"], new["e"])
    for k in ("s", "mag_mlp", "phase_mlp", "q0_mlp", "p0_mlp", "s_mlp"):
        if k in src_pytree and k in new:
            new[k] = src_pytree[k]
    return new


def _set_hf_envelope_coefficients(pytree, hf_coeffs, n_det):
    """Overwrite envelope coeff_up/coeff_dn in a fresh-at-twist pytree.

    Leaf paths to replace (per the inspect smoke):
      e/envelope/coeff_up/value : shape (n_det, n_up, n_pw)
      e/envelope/coeff_dn/value : same

    hf_coeffs has shape (n_pw, n_orbitals) — columns are eigenvectors.
    For each det idx, we set coeff[d, i, m] = hf_coeffs[m, i].
    Same coefficients for spin-up and spin-down (closed-shell, spin-symmetric).
    Uses tree_map_with_path because nnx.State doesn't support dict-style
    nested assignment.
    """
    new_coeff = np.broadcast_to(
        hf_coeffs.T[None, :, :],
        (n_det,) + hf_coeffs.T.shape,
    ).astype(np.complex128)
    new_coeff_j = jnp.asarray(new_coeff)

    def _path_to_str(path):
        parts = []
        for p in path:
            for attr in ("key", "name", "idx"):
                v = getattr(p, attr, None)
                if v is not None:
                    parts.append(str(v))
                    break
            else:
                parts.append(str(p))
        return "/".join(parts)

    def replace(path, leaf):
        s = _path_to_str(path)
        if s.endswith("envelope/coeff_up/value") or s.endswith("envelope/coeff_dn/value"):
            return new_coeff_j
        return leaf

    return jax.tree_util.tree_map_with_path(replace, pytree)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("config", type=str, help="YAML config used for training")
    p.add_argument("--chkpt", type=str, required=True,
                   help="path to .chk.npz of trained Γ-point V11 run")
    p.add_argument("--n-twists", type=int, default=60)
    p.add_argument("--twist-start", type=int, default=None,
                   help="(optional) start index in the Halton sequence "
                        "(parallel chunks)")
    p.add_argument("--twist-end", type=int, default=None,
                   help="(optional) end index (exclusive) in the Halton "
                        "sequence")
    p.add_argument("--chunk-tag", type=str, default=None,
                   help="(optional) tag appended to per-chunk output dir "
                        "for parallel runs, e.g. 'c0' .. 'c5'")
    p.add_argument("--walkers", type=int, default=1024)
    p.add_argument("--blocks", type=int, default=20,
                   help="eval blocks per twist (Weber uses few; with NN trial maybe more)")
    p.add_argument("--equil-blocks", type=int, default=5)
    p.add_argument("--steps-per-block", type=int, default=10)
    p.add_argument("--mc-timestep-R", type=float, default=0.1)
    p.add_argument("--mc-timestep-qc", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--out-dir", type=str, default=None)
    args = p.parse_args()

    with open(args.config) as f:
        cfg_yaml = yaml.safe_load(f)

    project = cfg_yaml.get("project", "v11_run") + "_tabc"
    if args.chunk_tag:
        project = f"{project}_{args.chunk_tag}"
    out_dir = Path(args.out_dir) if args.out_dir else Path("runs") / project
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "config_tabc.yaml", "w") as f:
        yaml.dump({**cfg_yaml, "tabc": vars(args)}, f, sort_keys=False)

    # Load Γ-trained chkpt
    chkpt = np.load(args.chkpt, allow_pickle=True)
    chkpt_params_flat = jnp.asarray(chkpt["params_flat"])
    print(f"[load] {args.chkpt}: {chkpt_params_flat.shape[0]} params, "
          f"iters={int(chkpt['n_iters_trained'])}, "
          f"E_final={float(chkpt['E_final_ha'])*1000:+.4f} mHa/e")

    # Generate quasi-random twists (2D; first dim sometimes called κ_x, κ_y).
    # Weber says Halton starting after skip; using skip=1 to avoid (0,0).
    # 2D twists in [-1/2, 1/2]^2 — passed directly to the 2D
    # complex-envelope builder (which expects shape (2,)).
    twists_all = _halton_sequence(args.n_twists, 2, skip_first=1)
    # Optional per-chunk slice for parallel runs
    t_start_idx = args.twist_start if args.twist_start is not None else 0
    t_end_idx = args.twist_end if args.twist_end is not None else args.n_twists
    twists = twists_all[t_start_idx:t_end_idx]
    print(f"[chunk] twists [{t_start_idx}:{t_end_idx}] "
          f"of {args.n_twists} (n={len(twists)})")
    np.save(out_dir / "twists.npy", twists)
    np.save(out_dir / "twist_indices.npy",
            np.arange(t_start_idx, t_end_idx))

    # Pre-build reference Γ optimizer once to verify chkpt shape.
    ref_opt = _build_optimizer_at_twist(cfg_yaml, twist=None,
                                         init_key=jax.random.key(args.seed))
    assert ref_opt.n_params == chkpt_params_flat.shape[0], (
        f"chkpt n_params {chkpt_params_flat.shape[0]} != "
        f"reconstructed Γ n_params {ref_opt.n_params}"
    )
    chkpt_pytree = ref_opt.unravel(chkpt_params_flat)
    print(f"[load] chkpt pytree keys: {list(chkpt_pytree.keys())}")

    # Extract V11 cos/sin coefficients ONCE — same for every twist.
    v11_real_coeffs = _extract_v11_real_envelope(chkpt_pytree)
    print(f"[load] V11 real env coeffs: "
          f"{[(k, v.shape) for k, v in v11_real_coeffs.items()]}")
    # Reference real basis (κ=0) — pw_basis_size matches V11's training
    sys_c0 = cfg_yaml["system"]
    an0 = cfg_yaml["ansatz"]
    if an0.get("full_determinant", False):
        v11_pw_size = (sys_c0["n_up"] + sys_c0["n_down"]) + int(an0.get("n_virt_pw", 0))
    else:
        v11_pw_size = max(sys_c0["n_up"], sys_c0["n_down"], 1) + int(an0.get("n_virt_pw", 0))
    L_ref = ref_opt.L_x
    v11_real_basis = enumerate_real_pw_basis_2d(v11_pw_size, L_ref)
    print(f"[load] V11 real basis: {v11_pw_size} cos/sin entries → "
          f"{v11_real_basis.kvecs.shape[0]} unique k-vectors")
    del ref_opt

    # ---- Twist loop ----
    log_path = out_dir / "tabc.log"
    e_per_twist = []
    sem_per_twist = []
    chkpt_e = float(chkpt["E_final_ha"])
    print(f"\n[TABC] {args.n_twists} twists × {args.blocks} blocks × "
          f"{args.steps_per_block} steps × {args.walkers} walkers")
    print(f"  (Γ reference E/N = {chkpt_e*1000:+.4f} mHa/e)")
    print(f"  output: {log_path}")
    with open(log_path, "w") as fout:
        print(f"# TABC eval — chkpt: {args.chkpt}", file=fout)
        print(f"# twists: {args.n_twists} (Halton)", file=fout)
        print(f"# per-twist: {args.blocks} blk × {args.steps_per_block} st × {args.walkers} w",
              file=fout)
        print(f"# twist_idx  kx  ky  E_per_e_ha  SEM_per_e_ha  Im_per_e_ha", file=fout)
        fout.flush()
        t_start = time.time()
        for local_i, κ in enumerate(twists):
            # Use GLOBAL twist index for log printing + RNG, so chunks
            # are reproducible and aggregate cleanly.
            i = t_start_idx + local_i
            t0 = time.time()
            # Fresh optimizer at this twist
            opt_κ = _build_optimizer_at_twist(
                cfg_yaml, twist=tuple(κ),
                init_key=jax.random.key(args.seed + i),
            )
            # Transfer chkpt params (skip envelope; envelope handled below)
            new_pytree = _transfer_chkpt_params(
                chkpt_pytree, opt_κ.unravel(opt_κ.params_flat),
            )
            # Convert V11's cos/sin (real) coefficients into the κ-shifted
            # complex envelope basis.  Matching is by integer n-vector, so
            # this is exact for κ=0 and consistent for κ≠0 (Bloch-state
            # ansatz: same coefficients, shifted PW basis).
            n_pw_actual = None
            for path, leaf in jax.tree_util.tree_leaves_with_path(new_pytree):
                if _path_to_str(path).endswith("envelope/coeff_up/value"):
                    n_pw_actual = leaf.shape[-1]; break
            assert n_pw_actual is not None, "envelope leaf not found"
            complex_basis_κ = enumerate_complex_pw_basis_2d(
                n_pw_actual, opt_κ.L_x, kappa=tuple(κ),
            )
            coeff_up_c, log_up = transfer_real_to_complex_envelope_coeffs(
                v11_real_coeffs["cos_up"], v11_real_coeffs["sin_up"],
                v11_real_basis, complex_basis_κ, opt_κ.L_x,
            )
            coeff_dn_c, log_dn = transfer_real_to_complex_envelope_coeffs(
                v11_real_coeffs["cos_dn"], v11_real_coeffs["sin_dn"],
                v11_real_basis, complex_basis_κ, opt_κ.L_x,
            )
            new_pytree = _set_complex_envelope_coeffs(
                new_pytree, coeff_up_c, coeff_dn_c,
            )
            # opt_κ.flatten is the optimizer's real-aware flatten that
            # matches its unravel (complex envelope coeffs → re/im).
            new_params_flat, _ = opt_κ.flatten(new_pytree)
            assert new_params_flat.shape[0] == opt_κ.n_params
            if local_i == 0:
                print(f"  [twist {i} (chunk 0) transfer "
                      f"up={log_up} dn={log_dn}]")

            # Eval
            stats = opt_κ.evaluate(
                jax.random.key(args.seed + i + 100000),
                params_flat=new_params_flat,
                num_walkers=args.walkers,
                num_blocks=args.blocks,
                num_blocks_equil=args.equil_blocks,
                num_steps_per_block=args.steps_per_block,
                mc_timestep_R=args.mc_timestep_R,
                mc_timestep_qc=args.mc_timestep_qc,
                verbose=0,
            )
            e_per_e = stats["E_per_elec_ha"]
            sem_per_e = stats["E_serr_per_e_ha"]
            im_per_e = stats["Im_per_e_ha"]
            e_per_twist.append(e_per_e)
            sem_per_twist.append(sem_per_e)
            dt = time.time() - t0
            line = (f"{i:4d}  {float(κ[0]):+.5f}  {float(κ[1]):+.5f}  "
                    f"{e_per_e:+.6e}  {sem_per_e:.3e}  {im_per_e:+.3e}"
                    f"   ({dt:.0f}s)")
            print(line, file=fout)
            fout.flush()
            # also to stdout, sparsely (use local index for chunk-relative)
            if local_i < 5 or (local_i + 1) % 5 == 0:
                print(line)
            # free
            del opt_κ
            jax.clear_caches()

        e_arr = np.asarray(e_per_twist)
        sem_arr = np.asarray(sem_per_twist)
        mean_E = float(e_arr.mean())
        sem_twist = float(e_arr.std(ddof=1) / np.sqrt(len(e_arr)))
        total_t = time.time() - t_start
        print(f"\n# TABC summary  ({total_t/60:.1f} min)", file=fout)
        print(f"# n_twists           = {len(e_arr)}", file=fout)
        print(f"# Γ-only E/N         = {chkpt_e*1000:+.4f} mHa/e", file=fout)
        print(f"# twist-averaged E/N = {mean_E*1000:+.4f} ± "
              f"{sem_twist*1000:.4f} mHa/e", file=fout)
        print(f"# per-twist mean SEM = {sem_arr.mean()*1000:.4f} mHa/e", file=fout)

    print(f"\n=== TABC SUMMARY ===")
    print(f"  Γ-only E/N           = {chkpt_e*1000:+.4f} mHa/e")
    print(f"  Twist-avg E/N        = {mean_E*1000:+.4f} ± {sem_twist*1000:.4f} mHa/e "
          f"({len(e_arr)} twists)")
    print(f"  Per-twist mean SEM   = {sem_arr.mean()*1000:.4f} mHa/e")
    print(f"  Total wall time      = {total_t/60:.1f} min")

    # Save summary JSON
    with open(out_dir / "tabc_summary.json", "w") as f:
        json.dump({
            "chkpt_path": args.chkpt,
            "chkpt_E_per_e_ha": chkpt_e,
            "n_twists": len(e_arr),
            "twists": twists.tolist(),
            "per_twist_E_per_e_ha": e_arr.tolist(),
            "per_twist_SEM_ha": sem_arr.tolist(),
            "twist_averaged_E_per_e_ha": mean_E,
            "twist_avg_SEM_ha": sem_twist,
            "wall_time_min": total_t / 60,
            "config_yaml": args.config,
            "n_walkers": args.walkers,
            "n_blocks": args.blocks,
        }, f, indent=2, default=float)
    print(f"  summary: {out_dir / 'tabc_summary.json'}")


if __name__ == "__main__":
    main()
