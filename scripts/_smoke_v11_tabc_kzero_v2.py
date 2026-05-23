"""V11 TABC sanity at κ=(0,0) — transfer cos/sin → complex envelope coeffs.

Math:
    cos(k_n·r) = (exp(+i·k_n·r) + exp(-i·k_n·r)) / 2
    sin(k_n·r) = (exp(+i·k_n·r) - exp(-i·k_n·r)) / (2i)

So orb_i(r) = Σ_n  c_cos[i,n] cos(k_n·r) + c_sin[i,n] sin(k_n·r)
            = Σ_n  [(c_cos[i,n] - i·c_sin[i,n])/2] · exp(+i·k_n·r)
            + Σ_n  [(c_cos[i,n] + i·c_sin[i,n])/2] · exp(-i·k_n·r)

For k=0: cos(0)=1, sin(0)=0 → complex_coeff[k=0] = c_cos[i, n_zero].

Sanity: with this transfer at κ=(0,0), the complex env should give exactly
V11's trained wavefunction, and eval should give ≈+137.10 mHa/e (Im=0).
"""
import math, sys, yaml
import jax, jax.numpy as jnp, numpy as np
jax.config.update("jax_enable_x64", True)
sys.path.insert(0, ".")

from jax.flatten_util import ravel_pytree
from OmegaQMC.psi.nn.heg_wf import HEGPsiFormerConfig, transfer_trained_params
from OmegaQMC.psi.nn.env_periodic import (
    enumerate_real_pw_basis_2d, enumerate_complex_pw_basis_2d,
)
from OmegaQMC.qed_vmcopt_nn_heg_l5 import _QEDL5Optimizer

CHKPT = "runs/l5_weber_fig1b_v11_on/l5_weber_fig1b_v11_on.chk.npz"
CFG = "inputs/2dheg_qed/l5_weber_fig1b_v11_on.yaml"

with open(CFG) as f: cfg_yaml = yaml.safe_load(f)
sys_c = cfg_yaml["system"]; cav = cfg_yaml["cavity"]; an = cfg_yaml["ansatz"]; oc = cfg_yaml["optimize"]
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

def build_opt(twist):
    return _QEDL5Optimizer(
        cfg, jax.random.key(12345),
        lr=oc["lr"], damping=oc["sr_damping"],
        omega=cav["omega"], coupling_lambda=cav["lambda"],
        coupling_polarization=cav.get("polarization", [1.0, 0.0]),
        coupling_op=cav.get("coupling_op", "P"),
        K_max=cav["K_max"],
        phase_mlp_hidden=tuple(cav["phase_mlp_hidden"]),
        mag_mlp_hidden=tuple(cav["mag_mlp_hidden"]),
        use_matter_photon_shift=True, use_matter_photon_pshift=True,
        q0_mlp_hidden=tuple(cav.get("q0_mlp_hidden", [32, 32])),
        p0_mlp_hidden=tuple(cav.get("p0_mlp_hidden", [32, 32])),
        v_ext_amp=float(cfg_yaml.get("v_ext", {}).get("amp", 0.0)),
        v_ext_a=cfg_yaml.get("v_ext", {}).get("a", None),
        include_vee=bool(sys_c.get("include_vee", True)),
        twist=twist,
    )


def _path_str(path):
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


def extract_envelope_coeffs(pytree, keys=("cos_up", "sin_up", "cos_dn", "sin_dn")):
    """Find envelope/coeff_{cos,sin}_{up,dn}/value leaves; return dict."""
    out = {}
    for path, leaf in jax.tree_util.tree_leaves_with_path(pytree):
        s = _path_str(path)
        for k in keys:
            if f"envelope/coeff_{k}/value" in s:
                out[k] = np.asarray(leaf)
    return out


def cos_sin_to_complex_coeffs(c_cos, c_sin, real_basis, complex_kvecs, tol=1e-6):
    """Convert real-env (cos/sin) coeffs to complex-env (exp) coeffs.

    Uses real_basis.basis_idx and basis_is_sin to track which
    (k, cos/sin) entries V11 actually trained.  Untrained coeffs
    (e.g., the un-used sin entry on the partial-shell cos-only k)
    are IGNORED to avoid leaking random init into the eval ansatz.

    Inputs:
      c_cos, c_sin: shape (n_det, n_orb, n_unique_real_k) real
      real_basis: RealPWBasis (kvecs, basis_idx, basis_is_sin)
      complex_kvecs: (n_complex_pw, 2)
    Returns: (n_det, n_orb, n_complex_pw) complex
    """
    n_det, n_orb, n_unique = c_cos.shape
    n_complex_pw = complex_kvecs.shape[0]
    out = np.zeros((n_det, n_orb, n_complex_pw), dtype=np.complex128)

    real_kvecs = np.asarray(real_basis.kvecs)
    basis_idx = np.asarray(real_basis.basis_idx)
    basis_is_sin = np.asarray(real_basis.basis_is_sin)

    # Per-k activity flags
    active_cos = np.zeros(n_unique, dtype=bool)
    active_sin = np.zeros(n_unique, dtype=bool)
    for k_i, is_s in zip(basis_idx, basis_is_sin):
        if int(is_s): active_sin[int(k_i)] = True
        else:          active_cos[int(k_i)] = True

    log = {"k_zero": 0, "paired": 0, "cos_only": 0, "sin_only": 0,
           "missing_antipode": 0, "unmapped": 0}

    for n in range(n_unique):
        if not (active_cos[n] or active_sin[n]):
            continue
        kn = real_kvecs[n]
        is_zero = np.linalg.norm(kn) < tol
        plus_idx, minus_idx = None, None
        for m in range(n_complex_pw):
            if np.allclose(complex_kvecs[m], kn, atol=tol):
                plus_idx = m
            elif np.allclose(complex_kvecs[m], -kn, atol=tol):
                minus_idx = m

        if is_zero:
            # k=0 entry — cos only (sin contribution vanishes identically)
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

        # Apply cos contribution: c·cos(k·r) = c/2 · (e^{+ikr} + e^{-ikr})
        if active_cos[n]:
            if plus_idx  is not None: out[:, :, plus_idx]  += c_cos[:, :, n] / 2.0
            if minus_idx is not None: out[:, :, minus_idx] += c_cos[:, :, n] / 2.0
        # Apply sin contribution: c·sin(k·r) = c/(2i) · (e^{+ikr} - e^{-ikr})
        #                                    = -ic/2 · e^{+ikr} + ic/2 · e^{-ikr}
        if active_sin[n]:
            if plus_idx  is not None: out[:, :, plus_idx]  += -1j * c_sin[:, :, n] / 2.0
            if minus_idx is not None: out[:, :, minus_idx] += +1j * c_sin[:, :, n] / 2.0

        if   active_cos[n] and active_sin[n]: log["paired"] += 1
        elif active_cos[n]:                   log["cos_only"] += 1
        else:                                 log["sin_only"] += 1

    return out, log


def set_complex_envelope(pytree, coeff_up, coeff_dn):
    """Path-based overwrite of envelope/coeff_{up,dn}/value."""
    cu_j = jnp.asarray(coeff_up)
    cd_j = jnp.asarray(coeff_dn)
    def replace(path, leaf):
        s = _path_str(path)
        if s.endswith("envelope/coeff_up/value"): return cu_j
        if s.endswith("envelope/coeff_dn/value"): return cd_j
        return leaf
    return jax.tree_util.tree_map_with_path(replace, pytree)


# Main
chkpt = np.load(CHKPT, allow_pickle=True)
chkpt_params_flat = jnp.asarray(chkpt["params_flat"])
print(f"chkpt: {chkpt_params_flat.shape[0]} params, "
      f"Γ E/N = {float(chkpt['E_final_ha'])*1000:+.4f} mHa/e")

# 1) Build REAL backbone reference (no twist), load chkpt
ref_opt = build_opt(twist=None)
print(f"ref (real, no twist) n_params: {ref_opt.n_params}")
assert ref_opt.n_params == chkpt_params_flat.shape[0]
ref_params = ref_opt.unravel(chkpt_params_flat)
real_coeffs = extract_envelope_coeffs(
    ref_params, keys=("cos_up", "sin_up", "cos_dn", "sin_dn"),
)
print(f"extracted real env coeffs: {[(k, v.shape) for k, v in real_coeffs.items()]}")

# 2) Build COMPLEX backbone at κ=(0,0)
tabc_opt = build_opt(twist=(0.0, 0.0))
print(f"tabc (complex, κ=0) n_params: {tabc_opt.n_params}")
tabc_params = tabc_opt.unravel(tabc_opt.params_flat)

# 3) Get kvecs for both — match heg_wf_module's auto-bump for complex
real_basis = enumerate_real_pw_basis_2d(18, L)
_kv = np.asarray(real_basis.kvecs)
_n_zero = int(np.sum(np.linalg.norm(_kv, axis=1) < 1e-9))
_antipodal_closure = _n_zero + 2 * (_kv.shape[0] - _n_zero)
_CLOSED_2D = [1, 5, 9, 13, 21, 25, 29, 37, 45, 57, 61, 81, 89, 97, 109]
complex_init = max(18, next((c for c in _CLOSED_2D if c >= _antipodal_closure),
                            max(_CLOSED_2D)))
complex_basis = enumerate_complex_pw_basis_2d(complex_init, L, kappa=(0.0, 0.0))
real_kvecs = np.asarray(real_basis.kvecs)
complex_kvecs = np.asarray(complex_basis.kvecs)
print(f"real_kvecs shape: {real_kvecs.shape}  (antipodal closure → {_antipodal_closure})")
print(f"complex_kvecs shape: {complex_kvecs.shape}  (auto-bumped to shell {complex_init})")

# 4) Transfer cos/sin → complex (basis_is_sin-aware)
coeff_up_complex, log_up = cos_sin_to_complex_coeffs(
    real_coeffs["cos_up"], real_coeffs["sin_up"],
    real_basis, complex_kvecs,
)
coeff_dn_complex, log_dn = cos_sin_to_complex_coeffs(
    real_coeffs["cos_dn"], real_coeffs["sin_dn"],
    real_basis, complex_kvecs,
)
print(f"transfer log (up): {log_up}")
print(f"transfer log (dn): {log_dn}")

# 5) Inject complex coeffs + transfer other params
tabc_params_e = transfer_trained_params(ref_params["e"], tabc_params["e"])
new_pytree = dict(tabc_params)
new_pytree["e"] = tabc_params_e
for k in ("s", "mag_mlp", "phase_mlp", "q0_mlp", "p0_mlp", "s_mlp"):
    if k in ref_params and k in new_pytree:
        new_pytree[k] = ref_params[k]
# Now override envelope coeffs with our cos/sin → complex transform
new_pytree = set_complex_envelope(new_pytree, coeff_up_complex, coeff_dn_complex)

new_flat, _ = tabc_opt.flatten(new_pytree)   # real-aware
print(f"new_flat shape = {new_flat.shape}, tabc_opt.n_params = {tabc_opt.n_params}")

# DIAG: Confirm complex coeffs were actually applied
cu_after = None
for path, leaf in jax.tree_util.tree_leaves_with_path(new_pytree):
    if _path_str(path).endswith("envelope/coeff_up/value"):
        cu_after = np.asarray(leaf); break
print(f"DIAG cu_after sum abs = {np.sum(np.abs(cu_after)):.4f}, "
      f"transferred sum abs = {np.sum(np.abs(coeff_up_complex)):.4f}, "
      f"match = {np.allclose(cu_after, coeff_up_complex)}")
print(f"DIAG cu_after[0,0,:5] = {cu_after[0,0,:5]}")
print(f"DIAG coeff_up_complex[0,0,:5] = {coeff_up_complex[0,0,:5]}")

# DIAG: also check coeff_dn
cd_after = None
for path, leaf in jax.tree_util.tree_leaves_with_path(new_pytree):
    if _path_str(path).endswith("envelope/coeff_dn/value"):
        cd_after = np.asarray(leaf); break
print(f"DIAG cd_after sum abs = {np.sum(np.abs(cd_after)):.4f}, "
      f"transferred sum abs = {np.sum(np.abs(coeff_dn_complex)):.4f}, "
      f"match = {np.allclose(cd_after, coeff_dn_complex)}")

# DIAG: list ALL non-envelope params and check they were transferred
ref_leaves = {_path_str(p): np.asarray(l)
              for p, l in jax.tree_util.tree_leaves_with_path(ref_params)}
new_leaves = {_path_str(p): np.asarray(l)
              for p, l in jax.tree_util.tree_leaves_with_path(new_pytree)}
print(f"\nDIAG ref_pytree leaves: {len(ref_leaves)}, new_pytree leaves: {len(new_leaves)}")

mismatches = []
common_keys = set(ref_leaves.keys()) & set(new_leaves.keys())
for k in sorted(common_keys):
    if "envelope/coeff" in k or "kvecs" in k or "k_sq" in k or "kappa_vec" in k:
        continue  # envelope already handled
    r_arr, n_arr = ref_leaves[k], new_leaves[k]
    if r_arr.shape != n_arr.shape:
        mismatches.append(f"  shape mismatch {k}: ref={r_arr.shape} new={n_arr.shape}")
    elif not np.allclose(r_arr, n_arr, atol=1e-10):
        mismatches.append(f"  value mismatch {k}: max_diff={np.max(np.abs(r_arr - n_arr)):.3e}")

ref_only = sorted(set(ref_leaves.keys()) - set(new_leaves.keys()))
new_only = sorted(set(new_leaves.keys()) - set(ref_leaves.keys()))
print(f"DIAG ref-only leaves ({len(ref_only)}):")
for k in ref_only[:20]: print(f"    {k}")
print(f"DIAG new-only leaves ({len(new_only)}):")
for k in new_only[:20]: print(f"    {k}")
print(f"DIAG mismatched (non-envelope) leaves ({len(mismatches)}):")
for m in mismatches[:20]: print(m)

# 6) Eval
stats = tabc_opt.evaluate(
    jax.random.key(7777),
    params_flat=new_flat,
    num_walkers=512,
    num_blocks=20,
    num_blocks_equil=5,
    num_steps_per_block=10,
    mc_timestep_R=0.1, mc_timestep_qc=0.5,
    verbose=0,
)
e_mha = stats["E_per_elec_ha"] * 1000
sem_mha = stats["E_serr_per_e_ha"] * 1000
im_mha = stats["Im_per_e_ha"] * 1000
print(f"\n=== RESULT ===")
print(f"  E/N      = {e_mha:+.4f} ± {sem_mha:.4f} mHa/e")
print(f"  Im/N     = {im_mha:+.4e} mHa/e (Hermiticity)")
print(f"  Γ ref    = +137.10 mHa/e")
print(f"  diff     = {e_mha - 137.10:+.4f} mHa/e")
match = abs(e_mha - 137.10) < max(5 * sem_mha, 1.0)
print(f"  match?   {'✓ YES' if match else '✗ NO'} (tol = max(5σ, 1.0 mHa/e))")
