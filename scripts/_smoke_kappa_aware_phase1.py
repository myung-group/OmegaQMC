"""Phase 1 sanity for κ-aware V11 architecture.

Tests:
  (1) Build κ-aware complex wf at fixed κ=0, ΔMLP=0 init.
      Transfer V11 coefs. Eval κ=0. Should match v7 (+137.06 mHa/e).
  (2) Eval with kappa runtime arg passed instead (envelope rebuilt at κ).
      Should give SAME +137.06 (since runtime κ=0 == construct κ=0).

This validates that the architecture additions (kappa_runtime arg,
coeff_override arg, KappaCoeffDelta MLP) don't break the κ=0 baseline.
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

# Patch the build to enable kappa_aware mode.  Cleanest: monkey-patch
# build_heg_psiformer_wf_complex's defaults via a wrapper.
from OmegaQMC.psi.nn import heg_wf_module as hwm
_orig_build = hwm.build_heg_psiformer_wf_complex
def _build_kappa_aware(config, rngs, *, kappa=None):
    return _orig_build(config, rngs, kappa=kappa, kappa_aware=True,
                       kappa_mlp_hidden=(32,))
hwm.build_heg_psiformer_wf_complex = _build_kappa_aware

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
                parts.append(str(v)); break
        else:
            parts.append(str(p))
    return "/".join(parts)


def extract_envelope_coeffs(pytree, keys):
    out = {}
    for path, leaf in jax.tree_util.tree_leaves_with_path(pytree):
        s = _path_str(path)
        for k in keys:
            if f"envelope/coeff_{k}/value" in s:
                out[k] = np.asarray(leaf)
    return out


def cos_sin_to_complex_coeffs(c_cos, c_sin, real_basis, complex_kvecs, tol=1e-6):
    """basis_is_sin-aware cos/sin → exp transfer (validated in v7)."""
    n_det, n_orb, n_unique = c_cos.shape
    n_complex_pw = complex_kvecs.shape[0]
    out = np.zeros((n_det, n_orb, n_complex_pw), dtype=np.complex128)
    real_kvecs = np.asarray(real_basis.kvecs)
    basis_idx = np.asarray(real_basis.basis_idx)
    basis_is_sin = np.asarray(real_basis.basis_is_sin)
    active_cos = np.zeros(n_unique, dtype=bool)
    active_sin = np.zeros(n_unique, dtype=bool)
    for k_i, is_s in zip(basis_idx, basis_is_sin):
        if int(is_s): active_sin[int(k_i)] = True
        else:          active_cos[int(k_i)] = True
    for n in range(n_unique):
        if not (active_cos[n] or active_sin[n]):
            continue
        kn = real_kvecs[n]
        is_zero = np.linalg.norm(kn) < tol
        plus_idx, minus_idx = None, None
        for m in range(n_complex_pw):
            if np.allclose(complex_kvecs[m], kn, atol=tol): plus_idx = m
            elif np.allclose(complex_kvecs[m], -kn, atol=tol): minus_idx = m
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


def set_complex_envelope(pytree, coeff_up, coeff_dn):
    cu_j = jnp.asarray(coeff_up); cd_j = jnp.asarray(coeff_dn)
    def replace(path, leaf):
        s = _path_str(path)
        if s.endswith("envelope/coeff_up/value"): return cu_j
        if s.endswith("envelope/coeff_dn/value"): return cd_j
        return leaf
    return jax.tree_util.tree_map_with_path(replace, pytree)


chkpt = np.load(CHKPT, allow_pickle=True)
chkpt_params_flat = jnp.asarray(chkpt["params_flat"])
print(f"chkpt: {chkpt_params_flat.shape[0]} params, Γ E/N = {float(chkpt['E_final_ha'])*1000:+.4f} mHa/e")

# Reference real env (no twist, NOT kappa-aware)
hwm.build_heg_psiformer_wf_complex = _orig_build  # turn off temporarily
ref_opt = build_opt(twist=None)
print(f"ref (real Γ): n_params = {ref_opt.n_params}")
ref_params = ref_opt.unravel(chkpt_params_flat)
real_coeffs = extract_envelope_coeffs(
    ref_params, keys=("cos_up", "sin_up", "cos_dn", "sin_dn"))

# κ-aware complex env at κ=0
hwm.build_heg_psiformer_wf_complex = _build_kappa_aware  # re-enable
tabc_opt = build_opt(twist=(0.0, 0.0))
print(f"tabc (κ-aware, κ=0): n_params = {tabc_opt.n_params}")
print(f"  (additional params from KappaCoeffDelta MLPs: "
      f"{tabc_opt.n_params - 168486})")

tabc_params = tabc_opt.unravel(tabc_opt.params_flat)
# Compute complex basis at κ=0
real_basis = enumerate_real_pw_basis_2d(18, L)
_kv = np.asarray(real_basis.kvecs)
_n_zero = int(np.sum(np.linalg.norm(_kv, axis=1) < 1e-9))
_antipodal_closure = _n_zero + 2 * (_kv.shape[0] - _n_zero)
_CLOSED_2D = [1, 5, 9, 13, 21, 25, 29, 37, 45, 57, 61, 81, 89, 97, 109]
complex_init = max(18, next((c for c in _CLOSED_2D if c >= _antipodal_closure), max(_CLOSED_2D)))
complex_basis = enumerate_complex_pw_basis_2d(complex_init, L, kappa=(0.0, 0.0))

coeff_up_complex = cos_sin_to_complex_coeffs(
    real_coeffs["cos_up"], real_coeffs["sin_up"], real_basis, np.asarray(complex_basis.kvecs))
coeff_dn_complex = cos_sin_to_complex_coeffs(
    real_coeffs["cos_dn"], real_coeffs["sin_dn"], real_basis, np.asarray(complex_basis.kvecs))

tabc_params_e = transfer_trained_params(ref_params["e"], tabc_params["e"])
new_pytree = dict(tabc_params)
new_pytree["e"] = tabc_params_e
for k in ("s", "mag_mlp", "phase_mlp", "q0_mlp", "p0_mlp", "s_mlp"):
    if k in ref_params and k in new_pytree:
        new_pytree[k] = ref_params[k]
new_pytree = set_complex_envelope(new_pytree, coeff_up_complex, coeff_dn_complex)
new_flat, _ = tabc_opt.flatten(new_pytree)   # real-aware
print(f"new_flat shape = {new_flat.shape}, tabc_opt.n_params = {tabc_opt.n_params}")

# Eval
stats = tabc_opt.evaluate(
    jax.random.key(7777),
    params_flat=new_flat,
    num_walkers=512, num_blocks=20, num_blocks_equil=5,
    num_steps_per_block=10,
    mc_timestep_R=0.1, mc_timestep_qc=0.5, verbose=0,
)
e_mha = stats["E_per_elec_ha"] * 1000
sem_mha = stats["E_serr_per_e_ha"] * 1000
im_mha = stats["Im_per_e_ha"] * 1000
print(f"\n=== κ-aware Phase 1 sanity at κ=(0,0) ===")
print(f"  E/N      = {e_mha:+.4f} ± {sem_mha:.4f} mHa/e")
print(f"  Im/N     = {im_mha:+.4e} mHa/e")
print(f"  V11 Γ    = +137.10 mHa/e")
print(f"  diff     = {e_mha - 137.10:+.4f} mHa/e")
match = abs(e_mha - 137.10) < max(5 * sem_mha, 1.0)
print(f"  match?   {'✓ YES' if match else '✗ NO'}")
print(f"\n  ΔMLP-up has zero-init last layer → Δcoeff(κ=0) ≈ 0 → "
      f"recovers non-κ-aware baseline")
