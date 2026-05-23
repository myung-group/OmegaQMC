"""Sanity test: evaluate V11 at κ=(0,0) via the TABC pipeline.
Should recover ~+137.10 mHa/e (V11 Γ-point reference).
If it does → TABC machinery is correct; large per-twist E at κ≠0 is real physics.
If not → bug somewhere.
"""
import math, sys, yaml
import jax, jax.numpy as jnp, numpy as np
jax.config.update("jax_enable_x64", True)
sys.path.insert(0, ".")

from jax.flatten_util import ravel_pytree
from OmegaQMC.psi.nn.heg_wf import HEGPsiFormerConfig, transfer_trained_params
from OmegaQMC.qed_vmcopt_nn_heg_l5 import _QEDL5Optimizer

# import HF init helpers from the TABC driver
sys.path.insert(0, "scripts")
from run_qed_l5_tabc import hf_orbitals_2d, _set_hf_envelope_coefficients

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

# Load chkpt + Γ reference pytree
chkpt = np.load(CHKPT, allow_pickle=True)
chkpt_params_flat = jnp.asarray(chkpt["params_flat"])
print(f"chkpt: {chkpt_params_flat.shape[0]} params, "
      f"Γ E/N = {float(chkpt['E_final_ha'])*1000:+.4f} mHa/e")

ref = build_opt(twist=None)  # Γ build
assert ref.n_params == chkpt_params_flat.shape[0]
chkpt_pytree = ref.unravel(chkpt_params_flat)
print(f"chkpt pytree keys: {list(chkpt_pytree.keys())}")

# === Test at κ=(0, 0) — should give Γ result ===
print("\n=== TABC eval at κ=(0,0) — should recover Γ ===")
opt_kz = build_opt(twist=(0.0, 0.0))
fresh_pytree = opt_kz.unravel(opt_kz.params_flat)
# Transfer chkpt → fresh: matter via transfer (skips envelope), rest direct
new_pytree = dict(fresh_pytree)
new_pytree["e"] = transfer_trained_params(chkpt_pytree["e"], fresh_pytree["e"])
for k in ("s", "mag_mlp", "phase_mlp", "q0_mlp", "p0_mlp", "s_mlp"):
    if k in chkpt_pytree and k in new_pytree:
        new_pytree[k] = chkpt_pytree[k]

# HF-init the envelope at κ=(0,0).  Should give back the V11 trained
# wavefunction (since matter is non-interacting + HF is exact at v_ext≠0).
# For full_determinant=True, n_pw = n_up + n_down.  Look up from leaves
# to avoid nnx-state string indexing issues.
n_pw_actual = None
flat = jax.tree_util.tree_leaves_with_path(new_pytree)
for path, leaf in flat:
    s = "/".join(
        str(getattr(p, "key", getattr(p, "name", getattr(p, "idx", p))))
        for p in path
    )
    if s.endswith("envelope/coeff_up/value"):
        n_pw_actual = leaf.shape[-1]
        break
assert n_pw_actual is not None, "could not find envelope/coeff_up/value leaf"
print(f"detected n_pw in envelope: {n_pw_actual}")
v_amp = float(cfg_yaml.get("v_ext", {}).get("amp", 0.0))
v_a = float(cfg_yaml.get("v_ext", {}).get("a", L))
hf_coeffs = hf_orbitals_2d(
    L=L, kappa=(0.0, 0.0), v_ext_amp=v_amp, v_ext_a=v_a,
    n_pw=n_pw_actual, n_orbitals=sys_c["n_up"],
)
print(f"HF-init envelope: n_pw={n_pw_actual}, hf_coeffs shape={hf_coeffs.shape}")
print(f"  |HF coeff[:, 0]|² (lowest orbital): {np.abs(hf_coeffs[:, 0])**2}")
new_pytree = _set_hf_envelope_coefficients(
    new_pytree, hf_coeffs, n_det=int(an["n_det"]),
)

new_flat, _ = ravel_pytree(new_pytree)
print(f"new_flat shape = {new_flat.shape}, opt_kz.n_params = {opt_kz.n_params}")
assert new_flat.shape[0] == opt_kz.n_params

# Run eval
stats = opt_kz.evaluate(
    jax.random.key(7777),
    params_flat=new_flat,
    num_walkers=256,
    num_blocks=10,
    num_blocks_equil=5,
    num_steps_per_block=10,
    mc_timestep_R=0.1,
    mc_timestep_qc=0.5,
    verbose=0,
)
e_per_e_mha = stats["E_per_elec_ha"] * 1000
sem_mha = stats["E_serr_per_e_ha"] * 1000
im_per_e_mha = stats["Im_per_e_ha"] * 1000
print(f"  E/N      = {e_per_e_mha:+.4f} ± {sem_mha:.4f} mHa/e")
print(f"  Im/N     = {im_per_e_mha:+.4e} mHa/e (Hermiticity)")
print(f"  Γ ref    = +137.10 mHa/e (V11 eval)")
print(f"  match?   {'✓ YES (within ~1 SEM)' if abs(e_per_e_mha - 137.10) < 5*sem_mha else '✗ NO — bug?'}")
