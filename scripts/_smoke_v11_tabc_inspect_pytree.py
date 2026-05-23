"""Inspect the param pytree structure after building a TABC optimizer.
Find where envelope coefficients live so we can overwrite them with HF init.
"""
import math, sys, yaml
import jax, jax.numpy as jnp, numpy as np
jax.config.update("jax_enable_x64", True)
sys.path.insert(0, ".")

from OmegaQMC.psi.nn.heg_wf import HEGPsiFormerConfig
from OmegaQMC.qed_vmcopt_nn_heg_l5 import _QEDL5Optimizer
from OmegaQMC.psi.nn.env_periodic import enumerate_complex_pw_basis_2d

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
opt = _QEDL5Optimizer(
    cfg, jax.random.key(0),
    lr=oc["lr"], damping=oc["sr_damping"],
    omega=cav["omega"], coupling_lambda=cav["lambda"],
    coupling_polarization=cav.get("polarization"),
    coupling_op=cav["coupling_op"],
    K_max=cav["K_max"],
    phase_mlp_hidden=tuple(cav["phase_mlp_hidden"]),
    mag_mlp_hidden=tuple(cav["mag_mlp_hidden"]),
    use_matter_photon_shift=True, use_matter_photon_pshift=True,
    q0_mlp_hidden=tuple(cav.get("q0_mlp_hidden", [32, 32])),
    p0_mlp_hidden=tuple(cav.get("p0_mlp_hidden", [32, 32])),
    v_ext_amp=float(cfg_yaml.get("v_ext", {}).get("amp", 0.0)),
    v_ext_a=cfg_yaml.get("v_ext", {}).get("a"),
    include_vee=bool(sys_c.get("include_vee", True)),
    twist=(0.1, -0.1),  # arbitrary nonzero
)
print(f"twist opt n_params = {opt.n_params}")

p = opt.unravel(opt.params_flat)
print(f"top-level keys: {list(p.keys())}")
print(f"\np['e'] type: {type(p['e'])}")
if isinstance(p["e"], dict):
    print(f"p['e'] keys: {list(p['e'].keys())}")

# Walk and find envelope-related leaves
print(f"\n--- all leaves with path containing 'envelope', 'coeff', 'kvec' ---")
flat = jax.tree_util.tree_leaves_with_path(p)
for path, leaf in flat:
    path_str = "/".join(
        getattr(pp, "key", getattr(pp, "name", str(pp))) for pp in path
    )
    if any(kw in path_str.lower() for kw in ["envelope", "coeff", "kvec"]):
        arr = np.asarray(leaf)
        print(f"  {path_str}: shape={arr.shape}, dtype={arr.dtype}")

# Also inspect the basis used
print(f"\n--- complex PW basis at twist (0.1, -0.1) ---")
basis = enumerate_complex_pw_basis_2d(9, L, kappa=(0.1, -0.1))
print(f"  n_pw = {basis.kvecs.shape[0]}")
print(f"  kvecs first 3: {np.asarray(basis.kvecs)[:3]}")
print(f"  k_sq first 5: {np.asarray(basis.k_sq)[:5]}")
