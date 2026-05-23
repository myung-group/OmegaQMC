"""Post-mortem: extract learned sigma_x, sigma_y from a saved L5 chkpt.

Usage:
    python scripts/inspect_l5_sigma.py runs/<project>
"""
import sys, os, math, yaml
import numpy as np
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + "/.."))
from OmegaQMC.psi.nn.heg_wf import HEGPsiFormerConfig
from OmegaQMC.qed_vmcopt_nn_heg_l5 import _QEDL5Optimizer


def _build(cfg_path, chk_path):
    with open(cfg_path) as f:
        y = yaml.safe_load(f)
    sysc = y["system"]
    an = y["ansatz"]
    cav = y["cavity"]
    oc = y["optimize"]
    rs = float(sysc["rs"])
    n_up = int(sysc["n_up"]); n_down = int(sysc["n_down"])
    N = n_up + n_down
    M = int(round(math.sqrt(N / 2)))
    assert 2 * M * M == N, f"N={N} not 2M^2"
    a = math.sqrt(2.0 / (math.sqrt(3.0) * (1.0 / (math.pi * rs * rs))))
    L_x = M * a
    L_y = M * a * math.sqrt(3.0)
    cfg = HEGPsiFormerConfig(
        n_up=n_up, n_down=n_down, L=L_x, L_y=L_y, dim=2,
        backbone=an["backbone"],
        embedding_dim=an["embedding_dim"], n_interactions=an["n_interactions"],
        two_particle_stream_dim=an["two_particle_stream_dim"],
        n_det=an["n_det"], full_determinant=an["full_determinant"],
        use_backflow=an["use_backflow"], use_cusp=an["use_cusp"],
        n_virt_pw=an["n_virt_pw"], use_ghost_atom=an["use_ghost_atom"],
        use_deep_jastrow=an["use_deep_jastrow"],
        envelope_type=an["envelope_type"],
        crystal_sigma_init=an["crystal_sigma_init"],
        crystal_spin_pattern=an["crystal_spin_pattern"],
        crystal_det_jitter=an["crystal_det_jitter"],
        crystal_lattice_type=an["crystal_lattice_type"],
        crystal_anisotropic_sigma=an["crystal_anisotropic_sigma"],
    )
    pol = tuple(cav.get("polarization", (1.0, 0.0)))
    opt = _QEDL5Optimizer(
        cfg, jax.random.key(0),
        lr=oc["lr"], damping=oc["sr_damping"], n_cg=oc.get("sr_n_cg", 20),
        ewald_n_real=y["ewald"]["n_real"], ewald_n_recip=y["ewald"]["n_recip"],
        ofname_chkpt="/tmp/_dummy.chk.h5",
        lr_schedule=oc.get("lr_schedule", "cosine"),
        lr_min=oc.get("lr_min", 1e-5),
        lr_T_max=oc.get("lr_T_max", oc.get("iters", 2000)),
        spring_mu=oc.get("spring_mu", 0.9),
        spring_norm_clip=oc.get("spring_norm_clip", 0.1),
        use_smw_sr=oc["use_smw_sr"], use_fused_step=oc["use_fused_step"],
        freeze_mlps=oc.get("freeze_mlps", False),
        omega=cav["omega"], coupling_lambda=cav["lambda"],
        coupling_polarization=pol, K_max=cav["K_max"],
        phase_mlp_hidden=tuple(cav["phase_mlp_hidden"]),
        mag_mlp_hidden=tuple(cav["mag_mlp_hidden"]),
        activation=cav.get("activation", "tanh"),
    )
    d = np.load(chk_path, allow_pickle=True)
    p_flat = jnp.asarray(d["params_flat"])
    params = opt.unravel(p_flat)
    return params, L_x, L_y


def find_sigmas(params, prefix=""):
    out = []
    if isinstance(params, dict):
        for k, v in params.items():
            out += find_sigmas(v, f"{prefix}/{k}")
    elif hasattr(params, "shape"):
        if "sigma" in prefix.lower():
            arr = np.asarray(params)
            out.append((prefix, arr))
    return out


def main(run_dir):
    project = os.path.basename(os.path.normpath(run_dir))
    cfg_path = os.path.join(run_dir, "config.yaml")
    chk_path = os.path.join(run_dir, f"{project}.chk.npz")
    params, L_x, L_y = _build(cfg_path, chk_path)
    print(f"\nRun: {project}  L_x={L_x:.2f} L_y={L_y:.2f} bohr")
    # First dump every JAX array leaf via tree_leaves_with_path
    print("--- param tree leaves (jax.tree_util.tree_leaves_with_path) ---")
    from jax.tree_util import tree_leaves_with_path, keystr
    leaves = tree_leaves_with_path(params)
    print(f"  total leaves: {len(leaves)}")
    sigma_leaves = []
    for path, leaf in leaves:
        s = keystr(path)
        arr = np.asarray(leaf)
        if "sigma" in s.lower():
            sigma_leaves.append((s, arr))
            print(f"  SIGMA  {s}: shape={arr.shape}")
        elif arr.size <= 4:
            print(f"  SMALL  {s}: shape={arr.shape}  vals={arr.flatten()}")
    print("--- sigma values (raw + softplus) ---")
    for name, raw in sigma_leaves:
        effective = jax.nn.softplus(jnp.asarray(raw)) + 1e-3
        eff = np.asarray(effective)
        print(f"  {name}: raw={raw.flatten()}  eff(softplus)={eff.flatten()}")
        if eff.size == 2:
            sx, sy = float(eff[0]), float(eff[1])
            print(f"      σ_x={sx:.6f}  σ_y={sy:.6f}  "
                  f"σ_x/σ_y={sx/sy:.6f}  "
                  f"asym=(σ_x-σ_y)/(σ_x+σ_y)={(sx-sy)/(sx+sy):+.6f}")


if __name__ == "__main__":
    main(sys.argv[1])
