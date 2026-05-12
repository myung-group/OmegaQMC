"""Re-run eval on a trained NN-VMC model to dump walker positions.

For density-chirality / 1-RDM post-processing. Loads the trained
params pickle, reconstructs the driver, runs eval with
dump_walker_positions=True, saves walkers to .npz.

Usage:
    python scripts/dump_walker_positions.py <yaml> [--params <pkl>]

If --params not given, looks for logs/<project>/<project>.params.pkl.
"""
from __future__ import annotations

import argparse
import os
import os.path as osp
import pickle
import sys
from datetime import datetime

import h5py
import jax
import jax.numpy as jnp
import numpy as np
import yaml


def _activate_x64():
    jax.config.update("jax_enable_x64", True)


def _load_yaml(path):
    with open(path) as f:
        return yaml.safe_load(f)


# Copy the molecule + cavity setup from run_qed_vmc.py
sys.path.insert(0, osp.join(osp.dirname(__file__)))
from run_qed_vmc import build_molecule


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config", help="path to original YAML")
    ap.add_argument("--params", default=None,
                    help="path to trained params .pkl "
                         "(default: logs/<project>/<project>.params.pkl)")
    ap.add_argument("--out", default=None,
                    help="output .npz path (default: alongside h5)")
    ap.add_argument("--num-walkers", type=int, default=None,
                    help="override eval num_walkers")
    ap.add_argument("--num-blocks", type=int, default=None,
                    help="override eval num_blocks")
    args = ap.parse_args()

    _activate_x64()

    cfg = _load_yaml(args.config)
    project = cfg["project"]
    log_dir = osp.join("logs", project)

    params_path = (
        args.params or osp.join(log_dir, f"{project}.params.pkl")
    )
    if not osp.exists(params_path):
        raise FileNotFoundError(f"no params pickle at {params_path}")
    with open(params_path, "rb") as f:
        params = pickle.load(f)
    print(f"loaded params from {params_path}", flush=True)

    mol = build_molecule(cfg["system"])
    print(f"system: {cfg['system']['type']}, "
          f"n_elec={mol.n_up + mol.n_down}", flush=True)

    cav = cfg["cavity"]
    omega = float(cav["omega"])
    lam = float(cav["lambda"])
    eps = jnp.array(cav.get("polarization", [0.0, 0.0, 1.0]),
                    dtype=jnp.float64)
    eps = eps / jnp.linalg.norm(eps)
    coupling_vec = lam * eps

    chiral_eps_y = None
    chiral_handedness = 1
    if "polarization_y" in cav:
        ey = jnp.array(cav["polarization_y"], dtype=jnp.float64)
        chiral_eps_y = ey / jnp.linalg.norm(ey)
        chiral_handedness = int(cav.get("chiral_handedness", 1))

    opt_cfg = cfg["optimizer"]
    seed = int(cfg.get("seed", 0))
    init_key = jax.random.key(seed)
    nph_max = int(opt_cfg.get("nph_max", 8))

    from OmegaQMC.qed_vmcopt_nn_sr import get_qed_vmcopt_nn_sr_func
    complex_psi_flag = bool(opt_cfg.get("complex_psi", False))
    opt = get_qed_vmcopt_nn_sr_func(
        mol,
        cfg["ansatz"]["type"],
        init_key,
        omega=omega,
        coupling_vec=coupling_vec,
        alpha_init=float(opt_cfg.get("alpha_init", 0.0)),
        alpha_train=bool(opt_cfg.get("alpha_train", True)),
        nph_max=nph_max,
        n_aware=bool(opt_cfg.get("n_aware", False)),
        fock_hidden_dim=int(opt_cfg.get("fock_hidden_dim", 64)),
        arch=opt_cfg.get("arch"),
        complex_psi=complex_psi_flag,
        chiral_eps_y=chiral_eps_y,
        chiral_handedness=chiral_handedness,
    )

    opt.driver.params = params  # inject trained params

    eval_cfg = opt_cfg.get("eval", {})
    n_walkers = args.num_walkers or int(eval_cfg.get("num_walkers", 512))
    n_blocks = args.num_blocks or int(eval_cfg.get("num_blocks", 30))

    print(f"running eval with {n_walkers} walkers, "
          f"{n_blocks} prod blocks, walker dump ON", flush=True)
    t0 = datetime.now()
    result = opt.driver(
        rng_key=jax.random.key(seed + 2),
        num_walkers=n_walkers,
        num_steps_per_block=int(eval_cfg.get("num_steps_per_block", 30)),
        num_blocks=n_blocks,
        num_blocks_equil=int(eval_cfg.get("num_blocks_equil", 10)),
        mc_timestep=float(eval_cfg.get("mc_timestep", 0.1)),
        verbose=1,
        dump_walker_positions=True,
    )
    elapsed = (datetime.now() - t0).total_seconds()
    print(f"eval+dump complete in {elapsed:.1f}s: "
          f"E={result['E_mean']:.6f} ± {result['E_serr']:.6f}, "
          f"<L_z>={result.get('l_z_mean', float('nan')):+.4f}",
          flush=True)

    out = args.out or osp.join(log_dir, f"{project}.walkers.npz")
    walkers = result["walker_positions"]
    print(f"walker_positions shape: {walkers.shape} "
          f"(blocks, walkers, electrons, xyz)", flush=True)
    np.savez_compressed(
        out,
        walker_positions=walkers,
        nuc_coords=np.array(mol.coords),
        charges=np.array(mol.charges),
        E_mean=result["E_mean"],
        E_serr=result["E_serr"],
        l_z_mean=result.get("l_z_mean", np.nan),
        l_z_serr=result.get("l_z_serr", np.nan),
        omega=omega,
        cavity_lambda=lam,
        chiral_handedness=chiral_handedness,
    )
    print(f"saved to {out}", flush=True)


if __name__ == "__main__":
    main()
