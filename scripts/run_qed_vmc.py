"""Run QED-NN-VMC SR optimization from a YAML config.

Usage:
    python scripts/run_qed_vmc.py inputs/qed_h2/<config>.yaml

Loads YAML, builds the QED driver + optimizer, runs SR training, evaluates
production-quality energy with the trained ansatz, and saves results to
HDF5 alongside the YAML config.

Output: logs/<project>/<project>.results.h5
"""
from __future__ import annotations

import argparse
import os
import os.path as osp
import sys
from datetime import datetime, timezone
import shutil

import h5py
import jax
import jax.numpy as jnp
import numpy as np
import yaml


def _activate_x64():
    jax.config.update("jax_enable_x64", True)


def _load_yaml(path: str) -> dict:
    with open(path, "r") as fh:
        return yaml.safe_load(fh)


def _build_h2_mol(R_bohr: float, spin: str = "singlet"):
    """H2 molecule along z-axis. spin: 'singlet' (n_up=n_down=1, X¹Σg+)
    or 'triplet' (n_up=2, n_down=0, Sz=1 component of a³Σu+)."""
    from OmegaQMC.utils import Mole_custom
    if spin == "triplet":
        n_up, n_down = 2, 0
    elif spin == "singlet":
        n_up, n_down = 1, 1
    else:
        raise ValueError(f"unknown spin={spin!r}; expected singlet|triplet")
    return Mole_custom.from_arrays(
        charges=[1, 1],
        coords=[
            [0.0, 0.0, -R_bohr / 2],
            [0.0, 0.0,  R_bohr / 2],
        ],
        n_up=n_up, n_down=n_down,
    )


def _build_h2_h2_mol(R_intermol_bohr: float, L_h2_bohr: float = 1.4010):
    """Two parallel H2 molecules along x, separated by R along z."""
    from OmegaQMC.utils import Mole_custom
    return Mole_custom.from_arrays(
        charges=[1, 1, 1, 1],
        coords=[
            [-L_h2_bohr / 2, 0.0, -R_intermol_bohr / 2],
            [ L_h2_bohr / 2, 0.0, -R_intermol_bohr / 2],
            [-L_h2_bohr / 2, 0.0,  R_intermol_bohr / 2],
            [ L_h2_bohr / 2, 0.0,  R_intermol_bohr / 2],
        ],
        n_up=2, n_down=2,
    )


def build_molecule(system_cfg: dict):
    sys_type = system_cfg["type"]
    if sys_type == "h2":
        return _build_h2_mol(
            R_bohr=float(system_cfg["R_bohr"]),
            spin=system_cfg.get("spin", "singlet"),
        )
    if sys_type == "h2_h2":
        return _build_h2_h2_mol(
            R_intermol_bohr=float(system_cfg["R_intermol_bohr"]),
            L_h2_bohr=float(system_cfg.get("L_h2_bohr", 1.4010)),
        )
    raise ValueError(f"unknown system type: {sys_type}")


class _Tee:
    """Mirror writes to multiple file handles (e.g. stdout + log file)."""

    def __init__(self, *streams):
        self._streams = streams

    def write(self, data):
        for s in self._streams:
            s.write(data)
            s.flush()

    def flush(self):
        for s in self._streams:
            s.flush()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="path to YAML config")
    args = parser.parse_args()

    _activate_x64()

    cfg = _load_yaml(args.config)
    project = cfg["project"]

    # Output directory matching the existing 2DHEG pattern.
    log_dir = osp.join("logs", project)
    os.makedirs(log_dir, exist_ok=True)
    shutil.copy(args.config, osp.join(log_dir, osp.basename(args.config)))

    # Persistent stdout log — stream all training/eval prints to a file
    # so progress can be `tail -f`ed even when invoked over ssh.
    log_path = osp.join(log_dir, "run.log")
    _log_fh = open(log_path, "w", buffering=1)  # line-buffered
    sys.stdout = _Tee(sys.__stdout__, _log_fh)
    sys.stderr = _Tee(sys.__stderr__, _log_fh)
    print(f"[run.log] persistent log path: {log_path}", flush=True)

    print(f"==== QED-VMC run: {project} ====", flush=True)
    print(f"host:    {os.uname().nodename}", flush=True)
    print(f"date:    {datetime.now(timezone.utc).isoformat()}", flush=True)
    print(f"jax devices: {jax.devices()}", flush=True)

    # System.
    mol = build_molecule(cfg["system"])
    print(f"system:  {cfg['system']['type']}, "
          f"n_elec={mol.n_up + mol.n_down}", flush=True)

    # Cavity.
    cav = cfg["cavity"]
    omega = float(cav["omega"])
    lam = float(cav["lambda"])
    eps = jnp.array(cav.get("polarization", [0.0, 0.0, 1.0]),
                    dtype=jnp.float64)
    eps = eps / jnp.linalg.norm(eps)
    coupling_vec = lam * eps
    print(f"cavity:  omega={omega:.4f} Ha, lambda={lam:.4f}, "
          f"eps={list(eps)}", flush=True)

    # Optimizer.
    opt_cfg = cfg["optimizer"]
    seed = int(cfg.get("seed", 0))
    init_key = jax.random.key(seed)
    nph_max = int(opt_cfg.get("nph_max", 8))

    from OmegaQMC.qed_vmcopt_nn_sr import get_qed_vmcopt_nn_sr_func
    opt = get_qed_vmcopt_nn_sr_func(
        mol,
        cfg["ansatz"]["type"],   # 'psiformer'
        init_key,
        omega=omega,
        coupling_vec=coupling_vec,
        alpha_init=float(opt_cfg.get("alpha_init", 0.0)),
        alpha_train=bool(opt_cfg.get("alpha_train", True)),
        nph_max=nph_max,
        n_aware=bool(opt_cfg.get("n_aware", False)),
        fock_hidden_dim=int(opt_cfg.get("fock_hidden_dim", 64)),
        arch=opt_cfg.get("arch"),
    )

    # Train.
    train_cfg = opt_cfg["train"]
    print(f"train:   "
          f"iters={train_cfg['num_iters']}, "
          f"walkers={train_cfg['num_walkers']}, "
          f"lr={train_cfg.get('lr', 0.05)}, "
          f"nph_max={nph_max}, "
          f"alpha_init={opt_cfg.get('alpha_init', 0.0)}, "
          f"alpha_train={opt_cfg.get('alpha_train', True)}",
          flush=True)
    t0 = datetime.now()
    params, history = opt(
        rng_key=jax.random.key(seed + 1),
        num_iters=int(train_cfg["num_iters"]),
        num_walkers=int(train_cfg["num_walkers"]),
        num_steps_per_block=int(train_cfg.get("num_steps_per_block", 50)),
        num_blocks_equil=int(train_cfg.get("num_blocks_equil", 5)),
        num_steps_decorr=int(train_cfg.get("num_steps_decorr", 5)),
        mc_timestep=float(train_cfg.get("mc_timestep", 0.1)),
        lr=float(train_cfg.get("lr", 0.05)),
        damping=float(train_cfg.get("damping", 1e-3)),
        cg_maxiter=int(train_cfg.get("cg_maxiter", 200)),
        max_param_change=float(train_cfg.get("max_param_change", 0.5)),
        jac_batch_size=int(train_cfg.get("jac_batch_size", 64)),
        alpha_lr_scale=float(train_cfg.get("alpha_lr_scale", 1.0)),
        verbose=int(train_cfg.get("verbose", 1)),
    )
    train_elapsed = (datetime.now() - t0).total_seconds()
    print(f"train complete in {train_elapsed:.1f}s", flush=True)

    # Production eval (longer sampling for tight stderr).
    eval_cfg = opt_cfg.get("eval", {})
    if eval_cfg.get("enabled", True):
        opt.driver.params = params  # apply trained params to driver
        eval_t0 = datetime.now()
        eval_result = opt.driver(
            rng_key=jax.random.key(seed + 2),
            num_walkers=int(eval_cfg.get("num_walkers", 512)),
            num_steps_per_block=int(eval_cfg.get("num_steps_per_block", 50)),
            num_blocks=int(eval_cfg.get("num_blocks", 50)),
            num_blocks_equil=int(eval_cfg.get("num_blocks_equil", 10)),
            mc_timestep=float(eval_cfg.get("mc_timestep", 0.1)),
            verbose=int(eval_cfg.get("verbose", 0)),
        )
        eval_elapsed = (datetime.now() - eval_t0).total_seconds()
        print(f"eval complete in {eval_elapsed:.1f}s: "
              f"E={eval_result['E_mean']:.6f} ± "
              f"{eval_result['E_serr']:.6f} Ha "
              f"<n>={eval_result['n_photon_mean']:.4f}",
              flush=True)
    else:
        eval_result = None

    # Save HDF5.
    out_path = osp.join(log_dir, f"{project}.results.h5")
    with h5py.File(out_path, "w") as f:
        f.attrs["project"] = project
        f.attrs["host"] = os.uname().nodename
        f.attrs["timestamp"] = datetime.now(timezone.utc).isoformat()
        f.attrs["omega"] = omega
        f.attrs["lambda"] = lam
        f.attrs["polarization"] = np.array(eps)
        f.attrs["nph_max"] = nph_max
        f.attrs["alpha_init"] = float(opt_cfg.get("alpha_init", 0.0))
        f.attrs["alpha_train"] = bool(opt_cfg.get("alpha_train", True))

        g_train = f.create_group("train")
        g_train.create_dataset("energies", data=np.array(history["energies"]))
        g_train.create_dataset("energy_serrs", data=np.array(history["energy_serrs"]))
        g_train.create_dataset("acceptances_r", data=np.array(history["acceptances_r"]))
        g_train.create_dataset("acceptances_n", data=np.array(history["acceptances_n"]))
        g_train.create_dataset("param_change_max", data=np.array(history["param_change_max"]))
        if history["alpha_history"]:
            g_train.create_dataset("alpha", data=np.array(history["alpha_history"]))
        g_train.attrs["elapsed_s"] = train_elapsed

        if eval_result is not None:
            g_eval = f.create_group("eval")
            g_eval.attrs["E_mean"] = eval_result["E_mean"]
            g_eval.attrs["E_serr"] = eval_result["E_serr"]
            g_eval.attrs["n_photon_mean"] = eval_result["n_photon_mean"]
            g_eval.attrs["acceptance_r"] = eval_result["acceptance_r"]
            g_eval.attrs["acceptance_n"] = eval_result["acceptance_n"]
            g_eval.attrs["sigma_r_final"] = eval_result["sigma_r_final"]
            g_eval.attrs["elapsed_s"] = eval_elapsed
            g_eval.create_dataset("E_blocks", data=np.array(eval_result["E_blocks"]))
            g_eval.create_dataset("n_photon_blocks",
                                  data=np.array(eval_result["n_photon_blocks"]))

    print(f"results saved to {out_path}", flush=True)


if __name__ == "__main__":
    main()
