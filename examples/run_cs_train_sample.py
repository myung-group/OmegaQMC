"""Train NN-VMC + sample walker bank for arbitrary small molecules.

Generic driver that handles BeH2/H2O/N2/LiH/H2/H4 from a single CLI.
Companion to ``run_cs_properties.py``: this script produces the
checkpoint + walker bank; that one consumes them and prints the
property comparison.

Usage examples:
    python examples/run_cs_train_sample.py --molecule beh2 --basis cc-pvdz \
        --train-iters 1000 --sample-blocks 100 --sample-walkers 256
    python examples/run_cs_train_sample.py --molecule h2o --basis cc-pvdz \
        --train-iters 2000 --sample-blocks 100 --sample-walkers 256
    python examples/run_cs_train_sample.py --molecule n2 --basis sto-3g \
        --R 2.4 --unit Angstrom --train-iters 2000 --sample-blocks 100 \
        --sample-walkers 256
"""

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")

import jax
import numpy as np

from OmegaQMC.utils import Mole_custom
from OmegaQMC.vmc_nn import get_vmc_nn_func
from OmegaQMC.vmcopt_nn_sr import get_vmcopt_nn_func

sys.path.insert(0, str(Path(__file__).parent))
from run_cs_properties import build_mol  # noqa: E402


def molecule_prefix(args) -> str:
    """Filesystem-safe label for the molecule/basis/geometry."""
    tag = args.molecule
    if args.geometry_tag:
        tag += f"_{args.geometry_tag}"
    R_str = f"R{args.R:.3f}".replace(".", "p")
    basis = args.basis.replace("*", "s").replace("+", "p")
    ansatz_tag = args.ansatz_tag or os.path.splitext(
        os.path.basename(args.ansatz))[0]
    return f"{tag}_{ansatz_tag}_{R_str}_{basis}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--molecule", required=True,
                   choices=["h2", "h4", "lih", "beh2", "h2o", "n2", "c2"])
    p.add_argument("--geometry-tag", default="",
                   help="for h4: 'linear' or 'square'")
    p.add_argument("--R", type=float, default=1.0,
                   help="bond length / geometry parameter (ignored for h2o)")
    p.add_argument("--basis", default="cc-pvdz")
    p.add_argument("--unit", default="Angstrom",
                   choices=["Bohr", "Angstrom"])
    p.add_argument("--n-alpha", type=int, default=None)
    p.add_argument("--n-beta", type=int, default=None)

    p.add_argument("--ansatz",
                   default=str(Path(__file__).parents[1] / "OmegaQMC" /
                               "psi" / "nn" / "conf" /
                               "ferminet_jastrow.yaml"))
    p.add_argument("--ansatz-tag", default=None)
    p.add_argument("--out-dir", default="cs_pilot_results")
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--train-iters", type=int, default=1000)
    p.add_argument("--train-walkers", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--mc-timestep", type=float, default=0.1)
    p.add_argument("--jac-batch", type=int, default=4)
    p.add_argument("--retrain", action="store_true")

    p.add_argument("--sample-blocks", type=int, default=100)
    p.add_argument("--sample-walkers", type=int, default=256)
    p.add_argument("--sample-steps-per-block", type=int, default=20)
    p.add_argument("--sample-equil-blocks", type=int, default=5)
    p.add_argument("--resample", action="store_true")
    args = p.parse_args()

    # Default electron counts per molecule
    defaults = {
        "h2": (1, 1), "h4": (2, 2), "lih": (2, 2),
        "beh2": (3, 3), "h2o": (5, 5), "n2": (7, 7),
        "c2": (6, 6),
    }
    if args.n_alpha is None or args.n_beta is None:
        args.n_alpha, args.n_beta = defaults[args.molecule]

    prefix = molecule_prefix(args)
    out_dir = Path(args.out_dir)
    work_dir = out_dir / prefix
    work_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = work_dir / f"{prefix}.chk.h5"
    bank_path = work_dir / f"{prefix}_walkers.h5"

    print(f"=== {prefix} ===")
    mol = build_mol(args.molecule, args.R, args.basis, args.unit,
                    args.geometry_tag)
    print(f"  mol: {mol.nao} AOs, {mol.nelec} electrons, "
          f"basis={args.basis}, unit={args.unit}")
    print(f"  work_dir = {work_dir}")

    rng = jax.random.key(args.seed)
    rng, init_key, opt_key, smp_key = jax.random.split(rng, 4)

    need_train = args.retrain or not checkpoint.exists()
    if need_train:
        print(f"\n[train] {args.train_iters} iters @ "
              f"{args.train_walkers} walkers, lr={args.lr}")
        opt = get_vmcopt_nn_func(mol, args.ansatz, init_key)
        opt(
            opt_key,
            num_iters=args.train_iters,
            num_walkers=args.train_walkers,
            lr=args.lr,
            mc_timestep=args.mc_timestep,
            jac_batch_size=args.jac_batch,
            prefix=str(work_dir / prefix),
            verbose=1,
        )
    else:
        print(f"\n[train] reusing checkpoint {checkpoint.name}")

    driver = get_vmc_nn_func(mol, args.ansatz, init_key,
                              prefix=str(work_dir / prefix))
    driver.load_checkpoint(str(checkpoint))

    need_sample = args.resample or not bank_path.exists()
    if need_sample:
        print(f"\n[sample] {args.sample_blocks} blocks x "
              f"{args.sample_walkers} walkers = "
              f"{args.sample_blocks * args.sample_walkers} walkers")
        result = driver(
            smp_key,
            num_walkers=args.sample_walkers,
            num_steps_per_block=args.sample_steps_per_block,
            num_blocks=args.sample_blocks,
            num_blocks_equil=args.sample_equil_blocks,
            mc_timestep=args.mc_timestep,
            compute_gradients=False,
            dump_walkers_path=str(bank_path),
            verbose=1,
        )
        print(f"  E_NN = {result['E_mean']:.6f} +/- {result['E_serr']:.6f} Ha")
    else:
        print(f"\n[sample] reusing bank {bank_path.name}")

    print(f"\nDone. Run properties next:")
    print(f"  python examples/run_cs_properties.py "
          f"--cell-dir {work_dir} "
          f"--geometry {args.molecule} "
          f"--R {args.R} --basis {args.basis} --unit {args.unit} "
          f"--ansatz {args.ansatz} "
          f"--n-alpha {args.n_alpha} --n-beta {args.n_beta}"
          + (f" --geometry-tag {args.geometry_tag}" if args.geometry_tag else "")
          + f" --out-json {work_dir / (prefix + '_properties.json')}")


if __name__ == "__main__":
    main()
