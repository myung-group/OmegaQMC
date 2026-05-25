"""Minimal H2 ground-state training + walker bank sampling.

Avoids run_h2_nesvmc.py's mandatory excited-state sampling step,
which fails if no excited-state checkpoint exists. This script just
trains the ground state via SR and dumps the walker bank, producing
exactly the artefacts the Pfau-NES + spectroscopy demo needs as its
warm-start.

Usage:
    python examples/run_h2_groundstate_only.py --R 2.5 --basis cc-pvdz
"""

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")

import jax

from OmegaQMC.utils import Mole_custom
from OmegaQMC.vmc_nn import get_vmc_nn_func
from OmegaQMC.vmcopt_nn_sr import get_vmcopt_nn_func as get_sr_func


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--R", type=float, default=2.5)
    p.add_argument("--basis", default="cc-pvdz")
    p.add_argument("--ansatz",
                   default=str(Path(__file__).parent / "inputs"
                               / "psiformer_small.yaml"))
    p.add_argument("--out-dir", default="cs_h2_nesvmc_results")
    p.add_argument("--seed", type=int, default=11)
    p.add_argument("--gs-iters", type=int, default=600)
    p.add_argument("--gs-walkers", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--mc-timestep", type=float, default=0.1)
    p.add_argument("--sample-blocks", type=int, default=50)
    p.add_argument("--sample-walkers", type=int, default=256)
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    R_str = f"R{args.R:.3f}".replace(".", "p")
    basis_tag = args.basis.replace("*", "s")
    prefix_gs = out_dir / f"h2_gs_{R_str}_{basis_tag}"
    chk_gs = prefix_gs.with_suffix(".chk.h5")
    bank_gs = out_dir / (prefix_gs.name + "_walkers.h5")

    mol = Mole_custom()
    mol.build(
        atom=[("H", [0, 0, 0]), ("H", [0, 0, args.R])],
        basis=args.basis, spin=0, charge=0, unit="Bohr", verbose=0,
    )
    print(f"=== H2 ground state training at R={args.R} Bohr, {args.basis} ===")
    print(f"  mol: {mol.nao} AOs, nelec={mol.nelec}")

    if not chk_gs.exists():
        print(f"\n[1/2] Training ground state ({args.gs_iters} SR iters)")
        rng = jax.random.key(args.seed)
        init_key, opt_key = jax.random.split(rng)
        sr = get_sr_func(mol, args.ansatz, init_key)
        sr(opt_key, num_iters=args.gs_iters, num_walkers=args.gs_walkers,
           lr=args.lr, mc_timestep=args.mc_timestep,
           jac_batch_size=4, prefix=str(prefix_gs), verbose=1)
    else:
        print(f"\n[1/2] Reusing GS checkpoint {chk_gs.name}")

    if not bank_gs.exists():
        print(f"\n[2/2] Sampling GS walker bank")
        rng_smp = jax.random.key(args.seed + 1000)
        init_key, smp_key = jax.random.split(rng_smp)
        driver_gs = get_vmc_nn_func(
            mol, args.ansatz, init_key, prefix=str(prefix_gs),
        )
        driver_gs.load_checkpoint(str(chk_gs))
        driver_gs(
            smp_key,
            num_walkers=args.sample_walkers,
            num_steps_per_block=20,
            num_blocks=args.sample_blocks,
            num_blocks_equil=5,
            mc_timestep=args.mc_timestep,
            compute_gradients=False,
            dump_walkers_path=str(bank_gs),
            verbose=1,
        )
    else:
        print(f"\n[2/2] Reusing GS bank {bank_gs.name}")

    print(f"\nDone. checkpoints: {chk_gs}, walker bank: {bank_gs}")


if __name__ == "__main__":
    main()
