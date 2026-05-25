"""Minimal H2O ground-state training + walker bank sampling.

Equilibrium geometry (re-OH = 0.957 Å, HOH = 104.5°), C2v point
group. Closed-shell singlet with 10 electrons.
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


def build_h2o(basis):
    # Standard equilibrium geometry; oxygen at origin, H atoms in xz plane,
    # bisector along +z; bond length 0.957 Å, HOH angle 104.5 deg.
    # Coords in Angstrom.
    import math
    r = 0.957
    theta = math.radians(104.5 / 2.0)
    h_x = r * math.sin(theta)
    h_z = r * math.cos(theta)
    mol = Mole_custom()
    mol.build(
        atom=[("O", [0.0, 0.0, 0.0]),
              ("H", [h_x, 0.0, h_z]),
              ("H", [-h_x, 0.0, h_z])],
        basis=basis, spin=0, charge=0, unit="Angstrom", verbose=0,
    )
    return mol


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--basis", default="cc-pvdz")
    p.add_argument("--ansatz",
                   default=str(Path(__file__).parent / "inputs"
                               / "psiformer_small.yaml"))
    p.add_argument("--out-dir", default="cs_h2o_pfau_gs")
    p.add_argument("--seed", type=int, default=11)
    p.add_argument("--gs-iters", type=int, default=1000)
    p.add_argument("--gs-walkers", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--mc-timestep", type=float, default=0.05)
    p.add_argument("--sample-blocks", type=int, default=50)
    p.add_argument("--sample-walkers", type=int, default=512)
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    basis_tag = args.basis.replace("*", "s")
    prefix_gs = out_dir / f"h2o_gs_{basis_tag}"
    chk_gs = prefix_gs.with_suffix(".chk.h5")
    bank_gs = out_dir / (prefix_gs.name + "_walkers.h5")

    mol = build_h2o(args.basis)
    print(f"=== H2O ground state training, {args.basis} ===")
    print(f"  mol: {mol.nao} AOs, nelec={mol.nelec}")

    if not chk_gs.exists():
        print(f"\n[1/2] Training ground state ({args.gs_iters} SR iters)")
        rng = jax.random.key(args.seed)
        init_key, opt_key = jax.random.split(rng)
        sr = get_sr_func(mol, args.ansatz, init_key)
        sr(opt_key, num_iters=args.gs_iters, num_walkers=args.gs_walkers,
           lr=args.lr, mc_timestep=args.mc_timestep,
           jac_batch_size=8, prefix=str(prefix_gs), verbose=1)
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

    print(f"\nDone. {chk_gs}, {bank_gs}")


if __name__ == "__main__":
    main()
