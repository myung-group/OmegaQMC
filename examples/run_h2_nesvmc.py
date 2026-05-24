"""H2 NES-VMC training: ground + first excited singlet via penalty method.

Trains a PsiFormer ground state for stretched H2 (or loads from
checkpoint), then trains an excited state orthogonalised against
it via the overlap penalty of
:mod:`OmegaQMC.vmcopt_nn_nes`. Dumps walker banks from both states
so the standard CS pipeline can recover two CI vectors.

Usage:
    python examples/run_h2_nesvmc.py --R 2.5 --basis cc-pvdz \
        --gs-iters 600 --es-iters 2000 --lambda-penalty 1.0
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
from OmegaQMC.vmcopt_nn_sr import get_vmcopt_nn_func as get_sr_func
from OmegaQMC.vmcopt_nn_nes import get_vmcopt_nn_nes_func


def build_h2(R_bohr: float, basis: str) -> Mole_custom:
    mol = Mole_custom()
    mol.build(
        atom=[("H", [0, 0, 0]), ("H", [0, 0, R_bohr])],
        basis=basis, spin=0, charge=0, unit="Bohr", verbose=0,
    )
    return mol


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--R", type=float, default=2.5)
    p.add_argument("--basis", default="cc-pvdz")
    p.add_argument("--ansatz",
                   default=str(Path(__file__).parent / "inputs"
                               / "psiformer_small.yaml"))
    p.add_argument("--out-dir", default="cs_h2_nesvmc_results")
    p.add_argument("--seed-gs", type=int, default=11)
    p.add_argument("--seed-es", type=int, default=22)

    p.add_argument("--gs-iters", type=int, default=600)
    p.add_argument("--gs-walkers", type=int, default=128)
    p.add_argument("--es-iters", type=int, default=2000)
    p.add_argument("--es-walkers", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--mc-timestep", type=float, default=0.1)
    p.add_argument("--lambda-penalty", type=float, default=1.0)

    p.add_argument("--sample-blocks", type=int, default=50)
    p.add_argument("--sample-walkers", type=int, default=256)
    p.add_argument("--skip-gs-train", action="store_true")
    p.add_argument("--skip-es-train", action="store_true")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    R_str = f"R{args.R:.3f}".replace(".", "p")
    basis_tag = args.basis.replace("*", "s")
    prefix_gs = out_dir / f"h2_gs_{R_str}_{basis_tag}"
    prefix_es = out_dir / f"h2_es_{R_str}_{basis_tag}"
    chk_gs = prefix_gs.with_suffix(".chk.h5")
    chk_es = prefix_es.with_suffix(".chk.h5")
    bank_gs = out_dir / (prefix_gs.name + "_walkers.h5")
    bank_es = out_dir / (prefix_es.name + "_walkers.h5")

    print(f"=== H2 NES-VMC at R={args.R} Bohr in {args.basis} ===")
    mol = build_h2(args.R, args.basis)
    print(f"  mol: {mol.nao} AOs, {mol.nelec} electrons")

    # ----- Ground state -----
    if not args.skip_gs_train and not chk_gs.exists():
        print(f"\n[1/4] Training ground state ({args.gs_iters} SR iters)")
        rng_gs = jax.random.key(args.seed_gs)
        init_key, opt_key = jax.random.split(rng_gs)
        sr = get_sr_func(mol, args.ansatz, init_key)
        sr(opt_key, num_iters=args.gs_iters, num_walkers=args.gs_walkers,
           lr=args.lr, mc_timestep=args.mc_timestep,
           jac_batch_size=4, prefix=str(prefix_gs), verbose=1)
    else:
        print(f"\n[1/4] Reusing ground-state checkpoint {chk_gs.name}")

    # Sample walker bank from ground state
    rng_smp = jax.random.key(args.seed_gs + 1000)
    init_key, smp_key = jax.random.split(rng_smp)
    driver_gs = get_vmc_nn_func(mol, args.ansatz, init_key, prefix=str(prefix_gs))
    driver_gs.load_checkpoint(str(chk_gs))
    if not bank_gs.exists():
        print(f"\n[2/4] Sampling ground-state walkers")
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
        print(f"\n[2/4] Reusing ground-state walker bank {bank_gs.name}")

    # ----- Excited state via NES-VMC -----
    if not args.skip_es_train and not chk_es.exists():
        print(f"\n[3/4] Training excited state ({args.es_iters} SR iters, "
              f"lambda_penalty={args.lambda_penalty})")
        rng_es = jax.random.key(args.seed_es)
        init_key, opt_key = jax.random.split(rng_es)
        nes = get_vmcopt_nn_nes_func(
            mol, args.ansatz, init_key,
            ground_state_checkpoint=str(chk_gs),
            lambda_penalty=args.lambda_penalty,
        )
        # IRAdam-style call signature (inherited): outer loop of
        # (resample walker snapshots, then run Adam epochs over them)
        nes(opt_key, num_iters=args.es_iters,
            num_walkers=args.es_walkers,
            num_epochs=10,
            num_sample_blocks=2,
            mc_timestep=args.mc_timestep,
            lr=args.lr, prefix=str(prefix_es), verbose=1)
    else:
        print(f"\n[3/4] Reusing excited-state checkpoint {chk_es.name}")

    # Sample walker bank from excited state
    rng_smp_es = jax.random.key(args.seed_es + 1000)
    init_key, smp_key = jax.random.split(rng_smp_es)
    driver_es = get_vmc_nn_func(mol, args.ansatz, init_key, prefix=str(prefix_es))
    driver_es.load_checkpoint(str(chk_es))
    if not bank_es.exists():
        print(f"\n[4/4] Sampling excited-state walkers")
        driver_es(
            smp_key,
            num_walkers=args.sample_walkers,
            num_steps_per_block=20,
            num_blocks=args.sample_blocks,
            num_blocks_equil=5,
            mc_timestep=args.mc_timestep,
            compute_gradients=False,
            dump_walkers_path=str(bank_es),
            verbose=1,
        )
    else:
        print(f"\n[4/4] Reusing excited-state walker bank {bank_es.name}")

    print("\nDone. Run CS recovery on both banks:")
    print(f"  python examples/run_cs_properties.py --cell-dir {out_dir}/<gs|es>"
          f" --geometry h2 --R {args.R} --basis {args.basis} --unit Bohr")


if __name__ == "__main__":
    main()
