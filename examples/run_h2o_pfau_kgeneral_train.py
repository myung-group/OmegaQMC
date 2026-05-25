"""H2O Pfau-NES K-general training (K >= 2).

Trains K orthogonal eigenstates jointly via the determinantal loss.
Designed to be the K=4 follow-up to the K=2 H2O demo when the K=2
random-init Pfau-NES doesn't capture the lowest dipole-allowed
singlet.

After training the K state checkpoints get saved as
``<prefix>_<k>.chk.h5`` (1-indexed); the downstream CS recovery /
rotation / transition / NEVPT2 stack consumes those one at a time.
"""

import argparse
import math
import os
import sys
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")

import jax

from OmegaQMC.utils import Mole_custom
from OmegaQMC.vmcopt_nn_pfau import get_vmcopt_nn_pfau_k_func


def build_h2o(basis):
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
    p.add_argument("--out-dir", default="cs_h2o_pfau_k4_results")
    p.add_argument("--gs-source-dir", default="cs_h2o_pfau_gs",
                   help="dir with the pre-trained GS checkpoint to "
                        "seed state 0 from")
    p.add_argument("--K", type=int, default=4)
    p.add_argument("--seed", type=int, default=77)
    p.add_argument("--iters", type=int, default=400)
    p.add_argument("--walkers", type=int, default=128,
                   help="K^3 cost scaling -- start small for K>=4")
    p.add_argument("--steps-per-block", type=int, default=50)
    p.add_argument("--steps-decorr", type=int, default=5)
    p.add_argument("--equil-blocks", type=int, default=3)
    p.add_argument("--mc-timestep", type=float, default=0.05)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--damping", type=float, default=1e-3)
    p.add_argument("--cg-maxiter", type=int, default=100)
    p.add_argument("--jac-batch", type=int, default=8)
    p.add_argument("--init-perturbation", type=float, default=0.5)
    p.add_argument("--init-random-states", type=int, default=None,
                   help="number of states (from K-1 backward) to give "
                        "fully random inits instead of GS-perturbed")
    args = p.parse_args()

    basis_tag = args.basis.replace("*", "s")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = out_dir / f"h2o_pfau_k{args.K}_{basis_tag}"

    src_dir = Path(args.gs_source_dir)
    chk_gs = src_dir / f"h2o_gs_{basis_tag}.chk.h5"
    if not chk_gs.exists():
        sys.exit(f"missing GS checkpoint {chk_gs}; run "
                 f"examples/run_h2o_groundstate_only.py first")

    print(f"=== H2O Pfau-NES K={args.K}, {args.basis} ===")
    print(f"  iters={args.iters}, walkers={args.walkers}, "
          f"lr={args.lr}, init_pert={args.init_perturbation}, "
          f"init_random={args.init_random_states}")
    mol = build_h2o(args.basis)
    print(f"  mol: {mol.nao} AOs, nelec={mol.nelec}")

    init_key = jax.random.key(args.seed)
    driver = get_vmcopt_nn_pfau_k_func(
        mol, args.ansatz, init_key,
        K=args.K,
        init_from_ground_checkpoint=str(chk_gs),
        init_perturbation=args.init_perturbation,
        init_random_states=args.init_random_states,
    )
    rng_opt = jax.random.split(init_key, 4)[3]
    driver(
        rng_opt,
        num_iters=args.iters,
        num_walkers=args.walkers,
        num_steps_per_block=args.steps_per_block,
        num_steps_decorr=args.steps_decorr,
        num_blocks_equil=args.equil_blocks,
        mc_timestep=args.mc_timestep,
        lr=args.lr,
        damping=args.damping,
        cg_maxiter=args.cg_maxiter,
        jac_batch_size=args.jac_batch,
        prefix=str(prefix),
        verbose=1,
    )

    print(f"\n[done] {args.K} checkpoints saved as {prefix}_<k>.chk.h5")


if __name__ == "__main__":
    main()
