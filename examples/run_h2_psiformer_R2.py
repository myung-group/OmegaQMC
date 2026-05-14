"""VMC + PGCS ZVZB forces for H2 with a PsiFormer ansatz.

Patterned on ``tests/test_vmc_H2_psiformer.py``
(``test_vmc_nn_h2_gradients_pgcs``): uses a *tilted* H2
geometry (small transverse offsets) to avoid the
``log|h_{ia,k}| = log|0| = -inf`` NaN that an exactly
on-axis H2 would trigger in the ZVZB kinetic correction
(see the test's docstring).

Stages:
  [1/3] SR optimization of PsiFormer at the tilted H2
        geometry with bond length R.
  [2/3] VMC evaluation with PGCS (``symmop_list``) and
        ``compute_gradients=True``; per-walker gradient
        samples written to ``<prefix>.grd.h5``.
  [3/3] ``postproc_h5_pgcs`` → symmetry-averaged nuclear
        forces F_{ia} (Ha/Bohr) with statistical errors,
        plus projection onto the bond axis.

Usage:
    python run_h2_psiformer_R2.py
    python run_h2_psiformer_R2.py --iters 1000
    python run_h2_psiformer_R2.py --skip-opt   # reuse checkpoint
    python run_h2_psiformer_R2.py --symmops E,Rz180,i
"""

import argparse
import os

os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
os.environ.setdefault('XLA_PYTHON_CLIENT_ALLOCATOR', 'platform')

import jax
import jax.numpy as jnp
import numpy as np

from OmegaQMC.utils import Mole_custom
from OmegaQMC.vmc_nn import get_vmc_nn_func
from OmegaQMC.vmcopt_nn_sr import get_vmcopt_nn_func
from OmegaQMC.observables.force import postproc_h5_pgcs


def build_h2_tilted(R_bohr: float, tilt_bohr: float) -> Mole_custom:
    """Tilted H2 with bond length exactly R_bohr.

    Geometry mirrors the style of
    ``test_vmc_nn_h2_gradients_pgcs``:
        r0 = (+tilt, 0, -z0),  r1 = (0, +tilt, +z0)
    with z0 chosen so |r1 - r0| = R.  The midpoint is
    at (tilt/2, tilt/2, 0) — fragments are centered at
    the fragment centroid by the driver, not at the
    global origin, so this is fine.
    """
    if tilt_bohr >= R_bohr / np.sqrt(2):
        raise ValueError(
            f"--tilt {tilt_bohr} too large for R={R_bohr}; "
            f"need tilt < R/sqrt(2) = {R_bohr/np.sqrt(2):.3f}"
        )
    z0 = np.sqrt((R_bohr**2 - 2 * tilt_bohr**2) / 4)
    r0 = [float(tilt_bohr), 0.0, -float(z0)]
    r1 = [0.0, float(tilt_bohr), +float(z0)]
    mol = Mole_custom.from_arrays(
        charges=[1, 1],
        coords=[r0, r1],
        n_up=1, n_down=1,
        unit='Bohr',
    )
    # Bond axis (normalized) from atom 0 → atom 1.
    bond_vec = np.array(r1) - np.array(r0)
    bond_len = float(np.linalg.norm(bond_vec))
    bond_axis = bond_vec / bond_len
    return mol, np.array([r0, r1]), bond_axis, bond_len


def parse_symmops(s: str):
    s = s.strip()
    if s.lower() == 'auto':
        return 'auto'
    if s.lower() in ('none', ''):
        return None
    return [tok.strip() for tok in s.split(',') if tok.strip()]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--R', type=float, default=2.0,
                   help='H-H bond length in Bohr (default 2.0)')
    p.add_argument('--tilt', type=float, default=0.1,
                   help=('Transverse offset in Bohr that tilts the bond '
                         'off-axis (default 0.1, following the test). '
                         'Set to 0 only if you accept NaN in transverse '
                         'force components.'))
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--iters', type=int, default=200,
                   help='SR iterations (default 200)')
    p.add_argument('--walkers', type=int, default=128,
                   help='MC walkers for SR (12 GB GPU: keep <=256)')
    p.add_argument('--lr', type=float, default=1e-2)
    p.add_argument('--mc-timestep', type=float, default=0.1)
    p.add_argument('--jac-batch', type=int, default=4,
                   help='Jacobian batch size for SR (lower = less VRAM)')
    p.add_argument('--ansatz', type=str, default='psiformer_small.yaml',
                   help=("Ansatz config: builtin name ('psiformer', "
                         "'ferminet', 'paulinet', 'deeperwin') or a "
                         "YAML path. Default 'psiformer_small.yaml' is "
                         "a compact 68k-param PsiFormer suitable for "
                         "H2-class systems (~31x smaller than the "
                         "default 2.1M-param 'psiformer')."))
    p.add_argument('--prefix', type=str, default=None,
                   help=("Output stem. Default encodes R, tilt, and "
                         "ansatz so a checkpoint is not silently reused "
                         "across incompatible geometries/architectures."))
    p.add_argument('--eval-blocks', type=int, default=20,
                   help='Production blocks for final VMC eval')
    p.add_argument('--eval-walkers', type=int, default=256)
    p.add_argument('--steps-per-block', type=int, default=20,
                   help=('MCMC steps per production block. Lower this '
                         'when --symmops uses many ops, since gradient '
                         'cost scales as walkers*steps*N_combos.'))
    p.add_argument('--equil-blocks', type=int, default=5)
    p.add_argument('--skip-opt', action='store_true',
                   help='Skip SR training; use existing checkpoint')
    p.add_argument('--no-forces', action='store_true',
                   help='Skip ZVZB force evaluation and PGCS postproc')
    p.add_argument('--symmops', type=str, default='E,Rz180,i',
                   help=("PGCS symmetry ops. Default 'E,Rz180,i' "
                         "matches test_vmc_nn_h2_gradients_pgcs. "
                         "Use 'auto' for full detected set (slow), "
                         "'none' to disable PGCS."))
    args = p.parse_args()

    symmop_list = parse_symmops(args.symmops)
    compute_forces = not args.no_forces
    if args.prefix is None:
        ansatz_tag = os.path.basename(args.ansatz)
        ansatz_tag = os.path.splitext(ansatz_tag)[0]
        args.prefix = (
            f"h2_{ansatz_tag}_R{args.R:.3f}"
            f"_tilt{args.tilt:.3f}"
        ).replace('.', 'p')

    mol, coords, bond_axis, bond_len = build_h2_tilted(
        args.R, args.tilt,
    )
    print(f"H2 PsiFormer VMC + ZVZB/PGCS forces")
    print(f"  R             = {args.R} Bohr "
          f"(|r1-r0| = {bond_len:.6f})")
    print(f"  tilt          = {args.tilt} Bohr")
    print(f"  r0            = {coords[0]}")
    print(f"  r1            = {coords[1]}")
    print(f"  bond axis     = {bond_axis}")
    print(f"  PGCS symmops  = {symmop_list!r}")
    print(f"  compute forces= {compute_forces}")
    print(f"  prefix        = {args.prefix}")

    rng = jax.random.key(args.seed)
    init_key, opt_key, run_key = jax.random.split(rng, 3)

    ckpt = args.prefix + '.chk.h5'

    if not args.skip_opt:
        print(f"\n[1/3] SR optimization: {args.iters} iters, "
              f"{args.walkers} walkers, lr = {args.lr}")
        # SR training does not use PGCS — correlated sampling
        # helps the force variance, not the energy gradient.
        opt = get_vmcopt_nn_func(mol, args.ansatz, init_key)
        params_final, E_data = opt(
            opt_key,
            num_iters=args.iters,
            num_walkers=args.walkers,
            lr=args.lr,
            mc_timestep=args.mc_timestep,
            jac_batch_size=args.jac_batch,
            prefix=args.prefix,
            verbose=1,
        )
        print(f"\nSR training final: {E_data}")
    else:
        print("\n[1/3] Skipping SR optimization (--skip-opt)")
        if not os.path.exists(ckpt):
            print(f"WARNING: checkpoint '{ckpt}' not found; "
                  f"evaluating with random init (expect NaN/noisy forces).")

    print(f"\n[2/3] VMC evaluation ({args.eval_blocks} blocks × "
          f"{args.steps_per_block} steps × {args.eval_walkers} walkers)"
          f"  compute_gradients={compute_forces}")
    driver = get_vmc_nn_func(
        mol, args.ansatz, init_key,
        prefix=args.prefix,
        symmop_list=symmop_list,
    )
    if os.path.exists(ckpt):
        meta = driver.load_checkpoint(ckpt)
        print(f"Loaded checkpoint {ckpt}  "
              f"(epoch={int(meta.get('epoch', -1))}, "
              f"train E={float(meta.get('energy', float('nan'))):.6f} Ha)")

    result = driver(
        run_key,
        num_walkers=args.eval_walkers,
        num_steps_per_block=args.steps_per_block,
        num_blocks=args.eval_blocks,
        num_blocks_equil=args.equil_blocks,
        mc_timestep=args.mc_timestep,
        compute_gradients=compute_forces,
        verbose=1,
    )

    E_mean = float(result['E_mean'])
    E_serr = float(result['E_serr'])
    print("\n=== VMC energy ===")
    print(f"  E_mean = {E_mean: .6f} Ha")
    print(f"  E_serr = {E_serr: .6f} Ha")
    print(f"  (near-exact reference for H2 at R={args.R} Bohr: "
          f"see your favorite FCI table)")

    if compute_forces:
        print(f"\n[3/3] PGCS post-processing → forces from "
              f"{args.prefix}.grd.h5")
        F, dF = postproc_h5_pgcs(
            prefix=args.prefix,
            walker_based_batch_size=4,
        )
        F = np.asarray(F)
        dF = np.asarray(dF)
        print("\n=== ZVZB+PGCS nuclear forces (Ha/Bohr) ===")
        for i, (Fi, dFi) in enumerate(zip(F, dF)):
            print(f"  atom {i}:  "
                  f"Fx={Fi[0]:+.4e} ± {dFi[0]:.1e},  "
                  f"Fy={Fi[1]:+.4e} ± {dFi[1]:.1e},  "
                  f"Fz={Fi[2]:+.4e} ± {dFi[2]:.1e}")
        # By Newton 3 the two atomic forces should be
        # opposite — check it as a sanity metric.
        Fsum = F[0] + F[1]
        print(f"  Σ F_i   = {Fsum} "
              f"(ideally 0 by Newton's 3rd)")
        # Project onto bond axis: positive value means
        # atom 1 feels a force pointing +bond_axis (i.e.
        # away from atom 0 → bond wants to lengthen).
        F_bond_1 = float(np.dot(F[1], bond_axis))
        dF_bond_1 = float(np.sqrt(
            np.dot(dF[1]**2, bond_axis**2)
        ))
        # dE/dR = -F_bond (force is -dE/dR).
        dEdR = -F_bond_1
        print(f"\n  F · n̂_bond (on atom 1) = "
              f"{F_bond_1:+.4e} ± {dF_bond_1:.1e} Ha/Bohr")
        print(f"  ⇒ dE/dR ≈ {dEdR:+.4e} Ha/Bohr at R={args.R} Bohr")
        print(f"     (R=2.0 is stretched past eq; expect dE/dR > 0)")


if __name__ == '__main__':
    main()
