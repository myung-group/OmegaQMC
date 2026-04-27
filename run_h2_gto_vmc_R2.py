"""GTO VMC + Padé Jastrow + ZVZB/PGCS forces for H2 at R = 2.0 Bohr.

Pipeline (patterned on tests/test_H2.py and tests/test_LiH_opt.py):

  [1/4] SCF (PySCF RHF)  at R with a chosen basis set.
  [2/4] Linear-method Jastrow optimization
        (OmegaQMC.vmcopt_gto_linear).  J2 cusp parameters
        are frozen at the electron-electron cusp values
        (0.25 for like-spin, 0.5 for unlike-spin).
  [3/4] VMC evaluation with PGCS and
        ``compute_gradients=True``; writes per-walker
        gradient samples to ``<prefix>.grd.h5``.  Uses
        the ZVZB estimator with space-warp redistribution
        (``gr_scheme='scheme1'``, MO-based).
  [4/4] ``postproc_h5_pgcs`` → symmetry-averaged forces
        F_{ia} (Ha/Bohr) with statistical errors.

Unlike the NN path, the GTO force estimator uses space-
warp coordinate transformations, so an on-axis H2 does
NOT trigger the ``log|h|`` NaN that the NN path does —
no geometry tilt needed.

Usage:
    python run_h2_gto_vmc_R2.py                  # defaults: 6-31G
    python run_h2_gto_vmc_R2.py --basis cc-pvdz
    python run_h2_gto_vmc_R2.py --skip-opt       # use stock Jastrow
    python run_h2_gto_vmc_R2.py --no-forces      # energy only
    python run_h2_gto_vmc_R2.py --symmops E,C2z
"""

import argparse
import os

os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
os.environ.setdefault('XLA_PYTHON_CLIENT_ALLOCATOR', 'platform')

import jax
import jax.numpy as jnp
import numpy as np

from OmegaQMC import generate_molecular_orbitals, get_vmc_gto_func
from OmegaQMC.utils import format_basis_name
from OmegaQMC.vmcopt_gto_linear import get_vmcopt_gto_func
from OmegaQMC.observables.force import postproc_h5_pgcs


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
    p.add_argument('--basis', type=str, default='6-31G',
                   help=('Gaussian basis set (PySCF name). '
                         'Examples: 6-31G, cc-pVDZ, cc-pVTZ, '
                         'aug-cc-pVTZ.'))
    p.add_argument('--seed', type=int, default=888)

    # Jastrow optimization
    p.add_argument('--epochs', type=int, default=20,
                   help='Linear-method optimization epochs (default 20)')
    p.add_argument('--opt-walkers', type=str, default='auto',
                   help=("Walkers for Jastrow opt. 'auto' queries free "
                         "GPU mem."))
    p.add_argument('--opt-samples', type=str, default='auto',
                   help=("Total opt samples per epoch; QMCPACK uses "
                         "16k-64k. 'auto' = 5 x walkers."))
    p.add_argument('--skip-opt', action='store_true',
                   help=('Skip Jastrow optimization; use hard-coded '
                         'good starting values from test_H2.py.'))

    # VMC evaluation
    p.add_argument('--eval-walkers', type=int, default=256)
    p.add_argument('--eval-blocks', type=int, default=20)
    p.add_argument('--steps-per-block', type=int, default=100)
    p.add_argument('--equil-blocks', type=int, default=5)
    p.add_argument('--mc-timestep', type=float, default=0.1)

    # Forces
    p.add_argument('--no-forces', action='store_true',
                   help='Skip force evaluation.')
    p.add_argument('--symmops', type=str, default='E,C2z',
                   help=("PGCS symmetry ops (default 'E,C2z' as in "
                         "test_H2.py). Use 'auto' for full detected "
                         "set, 'none' for identity only, or a comma "
                         "list (e.g. 'E,i,Rz180')."))
    p.add_argument('--gr-scheme', type=str, default='scheme1',
                   choices=['scheme1', 'scheme2'],
                   help=('Space-warp redistribution. scheme1 = MO-based, '
                         'scheme2 = distance-based r^-4.'))
    p.add_argument('--force-estimator', type=str, default='simple',
                   choices=['simple', 'zvzb2'],
                   help=('simple = standard ZVZB. zvzb2 = ZVZB with '
                         'parameter-response variance reduction.'))
    p.add_argument('--cusp-scheme', type=str, default='Quady2025',
                   help="Cusp correction scheme (default Quady2025).")

    p.add_argument('--prefix', type=str, default=None,
                   help=('Output stem. Default encodes R and basis so '
                         'checkpoints from different runs do not '
                         'collide.'))
    args = p.parse_args()

    if args.prefix is None:
        args.prefix = (
            f"h2_gto_{format_basis_name(args.basis)}"
            f"_R{args.R:.3f}"
        ).replace('.', 'p')

    symmop_list = parse_symmops(args.symmops)
    compute_forces = not args.no_forces

    print(f"H2 GTO VMC + Jastrow + forces")
    print(f"  R              = {args.R} Bohr")
    print(f"  basis          = {args.basis}")
    print(f"  PGCS symmops   = {symmop_list!r}")
    print(f"  force estimator= {args.force_estimator}")
    print(f"  space-warp     = {args.gr_scheme}")
    print(f"  cusp scheme    = {args.cusp_scheme}")
    print(f"  prefix         = {args.prefix}")

    # [1/4] SCF ---------------------------------------------------------
    atoms_string = (
        "H  0.0  0.0  {:.6f}   1\n"
        "H  0.0  0.0  {:.6f}   1\n"
    ).format(-args.R / 2, args.R / 2)
    print(f"\n[1/4] SCF ({args.basis})")
    modrv = generate_molecular_orbitals(
        atoms_string, units='Bohr', basis=args.basis,
    )

    # Jastrow parameter init.  Values from test_H2.py — optimized
    # for H2 at R=1.401, but a reasonable warm start at R=2.0; the
    # linear-method optimizer refines them quickly.
    params_jastrow_init = {
        "J1_pade": {"H": jnp.array([-0.05574627, 0.08272289])},
        "J2_pade": {
            "like": jnp.array([0.25, 0.6046799]),
            "unlike": jnp.array([0.5, 0.38077791]),
        },
    }
    # Freeze the e-e cusp constants (like=0.25, unlike=0.5) — these
    # enforce the Kato cusp condition analytically and must not be
    # optimized.
    frozen_keys = {
        "J2_pade": {"like": [0], "unlike": [0]},
    }

    rng = jax.random.key(args.seed)
    opt_key, run_key = jax.random.split(rng, 2)

    # [2/4] Jastrow optimization ----------------------------------------
    if args.skip_opt:
        print("\n[2/4] Skipping Jastrow optimization (--skip-opt)")
        params_jastrow = params_jastrow_init
    else:
        print(f"\n[2/4] Jastrow optimization: "
              f"{args.epochs} epochs, linear method")
        vmcopt_run = get_vmcopt_gto_func(modrv)
        opt_walkers = (args.opt_walkers
                       if args.opt_walkers == 'auto'
                       else int(args.opt_walkers))
        opt_samples = (args.opt_samples
                       if args.opt_samples == 'auto'
                       else int(args.opt_samples))
        params_jastrow, E_data = vmcopt_run(
            opt_key,
            params_corr_init=params_jastrow_init,
            frozen_keys=frozen_keys,
            num_epochs=args.epochs,
            num_walkers=opt_walkers,
            num_opt_samples=opt_samples,
            verbose=1,
        )
        print("\nOptimized Jastrow params:")
        for k, v in params_jastrow.items():
            print(f"  {k}: {v}")
        print(f"Opt energies (per epoch): {E_data}")

    # [3/4] VMC with forces ---------------------------------------------
    print(f"\n[3/4] VMC evaluation ({args.eval_blocks} blocks × "
          f"{args.steps_per_block} steps × {args.eval_walkers} walkers)"
          f"  compute_gradients={compute_forces}")
    vmc_run = get_vmc_gto_func(
        modrv, params_jastrow,
        cusp_scheme=args.cusp_scheme,
        gr_scheme=args.gr_scheme,
        force_estimator=args.force_estimator,
        prefix=args.prefix,
        symmop_list=symmop_list,
    )
    vmc_run(
        run_key,
        num_walkers=args.eval_walkers,
        num_steps_per_block=args.steps_per_block,
        num_blocks=args.eval_blocks,
        num_blocks_equil=args.equil_blocks,
        mc_timestep=args.mc_timestep,
        compute_gradients=compute_forces,
    )

    # [4/4] Force post-processing ---------------------------------------
    if compute_forces:
        print(f"\n[4/4] PGCS post-processing from "
              f"{args.prefix}.grd.h5")
        F, dF = postproc_h5_pgcs(prefix=args.prefix)
        F = np.asarray(F)
        dF = np.asarray(dF)
        print("\n=== GTO VMC ZVZB+PGCS forces (Ha/Bohr) ===")
        for i, (Fi, dFi) in enumerate(zip(F, dF)):
            print(f"  atom {i}:  "
                  f"Fx={Fi[0]:+.4e} ± {dFi[0]:.1e},  "
                  f"Fy={Fi[1]:+.4e} ± {dFi[1]:.1e},  "
                  f"Fz={Fi[2]:+.4e} ± {dFi[2]:.1e}")
        Fsum = F[0] + F[1]
        print(f"  Σ F_i  = {Fsum} "
              f"(ideally 0 by Newton's 3rd)")
        # H2 on z-axis: only F_z is physical. F_{0,z} positive
        # means atom 0 is pulled in +z → bond wants to shorten.
        Fz0 = F[0, 2]
        dFz0 = dF[0, 2]
        dEdR = -Fz0  # F on atom 0 along +z equals -dE/dR at R/2 shift
        print(f"\n  F_{{0,z}}  = {Fz0:+.4e} ± {dFz0:.1e} Ha/Bohr")
        print(f"  dE/dR ≈ {dEdR:+.4e} Ha/Bohr (at R={args.R} Bohr)")
        print(f"     (R=2.0 Bohr is stretched from eq ~1.4 Bohr; "
              f"expect dE/dR > 0.)")


if __name__ == '__main__':
    main()
