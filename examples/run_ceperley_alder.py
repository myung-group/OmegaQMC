#!/usr/bin/env python3
"""
Reproduce ground-state energies of the 3D homogeneous electron gas
from Ceperley & Alder, PRL 45, 566 (1980) using phaseless AFQMC.

Computes the total energy per electron at various r_s values for both
paramagnetic (unpolarized) and ferromagnetic (fully polarized) states.

Usage:
    python scripts/run_ceperley_alder.py [--quick]

The --quick flag uses fewer walkers/blocks for fast testing.
"""

import sys
import time
import numpy as np

from OmegaQMC.afqmc_pw_heg import (
    build_3deg_system,
    get_afqmc_3deg_func,
    run_twist_averaged_afqmc_3deg,
    hf_kinetic_energy_per_elec_ry,
    hf_exchange_energy_per_elec_ry,
    pz_correlation_energy,
)


def run_single_rs(rs, N_elec, N_pw, polarization, dt, num_walkers,
                  num_blocks, num_steps_per_block, num_eqlb_blocks,
                  n_twists):
    """Run AFQMC for a single rs value."""
    print(f"\n{'='*70}")
    print(f"  r_s = {rs}, N = {N_elec}, polarization = {polarization}")
    print(f"{'='*70}")

    base_system = build_3deg_system(
        rs=rs, N_elec=N_elec, N_pw=N_pw, polarization=polarization)

    if n_twists > 1:
        result = run_twist_averaged_afqmc_3deg(
            base_system, n_twists=n_twists, dt=dt,
            num_walkers=num_walkers, num_blocks=num_blocks,
            num_steps_per_block=num_steps_per_block,
            num_eqlb_blocks=num_eqlb_blocks, verbose=True)
        e_per_elec_ry = result['e_per_elec_ry']
        e_err_ry = result['energy_err'] / N_elec * 2.0
        # Use twist-averaged HF energy so FS errors cancel with QMC
        e_hf_ry = result['e_trial_per_elec_ry']
    else:
        driver = get_afqmc_3deg_func(
            base_system, dt=dt, include_coulomb=True, verbose=True)
        result = driver(
            num_walkers=num_walkers, num_blocks=num_blocks,
            num_steps_per_block=num_steps_per_block,
            num_eqlb_blocks=num_eqlb_blocks)
        e_per_elec_ry = result['e_per_elec_ry']
        e_err_ry = result['energy_err'] / N_elec * 2.0
        e_hf_ry = driver.e_trial / N_elec * 2.0

    # Correlation energy = E_AFQMC - E_HF (both twist-averaged)
    # FS errors in kinetic/exchange cancel between QMC and HF
    e_corr_ry = e_per_elec_ry - e_hf_ry

    # PZ81 parametrization of CA correlation energy (Hartree → Ry)
    pol_key = 'unpolarized' if polarization == 'unpolarized' else 'polarized'
    e_ca_corr = 2.0 * pz_correlation_energy(rs, pol_key)

    print(f"\n  E_HF/N (TABC) = {e_hf_ry:.6f} Ry")
    print(f"  E_tot/N (QMC)  = {e_per_elec_ry:.6f} +/- {e_err_ry:.6f} Ry")
    print(f"  E_c/N          = {e_corr_ry:.6f} Ry")
    print(f"  CA/PZ ref      = {e_ca_corr:.6f} Ry")
    print(f"  diff           = {e_corr_ry - e_ca_corr:.6f} Ry")

    return {
        'rs': rs,
        'polarization': polarization,
        'e_per_elec_ry': e_per_elec_ry,
        'e_err_ry': e_err_ry,
        'e_hf_ry': e_hf_ry,
        'e_corr_ry': e_corr_ry,
        'e_ca_ref': e_ca_corr,
    }


def main():
    quick = '--quick' in sys.argv

    # Parameters

    N_elec_unpol = 14  # closed shell for 3D: (1+6)*2 = 14
    N_elec_pol = 7     # polarized: 7 electrons, all spin-up
    #N_elec_pol = 19
    #N_elec_unpol = N_elec_pol * 2
    N_pw = 57           # ~3 complete shells beyond Fermi level

    if quick:
        print("QUICK MODE: reduced statistics for testing")
        rs_values = [1, 2]
        dt = 0.005
        num_walkers = 200
        num_blocks = 50
        num_steps_per_block = 25
        num_eqlb_blocks = 20
        n_twists = 10  # no twist averaging for quick test
    else:
        rs_values = [1, 2, 5, 10, 20]
        dt = 0.005
        num_walkers = 200
        num_blocks = 100
        num_steps_per_block = 25
        num_eqlb_blocks = 20
        n_twists = 50

    print("=" * 70)
    print("  Ceperley-Alder Reproduction via Phaseless AFQMC")
    print("  3D Homogeneous Electron Gas (Jellium)")
    print("=" * 70)
    print(f"  N_elec (unpol) = {N_elec_unpol}")
    print(f"  N_elec (pol)   = {N_elec_pol}")
    print(f"  N_pw           = {N_pw}")
    print(f"  dt             = {dt}")
    print(f"  num_walkers    = {num_walkers}")
    print(f"  num_blocks     = {num_blocks}")
    print(f"  n_twists       = {n_twists}")
    print(f"  r_s values     = {rs_values}")

    t_start = time.time()

    # Run paramagnetic calculations
    results_para = []
    for rs in rs_values:
        # Use larger system and more statistics for strongly correlated regime
        #if not quick and rs >= 10:
        #    n_elec = 38   # next closed shell: better finite-size scaling
        #    n_pw = 123    # ~3 shells beyond Fermi level for N=38
        #    nw = 400
        #    nb = 200
        #else:
        n_elec = N_elec_unpol
        n_pw = N_pw
        nw = num_walkers
        nb = num_blocks

        r = run_single_rs(
            rs, n_elec, n_pw, 'unpolarized', dt,
            nw, nb, num_steps_per_block,
            num_eqlb_blocks, n_twists)
        results_para.append(r)

    # Run ferromagnetic calculations
    results_ferro = []
    for rs in rs_values:
        #if not quick and rs >= 10:
        #    n_elec = 19   # polarized closed shell matching N=38 unpol
        #    n_pw = 123
        #    nw = 400
        #    nb = 200
        #else:
        n_elec = N_elec_pol
        n_pw = N_pw
        nw = num_walkers
        nb = num_blocks

        r = run_single_rs(
            rs, n_elec, n_pw, 'polarized', dt,
            nw, nb, num_steps_per_block,
            num_eqlb_blocks, n_twists)
        results_ferro.append(r)

    # Summary table
    print("\n" + "=" * 80)
    print("  SUMMARY: Paramagnetic (unpolarized) state")
    print("=" * 80)
    print(f"  {'r_s':>5} {'E_c/N (Ry)':>12} {'CA/PZ':>10} {'Diff':>10}")
    print("-" * 45)
    for r in results_para:
        diff = r['e_corr_ry'] - r['e_ca_ref']
        print(f"  {r['rs']:5.1f} {r['e_corr_ry']:12.6f} "
              f"{r['e_ca_ref']:10.4f} {diff:10.6f}")

    print("\n" + "=" * 80)
    print("  SUMMARY: Ferromagnetic (polarized) state")
    print("=" * 80)
    print(f"  {'r_s':>5} {'E_c/N (Ry)':>12} {'CA/PZ':>10} {'Diff':>10}")
    print("-" * 45)
    for r in results_ferro:
        diff = r['e_corr_ry'] - r['e_ca_ref']
        print(f"  {r['rs']:5.1f} {r['e_corr_ry']:12.6f} "
              f"{r['e_ca_ref']:10.4f} {diff:10.6f}")

    elapsed = time.time() - t_start
    print(f"\nTotal runtime: {elapsed:.1f} s")


if __name__ == '__main__':
    main()
