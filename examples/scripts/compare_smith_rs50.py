#!/usr/bin/env python
"""Compare fluid vs crystal at rs=50, N=58 (Smith 2024 spot-check size).

Reports:
  * variational energies + Bonsall-Maradudin static-WC reference
  * S(k) at the triangular reciprocal-lattice points (Bragg-peak signal
    of WC translational order)

Usage:  python scripts/compare_smith_rs50.py
"""

import json
from pathlib import Path
import numpy as np


def load(sector):
    path = (
        Path('runs')
        / f'heg2d_rs50_N58_unpol_{sector}_smith'
        / 'summary.json'
    )
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def main():
    f = load('fluid')
    c = load('crystal')

    print()
    print('=' * 80)
    print('rs=50, N=58 unpolarized (square cell, Smith 2024 spot-check size)')
    print('=' * 80)

    if f is None or c is None:
        print('Missing data:')
        print(f'  fluid:   {"loaded" if f else "MISSING runs/heg2d_rs50_N58_unpol_fluid_smith/"}')
        print(f'  crystal: {"loaded" if c else "MISSING runs/heg2d_rs50_N58_unpol_crystal_smith/"}')
        return

    rs = 50.0
    eps_M_BM_tri = -1.106103 / rs   # Bonsall-Maradudin triangular Madelung

    print()
    print('=== Energies (Ha/electron) ===')
    print(f'  Fluid VMC        = {f["e_vmc_ha"]:+.6f} +/- {f["e_vmc_serr_ha"]:.6f}')
    print(f'  Crystal VMC      = {c["e_vmc_ha"]:+.6f} +/- {c["e_vmc_serr_ha"]:.6f}')
    print(f'  HF (analytic)    = {f["e_hf_ha"]:+.6f}')
    print(f'  BM static WC ref = {eps_M_BM_tri:+.6f}  (-1.106103 / r_s)')
    print()

    delta = c['e_vmc_ha'] - f['e_vmc_ha']
    sd = np.sqrt(c['e_vmc_serr_ha'] ** 2 + f['e_vmc_serr_ha'] ** 2)
    sigma_units = delta / sd if sd > 0 else float('nan')
    print(f'  Delta (E_xtl - E_flu) = {delta * 1000:+.3f} mHa  '
          f'({sigma_units:+.1f} sigma)')
    if delta < -2 * sd:
        print('  CRYSTAL CLEARLY LOWER -> Wigner-crystal phase')
    elif delta > 2 * sd:
        print('  Fluid clearly lower (unexpected at rs=50; check convergence)')
    else:
        print('  Within statistical uncertainty')

    print()
    print('=== Structure factor S(k) ===')
    print('Bragg peaks at triangular reciprocal-lattice vectors should be')
    print('large for crystal, ~1 for fluid.')
    print()

    for sector_name, summary in [('Fluid', f), ('Crystal', c)]:
        obs = summary.get('observables', {})
        sk_tri = obs.get('sk_triangular_bragg')
        if sk_tri is None:
            print(f'  {sector_name}: no S(k) data')
            continue
        kvecs = np.array(sk_tri['k_vectors'])
        means = np.array(sk_tri['mean'])
        serrs = np.array(sk_tri['serr'])
        print(f'  --- {sector_name} ---')
        # First-shell average (vectors with smallest |k|)
        k_norms = np.linalg.norm(kvecs, axis=-1)
        order = np.argsort(k_norms)
        # Group by shell (rounded |k|)
        shells_seen = []
        for i in order:
            knrm = round(float(k_norms[i]), 4)
            if knrm not in shells_seen:
                shells_seen.append(knrm)
                shell_idx = np.where(np.abs(k_norms - k_norms[i]) < 1e-6)[0]
                shell_mean = float(np.mean(means[shell_idx]))
                shell_serr = (
                    float(np.sqrt(np.sum(serrs[shell_idx] ** 2)))
                    / np.sqrt(len(shell_idx))
                )
                print(
                    f'    |k|={knrm:.4f} (shell of {len(shell_idx)}): '
                    f'<S(k)> = {shell_mean:.3f} +/- {shell_serr:.3f}'
                )

    print()
    print('=' * 80)
    print('Reference values:')
    print(f'  BM static-WC Madelung at r_s=50: {eps_M_BM_tri:+.6f} Ha/elec')
    print(f'  Smith 2024 (PRL 133, 266504): r_s_c = 37 +/- 1')
    print(f'  Drummond-Needs 2009: r_s_c = 31 +/- 1')
    print('=' * 80)


if __name__ == '__main__':
    main()
