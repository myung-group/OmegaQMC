#!/usr/bin/env python
"""Compare fluid vs Wigner-crystal sector energies at low density.

Reads runs/heg2d_rs{rs}_N10_unpol_{fluid,crystal}/summary.json and
prints a table + identifies the energy crossing E_fluid(rs) =
E_crystal(rs).  Drummond-Needs 2009 reports rs_c = 31(1) for the
unpolarised 2D HEG.
"""

import json
from pathlib import Path
import numpy as np


def load(rs, sector):
    path = (
        Path('runs')
        / f'heg2d_rs{rs}_N10_unpol_{sector}'
        / 'summary.json'
    )
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def main():
    rs_values = [25, 30, 35, 40]
    print()
    print('=' * 80)
    print(
        'Fluid vs Wigner-crystal energy comparison (N=10, light settings).'
    )
    print('=' * 80)
    print()
    print(
        f'{"rs":>4} | '
        f'{"E fluid (Ha/elec)":>22} | '
        f'{"E crystal (Ha/elec)":>22} | '
        f'{"E_xtl-E_flu (mHa)":>18}'
    )
    print('-' * 80)

    deltas = []
    for rs in rs_values:
        f = load(rs, 'fluid')
        c = load(rs, 'crystal')
        if f is None or c is None:
            print(f'{rs:>4} | (incomplete data)')
            continue
        ef, sf = f['e_vmc_ha'], f['e_vmc_serr_ha']
        ec, sc = c['e_vmc_ha'], c['e_vmc_serr_ha']
        delta = ec - ef
        sd = np.sqrt(sf ** 2 + sc ** 2)
        sigma_units = delta / sd if sd > 0 else float('nan')
        print(
            f'{rs:>4} | '
            f'{ef:>+11.5f} +/- {sf:.5f} | '
            f'{ec:>+11.5f} +/- {sc:.5f} | '
            f'{delta*1000:>+10.3f} ({sigma_units:+.1f}sigma)'
        )
        deltas.append((rs, delta, sd))

    print('-' * 80)
    print()
    print('Interpretation: E_crystal - E_fluid')
    print('  > 0: fluid is variationally lower -> fluid phase')
    print('  < 0: crystal is variationally lower -> Wigner-crystal phase')
    print('  ~ 0: near phase boundary (rs_c, ~31 from Drummond-Needs 2009)')
    print()

    # If the data has a sign change, do a linear interpolation to
    # estimate r_s^c.
    if len(deltas) >= 2:
        for i in range(len(deltas) - 1):
            rs1, d1, s1 = deltas[i]
            rs2, d2, s2 = deltas[i + 1]
            if (d1 > 0) != (d2 > 0):
                # Linear interp: rs_c at delta = 0
                rs_c = rs1 + (rs2 - rs1) * (-d1) / (d2 - d1)
                print(
                    f'  Sign change between rs={rs1} and rs={rs2}: '
                    f'linear-interp rs_c ~= {rs_c:.1f}'
                )
                print(
                    f'  Drummond-Needs 2009 reference: rs_c = 31(1)'
                )


if __name__ == '__main__':
    main()
