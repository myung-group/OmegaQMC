#!/usr/bin/env python
"""Verify the Wigner-crystal transition by comparing fluid vs crystal
energies at rs=20 (control: fluid phase) and rs=40 (test: WC phase).

Usage:  python scripts/compare_wc_transition_v2.py
"""

import json
from pathlib import Path
import numpy as np


def load(rs, sector):
    path = (
        Path('runs')
        / f'heg2d_rs{rs}_N18_unpol_{sector}_v2'
        / 'summary.json'
    )
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def main():
    print()
    print('=' * 80)
    print(
        'WC transition test: fluid vs crystal at rs=20 (control) and '
        'rs=40 (test)'
    )
    print('N=18 unpolarized, 2000 SR iters, 512 walkers, sigma_init=0.15')
    print('=' * 80)
    print()
    print(
        f'{"rs":>4} | '
        f'{"E fluid (Ha/elec)":>22} | '
        f'{"E crystal (Ha/elec)":>22} | '
        f'{"E_xtl-E_flu (mHa)":>20}'
    )
    print('-' * 80)

    results = {}
    for rs in [20, 40]:
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
        marker = ''
        if delta > 2 * sd:
            marker = '  <- FLUID WINS'
        elif delta < -2 * sd:
            marker = '  <- CRYSTAL WINS'
        print(
            f'{rs:>4} | '
            f'{ef:>+11.5f} +/- {sf:.5f} | '
            f'{ec:>+11.5f} +/- {sc:.5f} | '
            f'{delta*1000:>+10.3f} ({sigma_units:+.1f}sig){marker}'
        )
        results[rs] = (delta, sd)

    print('-' * 80)
    print()

    if 20 in results and 40 in results:
        d20, s20 = results[20]
        d40, s40 = results[40]
        print('Transition verified if:')
        print('  rs=20: fluid lower (delta_20 > 0)')
        print('  rs=40: crystal lower (delta_40 < 0)')
        print()
        print(f'  rs=20 result: delta = {d20*1000:+.2f} +/- {s20*1000:.2f} mHa  '
              f'-> {"fluid wins" if d20 > 0 else "CRYSTAL WINS (unexpected)"}')
        print(f'  rs=40 result: delta = {d40*1000:+.2f} +/- {s40*1000:.2f} mHa  '
              f'-> {"CRYSTAL WINS (transition!)" if d40 < 0 else "fluid still wins"}')
        print()
        if d20 > 0 and d40 < 0:
            print('  TRANSITION VERIFIED: fluid -> WC ordering as rs grows.')
            print('  Drummond-Needs 2009 reference: rs_c = 31(1).')
        elif d20 > 0 and d40 > 0:
            print('  Crystal still above fluid at rs=40 -- need:')
            print('    (a) more iters (2000 -> 5000),')
            print('    (b) smaller sigma_init (0.15 -> 0.10), or')
            print('    (c) larger N (18 -> 32).')


if __name__ == '__main__':
    main()
