#!/usr/bin/env python
"""Compare local 2D HEG rs-scan results to Attaccalite 2002 references.

Usage:
    python scripts/compare_rs_scan.py
"""

import json
from pathlib import Path
import numpy as np

from OmegaQMC.heg_2d import (
    build_2deg_system,
    hf_energy_2d_finite,
    hf_energy_2d_td,
)


# Attaccalite 2002 PRL 88, 256601, FN-DMC backflow at N=58 unpolarized.
# These are the reference values our PsiFormer at converged TD limit
# should approach.
ATTACCALITE_BACKFLOW_UNPOL = {
    1:  -0.20372,
    2:  -0.25721,
    5:  -0.149518,
    10: -0.085427,
    20: -0.046385,
    30: -0.031941,
}


def load_summary(rs):
    """Load per-rs summary.json."""
    path = (
        Path('runs')
        / f'heg2d_rs{rs}_N10_unpol_500iter'
        / 'summary.json'
    )
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def main():
    rs_values = [1, 2, 5, 10, 20]

    print()
    print('=' * 92)
    print(f'2D HEG rs scan — local CPU run, N=10 unpolarized, 500 SR iters')
    print('=' * 92)
    print()
    print(
        f'{"rs":>4} | {"PsiFormer E/N":>18} | '
        f'{"HF (analytic, N=10)":>20} | {"corr (mHa)":>10} | '
        f'{"Attaccalite N=58":>16} | {"diff (mHa)":>10}'
    )
    print('-' * 92)

    rows = []
    for rs in rs_values:
        s = load_summary(rs)
        attac = ATTACCALITE_BACKFLOW_UNPOL.get(rs)

        if s is None:
            print(f'{rs:>4} | (no summary.json yet)')
            continue

        e_vmc = s['e_vmc_ha']
        serr = s['e_vmc_serr_ha']
        e_hf = s['e_hf_ha']
        corr = e_vmc - e_hf
        diff_attac = e_vmc - attac if attac is not None else None

        print(
            f'{rs:>4} | '
            f'{e_vmc:>+11.5f} +/- {serr:.5f} | '
            f'{e_hf:>20.5f} | '
            f'{corr*1000:>+10.2f} | '
            f'{attac:>+16.5f} | '
            f'{diff_attac*1000:>+10.2f}'
            if diff_attac is not None else
            f'{rs:>4} | '
            f'{e_vmc:>+11.5f} +/- {serr:.5f} | '
            f'{e_hf:>20.5f} | '
            f'{corr*1000:>+10.2f} |       (no ref) |'
        )
        rows.append({
            'rs': rs, 'e_vmc': e_vmc, 'serr': serr,
            'e_hf_finite': e_hf, 'corr_finite': corr,
            'attac_dmc': attac, 'diff_attac': diff_attac,
        })

    print('-' * 92)
    print()
    print('Notes:')
    print(
        '  * "PsiFormer E/N":  500-iter SR-VMC, N=10, embedding_dim=32, '
        'n_det=4'
    )
    print(
        '  * "HF (analytic, N=10)": closed-form 2D Hartree-Fock at the same '
        'finite N=10'
    )
    print(
        '  * "corr": (PsiFormer) - (HF) = correlation energy at this finite N'
    )
    print(
        '  * "Attaccalite": reference backflow-DMC at N=58 (different '
        'finite size)'
    )
    print(
        '  * "diff (mHa)": PsiFormer at N=10 minus Attaccalite at N=58 '
        '(includes finite-size effect, NOT a variational gap)'
    )
    print()

    # Correlation comparison vs Attaccalite TD-limit correlation
    print('-' * 92)
    print('Correlation energy: ours (N=10) vs Attaccalite TD limit')
    print('-' * 92)
    print(
        f'{"rs":>4} | {"corr ours (mHa)":>18} | '
        f'{"corr TD (mHa)":>15} | {"% recovered":>13}'
    )
    print('-' * 92)
    for row in rows:
        rs = row['rs']
        attac = row['attac_dmc']
        if attac is None:
            continue
        e_hf_td = hf_energy_2d_td(float(rs), 'unpolarized')
        corr_td = attac - e_hf_td   # Attaccalite TD - HF TD
        # Note: Attaccalite reports at N=58 not TD; their N=58 is very
        # close to TD for the fluid phase, so we treat as ~TD.
        pct = 100.0 * row['corr_finite'] / corr_td
        print(
            f'{rs:>4} | '
            f'{row["corr_finite"]*1000:>+18.2f} | '
            f'{corr_td*1000:>+15.2f} | '
            f'{pct:>12.1f}%'
        )
    print('-' * 92)


if __name__ == '__main__':
    main()
