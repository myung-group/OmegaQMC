"""H2 dissociation curve via standard FCI (lambda=0) and QED-FCI (lambda>0).

Phase H2-A reference: bare dissociation curve to compare against
NN-VMC tang_native. Cheap (FCI for H2 in aug-cc-pVDZ runs in seconds).

Output: CSV at logs/h2_dissoc/fci_baseline.csv
"""
from __future__ import annotations

import csv
import os
import os.path as osp
import time

import numpy as np
from pyscf import gto, scf

from OmegaQMC.qed_fci import qed_fci


_R_GRID = [1.0, 1.4, 2.0, 2.5, 3.0, 4.0, 5.0]   # Bohr
_BASIS = 'aug-cc-pvdz'
_OMEGA = 0.46672    # Tang's ~12.7 eV (irrelevant at lambda=0)
_NPH_MAX = 6
_LAMBDA = 0.0


def _h2_mol(R_bohr: float):
    return gto.M(
        atom=f'H 0 0 -{R_bohr/2}; H 0 0 {R_bohr/2}',
        unit='Bohr', basis=_BASIS, verbose=0,
    )


def main():
    out_dir = 'logs/h2_dissoc'
    os.makedirs(out_dir, exist_ok=True)
    csv_path = osp.join(out_dir, 'fci_baseline.csv')

    print(f'H2 dissociation curve, basis={_BASIS}, lambda={_LAMBDA}')
    print(f'{"R (Bohr)":>10}  {"E_HF":>14}  {"E_FCI":>14}  {"corr (mHa)":>12}  {"time (s)":>10}')
    rows = []
    for R in _R_GRID:
        t0 = time.time()
        mol = _h2_mol(R)
        mf = scf.RHF(mol).run(verbose=0)
        e_hf = float(mf.e_tot)
        eps = np.array([0., 0., 1.])
        res = qed_fci(
            mf, omega=_OMEGA, coupling_vec=_LAMBDA * eps,
            nph_max=_NPH_MAX, proper_dse=True,
        )
        e_fci = float(res['e_qed_fci'])
        corr_mHa = (e_fci - e_hf) * 1000.0
        dt = time.time() - t0
        print(f'{R:>10.4f}  {e_hf:>14.6f}  {e_fci:>14.6f}  {corr_mHa:>12.3f}  {dt:>10.2f}')
        rows.append((R, e_hf, e_fci, res.get('n_photon', 0.0)))

    with open(csv_path, 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['R_bohr', 'E_HF_Ha', 'E_FCI_Ha', 'n_photon'])
        for r in rows:
            w.writerow(r)
    print(f'\nwrote {csv_path}')


if __name__ == '__main__':
    main()
