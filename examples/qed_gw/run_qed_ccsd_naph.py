"""QED-CCSD ground-state energy of naphthalene/cc-pVDZ in an optical cavity.

Uses the spin-adapted closed-shell module qed_ccsd_rhf (Wick-derived DF
equations spin-traced to spatial orbitals, integral-block intermediates
built once per run): full QED-CCSD-21 fits in ~1.5 GB here, versus
~11 GB for a spin-orbital DF implementation of the same model.

The density-fitted equations are what make this molecule/basis reachable
at all: integral storage is naux*nmo^2 instead of the nso^4 (~134 GB)
a dense antisymmetrised spin-orbital ERI would require.

Model: QED-CCSD-21 (Deprince): t1_10 + t2_20 + t1_01 + t2_11 + t2_21.

Cavity: z-polarized (short in-plane axis of the acene), omega = 3 eV,
lambda = 0.05. Frozen core: 10 C 1s orbitals.

Writes qed_ccsd_naph_results.json next to this script.
"""
import json
import math
import os
import resource
import time

import numpy as np
from pyscf import gto

from OmegaQMC.addons.qed_hf import run_qed_hf
from OmegaQMC.addons import qed_ccsd_rhf

EV = 27.211386245988
BASIS = 'cc-pVDZ'
AUXBASIS = 'cc-pvdz-ri'
OMEGA_EV = 3.0
LAMBDA = (0.0, 0.0, 0.05)
FROZEN = 10              # C 1s cores
DO_T2_21 = True          # full QED-CCSD-21 (Deprince)


def acene(n_rings, aCC=1.40, aCH=1.09):
    """Idealized planar linear acene in the x-z plane (short axis = z),
    same construction as run_qed_absorption_naph.py."""
    ang = np.deg2rad([30, 90, 150, 210, 270, 330])
    cx0 = (n_rings - 1) / 2.0 * math.sqrt(3.0) * aCC
    verts = []
    for ir in range(n_rings):
        cx = ir * math.sqrt(3.0) * aCC - cx0
        for a in ang:
            v = (round(cx + aCC * math.cos(a), 6), round(aCC * math.sin(a), 6))
            if v not in verts:
                verts.append(v)
    atoms = [('C', x, z) for (x, z) in verts]
    ch = []
    for (x, z) in verts:
        # H on carbons with only two C neighbours
        nn = sum(1 for (x2, z2) in verts
                 if 0 < math.hypot(x - x2, z - z2) < 1.05 * aCC)
        if nn == 2:
            r = math.hypot(x, z)
            ch.append(('H', x * (1 + aCH / r), z * (1 + aCH / r)))
    return [(el, (x, 0.0, z)) for (el, x, z) in atoms + ch]


def main():
    geom = acene(2)
    assert sum(1 for el, _ in geom if el == 'C') == 10
    assert sum(1 for el, _ in geom if el == 'H') == 8
    mol = gto.M(atom=[(el, xyz) for el, xyz in geom], basis=BASIS,
                unit='Angstrom', symmetry=False, verbose=0)
    omega = OMEGA_EV / EV
    print(f"naphthalene / {BASIS}: nao = {mol.nao}, nelec = {mol.nelectron}")
    print(f"cavity: omega = {OMEGA_EV} eV, lambda = {LAMBDA}, "
          f"auxbasis = {AUXBASIS}, frozen = {FROZEN}")

    t0 = time.time()
    qedhf = run_qed_hf(mol, omega, LAMBDA, verbose=False, auxbasis=AUXBASIS)
    print(f"\nE_QED_HF = {qedhf['E_qed_hf']:.10f}  "
          f"(E_RHF = {qedhf['E_rhf']:.10f})   [{time.time()-t0:.1f} s]")

    res = qed_ccsd_rhf.run_qed_ccsd(
        qedhf,
        do_t1_01=True, do_t2_11=True, do_t2_21=DO_T2_21,
        do_t2_02=False, do_t2_12=False, do_t2_22=False,
        frozen=FROZEN, max_iter=60, tol=1e-8, tol_amp=1e-7,
        max_diis=8, diis_on_disk=False, verbose=True,
    )

    peak_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**30
    print(f"\nE_QED_CCSD corr  = {res['E_qed_ccsd_corr']:.10f}")
    print(f"E_QED_CCSD total = {res['E_qed_ccsd_total']:.10f}")
    print(f"converged = {res['converged']} in {res['iterations']} iterations,"
          f" {res['time_total']:.1f} s;  peak RSS = {peak_gb:.2f} GB")

    out = {
        'molecule': 'naphthalene (idealized acene geometry)',
        'basis': BASIS, 'auxbasis': AUXBASIS,
        'omega_eV': OMEGA_EV, 'lambda': list(LAMBDA), 'frozen': FROZEN,
        'flavour': ('QED-CCSD-21 (Deprince)' if DO_T2_21
                    else 'QED-CCSD-S1 (t1_01 + t2_11, no t2_21)'),
        'E_rhf': qedhf['E_rhf'], 'E_qed_hf': qedhf['E_qed_hf'],
        'E_qed_ccsd_corr': res['E_qed_ccsd_corr'],
        'E_qed_ccsd_total': res['E_qed_ccsd_total'],
        'converged': res['converged'], 'iterations': res['iterations'],
        'time_total_s': res['time_total'], 'peak_rss_gb': peak_gb,
    }
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        'qed_ccsd_naph_results.json')
    with open(path, 'w') as fh:
        json.dump(out, fh, indent=2)
    print(f"results written to {path}")


if __name__ == '__main__':
    main()
