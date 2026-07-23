"""Test the canonical-frame logic (as generate_molecular_orbitals applies
it) for symmetric + asymmetric molecules: input-independence, point-group
convention preservation, degenerate fallback, and g16 agreement."""
import pickle
import itertools
import numpy as np
from pyscf import gto, symm
from OmegaQMC.symm.point_groups import (charge_inertia_axes,
                                        canonicalize_symmetry_axes)

np.set_printoptions(precision=4, suppress=True)
RES = pickle.load(open("g16_results.pkl", "rb"))
from molecules import MOLECULES                                # noqa: E402


def rigid(c, seed):
    rng = np.random.default_rng(seed)
    v = rng.normal(size=3)
    v /= np.linalg.norm(v)
    th = rng.uniform(0.3, 2.8)
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    R = np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * (K @ K)
    return c @ R.T + rng.normal(size=3)


def canonical_coords(syms, coords):
    """Replicate the generate_molecular_orbitals symmetrization step."""
    mol = gto.M(atom=[[s, tuple(x)] for s, x in zip(syms, coords)],
                unit='Angstrom', basis='sto-3g', verbose=0)
    gpname, centroid, axes = symm.geom.detect_symm(mol._atom)
    if gpname == 'C1':
        canon = charge_inertia_axes(mol.atom_coords(), mol.atom_charges())
    else:
        canon = canonicalize_symmetry_axes(mol.atom_coords(),
                                           mol.atom_charges(), axes, gpname)
    if canon is not None:
        centroid, axes = canon
    # shift_atom works in Bohr; g16 reference coords are in Angstrom.
    bohr = 0.52917721092
    P = np.array([x for _, x in symm.geom.shift_atom(mol._atom,
                                                     centroid, axes)]) * bohr
    return gpname, P


def signflip_or_perm_match(P, g):
    best = 9e9
    for perm in itertools.permutations(range(3)):
        for sg in itertools.product((1, -1), repeat=3):
            S = np.zeros((3, 3))
            for c, (p, s) in enumerate(zip(perm, sg)):
                S[p, c] = s
            best = min(best, np.abs(P @ S - g).max())
    return best


print(f"{'mol':12s} {'gp':5s} {'input-indep':>11s} "
      f"{'vs_g16':>8s}  note")
for nm in ['water', 'ethylene', 'h2o2', 'hocl', 'fclethylene',
           'chfclbr', 'chfclbr2', 'sfclbr', 'ammonia', 'methane',
           'hcn', 'co2']:
    syms, c0 = MOLECULES[nm]()
    gpA, PA = canonical_coords(syms, rigid(c0, 11))
    gpB, PB = canonical_coords(syms, rigid(c0, 22))
    indep = np.abs(PA - PB).max()          # DIRECT (signs matter)
    # proper-rotation check on the frame taking input->canonical
    g = RES[nm]['standard'] if nm in RES else None
    vg = signflip_or_perm_match(PA, g) if g is not None else float('nan')
    note = ""
    # C1 and the fully-fixed symmetric groups (C2v, D2h, linear) should
    # match g16 exactly; Cs/C2 keep g16's principal axis but not its
    # "leftover azimuth" (FINDINGS.md), so ~0.1 A there is expected; the
    # degenerate groups (C3v, Td) fall back to pyscf's non-canonical axes.
    if indep > 1e-6 and gpA not in ('C3v', 'Td', 'Coov', 'Dooh'):
        note = "NOT input-independent!"
    elif indep > 1e-6:
        note = "degenerate fallback (pyscf, not canonical)"
    print(f"{nm:12s} {gpA:5s} {indep:11.1e} {vg:8.1e}  {note}")
