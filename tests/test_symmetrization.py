"""C2v symmetrization of a slightly asymmetric water geometry.

The input geometry (Angstrom, as typed by a user) is C2v only to about
1e-5 bohr after PySCF's alignment: the two H atoms' |y| and |z| differ
by that much, which breaks the point-group symmetry the PGCS and the
symmetry-adapted orbitals rely on.  ``symmetrization_level=2`` must
make the two H atoms exact mirror images (and the O-H bonds equal),
while ``symmetrization_level=0`` leaves the geometry as given.
"""
import numpy as np

from OmegaQMC import generate_molecular_orbitals

WATER = '''
O                0.000000    0.000000    0.000000
H                0.000000    0.000000    0.957800
H                0.927385    0.000000   -0.239451
'''


def _h_mismatch(level):
    mf = generate_molecular_orbitals(WATER, units="Angstrom",
                                     basis="sto-3g",
                                     symmetrization_level=level, spin=0)
    c = np.asarray(mf.mol.atom_coords())
    d_oh = np.linalg.norm(c[1:] - c[0], axis=1)
    return np.abs(np.abs(c[1]) - np.abs(c[2])), abs(d_oh[0] - d_oh[1])


def test_water_symmetrization_exact_c2v():
    mis0, bond0 = _h_mismatch(0)
    mis2, bond2 = _h_mismatch(2)
    # as given: the H atoms are mirror images only to ~1e-5 bohr
    assert mis0.max() > 1e-7
    # symmetrized: exact to round-off, O-H bonds equal
    assert mis2.max() < 1e-10 and bond2 < 1e-10
