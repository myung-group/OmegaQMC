"""End-to-end check of the canonical-frame transform in
generate_molecular_orbitals (C1 reorientation, input-independence,
symmetric-case regression, energy invariance)."""
import numpy as np
from OmegaQMC.vmc_gto import generate_molecular_orbitals

np.set_printoptions(precision=5, suppress=True)


def astr(syms, coords):
    return "; ".join(f"{s} {c[0]:.10f} {c[1]:.10f} {c[2]:.10f}"
                     for s, c in zip(syms, coords))


def rigid(coords, seed):
    rng = np.random.default_rng(seed)
    v = rng.normal(size=3)
    v /= np.linalg.norm(v)
    th = rng.uniform(0.2, 2.8)
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    R = np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * (K @ K)
    return coords @ R.T + rng.normal(size=3)


# --- CHFClBr, a C1 molecule (charges very different from masses) ---
syms = ['C', 'H', 'F', 'Cl', 'Br']
base = np.array([[0, 0, 0], [0.63, 0.63, 0.63], [0.78, -0.78, -0.78],
                 [-1.02, 1.02, -1.02], [-1.12, -1.12, 1.12]])

results = []
for seed in (1, 2):
    mf = generate_molecular_orbitals(
        astr(syms, rigid(base, seed)), units='Angstrom', basis='sto-3g',
        symmetrization_level=1, dm0_pkl=None)
    c = mf.mol.atom_coords()          # Bohr, after canonical transform
    z = mf.mol.atom_charges()
    cnc = (z[:, None] * c).sum(0) / z.sum()
    results.append((c, float(mf.e_tot), mf.mol.groupname))
    print(f"seed {seed}: gp={mf.mol.groupname} E={mf.e_tot:.8f} "
          f"|CNC|={np.linalg.norm(cnc):.1e}")

c1, e1, g1 = results[0]
c2, e2, g2 = results[1]
print(f"\nC1 canonical frame input-independent? "
      f"max|c1-c2|={np.abs(c1 - c2).max():.2e}")
print(f"C1 energy input-independent?  |dE|={abs(e1 - e2):.2e}")

# --- water: symmetric (C2v) regression, must stay on pyscf symmetry axes ---
w = np.array([[0, 0, 0], [0, 0.757, 0.587], [0, -0.757, 0.587]])
mfw = generate_molecular_orbitals(
    astr(['O', 'H', 'H'], rigid(w, 7)), units='Angstrom', basis='sto-3g',
    symmetrization_level=1, dm0_pkl=None)
cw = mfw.mol.atom_coords()
print(f"\nwater: gp={mfw.mol.groupname} E={mfw.e_tot:.8f}")
print("water canonical coords (Bohr):")
print(cw)
# C2v: O on the symmetry axis, H's mirror images; check one axis has all
# x (or y) equal-and-opposite for the H's and 0 for O.
print(f"O off-axis dist in xy: {np.linalg.norm(cw[0, :2]):.1e}")

# water input-independence through the full pipeline (was NOT the case
# before applying the canonical frame to symmetric molecules).
mfw2 = generate_molecular_orbitals(
    astr(['O', 'H', 'H'], rigid(w, 8)), units='Angstrom', basis='sto-3g',
    symmetrization_level=1, dm0_pkl=None)
print(f"water canonical input-independent (level 1)? "
      f"max|c-c2|={np.abs(cw - mfw2.mol.atom_coords()).max():.2e}")

# level-2 symmetrization must still work in the canonicalized frame
mfw3 = generate_molecular_orbitals(
    astr(['O', 'H', 'H'], rigid(w, 9)), units='Angstrom', basis='sto-3g',
    symmetrization_level=2, dm0_pkl=None)
print(f"water level-2: gp={mfw3.mol.groupname} E={mfw3.e_tot:.8f} "
      f"(|dE vs level-1|={abs(mfw3.e_tot - mfw.e_tot):.1e})")

# ethylene (D2h) input-independence through the full pipeline
e = np.array([[0, 0, 0.669], [0, 0, -0.669], [0, 0.925, 1.24],
              [0, -0.925, 1.24], [0, 0.925, -1.24], [0, -0.925, -1.24]])
esym = ['C', 'C', 'H', 'H', 'H', 'H']
me1 = generate_molecular_orbitals(astr(esym, rigid(e, 3)), units='Angstrom',
                                  basis='sto-3g', symmetrization_level=1,
                                  dm0_pkl=None)
me2 = generate_molecular_orbitals(astr(esym, rigid(e, 4)), units='Angstrom',
                                  basis='sto-3g', symmetrization_level=1,
                                  dm0_pkl=None)
print(f"ethylene(D2h): gp={me1.mol.groupname} E={me1.e_tot:.8f} "
      f"input-indep max|c-c2|="
      f"{np.abs(me1.mol.atom_coords() - me2.mol.atom_coords()).max():.2e}")
