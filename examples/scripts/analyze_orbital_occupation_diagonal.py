"""Project walker positions onto the CH3 1e' orbital pair (e'_x, e'_y)
and report the diagonal "occupation projection" of the e' shell.

This is NOT the rigorous c_k^dagger c_k expectation value — that requires the
off-diagonal 1-RDM. But it IS a quantitative measure of how much MC density
lies inside the e' shell, useful for sanity checking.

The cleaner argument for orbital-degeneracy lifting:

  In the orbital occupation basis, <L_z> = sum_k m_l(k) * n_k.
  In CH3, only the 1e' shell has nonzero m_l contribution; the SOMO
  (1a2'') has m_l=0 and all closed shells average to m_l=0.
  Therefore:
      <L_z> = 1 * n(e'_+) + (-1) * n(e'_-) = n(e'_+) - n(e'_-) = delta_e'.
  So our measured <L_z> directly equals the occupation asymmetry.

This script also generates a "fraction of density in e' shell" number,
which is a cross-check on whether our MC samples are consistent with
the expected ~4-electron occupation of the e' pair (2 per orbital).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from pyscf import gto


def build_ch3_mol():
    """Same molecule as build_ch3_mo_reference.py."""
    import math
    coords = [["C", (0.0, 0.0, 0.0)]]
    r_ch = 2.039
    for k in range(3):
        theta = 2 * math.pi * k / 3
        coords.append([
            "H", (r_ch * math.cos(theta), r_ch * math.sin(theta), 0.0),
        ])
    return gto.M(
        atom=coords, basis="cc-pVDZ",
        spin=1, unit="Bohr", symmetry=True, verbose=0,
    )


def evaluate_ao_at_positions(mol, positions):
    """Evaluate all AOs at given 3D positions. Returns (n_pos, nao) array."""
    # PySCF eval_ao for an array of positions in Bohr.
    return mol.eval_gto("GTOval_sph", positions)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("walkers_npz")
    ap.add_argument("--mo-ref", default="scripts/ch3_mo_reference.npz")
    args = ap.parse_args()

    ref = np.load(args.mo_ref)
    e_x_coef = ref["mo_coeff"][:, ref["e_prime_indices"][0]]   # (nao,)
    e_y_coef = ref["mo_coeff"][:, ref["e_prime_indices"][1]]   # (nao,)
    e_plus_coef = ref["e_prime_plus_coef"]    # complex (nao,)
    e_minus_coef = ref["e_prime_minus_coef"]

    walkers = np.load(args.walkers_npz)
    pos = walkers["walker_positions"].reshape(-1, 3)
    lz_mean = float(walkers["l_z_mean"])
    cavity_lam = float(walkers["cavity_lambda"])
    hand = int(walkers["chiral_handedness"])

    print(f"loaded {pos.shape[0]:,} electron positions")
    print(f"cavity: lambda={cavity_lam}, handedness={hand}")
    print(f"<L_z>_VMC: {lz_mean:+.4f}")

    mol = build_ch3_mol()
    # Evaluate AOs at all walker positions: shape (N_pos, n_ao)
    ao_at_pos = evaluate_ao_at_positions(mol, pos)

    # Project each electron onto e'_x and e'_y orbitals
    phi_x = ao_at_pos @ e_x_coef        # (N_pos,) real
    phi_y = ao_at_pos @ e_y_coef        # (N_pos,) real

    # Diagonal density-projection: how much density is in each orbital?
    # Note: |phi_+|^2 = |phi_-|^2 by complex-conjugate symmetry, so we
    # can't get the asymmetry this way. We compute n_x and n_y separately,
    # then construct the asymmetry via the cross term.
    n_x = np.mean(phi_x ** 2)    # average over all electron positions
    n_y = np.mean(phi_y ** 2)
    # Cross-density term (real; this is the diagonal of [phi_x.conj() phi_y]
    # weighted by density). Real because both orbitals are real-valued.
    n_xy = np.mean(phi_x * phi_y)

    n_e_shell_per_electron = n_x + n_y
    # Total density in e' shell (multiply by N electrons):
    n_e_shell_total = n_e_shell_per_electron * 9.0

    print(f"\nPer-electron density projections (averaged over MC samples):")
    print(f"  <|φ_x|²> = {n_x:.5f}  (in vacuum, expect ~2/9 = 0.222)")
    print(f"  <|φ_y|²> = {n_y:.5f}  (same expectation)")
    print(f"  <φ_x φ_y> = {n_xy:.5f}  (vacuum: 0 by symmetry)")
    print(f"\nTotal density in 1e' shell (assuming 9-electron weighting):")
    print(f"  N_e' ≈ {n_e_shell_total:.3f}  (vacuum expects ~4.0)")

    # Symmetry-breaking quantitative measures
    asym_xy = abs(n_x - n_y)
    print(f"\nSymmetry breaking diagnostics:")
    print(f"  |<|φ_x|²> - <|φ_y|²>|  = {asym_xy:.5f}  (vacuum: 0)")
    print(f"  <φ_x φ_y>              = {n_xy:.5f}  (vacuum: 0)")

    # The orbital-occupation asymmetry, by the analytical argument:
    # δ_e' = <L_z>_VMC / ℏ  (assuming all <L_z> comes from e' shell)
    delta_from_lz = lz_mean
    print(f"\nOrbital occupation asymmetry by analytical argument:")
    print(f"  δ_e' = n(e'_+) - n(e'_-) = <L_z>/ℏ = {delta_from_lz:+.4f}")
    print(f"  (assumes <L_z> dominated by e' shell, which is the only")
    print(f"   m_l ≠ 0 occupied shell in CH3·)")

    if cavity_lam > 0:
        rel_asym = abs(delta_from_lz) / 4.0   # 4 electrons in e' shell
        print(f"  → fractional asymmetry: {100*rel_asym:.2f}% of the e' "
              f"shell population")


if __name__ == "__main__":
    main()
