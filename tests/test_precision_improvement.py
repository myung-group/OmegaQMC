#!/usr/bin/env python3
"""
Test to demonstrate the improvement in numerical precision for water molecule
symmetry after applying C2v symmetrization.

This test specifically addresses the issue mentioned in the problem:
"the y-coordinate and z-coordinate of the two hydrogen atoms appear to differ
(up to a sign) by about 0.00001 Bohr" and shows the improvement.
"""

from vmc_mlsw import generate_molecular_orbitals


def test_numerical_precision_improvement():
    """Test the improvement in numerical precision for water molecule."""

    print("=" * 80)
    print("Testing Numerical Precision Improvement for Water Molecule")
    print("=" * 80)

    # Use the exact water geometry from the problem description
    water_string = '''
    O                0.000000    0.000000    0.000000
    H                0.000000    0.000000    0.957800
    H                0.927385    0.000000   -0.239451
    '''

    print("\nOriginal water geometry (Angstrom):")
    for line in water_string.strip().split('\n'):
        print(f"  {line.strip()}")

    # Test without symmetrization
    print("\n1. WITHOUT C2v Symmetrization")
    print("-" * 50)
    mf_no_symm = generate_molecular_orbitals(water_string, units="Angstrom",
                                             basis="cc-pVDZ",
                                             enable_symmetrization=False,
                                             spin=0)

    coords_no_symm = mf_no_symm.mol.atom_coords()
    print("Final coordinates after PySCF alignment (Bohr):")
    for i, atom in enumerate(coords_no_symm):
        symbol = mf_no_symm.mol.atom_symbol(i)
        print(f"  {symbol}: "
              f"[{atom[0]:12.8f}, {atom[1]:12.8f}, {atom[2]:12.8f}]")

    # Calculate the specific differences mentioned in the problem
    H1_pos = coords_no_symm[1]
    H2_pos = coords_no_symm[2]

    y_diff_no_symm = abs(H1_pos[1]) - abs(H2_pos[1])
    z_diff_no_symm = abs(H1_pos[2]) - abs(H2_pos[2])

    print("\nHydrogen coordinate differences (problematic values):")
    print(f"  |H1_y| - |H2_y|: {y_diff_no_symm:.8f} Bohr")
    print(f"  |H1_z| - |H2_z|: {z_diff_no_symm:.8f} Bohr")
    print("  Problem threshold: 0.00001 Bohr")
    print("  Exceeds threshold: {}"
          .format('YES'
                  if abs(y_diff_no_symm) > 1e-5 or abs(z_diff_no_symm) > 1e-5
                  else 'NO'))

    # Test with symmetrization
    print("\n2. WITH C2v Symmetrization")
    print("-" * 50)
    mf_symm = generate_molecular_orbitals(water_string, units="Angstrom",
                                          basis="cc-pVDZ",
                                          enable_symmetrization=True,
                                          spin=0)

    coords_symm = mf_symm.mol.atom_coords()
    print("Final coordinates after symmetrization (Bohr):")
    for i, atom in enumerate(coords_symm):
        symbol = mf_symm.mol.atom_symbol(i)
        print(f"  {symbol}: "
              f"[{atom[0]:12.8f}, {atom[1]:12.8f}, {atom[2]:12.8f}]")

    # Calculate the differences after symmetrization
    H1_pos = coords_symm[1]
    H2_pos = coords_symm[2]

    y_diff_symm = abs(H1_pos[1]) - abs(H2_pos[1])
    z_diff_symm = abs(H1_pos[2]) - abs(H2_pos[2])

    print("\nHydrogen coordinate differences (after symmetrization):")
    print(f"  |H1_y| - |H2_y|: {y_diff_symm:.8f} Bohr")
    print(f"  |H1_z| - |H2_z|: {z_diff_symm:.8f} Bohr")
    print("  Problem threshold: 0.00001 Bohr")
    print("  Exceeds threshold: {}"
          .format('YES'
                  if abs(y_diff_symm) > 1e-5 or abs(z_diff_symm) > 1e-5
                  else 'NO'))

    # Calculate improvement
    print("\n3. IMPROVEMENT ANALYSIS")
    print("-" * 50)

    y_improvement = y_diff_no_symm / max(y_diff_symm, 1e-15)
    z_improvement = z_diff_no_symm / max(z_diff_symm, 1e-15)

    print(f"Y-coordinate improvement: {y_improvement:.1f}x")
    print(f"Z-coordinate improvement: {z_improvement:.1f}x")

    # Check if the problem is solved
    problem_solved = (y_diff_symm < 1e-5 and z_diff_symm < 1e-5)
    high_precision = (y_diff_symm < 1e-10 and z_diff_symm < 1e-10)

    print("\n4. PROBLEM RESOLUTION")
    print("-" * 50)
    print("Original problem: Differences ~0.00001 Bohr "
          "causing numerical instabilities")
    print(f"Problem solved: {'✓ YES' if problem_solved else '✗ NO'}")
    print(f"High precision achieved: {'✓ YES' if high_precision else '✗ NO'}")
    print(f"Final precision: {max(y_diff_symm, z_diff_symm):.2e} Bohr")

    print("\n" + "=" * 80)
    print("SUMMARY:")
    print("  Before symmetrization: "
          f"y_diff={y_diff_no_symm:.2e}, "
          f"z_diff={z_diff_no_symm:.2e} Bohr")
    print("  After symmetrization:  "
          f"y_diff={y_diff_symm:.2e}, "
          f"z_diff={z_diff_symm:.2e} Bohr")
    print(f"  Improvement: {y_improvement:.0f}x (Y), {z_improvement:.0f}x (Z)")
    print("  Status: {}"
          .format('✓ PROBLEM SOLVED'
                  if problem_solved
                  else '✗ PROBLEM REMAINS'))
    print("=" * 80)

    return problem_solved


if __name__ == "__main__":
    success = test_numerical_precision_improvement()
    exit(0 if success else 1)
