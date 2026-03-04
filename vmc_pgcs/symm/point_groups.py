#!/usr/bin/env python3
"""
Point group symmetrization module for improving numerical precision
in molecular coordinates after PySCF symmetry alignment.

This module implements symmetrization algorithms for various point groups,
starting with C2v for water molecules and designed to be extensible.
"""

import numpy as np
import math
from typing import Optional

from ..utils import compute_center_of_mass


class PointGroupSymmetrizer:
    """Base class for point group symmetrization."""

    def __init__(self, name: str, operations: dict[str, np.ndarray]):
        self.name = name
        self.operations = operations

    def apply_operation(self, coords: np.ndarray,
                        operation_matrix: np.ndarray) -> np.ndarray:
        """Apply a symmetry operation to coordinates."""
        return coords @ operation_matrix.T

    def symmetrize(self, coords: np.ndarray,
                   center: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Apply all symmetry operations and average the results.

        Args:
            coords: Nuclear coordinates (N, 3)
            center: Center point for operations (default: origin)

        Returns:
            Symmetrized coordinates
        """
        if center is not None:
            coords_centered = coords - center
        else:
            coords_centered = coords

        # Apply all operations and collect results
        transformed_coords = []
        for op_name, op_matrix in self.operations.items():
            transformed = self.apply_operation(coords_centered, op_matrix)
            if center is not None:
                transformed = transformed + center
            transformed_coords.append(transformed)

        # Average all transformed coordinates
        symmetrized = np.mean(transformed_coords, axis=0)

        return symmetrized


class C2vSymmetrizer(PointGroupSymmetrizer):
    """C2v point group symmetrizer."""

    def __init__(self):
        # C2v symmetry operations
        operations = {
            'E': np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
            'C2_z': np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]]),
            'sigma_xz': np.array([[1, 0, 0], [0, -1, 0], [0, 0, 1]]),
            'sigma_yz': np.array([[-1, 0, 0], [0, 1, 0], [0, 0, 1]])
        }
        super().__init__('C2v', operations)

    def symmetrize_water_molecule(self, coords: np.ndarray,
                                  center: Optional[np.ndarray] = None,
                                  symbols: Optional[list[str]] = None) \
            -> np.ndarray:
        """
        Specialized symmetrization for water molecules.
        Ensures equal OH bond lengths and proper H-O-H angle
        while maintaining C2v symmetry.

        Args:
            coords: Nuclear coordinates (N, 3)
            center: Center point for operations
                    (default: calculate center of mass)
            symbols: List of element symbols (default: ['O', 'H', 'H'])
        """
        if len(coords) != 3:
            raise ValueError("Water symmetrization requires exactly 3 atoms")

        # Set default symbols if not provided
        if symbols is None:
            symbols = ['O', 'H', 'H']

        if len(symbols) != 3:
            raise ValueError("Water symmetrization requires exactly 3 symbols")

        O, H1, H2 = coords[0], coords[1], coords[2]

        # Calculate OH vectors and distances
        OH1_vec = H1 - O
        OH2_vec = H2 - O

        rOH1 = np.linalg.norm(OH1_vec)
        rOH2 = np.linalg.norm(OH2_vec)
        rOH = 0.5 * (rOH1 + rOH2)  # Average bond length

        # Calculate H-O-H angle
        cos_angle = np.dot(OH1_vec, OH2_vec) / (rOH1 * rOH2)
        angle = math.acos(np.clip(cos_angle, -1, 1))

        # For C2v water, we want H atoms symmetric about the z-axis
        # Place H atoms at equal distance from z-axis, symmetric in xy-plane
        # Use the average bond length and angle

        # Default water angle ~104.4776 degrees
        # Hoy1979 doi:10.1016/0022-2852(79)90019-5
        target_angle = angle if angle > 0.1 else 104.4776 * math.pi / 180

        # Place H atoms symmetrically about z-axis in xz-plane
        # H atoms at same z coordinate, opposite x coordinates
        z_offset = rOH * math.cos(target_angle / 2)
        x_offset = rOH * math.sin(target_angle / 2)

        # New coordinates with O at origin
        H1_new = np.array([x_offset, 0.0, z_offset])
        H2_new = np.array([-x_offset, 0.0, z_offset])

        # Move to original oxygen position
        O_new = O
        H1_new = H1_new + O_new
        H2_new = H2_new + O_new

        coords_adjusted = np.array([O_new, H1_new, H2_new])

        # Handle center of mass calculation if center is not provided
        if center is not None:
            return coords_adjusted - center
        else:
            # Calculate center of mass using atomic masses
            com = compute_center_of_mass(coords_adjusted, symbols)
            return coords_adjusted - com

    def symmetrize(self, coords: np.ndarray,
                   center: Optional[np.ndarray] = None,
                   symbols: Optional[list[str]] = None) -> np.ndarray:
        """
        Apply C2v symmetrization with special handling for water molecules.

        Args:
            coords: Nuclear coordinates (N, 3)
            center: Center point for operations
            symbols: List of element symbols (optional, for water molecules)
        """
        # Check if this looks like a water molecule
        # (3 atoms with O-H-H pattern)
        if len(coords) == 3:
            # Set default symbols if not provided
            if symbols is None:
                symbols = ['O', 'H', 'H']
            return self.symmetrize_water_molecule(coords, center, symbols)

        # For other molecules, use the general symmetrization
        parent_symmetrizer = super(C2vSymmetrizer, self)
        return parent_symmetrizer.symmetrize(coords, center)


class CsSymmetrizer(PointGroupSymmetrizer):
    """Cs point group symmetrizer (planar molecules)."""

    def __init__(self, plane: str = 'xy'):
        # Cs symmetry operations (only identity and one mirror plane)
        operations = {
            'E': np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])  # Identity
        }

        # Add the appropriate mirror plane
        if plane == 'xy':
            operations['sigma_xy'] \
                = np.array([[1, 0, 0], [0, 1, 0], [0, 0, -1]])
        elif plane == 'xz':
            operations['sigma_xz'] \
                = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 1]])
        elif plane == 'yz':
            operations['sigma_yz'] \
                = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, 1]])
        else:
            raise ValueError(f"Unknown plane for Cs symmetry: {plane}")

        self.plane = plane
        super().__init__('Cs', operations)


class C2hSymmetrizer(PointGroupSymmetrizer):
    """C2h point group symmetrizer."""

    def __init__(self):
        # C2h symmetry operations
        operations = {
            'E': np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
            'C2_z': np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]]),
            'i': np.array([[-1, 0, 0], [0, -1, 0], [0, 0, -1]]),
            'sigma_h': np.array([[1, 0, 0], [0, 1, 0], [0, 0, -1]])
        }
        super().__init__('C2h', operations)


class D2hSymmetrizer(PointGroupSymmetrizer):
    """D2h point group symmetrizer."""

    def __init__(self):
        # D2h symmetry operations (8 operations)
        operations = {
            'E': np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
            'C2_z': np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]]),
            'C2_y': np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]]),
            'C2_x': np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]]),
            'i': np.array([[-1, 0, 0], [0, -1, 0], [0, 0, -1]]),
            'sigma_xy': np.array([[1, 0, 0], [0, 1, 0], [0, 0, -1]]),
            'sigma_xz': np.array([[1, 0, 0], [0, -1, 0], [0, 0, 1]]),
            'sigma_yz': np.array([[-1, 0, 0], [0, 1, 0], [0, 0, 1]])
        }
        super().__init__('D2h', operations)


# Registry of available symmetrizers
SYMMETRIZER_REGISTRY = {
    'C2v': C2vSymmetrizer,
    'Cs': CsSymmetrizer,
    'C2h': C2hSymmetrizer,
    'D2h': D2hSymmetrizer,
}


def get_symmetrizer(point_group: str, **kwargs) -> PointGroupSymmetrizer:
    """
    Get appropriate symmetrizer for a point group.

    Args:
        point_group: Point group name (e.g., 'C2v', 'Cs')
        **kwargs: Additional arguments for symmetrizer

    Returns:
        Symmetrizer instance
    """
    if point_group not in SYMMETRIZER_REGISTRY:
        raise ValueError("No symmetrizer available for point group: "
                         f"{point_group}")

    return SYMMETRIZER_REGISTRY[point_group](**kwargs)


def auto_symmetrize_molecule(atom_coords: list,
                             detected_point_group: str,
                             center: Optional[np.ndarray] = None) -> list:
    """
    Automatically symmetrize molecular coordinates using the most appropriate
    point group operations based on the detected point group
    and molecular structure.

    Args:
        atom_coords: List of [symbol, coordinates_array]
                     or (symbol, x, y, z) tuples
        detected_point_group: Point group detected by PySCF
        center: Center point for operations (default: compute center of mass)

    Returns:
        Symmetrized atom coordinates in same format as input
    """
    # Handle both PySCF format [symbol, coords_array]
    # and tuple format (symbol, x, y, z)
    is_pyscf_format = len(atom_coords[0]) == 2 \
        and isinstance(atom_coords[0][1], np.ndarray)

    if is_pyscf_format:
        coords = np.array([atom[1] for atom in atom_coords])
        symbols = [atom[0] for atom in atom_coords]
    else:
        coords = np.array([(x, y, z) for _, x, y, z in atom_coords])
        symbols = [symbol for symbol, _, _, _ in atom_coords]

    # Determine the best symmetrizer
    # based on detected group and molecular structure
    num_atoms = len(symbols)

    # Special cases for water-like molecules
    if num_atoms == 3 \
            and symbols[0] == 'O' and symbols[1] == 'H' and symbols[2] == 'H':
        # Water molecule - use C2v symmetrizer regardless of detected group
        # (PySCF often detects Cs for planar water,
        # but it should have C2v symmetry)
        point_group = 'C2v'
    elif detected_point_group in SYMMETRIZER_REGISTRY:
        # Use the detected point group if we have a symmetrizer for it
        point_group = detected_point_group
    else:
        # No symmetrizer available for this point group
        return atom_coords

    # Get symmetrizer and apply symmetrization
    try:
        symmetrizer = get_symmetrizer(point_group)

        # Pass symbols to symmetrizer for water molecules (C2v case)
        if isinstance(symmetrizer, C2vSymmetrizer) and len(symbols) == 3:
            # C2vSymmetrizer supports symbols parameter
            symmetrized_coords = symmetrizer.symmetrize(coords, center,
                                                        symbols=symbols)
        else:
            symmetrized_coords = symmetrizer.symmetrize(coords, center)

        # Reconstruct atom list in same format as input
        if is_pyscf_format:
            symmetrized_atoms = [[symbols[i], symmetrized_coords[i]]
                                 for i in range(len(symbols))]
        else:
            symmetrized_atoms = [(symbols[i], symmetrized_coords[i, 0],
                                  symmetrized_coords[i, 1],
                                  symmetrized_coords[i, 2])
                                 for i in range(len(symbols))]

        return symmetrized_atoms

    except Exception as e:
        # If symmetrization fails, return original coordinates
        print("Warning: Symmetrization failed "
              f"for point group {point_group}: {e}")
        return atom_coords


def symmetrize_molecule(atom_coords: list,
                        point_group: str,
                        center: Optional[np.ndarray] = None) -> list:
    """
    Symmetrize molecular coordinates using point group operations.

    Args:
        atom_coords: List of [symbol, coordinates_array]
                     or (symbol, x, y, z) tuples
        point_group: Point group name
        center: Center point for operations (default: compute center of mass)

    Returns:
        Symmetrized atom coordinates in same format as input
    """
    # Handle both PySCF format [symbol, coords_array]
    # and tuple format (symbol, x, y, z)
    is_pyscf_format = len(atom_coords[0]) == 2 \
        and isinstance(atom_coords[0][1], np.ndarray)

    if is_pyscf_format:
        # PySCF format: [symbol, coordinates_array]
        coords = np.array([atom[1] for atom in atom_coords])
        symbols = [atom[0] for atom in atom_coords]
        output_format = 'pyscf'
    else:
        # Tuple format: (symbol, x, y, z)
        coords = np.array([(x, y, z) for _, x, y, z in atom_coords])
        symbols = [symbol for symbol, _, _, _ in atom_coords]
        output_format = 'tuple'

    # Get symmetrizer
    symmetrizer = get_symmetrizer(point_group)

    # Apply symmetrization
    symmetrized_coords = symmetrizer.symmetrize(coords, center)

    # Reconstruct atom list in same format as input
    if output_format == 'pyscf':
        symmetrized_atoms = [[symbols[i], symmetrized_coords[i]]
                             for i in range(len(symbols))]
    else:
        symmetrized_atoms = [(symbols[i], symmetrized_coords[i, 0],
                              symmetrized_coords[i, 1],
                              symmetrized_coords[i, 2])
                             for i in range(len(symbols))]

    return symmetrized_atoms


def detect_symmetry_quality(atom_coords: list,
                            point_group: str,
                            tolerance: float = 1e-5) -> dict[str, float]:
    """
    Analyze the quality of molecular symmetry.

    Args:
        atom_coords: List of [symbol, coordinates_array]
                     or (symbol, x, y, z) tuples
        point_group: Point group name
        tolerance: Tolerance for symmetry deviation

    Returns:
        Dictionary with symmetry quality metrics
    """
    # Handle both PySCF format [symbol, coords_array]
    # and tuple format (symbol, x, y, z)
    if len(atom_coords[0]) == 2 and isinstance(atom_coords[0][1], np.ndarray):
        # PySCF format: [symbol, coordinates_array]
        coords = np.array([atom[1] for atom in atom_coords])
        symbols = [atom[0] for atom in atom_coords]
    else:
        # Tuple format: (symbol, x, y, z)
        coords = np.array([(x, y, z) for _, x, y, z in atom_coords])
        symbols = [symbol for symbol, _, _, _ in atom_coords]

    # Get symmetrizer
    symmetrizer = get_symmetrizer(point_group)

    # Apply symmetrization
    symmetrized_coords = symmetrizer.symmetrize(coords)

    # Calculate deviations
    deviations = np.linalg.norm(coords - symmetrized_coords, axis=1)

    # Group by atom type
    atom_deviations = {}
    for i, symbol in enumerate(symbols):
        if symbol not in atom_deviations:
            atom_deviations[symbol] = []
        atom_deviations[symbol].append(deviations[i])

    # Calculate statistics
    quality_metrics = {
        'max_deviation': np.max(deviations),
        'mean_deviation': np.mean(deviations),
        'atom_max_deviations': {symbol: np.max(devs)
                                for symbol, devs in atom_deviations.items()},
        'atom_mean_deviations': {symbol: np.mean(devs)
                                 for symbol, devs in atom_deviations.items()},
        'needs_symmetrization': float(np.max(deviations)) > tolerance
    }

    return quality_metrics


if __name__ == "__main__":
    # Test with water molecule
    water_atoms = [
        ('O', 0.0, 0.0, 0.0),
        ('H', 0.0, 1.52610182, 1.12172672),
        ('H', 0.0, -1.51745721, 1.11537270)
    ]

    print("Original water coordinates:")
    for symbol, x, y, z in water_atoms:
        print(f"  {symbol}: [{x:10.6f}, {y:10.6f}, {z:10.6f}]")

    # Analyze symmetry quality
    quality = detect_symmetry_quality(water_atoms, 'C2v')
    print("\nSymmetry quality metrics:")
    print(f"  Max deviation: {quality['max_deviation']:.2e}")
    print(f"  Mean deviation: {quality['mean_deviation']:.2e}")
    print(f"  Needs symmetrization: {quality['needs_symmetrization']}")

    # Apply symmetrization
    symmetrized = symmetrize_molecule(water_atoms, 'C2v')
    print("\nSymmetrized water coordinates:")
    for symbol, x, y, z in symmetrized:
        print(f"  {symbol}: [{x:10.6f}, {y:10.6f}, {z:10.6f}]")

    # Check improvement
    quality_after = detect_symmetry_quality(symmetrized, 'C2v')
    print("\nAfter symmetrization:")
    print(f"  Max deviation: {quality_after['max_deviation']:.2e}")
    print(f"  Mean deviation: {quality_after['mean_deviation']:.2e}")
    print("  Improvement factor: "
          f"{quality['max_deviation']/quality_after['max_deviation']:.1f}x")
