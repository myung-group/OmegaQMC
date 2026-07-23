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


def _parse_atom_list(atom_coords):
    """Split an atom list into ``(symbols, coords, rebuild)``.

    Accepts any of the atom formats used in this codebase:

    * ``[symbol, coords_array]`` -- PySCF ``mol.atom`` / ``mol._atom``;
    * ``[symbol, coords_array, fragment_index]`` -- ``Mole_custom``, which
      carries a trailing per-atom fragment label;
    * ``(symbol, x, y, z)`` -- flat tuple.

    ``coords`` is an ``(N, 3)`` float array.  ``rebuild(new_coords)``
    reconstructs the list in the original format, preserving any trailing
    per-atom data (e.g. the fragment index) so it survives a
    symmetrization round-trip.
    """
    first = atom_coords[0]
    if isinstance(first[1], (list, tuple, np.ndarray)):
        # [symbol, coords_array, *extra]
        symbols = [a[0] for a in atom_coords]
        coords = np.array([np.asarray(a[1], dtype=float)
                           for a in atom_coords])
        extras = [list(a[2:]) for a in atom_coords]

        def rebuild(new_coords):
            return [[symbols[i], new_coords[i], *extras[i]]
                    for i in range(len(symbols))]
    else:
        # (symbol, x, y, z)
        symbols = [a[0] for a in atom_coords]
        coords = np.array([(a[1], a[2], a[3]) for a in atom_coords])

        def rebuild(new_coords):
            return [(symbols[i], new_coords[i, 0], new_coords[i, 1],
                     new_coords[i, 2]) for i in range(len(symbols))]

    return symbols, coords, rebuild


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
    symbols, coords, rebuild = _parse_atom_list(atom_coords)

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

        # Reconstruct atom list in the same format as the input, keeping
        # any trailing per-atom data (e.g. Mole_custom fragment indices).
        return rebuild(symmetrized_coords)

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
    _, coords, rebuild = _parse_atom_list(atom_coords)

    # Get symmetrizer
    symmetrizer = get_symmetrizer(point_group)

    # Apply symmetrization
    symmetrized_coords = symmetrizer.symmetrize(coords, center)

    # Reconstruct atom list in the same format as the input, keeping any
    # trailing per-atom data (e.g. Mole_custom fragment indices).
    return rebuild(symmetrized_coords)


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
    symbols, coords, _ = _parse_atom_list(atom_coords)

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


def _reference_atom(proj: np.ndarray, tol_rel: float = 1e-6) -> int:
    """Index of the atom most strongly projected onto an axis.

    Returns the atom with the largest ``|proj|``, breaking ties by the
    smallest atom index so the choice is independent of the input
    orientation.  Symmetry-equivalent atoms have equal ``|proj|`` in exact
    arithmetic but differ by rounding, which would otherwise let the raw
    ``argmax`` pick a different atom for a rotated copy of the same
    molecule; magnitudes are therefore bucketed to a relative tolerance
    before ranking so such near-ties fall to the index tie-break.
    """
    a = np.abs(proj)
    scale = max(1.0, float(a.max()))
    bucket = np.round(a / (tol_rel * scale))
    order = np.lexsort((np.arange(len(proj)), -bucket))
    return int(order[0])


def _charge_inertia_eigen(coords: np.ndarray, charges: np.ndarray):
    """Eigendecomposition of the charge-weighted inertia tensor.

    Builds the inertia-like tensor of the nuclei about their center of
    nuclear charge (CNC), using the nuclear charge (atomic number) in
    place of the mass, and diagonalizes it.

    Returns ``(cnc, x, evals, evecs)`` with ``x`` the CNC-centered
    coordinates, ``evals`` the ascending principal moments and ``evecs``
    their eigenvectors as columns.
    """
    coords = np.asarray(coords, dtype=float)
    q = np.asarray(charges, dtype=float)
    cnc = (q[:, None] * coords).sum(0) / q.sum()
    x = coords - cnc
    tensor = np.zeros((3, 3))
    for qi, xi in zip(q, x):
        tensor += qi * (xi @ xi * np.eye(3) - np.outer(xi, xi))
    evals, evecs = np.linalg.eigh(tensor)   # ascending; eigvecs are cols
    return cnc, x, evals, evecs


def _is_degenerate(evals: np.ndarray, degen_rtol: float) -> bool:
    """True if any two charge-inertia moments coincide to ``degen_rtol``.

    A (near-)degenerate tensor has ill-defined principal axes (linear
    molecules, or any group with a C_n / S_n axis of order >= 3 such as
    C3v, Td, C4v, D4h), so the caller should fall back to symmetry axes.
    """
    scale = max(1.0, float(np.abs(evals).max()))
    return bool(np.any(np.diff(evals) < degen_rtol * scale))


def _deterministic_frame(dirs: np.ndarray, x: np.ndarray,
                         tol_rel: float = 1e-8) -> np.ndarray:
    """Sign an orthonormal axis set deterministically (proper rotation).

    ``dirs`` is a (3, 3) array whose rows are orthonormal axis directions
    already in the desired (x, y, z) assignment, and ``x`` the CNC-centered
    nuclear coordinates.  Each axis is signed so that its most strongly
    projected atom has a positive coordinate (ties broken by atom index),
    a choice that depends only on the geometry and not on the input
    orientation.  Axes onto which every atom projects to (near) zero -- for
    instance the normal of a planar fragment -- are left undetermined here.
    The overall handedness is then fixed so the frame is a proper rotation
    (``det == +1``), never a reflection (which would invert chirality): an
    undetermined axis absorbs the flip when present, otherwise the
    least-determined axis does.
    """
    A = np.array(dirs, dtype=float).copy()
    scale = max(1.0, float(np.abs(x).max()))
    tol = tol_rel * scale
    scores = np.empty(3)
    for k in range(3):
        proj = x @ A[k]
        scores[k] = proj[_reference_atom(proj)]
    for k in range(3):
        if scores[k] < -tol:
            A[k] = -A[k]
            scores[k] = -scores[k]
    if np.linalg.det(A) < 0.0:
        undetermined = [k for k in range(3) if abs(scores[k]) <= tol]
        k = undetermined[0] if undetermined else int(np.argmin(scores))
        A[k] = -A[k]
    return A


def charge_inertia_axes(coords: np.ndarray,
                        charges: np.ndarray,
                        degen_rtol: float = 1e-3):
    """Canonical center + orientation from the charge-weighted inertia.

    Returns the center of nuclear charge together with the principal axes
    of the charge-weighted inertia tensor as a proper-rotation matrix with
    a deterministic, input-orientation-independent sign convention (see
    :func:`_deterministic_frame`).

    This reproduces the transformation Gaussian uses for its "Standard
    orientation" of molecules with no point-group symmetry: translate to
    the CNC and rotate to the principal axes ordered by ascending
    charge-weighted moment (smallest-moment axis to x, largest to z),
    matching g16's axis ordering.  See
    ``tests/orient-molecule/FINDINGS.md``.  The mass is deliberately *not*
    used, so the orientation is isotope-independent.

    Parameters
    ----------
    coords : ndarray, shape (N, 3)
        Nuclear coordinates.  The returned center is in these same units;
        the axes are scale-free.
    charges : ndarray, shape (N,)
        Nuclear charges (atomic numbers).
    degen_rtol : float, optional
        Relative tolerance for detecting a (near-)degenerate tensor, whose
        principal axes are ill-defined; ``None`` is returned so the caller
        can fall back to symmetry axes.

    Returns
    -------
    (center, axes) : (ndarray shape (3,), ndarray shape (3, 3)) or None
        ``center`` is the center of nuclear charge; ``axes`` rows are the
        principal axes expressed in the input coordinate frame, ordered by
        ascending moment, forming a right-handed frame (``det == +1``).
        The pair can be passed straight to
        :func:`pyscf.symm.geom.shift_atom` as ``(centroid, axes)``.
        ``None`` if the tensor is (near-)degenerate.
    """
    cnc, x, evals, evecs = _charge_inertia_eigen(coords, charges)
    if _is_degenerate(evals, degen_rtol):
        return None
    dirs = np.array([evecs[:, 0], evecs[:, 1], evecs[:, 2]])
    return cnc, _deterministic_frame(dirs, x)


def canonicalize_symmetry_axes(coords: np.ndarray,
                               charges: np.ndarray,
                               sym_axes: np.ndarray,
                               gpname: str,
                               degen_rtol: float = 1e-3):
    """Re-sign point-group symmetry axes deterministically.

    For a molecule *with* point-group symmetry the charge-weighted
    principal directions coincide with the symmetry axes returned by
    pyscf's ``detect_symm``, so those axes already carry the standard axis
    *assignment* the downstream operations rely on (principal C_n along z,
    mirror normals on the expected Cartesians, etc.).  pyscf's *signs*,
    however, depend on the input orientation.  This keeps pyscf's axis
    assignment but re-signs each axis with the same deterministic,
    input-independent rule used for asymmetric molecules
    (:func:`_deterministic_frame`), yielding a canonical frame while
    leaving the point-group convention -- and hence the supported symmetry
    operations and level-2 symmetrizers -- intact.

    Only the non-degenerate groups (Cs, Ci, C2, C2v, C2h, D2h), whose
    operations are pure coordinate-sign flips, reach the re-signing step;
    groups with a C_n / S_n axis of order >= 3 (C3v, Td, C4v, D4h) or
    linear groups have a (near-)degenerate charge-inertia tensor and
    return ``None`` so the caller keeps pyscf's axes unchanged.

    D2h and Ci additionally have an input-dependent *assignment* of their
    equal-role principal axes (pyscf may map the three C2 axes of D2h to
    x/y/z in any order).  Their operation set is invariant under axis
    permutation, so for these two groups the axes are reordered by
    ascending charge-weighted moment (as for asymmetric molecules) to
    remove that freedom; every other group keeps pyscf's assignment, whose
    special axis (C2 / mirror normal) is already pinned to the convention
    the downstream operations expect.

    Returns ``(cnc, axes)`` (suitable for
    :func:`pyscf.symm.geom.shift_atom`) or ``None`` when degenerate.
    """
    cnc, x, evals, evecs = _charge_inertia_eigen(coords, charges)
    if _is_degenerate(evals, degen_rtol):
        return None
    if gpname in ('D2h', 'Ci'):
        dirs = np.array([evecs[:, 0], evecs[:, 1], evecs[:, 2]])
    else:
        dirs = np.asarray(sym_axes, dtype=float)
    return cnc, _deterministic_frame(dirs, x)


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
