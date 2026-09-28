"""Point-group symmetry operations on Cartesian coordinates.

Each ``apply_*`` function maps coordinates of shape ``(..., 3)`` to
their image under one operation, acting on the last axis.
:data:`symmetry_operations_map` keys them by 16 canonical symbols:

* ``E``: identity.
* ``i``: inversion, ``(x, y, z) -> (-x, -y, -z)``.
* ``sx``, ``sy``, ``sz``: mirror planes yz, xz and xy, i.e. negate
  the named coordinate.
* ``sxy``, ``sxmy``: diagonal sigma_d mirrors,
  ``(y, x, z)`` and ``(-y, -x, z)``.
* ``Rz90``, ``Rz180``, ``Rz270``: rotation about z by 90 degrees
  counter-clockwise (``(-y, x, z)``), 180 and 270 degrees.
* ``Rx180``, ``Ry180``: C2 about x and about y.
* ``C2xy``, ``C2xmy``: C2 about the diagonals,
  ``(y, x, -z)`` and ``(-y, -x, -z)``.
* ``S4``, ``S4_3``: improper rotations,
  ``(y, -x, -z)`` and ``(-y, x, -z)``.

The axes are those of a *fragment-local* frame, not the lab frame.
For Point Group Correlated Sampling the drivers shift electrons to
the fragment centroid, rotate them into the frame built by
:func:`~OmegaQMC.symm.fragments.build_frag_transform_data` (principal
C_n / S_n axis along local z, sigma_v / C2' elements along local x
and y), apply the operation and transform back; see
:func:`~OmegaQMC.symm.fragments.make_apply_single_frag_symmop`.

:data:`POINT_GROUP_OPS` lists the operations of each supported point
group in that convention.  The linear groups Coov and Dooh are
represented by their C4v and D4h subgroups, and any other group is
treated as C1 (identity only).  :data:`POINT_GROUP_OP_ALIASES` maps
alternative spellings (``"C2z"``, ``"sigma_x"``, ``"-1"``, ...) to
the canonical symbols; user input should be normalised through it,
as :func:`~OmegaQMC.symm.fragments.build_frag_symmops` does.
"""

import jax
from pyscf import gto, symm


@jax.jit
def apply_identity(coords):
    return coords


@jax.jit
def apply_reflection_x(coords):
    """Apply reflection across yz-plane (- x-coordinate)."""
    return coords.at[..., 0].multiply(-1)


@jax.jit
def apply_reflection_y(coords):
    """Apply reflection across xz-plane (- y-coordinate)."""
    return coords.at[..., 1].multiply(-1)


@jax.jit
def apply_reflection_z(coords):
    """Apply reflection across xy-plane (- z-coordinate)."""
    return coords.at[..., 2].multiply(-1)


@jax.jit
def apply_rotation_z180(coords):
    """Apply 180-degree rotation about z-axis (- x,y-coordinate)."""
    return coords.at[..., [0, 1]].multiply(-1)


@jax.jit
def apply_rotation_z90(coords):
    """Apply 90-degree ccw rotation about z-axis (-y, x)."""
    return coords.at[..., [0, 1]].set(coords[..., [1, 0]]) \
        .at[..., 0].multiply(-1)


@jax.jit
def apply_rotation_z270(coords):
    """Apply 90-degree cw rotation about z-axis (y, -x)."""
    return coords.at[..., [0, 1]].set(coords[..., [1, 0]]) \
        .at[..., 1].multiply(-1)


@jax.jit
def apply_inversion(coords):
    """Negate all coordinates."""
    return coords.at[..., [0, 1, 2]].multiply(-1)


@jax.jit
def apply_rotation_x180(coords):
    """Apply 180-degree rotation about x-axis (x, -y, -z)."""
    return coords.at[..., [1, 2]].multiply(-1)


@jax.jit
def apply_rotation_y180(coords):
    """Apply 180-degree rotation about y-axis (-x, y, -z)."""
    return coords.at[..., [0, 2]].multiply(-1)


@jax.jit
def apply_S4(coords):
    """Apply S4 improper rotation (C4³ followed by σh): (x,y,z)
    → (y, -x, -z)."""
    result = coords.at[..., [0, 1]].set(coords[..., [1, 0]])
    return result.at[..., [1, 2]].multiply(-1)


@jax.jit
def apply_S4_3(coords):
    """Apply S4³ improper rotation (C4 followed by σh): (x,y,z)
    → (-y, x, -z)."""
    result = coords.at[..., [0, 1]].set(coords[..., [1, 0]])
    return result.at[..., [0, 2]].multiply(-1)


@jax.jit
def apply_rotation_xy_diagonal(coords):
    """Apply C2'' rotation about xy diagonal: (x,y,z) → (y, x, -z)."""
    result = coords.at[..., [0, 1]].set(coords[..., [1, 0]])
    return result.at[..., 2].multiply(-1)


@jax.jit
def apply_rotation_xmy_diagonal(coords):
    """Apply C2'' rotation about x,-y diagonal: (x,y,z) → (-y, -x, -z)."""
    result = coords.at[..., [0, 1]].set(coords[..., [1, 0]])
    return result.at[..., [0, 1, 2]].multiply(-1)


@jax.jit
def apply_reflection_xy_diagonal(coords):
    """Apply σd reflection (xy diagonal plane): (x,y,z) → (y, x, z)."""
    return coords.at[..., [0, 1]].set(coords[..., [1, 0]])


@jax.jit
def apply_reflection_xmy_diagonal(coords):
    """Apply σd reflection (x,-y diagonal plane): (x,y,z) → (-y, -x, z)."""
    result = coords.at[..., [0, 1]].set(coords[..., [1, 0]])
    return result.at[..., [0, 1]].multiply(-1)


# see also: pyscf.symm.param.OPERATOR_TABLE
# Keys are the canonical operation symbols used in
# ``POINT_GROUP_OPS``.  User-facing call sites should
# normalize through ``POINT_GROUP_OP_ALIASES`` before
# looking up a function here.
symmetry_operations_map = {
    'E': apply_identity,
    'i': apply_inversion,
    'sx': apply_reflection_x,
    'sy': apply_reflection_y,
    'sz': apply_reflection_z,
    'sxy': apply_reflection_xy_diagonal,
    'sxmy': apply_reflection_xmy_diagonal,
    'Rz90': apply_rotation_z90,
    'Rz180': apply_rotation_z180,
    'Rz270': apply_rotation_z270,
    'Rx180': apply_rotation_x180,
    'Ry180': apply_rotation_y180,
    'C2xy': apply_rotation_xy_diagonal,
    'C2xmy': apply_rotation_xmy_diagonal,
    'S4': apply_S4,
    'S4_3': apply_S4_3,
}


# Map point groups to symmetry operation lists
POINT_GROUP_OPS = {
    'C1': ['E'],
    'Cs': ['E', 'sz'],           # σ_h (horizontal mirror in xy-plane)
    'C2v': ['E', 'Rz180', 'sx', 'sy'],  # C2(z), σ_v(yz), σ_v(xz)
    'C2h': ['E', 'Rz180', 'i', 'sz'],  # C2(z), inversion, σ_h
    'D2h': ['E', 'Rz180', 'Rx180', 'Ry180', 'sx', 'sy', 'sz', 'i'],  # Full D2h
    # Linear molecule approximations
    'C4v': ['E', 'Rz90', 'Rz180', 'Rz270', 'sx', 'sy', 'sxy', 'sxmy'],
    'D4h': ['E', 'Rz90', 'Rz180', 'Rz270', 'i', 'S4_3', 'sz', 'S4',
            'Rx180', 'Ry180', 'C2xy', 'C2xmy', 'sx', 'sy', 'sxy', 'sxmy'],
    'Coov': ['E', 'Rz90', 'Rz180', 'Rz270', 'sx', 'sy', 'sxy', 'sxmy'],
    # → C4v
    'Dooh': ['E', 'Rz90', 'Rz180', 'Rz270', 'i', 'S4_3', 'sz', 'S4',
             'Rx180', 'Ry180', 'C2xy', 'C2xmy', 'sx', 'sy', 'sxy', 'sxmy'],
    # → D4h
}


# Aliases for the canonical operation symbols that
# appear in ``POINT_GROUP_OPS``.  Each key is an
# additional spelling a user might type in
# ``symmop_list``; each value is the canonical symbol
# found in some ``POINT_GROUP_OPS`` entry.  Callers
# that accept user input (e.g.
# ``build_frag_symmops``) can normalize through this
# map before comparing against a fragment's allowed
# ops.
POINT_GROUP_OP_ALIASES = {
    # Identity
    'I': 'E',
    '1': 'E',
    'identity': 'E',

    # Inversion
    '-I': 'i',
    '-1': 'i',
    'inv': 'i',
    'inverse': 'i',
    'inversion': 'i',

    # Mirror planes.  ``POINT_GROUP_OPS`` uses the
    # bare axis symbol for the reflection that
    # negates that Cartesian coordinate.
    'x': 'sx',
    'sigma_x': 'sx',
    'sigma_v_x': 'sx',
    'y': 'sy',
    'sigma_y': 'sy',
    'sigma_v_y': 'sy',
    'z': 'sz',
    'sh': 'sz',
    'sigma_z': 'sz',
    'sigma_h': 'sz',

    # Proper rotations about z
    'C2': 'Rz180',
    'C2z': 'Rz180',
    'Cp4': 'Rz90',
    'Cm4': 'Rz270',
    'C2x': 'Rx180',
    'C2y': 'Ry180',

    # Diagonal σ_d reflections in D4h / C4v
    'sd_xy': 'sxy',
    'sigma_d_xy': 'sxy',
    'sd_xmy': 'sxmy',
    'sigma_d_xmy': 'sxmy',
}


def get_global_symmops(mol: gto.Mole) -> list[str]:
    """Extract symmetry operations valid for the entire molecule.

    For single-fragment molecules, returns all fragment operations.
    For multi-fragment molecules, returns intersection of all fragment
    operations (only globally-valid ops that preserve the entire structure).

    Args:
        mol: PySCF Mole object with map_frag_symmops attribute

    Returns:
        List of symmetry operation strings (e.g.,
        ['E', 'Rz180', 'sx', 'sy'])
    """
    if not hasattr(mol, 'map_frag_symmops') or not mol.map_frag_symmops:
        return ['E']

    frag_ops_list = list(mol.map_frag_symmops.values())

    if len(frag_ops_list) == 0:
        return ['E']

    if len(frag_ops_list) == 1:
        # Single fragment: use all its operations
        return list(frag_ops_list[0])

    # Multi-fragment: compute intersection
    common_ops = set(frag_ops_list[0])
    for frag_ops in frag_ops_list[1:]:
        common_ops &= set(frag_ops)

    # Ensure 'E' (identity) is always present
    common_ops.add('E')

    return list(common_ops) if common_ops else ['E']


def populate_fragment_symmops(mol: gto.Mole):
    """Detect symmetry of each molecular fragment
    and populate map_frag_symmops.

    For each fragment, detects the point group using PySCF and maps it to
    a list of symmetry operation strings compatible
    with symmetry_operations_map.

    Supported point groups: C1, Cs, C2v, C2h, D2h, C4v, D4h
    Linear molecules (Coov, Dooh) are mapped to C4v, D4h respectively.

    Each fragment is detected on its own atoms with PySCF's default
    tolerance, so a slightly distorted fragment drops to a lower
    group (a distorted water is found as Cs rather than C2v).

    Args:
        mol: Molecule with the fragment maps ``map_frag_ctr`` and
            ``map_nuc_frag`` set by
            :func:`~OmegaQMC.utils.parse_molecular_inspheres`.

    Sets:
        ``mol.map_frag_symmops[fid]``: the operations of the
        detected group (:data:`POINT_GROUP_OPS`), or ``['E']`` for
        an unsupported group or a failed detection.

        ``mol.map_frag_axes[fid]``: ``(3, 3)`` rows are PySCF's
        standard-orientation axes in the lab frame, flipped if
        needed to a proper rotation; the identity when detection
        fails.  Multi-atom fragments that apply symmetry operations
        get their frame from a geometric fit in
        :func:`~OmegaQMC.symm.fragments.build_frag_transform_data`
        instead.
    """
    import numpy as _np

    mol.map_frag_symmops = {}
    # Per-fragment axes from PySCF's detect_symm: rows
    # are the standard-orientation axes expressed in the
    # lab frame.  Used by build_frag_transform_data so that
    # ``Rz180``, ``sx`` etc. are interpreted relative to
    # the fragment's actual symmetry axes (PySCF puts the
    # principal C_n along local-z), instead of the SVD
    # principal-moments frame which puts the plane normal
    # along local-z and silently rotates about the wrong
    # axis for C2v / C2h / etc.
    mol.map_frag_axes = {}

    # Build atom list with fragment assignments
    # (from parse_molecular_inspheres)
    # mol.map_nuc_frag[i] gives fragment ID for atom i
    # mol._atom[i] = (symbol, coords) for each atom

    for frag_id in mol.map_frag_ctr.keys():
        # Extract atoms belonging to this fragment
        frag_atoms = []
        for atom_idx, atom_frag_id in enumerate(mol.map_nuc_frag):
            if atom_frag_id == frag_id:
                frag_atoms.append(mol._atom[atom_idx])

        if len(frag_atoms) == 0:
            mol.map_frag_symmops[frag_id] = ['E']
            continue

        # Detect point group for this fragment
        try:
            gpname, _, frag_axes = symm.geom.detect_symm(frag_atoms)
            frag_axes = _np.asarray(frag_axes, dtype=float)
            # detect_symm may return an improper rotation
            # (det = -1).  apply_single_frag_symmop only
            # uses Vh as a basis change; both signs give a
            # consistent "lab ↔ local" map, but flipping
            # to a proper rotation keeps downstream ops
            # (e.g. operation matrices, atom permutations)
            # in a single chirality.
            if _np.linalg.det(frag_axes) < 0:
                frag_axes = frag_axes.copy()
                frag_axes[0] = -frag_axes[0]
            mol.map_frag_axes[frag_id] = frag_axes
        except Exception:
            gpname = 'C1'
            mol.map_frag_axes[frag_id] = _np.eye(3)

        # Map to supported operations (default to C1 if unknown)
        if gpname in POINT_GROUP_OPS:
            mol.map_frag_symmops[frag_id] = POINT_GROUP_OPS[gpname]
        else:
            # For unsupported point groups, fall back to identity only
            mol.map_frag_symmops[frag_id] = ['E']
