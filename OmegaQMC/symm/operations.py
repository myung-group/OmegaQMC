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
    """
    mol.map_frag_symmops = {}

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
            gpname, _, _ = symm.geom.detect_symm(frag_atoms)
        except Exception:
            gpname = 'C1'

        # Map to supported operations (default to C1 if unknown)
        if gpname in POINT_GROUP_OPS:
            mol.map_frag_symmops[frag_id] = POINT_GROUP_OPS[gpname]
        else:
            # For unsupported point groups, fall back to identity only
            mol.map_frag_symmops[frag_id] = ['E']
