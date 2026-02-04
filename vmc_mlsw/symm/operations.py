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
symmetry_operations_map = {
    'I': apply_identity,
    'E': apply_identity,
    '1': apply_identity,
    'i': apply_inversion,
    '-I': apply_inversion,
    '-1': apply_inversion,
    'x': apply_reflection_x,
    'y': apply_reflection_y,
    'z': apply_reflection_z,
    'sx': apply_reflection_x,
    'sy': apply_reflection_y,
    'sz': apply_reflection_z,
    'sh': apply_reflection_z,
    'Cp4': apply_rotation_z90,
    'Rz90': apply_rotation_z90,
    'Cm4': apply_rotation_z270,
    'Rz270': apply_rotation_z270,
    'C2x': apply_rotation_x180,
    'C2y': apply_rotation_y180,
    'S4': apply_S4,
    'S4_3': apply_S4_3,
    'C2xy': apply_rotation_xy_diagonal,
    'C2xmy': apply_rotation_xmy_diagonal,
    'sxy': apply_reflection_xy_diagonal,
    'sxmy': apply_reflection_xmy_diagonal,
    'xy': apply_rotation_z180,
    'Rz180': apply_rotation_z180,
    'C2z': apply_rotation_z180,
    'C2': apply_rotation_z180
}


def populate_fragment_symmops(mol: gto.Mole):
    """Detect symmetry of each molecular fragment
    and populate map_frag_symmops.

    For each fragment, detects the point group using PySCF and maps it to
    a list of symmetry operation strings compatible
    with symmetry_operations_map.

    Supported point groups: C1, Cs, C2v, C2h, D2h
    """
    # Map point groups to symmetry operation lists
    POINT_GROUP_OPS = {
        'C1': ['E'],
        'Cs': ['E', 'z'],           # σ_h (horizontal mirror in xy-plane)
        'C2v': ['E', 'C2', 'x', 'y'],  # C2(z), σ_v(yz), σ_v(xz)
        'C2h': ['E', 'C2', 'i', 'z'],  # C2(z), inversion, σ_h
        'D2h': ['E', 'C2z', 'C2x', 'C2y', 'x', 'y', 'z', 'i'],  # Full D2h
        # Linear molecule approximations
        'C4v': ['E', 'Rz90', 'C2z', 'Rz270', 'x', 'y', 'sxy', 'sxmy'],
        'D4h': ['E', 'Rz90', 'C2z', 'Rz270', 'i', 'S4_3', 'z', 'S4',
                'C2x', 'C2y', 'C2xy', 'C2xmy', 'x', 'y', 'sxy', 'sxmy'],
        'Coov': ['E', 'Rz90', 'C2z', 'Rz270', 'x', 'y', 'sxy', 'sxmy'],
        # → C4v
        'Dooh': ['E', 'Rz90', 'C2z', 'Rz270', 'i', 'S4_3', 'z', 'S4',
                 'C2x', 'C2y', 'C2xy', 'C2xmy', 'x', 'y', 'sxy', 'sxmy'],
        # → D4h
    }

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
