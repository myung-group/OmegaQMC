import jax


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
    """Apply S4 improper rotation (C4³ followed by σh): (x,y,z) → (y, -x, -z)."""
    result = coords.at[..., [0, 1]].set(coords[..., [1, 0]])
    return result.at[..., [1, 2]].multiply(-1)


@jax.jit
def apply_S4_3(coords):
    """Apply S4³ improper rotation (C4 followed by σh): (x,y,z) → (-y, x, -z)."""
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
