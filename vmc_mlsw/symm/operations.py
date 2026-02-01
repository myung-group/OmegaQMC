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


# see also: pyscf.symm.param.OPERATOR_TABLE
symmetry_operations_map = {
    'I': apply_identity,
    'E': apply_identity,
    '1': apply_identity,
    'i': apply_inversion,
    '-I': apply_inversion,
    '-1': apply_inversion,
    'inv': apply_inversion,
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
    'xy': apply_rotation_z180,
    'Rz180': apply_rotation_z180,
    'C2z': apply_rotation_z180,
    'C2': apply_rotation_z180
}
