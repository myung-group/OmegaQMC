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


symmetry_operations_map = {
    'I': apply_identity,
    '1': apply_identity,
    '-I': apply_inversion,
    '-1': apply_inversion,
    'inv': apply_inversion,
    'x': apply_reflection_x,
    'y': apply_reflection_y,
    'z': apply_reflection_z,
    'sigma_x': apply_reflection_x,
    'sigma_y': apply_reflection_y,
    'sigma_z': apply_reflection_z,
    'Cp4': apply_rotation_z90,
    'Rz90': apply_rotation_z90,
    'Cm4': apply_rotation_z270,
    'Rz270': apply_rotation_z270,
    'xy': apply_rotation_z180,
    'Rz180': apply_rotation_z180,
    'C2': apply_rotation_z180
}
