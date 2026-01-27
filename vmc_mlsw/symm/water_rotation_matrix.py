import jax.numpy as jnp
import jax


@jax.jit
def symmetrize_water_molecule(nuc_crds):
    """
    Symmetrize a water molecule by making OH bond lengths equal.
    
    Args:
        nuc_crds: nuclear coordinates [O, H1, H2] (3, 3)
    
    Returns:
        nuc_crds_sym: symmetrized nuclear coordinates
        rot_mat: rotation matrix for coordinate transformation
    """
    O, H1, H2 = nuc_crds[0], nuc_crds[1], nuc_crds[2]
    
    # Calculate OH vectors and distances
    OH1_vec = H1 - O
    OH2_vec = H2 - O
    
    rOH1 = jnp.linalg.norm(OH1_vec)
    rOH2 = jnp.linalg.norm(OH2_vec)
    rOH = 0.5 * (rOH1 + rOH2)  # Average bond length
    
    # Create symmetric H positions
    H1_new = O + rOH * OH1_vec / rOH1
    H2_new = O + rOH * OH2_vec / rOH2
    
    nuc_crds_sym = jnp.array([O, H1_new, H2_new])
    
    # Build coordinate system for rotation
    bisector = OH1_vec / rOH1 + OH2_vec / rOH2
    bisector = bisector / jnp.linalg.norm(bisector)
    
    normal = jnp.cross(OH1_vec, OH2_vec)
    normal = normal / jnp.linalg.norm(normal)
    
    reflection_normal = jnp.cross(bisector, normal)
    reflection_normal = reflection_normal / jnp.linalg.norm(reflection_normal)
    
    # Rotation matrix: normal->x, reflection_normal->y, bisector->z
    rot_mat = jnp.array([normal, reflection_normal, bisector])
    
    return nuc_crds_sym, rot_mat.T

