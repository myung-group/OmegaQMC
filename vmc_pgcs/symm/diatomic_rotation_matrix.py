import jax.numpy as jnp
import jax


@jax.jit
def rotate_vector_to_z_axis(vector):
    """
    Generate rotation matrix to align a vector with the z-axis.

    This function creates a rotation matrix R such that:
    R @ vector = [0, 0, |vector|]

    Args:
        vector: 3D vector (a, b, c) to be rotated to z-axis

    Returns:
        rotation_matrix: 3x3 rotation matrix

    Note:
        If the input vector is already aligned with z-axis (or -z-axis),
        returns identity matrix (or reflection matrix respectively).
    """
    vector = jnp.array(vector, dtype=jnp.float64)

    # Normalize the input vector
    norm = jnp.linalg.norm(vector)

    # Handle zero vector case using jnp.where
    def zero_case():
        return jnp.eye(3)

    def non_zero_case():
        # Normalized input vector
        v = vector / norm

        # Target vector (z-axis)
        z_axis = jnp.array([0.0, 0.0, 1.0])

        # Check if vector is already aligned with z-axis
        dot_product = jnp.dot(v, z_axis)

        # Use jax.lax.cond for conditional logic

        def aligned_pos_z():
            return jnp.eye(3)

        def aligned_neg_z():
            return jnp.diag(jnp.array([1.0, 1.0, -1.0]))

        def general_case():
            # Rotation axis: cross product of v and z_axis
            k = jnp.cross(v, z_axis)
            k = k / jnp.linalg.norm(k)  # Normalize rotation axis

            # Rotation angle
            cos_theta = dot_product
            sin_theta = jnp.sqrt(1.0 - cos_theta**2)

            # Rodrigues' rotation formula: R = I + sin(θ)[k]× + (1-cos(θ))[k]×²
            K = jnp.array([
                [0.0, -k[2], k[1]],
                [k[2], 0.0, -k[0]],
                [-k[1], k[0], 0.0]
            ])

            rotation_matrix = (jnp.eye(3) +
                               sin_theta * K +
                               (1.0 - cos_theta) * jnp.dot(K, K))

            return rotation_matrix

        # Handle the different cases for the dot product
        return jax.lax.cond(
            jnp.abs(dot_product - 1.0) < 1e-12,
            aligned_pos_z,
            lambda: jax.lax.cond(
                jnp.abs(dot_product + 1.0) < 1e-12,
                aligned_neg_z,
                general_case
            )
        )

    # Use jnp.where for the top-level condition on the vector norm
    return jax.lax.cond(
        norm < 1e-12,
        zero_case,
        non_zero_case
    )


def demonstrate_diatomic_rotation():
    """
    Demonstrate rotation matrix generation for diatomic molecules.
    """
    print("=== Diatomic Molecule Rotation Matrix Generator ===\n")

    # Example 1: Simple vector
    print("Example 1: Rotating vector (1, 1, 1) to z-axis")
    vector1 = jnp.array([1.0, 1.0, 1.0])
    R1 = rotate_vector_to_z_axis(vector1)

    print(f"Original vector: {vector1}")
    print(f"Rotation matrix:\n{R1}")

    rotated1 = R1 @ vector1
    print(f"Rotated vector: {rotated1}")
    print(f"Expected: [0, 0, {jnp.linalg.norm(vector1):.6f}]")
    print("Error: {:.2e}\n"
          .format(jnp.linalg.norm(
              rotated1 - jnp.array([0, 0, jnp.linalg.norm(vector1)])
              )))

    # Example 2: Diatomic molecule coordinates
    print("Example 2: H2 molecule")
    L = 0.7414      # H-H bond length ~0.7414 Å
    A_H2 = jnp.array([0.0, 0.0, -L/2])
    B_H2 = jnp.array([0.0, 0.0, L/2])
    AB_vector = B_H2 - A_H2

    R2 = rotate_vector_to_z_axis(AB_vector)

    print("H2 molecule:")
    print(f"  A (H1): {A_H2}")
    print(f"  B (H2): {B_H2}")
    print(f"  AB vector: {AB_vector}")
    print(f"Rotation matrix:\n{R2}")

    # Rotate both atoms
    A_rotated = R2 @ A_H2
    B_rotated = R2 @ B_H2
    AB_rotated = B_rotated - A_rotated

    print("After rotation:")
    print(f"  A (H1): {A_rotated}")
    print(f"  B (H2): {B_rotated}")
    print(f"  AB vector: {AB_rotated}")
    print(f"Bond length preserved: {jnp.linalg.norm(AB_rotated):.6f} "
          f"(original: {jnp.linalg.norm(AB_vector):.6f})\n")

    # Example 3: CO molecule
    print("Example 3: CO molecule")
    A_CO = jnp.array([1.2, -0.5, 0.8])  # C atom
    B_CO = jnp.array([2.3, 0.1, -0.4])  # O atom
    AB_CO = B_CO - A_CO

    R3 = rotate_vector_to_z_axis(AB_CO)

    print("CO molecule:")
    print(f"  A (C): {A_CO}")
    print(f"  B (O): {B_CO}")
    print(f"  AB vector: {AB_CO}")
    print(f"Rotation matrix:\n{R3}")

    A_CO_rot = R3 @ A_CO
    B_CO_rot = R3 @ B_CO
    AB_CO_rot = B_CO_rot - A_CO_rot

    print("After rotation:")
    print(f"  A (C): {A_CO_rot}")
    print(f"  B (O): {B_CO_rot}")
    print(f"  AB vector: {AB_CO_rot}")
    print(f"Z-alignment check: {jnp.abs(AB_CO_rot[0]):.2e}, "
          f"{jnp.abs(AB_CO_rot[1]):.2e}, {AB_CO_rot[2]:.6f}\n")

    # Verification: Check that rotation matrices are orthogonal
    print("=== Matrix Properties Verification ===")
    for i, (name, R) in enumerate([("Example 1", R1),
                                   ("Example 2", R2),
                                   ("Example 3", R3)], 1):
        det_R = jnp.linalg.det(R)
        orthogonality_error = jnp.linalg.norm(R @ R.T - jnp.eye(3))
        print(f"{name}: det(R) = {det_R:.6f}, "
              f"orthogonality error = {orthogonality_error:.2e}")


# Main execution
if __name__ == "__main__":
    demonstrate_diatomic_rotation()

    print("\n=== Quick Usage Guide ===")
    print("# For diatomic molecule AB with coordinates A and B:")
    print("AB_vector = B - A")
    print("rotation_matrix = rotate_vector_to_z_axis(AB_vector)")
    print("# Apply to coordinates:")
    print("A_rotated = rotation_matrix @ A")
    print("B_rotated = rotation_matrix @ B")
    print("# Now AB vector is aligned with z-axis")
