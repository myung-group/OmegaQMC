import jax.numpy as jnp
import jax


def diatomic_reflection_electrons(nuc_crds):
    """
    Args:
        nuc_crds: nuclear coordinates [A, B] (2, 3)
    """

    @jax.jit(static_argnames=['reflection'])
    def run_electron_reflection(r_electrons: jax.Array,
                               rescale: jax.Array,
                               reflection: str):

        if reflection in ['y']:
            r_electrons = r_electrons.at[:, 1].multiply(-1)
        elif reflection in ['x']:
            r_electrons = r_electrons.at[:, 0].multiply(-1)
        elif reflection in ['xy']:
            r_electrons = r_electrons.at[:, 0].multiply(-1)
            r_electrons = r_electrons.at[:, 1].multiply(-1)

        return r_electrons

    return run_electron_reflection


def water_reflection_electrons(nuc_crds):
    """
    Args:
        nuc_crds: nuclear coordinates [O, H1, H2] (3, 3)
    """
    r_O, r_H1, r_H2 = nuc_crds[0], nuc_crds[1], nuc_crds[2]

    # Step 1: Symmetrize water
    v_OH1 = r_H1 - r_O
    v_OH2 = r_H2 - r_O
    r_OH1 = jnp.linalg.norm(v_OH1)
    r_OH2 = jnp.linalg.norm(v_OH2)
    r_avg = 0.5 * (r_OH1 + r_OH2)

    r_H1_sym = r_O + r_avg * v_OH1 / r_OH1
    r_H2_sym = r_O + r_avg * v_OH2 / r_OH2
    nuc_sym_crds = jnp.concatenate([r_O, r_H1_sym, r_H2_sym], axis=-1).reshape(-1, 3)

    # Step 2: Rotation Matrix based on water nuclei
    r_H1_shifted = r_H1_sym - r_O
    r_H2_shifted = r_H2_sym - r_O

    midpoint = 0.5 * (r_H1_shifted + r_H2_shifted)
    z_axis = midpoint / jnp.linalg.norm(midpoint)

    v_H1H2 = r_H2_shifted - r_H1_shifted
    x_axis = jnp.cross(z_axis, v_H1H2)
    x_axis = x_axis / jnp.linalg.norm(x_axis)

    y_axis = jnp.cross(z_axis, x_axis)
    Rmat = jnp.column_stack([x_axis, y_axis, z_axis])

    @jax.jit(static_argnames=['reflection'])
    def run_electron_reflection(r_electrons: jax.Array,
                               rescale: jax.Array,  # (b, nelec, n_nuc)
                               reflection: str):
        # Step 3: transform electrons to standard frame
        r_electrons_sym = r_electrons + \
            jnp.einsum('nk,ben->bek',
                       nuc_sym_crds - nuc_crds, rescale)

        nbatch, nelec, ndim = r_electrons_sym.shape
        r_electrons_sym = r_electrons_sym.reshape(-1, ndim)
        r_elec_shifted = r_electrons_sym - r_O
        r_elec_std = (Rmat.T @ r_elec_shifted.T).T

        # Step 4: reflection electrons
        if reflection in ['y']:
            r_elec_std = r_elec_std.at[:, 1].multiply(-1)
        elif reflection in ['x']:
            r_elec_std = r_elec_std.at[:, 0].multiply(-1)
        elif reflection in ['xy', 'yx']:
            r_elec_std = r_elec_std.at[:, 0].multiply(-1)
            r_elec_std = r_elec_std.at[:, 1].multiply(-1)

        # Step 5: inverse transform electrons from standard frame to symmetrized water
        r_elec_sym = (Rmat @ r_elec_std.T).T + r_O

        r_elec_sym = r_elec_sym.reshape(nbatch, nelec, ndim)
        r_elec_orig = r_elec_sym - \
                jnp.einsum('nk,ben->bek',
                           nuc_sym_crds - nuc_crds, rescale)

        return r_elec_orig

    return run_electron_reflection


def water_dimer_reflection_electrons(nuc_crds):
    """
    Water dimer reflection with dynamic electron assignment based on proximity.

    Args:
        nuc_crds: nuclear coordinates [O1, H2, H3, O4, H5, H6] (6, 3)

    Key improvement: Electrons are assigned to water molecules based on which
    oxygen atom they are closer to, not by fixed indexing.
    """
    r_O1, r_H2, r_H3 = nuc_crds[0], nuc_crds[1], nuc_crds[2]
    r_O4, r_H5, r_H6 = nuc_crds[3], nuc_crds[4], nuc_crds[5]

    # Step 1: Symmetrize both water molecules
    # Water 1
    v_O1H2 = r_H2 - r_O1
    v_O1H3 = r_H3 - r_O1
    r_O1H2 = jnp.linalg.norm(v_O1H2)
    r_O1H3 = jnp.linalg.norm(v_O1H3)
    r_avg_1 = 0.5 * (r_O1H2 + r_O1H3)

    r_H2_sym = r_O1 + r_avg_1 * v_O1H2 / r_O1H2
    r_H3_sym = r_O1 + r_avg_1 * v_O1H3 / r_O1H3

    # Water 2
    v_O4H5 = r_H5 - r_O4
    v_O4H6 = r_H6 - r_O4
    r_O4H5 = jnp.linalg.norm(v_O4H5)
    r_O4H6 = jnp.linalg.norm(v_O4H6)
    r_avg_2 = 0.5 * (r_O4H5 + r_O4H6)

    r_H5_sym = r_O4 + r_avg_2 * v_O4H5 / r_O4H5
    r_H6_sym = r_O4 + r_avg_2 * v_O4H6 / r_O4H6

    nuc_sym_crds = jnp.concatenate([r_O1, r_H2_sym, r_H3_sym,
                                    r_O4, r_H5_sym, r_H6_sym], axis=-1).reshape(-1, 3)

    # Step 2: Build rotation matrices for both water molecules
    # Water 1 rotation matrix
    r_H2_shifted = r_H2_sym - r_O1
    r_H3_shifted = r_H3_sym - r_O1

    midpoint_1 = 0.5 * (r_H2_shifted + r_H3_shifted)
    z_axis_1 = midpoint_1 / jnp.linalg.norm(midpoint_1)

    v_H2H3 = r_H3_shifted - r_H2_shifted
    x_axis_1 = jnp.cross(z_axis_1, v_H2H3)
    x_axis_1 = x_axis_1 / jnp.linalg.norm(x_axis_1)

    y_axis_1 = jnp.cross(z_axis_1, x_axis_1)
    Rmat_wat1 = jnp.column_stack([x_axis_1, y_axis_1, z_axis_1])

    # Water 2 rotation matrix
    r_H5_shifted = r_H5_sym - r_O4
    r_H6_shifted = r_H6_sym - r_O4

    midpoint_2 = 0.5 * (r_H5_shifted + r_H6_shifted)
    z_axis_2 = midpoint_2 / jnp.linalg.norm(midpoint_2)

    v_H5H6 = r_H6_shifted - r_H5_shifted
    x_axis_2 = jnp.cross(z_axis_2, v_H5H6)
    x_axis_2 = x_axis_2 / jnp.linalg.norm(x_axis_2)

    y_axis_2 = jnp.cross(z_axis_2, x_axis_2)
    Rmat_wat2 = jnp.column_stack([x_axis_2, y_axis_2, z_axis_2])

    @jax.jit(static_argnames=['reflection'])
    def run_electron_reflection(r_electrons: jax.Array,
                               rescale: jax.Array,  # (b, nelec, n_nuc)
                               reflection: str):
        """
        Apply reflection to electrons with dynamic water assignment.

        Args:
            r_electrons: (nbatch, nelec, 3)
            rescale: (nbatch, nelec, n_nuc)
            reflection: 'x', 'y', or 'xy'
        """
        nbatch, nelec, ndim = r_electrons.shape

        # Step 3: Transform electrons to symmetrized coordinates
        r_electrons_sym = r_electrons + \
            jnp.einsum('nk,ben->bek',
                       nuc_sym_crds - nuc_crds, rescale)

        # Step 3.5: Determine which water each electron belongs to
        # Calculate distances to both oxygen atoms
        # Shape: (nbatch, nelec)
        dist_to_O1 = jnp.linalg.norm(r_electrons_sym - r_O1[None, None, :], axis=-1)
        dist_to_O4 = jnp.linalg.norm(r_electrons_sym - r_O4[None, None, :], axis=-1)

        # Create mask: True if electron belongs to water 1, False for water 2
        # Shape: (nbatch, nelec)
        belongs_to_wat1 = dist_to_O1 < dist_to_O4

        # Reshape for vectorized operations
        r_electrons_sym_flat = r_electrons_sym.reshape(-1, ndim)  # (nbatch*nelec, 3)
        belongs_to_wat1_flat = belongs_to_wat1.reshape(-1)  # (nbatch*nelec,)

        # Step 4: Transform to standard frame and apply reflection
        # We'll process all electrons but use different rotation matrices

        # Transform relative to O1 (for water 1 electrons)
        r_elec_shifted_wat1 = r_electrons_sym_flat - r_O1
        r_elec_std_wat1 = (Rmat_wat1.T @ r_elec_shifted_wat1.T).T

        # Transform relative to O4 (for water 2 electrons)
        r_elec_shifted_wat2 = r_electrons_sym_flat - r_O4
        r_elec_std_wat2 = (Rmat_wat2.T @ r_elec_shifted_wat2.T).T

        # Apply reflection to both sets
        if reflection == 'y':
            r_elec_std_wat1 = r_elec_std_wat1.at[:, 1].multiply(-1)
            r_elec_std_wat2 = r_elec_std_wat2.at[:, 1].multiply(-1)
        elif reflection == 'x':
            r_elec_std_wat1 = r_elec_std_wat1.at[:, 0].multiply(-1)
            r_elec_std_wat2 = r_elec_std_wat2.at[:, 0].multiply(-1)
        elif reflection == 'xy':
            r_elec_std_wat1 = r_elec_std_wat1.at[:, 0].multiply(-1)
            r_elec_std_wat1 = r_elec_std_wat1.at[:, 1].multiply(-1)
            r_elec_std_wat2 = r_elec_std_wat2.at[:, 0].multiply(-1)
            r_elec_std_wat2 = r_elec_std_wat2.at[:, 1].multiply(-1)

        # Step 5: Inverse transform back to symmetrized coordinates
        r_elec_sym_wat1 = (Rmat_wat1 @ r_elec_std_wat1.T).T + r_O1
        r_elec_sym_wat2 = (Rmat_wat2 @ r_elec_std_wat2.T).T + r_O4

        # Select the correct transformed coordinates based on assignment
        # Use where to select between the two transformations
        r_elec_sym_combined = jnp.where(
            belongs_to_wat1_flat[:, None],  # (nbatch*nelec, 1)
            r_elec_sym_wat1,                # (nbatch*nelec, 3)
            r_elec_sym_wat2                 # (nbatch*nelec, 3)
        )

        # Reshape back
        r_elec_sym = r_elec_sym_combined.reshape(nbatch, nelec, ndim)

        # Step 6: Transform back to original (unsymmetrized) coordinates
        r_elec_orig = r_elec_sym - \
                jnp.einsum('nk,ben->bek',
                           nuc_sym_crds - nuc_crds, rescale)

        return r_elec_orig

    return run_electron_reflection


def water_dimer_reflection_electrons_debug(nuc_crds):
    """
    Version with debugging output to verify electron assignments.
    """
    r_O1, r_H2, r_H3 = nuc_crds[0], nuc_crds[1], nuc_crds[2]
    r_O4, r_H5, r_H6 = nuc_crds[3], nuc_crds[4], nuc_crds[5]

    # Symmetrization (same as above)
    v_O1H2 = r_H2 - r_O1
    v_O1H3 = r_H3 - r_O1
    r_O1H2 = jnp.linalg.norm(v_O1H2)
    r_O1H3 = jnp.linalg.norm(v_O1H3)
    r_avg_1 = 0.5 * (r_O1H2 + r_O1H3)

    r_H2_sym = r_O1 + r_avg_1 * v_O1H2 / r_O1H2
    r_H3_sym = r_O1 + r_avg_1 * v_O1H3 / r_O1H3

    v_O4H5 = r_H5 - r_O4
    v_O4H6 = r_H6 - r_O4
    r_O4H5 = jnp.linalg.norm(v_O4H5)
    r_O4H6 = jnp.linalg.norm(v_O4H6)
    r_avg_2 = 0.5 * (r_O4H5 + r_O4H6)

    r_H5_sym = r_O4 + r_avg_2 * v_O4H5 / r_O4H5
    r_H6_sym = r_O4 + r_avg_2 * v_O4H6 / r_O4H6

    nuc_sym_crds = jnp.concatenate([r_O1, r_H2_sym, r_H3_sym,
                                    r_O4, r_H5_sym, r_H6_sym], axis=-1).reshape(-1, 3)

    # Rotation matrices (same as above)
    r_H2_shifted = r_H2_sym - r_O1
    r_H3_shifted = r_H3_sym - r_O1
    midpoint_1 = 0.5 * (r_H2_shifted + r_H3_shifted)
    z_axis_1 = midpoint_1 / jnp.linalg.norm(midpoint_1)
    v_H2H3 = r_H3_shifted - r_H2_shifted
    x_axis_1 = jnp.cross(z_axis_1, v_H2H3)
    x_axis_1 = x_axis_1 / jnp.linalg.norm(x_axis_1)
    y_axis_1 = jnp.cross(z_axis_1, x_axis_1)
    Rmat_wat1 = jnp.column_stack([x_axis_1, y_axis_1, z_axis_1])

    r_H5_shifted = r_H5_sym - r_O4
    r_H6_shifted = r_H6_sym - r_O4
    midpoint_2 = 0.5 * (r_H5_shifted + r_H6_shifted)
    z_axis_2 = midpoint_2 / jnp.linalg.norm(midpoint_2)
    v_H5H6 = r_H6_shifted - r_H5_shifted
    x_axis_2 = jnp.cross(z_axis_2, v_H5H6)
    x_axis_2 = x_axis_2 / jnp.linalg.norm(x_axis_2)
    y_axis_2 = jnp.cross(z_axis_2, x_axis_2)
    Rmat_wat2 = jnp.column_stack([x_axis_2, y_axis_2, z_axis_2])

    def run_electron_reflection_debug(r_electrons: jax.Array,
                                     rescale: jax.Array,
                                     reflection: str,
                                     print_assignment: bool = True):
        """Debug version that prints electron assignments."""
        nbatch, nelec, ndim = r_electrons.shape

        r_electrons_sym = r_electrons + \
            jnp.einsum('nk,ben->bek',
                       nuc_sym_crds - nuc_crds, rescale)

        # Determine assignments
        dist_to_O1 = jnp.linalg.norm(r_electrons_sym - r_O1[None, None, :], axis=-1)
        dist_to_O4 = jnp.linalg.norm(r_electrons_sym - r_O4[None, None, :], axis=-1)
        belongs_to_wat1 = dist_to_O1 < dist_to_O4

        if print_assignment:
            print("\nElectron assignments:")
            print(f"{'Electron':<10} {'Water':<8} {'Dist to O1':<12} {'Dist to O4':<12}")
            print("-" * 50)
            for i in range(nelec):
                water = "Water 1" if belongs_to_wat1[0, i] else "Water 2"
                print(f"{i:<10} {water:<8} {dist_to_O1[0, i]:<12.4f} {dist_to_O4[0, i]:<12.4f}")

            n_wat1 = belongs_to_wat1[0].sum()
            n_wat2 = nelec - n_wat1
            print(f"\nTotal: {n_wat1} electrons in Water 1, {n_wat2} electrons in Water 2")

        # Apply transformations (same as main function)
        r_electrons_sym_flat = r_electrons_sym.reshape(-1, ndim)
        belongs_to_wat1_flat = belongs_to_wat1.reshape(-1)

        r_elec_shifted_wat1 = r_electrons_sym_flat - r_O1
        r_elec_std_wat1 = (Rmat_wat1.T @ r_elec_shifted_wat1.T).T

        r_elec_shifted_wat2 = r_electrons_sym_flat - r_O4
        r_elec_std_wat2 = (Rmat_wat2.T @ r_elec_shifted_wat2.T).T

        if reflection == 'y':
            r_elec_std_wat1 = r_elec_std_wat1.at[:, 1].multiply(-1)
            r_elec_std_wat2 = r_elec_std_wat2.at[:, 1].multiply(-1)
        elif reflection == 'x':
            r_elec_std_wat1 = r_elec_std_wat1.at[:, 0].multiply(-1)
            r_elec_std_wat2 = r_elec_std_wat2.at[:, 0].multiply(-1)
        elif reflection == 'xy':
            r_elec_std_wat1 = r_elec_std_wat1.at[:, 0].multiply(-1)
            r_elec_std_wat1 = r_elec_std_wat1.at[:, 1].multiply(-1)
            r_elec_std_wat2 = r_elec_std_wat2.at[:, 0].multiply(-1)
            r_elec_std_wat2 = r_elec_std_wat2.at[:, 1].multiply(-1)

        r_elec_sym_wat1 = (Rmat_wat1 @ r_elec_std_wat1.T).T + r_O1
        r_elec_sym_wat2 = (Rmat_wat2 @ r_elec_std_wat2.T).T + r_O4

        r_elec_sym_combined = jnp.where(
            belongs_to_wat1_flat[:, None],
            r_elec_sym_wat1,
            r_elec_sym_wat2
        )

        r_elec_sym = r_elec_sym_combined.reshape(nbatch, nelec, ndim)
        r_elec_orig = r_elec_sym - \
                jnp.einsum('nk,ben->bek',
                           nuc_sym_crds - nuc_crds, rescale)

        return r_elec_orig

    return run_electron_reflection_debug


if __name__ == "__main__":
    # Test setup
    r_O1 = jnp.array([0.0, 0.0, 0.0])
    r_H2 = jnp.array([0.96, 0.0, 0.0])
    r_H3 = jnp.array([0.0, 0.99, 0.0])
    r_O4 = jnp.array([0.0, 0.0, 3.0])
    r_H5 = jnp.array([0.96, 0.0, 3.0])
    r_H6 = jnp.array([0.0, 0.99, 3.0])
    nuc_crds = jnp.concatenate([r_O1, r_H2, r_H3,
                                r_O4, r_H5, r_H6], axis=0).reshape(-1, 3)

    def redistribute_samples_scheme2(elec_crds, nuc_crds):
        diff = elec_crds[:, None, :] - nuc_crds[None, :, :]
        dist = jnp.sqrt(jnp.sum(diff**2, axis=-1))
        dist = jnp.where(dist < 1e-12, 1e-12, dist)
        weight = dist**(-4.0)
        return weight / jnp.sum(weight, axis=-1, keepdims=True)

    # Create test electrons
    idx_cnt = []
    for ia, iz in enumerate([8, 1, 1, 8, 1, 1]):
        idx_cnt.extend([ia] * iz)
    idx_cnt = jnp.array(idx_cnt)
    nelec = idx_cnt.shape[0]
    centers = nuc_crds[idx_cnt]
    walkers = centers[jnp.newaxis, :, :] + 0.1 * jax.random.normal(
        jax.random.key(0), (1, nelec, 3)
    )

    # Place some electrons specifically to test assignment
    walkers = walkers.at[:, -2].set([0.1, 1.0, 3.0])  # Near water 2
    walkers = walkers.at[:, -1].set([-0.2, 0.95, 3.1])  # Near water 2

    rescale = redistribute_samples_scheme2(walkers[0], nuc_crds)
    rescale = rescale[None, :, :]

    print("=" * 70)
    print("Testing Water Dimer Electron Reflection with Dynamic Assignment")
    print("=" * 70)

    # Test with debug version
    print("\n1. Testing with debug version (shows assignments):")
    run_elec_reflection_debug = water_dimer_reflection_electrons_debug(nuc_crds)
    walkers_reflection_debug = run_elec_reflection_debug(
        walkers, rescale, 'y', print_assignment=True
    )

    print("\n" + "=" * 90)
    print("2. Testing with optimized version (JIT-compiled):")
    run_elec_reflection = water_dimer_reflection_electrons_dynamic(nuc_crds)
    walkers_reflection = run_elec_reflection(walkers, rescale, 'y')

    # Verify results
    print("\nInitial electron positions:")
    print(f"{'Electron':<10} {'Position':<35} {'O1 dist':<10} {'O4 dist':<10}  "
          f"{'H1_dist ':<10} {'H2_dist':<10}")

    print("-" * 90)
    for i, r_e in enumerate(walkers[0]):
        dist_O1 = jnp.linalg.norm(r_e - r_O1)
        dist_O4 = jnp.linalg.norm(r_e - r_O4)
        if (dist_O1 < dist_O4):
            dist_H1 = jnp.linalg.norm(r_e - r_H2)
            dist_H2 = jnp.linalg.norm(r_e - r_H3)
        else:
            dist_H1 = jnp.linalg.norm(r_e - r_H5)
            dist_H2 = jnp.linalg.norm(r_e - r_H6)
        print(f"{i:<10} {str(r_e):<35} {dist_O1:<10.4f} {dist_O4:<10.4f}  "
              f"{dist_H1:<10.4f}  {dist_H2:<10.4f}")

    print("\nReflected electron positions:")
    print(f"{'Electron':<10} {'Position':<35} {'O1 dist':<10} {'O4 dist':<10}  "
          f"{'H1_dist ':<10} {'H2_dist':<10}")
    print("-" * 90)
    for i, r_e in enumerate(walkers_reflection[0]):
        dist_O1 = jnp.linalg.norm(r_e - r_O1)
        dist_O4 = jnp.linalg.norm(r_e - r_O4)
        if (dist_O1 < dist_O4):
            dist_H1 = jnp.linalg.norm(r_e - r_H2)
            dist_H2 = jnp.linalg.norm(r_e - r_H3)
        else:
            dist_H1 = jnp.linalg.norm(r_e - r_H5)
            dist_H2 = jnp.linalg.norm(r_e - r_H6)
        print(f"{i:<10} {str(r_e):<35} {dist_O1:<10.4f} {dist_O4:<10.4f}  "
              f"{dist_H1:<10.4f}  {dist_H2:<10.4f}")

    print("\n" + "=" * 90)
    print("Verification: Check if electrons stay with their assigned water")
    print("=" * 90)

    # Check if assignment is preserved
    dist_before_O1 = jnp.linalg.norm(walkers[0] - r_O1[None, :], axis=-1)
    dist_before_O4 = jnp.linalg.norm(walkers[0] - r_O4[None, :], axis=-1)
    assigned_to_wat1_before = dist_before_O1 < dist_before_O4

    dist_after_O1 = jnp.linalg.norm(walkers_reflection[0] - r_O1[None, :], axis=-1)
    dist_after_O4 = jnp.linalg.norm(walkers_reflection[0] - r_O4[None, :], axis=-1)
    assigned_to_wat1_after = dist_after_O1 < dist_after_O4

    assignment_preserved = jnp.all(assigned_to_wat1_before == assigned_to_wat1_after)
    print(f"\nAssignment preserved after reflection: {assignment_preserved}")

    if not assignment_preserved:
        changed = jnp.where(assigned_to_wat1_before != assigned_to_wat1_after)[0]
        print(f"⚠️  Electrons that changed water: {changed}")
    else:
        print("✓ All electrons stayed with their assigned water molecule")
