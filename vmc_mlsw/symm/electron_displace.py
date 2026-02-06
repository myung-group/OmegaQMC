"""
Electron relocation operations for VMC symmetry exploitation.

This module provides symmetry operations (reflections and rotations) for
electrons in molecular systems, used to improve sampling efficiency in
Variational Monte Carlo calculations.
"""

import jax
import jax.numpy as jnp
from collections.abc import Callable

from .operations import symmetry_operations_map, apply_reflection_z

# Canonical ordering of symmetry operations.
# This list includes all operations from symmetry_operations_map that might
# be used in symmop_list (from POINT_GROUP_OPS).
SYMM_OP_LABELS = [
    'E',      # 0: Identity (PySCF style)
    'x',      # 1: Reflect across yz-plane (sx)
    'y',      # 2: Reflect across xz-plane (sy)
    'z',      # 3: Reflect across xy-plane (sz, sh)
    'Rz90',   # 4: 90° rotation about z
    'Rz180',  # 5: 180° rotation about z (C2, C2z)
    'Rz270',  # 6: 270° rotation about z
    'i',      # 7: Inversion
    'C2x',    # 8: 180° rotation about x
    'C2y',    # 9: 180° rotation about y
    'S4',     # 10: S4 improper rotation
    'S4_3',   # 11: S4^3 improper rotation
    'C2xy',   # 12: 180° rotation about xy diagonal
    'C2xmy',  # 13: 180° rotation about x,-y diagonal
    'sxy',    # 14: Diagonal mirror (xy plane)
    'sxmy',   # 15: Diagonal mirror (x,-y plane)
]

# Tuple of symmetry-operation functions corresponding to SYMM_OP_LABELS.
_SYMM_OP_FUNCS = tuple(symmetry_operations_map[label]
                       for label in SYMM_OP_LABELS)

# Mapping from operation string labels to integer indices.
# Includes aliases for PySCF-style labels (E, C2, C2z, sx, sy, sz, sh).
SYMM_OP_STRING_TO_ID = {label: i for i, label in enumerate(SYMM_OP_LABELS)}
# Add aliases
SYMM_OP_STRING_TO_ID.update({
    'I': 0,       # Alias for E
    'C2': 5,      # Alias for Rz180
    'C2z': 5,     # Alias for Rz180
    'sx': 1,      # Alias for x
    'sy': 2,      # Alias for y
    'sz': 3,      # Alias for z
    'sh': 3,      # Alias for z (horizontal mirror)
    'xy': 5,      # Alias for Rz180
})


def _apply_symmetry_operation(
    r_electrons: jax.Array,
    symm_op_id: int
) -> jax.Array:
    """
    Apply a symmetry operation (reflection or rotation)
    to electron coordinates.

    Args:
        r_electrons: Electron positions with shape (nelec, 3)
        symm_op_id: Integer ID indexing `SYMM_OP_LABELS`

    Returns:
        Transformed electron positions with shape (nelec, 3)
    """
    # Use JAX control flow to select the appropriate symmetry function,
    # which is implemented in `vmc_mlsw/symm/operations.py`.
    return jax.lax.switch(symm_op_id, _SYMM_OP_FUNCS, r_electrons)


def _symmetrize_water(r_O: jax.Array, r_H1: jax.Array, r_H2: jax.Array):
    """
    Symmetrize a water molecule by averaging OH bond lengths.

    Args:
        r_O: Oxygen position (3,)
        r_H1: First hydrogen position (3,)
        r_H2: Second hydrogen position (3,)

    Returns:
        Tuple of (r_H1_symm, r_H2_symm) with equalized bond lengths
    """
    v_OH1 = r_H1 - r_O
    v_OH2 = r_H2 - r_O
    r_OH1 = jnp.linalg.norm(v_OH1)
    r_OH2 = jnp.linalg.norm(v_OH2)
    r_avg = 0.5 * (r_OH1 + r_OH2)

    r_H1_symm = r_O + r_avg * v_OH1 / r_OH1
    r_H2_symm = r_O + r_avg * v_OH2 / r_OH2

    return r_H1_symm, r_H2_symm


def _build_water_rotation_matrix(r_O: jax.Array,
                                 r_H1_symm: jax.Array,
                                 r_H2_symm: jax.Array) -> jax.Array:
    """
    Build rotation matrix for transforming to water's standard frame.

    The standard frame has:
    - z-axis along the bisector of the H-O-H angle
    - x-axis perpendicular to the molecular plane
    - y-axis completing the right-handed system

    Args:
        r_O: Oxygen position (3,)
        r_H1_symm: First symmetrized hydrogen position (3,)
        r_H2_symm: Second symmetrized hydrogen position (3,)

    Returns:
        Rotation matrix (3, 3)
    """
    r_H1_shifted = r_H1_symm - r_O
    r_H2_shifted = r_H2_symm - r_O

    # z-axis: bisector direction
    midpoint = 0.5 * (r_H1_shifted + r_H2_shifted)
    z_axis = midpoint / jnp.linalg.norm(midpoint)

    # x-axis: perpendicular to molecular plane
    v_H1H2 = r_H2_shifted - r_H1_shifted
    x_axis = jnp.cross(z_axis, v_H1H2)
    x_axis = x_axis / jnp.linalg.norm(x_axis)

    # y-axis: complete right-handed system
    y_axis = jnp.cross(z_axis, x_axis)

    return jnp.column_stack([x_axis, y_axis, z_axis])


def diatomic_reflection_electrons(nuc_crds: jax.Array) -> Callable:
    """
    Create electron reflection function for diatomic molecules.

    For diatomic molecules, reflection is applied directly without
    coordinate transformation since the molecule is already aligned.

    Args:
        nuc_crds: Nuclear coordinates with shape (2, 3)

    Returns:
        Function that applies a symmetry operation to electron coordinates
    """
    @jax.jit
    def run_electron_reflection(r_electrons: jax.Array,
                                rescale: jax.Array,
                                reflection_ID: int) -> jax.Array:
        # `reflection_ID` is kept for backward compatibility; it indexes
        # the unified symmetry-operation matrix table.
        return _apply_symmetry_operation(r_electrons, reflection_ID)

    return run_electron_reflection


def water_reflection_electrons(nuc_crds: jax.Array) -> Callable:
    """
    Create electron reflection function for a single water molecule.

    Args:
        nuc_crds: Nuclear coordinates [O, H1, H2] with shape (3, 3)

    Returns:
        Function that applies a symmetry operation to electron coordinates
    """
    r_O, r_H1, r_H2 = nuc_crds[0], nuc_crds[1], nuc_crds[2]

    # Symmetrize water molecule
    r_H1_symm, r_H2_symm = _symmetrize_water(r_O, r_H1, r_H2)
    nuc_symm_crds = jnp.stack([r_O, r_H1_symm, r_H2_symm])

    # Build rotation matrix
    Rmat = _build_water_rotation_matrix(r_O, r_H1_symm, r_H2_symm)

    # Precompute coordinate shift
    coord_shift = nuc_symm_crds - nuc_crds

    @jax.jit
    def run_electron_reflection(r_electrons: jax.Array,
                                rescale: jax.Array,
                                reflection_ID: int) -> jax.Array:
        """
        Apply reflection to electrons in water molecule frame.

        Args:
            r_electrons: Electron positions (nelec, 3)
            rescale: Weight matrix for coordinate transformation (nelec, 3)
            reflection_ID: Reflection operation ID

        Returns:
            Reflected electron positions (nelec, 3)
        """
        # Transform to symmetrized coordinates
        r_elec_symm = r_electrons \
            + jnp.einsum('nk,en->ek', coord_shift, rescale)

        # Transform to standard frame
        r_elec_shifted = r_elec_symm - r_O
        r_elec_std = jnp.einsum('ij,ej->ei', Rmat.T, r_elec_shifted)

        # Apply symmetry operation in the standard frame
        r_elec_std = _apply_symmetry_operation(r_elec_std, reflection_ID)

        # Transform back to symmetrized frame
        r_elec_symm = jnp.einsum('ij,ej->ei', Rmat, r_elec_std) + r_O

        # Transform back to original coordinates
        r_elec_orig = r_elec_symm \
            - jnp.einsum('nk,en->ek', coord_shift, rescale)

        return r_elec_orig

    return run_electron_reflection


def water_dimer_reflection_electrons(nuc_crds: jax.Array) -> Callable:
    """
    Create electron reflection function for water dimer.

    Electrons are dynamically assigned to the closer water molecule
    based on their distance to the oxygen atoms.

    Args:
        nuc_crds: Nuclear coordinates [O1, H2, H3, O4, H5, H6]
        with shape (6, 3)

    Returns:
        Function that applies a symmetry operation to electron coordinates
    """
    r_O1, r_H2, r_H3 = nuc_crds[0], nuc_crds[1], nuc_crds[2]
    r_O4, r_H5, r_H6 = nuc_crds[3], nuc_crds[4], nuc_crds[5]

    # Symmetrize both water molecules
    r_H2_symm, r_H3_symm = _symmetrize_water(r_O1, r_H2, r_H3)
    r_H5_symm, r_H6_symm = _symmetrize_water(r_O4, r_H5, r_H6)

    nuc_symm_crds = jnp.stack([r_O1, r_H2_symm, r_H3_symm,
                              r_O4, r_H5_symm, r_H6_symm])

    # Build rotation matrices for both water molecules
    Rmat_wat1 = _build_water_rotation_matrix(r_O1, r_H2_symm, r_H3_symm)
    Rmat_wat2 = _build_water_rotation_matrix(r_O4, r_H5_symm, r_H6_symm)

    # Precompute coordinate shift
    coord_shift = nuc_symm_crds - nuc_crds

    @jax.jit
    def run_electron_reflection(r_electrons: jax.Array,
                                rescale: jax.Array,
                                reflection_ID: int) -> jax.Array:
        """
        Apply reflection to electrons with dynamic water assignment.

        Args:
            r_electrons: Electron positions (nelec, 3)
            rescale: Weight matrix for coordinate transformation (nelec, 6)
            reflection_ID: Reflection operation ID

        Returns:
            Reflected electron positions (nelec, 3)
        """
        # Transform to symmetrized coordinates
        r_elec_symm = r_electrons \
            + jnp.einsum('nk,en->ek', coord_shift, rescale)

        # Assign electrons to water molecules based on distance to oxygen
        dist_to_O1 = jnp.linalg.norm(r_elec_symm - r_O1, axis=-1)
        dist_to_O4 = jnp.linalg.norm(r_elec_symm - r_O4, axis=-1)
        belongs_to_wat1 = dist_to_O1 < dist_to_O4  # (nelec,)

        # Transform to standard frame for both water molecules
        r_elec_std_wat1 = jnp.einsum('ij,ej->ei',
                                     Rmat_wat1.T, r_elec_symm - r_O1)
        r_elec_std_wat2 = jnp.einsum('ij,ej->ei',
                                     Rmat_wat2.T, r_elec_symm - r_O4)

        # Apply symmetry operation to both
        r_elec_std_wat1 = _apply_symmetry_operation(
            r_elec_std_wat1, reflection_ID
        )
        r_elec_std_wat2 = _apply_symmetry_operation(
            r_elec_std_wat2, reflection_ID
        )

        # Transform back to symmetrized frame
        r_elec_symm_wat1 = jnp.einsum('ij,ej->ei',
                                      Rmat_wat1, r_elec_std_wat1) + r_O1
        r_elec_symm_wat2 = jnp.einsum('ij,ej->ei',
                                      Rmat_wat2, r_elec_std_wat2) + r_O4

        # Select based on water assignment
        r_elec_symm_combined = jnp.where(
            belongs_to_wat1[:, None],
            r_elec_symm_wat1,
            r_elec_symm_wat2
        )

        # Transform back to original coordinates
        r_elec_orig = r_elec_symm_combined \
            - jnp.einsum('nk,en->ek', coord_shift, rescale)

        return r_elec_orig

    return run_electron_reflection


def water_cluster_reflection_electrons(
        nuc_crds: jax.Array,
        cluster_idx: list
        ) -> Callable:
    """
    Create electron reflection function for water dimer.

    Electrons are dynamically assigned to the closer water molecule
    based on their distance to the oxygen atoms.

    Args:
        nuc_crds: Nuclear coordinates [O1, H2, H3, O4, H5, H6]
        with shape (6, 3)

    Returns:
        Function that applies a symmetry operation to electron coordinates
    """
    r_O_ls = []
    Rmat_wat_ls = []
    symmetrized_waters = []
    for idx in cluster_idx:
        r_O, H1, H2 = nuc_crds[jnp.array(idx)]
        # Symmetrize both water molecules
        symm_H = _symmetrize_water(r_O, H1, H2)
        # Build rotation matrices for both water molecules
        Rmat_wat = _build_water_rotation_matrix(r_O, symm_H[0], symm_H[1])
        # Stack results
        r_O_ls.append(r_O)
        Rmat_wat_ls.append(jnp.array(Rmat_wat))
        symmetrized_waters.append(jnp.array([r_O, symm_H[0], symm_H[1]]))
    nuc_symm_crds = jnp.concatenate(symmetrized_waters)

    # Precompute coordinate shift
    coord_shift = nuc_symm_crds - nuc_crds

    @jax.jit
    def run_electron_reflection(r_electrons: jax.Array,
                                rescale: jax.Array,
                                reflection_ID: int) -> jax.Array:
        """
        Apply reflection to electrons with dynamic water assignment.

        Args:
            r_electrons: Electron positions (nelec, 3)
            rescale: Weight matrix for coordinate transformation (nelec, 6)
            reflection_ID: Reflection operation ID

        Returns:
            Reflected electron positions (nelec, 3)
        """
        # Transform to symmetrized coordinates
        r_elec_symm = r_electrons \
            + jnp.einsum('nk,en->ek', coord_shift, rescale)

        # Assign electrons to water molecules based on distance to oxygen
        dist_list = [
            jnp.linalg.norm(r_elec_symm - r_O_ls[i], axis=-1)
            for i in range(len(r_O_ls))
        ]
        r_elec_symm_combined = jnp.zeros_like(r_electrons)
        for i in range(len(r_O_ls)):
            dist_to_Oi = dist_list[i]
            belongs_to_wat_i_all = jnp.array([dist_to_Oi < dist
                                              for dist in dist_list])
            row_mask = (jnp.arange(belongs_to_wat_i_all.shape[0]) != i)
            belongs_to_wat_i_all_except_ii = jnp.where(
                row_mask[:, None],
                belongs_to_wat_i_all,
                True
            )
            belongs_to_wat_i = jnp.all(belongs_to_wat_i_all_except_ii, axis=0)

            # Transform to standard frame for this water molecule
            r_elec_std_wat_i = jnp.einsum('ij,ej->ei',
                                          Rmat_wat_ls[i].T,
                                          r_elec_symm - r_O_ls[i])
            # Apply symmetry operation in the standard frame
            r_elec_std_wat_i = _apply_symmetry_operation(
                r_elec_std_wat_i, reflection_ID
            )
            # Transform back to symmetrized frame
            r_elec_symm_wat_i = jnp.einsum('ij,ej->ei',
                                           Rmat_wat_ls[i],
                                           r_elec_std_wat_i) + r_O_ls[i]
            r_elec_symm_combined \
                = jnp.where(belongs_to_wat_i[:, None],
                            r_elec_symm_wat_i, r_elec_symm_combined)

        # Transform back to original coordinates
        r_elec_orig = r_elec_symm_combined \
            - jnp.einsum('nk,en->ek', coord_shift, rescale)

        return r_elec_orig

    return run_electron_reflection


def reflect_xy_planar(elec_crds: jax.Array,
                      frag_nuc_crds: jax.Array,
                      frag_centroid: jax.Array,
                      inradius: float) -> jax.Array:
    """Reflect electrons through a planar fragment's molecular plane.

    Only electrons within `inradius` of `frag_centroid` are reflected;
    the rest are left unchanged.

    Algorithm:
        1. SVD of centered fragment nuclear coords → rotation matrix Vh
           where Vh[2] is the plane normal (smallest singular value direction)
        2. Translate electrons so centroid is at origin
        3. Rotate electrons into fragment principal frame (plane → xy)
        4. Apply z-reflection (flip z-coordinate)
        5. Inverse rotate and translate back

    Args:
        elec_crds: Electron coordinates (nelec, 3)
        frag_nuc_crds: Nuclear coordinates of fragment atoms (n_frag_atoms, 3)
        frag_centroid: Center of mass of the fragment (3,)
        inradius: Fragment inradius — only electrons within this distance
                  of frag_centroid are reflected

    Returns:
        Proposed electron coordinates (nelec, 3)
    """
    # 1. Compute rotation matrix from fragment geometry
    centered_nucs = frag_nuc_crds - frag_centroid
    _, _, Vh = jnp.linalg.svd(centered_nucs, full_matrices=True)
    # Vh rows: principal axes. Vh[2] = plane normal (least variance).
    # R = Vh rotates the fragment plane onto xy.

    # 2. Translate electrons to fragment-centered frame
    elec_centered = elec_crds - frag_centroid

    # 3. Rotate into fragment principal frame (plane → xy)
    elec_rotated = elec_centered @ Vh.T

    # 4. Reflect through xy-plane (z → -z)
    elec_reflected = apply_reflection_z(elec_rotated)

    # 5. Rotate back and translate
    elec_proposed = elec_reflected @ Vh + frag_centroid

    # 6. Only apply to electrons within inradius of centroid
    dist = jnp.linalg.norm(elec_centered, axis=-1)   # (nelec,)
    mask = dist <= inradius                            # (nelec,)
    return jnp.where(mask[:, None], elec_proposed, elec_crds)


if __name__ == "__main__":
    # Test the reflection functions
    print("=" * 70)
    print("Testing Electron Reflection Functions")
    print("=" * 70)

    # Test water molecule setup
    r_O = jnp.array([0.0, 0.0, 0.0])
    r_H1 = jnp.array([0.96, 0.0, 0.0])
    r_H2 = jnp.array([0.0, 0.99, 0.0])
    nuc_crds_water = jnp.stack([r_O, r_H1, r_H2])

    # Test water dimer setup
    r_O1 = jnp.array([0.0, 0.0, 0.0])
    r_H2 = jnp.array([0.96, 0.0, 0.0])
    r_H3 = jnp.array([0.0, 0.99, 0.0])
    r_O4 = jnp.array([0.0, 0.0, 3.0])
    r_H5 = jnp.array([0.96, 0.0, 3.0])
    r_H6 = jnp.array([0.0, 0.99, 3.0])
    nuc_crds_dimer = jnp.stack([r_O1, r_H2, r_H3, r_O4, r_H5, r_H6])

    def compute_rescale(elec_crds, nuc_crds):
        """Compute rescaling weights based on inverse distance."""
        diff = elec_crds[:, None, :] - nuc_crds[None, :, :]
        dist = jnp.sqrt(jnp.sum(diff**2, axis=-1))
        dist = jnp.where(dist < 1e-12, 1e-12, dist)
        weight = dist**(-4.0)
        return weight / jnp.sum(weight, axis=-1, keepdims=True)

    # Create test electrons for water
    key = jax.random.key(42)
    nelec_water = 10
    walkers_water = jax.random.normal(key, (nelec_water, 3)) * 0.5
    rescale_water = compute_rescale(walkers_water, nuc_crds_water)

    # Test water reflection
    print("\n1. Testing single water molecule reflection:")
    reflect_water = water_reflection_electrons(nuc_crds_water)

    for ref_name, ref_id in SYMM_OP_IDS.items():
        reflected = reflect_water(walkers_water, rescale_water, ref_id)
        print(f"   Reflection '{ref_name}': shape = {reflected.shape}")

    # Create test electrons for dimer
    key = jax.random.key(123)
    nelec_dimer = 20
    walkers_dimer = jax.random.normal(key, (nelec_dimer, 3)) * 0.5
    walkers_dimer = walkers_dimer.at[10:, 2].add(3.0)
    # Shift half to second water
    rescale_dimer = compute_rescale(walkers_dimer, nuc_crds_dimer)

    # Test water dimer reflection
    print("\n2. Testing water dimer reflection:")
    reflect_dimer = water_dimer_reflection_electrons(nuc_crds_dimer)

    for ref_name, ref_id in SYMM_OP_IDS.items():
        reflected = reflect_dimer(walkers_dimer, rescale_dimer, ref_id)
        print(f"   Reflection '{ref_name}': shape = {reflected.shape}")

    # Verify assignment preservation
    print("\n3. Verifying electron assignment preservation:")
    dist_before_O1 = jnp.linalg.norm(walkers_dimer - r_O1, axis=-1)
    dist_before_O4 = jnp.linalg.norm(walkers_dimer - r_O4, axis=-1)
    assigned_before = dist_before_O1 < dist_before_O4

    reflected_y = reflect_dimer(walkers_dimer,
                                rescale_dimer, SYMM_OP_IDS['y'])
    dist_after_O1 = jnp.linalg.norm(reflected_y - r_O1, axis=-1)
    dist_after_O4 = jnp.linalg.norm(reflected_y - r_O4, axis=-1)
    assigned_after = dist_after_O1 < dist_after_O4

    preserved = jnp.all(assigned_before == assigned_after)
    n_wat1 = assigned_before.sum()
    print(f"   Electrons in Water 1: {n_wat1}, "
          f"Water 2: {nelec_dimer - n_wat1}")
    print(f"   Assignment preserved: {preserved}")

    # JIT compilation test
    print("\n4. JIT compilation test:")
    reflect_water_jit = jax.jit(reflect_water)
    reflect_dimer_jit = jax.jit(reflect_dimer)

    # Warm up
    _ = reflect_water_jit(walkers_water, rescale_water, 1)
    _ = reflect_dimer_jit(walkers_dimer, rescale_dimer, 1)

    print("   JIT compilation successful!")
    print("\n" + "=" * 70)
