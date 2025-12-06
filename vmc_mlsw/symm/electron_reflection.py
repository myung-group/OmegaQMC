"""
Electron reflection operations for VMC symmetry exploitation.

This module provides reflection operations for electrons in molecular systems,
used to improve sampling efficiency in Variational Monte Carlo calculations.
"""

import jax
import jax.numpy as jnp
from typing import Callable, List

# Reflection ID mapping
REFLECTION_IDS = {'I': 0, 'x': 1, 'y': 2, 'xy': 3}

# Pre-computed reflection matrices for efficiency
# Each matrix applies the corresponding reflection in the xy-plane
_REFLECTION_MATRICES = jnp.array([
    [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]],   # Identity
    [[-1., 0., 0.], [0., 1., 0.], [0., 0., 1.]],  # Reflect x
    [[1., 0., 0.], [0., -1., 0.], [0., 0., 1.]],  # Reflect y
    [[-1., 0., 0.], [0., -1., 0.], [0., 0., 1.]], # Reflect xy
])


def _apply_reflection(r_electrons: jax.Array, reflection_ID: int) -> jax.Array:
    """
    Apply reflection operation to electron coordinates.

    Args:
        r_electrons: Electron positions with shape (nelec, 3)
        reflection_ID: Integer ID (0=identity, 1=x, 2=y, 3=xy)

    Returns:
        Reflected electron positions with shape (nelec, 3)
    """
    # Use dynamic indexing to select the appropriate reflection matrix
    ref_matrix = jax.lax.dynamic_index_in_dim(
        _REFLECTION_MATRICES, reflection_ID, axis=0, keepdims=False
    )
    return jnp.einsum('ij,ej->ei', ref_matrix, r_electrons)


def _symmetrize_water(r_O: jax.Array, r_H1: jax.Array, r_H2: jax.Array):
    """
    Symmetrize a water molecule by averaging OH bond lengths.

    Args:
        r_O: Oxygen position (3,)
        r_H1: First hydrogen position (3,)
        r_H2: Second hydrogen position (3,)

    Returns:
        Tuple of (r_H1_sym, r_H2_sym) with equalized bond lengths
    """
    v_OH1 = r_H1 - r_O
    v_OH2 = r_H2 - r_O
    r_OH1 = jnp.linalg.norm(v_OH1)
    r_OH2 = jnp.linalg.norm(v_OH2)
    r_avg = 0.5 * (r_OH1 + r_OH2)

    r_H1_sym = r_O + r_avg * v_OH1 / r_OH1
    r_H2_sym = r_O + r_avg * v_OH2 / r_OH2

    return r_H1_sym, r_H2_sym


def _build_water_rotation_matrix(r_O: jax.Array,
                                  r_H1_sym: jax.Array,
                                  r_H2_sym: jax.Array) -> jax.Array:
    """
    Build rotation matrix for transforming to water's standard frame.

    The standard frame has:
    - z-axis along the bisector of the H-O-H angle
    - x-axis perpendicular to the molecular plane
    - y-axis completing the right-handed system

    Args:
        r_O: Oxygen position (3,)
        r_H1_sym: First symmetrized hydrogen position (3,)
        r_H2_sym: Second symmetrized hydrogen position (3,)

    Returns:
        Rotation matrix (3, 3)
    """
    r_H1_shifted = r_H1_sym - r_O
    r_H2_shifted = r_H2_sym - r_O

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
        Function that applies reflection to electron coordinates
    """
    def run_electron_reflection(r_electrons: jax.Array,
                                rescale: jax.Array,
                                reflection_ID: int) -> jax.Array:
        return _apply_reflection(r_electrons, reflection_ID)

    return run_electron_reflection


def water_reflection_electrons(nuc_crds: jax.Array) -> Callable:
    """
    Create electron reflection function for a single water molecule.

    Args:
        nuc_crds: Nuclear coordinates [O, H1, H2] with shape (3, 3)

    Returns:
        Function that applies reflection to electron coordinates
    """
    r_O, r_H1, r_H2 = nuc_crds[0], nuc_crds[1], nuc_crds[2]

    # Symmetrize water molecule
    r_H1_sym, r_H2_sym = _symmetrize_water(r_O, r_H1, r_H2)
    nuc_sym_crds = jnp.stack([r_O, r_H1_sym, r_H2_sym])

    # Build rotation matrix
    Rmat = _build_water_rotation_matrix(r_O, r_H1_sym, r_H2_sym)

    # Precompute coordinate shift
    coord_shift = nuc_sym_crds - nuc_crds

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
        r_elec_sym = r_electrons + jnp.einsum('nk,en->ek', coord_shift, rescale)

        # Transform to standard frame
        r_elec_shifted = r_elec_sym - r_O
        r_elec_std = jnp.einsum('ij,ej->ei', Rmat.T, r_elec_shifted)

        # Apply reflection
        r_elec_std = _apply_reflection(r_elec_std, reflection_ID)

        # Transform back to symmetrized frame
        r_elec_sym = jnp.einsum('ij,ej->ei', Rmat, r_elec_std) + r_O

        # Transform back to original coordinates
        r_elec_orig = r_elec_sym - jnp.einsum('nk,en->ek', coord_shift, rescale)

        return r_elec_orig

    return run_electron_reflection


def water_dimer_reflection_electrons(nuc_crds: jax.Array) -> Callable:
    """
    Create electron reflection function for water dimer.

    Electrons are dynamically assigned to the closer water molecule
    based on their distance to the oxygen atoms.

    Args:
        nuc_crds: Nuclear coordinates [O1, H2, H3, O4, H5, H6] with shape (6, 3)

    Returns:
        Function that applies reflection to electron coordinates
    """
    r_O1, r_H2, r_H3 = nuc_crds[0], nuc_crds[1], nuc_crds[2]
    r_O4, r_H5, r_H6 = nuc_crds[3], nuc_crds[4], nuc_crds[5]

    # Symmetrize both water molecules
    r_H2_sym, r_H3_sym = _symmetrize_water(r_O1, r_H2, r_H3)
    r_H5_sym, r_H6_sym = _symmetrize_water(r_O4, r_H5, r_H6)

    nuc_sym_crds = jnp.stack([r_O1, r_H2_sym, r_H3_sym, r_O4, r_H5_sym, r_H6_sym])

    # Build rotation matrices for both water molecules
    Rmat_wat1 = _build_water_rotation_matrix(r_O1, r_H2_sym, r_H3_sym)
    Rmat_wat2 = _build_water_rotation_matrix(r_O4, r_H5_sym, r_H6_sym)

    # Precompute coordinate shift
    coord_shift = nuc_sym_crds - nuc_crds

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
        r_elec_sym = r_electrons + jnp.einsum('nk,en->ek', coord_shift, rescale)

        # Assign electrons to water molecules based on distance to oxygen
        dist_to_O1 = jnp.linalg.norm(r_elec_sym - r_O1, axis=-1)
        dist_to_O4 = jnp.linalg.norm(r_elec_sym - r_O4, axis=-1)
        belongs_to_wat1 = dist_to_O1 < dist_to_O4  # (nelec,)

        # Transform to standard frame for both water molecules
        r_elec_std_wat1 = jnp.einsum('ij,ej->ei', Rmat_wat1.T, r_elec_sym - r_O1)
        r_elec_std_wat2 = jnp.einsum('ij,ej->ei', Rmat_wat2.T, r_elec_sym - r_O4)

        # Apply reflection to both
        r_elec_std_wat1 = _apply_reflection(r_elec_std_wat1, reflection_ID)
        r_elec_std_wat2 = _apply_reflection(r_elec_std_wat2, reflection_ID)

        # Transform back to symmetrized frame
        r_elec_sym_wat1 = jnp.einsum('ij,ej->ei', Rmat_wat1, r_elec_std_wat1) + r_O1
        r_elec_sym_wat2 = jnp.einsum('ij,ej->ei', Rmat_wat2, r_elec_std_wat2) + r_O4

        # Select based on water assignment
        r_elec_sym_combined = jnp.where(
            belongs_to_wat1[:, None],
            r_elec_sym_wat1,
            r_elec_sym_wat2
        )

        # Transform back to original coordinates
        r_elec_orig = r_elec_sym_combined - jnp.einsum('nk,en->ek', coord_shift, rescale)

        return r_elec_orig

    return run_electron_reflection


def water_cluster_reflection_electrons(
        nuc_crds: jax.Array, 
        cluster_idx: List
    ) -> Callable:
    """
    Create electron reflection function for water dimer.

    Electrons are dynamically assigned to the closer water molecule
    based on their distance to the oxygen atoms.

    Args:
        nuc_crds: Nuclear coordinates [O1, H2, H3, O4, H5, H6] with shape (6, 3)

    Returns:
        Function that applies reflection to electron coordinates
    """
    r_O_ls = []
    Rmat_wat_ls = []
    symmetrized_waters = []
    for idx in cluster_idx:
        r_O, H1, H2 = nuc_crds[jnp.array(idx)]
        # Symmetrize both water molecules
        sym_H = _symmetrize_water(r_O, H1, H2)
        # Build rotation matrices for both water molecules
        Rmat_wat = _build_water_rotation_matrix(r_O, sym_H[0], sym_H[1])
        # Stack results
        r_O_ls.append(r_O)
        Rmat_wat_ls.append(jnp.array(Rmat_wat))
        symmetrized_waters.append(jnp.array([r_O, sym_H[0], sym_H[1]]))
    nuc_sym_crds = jnp.concatenate(symmetrized_waters)

    # Precompute coordinate shift
    coord_shift = nuc_sym_crds - nuc_crds

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
        r_elec_sym = r_electrons + jnp.einsum('nk,en->ek', coord_shift, rescale)

        # Assign electrons to water molecules based on distance to oxygen
        dist_list = [
            jnp.linalg.norm(r_elec_sym - r_O_ls[i], axis=-1)
                for i in range(len(r_O_ls))
        ]       
        r_elec_sym_combined = jnp.zeros_like(r_electrons)
        for i in range(len(r_O_ls)):
            dist_to_Oi = dist_list[i]
            belongs_to_wat_i_all = jnp.array([dist_to_Oi < dist for dist in dist_list])
            row_mask = (jnp.arange(belongs_to_wat_i_all.shape[0]) != i)
            belongs_to_wat_i_all_except_ii = jnp.where(
                row_mask[:, None], 
                belongs_to_wat_i_all,         
                True              
            )
            belongs_to_wat_i = jnp.all(belongs_to_wat_i_all_except_ii, axis=0)

            # Transform to standard frame for both water molecules
            r_elec_std_wat_i = jnp.einsum('ij,ej->ei', Rmat_wat_ls[i].T, r_elec_sym - r_O_ls[i])
            # Apply reflection to both
            r_elec_std_wat_i = _apply_reflection(r_elec_std_wat_i, reflection_ID)
            # Transform back to symmetrized frame
            r_elec_sym_wat_i = jnp.einsum('ij,ej->ei', Rmat_wat_ls[i], r_elec_std_wat_i) + r_O_ls[i]
            r_elec_sym_combined = jnp.where(belongs_to_wat_i[:, None], r_elec_sym_wat_i, r_elec_sym_combined)

        # Transform back to original coordinates
        r_elec_orig = r_elec_sym_combined - jnp.einsum('nk,en->ek', coord_shift, rescale)

        return r_elec_orig

    return run_electron_reflection


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

    for ref_name, ref_id in REFLECTION_IDS.items():
        reflected = reflect_water(walkers_water, rescale_water, ref_id)
        print(f"   Reflection '{ref_name}': shape = {reflected.shape}")

    # Create test electrons for dimer
    key = jax.random.key(123)
    nelec_dimer = 20
    walkers_dimer = jax.random.normal(key, (nelec_dimer, 3)) * 0.5
    walkers_dimer = walkers_dimer.at[10:, 2].add(3.0)  # Shift half to second water
    rescale_dimer = compute_rescale(walkers_dimer, nuc_crds_dimer)

    # Test water dimer reflection
    print("\n2. Testing water dimer reflection:")
    reflect_dimer = water_dimer_reflection_electrons(nuc_crds_dimer)

    for ref_name, ref_id in REFLECTION_IDS.items():
        reflected = reflect_dimer(walkers_dimer, rescale_dimer, ref_id)
        print(f"   Reflection '{ref_name}': shape = {reflected.shape}")

    # Verify assignment preservation
    print("\n3. Verifying electron assignment preservation:")
    dist_before_O1 = jnp.linalg.norm(walkers_dimer - r_O1, axis=-1)
    dist_before_O4 = jnp.linalg.norm(walkers_dimer - r_O4, axis=-1)
    assigned_before = dist_before_O1 < dist_before_O4

    reflected_y = reflect_dimer(walkers_dimer, rescale_dimer, REFLECTION_IDS['y'])
    dist_after_O1 = jnp.linalg.norm(reflected_y - r_O1, axis=-1)
    dist_after_O4 = jnp.linalg.norm(reflected_y - r_O4, axis=-1)
    assigned_after = dist_after_O1 < dist_after_O4

    preserved = jnp.all(assigned_before == assigned_after)
    n_wat1 = assigned_before.sum()
    print(f"   Electrons in Water 1: {n_wat1}, Water 2: {nelec_dimer - n_wat1}")
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
