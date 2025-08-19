import jax.numpy as jnp
import jax
import h5py
from functools import partial
from vmc_mlsw.diatomic_rotation_matrix import rotate_vector_to_z_axis
from vmc_mlsw.water_rotation_matrix import symmetrize_water_molecule

@jax.jit
def redistribute_weight(e_crds, n_crds, sigma=0.5):
    """
    Compute redistribution weights based on distances to nuclei.
    
    Args:
        e_crds: electron coordinates (nelec, 3)
        n_crds: nuclear coordinates (nnuc, 3)
        sigma: decay parameter for exponential weighting
    
    Returns:
        weights: normalized weights (nelec, nnuc)
    """
    # Vectorized distance calculation
    distances = jnp.linalg.norm(e_crds[:, None, :] - n_crds[None, :, :], axis=-1)
    # Avoid division by zero and numerical instability
    distances = jnp.where(distances < 1e-12, 1e-12, distances)
    weights = jnp.exp(-distances / sigma)
    return weights / jnp.sum(weights, axis=-1, keepdims=True)



@jax.jit
def apply_coordinate_transformation(coords, center, rot_mat):
    """
    Apply translation and rotation to coordinates.
    
    Args:
        coords: coordinates to transform (..., 3)
        center: translation center (3,)
        rot_mat: rotation matrix (3, 3)
    
    Returns:
        transformed coordinates
    """
    # Translate to center
    coords_trans = coords - center
    # Rotate
    return jnp.einsum('...i,ij->...j', coords_trans, rot_mat)


@jax.jit
def apply_inverse_transformation(coords, center, rot_mat):
    """
    Apply inverse rotation and translation to coordinates.
    
    Args:
        coords: coordinates to transform (..., 3)
        center: original translation center (3,)
        rot_mat: rotation matrix (3, 3)
    
    Returns:
        transformed coordinates back to original frame
    """
    # Inverse rotate (transpose of rotation matrix)
    coords_rot_inv = jnp.einsum('...j,ij->...i', coords, rot_mat)
    # Translate back
    return coords_rot_inv + center


@jax.jit
def apply_space_warping(elc_coords, nuc_orig, nuc_sym, weights):
    """
    Apply space warping transformation to electron coordinates.
    
    Args:
        elc_coords: electron coordinates (nelec, 3)
        nuc_orig: original nuclear coordinates (nnuc, 3)
        nuc_sym: symmetrized nuclear coordinates (nnuc, 3)
        weights: redistribution weights (nelec, nnuc)
    
    Returns:
        warped electron coordinates
    """
    shift_nuc = nuc_sym - nuc_orig
    return elc_coords + jnp.einsum('nk,en->ek', shift_nuc, weights)


@jax.jit
def apply_reflection_x(coords):
    """Apply reflection across yz-plane (negate x-coordinate)."""
    return coords.at[..., 0].multiply(-1)


@jax.jit
def apply_reflection_y(coords):
    """Apply reflection across xz-plane (negate y-coordinate)."""
    return coords.at[..., 1].multiply(-1)


@jax.jit
def apply_reflection_xy(coords):
    """Apply reflection across yz-plane and xz-plane (negate x,y-coordinate)."""
    coords = coords.at[..., 0].multiply(-1) 
    coords = coords.at[..., 1].multiply(-1)
    return coords


def process_symmetric_water_molecule(chkfile_mc,
                                     chkfile_elc,
                                     sigma=0.5, 
                                     reflection_ops=None):
    """
    Complete processing pipeline for symmetric operations on water molecule.
    
    Args:
        chkfile_mc: h5py file which provides nuc_crds/elc_samples
        sigma: decay parameter for weight redistribution
        reflection_ops: list of reflection operations to apply
        
    Returns:
        dict containing all transformed coordinates and intermediate results
    """
    with h5py.File(chkfile_mc, 'r') as f:
        #nuc_crds: original nuclear coordinates [O, H1, H2] (3, 3)
        #elc_samples: electron coordinate samples (nsamples, nelec, 3)
        elc_samples = jnp.array(f['stacked_samples'][:])
        nuc_crds = jnp.array(f['nuc_crds'][:])

    if reflection_ops is None:
        reflection_ops = ['x']  # Default: reflection across yz-plane
    
    O = nuc_crds[0]  # Oxygen position for centering

    #dist = jnp.linalg.norm (elc_samples[0]-O, axis=-1)
    #print ('dist_ref', dist)
    # Step 1: Symmetrize the water molecule
    nuc_crds_sym, rot_mat = symmetrize_water_molecule(nuc_crds)
    
    print(f"Original bond lengths: {jnp.linalg.norm(nuc_crds[1] - nuc_crds[0]):.6f}, "
          f"{jnp.linalg.norm(nuc_crds[2] - nuc_crds[0]):.6f}")
    print(f"Symmetrized bond lengths: {jnp.linalg.norm(nuc_crds_sym[1] - nuc_crds_sym[0]):.6f}, "
          f"{jnp.linalg.norm(nuc_crds_sym[2] - nuc_crds_sym[0]):.6f}")
    
    # Step 2: Transform coordinates to symmetric frame
    elc_samples_trans_rot = apply_coordinate_transformation(elc_samples, O, rot_mat)
    nuc_crds_trans_rot = apply_coordinate_transformation(nuc_crds, O, rot_mat)
    nuc_crds_sym_trans_rot = apply_coordinate_transformation(nuc_crds_sym, O, rot_mat)
    
    #debug
    #dist = jnp.linalg.norm (elc_samples_trans_rot[0], axis=-1)
    #print ('dist_trans_rot', dist)

    # Step 3: Compute redistribution weights
    # Using vmap for efficient batch processing
    weights = jax.vmap(redistribute_weight, in_axes=(0, None, None))(
        elc_samples_trans_rot, nuc_crds_trans_rot, sigma
    )
    
    # Step 4: Apply space warping to symmetrize electron distribution
    elc_samples_sym_trans_rot = jax.vmap(
        apply_space_warping, in_axes=(0, None, None, 0)
    )(elc_samples_trans_rot, nuc_crds_trans_rot, nuc_crds_sym_trans_rot, weights)
    
    #dist = jnp.linalg.norm (elc_samples_sym_trans_rot[0], axis=-1)
    #print ('dist_space_warping', dist)

    # Step 5: Apply reflection operations
    reflection_map = {
        'x': apply_reflection_x,
        'y': apply_reflection_y, 
        'xy': apply_reflection_xy
    }
    
    results = {}
    results[f'reflection_E'] = elc_samples 
    
    for reflection_op in reflection_ops:
        if reflection_op not in reflection_map:
            print(f"Warning: Unknown reflection operation '{reflection_op}'. Skipping.")
            continue
        
        # Apply reflection
        elc_samples_reflected = reflection_map[reflection_op](elc_samples_sym_trans_rot)
        nuc_crds_sym_trans_rot_reflected = reflection_map[reflection_op](nuc_crds_sym_trans_rot)
        nuc_crds_trans_rot_reflected = reflection_map[reflection_op](nuc_crds_trans_rot)
        
        # Step 6: Transform back to original frame
        # First, reverse the space warping
        elc_samples_unwarped = jax.vmap(
            lambda coords, weights_i: coords - jnp.einsum('nk,en->ek', 
                                                        nuc_crds_sym_trans_rot_reflected - nuc_crds_trans_rot_reflected, 
                                                        weights_i)
        )(elc_samples_reflected, weights)
        
        #dist = jnp.linalg.norm (elc_samples_unwarped[0], axis=-1)
        #print ('dist_unwarped', reflection_op, dist)
        # Then apply inverse coordinate transformation
        elc_samples_final = apply_inverse_transformation(elc_samples_unwarped, O, rot_mat)
        
        # Store results
        results[f'reflection_{reflection_op}'] = elc_samples_final

    with h5py.File(chkfile_elc, 'w') as f:
        for key, elc_samples in results.items():
            f.create_dataset (key, data=elc_samples)

    


def process_symmetric_diatomic_molecule(chkfile_mc,
                                        chkfile_elc,
                                        reflection_ops=None):
    """
    Complete processing pipeline for symmetric operations on diatomic molecule.
    
    Args:
        chkfile_mc: h5py file which provides nuc_crds/elc_samples
        reflection_ops: list of reflection operations to apply
        
    Returns:
        dict containing all transformed coordinates and intermediate results
    """
    with h5py.File(chkfile_mc, 'r') as f:
        elc_samples = jnp.array(f['stacked_samples'][:])
        nuc_crds = jnp.array(f['nuc_crds'][:])

    if reflection_ops is None:
        reflection_ops = ['x']  # Default: reflection across yz-plane
    
    center = nuc_crds[0]  # A position for centering

    # Step 1: Symmetrize the water molecule
    vector = nuc_crds[1] - center
    rot_mat = rotate_vector_to_z_axis(vector)
    rot_mat = rot_mat.T 

    # Step 2: Transform coordinates to symmetric frame
    elc_samples_trans_rot = apply_coordinate_transformation(elc_samples, center, rot_mat)
    
    # Step 5: Apply reflection operations
    reflection_map = {
        'x': apply_reflection_x,
        'y': apply_reflection_y, 
        'xy': apply_reflection_xy
    }
    
    results = {}
    results[f'reflection_E'] = elc_samples 
    
    for reflection_op in reflection_ops:
        if reflection_op not in reflection_map:
            print(f"Warning: Unknown reflection operation '{reflection_op}'. Skipping.")
            continue
        
        # Apply reflection
        elc_samples_reflected = reflection_map[reflection_op](elc_samples_trans_rot)
        
        # Then apply inverse coordinate transformation
        elc_samples_final = apply_inverse_transformation(elc_samples_reflected, center, rot_mat)
        
        # Store results
        results[f'reflection_{reflection_op}'] = elc_samples_final
    
    with h5py.File(chkfile_elc, 'w') as f:
        for key, elc_samples in results.items():
            f.create_dataset (key, data=elc_samples)
    


if __name__ == "__main__":
    from pyscf import gto, scf
    from vmc_mlsw import get_vmc_func 
    import h5py
    
    mol = gto.M(
              atom='''
O        1.000000    0.000000    0.117307
H        1.000000    0.757216   -0.469229
H        1.000000   -0.807216   -0.469229
''',
              basis='6-31g*',
              # basis='cc-pvdz',
              unit='Ang'
          )
    mol.build()
    mf = scf.RHF(mol)
    mf.kernel()
    mf_grad = mf.nuc_grad_method()
    grad = mf_grad.kernel()

    nuc_crds = mol.atom_coords(unit='Bohr')
    rng_key = jax.random.key(777)
    # No optimizable Jastrow parameters:
    params_vmc_no_jastrow = jnp.array([])

    chkfile_mc =  'H2O_vmc_631gd_mc.hdf5'
    chkfile_enr = 'H2O_vmc_631gd_enr.hdf5'
    chkfile_grd = 'H2O_vmc_631gd_grd.hdf5'

    cgto_coeff = {
        1: jnp.array ([1, 1.0431879, -0.02914878, 0.78355617,
        -2.95081286, 5.43507108, -5.08491324, 1.94265234]),
        8: jnp.array ([1, 12.45593615, -2.38348643, 30.46159315,
        -125.8242091, 252.61904634, -239.5024989, 86.70950789])
    }
    
    vmc_run, vmc_energy, vmc_gradient_prep, vmc_grad =\
        get_vmc_func (mf, 
                      params_vmc_no_jastrow,
                      chkfile_mc=chkfile_mc,
                      chkfile_enr=chkfile_enr,
                      chkfile_grd=chkfile_grd,
                      cgto_coeff=None)
    
    
    vmc_run (rng_key, 
             num_steps=100000,
             num_equilibration=500000,
             step_size=0.05)
    
    vmc_energy () 

    with h5py.File(chkfile_mc, 'r') as f:
        sampled_elc_crds = jnp.array(f['stacked_samples'][:])

    results = process_symmetric_water_molecule(
        nuc_crds, sampled_elc_crds, 
        sigma=0.5,
        reflection_ops=['x','y', 'xy']
    )
    
