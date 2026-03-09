import jax
import jax.numpy as jnp
from pyscf import gto
import numpy as np  # For generating quadrature points
# from scipy.linalg import eig as sci_eig
from functools import partial
import optax
import pickle
import os

CUSP_PARAMS_FILE = os.path.join(os.path.dirname(__file__), 'cusp_params.pkl')

# Ground state spins (2S) for elements 1-36
# Based on ground state electronic configurations
GROUND_STATE_SPINS_2S = {
    "H": 1, "He": 0,
    "Li": 1, "Be": 0, "B": 1, "C": 2, "N": 3, "O": 2, "F": 1, "Ne": 0,
    "Na": 1, "Mg": 0, "Al": 1, "Si": 2, "P": 3, "S": 2, "Cl": 1, "Ar": 0,
    "K": 1, "Ca": 0, "Sc": 1, "Ti": 2, "V": 3, "Cr": 6, "Mn": 5, "Fe": 4,
    "Co": 3, "Ni": 2, "Cu": 1, "Zn": 0, "Ga": 1, "Ge": 2, "As": 3, "Se": 2,
    "Br": 1, "Kr": 0
}


def get_cusp_params(elem, basis_name):
    """
    Generates or retrieves from cache the cusp correction data
    for a given element and basis set.

    Args:
        elem (str): The element, e.g., H, He, Li, ...
        basis_name (str): The name of the basis set, e.g., '6-31g'.

    Returns:
        dict: A dictionary containing the 'q0' and 'coeff' for the given atom.
    """
    assert isinstance(basis_name, str)

    # Create a key for the cache
    mol_temp = gto.M(atom=f"{elem} 0 0 0",
                     basis=basis_name,
                     spin=GROUND_STATE_SPINS_2S[elem])
    key = (mol_temp.atom_symbol(0), basis_name)
    # If we're going to test and switch between different cusping schemes,
    # we would introduce a third element (label) in key above.
    # But I'm keeping it fixed to CGAOWS for now... --DCY

    # Load cache if it exists
    if os.path.exists(CUSP_PARAMS_FILE):
        with open(CUSP_PARAMS_FILE, 'rb') as f:
            cusp_data_cache = pickle.load(f)
    else:
        cusp_data_cache = {}

    # Check if data is in cache
    if key in cusp_data_cache:
        print(f"Loading cached cusp data for {key}")
        return {key[0]: cusp_data_cache[key]}

    print(f"Generating cusp data for {key}")
    # If not in cache, generate it
    mol = gto.M(
        atom=f"{elem} 0 0 0",
        basis=basis_name,
        unit='Bohr',
        spin=GROUND_STATE_SPINS_2S[elem]
    )
    mol.build()

    symb = mol._atom[0][0]
    Z = mol.atom_charges()[0]
    rc = 0.1 if Z == 1 else 0.2
    basis = mol._basis[symb]
    # Find the s-shell (l=0) with the most primitives
    # — this is the contracted 1s
    s_shells = [sh for sh in basis if sh[0] == 0]
    basis_1s = max(s_shells, key=lambda sh: len(sh) - 1)

    g_alpha, g_norm = gto_1s_alpha_norm([basis_1s])
    coeff_vec = cusp_coeff_vec(g_alpha, g_norm, Z, rc)
    q0_min = minimize_q0(g_alpha, g_norm, Z, rc, coeff_vec)

    data = {
        'q0': q0_min.tolist()[0],
        'coeff': coeff_vec.tolist()
    }

    # Save to cache
    cusp_data_cache[key] = data
    with open(CUSP_PARAMS_FILE, 'wb') as f:
        pickle.dump(cusp_data_cache, f)

    return {key[0]: data}


# Helper function to generate Gauss-Legendre quadrature points and weights
def get_gauss_legendre(n):
    nodes, weights = np.polynomial.legendre.leggauss(n)
    return jnp.array(nodes), jnp.array(weights)


# JIT-compiled function for evaluating b function
@jax.jit
def evaluate_b_func(r, rc):
    s = r / rc
    s2 = s * s
    s3 = s2 * s
    s4 = s2 * s2
    s5 = s3 * s2
    return jnp.where(r < rc,
                     1.0 - 10.0 * s3 + 15.0 * s4 - 6.0 * s5,
                     0.0)


# JIT-compiled Slater function
@jax.jit
def evaluate_slater_func(r, q0, Z):
    return q0 * jnp.exp(-Z * r)


@partial(jax.jit, static_argnames=['Z', 'rc'])
def radial_basis_one_minus_b_X(r, g_alpha, g_norm, Z, rc):
    r2 = r * r
    b = evaluate_b_func(r, rc)
    return (1 - b) * jnp.sum(jnp.exp(-g_alpha * r2) * g_norm)


@partial(jax.jit, static_argnames=['Z', 'rc', 'order'])
def radial_basis_b_Q(r, g_alpha, g_norm, Z, rc, order):
    # g_alpha and g_norm are unused but kept for consistent signature if needed
    s = evaluate_slater_func(r, 1.0, Z)
    b = evaluate_b_func(r, rc)
    return b * s * (r/rc)**order


# Full basis function wrapper
@partial(jax.jit, static_argnames=['Z', 'rc', 'radial_func', 'order'])
def basis_func_wrapper(e_pos, g_alpha, g_norm, Z, rc, radial_func, order=None):
    r = jnp.sqrt(jnp.sum(e_pos * e_pos, axis=-1))
    if order is not None:
        return radial_func(r, g_alpha, g_norm, Z, rc, order=order)
    return radial_func(r, g_alpha, g_norm, Z, rc)


# JIT-compiled integrand for overlap (S) matrix
@partial(jax.jit, static_argnames=['bra_func', 'ket_func', 'Z', 'rc',
                                   'bra_order', 'ket_order'])
def integrand_at_rtp(r, theta, phi, bra_func, ket_func,
                     g_alpha, g_norm, Z, rc,
                     bra_order=None, ket_order=None):
    sint = jnp.sin(theta)
    r2 = r * r
    r2sint = r2 * sint

    xoc = r * sint * jnp.cos(phi)
    yoc = r * sint * jnp.sin(phi)
    zoc = r * jnp.cos(theta)
    e_pos = jnp.stack([xoc, yoc, zoc], axis=-1)

    # Vmap to evaluate the function for all points
    def bra_map(e):
        return basis_func_wrapper(e, g_alpha, g_norm, Z, rc,
                                  bra_func, bra_order)

    def ket_map(e):
        return basis_func_wrapper(e, g_alpha, g_norm, Z, rc,
                                  ket_func, ket_order)

    bra = jax.vmap(bra_map)(e_pos)
    ket = jax.vmap(ket_map)(e_pos)

    return bra * ket * r2sint


# JIT-compiled integrand for Hamiltonian (H) matrix
@partial(jax.jit, static_argnames=['bra_func', 'ket_func', 'Z', 'rc',
                                   'bra_order', 'ket_order'])
def integrand_at_rtp_H(r, theta, phi, bra_func, ket_func,
                       g_alpha, g_norm, Z, rc,
                       bra_order=None, ket_order=None):
    # dV = r*r sin(theta) d_r d_theta d_phi
    sint = jnp.sin(theta)
    r2 = r * r
    r2sint = r2 * sint

    # 1. Evaluate r and bra/ket values at (r, theta, phi)
    r_val = r

    # We define a function of r only, and then vmap on r_val
    def bra_rad(r_in):
        return bra_func(r_in, g_alpha, g_norm, Z, rc, order=bra_order) \
            if bra_order is not None \
            else bra_func(r_in, g_alpha, g_norm, Z, rc)

    def ket_rad(r_in):
        return ket_func(r_in, g_alpha, g_norm, Z, rc, order=ket_order) \
            if ket_order is not None \
            else ket_func(r_in, g_alpha, g_norm, Z, rc)

    bra = jax.vmap(bra_rad)(r_val)
    ket = jax.vmap(ket_rad)(r_val)

    # 2. Compute Kinetic Energy (T) contribution using 1D Laplacian formula:
    # T_ket = -0.5 * Nabla^2_ket = -0.5 * (d^2/dr^2 + 2/r * d/dr) ket

    # JAX grad of the radial function (ket_rad(r))
    dket_dr = jax.vmap(jax.grad(ket_rad))(r_val)
    d2ket_dr2 = jax.vmap(jax.grad(jax.grad(ket_rad)))(r_val)

    # 1D Laplacian on a spherically symmetric function
    # Handle r=0 singularity by using jnp.where
    laplacian_ket = d2ket_dr2 \
        + jnp.where(r_val > 1e-10, 2.0 / r_val * dket_dr, 0.0)

    # Kinetic energy integrand part: bra * (-0.5 * Nabla^2_ket) * dV
    retval_kE = bra * (-0.5 * laplacian_ket) * r2sint

    # 3. Compute Electron-Nuclear Potential Energy (V_eN)
    # V_eN = bra * (-Z/r) * ket * dV
    # Handle r=0 singularity in V_eN with a small epsilon
    potential_term = -Z / jnp.where(r_val > 1e-10, r_val, 1e-10)
    retval_eN = bra * potential_term * ket * r2sint

    return retval_kE + retval_eN


# JIT-compiled integration function using Gauss-Legendre quadrature
@partial(jax.jit, static_argnames=['bra_func', 'ket_func', 'Z', 'rc',
                                   'n_points', 'bra_order', 'ket_order'])
def test_integrand_at_rtp(bra_func, ket_func,
                          g_alpha, g_norm, Z, rc, n_points=32,
                          bra_order=None, ket_order=None):
    # Gauss-Legendre quadrature points and weights
    r_nodes, w_r = get_gauss_legendre(n_points)
    theta_nodes, w_theta = get_gauss_legendre(n_points)
    phi_nodes, w_phi = get_gauss_legendre(n_points)

    rgrid = rc * 0.5 * (r_nodes + 1.0)
    w_r = rc * 0.5 * w_r

    theta = 0.5 * (theta_nodes + 1.0) * jnp.pi
    w_theta = 0.5 * jnp.pi * w_theta

    phi = 0.5 * (phi_nodes + 1.0) * 2.0 * jnp.pi
    w_phi = 0.5 * 2.0 * jnp.pi * w_phi

    # Create 3D grid of quadrature points
    r_grid, theta_grid, phi_grid \
        = jnp.meshgrid(rgrid, theta, phi, indexing='ij')
    w_r_grid, w_theta_grid, w_phi_grid \
        = jnp.meshgrid(w_r, w_theta, w_phi, indexing='ij')

    # Compute total weights
    weights = (w_r_grid * w_theta_grid * w_phi_grid).ravel()

    r_flat = r_grid.ravel()
    theta_flat = theta_grid.ravel()
    phi_flat = phi_grid.ravel()

    # Evaluate integrands on the grid
    S_vals = integrand_at_rtp(r_flat, theta_flat, phi_flat,
                              bra_func, ket_func,
                              g_alpha, g_norm, Z, rc,
                              bra_order, ket_order)
    # print ('S_vals', S_vals.shape)
    H_vals = integrand_at_rtp_H(r_flat, theta_flat, phi_flat,
                                bra_func, ket_func,
                                g_alpha, g_norm, Z, rc,
                                bra_order, ket_order)

    # Sum over all quadrature points
    S_rc = jnp.sum(S_vals * weights)
    H_rc = jnp.sum(H_vals * weights)

    return S_rc, H_rc


# JIT-compiled basis functions
radial_basis_one_minus_b_X_wrapped = radial_basis_one_minus_b_X


@partial(jax.jit, static_argnames=['Z', 'rc', 'order'])
def radial_basis_b_Q_wrapped(r, g_alpha, g_norm, Z, rc, order):
    return radial_basis_b_Q(r, g_alpha, g_norm, Z, rc, order)


def cusp_coeff_vec(g_alpha, g_norm, Z, rc):
    # Define basis functions for different orders
    basis_funcs_with_orders = [
        (radial_basis_one_minus_b_X_wrapped, None),     # (1-b)*X
        (radial_basis_b_Q_wrapped, 0),  # b*Q * r^0
        (radial_basis_b_Q_wrapped, 2),  # b*Q * r^2 (order 1 skipped, cusp)
        (radial_basis_b_Q_wrapped, 3),  # b*Q * r^3
        (radial_basis_b_Q_wrapped, 4),
        (radial_basis_b_Q_wrapped, 5),
        (radial_basis_b_Q_wrapped, 6),
        (radial_basis_b_Q_wrapped, 7)
    ]

    n_basis = len(basis_funcs_with_orders)
    H_mat = jnp.zeros((n_basis, n_basis))
    S_mat = jnp.zeros((n_basis, n_basis))

    # Compute matrix elements
    for i, (bra_func, bra_order) in enumerate(basis_funcs_with_orders):
        for j, (ket_func, ket_order) in enumerate(basis_funcs_with_orders):
            S_rc, H_rc = test_integrand_at_rtp(bra_func, ket_func,
                                               g_alpha, g_norm, Z, rc,
                                               bra_order=bra_order,
                                               ket_order=ket_order)
            H_mat = H_mat.at[i, j].set(H_rc)
            S_mat = S_mat.at[i, j].set(S_rc)

    L = jnp.linalg.cholesky(S_mat)
    # If Cholesky fails, add small regularization
    if jnp.any(jnp.isnan(L)) or jnp.any(jnp.isinf(L)):
        reg = 1e-12 * jnp.eye(S_mat.shape[0])
        L = jnp.linalg.cholesky(S_mat + reg)

    # Solve L^{-1} H L^{-T}
    L_inv = jnp.linalg.inv(L)
    L_inv_T = L_inv.T

    # Transform Hamiltonian: H' = L^{-1} H L^{-T}
    H_transformed = L_inv @ H_mat @ L_inv_T

    # Solve standard eigenvalue problem
    eigenvalues, y = jnp.linalg.eigh(H_transformed)

    # Transform eigenvectors back: q = L^{-T} y
    eigenvectors = L_inv_T @ y

    idx_cho = eigenvalues.argsort()
    evec_best = eigenvectors[:, idx_cho[0]]
    coeff_vec = evec_best / evec_best[0]

    return coeff_vec


@jax.jit
def gto_1s_alpha_norm(basis):
    ish = basis[0]      # 1s orbital
    # Pre-compute primitive pairs more efficiently
    nprim = len(ish[1:])
    ip_indices, jp_indices = jnp.triu_indices(nprim, k=0)

    primitives = jnp.array(ish[1:])
    alphas = primitives[:, 0]
    coeffs = primitives[:, 1]

    # Vectorized normalization calculation
    cnorm = jnp.exp(0.75 * jnp.log(2.0 * alphas / jnp.pi))
    norm = coeffs * cnorm

    # Vectorized overlap calculation
    alpha_sum = alphas[ip_indices] + alphas[jp_indices]
    fac = alpha_sum * jnp.sqrt(alpha_sum)
    overlap_terms = norm[ip_indices] * norm[jp_indices] / fac

    # Handle diagonal vs off-diagonal terms
    diagonal_mask = ip_indices == jp_indices
    overlap_terms = jnp.where(diagonal_mask,
                              overlap_terms, 2.0 * overlap_terms)

    pi32 = jnp.pi**(1.5)
    facs = 1.0 / jnp.sqrt(jnp.sum(overlap_terms) * pi32)

    return alphas, norm * facs


@jax.jit
def evaluate_cusp_s(r, rc, Z, q0, rad_s, coeff):
    """Evaluate cusp-corrected s orbital."""
    b = evaluate_b_func(r, rc)
    s = evaluate_slater_func(r, q0, Z)
    r_over_rc = r / rc

    # Vectorized computation of powers
    r_powers = jnp.power(r_over_rc, jnp.arange(2, 8))

    # Construct the terms
    terms = jnp.concatenate([
        jnp.array([(1-b)*rad_s, b*s]),
        b*s*r_powers
    ])

    return jnp.dot(coeff, terms)


def minimize_q0(g_alpha, g_norm, Z, rc, coeff):
    # 1. Define GTO radial function for integration
    @partial(jax.jit, static_argnames=['Z', 'rc'])
    def radial_gto_1s_orb(r, g_alpha, g_norm, Z, rc):
        r2 = r*r
        rad_s = jnp.sum(jnp.exp(-g_alpha*r2)*g_norm)
        return rad_s

    # 2. Integrate GTO overlap and Hamiltonian
    # (using the efficient radial-only integrand)
    int_S_gto, int_H_gto = test_integrand_at_rtp(radial_gto_1s_orb,
                                                 radial_gto_1s_orb,
                                                 g_alpha, g_norm, Z, rc)

    # Simplified cusp function for the rest of minimize_q0
    @partial(jax.jit, static_argnames=['Z', 'rc'])
    def cusp_1s_orb_min_func(r, g_alpha, g_norm, Z, rc, q0, coeff_vec):
        r2 = r * r
        rad_s = jnp.sum(jnp.exp(-g_alpha*r2)*g_norm)
        cgs = evaluate_cusp_s(r, rc, Z, q0, rad_s, coeff_vec)
        return cgs

    @jax.jit
    def min_func(params):
        qq0 = params[0]
        # coeff0 = coeff.at[0].set(params[1])
        # Define a partial function to be used in the integrator
        cusp_func_partial = partial(cusp_1s_orb_min_func,
                                    q0=qq0, coeff_vec=coeff)

        # Use the optimized integration function
        int_S_cusp, int_H_cusp = test_integrand_at_rtp(cusp_func_partial,
                                                       cusp_func_partial,
                                                       g_alpha, g_norm, Z, rc)

        return (int_S_cusp - int_S_gto)**2 + (int_H_cusp-int_H_gto)**2

    cur_params = jnp.array([1.0])
    if jax.config.jax_logging_level in ["DEBUG", "INFO"]:
        print('Objective function (first call): {:.2E}'
              .format(min_func(cur_params)))

    solver = optax.lbfgs()
    opt_state = solver.init(cur_params)
    loss_and_grad = optax.value_and_grad_from_state(min_func)
    tolerance = 1e-8

    max_iter = 500
    l_not_converged = True
    for step in range(max_iter):
        loss_val, grads = loss_and_grad(cur_params, state=opt_state)
        updates, opt_state \
            = solver.update(grads, opt_state, cur_params,
                            value=loss_val, grad=grads, value_fn=min_func)
        cur_params = optax.apply_updates(cur_params, updates)
        print(step+1, loss_val)
        if jax.config.jax_logging_level in ["DEBUG", "INFO"]:
            print('Objective function: {:.2E}'.format(min_func(cur_params)))

        if jnp.abs(loss_val) < tolerance:
            l_not_converged = False
            break

    if jax.config.jax_logging_level in ["DEBUG", "WARNING"] \
            and l_not_converged:
        print("Warning: minimize_q0 has not converged.")

    return cur_params
