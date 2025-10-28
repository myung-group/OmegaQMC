import jax
import jax.numpy as jnp
from pyscf import gto
import numpy as np  # For generating quadrature points
# from scipy.linalg import eig as sci_eig
from functools import partial
import jaxopt

jax.config.update("jax_enable_x64", True)


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


# JIT-compiled integrand for overlap (S) matrix
@partial(jax.jit, static_argnames=['bra_func', 'ket_func', 'Z', 'rc'])
# def integrand_at_rtp(r, theta, phi, bra_func, ket_func,
#                      g_alpha, g_norm, Z, rc, static_args=(0, 1)):
def integrand_at_rtp(r, theta, phi, bra_func, ket_func,
                     g_alpha, g_norm, Z, rc):
    sint = jnp.sin(theta)
    r2 = r * r
    r2sint = r2 * sint

    xoc = r * sint * jnp.cos(phi)
    yoc = r * sint * jnp.sin(phi)
    zoc = r * jnp.cos(theta)
    e_pos = jnp.stack([xoc, yoc, zoc], axis=-1)

    bra = jax.vmap(bra_func,
                   in_axes=(0, None, None, None, None))(e_pos,
                                                        g_alpha, g_norm,
                                                        Z, rc)
    ket = jax.vmap(ket_func,
                   in_axes=(0, None, None, None, None))(e_pos,
                                                        g_alpha, g_norm,
                                                        Z, rc)

    return bra * ket * r2sint


# JIT-compiled integrand for Hamiltonian (H) matrix
# @jax.jit
@partial(jax.jit, static_argnames=['bra_func', 'ket_func', 'Z', 'rc'])
def integrand_at_rtp_H(r, theta, phi, bra_func, ket_func,
                       g_alpha, g_norm, Z, rc):
    # dV = r*r sin(theta) d_r d_theta d_phi
    sint = jnp.sin(theta)
    r2 = r * r
    r2sint = r2 * sint

    xoc = r * sint * jnp.cos(phi)
    yoc = r * sint * jnp.sin(phi)
    zoc = r * jnp.cos(theta)
    e_pos = jnp.stack([xoc, yoc, zoc], axis=-1)

    bra = jax.vmap(bra_func,
                   in_axes=(0, None, None, None, None))(e_pos,
                                                        g_alpha, g_norm,
                                                        Z, rc)
    ket = jax.vmap(ket_func,
                   in_axes=(0, None, None, None, None))(e_pos,
                                                        g_alpha, g_norm,
                                                        Z, rc)

    # Compute Hessian trace efficiently
    hess_ket = jax.vmap(
        lambda x: jnp.trace(jax.hessian(ket_func)(x, g_alpha, g_norm, Z, rc))
        )(e_pos)

    retval_kE = bra * (-0.5 * hess_ket) * r2sint
    retval_eN = bra * (-Z/r) * ket * r2sint

    return retval_kE + retval_eN


# JIT-compiled integration function using Gauss-Legendre quadrature
# @jax.jit(static_argnums=(0, 1))  # Mark bra_func and ket_func as static
@partial(jax.jit, static_argnames=['bra_func', 'ket_func', 'Z', 'rc'])
def test_integrand_at_rtp(bra_func, ket_func,
                          g_alpha, g_norm, Z, rc, n_points=16):
    # Gauss-Legendre quadrature for r
    g16, w_r = get_gauss_legendre(n_points)
    rgrid = rc * 0.5 * (g16 + 1.0)
    w_r = rc * 0.5 * w_r

    # Gauss-Legendre quadrature for theta [0, pi]
    theta_nodes, w_theta = get_gauss_legendre(n_points)
    theta = 0.5 * (theta_nodes + 1.0) * jnp.pi
    w_theta = 0.5 * jnp.pi * w_theta

    # Gauss-Legendre quadrature for phi [0, 2pi]
    phi_nodes, w_phi = get_gauss_legendre(n_points)
    phi = 0.5 * (phi_nodes + 1.0) * 2.0 * jnp.pi
    w_phi = 0.5 * 2.0 * jnp.pi * w_phi

    # Create 3D grid of quadrature points
    r_grid, theta_grid, phi_grid \
        = jnp.meshgrid(rgrid, theta, phi, indexing='ij')
    w_r_grid, w_theta_grid, w_phi_grid \
        = jnp.meshgrid(w_r, w_theta, w_phi, indexing='ij')

    # Compute total weights
    weights = w_r_grid * w_theta_grid * w_phi_grid

    # print ('r_grid', r_grid.shape)
    # print ('weights', weights.shape)
    r_grid = r_grid.reshape(-1)
    theta_grid = theta_grid.reshape(-1)
    phi_grid = phi_grid.reshape(-1)
    weights = weights.reshape(-1)

    # Evaluate integrands on the grid
    S_vals = integrand_at_rtp(r_grid, theta_grid, phi_grid,
                              bra_func, ket_func,
                              g_alpha, g_norm, Z, rc)
    # print ('S_vals', S_vals.shape)
    H_vals = integrand_at_rtp_H(r_grid, theta_grid, phi_grid,
                                bra_func, ket_func,
                                g_alpha, g_norm, Z, rc)

    # Sum over all quadrature points
    S_rc = jnp.sum(S_vals * weights)
    H_rc = jnp.sum(H_vals * weights)

    return S_rc, H_rc


# JIT-compiled basis functions
# @jax.jit
@partial(jax.jit, static_argnames=['Z', 'rc'])
def basis_one_minus_b_X(e_pos, g_alpha, g_norm, Z, rc):
    # print ('e_pos', e_pos.shape)
    r2 = jnp.sum(e_pos * e_pos, axis=-1)
    r = jnp.sqrt(r2)
    b = evaluate_b_func(r, rc)
    # print ('r2', r2.shape)
    # print ('b', b.shape)
    # print ('g_alpha', g_alpha.shape)
    return (1 - b) * jnp.sum(jnp.exp(-g_alpha * r2) * g_norm)


@partial(jax.jit, static_argnames=['Z', 'rc', 'order'])
def basis_b_Q(e_pos, g_alpha, g_norm, Z, rc, order):
    r = jnp.sqrt(jnp.sum(e_pos * e_pos, axis=-1))
    s = evaluate_slater_func(r, 1.0, Z)
    b = evaluate_b_func(r, rc)
    return b * s * (r/rc)**order


def cusp_coeff_vec(g_alpha, g_norm, Z, rc):
    # Define basis functions for different orders
    basis_funcs = [
        basis_one_minus_b_X,
        lambda e, ga, gn, z, r: basis_b_Q(e, ga, gn, z, r, order=0),
        lambda e, ga, gn, z, r: basis_b_Q(e, ga, gn, z, r, order=2),
        lambda e, ga, gn, z, r: basis_b_Q(e, ga, gn, z, r, order=3),
        lambda e, ga, gn, z, r: basis_b_Q(e, ga, gn, z, r, order=4),
        lambda e, ga, gn, z, r: basis_b_Q(e, ga, gn, z, r, order=5),
        lambda e, ga, gn, z, r: basis_b_Q(e, ga, gn, z, r, order=6),
        lambda e, ga, gn, z, r: basis_b_Q(e, ga, gn, z, r, order=7)
    ]

    n_basis = len(basis_funcs)
    H_mat = jnp.zeros((n_basis, n_basis))
    S_mat = jnp.zeros((n_basis, n_basis))

    # Compute matrix elements
    for i, bra_func in enumerate(basis_funcs):
        for j, ket_func in enumerate(basis_funcs):
            S_rc, H_rc = test_integrand_at_rtp(bra_func, ket_func,
                                               g_alpha, g_norm, Z, rc)
            H_mat = H_mat.at[i, j].set(H_rc)
            S_mat = S_mat.at[i, j].set(S_rc)

    '''
    print("Hamiltonian matrix:")
    print(H_mat)
    print("\nOverlap matrix:")
    print(S_mat)

    # using scipy
    evals, evec_mat = sci_eig (H_mat, S_mat)
    idx = evals.argsort()
    evec_best = evec_mat[:, idx[0]]
    print ('evec_best', evec_best)
    print ('evals', evals)
    print ('idx', idx)
    coeff_vec = evec_best /evec_best[0]
    print ('coeff_vec', coeff_vec)

    svals_S, evec_S = jnp.linalg.eigh (S_mat)
    print ('svals_S', svals_S)
    '''
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
    # print ('evec_best', evec_best)
    # print ('evals', eigenvalues)
    # print ('eigenvectors', eigenvectors)
    coeff_vec = evec_best / evec_best[0]

    return coeff_vec


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
    n_points = 16
    # Gauss-Legendre quadrature for r
    g16, w_r = get_gauss_legendre(n_points)
    rgrid = rc * 0.5 * (g16 + 1.0)
    w_r = rc * 0.5 * w_r

    # Gauss-Legendre quadrature for theta [0, pi]
    theta_nodes, w_theta = get_gauss_legendre(n_points)
    theta = 0.5 * (theta_nodes + 1.0) * jnp.pi
    w_theta = 0.5 * jnp.pi * w_theta

    # Gauss-Legendre quadrature for phi [0, 2pi]
    phi_nodes, w_phi = get_gauss_legendre(n_points)
    phi = 0.5 * (phi_nodes + 1.0) * 2.0 * jnp.pi
    w_phi = 0.5 * 2.0 * jnp.pi * w_phi

    # Create 3D grid of quadrature points
    r_grid, theta_grid, phi_grid \
        = jnp.meshgrid(rgrid, theta, phi, indexing='ij')
    w_r_grid, w_theta_grid, w_phi_grid \
        = jnp.meshgrid(w_r, w_theta, w_phi, indexing='ij')

    # Compute total weights
    weights = w_r_grid * w_theta_grid * w_phi_grid

    # print ('r_grid', r_grid.shape)
    # print ('weights', weights.shape)
    r_grid = r_grid.reshape(-1)
    theta_grid = theta_grid.reshape(-1)
    phi_grid = phi_grid.reshape(-1)
    weights = weights.reshape(-1)

    sint = jnp.sin(theta_grid)
    r2 = r_grid*r_grid
    r2sint = r2 * sint

    xoc = r_grid * sint * jnp.cos(phi_grid)
    yoc = r_grid * sint * jnp.sin(phi_grid)
    zoc = r_grid * jnp.cos(theta_grid)
    e_pos = jnp.stack([xoc, yoc, zoc], axis=-1)

    def gto_1s_orb(epos):
        r2 = jnp.einsum('i,i->', epos, epos)
        # r = jnp.sqrt(r2)
        rad_s = jnp.sum(jnp.exp(-g_alpha*r2)*g_norm)
        return rad_s

    def cusp_1s_orb(epos, q0):
        r2 = jnp.einsum('i,i->', epos, epos)
        r = jnp.sqrt(r2)
        rad_s = jnp.sum(jnp.exp(-g_alpha*r2)*g_norm)
        cgs = evaluate_cusp_s(r, rc, Z, q0, rad_s, coeff)
        return cgs

    int_gto_1s = jax.vmap(gto_1s_orb)(e_pos)
    int_gto = jnp.sum(int_gto_1s*int_gto_1s*r2sint*weights)

    def min_func(q0):
        int_cusp_orb = jax.vmap(cusp_1s_orb, in_axes=(0, None))(e_pos, q0[0])
        int_cusp = jnp.sum(int_cusp_orb * int_cusp_orb * r2sint * weights)
        return (int_cusp - int_gto)**2

    solver = jaxopt.LBFGS(min_func, maxiter=500)
    init_params = jnp.array([1.0])
    res = solver.run(init_params)
    params, state = res
    return params


if __name__ == "__main__":
    # from scipy.linalg import eig as sci_eig
    # import json
    # Initialize molecule
    mol = gto.M(
        atom='Li 0.000000 0.000000 0.000',
        basis='6-31g',
        unit='Bohr',
        spin=1
    )
    mol.build()
    # print ('mol', mol.atom_coords())
    # Extract basis information
    symb = mol._atom[0][0]
    Z = mol.atom_charges()[0]
    rc = 0.1 if Z == 1 else 0.2
    print('Z', Z, 'rc', rc)
    basis = mol._basis[symb]

    g_alpha, g_norm = gto_1s_alpha_norm(basis)

    # Compute matrices
    coeff_vec = cusp_coeff_vec(g_alpha, g_norm, Z, rc)

    q0_min = minimize_q0(g_alpha, g_norm, Z, rc, coeff_vec)

    data = {
            int(Z): {
                'q0': q0_min.tolist()[0],
                'coeff': coeff_vec.tolist()
                }
            }

    # json_data = json.dumps(data, indent=4)
    print('Result')
    print(data)
