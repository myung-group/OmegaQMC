import jax
import jax.numpy as jnp
from dataclasses import dataclass
#from functools import partial


# Data class to represent a Gaussian basis shell
@dataclass
class ShellType:
    """
    Represents a Gaussian basis shell with its properties and parameters.
    Used to store information about atomic orbitals in
    quantum chemistry calculations.
    """
    am: int = 0    # Angular momentum quantum number (s=0, p=1, d=2, etc.)
    iat: int = 0   # Index of the atom this shell belongs to
    isgs: int = 0   # Index to spherical Gaussian type orbital (GTO)
    icgs: int = 0   # Index to cartesian GTO
    nprim: int = 0  # Number of primitive Gaussians in the contraction
    ncgs: int = 0  # Number of cartesian Gaussian shells
    nsgs: int = 0  # Number of spherical Gaussian shells
    alpha: jax.Array = None  # Exponents of primitive Gaussians
    norm: jax.Array = None  # Normalized contraction coefficients
    is_cusp: int = 0 # int 1: True


# Constant used in normalization of Gaussian functions
pi32 = jnp.pi**(1.5)  # π^(3/2) appears in normalization of Gaussian functions

@jax.jit
def evaluate_b_func(r, rc):
    """Evaluate B function for cusp correction."""
    s = r / rc
    s2 = s * s
    s3 = s2 * s
    s4 = s2 * s2
    s5 = s3 * s2
    return jnp.where(r < rc,
                     1.0 - 10.0 * s3 + 15.0 * s4 - 6.0 * s5,
                     0.0)

@jax.jit
def evaluate_slater_func(r, q0, Z):
    """Evaluate Slater function."""
    return q0 * jnp.exp(-Z * r)

@jax.jit
def evaluate_cusp_s(r, rc, Z, rad_s, q0, coeff):
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


@jax.jit
def evaluate_cusp_s_vgl(
    r, rc, Z, rad_s, rad_s_p_r, rad_s_lap, q0, coeff,
):
    """Cusp-corrected s: value, grad/r, and radial Laplacian.

    Returns the triple ``(f, f'/r, f'' + 2 f'/r)`` at scalar
    ``r``, where ``f`` is the same scalar that
    :func:`evaluate_cusp_s` returns.  These are the quantities
    the Slater block needs to build ``∇χ = (f'/r) · dr`` and
    ``∇²χ = f'' + 2 f'/r`` for an s-orbital (l = 0).

    ``rad_s``, ``rad_s_p_r``, ``rad_s_lap`` carry the
    uncorrected primitive sum ``R_0(r)``, ``R_0'(r)/r``, and
    ``R_0''(r) + 2 R_0'(r)/r``.  Expressed this way the
    ``(1 - b) R_0`` branch contains no bare ``1/r``.  The
    cusp-polynomial branch ``Q(r) = b(r) q0 e^{-Zr} P(r)``
    genuinely has a nuclear-cusp singularity (``Q'(0) ≠ 0``),
    so ``Q'/r`` is kept as a plain division — finite at any
    ``r`` a Metropolis walker visits.
    """
    s = r / rc
    one_minus_s = 1.0 - s
    inside = r < rc

    # b(r) and its radial derivatives — all forms below are
    # division-free (and regular at r = 0).  Factorisations:
    #   b'(r)  = -(30/rc)·s²·(1-s)²
    #   b'(r)/r = -(30/rc³)·r·(1-s)²
    #   b''(r) = -(60/rc²)·s·(1-s)·(1-2s)
    b_core = 1.0 - 10.0*s**3 + 15.0*s**4 - 6.0*s**5
    bp_core = -(30.0 / rc) * s**2 * one_minus_s**2
    bp_over_r_core = -(30.0 / rc**3) * r * one_minus_s**2
    bpp_core = (
        -(60.0 / rc**2) * s * one_minus_s * (1.0 - 2.0*s)
    )
    BL_core = bpp_core + 2.0 * bp_over_r_core

    b = jnp.where(inside, b_core, 0.0)
    bp = jnp.where(inside, bp_core, 0.0)
    bp_over_r = jnp.where(inside, bp_over_r_core, 0.0)
    BL = jnp.where(inside, BL_core, 0.0)

    # Part A: coeff[0] · (1 - b(r)) · R_0(r)
    c0 = coeff[0]
    A_val = c0 * (1.0 - b) * rad_s
    A_grad_over_r = c0 * (
        -bp_over_r * rad_s + (1.0 - b) * rad_s_p_r
    )
    A_lap = c0 * (
        -BL * rad_s
        + (1.0 - b) * rad_s_lap
        - 2.0 * r * bp * rad_s_p_r
    )

    # Part B: Q(r) = b(r) · q0·e^{-Zr} · P(r), a scalar
    # function of r only.  Second-order scalar autodiff is
    # O(1) memory — the Hessian-memory pathology this module
    # avoids is about vmap over 3N-dim coord vectors, not
    # scalar r.
    def _Q(rv):
        return evaluate_cusp_s(rv, rc, Z, 0.0, q0, coeff)
    Q_val = _Q(r)
    Q_grad = jax.grad(_Q)(r)
    Q_hess = jax.grad(jax.grad(_Q))(r)
    Q_grad_over_r = Q_grad / r
    Q_lap = Q_hess + 2.0 * Q_grad_over_r

    f_val = A_val + Q_val
    f_grad_over_r = A_grad_over_r + Q_grad_over_r
    f_lap = A_lap + Q_lap
    return f_val, f_grad_over_r, f_lap

def read_shell(ish_basis, ia, nsgs, ncgs):
    """
    Reads and processes a single basis shell from PySCF basis set format.
    Optimized for JAX with pre-computed arrays and vectorized operations.
    """
    # Extract angular momentum from first element
    am = ish_basis[0]

    primitives = jnp.array(ish_basis[1:])
    nprim, ncoef = primitives.shape

    # Pre-compute primitive pairs more efficiently
    ip_indices, jp_indices = jnp.triu_indices(nprim, k=0)
    diagonal_mask = ip_indices == jp_indices

    alphas = primitives[:, 0]
    shell_list = []
    curr_nsgs = nsgs
    curr_ncgs = ncgs
    for ic in range (1, ncoef):
        # Initialize shell object
        shell = ShellType()
        shell.iat = ia
        shell.isgs = curr_nsgs
        shell.icgs = curr_ncgs
        shell.am = am
        shell.nprim = nprim

        coeffs = primitives[:, ic]

        if am == 0:
            shell.ncgs = 1
            shell.nsgs = 1

            # Vectorized normalization calculation
            cnorm = jnp.exp(0.75 * jnp.log(2.0 * alphas / jnp.pi))
            norm = coeffs * cnorm

            # Vectorized overlap calculation
            alpha_sum = alphas[ip_indices] + alphas[jp_indices]
            fac = alpha_sum * jnp.sqrt(alpha_sum)
            overlap_terms = norm[ip_indices] * norm[jp_indices] / fac

            # Handle diagonal vs off-diagonal terms
            overlap_terms = jnp.where(diagonal_mask, overlap_terms, 2.0 * overlap_terms)

            facs = 1.0 / jnp.sqrt(jnp.sum(overlap_terms) * pi32)
            norm = norm * facs

        elif am == 1:
            shell.ncgs = 3
            shell.nsgs = 3

            # Vectorized p-orbital normalization
            cnorm = jnp.exp(0.75 * jnp.log(2.0 * alphas / jnp.pi))
            cnorm = cnorm * jnp.sqrt(4.0 * alphas)
            norm = coeffs * cnorm

            # Vectorized overlap calculation for p-orbitals
            alpha_sum = alphas[ip_indices] + alphas[jp_indices]
            fac = alpha_sum * jnp.sqrt(alpha_sum)
            overlap_terms = 0.5 * norm[ip_indices] * norm[jp_indices] / (alpha_sum * fac)

            diagonal_mask = ip_indices == jp_indices
            overlap_terms = jnp.where(diagonal_mask, overlap_terms, 2.0 * overlap_terms)

            facs = 1.0 / jnp.sqrt(jnp.sum(overlap_terms) * pi32)

            norm = norm * facs

        elif am == 2:
            shell.ncgs = 6
            shell.nsgs = 5

            # Vectorized d-orbital normalization
            cnorm = jnp.exp(0.75 * jnp.log(2.0 * alphas / jnp.pi))*(4.0*alphas)
            cnorm = cnorm / jnp.sqrt(3.0)
            norm = coeffs * cnorm

        elif am == 3:
            shell.ncgs = 10
            shell.nsgs = 7

            # Vectorized d-orbital normalization
            cnorm = jnp.exp(0.75 * jnp.log(2.0 * alphas / jnp.pi))*(4.0*alphas)**(3/2)
            cnorm = cnorm / jnp.sqrt(15.0)
            norm = coeffs * cnorm

        elif am == 4:
            shell.ncgs = 15
            shell.nsgs = 9

            # Vectorized d-orbital normalization
            cnorm = jnp.exp(0.75 * jnp.log(2.0 * alphas / jnp.pi))*(4.0*alphas)**2
            cnorm = cnorm / jnp.sqrt(7.0*15.0)
            norm = coeffs * cnorm

        else:
            raise NotImplementedError(f"Angular momentum {am} not yet implemented")

        shell.alpha = alphas
        shell.norm = norm
        curr_nsgs = curr_nsgs + shell.nsgs
        curr_ncgs = curr_ncgs + shell.ncgs
        shell_list.append (shell)

    return shell_list

