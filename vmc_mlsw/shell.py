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

def read_shell(ish_basis, ia, nsgs, ncgs):
    """
    Reads and processes a single basis shell from PySCF basis set format.
    Optimized for JAX with pre-computed arrays and vectorized operations.
    """
    # Extract angular momentum from first element
    am = ish_basis[0]
    nprim = len(ish_basis[1:])

    # Pre-compute primitive pairs more efficiently
    ip_indices, jp_indices = jnp.triu_indices(nprim, k=0)

    # Initialize shell object
    shell = ShellType()
    shell.iat = ia
    shell.isgs = nsgs
    shell.icgs = ncgs
    shell.am = am
    shell.nprim = nprim

    # Extract alphas and coefficients vectorized
    primitives = jnp.array(ish_basis[1:])
    alphas = primitives[:, 0]
    coeffs = primitives[:, 1]

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
        diagonal_mask = ip_indices == jp_indices
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

    return shell
