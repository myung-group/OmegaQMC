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



def read_two_shells (ish_basis, ia, nsgs, ncgs):
    """
    Reads and processes a split-valence basis shell
    from PySCF basis set format.
    This function handles shells that have two sets of contraction coefficients
    (e.g., for double-zeta basis sets).

    Args:
        ish_basis: List containing basis set information for a shell
                  [angular_momentum, (alpha1, coeff1_1, coeff2_1), ...]
        ia: Index of the atom this shell belongs to
        nsgs: Current index for spherical Gaussian orbitals
        ncgs: Current index for cartesian Gaussian orbitals

    Returns:
        shell1, shell2: Two ShellType objects containing
                        processed basis shell information
    """
    # Extract angular momentum from first element
    am = ish_basis[0]
    # Count number of primitive Gaussians
    nprim = len(ish_basis[1:])
    # Generate pairs of primitive indices for normalization
    nprim_pairs = [(ip, jp) for ip in range(nprim) for jp in range(ip+1)]

    # Initialize two shell objects for split-valence basis
    shell1 = ShellType()
    shell2 = ShellType()

    # Set common properties for both shells
    shell1.iat = ia         # Atom index
    shell1.am = am          # Angular momentum
    shell1.nprim = nprim    # Number of primitives

    shell2.iat = ia         # Same atom index
    shell2.am = am          # Same angular momentum
    shell2.nprim = nprim    # Same number of primitives

    if am == 0:
        # For s-orbitals (l=0), each shell has one component
        # in both cartesian and spherical
        shell1.ncgs = 1  # One cartesian Gaussian shell for first contraction
        shell1.nsgs = 1  # One spherical Gaussian shell for first contraction
        shell2.ncgs = 1  # One cartesian Gaussian shell for second contraction
        shell2.nsgs = 1  # One spherical Gaussian shell for second contraction

        # Lists to store exponents and normalized coefficients
        # for both contractions
        shell_alpha = []   # Common exponents for both contractions
        shell_norm1 = []   # Normalized coefficients for first contraction
        shell_norm2 = []   # Normalized coefficients for second contraction

        # Process each primitive Gaussian
        for iprm_basis in ish_basis[1:]:
            alpha, coeff1, coeff2 = iprm_basis
            # Calculate normalization factor for primitive s-orbital
            # For s-orbitals: N = (2α/π)^(3/4)
            cnorm = jnp.exp(0.75*jnp.log(2.0*alpha/jnp.pi))
            # Apply normalization to both sets of coefficients
            norm1 = coeff1*cnorm
            norm2 = coeff2*cnorm

            shell_alpha.append(alpha)
            shell_norm1.append(norm1)
            shell_norm2.append(norm2)

        # Calculate overlap normalization factors for both contractions
        facs1 = 0.0  # For first contraction
        facs2 = 0.0  # For second contraction

        # Compute overlap integrals for both contractions
        for ip, jp in nprim_pairs:
            aa = shell_alpha[ip] + shell_alpha[jp]  # Sum of exponents
            fac = aa * jnp.sqrt(aa)  # Factor in overlap integral
            # Calculate overlap contributions
            dum1 = shell_norm1[ip]*shell_norm1[jp]/fac  # For first contraction
            dum2 = shell_norm2[ip]*shell_norm2[jp]/fac  # For second contraction
            if ip != jp:
                dum1 = dum1+dum1  # Double counting for off-diagonal terms
                dum2 = dum2+dum2  # Double counting for off-diagonal terms
            facs1 += dum1
            facs2 += dum2

        # Final normalization factors
        facs1 = 1.0/jnp.sqrt(facs1*pi32)
        facs2 = 1.0/jnp.sqrt(facs2*pi32)

        # Store exponents and normalized coefficients
        shell1.alpha = jnp.array(shell_alpha)
        shell2.alpha = jnp.array(shell_alpha)
        shell_norm1 = jnp.array(shell_norm1)
        shell_norm2 = jnp.array(shell_norm2)
        shell1.norm = jax.lax.mul(shell_norm1, facs1)
        shell2.norm = jax.lax.mul(shell_norm2, facs2)

    elif am == 1:
        # For p-orbitals (l=1), each shell has 3 components (px, py, pz)
        # in both cartesian and spherical
        shell1.ncgs = 3  # Three cartesian Gaussian shells for first contraction
        shell1.nsgs = 3  # Three spherical Gaussian shells for first contraction
        shell2.ncgs = 3  # Three cartesian Gaussian shells for second contraction
        shell2.nsgs = 3  # Three spherical Gaussian shells for second contraction

        # Lists to store exponents and normalized coefficients
        # for both contractions
        shell_alpha = []   # Common exponents for both contractions
        shell_norm1 = []   # Normalized coefficients for first contraction
        shell_norm2 = []   # Normalized coefficients for second contraction

        # Process each primitive Gaussian
        for iprm_basis in ish_basis[1:]:
            alpha, coeff1, coeff2 = iprm_basis

            # Calculate normalization factor for primitive p-orbital
            # For p-orbitals: N = (2α/π)^(3/4) * sqrt(4α)
            cnorm = jnp.exp(0.75*jnp.log(2.0*alpha/jnp.pi))
            cnorm = cnorm * jnp.sqrt(4.0*alpha)  # Additional factor for p-orbital
            # Apply normalization to both sets of coefficients
            norm1 = coeff1*cnorm
            norm2 = coeff2*cnorm

            shell_alpha.append(alpha)
            shell_norm1.append(norm1)
            shell_norm2.append(norm2)

        # Calculate overlap normalization factors for both contractions
        facs1 = 0.0  # For first contraction
        facs2 = 0.0  # For second contraction

        # Compute overlap integrals for both contractions
        for ip, jp in nprim_pairs:
            aa = shell_alpha[ip] + shell_alpha[jp]  # Sum of exponents
            fac = aa * jnp.sqrt(aa)  # Factor in overlap integral
            # Factor of 0.5 comes from p-orbital overlap integral
            dum1 = 0.5*shell_norm1[ip]*shell_norm1[jp]/(aa*fac)  # For first contraction
            dum2 = 0.5*shell_norm2[ip]*shell_norm2[jp]/(aa*fac)  # For second contraction
            if ip != jp:
                dum1 = dum1+dum1  # Double counting for off-diagonal terms
                dum2 = dum2+dum2  # Double counting for off-diagonal terms
            facs1 += dum1
            facs2 += dum2

        # Final normalization factors
        facs1 = 1.0/jnp.sqrt(facs1*pi32)
        facs2 = 1.0/jnp.sqrt(facs2*pi32)

        # Store exponents and normalized coefficients
        shell1.alpha = jnp.array(shell_alpha)
        shell2.alpha = jnp.array(shell_alpha)
        shell_norm1 = jnp.array(shell_norm1)
        shell_norm2 = jnp.array(shell_norm2)
        shell1.norm = jax.lax.mul(shell_norm1, facs1)
        shell2.norm = jax.lax.mul(shell_norm2, facs2)

    else:
        print("NOT YET")  # Higher angular momentum not yet implemented

    # Set indices for the shells
    shell1.isgs = nsgs  # First shell starts at current spherical index
    shell1.icgs = ncgs  # First shell starts at current cartesian index
    shell2.isgs = nsgs + shell1.nsgs  # Second shell starts after first shell's spherical orbitals
    shell2.icgs = ncgs + shell1.ncgs  # Second shell starts after first shell's cartesian orbitals

    return shell1, shell2  # Return both shells for split-valence basis
