import jax
import jax.numpy as jnp
from dataclasses import dataclass
# import numpy as np
# from functools import partial
# import sys


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


# Constant used in normalization of Gaussian functions
pi32 = jnp.pi**(1.5)  # π^(3/2) appears in normalization of Gaussian functions


def read_shell(ish_basis, ia, nsgs, ncgs):
    """
    Reads and processes a single basis shell from PySCF basis set format.

    Args:
        ish_basis: List containing basis set information for a shell
                  [angular_momentum, (alpha1, coeff1), (alpha2, coeff2), ...]
        ia: Index of the atom this shell belongs to
        nsgs: Current index for spherical Gaussian orbitals
        ncgs: Current index for cartesian Gaussian orbitals

    Returns:
        shell: ShellType object containing processed basis shell information
    """
    # Extract angular momentum from first element
    am = ish_basis[0]
    # Count number of primitive Gaussians
    nprim = len(ish_basis[1:])
    # Generate pairs of primitive indices for normalization
    nprim_pairs = [(ip, jp) for ip in range(nprim) for jp in range(ip+1)]

    # Initialize shell object with basic information
    shell = ShellType()
    shell.iat = ia     # Atom index
    shell.isgs = nsgs  # Spherical GTO index
    shell.icgs = ncgs  # Cartesian GTO index
    shell.am = am      # Angular momentum
    shell.nprim = nprim  # Number of primitives

    if am == 0:
        # For s-orbitals (l=0), there is one shell
        # in both cartesian and spherical coordinates
        shell.ncgs = 1  # One cartesian Gaussian shell
        shell.nsgs = 1  # One spherical Gaussian shell

        shell_alpha = []  # List to store exponents
        shell_norm = []   # List to store normalized coefficients

        # Process each primitive Gaussian in the contraction
        for iprm_basis in ish_basis[1:]:
            alpha, coeff = iprm_basis
            # Calculate normalization factor for primitive Gaussian
            # For s-orbitals: N = (2α/π)^(3/4)
            cnorm = jnp.exp(0.75*jnp.log(2.0*alpha/jnp.pi))
            norm = coeff*cnorm

            shell_alpha.append(alpha)
            shell_norm.append(norm)

        # Calculate overlap normalization factor
        facs = 0.0

        # Double loop over primitives to compute overlap integrals
        for ip in range(nprim):
            for jp in range(ip+1):
                aa = shell_alpha[ip] + shell_alpha[jp]  # Sum of exponents
                fac = aa * jnp.sqrt(aa)  # Factor in overlap integral
                dum = shell_norm[ip]*shell_norm[jp]/fac
                if ip != jp:
                    dum = dum+dum  # Double counting for off-diagonal terms
                facs += dum

        # Final normalization factor
        facs = 1.0/jnp.sqrt(facs*pi32)

        # Store exponents and normalized coefficients
        shell.alpha = jnp.array(shell_alpha)
        shell_norm = jnp.array(shell_norm)
        shell.norm = jax.lax.mul(shell_norm, facs)

    elif am == 1:
        # For p-orbitals (l=1), there are 3 components (px, py, pz)
        # in both cartesian and spherical
        shell.ncgs = 3  # Three cartesian Gaussian shells
        shell.nsgs = 3  # Three spherical Gaussian shells

        shell_alpha = []  # List to store exponents
        shell_norm = []   # List to store normalized coefficients

        # Process each primitive Gaussian in the contraction
        for iprm_basis in ish_basis[1:]:
            alpha, coeff = iprm_basis

            # Calculate normalization factor for primitive p-orbital
            # For p-orbitals: N = (2α/π)^(3/4) * sqrt(4α)
            cnorm = jnp.exp(0.75*jnp.log(2.0*alpha/jnp.pi))
            cnorm = cnorm * jnp.sqrt(4.0*alpha)
            # ^ Additional factor for p-orbital
            norm = coeff*cnorm

            shell_alpha.append(alpha)
            shell_norm.append(norm)

        # Calculate overlap normalization factor
        facs = 0.0

        # Compute overlap integrals using pre-generated primitive pairs
        for ip, jp in nprim_pairs:
            aa = shell_alpha[ip] + shell_alpha[jp]  # Sum of exponents
            fac = aa * jnp.sqrt(aa)  # Factor in overlap integral
            # Factor of 0.5 comes from p-orbital overlap integral
            dum = 0.5*shell_norm[ip]*shell_norm[jp]/(aa*fac)
            if ip != jp:
                dum = dum+dum  # Double counting for off-diagonal terms
            facs += dum

        # Final normalization factor
        facs = 1.0/jnp.sqrt(facs*pi32)

        # Store exponents and normalized coefficients
        shell.alpha = jnp.array(shell_alpha)
        shell_norm = jnp.array(shell_norm)
        shell.norm = jax.lax.mul(shell_norm, facs)

    elif shell.am == 2:
        # For d-orbitals (l=2):
        # - 6 cartesian components (xx, xy, xz, yy, yz, zz)
        # - 5 spherical components
        #         (d_{z^2}, d_{xz}, d_{yz}, d_{x^2-y^2}, d_{xy})
        shell.ncgs = 6  # Six cartesian Gaussian shells
        shell.nsgs = 5  # Five spherical Gaussian shells

        shell_alpha = []  # List to store exponents
        shell_norm = []   # List to store normalized coefficients

        # Process each primitive Gaussian in the contraction
        for iprm_basis in ish_basis[1:]:
            alpha, coeff = iprm_basis

            # Calculate normalization factor for primitive d-orbital
            # For d-orbitals: N = (2α/π)^(3/4) * 4α/√3
            cnorm = jnp.exp(0.75*jnp.log(2.0*alpha/jnp.pi))
            cnorm = cnorm * 4.0*alpha  # Additional α factor for d-orbital
            cnorm = cnorm / jnp.sqrt(3.0)  # Normalization for d-orbital
            norm = coeff*cnorm

            shell_alpha.append(alpha)
            shell_norm.append(norm)

        # Store exponents and normalized coefficients
        shell.alpha = jnp.array(shell_alpha)
        shell.norm = jnp.array(shell_norm)

    else:
        print("NOT YET")  # Higher angular momentum not yet implemented

    return shell


def read_shell_two(ish_basis, ia, nsgs, ncgs):
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


def get_psi_fun(mf):
    """
    Creates functions for evaluating the wavefunction
    and local energy components from a PySCF mean-field calculation.

    Args:
        mf: PySCF mean-field object containing molecular
            and electronic structure information

    Returns:
        log_trial_wavefunction: Function to evaluate log of trial wavefunction
        local_energy: Tuple of functions to evaluate local energy components
    """
    # Extract basic molecular information
    mol = mf.mol
    l_spherical = not mol.cart  # Whether to use spherical (True) or cartesian (False) Gaussians
    nocc = jnp.count_nonzero(mf.mo_occ > 0)  # Number of occupied orbitals
    mo_occ_coeff = mf.mo_coeff[:, :nocc]  # Coefficients for occupied molecular orbitals

    # Get nuclear and electronic charges
    Z_charges = mol.atom_charges()  # Nuclear charges
    nelec = mf.mol.tot_electrons()  # Total number of electrons
    e_charges = -jnp.ones((nelec))  # Electronic charges (all -1)

    # Initialize counters for Gaussian shell indices
    nsgs = 0  # Counter for spherical Gaussian shells
    ncgs = 0  # Counter for cartesian Gaussian shells

    # List to store all basis shells
    shell_list = []

    # Process basis functions for each atom
    for ia, atom in enumerate(mol._atom):
        symb = atom[0]  # Get atomic symbol
        basis = mol._basis[symb]  # Get basis set for this atom type

        # Process each shell in the basis set
        for ish, ish_basis in enumerate(basis):
            # Get number of coefficients in first primitive to determine basis type
            ncoeff = len(ish_basis[1])

            if ncoeff == 2:
                # Regular basis shell (one set of coefficients)
                shell = read_shell(ish_basis, ia, nsgs, ncgs)
                # Update shell counters
                nsgs = nsgs + shell.nsgs  # Add number of spherical shells
                ncgs = ncgs + shell.ncgs  # Add number of cartesian shells
                shell_list.append(shell)
            else:
                # Split-valence basis shell (two sets of coefficients)
                shell1, shell2 = read_shell_two(ish_basis, ia, nsgs, ncgs)
                # Update shell counters for both shells
                nsgs = nsgs + shell1.nsgs + shell2.nsgs  # Add spherical shells from both
                ncgs = ncgs + shell1.ncgs + shell2.ncgs  # Add cartesian shells from both
                shell_list.append(shell1)
                shell_list.append(shell2)

    def cgs_cart_get(elec_crds, nuc_crds):
        """
        Evaluates cartesian Gaussian-type orbital (GTO) basis functions.

        Args:
            elec_crds: Coordinates of electron
            nuc_crds: Coordinates of all nuclei

        Returns:
            ao_val: Array of atomic orbital values at electron position
        """
        # Get dimensions
        n_nuc = nuc_crds.shape[0]  # Number of nuclei
        shell_ncgs = shell_list[-1].icgs + shell_list[-1].ncgs  # Total number of cartesian GTOs
        ao_val = jnp.zeros((n_nuc, shell_ncgs))  # Array to store AO values

        # Loop over all shells
        for ishell in shell_list:
            # Get shell parameters
            am = ishell.am      # Angular momentum
            alpha = ishell.alpha  # Exponents
            norm = ishell.norm    # Normalization coefficients
            ic = ishell.icgs     # Starting index for this shell
            ncgs = ishell.ncgs   # Number of cartesian components

            # Get nuclear position for this shell
            iat = ishell.iat
            pos = nuc_crds[iat]

            # Calculate electron-nuclear distance
            dr = elec_crds - pos  # Vector from nucleus to electron
            r2 = (dr*dr).sum(axis=-1)  # Distance squared

            # Calculate radial part of GTO
            rad_s = (jnp.exp(-alpha*r2)*norm).sum()  # Sum over primitives
            cgs = rad_s  # For s-orbitals (am=0)

            # Add angular part based on angular momentum
            if am == 1:
                # p-orbitals: multiply by x, y, or z
                cgs = rad_s*dr
            elif am == 2:
                # d-orbitals: multiply by xx, xy, xz, yy, yz, zz
                x, y, z = dr
                cgs = rad_s * jnp.array([
                    x*x, x*y, x*z, y*y, y*z, z*z
                ])

            # Store values in output array
            ao_val = ao_val.at[iat, ic:ic+ncgs].set(cgs)

        return ao_val

    def cgs_sph_get(elec_crds, nuc_crds):
        """
        Evaluates spherical Gaussian-type orbital (GTO) basis functions.

        Args:
            elec_crds: Coordinates of electron
            nuc_crds: Coordinates of all nuclei

        Returns:
            ao_val: Array of atomic orbital values at electron position
        """
        # Get dimensions
        n_nuc = nuc_crds.shape[0]  # Number of nuclei
        shell_nsgs = shell_list[-1].isgs + shell_list[-1].nsgs  # Total number of spherical GTOs
        ao_val = jnp.zeros((n_nuc, shell_nsgs))  # Array to store AO values

        # Loop over all shells
        for ishell in shell_list:
            # Get shell parameters
            am = ishell.am      # Angular momentum
            alpha = ishell.alpha  # Exponents
            norm = ishell.norm    # Normalization coefficients
            ic = ishell.isgs     # Starting index for this shell
            nsgs = ishell.nsgs   # Number of spherical components

            # Get nuclear position for this shell
            iat = ishell.iat
            pos = nuc_crds[iat]

            # Calculate electron-nuclear distance
            dr = elec_crds - pos  # Vector from nucleus to electron
            r2 = (dr*dr).sum(axis=-1)  # Distance squared

            # Calculate radial part of GTO
            rad_s = (jnp.exp(-alpha*r2)*norm).sum()  # Sum over primitives
            cgs = rad_s  # For s-orbitals (am=0)

            # Add angular part based on angular momentum
            if am == 1:
                # p-orbitals: same as cartesian (px, py, pz)
                cgs = rad_s*dr
            elif am == 2:
                # d-orbitals: transform to spherical harmonics
                cd1 = jnp.sqrt(3.0)  # Normalization constant
                cd2 = cd1*0.5        # Half of normalization constant

                x, y, z = dr
                # Five spherical d-orbitals:
                # dxy, dyz, dz2, dxz, dx2-y2
                cgs = rad_s * jnp.array([
                    cd1*x*y,                      # dxy
                    cd1*y*z,                      # dyz
                    0.5*(2.0*z*z - x*x - y*y),   # dz2
                    -cd1*x*z,                     # dxz
                    cd2*(x*x - y*y)               # dx2-y2
                ])

            # Store values in output array
            ao_val = ao_val.at[iat, ic:ic+nsgs].set(cgs)

        return ao_val

    def get_psi_mo(elec_crds, nuc_crds):
        """
        Evaluates molecular orbitals at given electron coordinates.

        Args:
            elec_crds: Array of electron coordinates
            nuc_crds: Array of nuclear coordinates

        Returns:
            mo_val: Molecular orbital values at electron positions
                   Shape: (n_nuc, n_elec, n_occ)
        """
        # Evaluate atomic orbitals for each electron using vectorized operations
        # ao_val shape: [n_elec, n_nuc, shell_nsgs]
        if l_spherical:
            # Use spherical Gaussian basis functions
            ao_val = jax.vmap(cgs_sph_get, in_axes=(0, None))(
                elec_crds, nuc_crds
            )
        else:
            # Use cartesian Gaussian basis functions
            ao_val = jax.vmap(cgs_cart_get, in_axes=(0, None))(
                elec_crds, nuc_crds
            )

        # Transform atomic orbitals to molecular orbitals using occupied MO coefficients
        # mo_val shape: (n_nuc, n_elec, n_occ)
        mo_val = jnp.einsum('ena,am->nem', ao_val, mo_occ_coeff)

        return mo_val

    def log_slater_determinant(elec_crds, nuc_crds):
        """
        Calculates the log of the Slater determinant wavefunction.

        The Slater determinant for a restricted Hartree-Fock (RHF) wavefunction is:
        |Ψ⟩ = (1/√N!)|χ₁(x₁) χ₂(x₁) ...|
                     |χ₁(x₂) χ₂(x₂) ...|

        where χᵢ are spin-orbitals:
        χ_even(x) = ψ(x)α  (spin up)
        χ_odd(x)  = ψ(x)β (spin down)

        For RHF: nocc = nelec // 2 (paired electrons)

        Args:
            elec_crds: Array of electron coordinates
            nuc_crds: Array of nuclear coordinates

        Returns:
            log_det: Log of the Slater determinant
        """
        # Get molecular orbital values and sum over nuclear centers
        mo_val = get_psi_mo(elec_crds, nuc_crds)
        mo_val = mo_val.sum(axis=0)  # Shape: (n_elec, n_occ)

        # Split electrons into spin up (alpha) and spin down (beta)
        alpha_indices = jnp.arange(0, nelec, 2)  # Even indices for spin up
        beta_indices = jnp.arange(1, nelec, 2)   # Odd indices for spin down

        # Extract spin up and spin down matrices
        alpha_matrix = mo_val[alpha_indices, :]  # Spin up determinant
        beta_matrix = mo_val[beta_indices, :]    # Spin down determinant

        # Calculate log determinants for both spin components
        sign_alpha, log_det_alpha = jnp.linalg.slogdet(alpha_matrix)
        sign_beta, log_det_beta = jnp.linalg.slogdet(beta_matrix)

        # Total log determinant is sum of up and down contributions
        log_det = log_det_alpha + log_det_beta
        return log_det

    def jastrow_factor(elec_crds, params_vmc):
        """
        Calculates the Jastrow correlation factor.
        Currently returns 1.0 (no correlation).
        """
        return 1.0

    def log_trial_wavefunction(elec_crds, nuc_crds, params_vmc):
        """
        Calculates the log of the trial wavefunction.
        Currently just uses the Slater determinant (no Jastrow factor).

        Args:
            elec_crds: Array of electron coordinates
            nuc_crds: Array of nuclear coordinates
            params_vmc: VMC parameters (for Jastrow factor)

        Returns:
            log_psi: Log of the trial wavefunction
        """
        return log_slater_determinant(elec_crds, nuc_crds)

    def classical_intra_coulomb_energy(crds, chgs):
        """
        Calculates the Coulomb energy between particles of the same type
        (electron-electron or nuclear-nuclear).

        Args:
            crds: Coordinates of particles
            chgs: Charges of particles

        Returns:
            e: Coulomb energy
        """
        # Get upper triangular indices to avoid double counting
        i, j = jnp.triu_indices(crds.shape[-2], k=1)
        # Calculate pairwise differences and distances
        diffs_ij = (crds.reshape(-1, 1, 3) - crds.reshape(1, -1, 3))[i, j]
        dists_ij = jnp.sqrt((diffs_ij*diffs_ij).sum(axis=-1))  # Add small constant to avoid division by zero
        # Calculate charge products
        chgs_ij = (chgs.reshape(-1, 1) * chgs.reshape(1, -1))[i, j]

        # Sum up Coulomb interactions
        e = jnp.einsum('i,i->', chgs_ij, 1.0/dists_ij)

        return e

    def classical_inter_coulomb_energy(elec_crds, nuc_crds,
                                       elec_chgs, nuc_chgs):
        """
        Calculates the Coulomb energy between electrons and nuclei.

        Args:
            elec_crds: Electron coordinates
            nuc_crds: Nuclear coordinates
            elec_chgs: Electron charges
            nuc_chgs: Nuclear charges

        Returns:
            e: Electron-nuclear Coulomb energy
        """
        # Calculate electron-nuclear distances
        diffs = elec_crds.reshape(-1, 1, 3)-nuc_crds  # Shape: (n_elec, n_nuc, 3)
        dists = jnp.sqrt(1.0e-12 + (diffs*diffs).sum(axis=-1))  # Add small constant to avoid division by zero
        # Sum up Coulomb interactions
        e = jnp.einsum('i,ij,j->', elec_chgs, 1.0/dists, nuc_chgs)
        return e

    def H_psi_over_psi(elec_crds, nuc_crds, params_vmc):
        """
        Calculates the local energy from kinetic energy terms.
        This involves computing the Laplacian of log(ψ) and the square of its gradient.

        Args:
            elec_crds: Electron coordinates
            nuc_crds: Nuclear coordinates
            params_vmc: VMC parameters

        Returns:
            ke: Local kinetic energy
        """
        def log_psi_of_single_electron_crds(r_elec_crds, elec_idx_fixed):
            """Helper function to compute log(ψ) with one electron moved"""
            temp_crds = elec_crds.at[elec_idx_fixed].set(r_elec_crds)
            return log_trial_wavefunction(temp_crds, nuc_crds, params_vmc)

        # Initialize sums for Laplacian and gradient squared terms
        lap_log_psi_sum = 0.0
        grad_log_psi_sq_sum = 0.0

        # Loop over electrons
        for i in range(elec_crds.shape[0]):
            # Calculate Laplacian (trace of Hessian) for electron i
            hess_func_i = jax.hessian(
                    lambda r_i_var: log_psi_of_single_electron_crds(r_i_var, i)
                    )
            lap_log_psi_sum += jnp.trace(hess_func_i(elec_crds[i]))

            # Calculate gradient squared for electron i
            grad_func_i = jax.grad(
                    lambda r_i_var: log_psi_of_single_electron_crds(r_i_var, i)
                    )
            grad_log_psi_i = grad_func_i(elec_crds[i])
            grad_log_psi_sq_sum += jnp.sum(grad_log_psi_i**2)

        # Return local kinetic energy
        return -0.5*(lap_log_psi_sum + grad_log_psi_sq_sum)

    def local_energy_ee(elec_crds):
        """Calculates electron-electron Coulomb energy"""
        ee_pot_energy = classical_intra_coulomb_energy(elec_crds, e_charges)
        return ee_pot_energy

    def local_energy_nn(nuc_crds):
        """Calculates nuclear-nuclear Coulomb energy"""
        nn_pot_energy = classical_intra_coulomb_energy(nuc_crds, Z_charges)
        return nn_pot_energy

    def local_energy_en(elec_crds, nuc_crds):
        """Calculates electron-nuclear Coulomb energy"""
        en_pot_energy = classical_inter_coulomb_energy(
            elec_crds, nuc_crds, e_charges, Z_charges
            )
        return en_pot_energy

    
    def local_energy_ke(elec_crds, nuc_crds, params_vmc):
        """Calculates local kinetic energy"""
        kin_energy = H_psi_over_psi(elec_crds, nuc_crds, params_vmc)
        return kin_energy
    '''
    def local_energy_ke(elec_crds, nuc_crds, params_vmc):
        """Calculates local kinetic energy"""

        def _log_psi_flat(p_flat):
            return log_trial_wavefunction(p_flat.reshape(-1, 3),
                                          nuc_crds,
                                          params_vmc)
        hessian_matrix = jax.hessian(_log_psi_flat)(elec_crds.flatten())
        lap_term = jnp.trace(hessian_matrix)

        grad_log_psi = jax.grad(log_trial_wavefunction, argnums=0)(elec_crds, nuc_crds, params_vmc)
        grad_term_sq = jnp.sum(grad_log_psi**2)

        return -0.5*(lap_term + grad_term_sq)
    '''
    return log_trial_wavefunction, \
        (local_energy_ee, local_energy_nn, local_energy_en, local_energy_ke), \
        get_psi_mo
