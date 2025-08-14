import jax
import jax.numpy as jnp
#from functools import partial
from vmc_mlsw.shell import read_shell
from vmc_mlsw.constants import JASTROW_EE_L_CUT, JASTROW_EE_M_POWER, EE_CUSP_VALUE


def get_psi_fun(mf):
    """
    Creates optimized functions for evaluating the wavefunction
    and local energy components from a PySCF mean-field calculation.
    """
    # Extract basic molecular information
    mol = mf.mol
    l_spherical = not mol.cart
    nocc = jnp.count_nonzero(mf.mo_occ > 0)
    mo_occ_coeff = mf.mo_coeff[:, :nocc]

    # Get nuclear and electronic charges
    Z_charges = mol.atom_charges()
    nelec = mf.mol.tot_electrons()
    e_charges = -jnp.ones((nelec))
    
    # Initialize counters
    nsgs = 0
    ncgs = 0
    shell_list = []

    # Process basis functions for each atom
    for ia, atom in enumerate(mol._atom):
        symb = atom[0]
        basis = mol._basis[symb]

        for ish, ish_basis in enumerate(basis):
            shell = read_shell(ish_basis, ia, nsgs, ncgs)
            shell.is_cusp = 0
            nsgs = nsgs + shell.nsgs
            ncgs = ncgs + shell.ncgs
            shell_list.append(shell)

    '''
    # Pre-compute shell parameters for vectorized operations
    n_shells = len(shell_list)
    max_nsgs = max(shell.nsgs for shell in shell_list)
    max_nprim = max(shell.nprim for shell in shell_list)
    
    # Create structured arrays for better memory access
    shell_am = jnp.array([shell.am for shell in shell_list])
    shell_iat = jnp.array([shell.iat for shell in shell_list])
    shell_isgs = jnp.array([shell.isgs for shell in shell_list])
    shell_nsgs = jnp.array([shell.nsgs for shell in shell_list])
    shell_is_cusp = jnp.array([shell.is_cusp for shell in shell_list])
    
    # Pad arrays for vectorization
    shell_alphas = jnp.zeros((n_shells, max_nprim))
    shell_norms = jnp.zeros((n_shells, max_nprim))
    
    for i, shell in enumerate(shell_list):
        shell_alphas = shell_alphas.at[i, :shell.nprim].set(shell.alpha)
        shell_norms = shell_norms.at[i, :shell.nprim].set(shell.norm)
    '''

    @jax.jit
    def cgs_sph_get_optimized(elec_crds, nuc_crds):
        """
        Optimized spherical GTO evaluation using vectorized operations.
        """
        n_nuc = nuc_crds.shape[0]
        shell_nsgs_total = shell_list[-1].isgs + shell_list[-1].nsgs
        ao_val = jnp.zeros((n_nuc, shell_nsgs_total))
        ao_val_s = jnp.zeros((n_nuc, shell_nsgs_total))
        # Vectorized distance calculations
        for i, shell in enumerate(shell_list):
            pos = nuc_crds[shell.iat]
            dr = elec_crds - pos
            r2 = jnp.sum(dr * dr, axis=-1)
            r = jnp.sqrt(r2)
            
            # Vectorized radial evaluation
            alpha = shell.alpha
            norm = shell.norm
            rad_s = jnp.sum(jnp.exp(-alpha * r2) * norm)
            cgs = rad_s
            # Angular part based on angular momentum
            if shell.am == 0:
                ao_val_s = ao_val_s.at[shell.iat, shell.isgs:shell.isgs+shell.nsgs].set(cgs)
            elif shell.am == 1:
                cgs = rad_s * dr
            elif shell.am == 2:
                cd1 = jnp.sqrt(3.0)
                cd2 = cd1 * 0.5
                x, y, z = dr
                cgs = rad_s * jnp.array([
                    cd1*x*y,
                    cd1*y*z,
                    0.5*(2.0*z*z - x*x - y*y),
                    -cd1*x*z,
                    cd2*(x*x - y*y)
                ])
            
            ao_val = ao_val.at[shell.iat, 
                               shell.isgs:shell.isgs+shell.nsgs].set(cgs)

        return ao_val, ao_val_s

    @jax.jit
    def get_psi_mo_optimized(elec_crds, nuc_crds):
        """Optimized molecular orbital evaluation."""
        ao_val, ao_val_s = jax.vmap(cgs_sph_get_optimized, in_axes=(0, None))(elec_crds, nuc_crds)
        mo_val = jnp.einsum('ena,am->nem', ao_val, mo_occ_coeff)
        mo_val_s = jnp.einsum('ena,am->nem', ao_val_s, mo_occ_coeff)
        
        return mo_val, mo_val_s

    @jax.jit
    def log_slater_determinant_optimized(elec_crds, nuc_crds):
        """Optimized Slater determinant calculation."""
        mo_val, _ = get_psi_mo_optimized(elec_crds, nuc_crds)
        mo_val = mo_val.sum(axis=0)
        
        # More efficient spin splitting
        alpha_matrix = mo_val[::2, :]
        beta_matrix = mo_val[1::2, :]
        
        # Stable log determinant calculation
        sign_alpha, log_det_alpha = jnp.linalg.slogdet(alpha_matrix)
        sign_beta, log_det_beta = jnp.linalg.slogdet(beta_matrix)
        
        return log_det_alpha + log_det_beta

    @jax.jit
    def J2_aa(elec_crds, params_vmc):
        """
        Two-body Jastrow for like-spin electron pairs.

        Uses a cutoff polynomial form that satisfies the electron-electron
        cusp condition for same-spin pairs:
            u_aa(r) = a * r * max(1 - r/L, 0)^M
        where a = 1/4 for like-spin, L and M are global constants.
        """
        # Pairwise distances between electrons
        diffs = elec_crds[:, None, :] - elec_crds[None, :, :]
        dists = jnp.sqrt(jnp.sum(diffs * diffs, axis=-1))

        # Upper-triangular pairs (i < j)
        ne = elec_crds.shape[0]
        iu, ju = jnp.triu_indices(ne, k=1)
        r_ij = dists[iu, ju]

        # Same-spin mask assuming alternating spin ordering
        # (0:A, 1:B, 2:A, 3:B, ...)
        same_spin_mask = (iu % 2) == (ju % 2)
        # Avoid boolean indexing inside jitted code
        # (causes NonConcreteBooleanIndexError)
        same_spin_mask_f = same_spin_mask.astype(r_ij.dtype)

        # Jastrow parameters
        a_cusp = EE_CUSP_VALUE  # 1/4 for like-spin electrons
        # L_cut = JASTROW_EE_L_CUT
        # m_pow = JASTROW_EE_M_POWER

        # Cutoff polynomial (C^M-1 continuous at r = L_cut)
        # one_minus = 1.0 - r_ij / L_cut
        # cutoff = jnp.clip(one_minus, a_min=0.0)  # max(1 - r/L, 0)
        # u_pairs = a_cusp * r_ij * cutoff**m_pow
        u_pairs = a_cusp * r_ij / (1. + params_vmc[0]*r_ij)

        # Sum only same-spin contributions via masking
        return jnp.sum(u_pairs * same_spin_mask_f)

    @jax.jit
    def J2_ab(elec_crds, params_vmc):
        """
        Two-body Jastrow for opposite-spin electron pairs.

        Uses the same cutoff polynomial form as J2_aa but with the
        unlike-spin cusp a = 1/2:
            u_ab(r) = a * r * max(1 - r/L, 0)^M
        """
        # Pairwise distances between electrons
        diffs = elec_crds[:, None, :] - elec_crds[None, :, :]
        dists = jnp.sqrt(jnp.sum(diffs * diffs, axis=-1))

        # Upper-triangular pairs (i < j)
        ne = elec_crds.shape[0]
        iu, ju = jnp.triu_indices(ne, k=1)
        r_ij = dists[iu, ju]

        # Opposite-spin mask assuming alternating spin ordering
        opp_spin_mask = (iu % 2) != (ju % 2)
        # Avoid boolean indexing; use mask multiplication
        opp_spin_mask_f = opp_spin_mask.astype(r_ij.dtype)

        # Jastrow parameters
        a_cusp = 2.0 * EE_CUSP_VALUE  # 1/2 for unlike-spin electrons
        # L_cut = JASTROW_EE_L_CUT
        # m_pow = JASTROW_EE_M_POWER

        # Cutoff polynomial
        # one_minus = 1.0 - r_ij / L_cut
        # cutoff = jnp.clip(one_minus, a_min=0.0)
        # u_pairs = a_cusp * r_ij * cutoff**m_pow
        u_pairs = a_cusp * r_ij / (1. + params_vmc[0]*r_ij)

        # Sum only opposite-spin contributions via masking
        return jnp.sum(u_pairs * opp_spin_mask_f)

    @jax.jit
    def log_trial_wavefunction(elec_crds, nuc_crds, params_vmc):
        """Optimized trial wavefunction."""
        return log_slater_determinant_optimized(elec_crds, nuc_crds) \
            + J2_aa(elec_crds, params_vmc)

#             + J2_ab(elec_crds, params_vmc)

    @jax.jit
    def classical_coulomb_optimized(crds1, chgs1, crds2=None, chgs2=None):
        """Optimized Coulomb interaction calculation."""
        eps = 0.0
        if crds2 is None:
            # Intra-particle interactions
            i, j = jnp.triu_indices(crds1.shape[0], k=1)
            diffs = crds1[i] - crds1[j]
            dists = jnp.sqrt(jnp.sum(diffs * diffs, axis=-1) + eps)
            return jnp.sum(chgs1[i] * chgs1[j] / dists)
        elif chgs2 is not None:
            # Inter-particle interactions
            diffs = crds1[:, None, :] - crds2[None, :, :]
            dists = jnp.sqrt(jnp.sum(diffs * diffs, axis=-1) + eps)
            return jnp.sum(chgs1[:, None] * chgs2[None, :] / dists)
        else:
            raise ValueError("chgs2 is None")
        
    @jax.jit
    def local_energy_ee(elec_crds):
        """Optimized electron-electron energy."""
        return classical_coulomb_optimized(elec_crds, e_charges)

    @jax.jit
    def local_energy_nn(nuc_crds):
        """Optimized nuclear-nuclear energy."""
        return classical_coulomb_optimized(nuc_crds, Z_charges)

    @jax.jit
    def local_energy_en(elec_crds, nuc_crds):
        """Optimized electron-nuclear energy."""
        return classical_coulomb_optimized(elec_crds, e_charges, nuc_crds, Z_charges)

    @jax.jit
    def local_energy_ke(elec_crds, nuc_crds, params_vmc):
        """Optimized kinetic energy calculation."""
        def _log_psi_flat(p_flat):
            return log_trial_wavefunction(p_flat.reshape(-1, 3), nuc_crds, params_vmc)
        
        # Use more efficient gradient calculations
        grad_fn = jax.grad(_log_psi_flat)
        hess_fn = jax.hessian(_log_psi_flat)
        
        p_flat = elec_crds.flatten()
        grad_log_psi = grad_fn(p_flat)
        hess_log_psi = hess_fn(p_flat)
        # jax.debug.print("!!neneA!: {x}", x=jnp.isnan(p_flat).any())
        # jax.debug.print("!!neneB!:  {x}", x=jnp.isnan(hess_log_psi).any())
        # jax.debug.print("!!neneC!:   {x}", x=jnp.isnan(grad_log_psi).any())
        
        lap_term = jnp.trace(hess_log_psi)
        grad_term_sq = jnp.sum(grad_log_psi**2)
        
        return -0.5 * (lap_term + grad_term_sq)

    return (log_trial_wavefunction, 
            (local_energy_ee, local_energy_nn, local_energy_en, local_energy_ke), 
            get_psi_mo_optimized)
