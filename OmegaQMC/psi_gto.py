import jax
import jax.numpy as jnp
# from functools import partial
from OmegaQMC.shell import read_shell, evaluate_cusp_s
# from OmegaQMC.constants import EE_CUSP_VALUE
# JASTROW_EE_L_CUT, JASTROW_EE_M_POWER


def get_psi_fun(mf, params_cusp=None, trial=None):
    """
    Creates functions for evaluating the wavefunction
    and local energy components from a PySCF mean-field calculation.
    """
    # Extract basic molecular information
    mol = mf.mol
    # l_spherical = not mol.cart
    nocc = jnp.count_nonzero(mf.mo_occ > 0)
    if trial is not None:
        import numpy as np
        mo_coeff_full = jnp.array(
            np.asarray(trial['mo_coeff'])
        )
        ci_coeffs = trial['ci_coeffs']
        occ_up = trial['occ_up']
        occ_dn = trial['occ_dn']
        # Dominant det's occupied MOs for get_psi_mo
        mo_occ_coeff = mo_coeff_full[:, occ_up[0]]
    else:
        mo_occ_coeff = mf.mo_coeff[:, :nocc]

    # Get nuclear and electronic charges
    Z_charges = mol.atom_charges()
    nelec = mol.tot_electrons()
    e_charges = -jnp.ones((nelec))

    # Initialize counters
    nsgs = 0
    ncgs = 0
    shell_list = []

    l_cgto = params_cusp is not None

    Z_rc = jnp.array([])
    Z_cgao_q0 = jnp.array([])
    Z_cgao_coeff = jnp.array([])
    if l_cgto:
        Z_rc = jnp.array([0.1 if Z == 1 else 0.2 for Z in Z_charges])
        Z_cgao_q0 = jnp.array([params_cusp[mol.atom_symbol(i)]['q0']
                               for i in range(mol.natm)])
        Z_cgao_coeff = jnp.array([params_cusp[mol.atom_symbol(i)]['coeff']
                                  for i in range(mol.natm)])

    # Process basis functions for each atom
    for ia, atom in enumerate(mol._atom):
        symb = atom[0]
        basis = mol._basis[symb]

        for ish, ish_basis in enumerate(basis):
            shells = read_shell(ish_basis, ia, nsgs, ncgs)

            for jsh, shell in enumerate(shells):
                shell.is_cusp = True \
                    if l_cgto and ish == 0 and jsh == 0 \
                    else False

                nsgs = nsgs + shell.nsgs
                ncgs = ncgs + shell.ncgs
                shell_list.append(shell)

    # Build per-element → per-atom mapping for J1
    unique_elements = []
    atom_to_elem_idx = []
    for ia in range(mol.natm):
        sym = mol.atom_symbol(ia)
        if sym not in unique_elements:
            unique_elements.append(sym)
        atom_to_elem_idx.append(unique_elements.index(sym))
    atom_to_elem_idx = jnp.array(atom_to_elem_idx)

    @jax.jit
    def cgs_sph_get(elec_crds, nuc_crds):
        """
        spherical GTO evaluation using vectorized operations.
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
            # Angular part based on angular momentum
            if shell.am == 0:
                if shell.is_cusp:
                    cgs = evaluate_cusp_s(r,
                                          Z_rc[shell.iat],
                                          Z_charges[shell.iat],
                                          rad_s,
                                          Z_cgao_q0[shell.iat],
                                          Z_cgao_coeff[shell.iat])
                else:
                    cgs = rad_s
                ao_val_s = ao_val_s.at[shell.iat,
                                       shell.isgs:shell.isgs+shell.nsgs] \
                    .set(cgs)
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
                    cd1*x*z,
                    cd2*(x*x - y*y)
                ])
            elif shell.am == 3:
                cf1 = jnp.sqrt(2.5)*0.5
                cf2 = 3.0*cf1
                cf3 = jnp.sqrt(15.0)
                cf4 = jnp.sqrt(1.5)*0.5
                cf5 = jnp.sqrt(6.0)
                cf6 = 1.5
                cf7 = cf3*0.5
                x, y, z = dr
                cgs = rad_s * jnp.array([
                    y*(cf2*x*x - cf1*y*y),  # xxy, yyy
                    cf3*x*y*z,
                    y*(cf5*z*z-cf4*(x*x+y*y)),
                    z*(z*z - cf6*(x*x+y*y)),
                    x*(cf5*z*z - cf4*(x*x+y*y)),
                    z*cf7*(x*x-y*y),
                    -x*(cf2*y*y-cf1*x*x)
                ])
            elif shell.am == 4:
                cg1 = 2.9580398915498085
                cg2 = 6.2749501990055672
                cg3 = 2.0916500663351894
                cg4 = 1.1180339887498949
                cg5 = 6.7082039324993694
                cg6 = 2.3717082451262845
                cg7 = 3.1622776601683795
                cg8 = 0.55901699437494745
                cg9 = 3.3541019662496847
                cg10 = 0.73950997288745213
                cg11 = 4.4370598373247132
                x, y, z = dr
                cgs = rad_s * jnp.array([
                    cg1*(x*x*x*y-x*y*y*y),
                    y*z*(cg2*x*x - cg3*y*y),
                    x*y*cg4*(-x*x - y*y) + cg5*x*y*z*z,
                    -cg6*x*x*y*z - cg6*y*y*y*z + cg7*y*z*z*z,
                    (0.375*(x*x*x*x + y*y*y*y + 2.0*x*x*y*y) +
                     z*z*z*z - 3.0*z*z*(x*x + y*y)),
                    -(cg6*x*x*x*z + cg6*x*y*y*z - cg7*x*z*z*z),
                    cg8*(y*y*y*y - x*x*x*x) + cg9*z*z*(x*x - y*y),
                    -x*z*(cg2*y*y - cg3*x*x),
                    cg10*(x*x*x*x + y*y*y*y) - cg11*x*x*y*y
                ])
            else:
                raise ValueError("shell.am > 4 is not supported yet.")

            ao_val = ao_val.at[shell.iat,
                               shell.isgs:shell.isgs+shell.nsgs].set(cgs)

        return ao_val, ao_val_s

    @jax.jit
    def get_psi_mo(elec_crds, nuc_crds):
        """Molecular orbital evaluation."""
        ao_val, ao_val_s \
            = jax.vmap(cgs_sph_get, in_axes=(0, None))(elec_crds, nuc_crds)
        mo_val = jnp.einsum('ena,am->em', ao_val, mo_occ_coeff)
        mo_val_s = jnp.einsum('ena,am->nem', ao_val_s, mo_occ_coeff)

        return mo_val, mo_val_s

    @jax.jit
    def get_ao_val(elec_crds, nuc_crds):
        """AO evaluation only (no MO contraction)."""
        ao_val, _ = jax.vmap(cgs_sph_get, in_axes=(0, None))(elec_crds,
                                                              nuc_crds)
        return ao_val   # (nelec, natm, nao_total)

    @jax.jit
    def log_slater_determinant_C(elec_crds, nuc_crds, C):
        """Slater determinant with explicit MO coefficients C."""
        ao_val, _ = jax.vmap(cgs_sph_get, in_axes=(0, None))(elec_crds,
                                                              nuc_crds)
        mo_val = jnp.einsum('ena,am->em', ao_val, C)
        alpha_matrix = mo_val[::2, :]
        beta_matrix = mo_val[1::2, :]
        _, log_det_alpha = jnp.linalg.slogdet(alpha_matrix)
        _, log_det_beta = jnp.linalg.slogdet(beta_matrix)
        return log_det_alpha + log_det_beta

    @jax.jit
    def log_trial_wavefunction_C(elec_crds, nuc_crds, curr_params, C):
        """Trial wavefunction with explicit MO coefficients C."""
        ln_slater = log_slater_determinant_C(elec_crds, nuc_crds, C)
        jastrow_term = 0.0
        if "J1_pade" in curr_params:
            jastrow_term += J1(elec_crds, nuc_crds, curr_params["J1_pade"])
        if "J2_pade" in curr_params:
            jastrow_term += J2_aa(elec_crds, curr_params["J2_pade"]) \
                + J2_ab(elec_crds, curr_params["J2_pade"])
        return ln_slater + jastrow_term

    @jax.jit
    def local_energy_ke_C(elec_crds, nuc_crds, curr_params, C):
        """Kinetic energy with explicit MO coefficients C."""
        def _log_psi_flat(p_flat):
            return log_trial_wavefunction_C(p_flat.reshape(-1, 3),
                                            nuc_crds, curr_params, C)
        grad_fn = jax.grad(_log_psi_flat)
        hess_fn = jax.hessian(_log_psi_flat)
        p_flat = elec_crds.flatten()
        grad_log_psi = grad_fn(p_flat)
        hess_log_psi = hess_fn(p_flat)
        lap_term = jnp.trace(hess_log_psi)
        grad_term_sq = jnp.sum(grad_log_psi**2)
        return -0.5 * (lap_term + grad_term_sq)

    @jax.jit
    def log_slater_determinant(elec_crds, nuc_crds):
        """Slater determinant calculation."""
        mo_val, _ = get_psi_mo(elec_crds, nuc_crds)

        # More efficient spin splitting
        alpha_matrix = mo_val[::2, :]
        beta_matrix = mo_val[1::2, :]

        # Stable log determinant calculation
        sign_alpha, log_det_alpha = jnp.linalg.slogdet(alpha_matrix)
        sign_beta, log_det_beta = jnp.linalg.slogdet(beta_matrix)

        return log_det_alpha + log_det_beta

    if trial is not None:
        @jax.jit
        def log_slater_multidet(elec_crds, nuc_crds):
            """Multi-determinant Slater using log-sum-exp."""
            ao_val, _ = jax.vmap(
                cgs_sph_get, in_axes=(0, None)
            )(elec_crds, nuc_crds)
            mo_val = jnp.einsum(
                'ena,am->em', ao_val, mo_coeff_full
            )
            mo_alpha = mo_val[::2, :]
            mo_beta = mo_val[1::2, :]

            def eval_one_det(occ_a, occ_b):
                sign_a, logdet_a = jnp.linalg.slogdet(
                    mo_alpha[:, occ_a]
                )
                sign_b, logdet_b = jnp.linalg.slogdet(
                    mo_beta[:, occ_b]
                )
                return sign_a, logdet_a, sign_b, logdet_b

            signs_a, logdets_a, signs_b, logdets_b = \
                jax.vmap(eval_one_det)(occ_up, occ_dn)

            log_abs_dets = logdets_a + logdets_b
            phase_signs = (
                signs_a * signs_b * jnp.sign(ci_coeffs)
            )
            log_abs_ci = jnp.log(jnp.abs(ci_coeffs))
            log_contributions = log_abs_ci + log_abs_dets
            logmax = jnp.max(log_contributions)
            sum_val = jnp.sum(
                phase_signs
                * jnp.exp(log_contributions - logmax)
            )
            return jnp.log(jnp.abs(sum_val)) + logmax

    _log_slater = log_slater_multidet \
        if trial is not None \
        else log_slater_determinant

    @jax.jit
    def J2_aa(elec_crds, curr_params):
        """
        Two-body Jastrow for like-spin electron pairs.

        Uses a Padé form:
            u_aa(r) = a * r / (1 + b * r)
        where a, b = curr_params["like"].
        To satisfy the e-e cusp condition, set a = 1/4.
        """
        # Pairwise distances between electrons
        # Upper-triangular pairs (i < j)
        i, j = jnp.triu_indices(elec_crds.shape[0], k=1)
        diffs = elec_crds[i] - elec_crds[j]
        r_ij = jnp.sqrt(jnp.sum(diffs*diffs, axis=-1))

        # Same-spin mask assuming alternating spin ordering
        # (0:A, 1:B, 2:A, 3:B, ...)
        same_spin_mask = (i % 2) == (j % 2)
        # Avoid boolean indexing inside jitted code
        # (causes NonConcreteBooleanIndexError)
        same_spin_mask_f = same_spin_mask.astype(r_ij.dtype)

        u_pairs = curr_params["like"][0] * r_ij \
            / (1. + curr_params["like"][1] * r_ij)

        # Sum only same-spin contributions via masking
        return jnp.sum(u_pairs * same_spin_mask_f)

    @jax.jit
    def J2_ab(elec_crds, curr_params):
        """
        Two-body Jastrow for opposite-spin electron pairs.

        Uses a Padé form:
            u_ab(r) = a * r / (1 + b * r)
        where a, b = curr_params["unlike"].
        To satisfy the e-e cusp condition, set a = 1/2.
        """
        # Pairwise distances between electrons
        # Upper-triangular pairs (i < j)
        i, j = jnp.triu_indices(elec_crds.shape[0], k=1)
        diffs = elec_crds[i] - elec_crds[j]
        r_ij = jnp.sqrt(jnp.sum(diffs*diffs, axis=-1))

        # Opposite-spin mask assuming alternating spin ordering
        opp_spin_mask = (i % 2) != (j % 2)
        # Avoid boolean indexing; use mask multiplication
        opp_spin_mask_f = opp_spin_mask.astype(r_ij.dtype)

        u_pairs = curr_params["unlike"][0] * r_ij \
            / (1. + curr_params["unlike"][1] * r_ij)

        # Sum only opposite-spin contributions via masking
        return jnp.sum(u_pairs * opp_spin_mask_f)

    @jax.jit
    def J1(elec_crds, nuc_crds, curr_params):
        if l_cgto:
            # Cusp handled by orbitals → both a and b from params
            ab_arr = jnp.stack([curr_params[sym]
                                for sym in unique_elements])
            a_per_atom = ab_arr[atom_to_elem_idx, 0]
            b_per_atom = ab_arr[atom_to_elem_idx, 1]
        else:
            # No cusp scheme → a = -Z (frozen), b from params
            b_arr = jnp.array([curr_params[sym]
                               for sym in unique_elements])
            b_per_atom = b_arr[atom_to_elem_idx]
            a_per_atom = -Z_charges
        diffs = elec_crds[None, :, :] - nuc_crds[:, None, :]
        r = jnp.linalg.norm(diffs, axis=-1)
        u_vals = a_per_atom[:, None] * r \
            / (1.0 + b_per_atom[:, None] * r)
        return jnp.sum(u_vals)

    @jax.jit
    def log_trial_wavefunction(elec_crds, nuc_crds, curr_params):
        """Trial wavefunction."""
        ln_slater = _log_slater(elec_crds, nuc_crds)

        jastrow_term = 0.0
        if "J1_pade" in curr_params:
            jastrow_term += J1(elec_crds, nuc_crds, curr_params["J1_pade"])
        if "J2_pade" in curr_params:
            jastrow_term += J2_aa(elec_crds, curr_params["J2_pade"]) \
                + J2_ab(elec_crds, curr_params["J2_pade"])

        return ln_slater + jastrow_term

    @jax.jit
    def classical_coulomb_energy(crds1, chgs1, crds2=None, chgs2=None):
        """Coulomb interaction calculation."""
        eps = jnp.finfo(crds1.dtype).eps
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
        """Electron-electron energy."""
        return classical_coulomb_energy(elec_crds, e_charges)

    @jax.jit
    def local_energy_nn(nuc_crds):
        """Nuclear-nuclear energy."""
        return classical_coulomb_energy(nuc_crds, Z_charges)

    @jax.jit
    def local_energy_en(elec_crds, nuc_crds):
        """Electron-nuclear energy."""
        return classical_coulomb_energy(elec_crds, e_charges,
                                        nuc_crds, Z_charges)

    @jax.jit
    def local_energy_ke(elec_crds, nuc_crds, curr_params):
        """Kinetic energy calculation."""
        def _log_psi_flat(p_flat):
            return log_trial_wavefunction(p_flat.reshape(-1, 3),
                                          nuc_crds, curr_params)

        # Use more efficient gradient calculations
        grad_fn = jax.grad(_log_psi_flat)
        hess_fn = jax.hessian(_log_psi_flat)

        p_flat = elec_crds.flatten()
        grad_log_psi = grad_fn(p_flat)
        hess_log_psi = hess_fn(p_flat)

        lap_term = jnp.trace(hess_log_psi)
        grad_term_sq = jnp.sum(grad_log_psi**2)

        return -0.5 * (lap_term + grad_term_sq)

    return (log_trial_wavefunction,
            (local_energy_ee, local_energy_nn,
             local_energy_en, local_energy_ke),
            get_psi_mo,
            (log_trial_wavefunction_C, local_energy_ke_C))
