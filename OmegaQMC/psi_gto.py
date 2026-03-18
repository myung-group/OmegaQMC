import jax
import jax.numpy as jnp
# from functools import partial
from OmegaQMC.shell import read_shell, evaluate_cusp_s
# from OmegaQMC.constants import EE_CUSP_VALUE
# JASTROW_EE_L_CUT, JASTROW_EE_M_POWER


# --- B-spline helpers (QMCPACK BsplineFunctor convention) ---

def _build_bspline_coefs(params, delta_r, cusp_val):
    """Map N user params → (N+4) full B-spline coefficients.

    Following QMCPACK's BsplineFunctor.h:
      coefs[1] = P[0], coefs[2] = P[1], ..., coefs[N] = P[N-1]
      coefs[0] = coefs[2] - 2 * delta_r * cusp_val   (cusp)
      coefs[N+1] = coefs[N+2] = coefs[N+3] = 0       (boundary)
    """
    # n = params.shape[0]
    c0 = params[1] - 2.0 * delta_r * cusp_val
    coefs = jnp.concatenate([
        c0[jnp.newaxis],
        params,
        jnp.zeros(3),
    ])
    return coefs


def _bspline_eval(r, coefs, delta_r_inv, max_index):
    """Evaluate cubic B-spline at distances r.

    Uses the uniform-knot cubic basis from QMCPACK.
    r is an array of distances; returns u(r) for each.
    """
    u = r * delta_r_inv
    i = jnp.clip(
        jnp.floor(u).astype(jnp.int32), 0, max_index
    )
    t = u - i

    # Cubic B-spline basis values (uniform knots)
    tp = jnp.stack([t * t * t, t * t, t, jnp.ones_like(t)])
    basis = jnp.array([
        [-1,  3, -3, 1],
        [ 3, -6,  3, 0],
        [-3,  0,  3, 0],
        [ 1,  4,  1, 0],
    ], dtype=r.dtype) / 6.0

    w = basis.T @ tp                      # (4, len(r))

    c0 = coefs[i]
    c1 = coefs[i + 1]
    c2 = coefs[i + 2]
    c3 = coefs[i + 3]

    return w[0] * c0 + w[1] * c1 + w[2] * c2 + w[3] * c3


def _angular_cartesian(am, dr, rad_s):
    """Return GTO angular part for angular momentum am.

    am is a Python int (unrolled at JAX trace time).
    """
    if am == 0:
        return rad_s
    elif am == 1:
        return rad_s * dr
    elif am == 2:
        cd1 = jnp.sqrt(3.0)
        cd2 = cd1 * 0.5
        x, y, z = dr
        return rad_s * jnp.array([
            cd1*x*y,
            cd1*y*z,
            0.5*(2.0*z*z - x*x - y*y),
            cd1*x*z,
            cd2*(x*x - y*y)
        ])
    elif am == 3:
        cf1 = jnp.sqrt(2.5)*0.5
        cf2 = 3.0*cf1
        cf3 = jnp.sqrt(15.0)
        cf4 = jnp.sqrt(1.5)*0.5
        cf5 = jnp.sqrt(6.0)
        cf6 = 1.5
        cf7 = cf3*0.5
        x, y, z = dr
        return rad_s * jnp.array([
            y*(cf2*x*x - cf1*y*y),
            cf3*x*y*z,
            y*(cf5*z*z-cf4*(x*x+y*y)),
            z*(z*z - cf6*(x*x+y*y)),
            x*(cf5*z*z - cf4*(x*x+y*y)),
            z*cf7*(x*x-y*y),
            -x*(cf2*y*y-cf1*x*x)
        ])
    elif am == 4:
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
        return rad_s * jnp.array([
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


class _PsiGTO:
    """
    Holds all JAX-compiled wavefunction and energy functions
    derived from a PySCF mean-field object.
    """

    def __init__(self, mf, params_cusp=None, trial=None,
                 bspline_config=None):
        self.mf = mf
        self.trial = trial
        self._parse_mol(mf, params_cusp, trial)
        self._parse_shells(mf)
        self._parse_elements(mf)
        self._parse_bspline_cfg(bspline_config)
        self._build_ao_fns()
        self._build_slater_fns()
        self._build_pade_jastrow_fns()
        self._build_bspline_jastrow_fns()
        self._build_trial_wf_fns()
        self._build_coulomb_fns()
        self._build_ke_fns()

    def _parse_mol(self, mf, params_cusp, trial):
        """Extract MO coefficients, charges, and cusp arrays."""
        mol = mf.mol
        nocc = jnp.count_nonzero(mf.mo_occ > 0)
        if trial is not None:
            import numpy as np
            self.mo_coeff_full = jnp.array(
                np.asarray(trial['mo_coeff'])
            )
            self.ci_coeffs = trial['ci_coeffs']
            self.occ_up = trial['occ_up']
            self.occ_dn = trial['occ_dn']
            self.mo_occ_coeff = self.mo_coeff_full[
                :, self.occ_up[0]
            ]
        else:
            self.mo_occ_coeff = mf.mo_coeff[:, :nocc]

        self.Z_charges = mol.atom_charges()
        self.nelec = mol.tot_electrons()
        self.e_charges = -jnp.ones((self.nelec,))
        self.l_cgto = params_cusp is not None

        if self.l_cgto:
            self.Z_rc = jnp.array(
                [0.1 if Z == 1 else 0.2
                 for Z in self.Z_charges]
            )
            self.Z_cgao_q0 = jnp.array(
                [params_cusp[mol.atom_symbol(i)]['q0']
                 for i in range(mol.natm)]
            )
            self.Z_cgao_coeff = jnp.array(
                [params_cusp[mol.atom_symbol(i)]['coeff']
                 for i in range(mol.natm)]
            )
        else:
            self.Z_rc = jnp.array([])
            self.Z_cgao_q0 = jnp.array([])
            self.Z_cgao_coeff = jnp.array([])

    def _parse_shells(self, mf):
        """Build shell_list from basis, assigning is_cusp flags."""
        mol = mf.mol
        nsgs = 0
        ncgs = 0
        shell_list = []
        for ia, atom in enumerate(mol._atom):
            symb = atom[0]
            basis = mol._basis[symb]
            for ish, ish_basis in enumerate(basis):
                shells = read_shell(ish_basis, ia, nsgs, ncgs)
                for jsh, shell in enumerate(shells):
                    shell.is_cusp = (
                        self.l_cgto and ish == 0 and jsh == 0
                    )
                    nsgs = nsgs + shell.nsgs
                    ncgs = ncgs + shell.ncgs
                    shell_list.append(shell)
        self.shell_list = shell_list

    def _parse_elements(self, mf):
        """Build unique element list and per-atom element index."""
        mol = mf.mol
        unique_elements = []
        atom_to_elem_idx = []
        for ia in range(mol.natm):
            sym = mol.atom_symbol(ia)
            if sym not in unique_elements:
                unique_elements.append(sym)
            atom_to_elem_idx.append(
                unique_elements.index(sym)
            )
        self.unique_elements = unique_elements
        self.atom_to_elem_idx = jnp.array(atom_to_elem_idx)

    def _parse_bspline_cfg(self, bspline_config):
        """Extract r_cut values from bspline_config dict."""
        self._bs_j2_cfg = None
        self._bs_j1_cfgs = {}
        if bspline_config is None:
            return
        if "J2" in bspline_config:
            r_cut = float(bspline_config["J2"]["r_cut"])
            self._bs_j2_cfg = {"r_cut": r_cut}
        if "J1" in bspline_config:
            for sym in self.unique_elements:
                if sym in bspline_config["J1"]:
                    rc = float(
                        bspline_config["J1"][sym]["r_cut"]
                    )
                    self._bs_j1_cfgs[sym] = {"r_cut": rc}

    def _build_ao_fns(self):
        """Build AO/MO evaluation functions."""
        shell_list = self.shell_list
        # l_cgto = self.l_cgto
        Z_rc = self.Z_rc
        Z_charges = self.Z_charges
        Z_cgao_q0 = self.Z_cgao_q0
        Z_cgao_coeff = self.Z_cgao_coeff
        mo_occ_coeff = self.mo_occ_coeff

        @jax.jit
        def cgs_sph_get(elec_crds, nuc_crds):
            """Spherical GTO evaluation."""
            n_nuc = nuc_crds.shape[0]
            shell_nsgs_total = (
                shell_list[-1].isgs + shell_list[-1].nsgs
            )
            ao_val = jnp.zeros((n_nuc, shell_nsgs_total))
            ao_val_s = jnp.zeros((n_nuc, shell_nsgs_total))
            for i, shell in enumerate(shell_list):
                pos = nuc_crds[shell.iat]
                dr = elec_crds - pos
                r2 = jnp.sum(dr * dr, axis=-1)
                r = jnp.sqrt(r2)
                alpha = shell.alpha
                norm = shell.norm
                rad_s = jnp.sum(
                    jnp.exp(-alpha * r2) * norm
                )
                cgs = _angular_cartesian(shell.am, dr, rad_s)
                if shell.am == 0 and shell.is_cusp:
                    cgs = evaluate_cusp_s(
                        r,
                        Z_rc[shell.iat],
                        Z_charges[shell.iat],
                        rad_s,
                        Z_cgao_q0[shell.iat],
                        Z_cgao_coeff[shell.iat]
                    )
                if shell.am == 0:
                    ao_val_s = ao_val_s.at[
                        shell.iat,
                        shell.isgs:shell.isgs+shell.nsgs
                    ].set(cgs)
                ao_val = ao_val.at[
                    shell.iat,
                    shell.isgs:shell.isgs+shell.nsgs
                ].set(cgs)
            return ao_val, ao_val_s

        @jax.jit
        def get_psi_mo(elec_crds, nuc_crds):
            """Molecular orbital evaluation."""
            ao_val, ao_val_s = jax.vmap(
                cgs_sph_get, in_axes=(0, None)
            )(elec_crds, nuc_crds)
            mo_val = jnp.einsum(
                'ena,am->em', ao_val, mo_occ_coeff
            )
            mo_val_s = jnp.einsum(
                'ena,am->nem', ao_val_s, mo_occ_coeff
            )
            return mo_val, mo_val_s

        @jax.jit
        def get_ao_val(elec_crds, nuc_crds):
            """AO evaluation only (no MO contraction)."""
            ao_val, _ = jax.vmap(
                cgs_sph_get, in_axes=(0, None)
            )(elec_crds, nuc_crds)
            return ao_val

        @jax.jit
        def log_slater_det_C(elec_crds, nuc_crds, C):
            """Slater determinant with explicit MO coefficients."""
            ao_val, _ = jax.vmap(
                cgs_sph_get, in_axes=(0, None)
            )(elec_crds, nuc_crds)
            mo_val = jnp.einsum('ena,am->em', ao_val, C)
            alpha_matrix = mo_val[::2, :]
            beta_matrix = mo_val[1::2, :]
            _, log_det_alpha = jnp.linalg.slogdet(
                alpha_matrix
            )
            _, log_det_beta = jnp.linalg.slogdet(
                beta_matrix
            )
            return log_det_alpha + log_det_beta

        self.cgs_sph_get = cgs_sph_get
        self.get_psi_mo = get_psi_mo
        self.get_ao_val = get_ao_val
        self._log_slater_det_C = log_slater_det_C

    def _build_slater_fns(self):
        """Build single- or multi-determinant Slater functions."""
        get_psi_mo = self.get_psi_mo
        cgs_sph_get = self.cgs_sph_get

        @jax.jit
        def log_slater_determinant(elec_crds, nuc_crds):
            """Single-determinant Slater."""
            mo_val, _ = get_psi_mo(elec_crds, nuc_crds)
            alpha_matrix = mo_val[::2, :]
            beta_matrix = mo_val[1::2, :]
            _, log_det_alpha = jnp.linalg.slogdet(
                alpha_matrix
            )
            _, log_det_beta = jnp.linalg.slogdet(
                beta_matrix
            )
            return log_det_alpha + log_det_beta

        if self.trial is not None:
            mo_coeff_full = self.mo_coeff_full
            ci_coeffs = self.ci_coeffs
            occ_up = self.occ_up
            occ_dn = self.occ_dn

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
                    signs_a * signs_b
                    * jnp.sign(ci_coeffs)
                )
                log_abs_ci = jnp.log(jnp.abs(ci_coeffs))
                log_contributions = log_abs_ci + log_abs_dets
                logmax = jnp.max(log_contributions)
                sum_val = jnp.sum(
                    phase_signs
                    * jnp.exp(log_contributions - logmax)
                )
                return jnp.log(jnp.abs(sum_val)) + logmax

            self._log_slater = log_slater_multidet
        else:
            self._log_slater = log_slater_determinant

    def _build_pade_jastrow_fns(self):
        """Build Padé-form Jastrow functions J2_aa, J2_ab, J1."""
        l_cgto = self.l_cgto
        unique_elements = self.unique_elements
        atom_to_elem_idx = self.atom_to_elem_idx
        Z_charges = self.Z_charges

        @jax.jit
        def J2_aa(elec_crds, curr_params):
            """Two-body Jastrow, like-spin pairs (Padé)."""
            i, j = jnp.triu_indices(elec_crds.shape[0], k=1)
            diffs = elec_crds[i] - elec_crds[j]
            r_ij = jnp.sqrt(jnp.sum(diffs*diffs, axis=-1))
            same_spin_mask = (i % 2) == (j % 2)
            same_spin_mask_f = same_spin_mask.astype(
                r_ij.dtype
            )
            u_pairs = curr_params["like"][0] * r_ij \
                / (1. + curr_params["like"][1] * r_ij)
            return jnp.sum(u_pairs * same_spin_mask_f)

        @jax.jit
        def J2_ab(elec_crds, curr_params):
            """Two-body Jastrow, opposite-spin pairs (Padé)."""
            i, j = jnp.triu_indices(elec_crds.shape[0], k=1)
            diffs = elec_crds[i] - elec_crds[j]
            r_ij = jnp.sqrt(jnp.sum(diffs*diffs, axis=-1))
            opp_spin_mask = (i % 2) != (j % 2)
            opp_spin_mask_f = opp_spin_mask.astype(r_ij.dtype)
            u_pairs = curr_params["unlike"][0] * r_ij \
                / (1. + curr_params["unlike"][1] * r_ij)
            return jnp.sum(u_pairs * opp_spin_mask_f)

        @jax.jit
        def J1(elec_crds, nuc_crds, curr_params):
            """One-body Jastrow (Padé)."""
            if l_cgto:
                ab_arr = jnp.stack(
                    [curr_params[sym]
                     for sym in unique_elements]
                )
                a_per_atom = ab_arr[atom_to_elem_idx, 0]
                b_per_atom = ab_arr[atom_to_elem_idx, 1]
            else:
                b_arr = jnp.array(
                    [curr_params[sym]
                     for sym in unique_elements]
                )
                b_per_atom = b_arr[atom_to_elem_idx]
                a_per_atom = -Z_charges
            diffs = (
                elec_crds[None, :, :]
                - nuc_crds[:, None, :]
            )
            r = jnp.linalg.norm(diffs, axis=-1)
            u_vals = a_per_atom[:, None] * r \
                / (1.0 + b_per_atom[:, None] * r)
            return jnp.sum(u_vals)

        self.J2_aa = J2_aa
        self.J2_ab = J2_ab
        self.J1 = J1

    def _build_bspline_jastrow_fns(self):
        """Build B-spline Jastrow functions."""
        l_cgto = self.l_cgto
        unique_elements = self.unique_elements
        atom_to_elem_idx = self.atom_to_elem_idx
        Z_charges = self.Z_charges
        _bs_j2_cfg = self._bs_j2_cfg
        _bs_j1_cfgs = self._bs_j1_cfgs

        @jax.jit
        def J2_bspline_aa(elec_crds, curr_params):
            """Two-body B-spline Jastrow, like-spin pairs."""
            n = curr_params["like"].shape[0]
            r_cut = _bs_j2_cfg["r_cut"]
            delta_r = r_cut / (n + 1)
            delta_r_inv = 1.0 / delta_r
            max_index = n
            coefs = _build_bspline_coefs(
                curr_params["like"], delta_r, -0.25
            )
            i_idx, j_idx = jnp.triu_indices(
                elec_crds.shape[0], k=1
            )
            diffs = elec_crds[i_idx] - elec_crds[j_idx]
            r_ij = jnp.sqrt(
                jnp.sum(diffs * diffs, axis=-1)
            )
            same_spin = (
                (i_idx % 2) == (j_idx % 2)
            ).astype(r_ij.dtype)
            cutoff_mask = (r_ij < r_cut).astype(r_ij.dtype)
            u = _bspline_eval(
                r_ij, coefs, delta_r_inv, max_index
            )
            return jnp.sum(u * same_spin * cutoff_mask)

        @jax.jit
        def J2_bspline_ab(elec_crds, curr_params):
            """Two-body B-spline Jastrow, unlike-spin pairs."""
            n = curr_params["unlike"].shape[0]
            r_cut = _bs_j2_cfg["r_cut"]
            delta_r = r_cut / (n + 1)
            delta_r_inv = 1.0 / delta_r
            max_index = n
            coefs = _build_bspline_coefs(
                curr_params["unlike"], delta_r, -0.5
            )
            i_idx, j_idx = jnp.triu_indices(
                elec_crds.shape[0], k=1
            )
            diffs = elec_crds[i_idx] - elec_crds[j_idx]
            r_ij = jnp.sqrt(
                jnp.sum(diffs * diffs, axis=-1)
            )
            opp_spin = (
                (i_idx % 2) != (j_idx % 2)
            ).astype(r_ij.dtype)
            cutoff_mask = (r_ij < r_cut).astype(r_ij.dtype)
            u = _bspline_eval(
                r_ij, coefs, delta_r_inv, max_index
            )
            return jnp.sum(u * opp_spin * cutoff_mask)

        @jax.jit
        def J1_bspline_fn(elec_crds, nuc_crds, curr_params):
            """One-body B-spline Jastrow."""
            total = 0.0
            for ie, sym in enumerate(unique_elements):
                if sym not in _bs_j1_cfgs:
                    continue
                r_cut = _bs_j1_cfgs[sym]["r_cut"]
                p = curr_params[sym]
                n = p.shape[0]
                delta_r = r_cut / (n + 1)
                delta_r_inv = 1.0 / delta_r
                max_index = n
                cusp_val = 0.0 if l_cgto else -float(
                    Z_charges[atom_to_elem_idx == ie][0]
                )
                coefs = _build_bspline_coefs(
                    p, delta_r, cusp_val
                )
                elem_mask = (
                    atom_to_elem_idx == ie
                ).astype(elec_crds.dtype)
                diffs = (
                    elec_crds[None, :, :]
                    - nuc_crds[:, None, :]
                )
                r = jnp.linalg.norm(diffs, axis=-1)
                u = _bspline_eval(
                    r.ravel(), coefs, delta_r_inv, max_index
                ).reshape(r.shape)
                cutoff = (r < r_cut).astype(r.dtype)
                total += jnp.sum(
                    u * cutoff * elem_mask[:, None]
                )
            return total

        self.J2_bspline_aa = J2_bspline_aa
        self.J2_bspline_ab = J2_bspline_ab
        self.J1_bspline_fn = J1_bspline_fn

    def _build_trial_wf_fns(self):
        """Build log_trial_wavefunction and _C variant."""
        _log_slater = self._log_slater
        _log_slater_det_C = self._log_slater_det_C
        J1 = self.J1
        J2_aa = self.J2_aa
        J2_ab = self.J2_ab
        J1_bspline_fn = self.J1_bspline_fn
        J2_bspline_aa = self.J2_bspline_aa
        J2_bspline_ab = self.J2_bspline_ab

        @jax.jit
        def log_trial_wavefunction(
            elec_crds, nuc_crds, curr_params
        ):
            """Trial wavefunction."""
            ln_slater = _log_slater(elec_crds, nuc_crds)
            jastrow_term = 0.0
            if "J1_pade" in curr_params:
                jastrow_term += J1(
                    elec_crds, nuc_crds,
                    curr_params["J1_pade"]
                )
            if "J2_pade" in curr_params:
                jastrow_term += J2_aa(
                    elec_crds, curr_params["J2_pade"]
                ) + J2_ab(
                    elec_crds, curr_params["J2_pade"]
                )
            if "J1_bspline" in curr_params:
                jastrow_term += J1_bspline_fn(
                    elec_crds, nuc_crds,
                    curr_params["J1_bspline"]
                )
            if "J2_bspline" in curr_params:
                jastrow_term += J2_bspline_aa(
                    elec_crds, curr_params["J2_bspline"]
                ) + J2_bspline_ab(
                    elec_crds, curr_params["J2_bspline"]
                )
            return ln_slater + jastrow_term

        @jax.jit
        def log_trial_wavefunction_C(
            elec_crds, nuc_crds, curr_params, C
        ):
            """Trial wavefunction with explicit MO coefficients."""
            ln_slater = _log_slater_det_C(
                elec_crds, nuc_crds, C
            )
            jastrow_term = 0.0
            if "J1_pade" in curr_params:
                jastrow_term += J1(
                    elec_crds, nuc_crds,
                    curr_params["J1_pade"]
                )
            if "J2_pade" in curr_params:
                jastrow_term += J2_aa(
                    elec_crds, curr_params["J2_pade"]
                ) + J2_ab(
                    elec_crds, curr_params["J2_pade"]
                )
            if "J1_bspline" in curr_params:
                jastrow_term += J1_bspline_fn(
                    elec_crds, nuc_crds,
                    curr_params["J1_bspline"]
                )
            if "J2_bspline" in curr_params:
                jastrow_term += J2_bspline_aa(
                    elec_crds, curr_params["J2_bspline"]
                ) + J2_bspline_ab(
                    elec_crds, curr_params["J2_bspline"]
                )
            return ln_slater + jastrow_term

        @jax.jit
        def local_energy_ke_C(
            elec_crds, nuc_crds, curr_params, C
        ):
            """Kinetic energy with explicit MO coefficients."""
            def _log_psi_flat(p_flat):
                return log_trial_wavefunction_C(
                    p_flat.reshape(-1, 3),
                    nuc_crds, curr_params, C
                )
            grad_fn = jax.grad(_log_psi_flat)
            hess_fn = jax.hessian(_log_psi_flat)
            p_flat = elec_crds.flatten()
            grad_log_psi = grad_fn(p_flat)
            hess_log_psi = hess_fn(p_flat)
            lap_term = jnp.trace(hess_log_psi)
            grad_term_sq = jnp.sum(grad_log_psi**2)
            return -0.5 * (lap_term + grad_term_sq)

        self.log_trial_wavefunction = log_trial_wavefunction
        self.log_trial_wavefunction_C = log_trial_wavefunction_C
        self.local_energy_ke_C = local_energy_ke_C

    def _build_coulomb_fns(self):
        """Build Coulomb interaction energy functions."""
        e_charges = self.e_charges
        Z_charges = self.Z_charges

        @jax.jit
        def classical_coulomb_energy(
            crds1, chgs1, crds2=None, chgs2=None
        ):
            """Coulomb interaction calculation."""
            eps = jnp.finfo(crds1.dtype).eps
            if crds2 is None:
                i, j = jnp.triu_indices(crds1.shape[0], k=1)
                diffs = crds1[i] - crds1[j]
                dists = jnp.sqrt(
                    jnp.sum(diffs * diffs, axis=-1) + eps
                )
                return jnp.sum(chgs1[i] * chgs1[j] / dists)
            elif chgs2 is not None:
                diffs = (
                    crds1[:, None, :] - crds2[None, :, :]
                )
                dists = jnp.sqrt(
                    jnp.sum(diffs * diffs, axis=-1) + eps
                )
                return jnp.sum(
                    chgs1[:, None] * chgs2[None, :] / dists
                )
            else:
                raise ValueError("chgs2 is None")

        @jax.jit
        def local_energy_ee(elec_crds):
            """Electron-electron energy."""
            return classical_coulomb_energy(
                elec_crds, e_charges
            )

        @jax.jit
        def local_energy_nn(nuc_crds):
            """Nuclear-nuclear energy."""
            return classical_coulomb_energy(
                nuc_crds, Z_charges
            )

        @jax.jit
        def local_energy_en(elec_crds, nuc_crds):
            """Electron-nuclear energy."""
            return classical_coulomb_energy(
                elec_crds, e_charges, nuc_crds, Z_charges
            )

        self.classical_coulomb_energy = classical_coulomb_energy
        self.local_energy_ee = local_energy_ee
        self.local_energy_nn = local_energy_nn
        self.local_energy_en = local_energy_en

    def _build_ke_fns(self):
        """Build kinetic energy function."""
        log_trial_wavefunction = self.log_trial_wavefunction

        @jax.jit
        def local_energy_ke(elec_crds, nuc_crds, curr_params):
            """Kinetic energy calculation."""
            def _log_psi_flat(p_flat):
                return log_trial_wavefunction(
                    p_flat.reshape(-1, 3),
                    nuc_crds, curr_params
                )
            grad_fn = jax.grad(_log_psi_flat)
            hess_fn = jax.hessian(_log_psi_flat)
            p_flat = elec_crds.flatten()
            grad_log_psi = grad_fn(p_flat)
            hess_log_psi = hess_fn(p_flat)
            lap_term = jnp.trace(hess_log_psi)
            grad_term_sq = jnp.sum(grad_log_psi**2)
            return -0.5 * (lap_term + grad_term_sq)

        self.local_energy_ke = local_energy_ke


def get_psi_fun(mf, params_cusp=None, trial=None,
                bspline_config=None):
    """
    Creates functions for evaluating the wavefunction
    and local energy components from a PySCF mean-field
    calculation.
    """
    obj = _PsiGTO(
        mf,
        params_cusp=params_cusp,
        trial=trial,
        bspline_config=bspline_config,
    )
    return (
        obj.log_trial_wavefunction,
        (obj.local_energy_ee, obj.local_energy_nn,
         obj.local_energy_en, obj.local_energy_ke),
        obj.get_psi_mo,
        (obj.log_trial_wavefunction_C,
         obj.local_energy_ke_C),
    )
