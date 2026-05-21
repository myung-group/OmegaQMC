"""Nuclear force gradient computation, storage, and
PGCS post-processing.

:func:`vmc_gto_gradients` / :func:`vmc_nn_gradients_zvzb`
build JIT-compiled callables that evaluate the
decomposed per-walker gradient components
(Hellmann-Feynman, kinetic, :math:`\\log|\\psi|`) for
GTO and NN trial wavefunctions respectively.

:func:`save_gto_gradients` / :func:`save_nn_gradients`
accumulate and write per-block gradient data
(reference + optional symmetry-related secondary
configurations) to HDF5 using a shared schema.

:func:`postproc_h5_pgcs` reads either HDF5 file and
applies Point Group Correlated Sampling (PGCS) to
obtain symmetry-averaged nuclear force estimates.
"""

import sys

import jax
import h5py
import numpy as np
import jax.numpy as jnp

from ..utils import (
    batched_binning_analysis_grds,
    blue_combine_states,
    compute_torque_with_error,
)

PSI2_RATIO_THRESHOLD = 1e-4


def vmc_gto_gradients(
    local_energy_ee,
    local_energy_en,
    local_energy_ke,
    log_trial_wavefunction,
    nuc_crds,
    params_corr,
    get_psi_mo_partition_vg,
    eps,
    gr_scheme,
    mo_relax=False,
    local_energy_ke_C=None,
    log_trial_wavefunction_C=None,
    C0=None,
    mo1s=None,
    num_nuc=None,
):
    """Build a JIT-compiled nuclear-gradient batch function.

    Constructs all intermediate gradient closures
    (Hellmann-Feynman, kinetic, Pulay, and optional
    MO-relaxation corrections) and returns a single
    JIT-compiled callable that evaluates them for a
    batch of walker positions.

    Parameters
    ----------
    local_energy_ee : callable
        Electron-electron energy ``(elec_crds) -> float``.
    local_energy_en : callable
        Electron-nuclear energy
        ``(elec_crds, nuc_crds) -> float``.
    local_energy_ke : callable
        Kinetic energy
        ``(elec_crds, nuc_crds, params) -> float``.
    log_trial_wavefunction : callable
        Log trial wavefunction
        ``(elec_crds, nuc_crds, params) -> float``.
    nuc_crds : jnp.ndarray
        Nuclear coordinates, shape ``(natom, 3)``.
    params_corr : pytree
        Jastrow / correlation parameters.
    get_psi_mo_partition_vg : callable
        Partitioned MO value+gradient evaluator
        ``(elec_crds, nuc_crds) -> (mo_val_s, mo_grad_s)``
        with shapes ``(n_nuc, n_e, n_mo)`` and
        ``(n_nuc, n_e, n_mo, 3)``.  Used by redistribution
        scheme 1 to compute closed-form value and
        electron-diagonal gradient of the SWCT weights.
    eps : float
        Machine epsilon for the coordinate dtype.
    gr_scheme : str
        ``'scheme1'`` (MO-based) or ``'scheme2'``
        (distance-based) space-warping redistribution.
    mo_relax : bool, optional
        Enable CPHF MO-relaxation correction.
        Default ``False``.
    local_energy_ke_C : callable, optional
        KE as a function of MO coefficients (for CPHF).
    log_trial_wavefunction_C : callable, optional
        log|ψ| as a function of MO coefficients.
    C0 : jnp.ndarray, optional
        Reference MO coefficient matrix.
    mo1s : jnp.ndarray, optional
        Orbital response tensors from CPHF,
        shape ``(natom, 3, nao, nocc)``.
    num_nuc : int, optional
        Number of nuclei (required when *mo_relax*
        is ``True``).

    Returns
    -------
    callable
        ``(batch_samples) -> (grd_ee, grd_en,
        grd_ke, grd_logpsi)`` where each component
        has shape ``(batch, natom, 3)``.
    """
    # --- Space-warping redistribution: closed-form value and
    # electron-diagonal gradient.
    #
    # Both schemes return ``(w, diag_grad)`` where
    #   w[e, n]              = SWCT weight (Σ_n w[e,n] = 1 ∀ e)
    #   diag_grad[e, n, K]   = ∂w[e,n] / ∂r[e,K]
    # The downstream code only ever consumes the per-electron
    # diagonal of the rescale Jacobian (see the
    # ``'benK->bnK'`` einsum in ``_vmc_gradient_batch``), so
    # off-diagonal entries are never built — that's the v6
    # memory win.  Off-diagonal entries are zero anyway by
    # chain rule (each electron's weight depends only on its
    # own coordinate), so this is exact, not an approximation.
    @jax.jit
    def _redistribute_scheme1_vg(elec_crds):
        mo_val_s, mo_grad_s = get_psi_mo_partition_vg(
            elec_crds, nuc_crds,
        )
        # Per-atom MO densities and totals.
        q = jnp.einsum('nem,nem->en', mo_val_s, mo_val_s)
        Q = jnp.sum(q, axis=-1, keepdims=True)
        w = q / Q
        # ∂q[e,n] / ∂r[e,K] = 2 Σ_o φ^n_{e,o} ∂φ^n_{e,o}/∂r_{e,K}
        dq = 2.0 * jnp.einsum(
            'nem,nemx->enx', mo_val_s, mo_grad_s,
        )
        # ∂Q[e]/∂r[e,K] = Σ_n ∂q[e,n]/∂r[e,K]
        dQ = jnp.sum(dq, axis=1, keepdims=True)
        # Quotient rule: ∂(q/Q) = (∂q − w · ∂Q) / Q
        diag_grad = (dq - w[..., None] * dQ) / Q[..., None]
        return w, diag_grad

    @jax.jit
    def _redistribute_scheme2_vg(elec_crds):
        diff = elec_crds[:, None, :] - nuc_crds[None, :, :]
        d2 = jnp.sum(diff * diff, axis=-1)
        d = jnp.sqrt(d2)
        d_safe = jnp.where(d < eps, eps, d)
        # g[e,n] = d^{-4}
        g = d_safe ** (-4.0)
        G = jnp.sum(g, axis=-1, keepdims=True)
        w = g / G
        # ∂g[e,n]/∂r[e,K] = −4 d^{-5} ê_K = −4 g · diff_K / d²
        # The inactive ``d < eps`` clamp zeroes the gradient
        # at coincident e–n positions (matches the original
        # scheme's behavior under autodiff of the clamp).
        active = (d >= eps)
        inv_d2 = jnp.where(active, 1.0 / (d_safe * d_safe), 0.0)
        dg = -4.0 * g[..., None] * diff * inv_d2[..., None]
        dG = jnp.sum(dg, axis=1, keepdims=True)
        diag_grad = (dg - w[..., None] * dG) / G[..., None]
        return w, diag_grad

    rescale_value_and_diag_grad_fn = (
        _redistribute_scheme2_vg
        if 'scheme2' in gr_scheme
        else _redistribute_scheme1_vg
    )

    # --- Per-walker gradient functions ---
    @jax.jit
    def _grad_fn_ee(e_pos):
        return jax.grad(local_energy_ee)(e_pos)

    @jax.jit
    def _grad_fn_en(e_pos):
        return jax.grad(
            local_energy_en, argnums=(0, 1),
        )(e_pos, nuc_crds)

    @jax.jit
    def _grad_fn_ke(e_pos):
        return jax.grad(
            local_energy_ke, argnums=(0, 1),
        )(e_pos, nuc_crds, params_corr)

    @jax.jit
    def _grad_fn_logpsi(e_pos):
        return jax.grad(
            log_trial_wavefunction, argnums=(0, 1),
        )(e_pos, nuc_crds, params_corr)

    if mo_relax:
        @jax.jit
        def _grad_fn_ke_mo(e_pos):
            """dE_ke/dC · dC/dR via JVP."""
            def ke_of_C(C):
                return local_energy_ke_C(
                    e_pos, nuc_crds, params_corr, C,
                )
            results = jnp.zeros((num_nuc, 3))
            for ia in range(num_nuc):
                for K in range(3):
                    _, dke = jax.jvp(
                        ke_of_C,
                        (C0,), (mo1s[ia, K],),
                    )
                    results = results.at[ia, K].set(dke)
            return results

        @jax.jit
        def _grad_fn_logpsi_mo(e_pos):
            """dlog|ψ|/dC · dC/dR via JVP."""
            def logpsi_of_C(C):
                return log_trial_wavefunction_C(
                    e_pos, nuc_crds, params_corr, C,
                )
            results = jnp.zeros((num_nuc, 3))
            for ia in range(num_nuc):
                for K in range(3):
                    _, dlp = jax.jvp(
                        logpsi_of_C,
                        (C0,), (mo1s[ia, K],),
                    )
                    results = results.at[ia, K].set(dlp)
            return results

    # --- Batched gradient function ---
    @jax.jit
    def _vmc_gradient_batch(batch_samples):
        grd_ee_elc = jax.vmap(
            _grad_fn_ee,
        )(batch_samples)
        grd_en_elc, grd_en_nuc = jax.vmap(
            _grad_fn_en,
        )(batch_samples)
        grd_ke_elc, grd_ke_nuc = jax.vmap(
            _grad_fn_ke,
        )(batch_samples)
        grd_logpsi_elc, grd_logpsi_nuc = jax.vmap(
            _grad_fn_logpsi,
        )(batch_samples)

        rescale, diag_rescale_elc = jax.vmap(
            rescale_value_and_diag_grad_fn,
        )(batch_samples)
        novel_correction = 0.5 * jnp.einsum(
            'benK->bnK', diag_rescale_elc,
        )

        grd_ee = jnp.einsum(
            'beK,ben->bnK', grd_ee_elc, rescale,
        )
        grd_en = grd_en_nuc + jnp.einsum(
            'beK,ben->bnK', grd_en_elc, rescale,
        )
        grd_ke = grd_ke_nuc + jnp.einsum(
            'beK,ben->bnK', grd_ke_elc, rescale,
        )

        grd_logpsi = grd_logpsi_nuc + jnp.einsum(
            'beK,ben->bnK',
            grd_logpsi_elc, rescale,
        )
        grd_logpsi += novel_correction

        if mo_relax:
            grd_ke_mo_batch = jax.vmap(
                _grad_fn_ke_mo,
            )(batch_samples)
            grd_logpsi_mo_batch = jax.vmap(
                _grad_fn_logpsi_mo,
            )(batch_samples)

            grd_ke = grd_ke + grd_ke_mo_batch
            grd_logpsi = grd_logpsi + grd_logpsi_mo_batch

        return grd_ee, grd_en, grd_ke, grd_logpsi

    return _vmc_gradient_batch


def _build_param_response_fns(
    log_trial_wavefunction,
    local_energy_ke,
    local_energy_en,
    nuc_crds,
    params_corr,
):
    """Build per-walker Jastrow parameter-response
    derivative functions for the ZVZB2 estimator.

    Returns a JIT-compiled batch function and the
    number of flattened Jastrow parameters.

    The returned callable evaluates, for each walker:

    - ``O_flat``: :math:`\\partial\\log|\\psi|/
      \\partial p_i` (flattened Jastrow param grad)
    - ``dEL_dp_flat``: :math:`\\partial E_L/
      \\partial p_i` (flattened; only KE contributes)
    - ``dEL_dR_nuc``: nuclear-only
      :math:`\\partial E_L/\\partial R` (no SWCT)
    - ``dlogpsi_dR_nuc``: nuclear-only
      :math:`\\partial\\log|\\psi|/\\partial R`
      (no SWCT)

    Parameters
    ----------
    log_trial_wavefunction : callable
        ``(elec_crds, nuc_crds, params) -> float``
    local_energy_ke : callable
        ``(elec_crds, nuc_crds, params) -> float``
    local_energy_en : callable
        ``(elec_crds, nuc_crds) -> float``
    nuc_crds : jnp.ndarray
        Nuclear coordinates, shape ``(natom, 3)``.
    params_corr : pytree
        Jastrow / correlation parameters.

    Returns
    -------
    param_response_batch : callable
        ``(walkers) -> (O_flat, dEL_dp_flat,
        dEL_dR_nuc, dlogpsi_dR_nuc)``
        where shapes are ``(batch, n_flat)``,
        ``(batch, n_flat)``,
        ``(batch, natom, 3)``,
        ``(batch, natom, 3)``.
    n_flat : int
        Number of flattened Jastrow parameters.
    """
    from jax.flatten_util import ravel_pytree

    flat_p, _ = ravel_pytree(params_corr)
    n_flat = flat_p.shape[0]

    def _single_walker(e_pos):
        # dlog|psi|/dp_i
        O_tree = jax.grad(
            log_trial_wavefunction, argnums=2,
        )(e_pos, nuc_crds, params_corr)
        O_flat, _ = ravel_pytree(O_tree)

        # dE_L/dp_i (only KE depends on params)
        dEL_tree = jax.grad(
            local_energy_ke, argnums=2,
        )(e_pos, nuc_crds, params_corr)
        dEL_flat, _ = ravel_pytree(dEL_tree)

        # Nuclear-only dE_L/dR (no SWCT)
        dEL_dR = jax.grad(
            local_energy_ke, argnums=1,
        )(e_pos, nuc_crds, params_corr)
        dEL_dR = dEL_dR + jax.grad(
            local_energy_en, argnums=1,
        )(e_pos, nuc_crds)

        # Nuclear-only dlog|psi|/dR (no SWCT)
        dlogpsi_dR = jax.grad(
            log_trial_wavefunction, argnums=1,
        )(e_pos, nuc_crds, params_corr)

        return (
            O_flat, dEL_flat,
            dEL_dR, dlogpsi_dR,
        )

    @jax.jit
    def param_response_batch(walkers):
        return jax.vmap(_single_walker)(walkers)

    return param_response_batch, n_flat


def vmc_nn_gradients_zvzb(
    log_psi,
    nuc_crds,
    charges,
    nelec,
    params,
    lap_grad=None,
):
    """Build a JIT-compiled ZVZB gradient batch function.

    Implements the zero-variance zero-bias (ZVZB)
    nuclear force estimator of Assaraf & Caffarel
    :cite:`Assaraf2003`, but returns the
    *negated* force decomposed into the same three
    per-walker components produced by
    :func:`vmc_gto_gradients`, so that both backends
    feed the same downstream postprocessing
    (:func:`postproc_h5_pgcs`).

    Per-walker decomposition (no E-dependence):

    * ``grd_ee_en = -F_en``: bare nuclear-electron
      Coulomb-potential gradient (the ee piece is
      identically zero — V_ee has no nuclear
      coordinate dependence — and is folded in only
      to keep the GTO-side group name).
    * ``grd_ke = (KE_dpsi - KE_psi) * grad_lp``: ZV
      kinetic correction.  Uses the identity
      :math:`\\log|\\partial\\psi/\\partial R_{ia,k}|
      = \\log|\\psi| + \\log|h_{ia,k}|` with
      :math:`h = \\partial\\log|\\psi|/\\partial R`
      (product rule), so the same autodiff-Laplacian
      machinery is reused for every ``(ia, k)``.
    * ``grd_logpsi = grad_lp``: nuclear gradient of
      :math:`\\log|\\psi|`; consumed downstream as
      the Pulay/ZB coefficient
      :math:`2(E_L - \\bar E)\\,\\nabla_R\\log|\\psi|`.

    The nuclear-nuclear gradient is stored separately
    by the driver (``grd_nn``) and added in the
    postprocessing sum.

    Parameters
    ----------
    log_psi : callable
        ``(elec_crds, nuc_crds, params) -> float``.
        Log absolute value of the NN trial
        wavefunction for a single walker.
    nuc_crds : jnp.ndarray
        Nuclear coordinates, shape ``(natom, 3)``.
    charges : jnp.ndarray
        Nuclear charges, shape ``(natom,)``.
    nelec : int
        Total number of electrons.
    params : pytree
        NN wavefunction parameters (NNX State).
    lap_grad : callable, optional
        ``(elec_crds, nuc_crds, params) -> (lap_x, grad_x)``
        returning the electron-coord Laplacian and gradient
        of ``log|ψ|`` for a single walker.  If supplied and
        carrying ``lap_grad.use_vgl == True`` (as produced by
        the analytic-VGL path of :func:`make_nn_log_psi`), the
        per-``(ia, k)`` ``fori_loop`` over
        ``ke_dpsi - ke_psi`` is replaced by a single
        ``jax.jacfwd``-over-nuclei pass on
        :math:`(\\nabla_x\\log|\\psi|, \\Delta_x\\log|\\psi|)`,
        using the identity (derived from
        :math:`q = \\log|\\psi| + \\log|h_{ia,k}|` with
        :math:`h = \\nabla_R\\log|\\psi|`)

        .. math::
           \\mathrm{grd\\_ke}_{ia,k}
           = -\\tfrac{1}{2}\\,\\Delta_x h_{ia,k}
             - \\nabla_x\\log|\\psi|\\cdot\\nabla_x h_{ia,k},

        in which the :math:`1/h_{ia,k}` factor cancels
        against the outer :math:`\\nabla_R\\log|\\psi|` factor
        in :math:`(\\mathrm{KE}_{d\\psi}-\\mathrm{KE}_\\psi)\\,
        \\nabla_R\\log|\\psi|`.  When omitted or with
        ``use_vgl == False``, the original
        ``laplacian``-per-component path is used.

    Returns
    -------
    callable
        ``(batch_walkers) -> (grd_ee_en, grd_ke,
        grd_logpsi)`` where *batch_walkers* has shape
        ``(batch, nelec, 3)`` and each output has
        shape ``(batch, natom, 3)``.
    """
    from ..psi.nn.physics import laplacian

    use_vgl = bool(getattr(lap_grad, 'use_vgl', False))

    n_nuc = len(charges)
    charges_elec = -jnp.ones(nelec)

    # --- Coulomb force utility ---
    def _coulomb_force(
        r1, r2, c1, c2, remove_self=False,
    ):
        """Pairwise Coulomb force on particles in r1.

        ``F_i = sum_j c_i c_j (r_i - r_j)/|r_i-r_j|^3``
        """
        diffs = r1[:, None] - r2[None]
        norms = jnp.linalg.norm(
            diffs, axis=-1, keepdims=True,
        )
        force = (
            (c1[:, None] * c2[None])[..., None]
            * diffs
            / norms ** 3
        )
        if remove_self:
            n = len(r1)
            mask = ~jnp.eye(n, dtype=bool)
            force = force * mask[..., None]
        return force.sum(-2)

    # --- Nuclear-electron HF force (single walker) ---
    @jax.jit
    def _force_en_bare(elec_crds):
        return _coulomb_force(
            nuc_crds, elec_crds,
            charges, charges_elec, False,
        )

    # --- Grad of log|psi| w.r.t. nuclear coords ---
    @jax.jit
    def _grad_nuc_log_psi(elec_crds):
        return jax.grad(
            log_psi, argnums=1,
        )(elec_crds, nuc_crds, params)

    # --- Kinetic energy of psi (single walker) ---
    @jax.jit
    def _ke_psi(elec_crds):
        def _log_psi_flat(r_flat):
            r = r_flat.reshape(nelec, 3)
            return log_psi(r, nuc_crds, params)
        r_flat = elec_crds.reshape(-1)
        lap_fn = laplacian(_log_psi_flat)
        lap_val, grad_val = lap_fn(r_flat)
        return -0.5 * (
            lap_val + jnp.dot(grad_val, grad_val)
        )

    # --- KE of dpsi/dR_{ij} for one (i,j) ---
    def _ke_dpsi_component(elec_crds, ia, k):
        r"""KE of `\partial\psi/\partial R_{ia,k}`.

        Uses the identity
        `\log|\partial\psi/\partial R_{ia,k}|`
        `= \log|\psi| + \log|h_{ia,k}|`
        where `h = \partial\log|\psi|/\partial R`.
        """
        def _log_abs_dpsi(r_flat):
            r = r_flat.reshape(nelec, 3)
            lp = log_psi(r, nuc_crds, params)
            h = jax.grad(
                log_psi, argnums=1,
            )(r, nuc_crds, params)[ia, k]
            return lp + jnp.log(jnp.abs(h))
        r_flat = elec_crds.reshape(-1)
        lap_fn = laplacian(_log_abs_dpsi)
        lap_val, grad_val = lap_fn(r_flat)
        return -0.5 * (
            lap_val + jnp.dot(grad_val, grad_val)
        )

    # --- VGL-based ZV kinetic correction ---
    # Replaces the ``ke_dpsi - ke_psi`` fori_loop with a
    # single ``jax.jacfwd`` over nuclear coordinates on the
    # ``(grad_x_lp, lap_x_lp)`` pair returned by ``lap_grad``.
    # The mixed Hessian/Laplacian-of-grad identity above lets
    # the per-(ia, k) Laplacian be computed once for all
    # nuclear components.
    if use_vgl:
        def _grd_ke_vgl(elec_crds):
            def _lap_grad_of_R(R):
                return lap_grad(elec_crds, R, params)
            (lap_x_lp,
             grad_x_lp) = lap_grad(elec_crds, nuc_crds, params)
            (lap_x_h,
             grad_x_h) = jax.jacfwd(_lap_grad_of_R)(nuc_crds)
            # lap_x_h shape (n_nuc, 3); grad_x_h shape
            # (D, n_nuc, 3); grad_x_lp shape (D,).
            return -0.5 * lap_x_h - jnp.einsum(
                'd,dab->ab', grad_x_lp, grad_x_h,
            )

    # --- Decomposed ZVZB gradient (single walker) ---
    @jax.jit
    def _grd_zvzb(elec_crds):
        f_en = _force_en_bare(elec_crds)
        grad_lp = _grad_nuc_log_psi(elec_crds)

        if use_vgl:
            # VGL fast path: compute the ZV kinetic
            # correction directly from
            # ``(∇_x lp, ∇_x h, Δ_x h)`` (h = ∇_R lp).
            grd_ke = _grd_ke_vgl(elec_crds)
        else:
            ke_psi = _ke_psi(elec_crds)

            # KE of dpsi/dR_{ia,k} for all (ia, k)
            def body_fn(idx, val):
                ia = idx // 3
                k = idx % 3
                ke_d = _ke_dpsi_component(
                    elec_crds, ia, k,
                )
                return val.at[ia, k].set(ke_d)

            ke_dpsi_all = jax.lax.fori_loop(
                0, n_nuc * 3, body_fn,
                jnp.zeros((n_nuc, 3)),
            )
            # ZV kinetic correction (gradient sign)
            grd_ke = (ke_dpsi_all - ke_psi) * grad_lp

        # gradient = -force; bare HF: grd_ee_en = -f_en
        # (V_ee has no R-dependence, so the ee piece
        # is identically zero)
        grd_ee_en = -f_en
        # Pulay/ZB coefficient: stored bare, multiplied
        # by 2*(E_L - <E>) downstream
        grd_logpsi = grad_lp
        return grd_ee_en, grd_ke, grd_logpsi

    # --- Batched version ---
    @jax.jit
    def _grd_zvzb_batch(batch_walkers):
        return jax.vmap(_grd_zvzb)(batch_walkers)

    return _grd_zvzb_batch


def save_nn_gradients(
    block_cnt,
    sampled_walkers,
    local_energies,
    batch_size,
    num_batches,
    single_frag_combos,
    ofname_grd,
    nn_gradient_batch,
    log_psi_batch=None,
    local_energy_batch=None,
    apply_single_frag_symmop=None,
):
    """Save per-block NN ZVZB gradient data.

    Evaluates the decomposed ZVZB gradient
    components built by
    :func:`vmc_nn_gradients_zvzb` on batches of walker
    positions and writes them to the gradient HDF5
    file using the same schema as
    :func:`save_gto_gradients`, so that
    :func:`postproc_h5_pgcs` works on either backend.

    HDF5 layout (per block ``<blk>``):

    * ``grd_ee_en/<blk>``, ``grd_ke/<blk>``,
      ``grd_logpsi/<blk>``: per-walker gradient
      components, shape ``(num_samples, natom, 3)``.
    * ``local_energies/<blk>``: local energies, shape
      ``(num_steps_per_block, num_walkers)`` so the
      downstream postproc can recover the (steps,
      walkers) layout.
    * ``grd_ee_en/<label>/<blk>``,
      ``grd_ke/<label>/<blk>``,
      ``grd_logpsi/<label>/<blk>``,
      ``local_energies/<label>/<blk>``,
      ``fragment_weights/<label>/<blk>``: per-combo
      secondary data, written when
      *single_frag_combos* is non-empty.

    Block-level restart logic deletes any existing
    datasets at ``<blk>`` before writing.

    Parameters
    ----------
    block_cnt : int
        Current production-block index.
    sampled_walkers : jnp.ndarray
        Walker positions, shape
        ``(num_samples, nelec, 3)`` where
        ``num_samples == num_steps_per_block *
        num_walkers``.
    local_energies : jnp.ndarray
        Local energies, shape
        ``(num_steps_per_block, num_walkers)``.
    batch_size : int
        Number of walkers per gradient batch.
    num_batches : int
        Number of batches.
    single_frag_combos : iterable
        Sequence of ``(frag_pos, op, label)`` tuples
        describing fragment symmetry operations.
        Empty for the current NN driver.
    ofname_grd : str
        Path to the gradient HDF5 file.
    nn_gradient_batch : callable
        JIT-compiled ``(walkers) -> (grd_ee_en,
        grd_ke, grd_logpsi)`` with each output shape
        ``(batch, natom, 3)``.
    log_psi_batch, local_energy_batch,
    apply_single_frag_symmop : callable, optional
        Required only when *single_frag_combos* is
        non-empty (combo screening + secondary
        evaluation).

    Returns
    -------
    dict
        Per-combo weighted-mean block energies, or
        empty dict if *single_frag_combos* is empty.
    """
    num_samples = sampled_walkers.shape[0]
    has_combos = bool(single_frag_combos)
    if has_combos:
        from ..symm.operations import (
            symmetry_operations_map,
        )
        if (
            apply_single_frag_symmop is None
            or log_psi_batch is None
            or local_energy_batch is None
        ):
            raise ValueError(
                "single_frag_combos requires "
                "apply_single_frag_symmop, "
                "log_psi_batch, and "
                "local_energy_batch."
            )

    # Reference accumulators
    w_grd_ee_en = []
    w_grd_ke = []
    w_grd_logpsi = []

    # Per-combo accumulators
    combo_grd_ee_en = {
        label: [] for _, _, label in single_frag_combos
    }
    combo_grd_ke = {
        label: [] for _, _, label in single_frag_combos
    }
    combo_grd_logpsi = {
        label: [] for _, _, label in single_frag_combos
    }
    combo_weights = {
        label: [] for _, _, label in single_frag_combos
    }
    combo_E_local = {
        label: [] for _, _, label in single_frag_combos
    }

    for batch_idx in range(num_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, num_samples)
        batch_w = sampled_walkers[start:end]

        g_ee_en, g_ke, g_lp = nn_gradient_batch(batch_w)
        w_grd_ee_en.append(g_ee_en)
        w_grd_ke.append(g_ke)
        w_grd_logpsi.append(g_lp)

        if has_combos:
            log_psi_orig = log_psi_batch(batch_w)
            for frag_pos, op, label in single_frag_combos:
                batch_trans = apply_single_frag_symmop(
                    batch_w, frag_pos,
                    symmetry_operations_map[op],
                )
                log_psi_trans = log_psi_batch(batch_trans)
                psi2_ratio = jnp.exp(
                    2.0 * (log_psi_trans - log_psi_orig)
                )
                safe = (psi2_ratio > PSI2_RATIO_THRESHOLD)
                batch_trans = jnp.where(
                    safe[:, None, None],
                    batch_trans, batch_w,
                )
                weight = jnp.where(safe, psi2_ratio, 1.0)

                g_ee_en_t, g_ke_t, g_lp_t = (
                    nn_gradient_batch(batch_trans)
                )
                combo_grd_ee_en[label].append(g_ee_en_t)
                combo_grd_ke[label].append(g_ke_t)
                combo_grd_logpsi[label].append(g_lp_t)
                combo_weights[label].append(weight)

                E_trans = local_energy_batch(batch_trans)
                combo_E_local[label].append(E_trans)

    w_grd_ee_en = jnp.vstack(w_grd_ee_en)
    w_grd_ke = jnp.vstack(w_grd_ke)
    w_grd_logpsi = jnp.vstack(w_grd_logpsi)

    with h5py.File(ofname_grd, 'a') as f:
        block_cnt_str = f'{block_cnt}'

        grp_names = [
            'grd_ee_en', 'grd_ke',
            'grd_logpsi', 'local_energies',
        ]
        if has_combos:
            grp_names.append('fragment_weights')
        for k in grp_names:
            if k not in f.keys():
                f.create_group(k)

        # Restart cleanup
        if block_cnt_str in f['grd_ee_en'].keys():
            del f['grd_ee_en'][block_cnt_str]
            del f['grd_ke'][block_cnt_str]
            del f['grd_logpsi'][block_cnt_str]
            del f['local_energies'][block_cnt_str]
            for _, _, label in single_frag_combos:
                if (
                    label in f['grd_ee_en']
                    and block_cnt_str
                    in f['grd_ee_en'][label]
                ):
                    del f['grd_ee_en'][label][
                        block_cnt_str
                    ]
                    del f['grd_ke'][label][block_cnt_str]
                    del f['grd_logpsi'][label][
                        block_cnt_str
                    ]
                    del f['fragment_weights'][label][
                        block_cnt_str
                    ]
                if (
                    label in f['local_energies']
                    and block_cnt_str
                    in f['local_energies'][label]
                ):
                    del f['local_energies'][label][
                        block_cnt_str
                    ]

        # A. Reference gradients
        f['grd_ee_en'].create_dataset(
            block_cnt_str, data=w_grd_ee_en,
        )
        f['grd_ke'].create_dataset(
            block_cnt_str, data=w_grd_ke,
        )
        f['grd_logpsi'].create_dataset(
            block_cnt_str, data=w_grd_logpsi,
        )
        f['local_energies'].create_dataset(
            block_cnt_str, data=local_energies,
        )

        # B. Per-combo secondary data
        for _, _, label in single_frag_combos:
            c_ee_en = jnp.vstack(
                combo_grd_ee_en[label],
            )
            c_ke = jnp.vstack(combo_grd_ke[label])
            c_logpsi = jnp.vstack(
                combo_grd_logpsi[label],
            )
            c_w = jnp.concatenate(combo_weights[label])
            c_E = jnp.concatenate(combo_E_local[label])

            for grp, data in [
                ('grd_ee_en', c_ee_en),
                ('grd_ke', c_ke),
                ('grd_logpsi', c_logpsi),
            ]:
                if label not in f[grp]:
                    f[grp].create_group(label)
                f[grp][label].create_dataset(
                    block_cnt_str, data=data,
                )

            if label not in f['fragment_weights']:
                f['fragment_weights'].create_group(label)
            f['fragment_weights'][label].create_dataset(
                block_cnt_str, data=c_w,
            )

            if label not in f['local_energies']:
                f['local_energies'].create_group(label)
            f['local_energies'][label].create_dataset(
                block_cnt_str, data=c_E,
            )

    # Per-combo weighted-mean block energies
    combo_block_E = {}
    for _, _, label in single_frag_combos:
        c_E = jnp.concatenate(combo_E_local[label])
        w = jnp.concatenate(combo_weights[label])
        combo_block_E[label] = float(
            jnp.sum(w * c_E) / jnp.sum(w)
        )
    return combo_block_E


def save_gto_gradients(
    block_cnt,
    sampled_walkers,
    local_energies,
    batch_size,
    num_batches,
    single_frag_combos,
    ofname_grd,
    vmc_gradient_batch,
    log_psi_batch,
    local_energy_batch,
    apply_single_frag_symmop,
    param_response_batch=None,
):
    """Save per-block nuclear-force gradient data.

    Evaluates gradient components (Hellmann-Feynman,
    kinetic, Pulay) at the reference walker positions
    and at each symmetry-related secondary
    configuration, then writes everything to an HDF5
    file.

    This is a free-function equivalent of the former
    ``_VMCDriverGTO._gradient_save`` method.  All
    JIT-compiled kernels are passed in explicitly,
    so no performance is lost.

    Args:
        block_cnt: Current production-block index.
        sampled_walkers: Walker positions, shape
            ``(num_steps_per_block * num_walkers,
            nelec, 3)``.
        local_energies: Local energies, shape
            ``(num_steps_per_block, num_walkers)``.
        batch_size: Walkers per gradient batch.
        num_batches: Number of batches.
        single_frag_combos: List of
            ``(frag_pos, op, label)`` tuples
            describing fragment symmetry operations.
        ofname_grd: Path to the gradient HDF5 file.
        vmc_gradient_batch: JIT-compiled callable
            ``(walkers) -> (g_ee, g_en, g_ke,
            g_logpsi)``.
        log_psi_batch: JIT-compiled callable
            ``(walkers) -> log|ψ|`` (batched).
        local_energy_batch: JIT-compiled callable
            ``(walkers) -> E_local`` (batched).
        apply_single_frag_symmop: JIT-compiled
            callable
            ``(walkers, frag_pos, op_matrix)
            -> transformed_walkers``.
        param_response_batch : callable or None
            When not ``None``, a JIT-compiled callable
            ``(walkers) -> (O_flat, dEL_dp_flat,
            dEL_dR_nuc, dlogpsi_dR_nuc)`` providing
            per-walker Jastrow parameter derivatives
            for the ZVZB2 estimator.

    Returns:
        Dict mapping combo labels to their
        weighted-mean block energies, or empty dict
        if *single_frag_combos* is empty.
    """
    from ..symm.operations import (
        symmetry_operations_map,
    )

    num_samples_per_block = sampled_walkers.shape[0]

    # Reference gradient accumulators
    w_grd_ee_en = []
    w_grd_ke = []
    w_grd_logpsi = []

    # Parameter-response accumulators (ZVZB2)
    w_O_params = []
    w_dEL_dparams = []
    w_dEL_dR_nuc = []
    w_dlogpsi_dR_nuc = []

    # Per-combo accumulators
    combo_grd_ee_en = {label: [] for _, _, label in single_frag_combos}
    combo_grd_ke = {label: [] for _, _, label in single_frag_combos}
    combo_grd_logpsi = {label: [] for _, _, label in single_frag_combos}
    combo_weights = {label: [] for _, _, label in single_frag_combos}
    combo_E_local = {label: [] for _, _, label in single_frag_combos}

    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(
            start_idx + batch_size,
            num_samples_per_block,
        )
        batch_orig = (sampled_walkers[start_idx:end_idx, :, :])

        # Reference gradients
        g_ee, g_en, g_ke, g_logpsi = (vmc_gradient_batch(batch_orig))
        w_grd_ee_en.append(g_ee + g_en)
        w_grd_ke.append(g_ke)
        w_grd_logpsi.append(g_logpsi)

        # Parameter-response derivatives (ZVZB2)
        if param_response_batch is not None:
            O_f, dEL_f, dEL_R, dlp_R = param_response_batch(batch_orig)
            w_O_params.append(O_f)
            w_dEL_dparams.append(dEL_f)
            w_dEL_dR_nuc.append(dEL_R)
            w_dlogpsi_dR_nuc.append(dlp_R)

        # log|ψ| at original positions (once)
        log_psi_orig = log_psi_batch(batch_orig)

        # Single-fragment symmetry combos
        for frag_pos, op, label in single_frag_combos:
            batch_trans = apply_single_frag_symmop(
                batch_orig, frag_pos,
                symmetry_operations_map[op],
            )

            # Screen: fall back where |ψ|² drops
            log_psi_trans = log_psi_batch(
                batch_trans,
            )
            psi2_ratio = jnp.exp(2.0 * (log_psi_trans - log_psi_orig))
            safe = (psi2_ratio > PSI2_RATIO_THRESHOLD)
            batch_trans = jnp.where(
                safe[:, None, None],
                batch_trans, batch_orig,
            )

            # Weight: J * |ψ(r')|²/|ψ(r)|²
            weight = jnp.where(
                safe, psi2_ratio, 1.0,
            )

            g_ee, g_en, g_ke, g_logpsi = (vmc_gradient_batch(batch_trans))
            combo_grd_ee_en[label].append(
                g_ee + g_en,
            )
            combo_grd_ke[label].append(g_ke)
            combo_grd_logpsi[label].append(
                g_logpsi,
            )
            combo_weights[label].append(weight)

            E_trans = local_energy_batch(
                batch_trans,
            )
            combo_E_local[label].append(E_trans)

    # Stack all batches
    w_grd_ee_en = jnp.vstack(w_grd_ee_en)
    w_grd_ke = jnp.vstack(w_grd_ke)
    w_grd_logpsi = jnp.vstack(w_grd_logpsi)

    has_pr = param_response_batch is not None
    if has_pr:
        w_O_params = jnp.vstack(w_O_params)
        w_dEL_dparams = jnp.vstack(w_dEL_dparams)
        w_dEL_dR_nuc = jnp.vstack(w_dEL_dR_nuc)
        w_dlogpsi_dR_nuc = jnp.vstack(
            w_dlogpsi_dR_nuc,
        )

    # Save to HDF5
    with h5py.File(ofname_grd, "a") as f:
        block_cnt_str = f'{block_cnt}'

        grp_names = [
            'grd_ee_en', 'grd_ke',
            'grd_logpsi', 'local_energies',
        ]
        if has_pr:
            grp_names += [
                'O_params', 'dEL_dparams',
                'dEL_dR_nuc', 'dlogpsi_dR_nuc',
            ]
        if single_frag_combos:
            grp_names.append('fragment_weights')
        for k in grp_names:
            if k not in f.keys():
                f.create_group(k)

        # Clean up existing block (restart)
        if block_cnt_str in f['grd_ee_en'].keys():
            del f['grd_ee_en'][block_cnt_str]
            del f['grd_ke'][block_cnt_str]
            del f['grd_logpsi'][block_cnt_str]
            del f['local_energies'][block_cnt_str]
            if has_pr:
                for prk in [
                    'O_params', 'dEL_dparams',
                    'dEL_dR_nuc', 'dlogpsi_dR_nuc',
                ]:
                    if prk in f and block_cnt_str in f[prk]:
                        del f[prk][block_cnt_str]
            for _, _, label in single_frag_combos:
                if label in f['grd_ee_en'] \
                        and block_cnt_str in f['grd_ee_en'][label]:
                    del f['grd_ee_en'][label][block_cnt_str]
                    del f['grd_ke'][label][block_cnt_str]
                    del f['grd_logpsi'][label][block_cnt_str]
                    del f['fragment_weights'][label][block_cnt_str]
                if (
                    label in f['local_energies']
                    and block_cnt_str in f['local_energies'][label]
                ):
                    del f['local_energies'][label][block_cnt_str]

        # A. Reference gradients
        f['grd_ee_en'].create_dataset(
            block_cnt_str, data=w_grd_ee_en,
        )
        f['grd_ke'].create_dataset(
            block_cnt_str, data=w_grd_ke,
        )
        f['grd_logpsi'].create_dataset(
            block_cnt_str, data=w_grd_logpsi,
        )
        f['local_energies'].create_dataset(
            block_cnt_str, data=local_energies,
        )

        # A2. Parameter-response data (ZVZB2)
        if has_pr:
            f['O_params'].create_dataset(
                block_cnt_str, data=w_O_params,
            )
            f['dEL_dparams'].create_dataset(
                block_cnt_str, data=w_dEL_dparams,
            )
            f['dEL_dR_nuc'].create_dataset(
                block_cnt_str, data=w_dEL_dR_nuc,
            )
            f['dlogpsi_dR_nuc'].create_dataset(
                block_cnt_str,
                data=w_dlogpsi_dR_nuc,
            )

        # B. Per-combo secondary gradients/weights
        for _, _, label in single_frag_combos:
            c_ee_en = jnp.vstack(
                combo_grd_ee_en[label],
            )
            c_ke = jnp.vstack(
                combo_grd_ke[label],
            )
            c_logpsi = jnp.vstack(
                combo_grd_logpsi[label],
            )
            c_w = jnp.concatenate(
                combo_weights[label],
            )

            for grp, data in [
                ('grd_ee_en', c_ee_en),
                ('grd_ke', c_ke),
                ('grd_logpsi', c_logpsi),
            ]:
                if label not in f[grp]:
                    f[grp].create_group(label)
                f[grp][label].create_dataset(
                    block_cnt_str, data=data,
                )

            if label not in f['fragment_weights']:
                f['fragment_weights'].create_group(
                    label,
                )
            f['fragment_weights'][label].create_dataset(
                block_cnt_str, data=c_w,
            )

            c_E = jnp.concatenate(
                combo_E_local[label],
            )
            if label not in f['local_energies']:
                f['local_energies'].create_group(
                    label,
                )
            f['local_energies'][label].create_dataset(
                block_cnt_str, data=c_E,
            )

    # Return per-combo weighted-mean block energies
    combo_weights_all = {
        label: jnp.concatenate(
            combo_weights[label],
        )
        for _, _, label in single_frag_combos
    }
    combo_block_E = {}
    for _, _, label in single_frag_combos:
        c_E = jnp.concatenate(
            combo_E_local[label],
        )
        w = combo_weights_all[label]
        combo_block_E[label] = float(jnp.sum(w * c_E) / jnp.sum(w))
    return combo_block_E


def _solve_param_response(
    f, block_nums, enr_mean,
):
    """Estimate Jastrow parameter response dp/dR.

    Uses analytical linear response at the VMC
    optimum: ``dp/dR = -H_pp^{-1} · H_pR`` where
    ``H_pp`` is the energy Hessian w.r.t. Jastrow
    parameters and ``H_pR`` is the mixed
    parameter–nuclear second derivative.

    Leading-order estimators:

    .. math::

        H_{pp}[i,j] \\approx 2\\,\\mathrm{Cov}(
            E_L^{(i)}, O_j) + 2\\,\\mathrm{Cov}(
            E_L^{(j)}, O_i)

        H_{pR}[i,A,k] \\approx 2\\,\\mathrm{Cov}(
            \\partial E_L/\\partial R_{A,k}, O_i)
            + 2\\,\\mathrm{Cov}(E_L^{(i)},
            \\partial\\log|\\psi|/\\partial R_{A,k})

    Per-block walker-axis covariance moments are
    stream-accumulated so peak host RAM stays
    O(per-block) instead of O(total-walkers).

    Parameters
    ----------
    f : h5py.File
        Open HDF5 file handle for the gradient data.
    block_nums : list of int
        Sorted list of block indices.
    enr_mean : float
        Global mean local energy.

    Returns
    -------
    dp_dR : jnp.ndarray, shape (n_params, natom, 3)
        Parameter response vector, or ``None`` if
        the Hessian is singular.
    """
    # Stream over blocks: accumulate per-walker sums
    # and sums of products (the moments that determine
    # the covariances).  Each block array sees only
    # this loop iteration.
    s_O = None
    s_dEL_dp = None
    s_dEL_dR = None
    s_dlp_dR = None
    # Outer-product accumulators:
    s_dEL_dp_O = None       # (n_p, n_p)
    s_dEL_dR_O = None       # (n_nuc, 3, n_p)
    s_dEL_dp_dlp_dR = None  # (n_p, n_nuc, 3)
    n_w_total = 0

    for bc in block_nums:
        bcs = f'{bc}'
        O = np.asarray(
            f['O_params'][bcs][()], dtype=np.float64,
        )
        dEL_dp = np.asarray(
            f['dEL_dparams'][bcs][()], dtype=np.float64,
        )
        dEL_dR = np.asarray(
            f['dEL_dR_nuc'][bcs][()], dtype=np.float64,
        )
        dlp_dR = np.asarray(
            f['dlogpsi_dR_nuc'][bcs][()], dtype=np.float64,
        )
        n_w_total += O.shape[0]

        b_s_O = O.sum(axis=0)
        b_s_dEL_dp = dEL_dp.sum(axis=0)
        b_s_dEL_dR = dEL_dR.sum(axis=0)
        b_s_dlp_dR = dlp_dR.sum(axis=0)

        b_s_dEL_dp_O = dEL_dp.T @ O                # (n_p, n_p)
        b_s_dEL_dR_O = np.einsum(
            'wAk,wj->Akj', dEL_dR, O,
        )                                          # (n_nuc, 3, n_p)
        b_s_dEL_dp_dlp_dR = np.einsum(
            'wi,wAk->iAk', dEL_dp, dlp_dR,
        )                                          # (n_p, n_nuc, 3)

        if s_O is None:
            s_O = b_s_O
            s_dEL_dp = b_s_dEL_dp
            s_dEL_dR = b_s_dEL_dR
            s_dlp_dR = b_s_dlp_dR
            s_dEL_dp_O = b_s_dEL_dp_O
            s_dEL_dR_O = b_s_dEL_dR_O
            s_dEL_dp_dlp_dR = b_s_dEL_dp_dlp_dR
        else:
            s_O += b_s_O
            s_dEL_dp += b_s_dEL_dp
            s_dEL_dR += b_s_dEL_dR
            s_dlp_dR += b_s_dlp_dR
            s_dEL_dp_O += b_s_dEL_dp_O
            s_dEL_dR_O += b_s_dEL_dR_O
            s_dEL_dp_dlp_dR += b_s_dEL_dp_dlp_dR

    n_w = float(n_w_total)
    O_mean = s_O / n_w
    dEL_dp_mean = s_dEL_dp / n_w
    dEL_dR_mean = s_dEL_dR / n_w
    dlp_dR_mean = s_dlp_dR / n_w

    n_p = O_mean.shape[0]
    n_nuc = dEL_dR_mean.shape[0]

    # Cov(X, Y) = E[XY] - E[X]E[Y]
    cov_dEL_dp_O = (
        s_dEL_dp_O / n_w
        - np.outer(dEL_dp_mean, O_mean)
    )                                              # (n_p, n_p)
    cov_dEL_dR_O = (
        s_dEL_dR_O / n_w
        - np.einsum('Ak,j->Akj', dEL_dR_mean, O_mean)
    )                                              # (n_nuc, 3, n_p)
    cov_dEL_dp_dlp_dR = (
        s_dEL_dp_dlp_dR / n_w
        - np.einsum('i,Ak->iAk', dEL_dp_mean, dlp_dR_mean)
    )                                              # (n_p, n_nuc, 3)

    # H_pp = 2*(Cov(dEL_dp, O) + Cov(dEL_dp, O).T)
    H_pp = jnp.asarray(
        2.0 * (cov_dEL_dp_O + cov_dEL_dp_O.T)
    )

    # H_pR[i,A,k] = 2*(Cov(dEL_dR[A,k], O[i])
    #               + Cov(dEL_dp[i], dlp_dR[A,k]))
    H_pR = jnp.asarray(
        2.0 * (
            cov_dEL_dR_O.transpose(2, 0, 1)
            + cov_dEL_dp_dlp_dR
        )
    )                                              # (n_p, n_nuc, 3)

    # Solve dp/dR = -H_pp^{-1} H_pR
    cond_hpp = float(jnp.linalg.cond(H_pp))
    H_pR_flat = H_pR.reshape(n_p, -1)
    dp_dR_flat = -jnp.linalg.solve(
        H_pp, H_pR_flat,
    )
    dp_dR = dp_dR_flat.reshape(n_p, n_nuc, 3)

    if (
        not jnp.all(jnp.isfinite(dp_dR))
        or cond_hpp > 1e12
    ):
        print(
            f"  ZVZB2: H_pp ill-conditioned "
            f"(cond={cond_hpp:.2e}), skipping "
            f"parameter response correction"
        )
        dp_dR = None
    else:
        print(
            f"  ZVZB2: param response solved "
            f"({n_p} params, "
            f"cond(H_pp)={cond_hpp:.2e})"
        )

    return dp_dR


def postproc_h5_pgcs(
        prefix: str = "vmc",
        logfile: bool | str = False,
        walker_based_batch_size: int = 10
        ) -> jnp.ndarray:
    """Post-process VMC gradient data to obtain \
nuclear forces using PGCS.

    Reads the gradient HDF5 file written by
    :meth:`_VMCDriverGTO.__call__` and applies Point
    Group Correlated Sampling (PGCS) to obtain
    symmetry-averaged estimates of the nuclear forces
    and their statistical errors.

    Parameters
    ----------
    prefix : str, optional
        File-name stem used when the VMC run was set
        up (the ``prefix`` argument of
        :func:`get_vmc_gto_func`).  The function looks
        for ``<prefix>.grd.h5``; trailing ``.chk.h5``
        or ``.grd.h5`` suffixes are stripped
        automatically.  Default is ``"vmc"``.
    logfile : bool or str, optional
        Controls logging output.  ``False`` (default)
        suppresses logging.  ``True`` writes to
        ``<prefix>.log``.  A string is used as the log
        file path directly (a ``.log`` extension is
        appended if absent).
    walker_based_batch_size : int, optional
        Number of walker blocks to load into memory at
        once when summing gradient contributions.
        Reduce this value if GPU memory is tight.
        Default is 10.

    Returns
    -------
    grd : jnp.ndarray, shape (num_atoms, 3)
        Mean nuclear forces (negative energy gradient)
        in Hartree/Bohr.
    grd_err : jnp.ndarray, shape (num_atoms, 3)
        Statistical error (standard error of the mean)
        of each force component.
    """
    from pyscf import gto

    suffixes_checked = [".chk.h5", ".grd.h5"]
    for s in suffixes_checked:
        if prefix.endswith(s):
            prefix = prefix[:-len(s)]
    # ofname_chkpt = prefix + ".chk.h5"
    ofname_grd = prefix + ".grd.h5"
    if not logfile or (
        isinstance(logfile, str) and logfile == ""
    ):
        ofname_log = None
    else:
        ofname_log = logfile.strip() \
            if logfile.endswith(".log") \
            else logfile.strip() + ".log"

    # 64 MiB chunk cache — coalesces multi-dataset
    # per-block reads into fewer libhdf5 calls.
    with h5py.File(
        ofname_grd, 'r', rdcc_nbytes=64 * 1024 ** 2,
    ) as f:
        atom_symbols = (
            f["system"]["atom_symbols"][()].split()
        )
        atom_coords = f["system"]["atom_coords"]
        myUnits = (
            f["system"]["units"][()].decode()
        )
        mole_data = [
            (atom_symbols[i].decode(),
             atom_coords[i, :])
            for i in range(len(atom_symbols))
        ]
        myMol = gto.M(
            atom=mole_data, basis="mini",
            unit=myUnits,
        )

        if "atom_fragment_map" in f["system"]:
            atom_frag_map = list(
                f["system"]["atom_fragment_map"][:]
            )
        else:
            atom_frag_map = None

        # Per-block arrays are read lazily from ``f``
        # inside the streaming block loop below — peak
        # host RAM stays O(per-block) instead of
        # O(total file size).

        block_nums = sorted(
            int(k) for k in f['local_energies'].keys()
            if k.isdigit()
        )

        # Streaming mean over blocks — read only one
        # block of ``local_energies`` at a time.
        _sum = 0.0
        _cnt = 0
        for block_cnt in block_nums:
            le_block = f['local_energies'][f'{block_cnt}']
            _sum += float(
                np.asarray(le_block[()]).sum(
                    dtype=np.float64,
                )
            )
            _cnt += le_block.size
        enr_mean = _sum / _cnt

        grd_nn = jnp.asarray(f['grd_nn'][()])

        # Identify combo labels from fragment_weights
        combo_labels = []
        if 'fragment_weights' in f:
            combo_labels = sorted(
                k for k in f['fragment_weights']
                if isinstance(
                    f['fragment_weights'][k], h5py.Group,
                )
            )
        states = [None] + combo_labels

        # Detect parameter-response data (ZVZB2).
        # ``_solve_param_response`` streams per-block.
        has_pr = ('O_params' in f)
        dp_dR = None
        if has_pr and states[0] is None:
            dp_dR = _solve_param_response(
                f, block_nums, enr_mean,
            )

        if ofname_log is None:
            fout = sys.stdout
        else:
            fout = open(ofname_log, 'w', 1)

        ref_grd_tot = None
        ref_grd_err = None
        all_state_results = {}

        # Per-state accumulators, indexed by state_label.
        # Loop order is inverted vs the eager-preload
        # version so each block is read from disk once
        # and all states are processed before moving on
        # to the next block.
        valid_samples_count = {s: 0 for s in states}
        grd_ee_en_sum = {s: 0.0 for s in states}
        grd_ke_sum = {s: 0.0 for s in states}
        grd_pulay_sum = {s: 0.0 for s in states}
        grd_tot_list = {s: [] for s in states}
        grd_err_list = {s: [] for s in states}

        for block_cnt in block_nums:
            bcs = f'{block_cnt}'

            # Reference local energies feed the Pulay
            # term for every state in this block.
            local_energies = jnp.asarray(
                f['local_energies'][bcs][()]
            )
            d_enr = local_energies - enr_mean
            num_steps_per_block, num_walkers = (
                local_energies.shape
            )

            # Param-response per-block tensors (loaded
            # once per block, reused only on the
            # reference state).
            if dp_dR is not None:
                O_p_block = jnp.asarray(
                    f['O_params'][bcs][()]
                )
                dEL_p_block = jnp.asarray(
                    f['dEL_dparams'][bcs][()]
                )

            # Hoist all per-state HDF5 dataset reads
            # for this block up front: one indexed
            # lookup per dataset instead of repeating
            # ``f['grd_ee_en'][...][bcs][()]`` style
            # chains inside the state loop.  Caches
            # group handles too.
            block_data = {}
            ee_en_grp = f['grd_ee_en']
            ke_grp = f['grd_ke']
            lp_grp = f['grd_logpsi']
            block_data[None] = (
                jnp.asarray(ee_en_grp[bcs][()]),
                jnp.asarray(ke_grp[bcs][()]),
                jnp.asarray(lp_grp[bcs][()]),
                None,
            )
            if combo_labels:
                fw_grp = f['fragment_weights']
                for label in combo_labels:
                    block_data[label] = (
                        jnp.asarray(
                            ee_en_grp[label][bcs][()]
                        ),
                        jnp.asarray(
                            ke_grp[label][bcs][()]
                        ),
                        jnp.asarray(
                            lp_grp[label][bcs][()]
                        ),
                        jnp.asarray(
                            fw_grp[label][bcs][()]
                        ),
                    )

            for state_label in states:
                grd_ee_en, grd_ke, grd_logpsi, frag_w = (
                    block_data[state_label]
                )

                _, num_nuc, _ = grd_ee_en.shape

                grd_ee_en = grd_ee_en.reshape(
                    num_steps_per_block,
                    num_walkers, num_nuc, 3,
                )
                grd_ke = grd_ke.reshape(
                    num_steps_per_block,
                    num_walkers, num_nuc, 3,
                )
                grd_logpsi = grd_logpsi.reshape(
                    num_steps_per_block,
                    num_walkers, num_nuc, 3,
                )
                grd_pulay = 2.0 * jnp.einsum(
                    'sw,swnK->swnK',
                    d_enr, grd_logpsi,
                )

                grd_nn_sw = jnp.broadcast_to(
                    grd_nn[jnp.newaxis, jnp.newaxis, :, :],
                    (num_steps_per_block, num_walkers,
                     num_nuc, 3),
                )
                grd_arrays = [
                    grd_nn_sw,
                    grd_ee_en, grd_ke,
                    grd_pulay,
                ]
                grd_tot_sw = jnp.stack(
                    grd_arrays, axis=0,
                ).sum(axis=0)

                # Parameter-response correction
                if (
                    dp_dR is not None
                    and state_label is None
                ):
                    n_jp = O_p_block.shape[-1]
                    O_p = O_p_block.reshape(
                        num_steps_per_block,
                        num_walkers, n_jp,
                    )
                    dEL_p = dEL_p_block.reshape(
                        num_steps_per_block,
                        num_walkers, n_jp,
                    )
                    cv = (
                        dEL_p
                        + 2.0 * (d_enr[..., None] * O_p)
                    )
                    delta_f = jnp.einsum(
                        'swi,inK->swnK', cv, dp_dR,
                    )
                    grd_tot_sw = grd_tot_sw + delta_f

                # ``frag_w`` for combo states was already
                # read into ``block_data`` above.
                if frag_w is not None:
                    frag_w = frag_w.reshape(
                        num_steps_per_block,
                        num_walkers,
                    )

                xbar, serr, sdev, kappa = (
                    batched_binning_analysis_grds(
                        grd_tot_sw,
                        walker_based_batch_size,
                        weights=frag_w,
                    )
                )
                grd_tot_list[state_label].append(
                    xbar[None, :, :, :]
                )
                grd_err_list[state_label].append(
                    serr[None, :, :, :]
                )

                grd_ee_en_sum[state_label] = (
                    grd_ee_en_sum[state_label]
                    + grd_ee_en.sum(axis=0)
                )
                grd_ke_sum[state_label] = (
                    grd_ke_sum[state_label]
                    + grd_ke.sum(axis=0)
                )
                grd_pulay_sum[state_label] = (
                    grd_pulay_sum[state_label]
                    + grd_pulay.sum(axis=0)
                )
                valid_samples_count[state_label] += (
                    local_energies.shape[0]
                )

        # --- Per-state post-loop averaging + report ---
        # ``num_nuc`` is reused below and is constant
        # across states / blocks; pull it from the
        # reference NN gradient tensor.
        num_nuc = int(grd_nn.shape[0])

        for state_label in states:
            vsc = valid_samples_count[state_label]
            if vsc > 0:
                grd_ee_en = (
                    grd_ee_en_sum[state_label] / vsc
                )
                grd_ke = grd_ke_sum[state_label] / vsc
                grd_pulay = (
                    grd_pulay_sum[state_label] / vsc
                )

                grd_tot_bw = jnp.concatenate(
                    grd_tot_list[state_label], axis=0,
                )
                grd_err_bw = jnp.concatenate(
                    grd_err_list[state_label], axis=0,
                )

                # mean over blocks, then walkers
                grd_tot = grd_tot_bw.mean(axis=0).squeeze()
                grd_tot = grd_tot.mean(axis=0)

                grd_err = (
                    jnp.linalg.norm(
                        grd_err_bw, axis=0,
                    ).squeeze()
                    / grd_err_bw.shape[0]
                )
                grd_err = (
                    jnp.linalg.norm(grd_err, axis=0)
                    / grd_err.shape[0]
                )

                torque, dtau = (
                    compute_torque_with_error(
                        myMol, grd_tot, grd_err,
                    )
                )

                grd_ee_en = jnp.mean(grd_ee_en, axis=0)
                grd_ke = jnp.mean(grd_ke, axis=0)
                grd_pulay = jnp.mean(grd_pulay, axis=0)

                grd_block_series = np.asarray(
                    grd_tot_bw.mean(axis=1)
                )
            else:
                grd_ee_en = jnp.zeros_like(grd_nn)
                grd_ke = jnp.zeros_like(grd_nn)
                grd_pulay = jnp.zeros_like(grd_nn)
                grd_tot = jnp.zeros_like(grd_nn)
                grd_err = jnp.zeros_like(grd_nn)
                torque = jnp.zeros(3)
                dtau = jnp.zeros(3)
                grd_block_series = None

            if state_label is None:
                ref_grd_tot = grd_tot
                ref_grd_err = grd_err

            all_state_results[state_label] = (
                grd_tot, grd_err, grd_block_series,
            )

            with jnp.printoptions(
                precision=12, suppress=True,
            ):
                if state_label is not None:
                    print("\n--- Secondary state:"
                          f" {state_label} ---",
                          file=fout)
                else:
                    print("\n--- Reference state ---",
                          file=fout)

                print('NN gradients\n', grd_nn, file=fout)
                print('ee+eN gradients\n', grd_ee_en,
                      file=fout)
                print('KE gradients\n', grd_ke, file=fout)
                print('Pulay gradients\n', grd_pulay,
                      file=fout)
                print('Total gradients\n', grd_tot,
                      file=fout)

                fout.write("Total forces (-gradients)\n")
                for i in range(num_nuc):
                    fout.write(
                        "{:4s}"
                        "{:>16.6g} ± {:>12.6g}"
                        "{:>16.6g} ± {:>12.6g}"
                        "{:>16.6g} ± {:>12.6g}\n"
                        .format(
                            myMol.atom_symbol(i),
                            -grd_tot[i, 0],
                            grd_err[i, 0],
                            -grd_tot[i, 1],
                            grd_err[i, 1],
                            -grd_tot[i, 2],
                            grd_err[i, 2],
                        )
                    )
                fout.write("Total torque\n")
                fout.write(
                    "    {:>16.6g} ± {:>12.6g}"
                    "{:>16.6g} ± {:>12.6g}"
                    "{:>16.6g} ± {:>12.6g}\n"
                    .format(
                        torque[0], dtau[0],
                        torque[1], dtau[1],
                        torque[2], dtau[2],
                    )
                )
                fout.write("\n")

        # --- Fragment-wise BLUE averaging of PGCS ---
        if atom_frag_map is not None and combo_labels:
            frag_to_states = {}
            for fid in set(atom_frag_map):
                frag_to_states[fid] = [None]

            for label in combo_labels:
                for part in label.split(','):
                    fid_str, op = part.split(':')
                    if op != 'E':
                        frag_to_states[int(fid_str)].append(label)
                        break

            avg_grd_tot_np = np.zeros((num_nuc, 3))
            avg_grd_err_np = np.zeros((num_nuc, 3))

            # Per-(atom, direction) BLUE combination across
            # the states that touched atom i's fragment.
            # The (num_blocks,) scalar series feeding BLUE
            # is the per-block walker-averaged force
            # component ``F_{i,k}`` for each relevant state.
            for i in range(num_nuc):
                fid = atom_frag_map[i]
                relevant_states = frag_to_states.get(
                    fid, [None],
                )
                series_block = {
                    s: all_state_results[s][2]
                    for s in relevant_states
                }
                # Fallback when any state has no valid
                # samples (no time series available): use
                # the per-state mean/error directly.
                if any(
                    series_block[s] is None
                    for s in relevant_states
                ):
                    forces = jnp.stack([
                        all_state_results[s][0][i]
                        for s in relevant_states
                    ])
                    errors = jnp.stack([
                        all_state_results[s][1][i]
                        for s in relevant_states
                    ])
                    N = len(relevant_states)
                    avg_grd_tot_np[i] = np.asarray(
                        forces.mean(axis=0)
                    )
                    avg_grd_err_np[i] = np.asarray(
                        jnp.sqrt(jnp.sum(errors ** 2, axis=0))
                        / N
                    )
                    continue

                for k in range(3):
                    series_ik = {
                        s: series_block[s][:, i, k]
                        for s in relevant_states
                    }
                    mean, err, _, _, _ = (
                        blue_combine_states(
                            series_ik, relevant_states,
                        )
                    )
                    avg_grd_tot_np[i, k] = mean
                    avg_grd_err_np[i, k] = err

            avg_grd_tot = jnp.asarray(avg_grd_tot_np)
            avg_grd_err = jnp.asarray(avg_grd_err_np)

            avg_torque, avg_dtau = (
                compute_torque_with_error(
                    myMol, avg_grd_tot, avg_grd_err,
                )
            )

            with jnp.printoptions(
                precision=12, suppress=True,
            ):
                fout.write("\nℹ️\tFragment-wise averaged forces\n")
                fout.write("Total gradients (averaged)\n")
                fout.write(f" {avg_grd_tot}\n")
                fout.write("Total forces (-gradients, averaged)\n")
                for i in range(num_nuc):
                    fid = atom_frag_map[i]
                    n_states = len(
                        frag_to_states.get(
                            fid, [None],
                        )
                    )
                    fout.write(
                        "{:4s}"
                        "{:>16.6g} ± {:>12.6g}"
                        "{:>16.6g} ± {:>12.6g}"
                        "{:>16.6g} ± {:>12.6g}"
                        "  (fragment {},"
                        " over {} states)\n"
                        .format(
                            myMol.atom_symbol(i),
                            -avg_grd_tot[i, 0], avg_grd_err[i, 0],
                            -avg_grd_tot[i, 1], avg_grd_err[i, 1],
                            -avg_grd_tot[i, 2], avg_grd_err[i, 2],
                            fid, n_states,
                        )
                    )
                fout.write(
                    "Total torque (averaged)\n"
                )
                fout.write(
                    "    {:>16.6g} ± {:>12.6g}"
                    "{:>16.6g} ± {:>12.6g}"
                    "{:>16.6g} ± {:>12.6g}\n"
                    .format(
                        avg_torque[0], avg_dtau[0],
                        avg_torque[1], avg_dtau[1],
                        avg_torque[2], avg_dtau[2],
                    )
                )
                fout.write("\n")

            ref_grd_tot = avg_grd_tot
            ref_grd_err = avg_grd_err

        if ofname_log is not None:
            fout.close()

        return -ref_grd_tot, ref_grd_err
