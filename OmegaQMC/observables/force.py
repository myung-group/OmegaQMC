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

import ast
import os
import re
import warnings
from datetime import datetime

import jax
import h5py
import numpy as np
import jax.numpy as jnp
from pyscf.data.nist import BOHR          # Angstrom per Bohr

from ..utils import (
    batched_binning_analysis_grds,
    blue_combine_states,
    compute_torque_with_error,
    equilibration_length,
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
    log_psi_batch, local_energy_batch, apply_single_frag_symmop :
        callable, optional Required only when *single_frag_combos* is
        non-empty (combo screening + secondary evaluation).

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


def _project_force_sum_rules(grd, grd_err, coords, ref):
    """Project a force/gradient estimate onto the translational and
    rotational sum-rule subspace via constrained generalized least
    squares (Lagrange multipliers).

    For an isolated molecule at any geometry the exact forces obey six
    exact linear constraints,

        sum_i F_i = 0          (translational invariance of E)
        sum_i r_i x F_i = 0    (rotational invariance of E)

    which SWCT enforces per state but which the fragment-wise PGCS
    average breaks (different atoms are averaged over different state
    subsets).  Given the noisy averaged gradient ``grd`` with diagonal
    covariance built from ``grd_err``, the minimum-chi-squared
    correction satisfying ``C g = 0`` is the oblique projection

        g_proj = (I - Sigma C^T (C Sigma C^T)^+ C) g ,

    where ``C`` is the 6 x 3N constraint matrix (3 translational rows
    plus 3 rows of the r_i-cross skew blocks about ``ref``) and
    ``Sigma`` is ``diag(grd_err**2)``.  The residual is absorbed mostly
    by the least well-determined components, the result satisfies both
    sum rules exactly, and the propagated covariance
    ``(I - P) Sigma (I - P)^T`` does not exceed the input variances.
    A pseudo-inverse is used so degenerate constraints (e.g. the
    torque about the axis of a linear molecule, whose constraint row
    vanishes) are handled gracefully.

    Parameters
    ----------
    grd, grd_err : array_like, shape (N, 3)
        Averaged gradient estimate and its per-component 1-sigma error.
    coords : array_like, shape (N, 3)
        Nuclear coordinates.
    ref : array_like, shape (3,)
        Reference point for the rotational (torque) constraint.

    Returns
    -------
    grd_proj : ndarray, shape (N, 3)
        Sum-rule-consistent gradient.
    grd_err_proj : ndarray, shape (N, 3)
        Propagated 1-sigma error of ``grd_proj``.
    """
    grd = np.asarray(grd, dtype=float)
    grd_err = np.asarray(grd_err, dtype=float)
    coords = np.asarray(coords, dtype=float)
    ref = np.asarray(ref, dtype=float)
    n = grd.shape[0]
    g = grd.reshape(-1)                       # (3N,)
    sigma2 = (grd_err.reshape(-1)) ** 2        # (3N,)

    # Build the 6 x 3N constraint matrix C.
    rel = coords - ref                         # (N, 3)
    C = np.zeros((6, 3 * n))
    for i in range(n):
        x, y, z = rel[i]
        # Translational rows.
        C[0, 3 * i + 0] = 1.0
        C[1, 3 * i + 1] = 1.0
        C[2, 3 * i + 2] = 1.0
        # Rotational rows: tau = sum_i r_i x F_i.
        # tau_x = sum (y F_z - z F_y)
        C[3, 3 * i + 2] += y
        C[3, 3 * i + 1] += -z
        # tau_y = sum (z F_x - x F_z)
        C[4, 3 * i + 0] += z
        C[4, 3 * i + 2] += -x
        # tau_z = sum (x F_y - y F_x)
        C[5, 3 * i + 1] += x
        C[5, 3 * i + 0] += -y

    sc_t = sigma2[:, None] * C.T               # Sigma C^T, (3N, 6)
    csct = C @ sc_t                            # C Sigma C^T, (6, 6)
    csct_pinv = np.linalg.pinv(csct, rcond=1e-10)
    p = sc_t @ csct_pinv @ C                   # P = Sigma C^T (..)^+ C
    m = np.eye(3 * n) - p                       # I - P

    g_proj = m @ g
    cov_proj = (m * sigma2[None, :]) @ m.T      # (I-P) Sigma (I-P)^T
    var_proj = np.clip(np.diag(cov_proj), 0.0, None)

    grd_proj = g_proj.reshape(n, 3)
    grd_err_proj = np.sqrt(var_proj).reshape(n, 3)
    return grd_proj, grd_err_proj


# Markers delimiting the geometry block echoed at the head of a
# postprocessing log; ``read_omega_xyz_block`` slices between them.
XYZ_BEGIN = "--- Geometry (xyz, Angstrom) ---"
XYZ_END = "--- End geometry ---"


def _format_xyz_lines(symbols, coords_ang, frag_ids=None,
                      comment=None, ndec=10):
    """Render a geometry as xyz lines with a fragment-index column.

    Produces the project's extended-xyz flavour: an atom count, a
    comment line carrying a ``Properties=`` spec, then one
    ``symbol x y z fragment`` row per atom, wrapped in the
    :data:`XYZ_BEGIN` / :data:`XYZ_END` markers.

    Parameters
    ----------
    symbols : sequence of str
        Element symbols, one per atom.
    coords_ang : ndarray, shape (N, 3)
        Nuclear coordinates **in Angstrom** (xyz convention).
    frag_ids : sequence of int, optional
        Per-atom fragment labels.  ``None`` emits ``0`` for every atom,
        matching the default of
        :func:`~OmegaQMC.utils.parse_molecular_inspheres` when an xyz
        file carries no fragment column.
    comment : str, optional
        Appended to the ``Properties=`` spec on the comment line.
    ndec : int, optional
        Decimal places for the coordinates.  The default of 10 keeps the
        Bohr -> Angstrom conversion lossless well below any tolerance
        used downstream.

    Returns
    -------
    list of str
        The block's lines, without trailing newlines.
    """
    coords_ang = np.asarray(coords_ang, dtype=float)
    n = len(symbols)
    if frag_ids is None:
        frag_ids = [0] * n
    # ``parse_molecular_inspheres`` warns unless the comment line
    # advertises a molecule (fragment) column, so always emit the spec.
    head = "Properties=species:S:1:pos:R:3:molecule:I:1"
    if comment:
        head = f"{head}  {comment}"
    lines = [XYZ_BEGIN, str(n), head]
    w = ndec + 8
    for i in range(n):
        x, y, z = coords_ang[i]
        lines.append(
            f"{str(symbols[i]):<2s} {x:>{w}.{ndec}f} "
            f"{y:>{w}.{ndec}f} {z:>{w}.{ndec}f} "
            f"{int(frag_ids[i]):>4d}"
        )
    lines.append(XYZ_END)
    return lines


def postproc_h5_pgcs(
        prefix: str = "vmc",
        logfile: bool | str = False,
        walker_based_batch_size: int = 10,
        project_states: bool = False,
        equil_cutoff: int | str = 0,
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
    project_states : bool, optional
        When ``True``, apply the translational +
        rotational sum-rule projection
        (:func:`_project_force_sum_rules`) to *each*
        per-state force vector before it is printed.
        SWCT already makes the translational sum exact
        per state, but the rotational sum rule holds only
        in expectation, so without projection a state's
        ``Total torque`` is a noisy estimate of zero; this
        flag cleans up the per-state torque rows.  It does
        **not** change the fragment-wise averaged result,
        which is always projected separately (the
        per-(atom, component) BLUE recombination re-breaks
        the sum rules regardless, so the averaged vector is
        projected on its own).  Default is ``False``.
    equil_cutoff : int or str, optional
        Number of leading blocks to discard as
        equilibration before the time-series analysis;
        equivalently the block index (in sorted order) at
        which the analysis starts.  Applied uniformly to
        the energy mean, parameter-response solve, and the
        per-state force series.  Pass the string ``"auto"``
        to detect the cutoff from the reference per-block
        energy series via
        :func:`OmegaQMC.utils.equilibration_length`.
        Default is ``0`` (use all blocks).

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
    elif logfile is True:
        # Documented behaviour: ``True`` -> "<prefix>.log".
        ofname_log = prefix + ".log"
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
        # inside the streaming block loop below
        # — peak host RAM stays O(per-block) instead of O(total file size).

        block_nums = sorted(
            int(k) for k in f['local_energies'].keys()
            if k.isdigit()
        )

        # Streaming pass over blocks — read only one block
        # of ``local_energies`` at a time, keeping the
        # per-block sum/size so the energy mean can be
        # formed after the equilibration cutoff is applied
        # (and, for ``"auto"``, the per-block mean series
        # the cutoff is detected from).
        block_sums = []
        block_sizes = []
        for block_cnt in block_nums:
            le_block = f['local_energies'][f'{block_cnt}']
            block_sums.append(
                float(np.asarray(le_block[()]).sum(
                    dtype=np.float64,
                ))
            )
            block_sizes.append(le_block.size)

        # Resolve the equilibration cutoff (auto or int);
        # the single slice below feeds the energy mean, the
        # parameter-response solve, and the force series.
        auto_msg = None
        if equil_cutoff == "auto":
            ref_series = [
                s / n for s, n in zip(block_sums, block_sizes)
            ]
            equil_cutoff = equilibration_length(ref_series)
            auto_msg = (
                "Auto-detected equilibration at block"
                f" index {equil_cutoff}"
            )
        elif isinstance(equil_cutoff, bool) \
                or not isinstance(equil_cutoff, int):
            raise ValueError(
                "equil_cutoff must be a non-negative int"
                f" or 'auto', got {equil_cutoff!r}"
            )
        elif equil_cutoff < 0:
            raise ValueError(
                "equil_cutoff must be non-negative,"
                f" got {equil_cutoff}"
            )
        if equil_cutoff >= len(block_nums):
            raise ValueError(
                f"equil_cutoff ({equil_cutoff}) discards"
                f" all {len(block_nums)} blocks"
            )
        block_nums = block_nums[equil_cutoff:]
        enr_mean = (
            sum(block_sums[equil_cutoff:])
            / sum(block_sizes[equil_cutoff:])
        )

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

        # Echo the geometry the forces are computed from, first thing,
        # so the log is self-contained: the frame-transform pipeline
        # (:func:`make_reframe_log`) reads it back to align this run
        # against another package's standard orientation.
        # ``atom_coords`` is always Bohr (written from pyscf
        # ``mol.atom_coords()``); convert here rather than going through
        # ``myMol``, whose ``unit=myUnits`` round-trip misreads runs that
        # were set up in Angstrom.
        for line in _format_xyz_lines(
            [s.decode() for s in atom_symbols],
            np.asarray(atom_coords[:]) * BOHR,
            frag_ids=atom_frag_map,
            comment=("OmegaQMC working frame "
                     "(center of nuclear charge + "
                     "charge-inertia axes)"),
        ):
            print(line, file=fout)

        if auto_msg is not None:
            print(auto_msg, file=fout)

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
        # Per-block walker-averaged force series, feeding
        # both the final mean and the BLUE block series.
        # Each entry is reduced to (num_nuc, 3) per block so
        # peak RAM stays O(num_nuc) rather than growing as
        # O(num_blocks * num_walkers).
        grd_blockmean_list = {s: [] for s in states}
        # Running sum of per-walker squared standard errors,
        # already reduced over walkers each block: (num_nuc,
        # 3).  Paired with the walker count below to form the
        # combined error sqrt(sum_{b,w} serr^2) / (B * W).
        grd_err_sq_sum = {s: 0.0 for s in states}
        grd_err_w_count = {s: 0 for s in states}

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

                # Sum the gradient contributions directly
                # instead of ``jnp.stack(...).sum(axis=0)``,
                # which would materialize a 4x ``(s, w, n, 3)``
                # tensor only to immediately reduce it away.
                # ``grd_nn`` broadcasts over the leading
                # step/walker axes on its own, so the explicit
                # ``broadcast_to`` is unnecessary too.
                grd_tot_sw = (
                    grd_nn[jnp.newaxis, jnp.newaxis, :, :]
                    + grd_ee_en + grd_ke + grd_pulay
                )

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
                # Reduce this block immediately: store only
                # the walker mean (force-series term) and
                # fold the squared standard error over
                # walkers into the running accumulator,
                # instead of retaining the full (W, num_nuc,
                # 3) arrays for every block.
                grd_blockmean_list[state_label].append(
                    xbar.mean(axis=0)
                )
                grd_err_sq_sum[state_label] = (
                    grd_err_sq_sum[state_label]
                    + (serr ** 2).sum(axis=0)
                )
                grd_err_w_count[state_label] += (
                    serr.shape[0]
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

                # Per-block walker-averaged force series,
                # (num_blocks, num_nuc, 3); ``grd_tot`` is
                # its mean over blocks.  Both follow directly
                # from the per-block reductions accumulated in
                # the streaming loop above.
                grd_block_series = jnp.stack(
                    grd_blockmean_list[state_label], axis=0,
                )
                grd_tot = grd_block_series.mean(axis=0)

                # Combined standard error, identical to the
                # previous nested-norm reduction:
                # sqrt(sum_{b,w} serr^2) / (B * W), with
                # ``grd_err_w_count`` == B * W.
                grd_err = (
                    jnp.sqrt(grd_err_sq_sum[state_label])
                    / grd_err_w_count[state_label]
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
                    grd_block_series
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

            # Per-state printing optionally uses sum-rule-
            # projected copies (``project_states``).  The
            # values stored in ``all_state_results`` above
            # and the returned reference force stay
            # unprojected, so neither the fragment-wise
            # average nor the return value is affected; this
            # only cleans up the printed per-state rows
            # (chiefly the rotational ``Total torque``, which
            # SWCT does not enforce per state).
            if project_states and grd_block_series is not None:
                coords_state = np.asarray(myMol.atom_coords())
                masses_state = np.asarray(
                    myMol.atom_mass_list()
                )
                com_state = np.average(
                    coords_state, axis=0, weights=masses_state,
                )
                grd_tot_p, grd_err_p = (
                    _project_force_sum_rules(
                        grd_tot, grd_err,
                        coords_state, com_state,
                    )
                )
                grd_tot_print = jnp.asarray(grd_tot_p)
                grd_err_print = jnp.asarray(grd_err_p)
                torque_print, dtau_print = (
                    compute_torque_with_error(
                        myMol, grd_tot_print, grd_err_print,
                    )
                )
            else:
                grd_tot_print = grd_tot
                grd_err_print = grd_err
                torque_print = torque
                dtau_print = dtau

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
                print('Total gradients\n', grd_tot_print,
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
                            -grd_tot_print[i, 0],
                            grd_err_print[i, 0],
                            -grd_tot_print[i, 1],
                            grd_err_print[i, 1],
                            -grd_tot_print[i, 2],
                            grd_err_print[i, 2],
                        )
                    )
                F_sys = -np.asarray(grd_tot_print).sum(axis=0)
                dF_sys = np.sqrt(
                    (np.asarray(grd_err_print) ** 2).sum(axis=0)
                )
                fout.write("Total force on system\n")
                fout.write(
                    "    {:>16.6g} ± {:>12.6g}"
                    "{:>16.6g} ± {:>12.6g}"
                    "{:>16.6g} ± {:>12.6g}\n"
                    .format(
                        F_sys[0], dF_sys[0],
                        F_sys[1], dF_sys[1],
                        F_sys[2], dF_sys[2],
                    )
                )
                coords_np = np.asarray(myMol.atom_coords())
                centroid = coords_np.mean(axis=0)
                F_np = -np.asarray(grd_tot_print)
                dF_np = np.asarray(grd_err_print)
                fout.write(
                    "Torque per atom (about centroid)\n"
                )
                for i in range(num_nuc):
                    x, y, z = coords_np[i] - centroid
                    Fx, Fy, Fz = F_np[i]
                    dFx, dFy, dFz = dF_np[i]
                    tau_i = np.array([
                        y * Fz - z * Fy,
                        z * Fx - x * Fz,
                        x * Fy - y * Fx,
                    ])
                    dtau_i = np.sqrt(np.array([
                        (y * dFz) ** 2 + (z * dFy) ** 2,
                        (z * dFx) ** 2 + (x * dFz) ** 2,
                        (x * dFy) ** 2 + (y * dFx) ** 2,
                    ]))
                    fout.write(
                        "{:4s}"
                        "{:>16.6g} ± {:>12.6g}"
                        "{:>16.6g} ± {:>12.6g}"
                        "{:>16.6g} ± {:>12.6g}\n"
                        .format(
                            myMol.atom_symbol(i),
                            tau_i[0], dtau_i[0],
                            tau_i[1], dtau_i[1],
                            tau_i[2], dtau_i[2],
                        )
                    )
                fout.write("Total torque\n")
                fout.write(
                    "    {:>16.6g} ± {:>12.6g}"
                    "{:>16.6g} ± {:>12.6g}"
                    "{:>16.6g} ± {:>12.6g}\n"
                    .format(
                        torque_print[0], dtau_print[0],
                        torque_print[1], dtau_print[1],
                        torque_print[2], dtau_print[2],
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

            # Enforce the exact translational + rotational sum
            # rules (sum F = 0, sum r x F = 0) that the
            # fragment-wise averaging breaks, via constrained GLS
            # projection weighted by the per-component errors. The
            # rotational constraint is taken about the center of
            # mass, matching ``compute_torque_with_error`` below, so
            # the reported total force and total torque are both
            # exactly zero and consistent with the per-atom values.
            coords_for_proj = np.asarray(myMol.atom_coords())
            masses_for_proj = np.asarray(myMol.atom_mass_list())
            com_for_proj = np.average(
                coords_for_proj, axis=0, weights=masses_for_proj,
            )
            avg_grd_tot_np, avg_grd_err_np = _project_force_sum_rules(
                avg_grd_tot_np, avg_grd_err_np,
                coords_for_proj, com_for_proj,
            )

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
                F_sys = -np.asarray(avg_grd_tot).sum(axis=0)
                dF_sys = np.sqrt(
                    (np.asarray(avg_grd_err) ** 2).sum(axis=0)
                )
                fout.write(
                    "Total force on system (averaged)\n"
                )
                fout.write(
                    "    {:>16.6g} ± {:>12.6g}"
                    "{:>16.6g} ± {:>12.6g}"
                    "{:>16.6g} ± {:>12.6g}\n"
                    .format(
                        F_sys[0], dF_sys[0],
                        F_sys[1], dF_sys[1],
                        F_sys[2], dF_sys[2],
                    )
                )
                coords_np = np.asarray(myMol.atom_coords())
                centroid = coords_np.mean(axis=0)
                F_np = -np.asarray(avg_grd_tot)
                dF_np = np.asarray(avg_grd_err)
                fout.write(
                    "Torque per atom (about centroid,"
                    " averaged)\n"
                )
                for i in range(num_nuc):
                    x, y, z = coords_np[i] - centroid
                    Fx, Fy, Fz = F_np[i]
                    dFx, dFy, dFz = dF_np[i]
                    tau_i = np.array([
                        y * Fz - z * Fy,
                        z * Fx - x * Fz,
                        x * Fy - y * Fx,
                    ])
                    dtau_i = np.sqrt(np.array([
                        (y * dFz) ** 2 + (z * dFy) ** 2,
                        (z * dFx) ** 2 + (x * dFz) ** 2,
                        (x * dFy) ** 2 + (y * dFx) ** 2,
                    ]))
                    fout.write(
                        "{:4s}"
                        "{:>16.6g} ± {:>12.6g}"
                        "{:>16.6g} ± {:>12.6g}"
                        "{:>16.6g} ± {:>12.6g}\n"
                        .format(
                            myMol.atom_symbol(i),
                            tau_i[0], dtau_i[0],
                            tau_i[1], dtau_i[1],
                            tau_i[2], dtau_i[2],
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


# ---------------------------------------------------------------------------
# Cross-code frame transforms ("reframing")
# ---------------------------------------------------------------------------
# OmegaQMC reorients a molecule into its own canonical frame (center of
# nuclear charge + charge-weighted-inertia axes) before doing any work, and
# every other quantum-chemistry package does something similar with its own
# convention for the degrees of freedom that symmetry does not fix (for a
# Cs system both may put the mirror plane at z = 0, yet the in-plane axes
# differ by a rotation about z -- see tests/orient-molecule/FINDINGS.md).
#
# The two working geometries are therefore the *same rigid body* expressed
# in different frames.  Rather than reproduce any package's orientation
# convention, we recover the map between the frames and record it compactly
# as a translation vector and a rotation matrix in a ``reframe.log``:
#
#     x_target = R @ (x_source + t)        (translate first, then rotate)
#
# Forces are vectors and torques are pseudovectors, so with det(R) = +1
#
#     F_target = R @ F_source     tau_target = R @ tau_source
#
# and the translation cancels for both.  Everything below reads plain-text
# stdout/log files; the HDF5 gradient file is not re-read.

# per-atom force line:  "O   <Fx> ± <dFx> <Fy> ± <dFy> <Fz> ± <dFz> ..."
# The symbol may carry a numeric tag ("O1") when atoms were labelled.
_FORCE_LINE = re.compile(
    r"^\s*([A-Za-z]{1,2}\d*)\s+"
    r"([-+0-9.eE]+|[-+]?(?:nan|inf))\s*±\s*([-+0-9.eE]+|[-+]?(?:nan|inf))\s+"
    r"([-+0-9.eE]+|[-+]?(?:nan|inf))\s*±\s*([-+0-9.eE]+|[-+]?(?:nan|inf))\s+"
    r"([-+0-9.eE]+|[-+]?(?:nan|inf))\s*±\s*([-+0-9.eE]+|[-+]?(?:nan|inf))")

# Force/torque block headers written by postproc_h5_pgcs.
_BLOCK_HEADERS = {
    "Total forces (-gradients)": ("force", "state"),
    "Total forces (-gradients, averaged)": ("force", "averaged"),
    "Torque per atom (about centroid)": ("torque", "state"),
    "Torque per atom (about centroid, averaged)": ("torque", "averaged"),
}

SUPPORTED_PACKAGES = ("g16", "nwchem")


def _bare_symbol(sym):
    """Strip a trailing numeric tag from an element symbol ("O1" -> "O")."""
    return re.sub(r"\d+$", "", str(sym)).strip()


# ---------------------------------------------------------------------------
# Readers: OmegaQMC postprocessing log
# ---------------------------------------------------------------------------

def read_omega_xyz_block(logpath):
    """Parse the geometry echoed at the head of a postprocessing log.

    Reads the block that :func:`postproc_h5_pgcs` writes between the
    :data:`XYZ_BEGIN` and :data:`XYZ_END` markers.

    Returns
    -------
    (symbols, coords_ang, frag_ids)
        Element symbols, an ``(N, 3)`` array of coordinates **in
        Angstrom**, and the per-atom fragment labels.

    Raises
    ------
    ValueError
        If the log carries no geometry block (e.g. it predates this
        feature); callers may then fall back to
        :func:`read_omega_orientation`.
    """
    with open(logpath) as fh:
        lines = fh.read().splitlines()
    try:
        i0 = next(i for i, ln in enumerate(lines)
                  if ln.strip() == XYZ_BEGIN)
    except StopIteration:
        raise ValueError(
            f"No geometry block ({XYZ_BEGIN!r}) in {logpath!r}") from None
    try:
        i1 = next(i for i in range(i0 + 1, len(lines))
                  if lines[i].strip() == XYZ_END)
    except StopIteration:
        raise ValueError(
            f"Unterminated geometry block in {logpath!r}") from None

    body = lines[i0 + 1:i1]
    if len(body) < 2:
        raise ValueError(f"Empty geometry block in {logpath!r}")
    natoms = int(body[0].split()[0])
    symbols, coords, frags = [], [], []
    for ln in body[2:2 + natoms]:
        parts = ln.split()
        symbols.append(parts[0])
        coords.append([float(parts[1]), float(parts[2]), float(parts[3])])
        frags.append(int(parts[4]) if len(parts) > 4 else 0)
    if len(symbols) != natoms:
        raise ValueError(
            f"Geometry block in {logpath!r} declares {natoms} atoms but "
            f"carries {len(symbols)} rows")
    return symbols, np.array(coords, dtype=float), frags


def read_omega_orientation(logpath):
    """Parse OmegaQMC's reoriented geometry from a VMC run's stdout.

    Legacy/alternate source, kept for logs written before
    :func:`postproc_h5_pgcs` began echoing its geometry (see
    :func:`read_omega_xyz_block`, which should be preferred).
    ``generate_molecular_orbitals`` echoes the post-transformation
    geometry as a Python-literal list of ``(symbol, [x, y, z],
    fragment_id)`` tuples when ``mol.verbose >= 3``.

    Returns ``(symbols, coords_bohr, frag_ids)`` -- note the
    coordinates are **in Bohr**, unlike
    :func:`read_omega_xyz_block`.
    """
    with open(logpath) as fh:
        for line in fh:
            s = line.strip()
            if s.startswith("[('") or s.startswith("[("):
                atoms = ast.literal_eval(s)
                symbols = [a[0] for a in atoms]
                coords = np.array([a[1] for a in atoms], float)
                frags = [a[2] if len(a) > 2 else 0 for a in atoms]
                return symbols, coords, frags
    raise ValueError(f"No OmegaQMC geometry line found in {logpath!r}")


def read_omega_force_blocks(logpath):
    """Parse every per-atom force/torque block in a postprocessing log.

    Scans the whole log, tracking the ``--- Reference state ---`` /
    ``--- Secondary state: <label> ---`` markers, so each block is
    labelled by the state it belongs to.  Blocks whose header mentions
    ``reframed`` are skipped, so re-reading a file that already contains
    reframed output cannot rotate it a second time.

    Returns
    -------
    list of dict
        One entry per block, in file order, with keys ``quantity``
        (``"force"`` / ``"torque"``), ``kind`` (``"state"`` /
        ``"averaged"``), ``label`` (``None`` for the reference state,
        else the secondary-state label), ``header``, ``symbols``,
        ``values`` and ``errors`` (both ``(N, 3)``), plus ``frag_ids``
        and ``n_states`` where the averaged block records them.
    """
    with open(logpath) as fh:
        lines = fh.read().splitlines()

    blocks = []
    state_label = None
    i = 0
    while i < len(lines):
        s = lines[i].strip()
        if s.startswith("--- Reference state"):
            state_label = None
        elif s.startswith("--- Secondary state:"):
            state_label = s.split(":", 1)[1].strip().rstrip("-").strip()
        elif s in _BLOCK_HEADERS and "reframed" not in s:
            quantity, kind = _BLOCK_HEADERS[s]
            syms, vals, errs, frags, nsts = [], [], [], [], []
            j = i + 1
            while j < len(lines):
                m = _FORCE_LINE.match(lines[j])
                if not m:
                    break
                g = [float(m.group(k)) for k in range(2, 8)]
                syms.append(m.group(1))
                vals.append(g[0::2])
                errs.append(g[1::2])
                tail = re.search(r"\(fragment\s+(\d+),\s*over\s+(\d+)",
                                 lines[j])
                frags.append(int(tail.group(1)) if tail else None)
                nsts.append(int(tail.group(2)) if tail else None)
                j += 1
            if syms:
                blocks.append({
                    "quantity": quantity, "kind": kind,
                    # The fragment-averaged block combines every state,
                    # so it belongs to none of them.
                    "label": None if kind == "averaged" else state_label,
                    "header": s,
                    "symbols": syms,
                    "values": np.array(vals, float),
                    "errors": np.array(errs, float),
                    "frag_ids": (frags if any(x is not None for x in frags)
                                 else None),
                    "n_states": (nsts[0] if nsts and nsts[0] is not None
                                 else None),
                })
            i = j
            continue
        i += 1
    if not blocks:
        raise ValueError(f"No force/torque blocks found in {logpath!r}")
    return blocks


def read_omega_forces(logpath, kind="averaged"):
    """Parse OmegaQMC total forces from a postprocessing log.

    Thin wrapper over :func:`read_omega_force_blocks` kept for
    backwards compatibility.  ``kind="averaged"`` selects the
    fragment-wise BLUE-averaged block (falling back to the reference
    state when a run produced no secondary states); ``kind="state"``
    selects the **reference** state.

    .. note::
       Earlier revisions returned the *last secondary* state for
       ``kind="state"``, not the reference state.

    Returns ``(symbols, forces, forces_err)`` in Hartree/Bohr.
    """
    if kind not in ("averaged", "state"):
        raise ValueError("kind must be 'averaged' or 'state'")
    blocks = [b for b in read_omega_force_blocks(logpath)
              if b["quantity"] == "force"]
    chosen = None
    if kind == "averaged":
        chosen = next((b for b in blocks if b["kind"] == "averaged"), None)
    if chosen is None:
        chosen = next((b for b in blocks
                       if b["kind"] == "state" and b["label"] is None), None)
    if chosen is None:
        raise ValueError(f"No matching force block in {logpath!r}")
    return chosen["symbols"], chosen["values"], chosen["errors"]


# ---------------------------------------------------------------------------
# Readers: target quantum-chemistry packages
# ---------------------------------------------------------------------------

def read_g16_standard_orientation(logpath, strict=False):
    """Parse the "Standard orientation" block of a Gaussian 16 log file.

    Returns ``(symbols, coords)`` with ``coords`` an ``(N, 3)`` array in
    Angstrom (Gaussian prints orientation blocks in Angstrom).  The
    *last* "Standard orientation" block is used.  If the log has none
    (Gaussian omits it when symmetry is turned off) the last "Input
    orientation" block is used instead, which then coincides with the
    working frame; set ``strict=True`` to raise instead of falling back.
    """
    from pyscf.data.elements import ELEMENTS

    with open(logpath) as fh:
        text = fh.read()
    header = "Standard orientation:"
    if header not in text:
        if strict:
            raise ValueError(
                f"No 'Standard orientation:' block in {logpath!r}")
        header = "Input orientation:"
        warnings.warn(
            f"{logpath!r} has no 'Standard orientation:' block; falling "
            "back to 'Input orientation:'.  These coincide only when "
            "Gaussian ran with symmetry turned off.", stacklevel=2)
    idx = text.rfind(header)
    if idx < 0:
        raise ValueError(f"No orientation block found in {logpath!r}")

    rows = []
    # Skip the header line + 4 ruler/column-label lines; data rows follow.
    for line in text[idx:].splitlines()[5:]:
        if set(line.strip()) <= set("- "):
            if rows:
                break          # closing ruler of the table
            continue           # ruler above the data
        parts = line.split()
        if len(parts) == 6 and parts[0].isdigit():
            rows.append((ELEMENTS[int(parts[1])],
                         [float(parts[3]), float(parts[4]),
                          float(parts[5])]))
        elif rows:
            break
    if not rows:
        raise ValueError(f"Could not parse any atoms from {logpath!r}")
    return [r[0] for r in rows], np.array([r[1] for r in rows], float)


def read_nwchem_orientation(path):
    """Parse NWChem's output geometry (not implemented yet).

    NWChem echoes its working frame under a ``Geometry "geometry" ->
    "geometry"`` heading followed by ``Output coordinates in angstroms``
    and rows of ``index tag charge x y z``; the *last* such block is the
    geometry the calculation ran on.  Implement this to return
    ``(symbols, coords_ang)`` exactly like
    :func:`read_g16_standard_orientation`, and the rest of the reframing
    pipeline works unchanged.
    """
    raise NotImplementedError(
        "NWChem output parsing is not implemented yet; "
        "implement read_nwchem_orientation() to add it.")


def _sniff_qc_package(path, nlines=400):
    """Identify which quantum-chemistry package wrote a text output file."""
    head = []
    with open(path, errors="ignore") as fh:
        for _, line in zip(range(nlines), fh):
            head.append(line)
    text = "".join(head)
    if "Gaussian, Inc." in text or "Entering Gaussian System" in text:
        return "g16"
    if ("Northwest Computational Chemistry" in text
            or "NWChem" in text):
        return "nwchem"
    raise ValueError(
        f"Could not identify the quantum-chemistry package that wrote "
        f"{path!r}; pass package= explicitly.  Supported: "
        f"{', '.join(SUPPORTED_PACKAGES)}.")


def read_target_orientation(path, package="auto"):
    """Read a target package's working orientation.

    Dispatches to the per-package reader and returns everything in a
    common form.  ``package="auto"`` sniffs the file (see
    :func:`_sniff_qc_package`); pass ``"g16"``/``"gaussian"`` or
    ``"nwchem"`` to force it.

    Returns
    -------
    (symbols, coords_ang, package)
        Element symbols, an ``(N, 3)`` array of coordinates **in
        Angstrom**, and the resolved package name.
    """
    pkg = _sniff_qc_package(path) if package == "auto" else str(package)
    pkg = {"gaussian": "g16", "g09": "g16", "g16": "g16",
           "nwchem": "nwchem"}.get(pkg.lower(), pkg.lower())
    if pkg == "g16":
        symbols, coords = read_g16_standard_orientation(path)
    elif pkg == "nwchem":
        symbols, coords = read_nwchem_orientation(path)
    else:
        raise ValueError(
            f"Unsupported package {package!r}; supported: "
            f"{', '.join(SUPPORTED_PACKAGES)}.")
    return symbols, coords, pkg


# ---------------------------------------------------------------------------
# Rigid-body checks and the frame transform
# ---------------------------------------------------------------------------

def check_rigid_body(src_coords, tgt_coords, src_symbols=None,
                     tgt_symbols=None, atom_map=None, dist_tol=1e-4):
    """Compare two geometries for rigid-body equivalence, atom for atom.

    The pairwise distance matrix is the complete frame-independent
    invariant of a rigid body, so comparing it needs no superposition
    and localizes the offending atom pair.  It is, however, **blind to
    chirality**: an enantiomer has an identical distance matrix.  The
    post-superposition RMSD computed by
    :func:`compute_reframe_transform` is what rejects mirror images
    (note that ``det(R)`` does *not*: Kabsch always returns a proper
    rotation, so a reflected target simply fails to superpose).

    Returns a diagnostics dict with ``max_dist_dev``, ``worst_pair``,
    ``symbols_match`` and ``natoms``; it never raises.  Use
    :func:`assert_rigid_body` for the raising form.
    """
    src = np.asarray(src_coords, float)
    tgt = np.asarray(tgt_coords, float)
    if atom_map is not None:
        src = src[list(atom_map)]
        if src_symbols is not None:
            src_symbols = [src_symbols[i] for i in atom_map]
    if src.shape != tgt.shape:
        return {"natoms": (len(src), len(tgt)), "symbols_match": False,
                "max_dist_dev": float("inf"), "worst_pair": None,
                "shape_mismatch": True}

    symbols_match = True
    if src_symbols is not None and tgt_symbols is not None:
        symbols_match = ([_bare_symbol(s) for s in src_symbols]
                         == [_bare_symbol(s) for s in tgt_symbols])

    d_src = np.linalg.norm(src[:, None] - src[None], axis=-1)
    d_tgt = np.linalg.norm(tgt[:, None] - tgt[None], axis=-1)
    dev = np.abs(d_src - d_tgt)
    k = int(np.argmax(dev))
    return {"natoms": (len(src), len(tgt)), "symbols_match": symbols_match,
            "max_dist_dev": float(dev.max()),
            "worst_pair": (k // len(src), k % len(src)),
            "shape_mismatch": False,
            "dist_ok": bool(dev.max() <= dist_tol)}


def assert_rigid_body(src_coords, tgt_coords, src_symbols=None,
                      tgt_symbols=None, atom_map=None, dist_tol=1e-4):
    """Raise :class:`ValueError` unless two geometries are the same body.

    See :func:`check_rigid_body`; this is the enforcing wrapper.
    """
    d = check_rigid_body(src_coords, tgt_coords, src_symbols,
                         tgt_symbols, atom_map, dist_tol)
    if d.get("shape_mismatch"):
        raise ValueError(
            f"Atom-count mismatch: source has {d['natoms'][0]} atoms, "
            f"target has {d['natoms'][1]}.")
    if not d["symbols_match"]:
        raise ValueError(
            "Element symbols differ between the source and target "
            "geometries (after any atom_map); pass atom_map= to line "
            "the two atom orders up.")
    if not d["dist_ok"]:
        i, j = d["worst_pair"]
        raise ValueError(
            f"Geometries are not the same rigid body: interatomic "
            f"distances differ by {d['max_dist_dev']:.3e} (tolerance "
            f"{dist_tol:.1e}), worst for the atom pair ({i}, {j}).")
    return d


def _rotation_angle_axis(R):
    """Return the rotation angle (degrees) and unit axis of ``R``.

    Uses the general ``theta = arccos((tr R - 1) / 2)`` with the axis
    from the antisymmetric part, valid for any proper rotation -- unlike
    ``arctan2(R[1, 0], R[0, 0])``, which is only the rotation about z.
    """
    R = np.asarray(R, float)
    cos_t = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    theta = np.arccos(cos_t)
    if np.isclose(theta, 0.0):
        return 0.0, np.array([0.0, 0.0, 1.0])
    if np.isclose(theta, np.pi):
        # sin(theta) == 0: recover the axis from R + I, whose columns
        # are all parallel to it.
        w, v = np.linalg.eigh((R + np.eye(3)) / 2.0)
        axis = v[:, int(np.argmax(w))]
    else:
        axis = np.array([R[2, 1] - R[1, 2],
                         R[0, 2] - R[2, 0],
                         R[1, 0] - R[0, 1]]) / (2.0 * np.sin(theta))
    n = np.linalg.norm(axis)
    axis = axis / n if n > 0 else np.array([0.0, 0.0, 1.0])
    return float(np.degrees(theta)), axis


def compute_reframe_transform(src_coords, tgt_coords, src_symbols=None,
                              tgt_symbols=None, atom_map=None,
                              rmsd_tol=1e-4, dist_tol=1e-4, sv_tol=1e-6,
                              strict=True):
    """Recover the rigid map taking a source frame onto a target frame.

    Kabsch superposition, reported in **translate-then-rotate** form::

        x_target = R @ (x_source + t)

    Kabsch natively yields ``x_t = R x_s + t_post`` with ``t_post =
    c_t - R c_s``; the two are related by ``t = R.T @ t_post = R.T @ c_t
    - c_s``.  Both are returned, since confusing them would corrupt any
    reframing of *positions* while leaving forces (which need only
    ``R``) looking correct.

    Parameters
    ----------
    src_coords, tgt_coords : ndarray, shape (N, 3)
        Source and target coordinates in the **same length unit**.
    src_symbols, tgt_symbols : sequence of str, optional
        Element symbols, checked when both are given.
    atom_map : sequence of int, optional
        Permutation lining ``src_coords[atom_map]`` up with
        ``tgt_coords``.
    rmsd_tol, dist_tol : float, optional
        Superposition and interatomic-distance tolerances, in the
        coordinate unit.  Defaults suit Angstrom input: Gaussian prints
        6 decimals, so ~2e-6 A of disagreement is expected from print
        rounding alone.
    sv_tol : float, optional
        Relative threshold for calling the fit degenerate (see below).
    strict : bool, optional
        When ``True`` (default) failed checks raise; otherwise they warn.

    Returns
    -------
    dict
        ``R``, ``t``, ``t_post``, ``rmsd``, ``max_dist_dev``, ``det_R``,
        ``singular_values``, ``angle_deg``, ``axis``, ``degenerate``.

    Notes
    -----
    A **linear** molecule leaves the rotation about its own axis
    undetermined (two vanishing singular values), so transverse force
    components cannot be reframed; this is reported via ``degenerate``
    and raised under ``strict``.  A **planar** molecule (one vanishing
    singular value) is *not* a problem -- the proper-rotation constraint
    fixes the frame uniquely.
    """
    src = np.asarray(src_coords, float)
    tgt = np.asarray(tgt_coords, float)
    if atom_map is not None:
        src = src[list(atom_map)]
        if src_symbols is not None:
            src_symbols = [src_symbols[i] for i in atom_map]

    checks = (assert_rigid_body if strict else check_rigid_body)(
        src, tgt, src_symbols, tgt_symbols, None, dist_tol)

    cs, ct = src.mean(0), tgt.mean(0)
    H = (src - cs).T @ (tgt - ct)
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    t_post = ct - R @ cs                 # x_t = R x_s + t_post
    t = R.T @ ct - cs                    # x_t = R (x_s + t)

    rmsd = float(np.sqrt((((src + t) @ R.T - tgt) ** 2).sum(1).mean()))
    det_R = float(np.linalg.det(R))
    scale = max(S[0], 1e-300)
    # Two vanishing singular values => linear molecule => the rotation
    # about the molecular axis is unobservable from the geometry.
    degenerate = bool(S[1] < sv_tol * scale)
    planar = bool(S[2] < sv_tol * scale) and not degenerate

    msgs = []
    if rmsd > rmsd_tol:
        msgs.append(
            f"superposition RMSD {rmsd:.3e} exceeds {rmsd_tol:.1e}; the "
            "geometries are not the same rigid body (a mirror image / "
            "enantiomer fails here even though interatomic distances "
            "match), or the atom order differs")
    if abs(det_R - 1.0) > 1e-8:
        msgs.append(f"det(R) = {det_R:+.6f}, not +1")
    if degenerate:
        msgs.append(
            "the geometry is linear, so the rotation about the molecular "
            "axis is undetermined and transverse force components cannot "
            "be reframed; supply the rotation explicitly")
    if msgs:
        text = "compute_reframe_transform: " + "; ".join(msgs) + "."
        if strict:
            raise ValueError(text)
        warnings.warn(text, stacklevel=2)
    if planar:
        warnings.warn(
            "compute_reframe_transform: planar geometry (one vanishing "
            "singular value).  The proper-rotation constraint still "
            "determines the frame uniquely; proceeding.", stacklevel=2)

    angle_deg, axis = _rotation_angle_axis(R)
    return {"R": R, "t": t, "t_post": t_post, "rmsd": rmsd,
            "max_dist_dev": checks["max_dist_dev"], "det_R": det_R,
            "singular_values": S, "angle_deg": angle_deg, "axis": axis,
            "degenerate": degenerate, "planar": planar}


def rotate_force_covariance(R, err, cov=None):
    """Rotate per-atom force (or torque) uncertainties into a new frame.

    With only per-component error bars (``cov is None``) the source
    covariance is taken to be diagonal, ``C_i = diag(err_i ** 2)``, and
    the rotated error bars are the axis half-widths of the rotated error
    ellipsoid::

        C' = R C R.T        s'_k = sqrt(sum_j R_kj**2 * s_j**2)

    This is exact for that covariance, and preserves ``sum_k s_k**2``.
    It does ignore correlations *between* the components of one atom's
    force, which the postprocessing log does not record.  Pass an
    ``(N, 3, 3)`` ``cov`` to propagate the full covariance exactly.

    Returns
    -------
    (err_rot, cov_rot)
        Rotated per-component error bars ``(N, 3)``, and the rotated
        covariance ``(N, 3, 3)`` when ``cov`` was supplied, else
        ``None``.
    """
    R = np.asarray(R, float)
    err = np.asarray(err, float)
    if cov is None:
        return np.sqrt((err ** 2) @ (R.T ** 2)), None
    cov = np.asarray(cov, float)
    cov_rot = np.einsum('ij,njk,lk->nil', R, cov, R)
    err_rot = np.sqrt(np.clip(np.einsum('nii->ni', cov_rot), 0.0, None))
    return err_rot, cov_rot


# ---------------------------------------------------------------------------
# reframe.log:  writing, reading, and the driver
# ---------------------------------------------------------------------------

REFRAME_LOG = "reframe.log"


def format_reframe_log(xf, source_log=None, target_log=None,
                       package=None, symbols=None, atom_map=None,
                       length_unit="angstrom"):
    """Render a reframe transform as the text of a ``reframe.log``.

    The layout is meant to be read by both people and
    :func:`read_reframe_log`: ``#`` starts a comment, ``key: value``
    gives a scalar or whitespace-separated vector, and a bare ``key:``
    introduces a matrix whose indented rows follow.

    Returns a list of lines (no trailing newlines).
    """
    R, t = np.asarray(xf["R"], float), np.asarray(xf["t"], float)
    n = len(symbols) if symbols is not None else 0
    lines = [
        "# OmegaQMC reframe transform",
        "# Maps the OmegaQMC working frame onto a target package's frame.",
        "# Convention:  x_target = R @ (x_source + t)",
        "#              (translate first, then rotate)",
        "# Forces:      F_target = R @ F_source",
        "# Torques:     tau_target = R @ tau_source",
        "#              (the translation cancels for both; the torque",
        "#               rule needs det(R) = +1, asserted below)",
        "# Error bars:  rotated as the axis half-widths of the error",
        "#              ellipsoid, s'_k = sqrt(sum_j R_kj^2 s_j^2),",
        "#              i.e. assuming the source covariance is diagonal",
        "#              (the source log records no cross-component",
        "#               correlations).",
        f"# Generated:   {datetime.now().isoformat(timespec='seconds')}",
        "version: 1",
    ]
    if source_log is not None:
        lines.append(f"source_log: {os.path.abspath(source_log)}")
    if target_log is not None:
        lines.append(f"target_log: {os.path.abspath(target_log)}")
    if package is not None:
        lines.append(f"target_package: {package}")
    lines.append(f"length_unit: {length_unit}")
    if symbols is not None:
        lines.append(f"natoms: {n}")
        lines.append("symbols: " + " ".join(_bare_symbol(s)
                                            for s in symbols))
    if atom_map is not None:
        lines.append("atom_map: " + " ".join(str(int(i))
                                             for i in atom_map))
    lines += [
        f"rmsd: {xf['rmsd']:.6e}",
        f"max_dist_dev: {xf['max_dist_dev']:.6e}",
        f"det_R: {xf['det_R']:+.12f}",
        "singular_values: " + " ".join(f"{v:.6e}"
                                       for v in xf["singular_values"]),
        f"rotation_angle_deg: {xf['angle_deg']:.6f}",
        "rotation_axis: " + " ".join(f"{v:.12f}" for v in xf["axis"]),
        "translation:",
        "    " + " ".join(f"{v:>21.14e}" for v in t),
        "rotation:",
    ]
    for row in R:
        lines.append("    " + " ".join(f"{v:>21.14e}" for v in row))
    return lines


def write_reframe_log(path, xf, **kwargs):
    """Write a ``reframe.log``; see :func:`format_reframe_log`.

    Returns the path written.
    """
    with open(path, "w") as fh:
        fh.write("\n".join(format_reframe_log(xf, **kwargs)) + "\n")
    return path


def read_reframe_log(path):
    """Parse a ``reframe.log`` written by :func:`write_reframe_log`.

    Returns a dict with ``R`` (3, 3), ``t`` (3,) and whichever metadata
    the file carries (``symbols``, ``natoms``, ``length_unit``,
    ``target_package``, ``rmsd``, ...).  ``R`` is re-validated on load
    (orthogonality to 1e-10 and ``det(R) = +1``) so a hand-edited or
    truncated file cannot silently corrupt the reframed forces.
    """
    data, matrices = {}, {}
    key = None
    with open(path) as fh:
        for raw in fh:
            line = raw.rstrip("\n")
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            if line[:1].isspace() and key is not None:
                matrices[key].append([float(v) for v in line.split()])
                continue
            key = None
            if ":" not in line:
                continue
            k, _, v = line.partition(":")
            k, v = k.strip(), v.strip()
            if v == "":
                key = k
                matrices[key] = []
            else:
                data[k] = v

    out = {}
    for k, v in data.items():
        parts = v.split()
        if k in ("symbols", "target_package", "length_unit",
                 "source_log", "target_log"):
            out[k] = parts if k == "symbols" else v
            continue
        try:
            nums = [float(x) for x in parts]
        except ValueError:
            out[k] = v
            continue
        if k in ("natoms", "version"):
            out[k] = int(nums[0])
        elif k == "atom_map":
            out[k] = [int(x) for x in nums]
        else:
            out[k] = nums[0] if len(nums) == 1 else np.array(nums)

    for k, rows in matrices.items():
        out[k] = np.array(rows, float)

    if "rotation" not in out or "translation" not in out:
        raise ValueError(
            f"{path!r} is missing a 'rotation' or 'translation' block")
    R = out["rotation"]
    if R.shape != (3, 3):
        raise ValueError(f"{path!r}: rotation must be 3x3, got {R.shape}")
    t = np.asarray(out["translation"], float).reshape(-1)
    if t.size != 3:
        raise ValueError(f"{path!r}: translation must have 3 components")
    if not np.allclose(R @ R.T, np.eye(3), atol=1e-10):
        raise ValueError(f"{path!r}: rotation is not orthogonal")
    if abs(np.linalg.det(R) - 1.0) > 1e-10:
        raise ValueError(
            f"{path!r}: det(rotation) = {np.linalg.det(R):+.12f}, not +1; "
            "reflections would flip the sign of pseudovectors (torque).")
    out["R"], out["t"] = R, t
    return out


def _read_source_geometry(main_log):
    """Return ``(symbols, coords_ang, frag_ids)`` for an OmegaQMC log.

    Prefers the xyz block echoed by :func:`postproc_h5_pgcs`; falls back
    to the geometry line echoed in a VMC run's stdout (Bohr, converted
    here) so logs written before that feature still work.
    """
    try:
        return read_omega_xyz_block(main_log)
    except ValueError:
        symbols, coords_bohr, frags = read_omega_orientation(main_log)
        return symbols, coords_bohr * BOHR, frags


def make_reframe_log(main_log, target_log, package="auto", out=None,
                     atom_map=None, rmsd_tol=1e-4, dist_tol=1e-4,
                     sv_tol=1e-6, strict=True):
    """Build a ``reframe.log`` from an OmegaQMC log and a target output.

    Reads OmegaQMC's working geometry (the xyz block at the head of a
    postprocessing log, or a VMC run's stdout as a fallback), reads the
    target package's working frame via
    :func:`read_target_orientation`, asserts the two are the same rigid
    body, computes the translate-then-rotate transform, and writes it to
    ``reframe.log``.

    Both geometries are handled in Angstrom, so no Bohr conversion
    enters the transform and the packages' differing Bohr constants
    never matter.

    Parameters
    ----------
    main_log : str
        OmegaQMC postprocessing log (or VMC run stdout).
    target_log : str
        Output file of the target package.
    package : str, optional
        ``"auto"`` (default), ``"g16"`` or ``"nwchem"``.
    out : str, optional
        Where to write.  Defaults to ``reframe.log`` in the directory of
        *main_log* (the run path).
    atom_map : sequence of int, optional
        Permutation lining the OmegaQMC atom order up with the target's.
    rmsd_tol, dist_tol, sv_tol, strict :
        Passed to :func:`compute_reframe_transform`.

    Returns
    -------
    dict
        The transform dict, with ``path`` (the file written),
        ``symbols`` and ``package`` added.
    """
    src_syms, src_xyz, _ = _read_source_geometry(main_log)
    tgt_syms, tgt_xyz, pkg = read_target_orientation(target_log, package)

    xf = compute_reframe_transform(
        src_xyz, tgt_xyz, src_symbols=src_syms, tgt_symbols=tgt_syms,
        atom_map=atom_map, rmsd_tol=rmsd_tol, dist_tol=dist_tol,
        sv_tol=sv_tol, strict=strict)

    if out is None:
        out = os.path.join(
            os.path.dirname(os.path.abspath(main_log)), REFRAME_LOG)
    write_reframe_log(out, xf, source_log=main_log, target_log=target_log,
                      package=pkg, symbols=tgt_syms, atom_map=atom_map,
                      length_unit="angstrom")
    xf["path"] = out
    xf["symbols"] = list(tgt_syms)
    xf["package"] = pkg
    return xf


# ---------------------------------------------------------------------------
# Applying a recorded transform to printed forces
# ---------------------------------------------------------------------------

def reframe_forces(force_log, reframe_log=None, blocks="averaged",
                   include_torque=False, fout=sys.stdout,
                   check_invariants=True):
    """Express forces from a postprocessing log in the target frame.

    Reads the rotation recorded in a ``reframe.log`` and the per-atom
    blocks printed by :func:`postproc_h5_pgcs`, rotates each vector
    (``F -> R F``) and its error bars (see
    :func:`rotate_force_covariance`), and prints the result in the same
    column layout as the source blocks.

    Parameters
    ----------
    force_log : str
        Postprocessing log holding the force blocks.
    reframe_log : str, optional
        Transform to apply.  Defaults to ``reframe.log`` beside
        *force_log*.
    blocks : str, optional
        ``"averaged"`` (default; the production result, falling back to
        the reference state), ``"reference"``, ``"all"``, or a specific
        secondary-state label.
    include_torque : bool, optional
        Also reframe the torque blocks.  Torque is a pseudovector; under
        a proper rotation it transforms as ``tau' = R tau`` and the
        translation cancels.  Default ``False``.
    fout : file object, optional
        Where to print.  Defaults to ``sys.stdout``.
    check_invariants : bool, optional
        Report ``max |d|F_i||`` and ``max |d sum_k s_k^2|``, both of
        which a rotation must preserve.  These validate the code path,
        not the diagonal-covariance assumption.

    Returns
    -------
    list of dict
        The reframed blocks, each the source dict with ``values``,
        ``errors`` and ``header`` replaced.
    """
    if reframe_log is None:
        reframe_log = os.path.join(
            os.path.dirname(os.path.abspath(force_log)), REFRAME_LOG)
    xf = read_reframe_log(reframe_log)
    R = xf["R"]
    pkg = xf.get("target_package", "target")

    found = read_omega_force_blocks(force_log)
    wanted = [b for b in found
              if include_torque or b["quantity"] == "force"]
    if blocks == "all":
        chosen = wanted
    elif blocks == "averaged":
        chosen = [b for b in wanted if b["kind"] == "averaged"]
        if not chosen:
            chosen = [b for b in wanted
                      if b["kind"] == "state" and b["label"] is None]
    elif blocks == "reference":
        chosen = [b for b in wanted
                  if b["kind"] == "state" and b["label"] is None]
    else:
        chosen = [b for b in wanted if b["label"] == blocks]
    if not chosen:
        raise ValueError(
            f"No blocks matching {blocks!r} in {force_log!r}")

    angle, axis = xf.get("rotation_angle_deg"), xf.get("rotation_axis")
    print(f"# Forces reframed into the {pkg} working frame", file=fout)
    if angle is not None and axis is not None:
        print(f"#   rotation {float(angle):.6f} deg about "
              f"[{axis[0]:.6f}, {axis[1]:.6f}, {axis[2]:.6f}]; "
              f"det(R) = {np.linalg.det(R):+.6f}", file=fout)
    if "rmsd" in xf:
        print(f"#   frame alignment RMSD {float(xf['rmsd']):.3e} "
              f"{xf.get('length_unit', '')}".rstrip() + " (from "
              f"{os.path.basename(reframe_log)})", file=fout)
    print("#   error bars: diagonal-covariance ellipsoid projection; "
          "correlations", file=fout)
    print("#   between force components are not recorded by the source "
          "log.", file=fout)

    out_blocks = []
    for b in chosen:
        v_rot = b["values"] @ R.T
        e_rot, _ = rotate_force_covariance(R, b["errors"])
        header = b["header"].rstrip(")")
        header = (f"{header}, reframed to {pkg})" if b["header"].endswith(")")
                  else f"{b['header']} (reframed to {pkg})")
        print("", file=fout)
        if b["label"] is not None:
            print(f"--- Secondary state: {b['label']} ---", file=fout)
        print(header, file=fout)
        for i, s in enumerate(b["symbols"]):
            tail = ""
            if b["frag_ids"] is not None and b["frag_ids"][i] is not None:
                tail = (f"  (fragment {b['frag_ids'][i]},"
                        f" over {b['n_states']} states)")
            print("{:4s}{:>16.6g} ± {:>12.6g}{:>16.6g} ± {:>12.6g}"
                  "{:>16.6g} ± {:>12.6g}{}".format(
                      s, v_rot[i, 0], e_rot[i, 0], v_rot[i, 1],
                      e_rot[i, 1], v_rot[i, 2], e_rot[i, 2], tail),
                  file=fout)
        tot = v_rot.sum(0)
        dtot = np.sqrt((e_rot ** 2).sum(0))
        label = ("Total force on system" if b["quantity"] == "force"
                 else "Total torque")
        print(f"{label} (reframed to {pkg})", file=fout)
        print("    {:>16.6g} ± {:>12.6g}{:>16.6g} ± {:>12.6g}"
              "{:>16.6g} ± {:>12.6g}".format(
                  tot[0], dtot[0], tot[1], dtot[1], tot[2], dtot[2]),
              file=fout)

        if check_invariants:
            dn = np.abs(np.linalg.norm(v_rot, axis=1)
                        - np.linalg.norm(b["values"], axis=1)).max()
            dt = np.abs((e_rot ** 2).sum(1)
                        - (b["errors"] ** 2).sum(1)).max()
            print(f"#   invariants: max |d|v_i|| = {dn:.2e}, "
                  f"max |d sum_k s_k^2| = {dt:.2e}", file=fout)

        nb = dict(b)
        nb.update({"values": v_rot, "errors": e_rot, "header": header})
        out_blocks.append(nb)
    return out_blocks


def align_forces_to_reference(ref_coords, work_coords, work_forces,
                              work_forces_err=None, atom_map=None,
                              rmsd_warn=1e-2):
    """Rotate per-atom force vectors into a reference geometry's frame.

    Convenience wrapper around :func:`compute_reframe_transform` and
    :func:`rotate_force_covariance` for callers holding arrays rather
    than log files.  ``ref_coords`` and ``work_coords`` must share a
    length unit and atom order (see ``atom_map``).

    Returns a dict with ``R``, ``t`` (translate-then-rotate), ``t_post``
    (rotate-then-translate), ``rmsd``, ``forces`` in the reference
    frame, ``forces_err`` (or ``None``) and ``ref_coords``.
    """
    ref = np.asarray(ref_coords, float)
    work = np.asarray(work_coords, float)
    forces = np.asarray(work_forces, float)
    err = None if work_forces_err is None \
        else np.asarray(work_forces_err, float)
    if atom_map is not None:
        sel = list(atom_map)
        work, forces = work[sel], forces[sel]
        if err is not None:
            err = err[sel]

    xf = compute_reframe_transform(work, ref, rmsd_tol=rmsd_warn,
                                   dist_tol=rmsd_warn, strict=False)
    err_rot = None if err is None \
        else rotate_force_covariance(xf["R"], err)[0]
    return {"R": xf["R"], "t": xf["t"], "t_post": xf["t_post"],
            "rmsd": xf["rmsd"], "forces": forces @ xf["R"].T,
            "forces_err": err_rot, "ref_coords": ref}


def compare_forces_with_g16(g16_log, omq_geom_log, omq_force_log=None,
                            forces=None, forces_err=None,
                            force_kind="averaged", atom_map=None,
                            print_table=True):
    """Deprecated: use :func:`make_reframe_log` + :func:`reframe_forces`.

    Kept as a shim over the generalized pipeline.  The replacement is::

        make_reframe_log(omq_geom_log, g16_log, package="g16")
        reframe_forces(omq_force_log)
    """
    warnings.warn(
        "compare_forces_with_g16 is deprecated; use make_reframe_log() "
        "to record the frame transform and reframe_forces() to apply it.",
        DeprecationWarning, stacklevel=2)

    src_syms, src_xyz, _ = _read_source_geometry(omq_geom_log)
    tgt_syms, tgt_xyz, _ = read_target_orientation(g16_log, "g16")
    xf = compute_reframe_transform(
        src_xyz, tgt_xyz, src_symbols=src_syms, tgt_symbols=tgt_syms,
        atom_map=atom_map, strict=False)

    if forces is None:
        if omq_force_log is None:
            raise ValueError("provide either `forces` or `omq_force_log`")
        _, forces, forces_err = read_omega_forces(omq_force_log,
                                                  kind=force_kind)
    forces = np.asarray(forces, float)
    forces_err = None if forces_err is None \
        else np.asarray(forces_err, float)
    err_rot = None if forces_err is None \
        else rotate_force_covariance(xf["R"], forces_err)[0]

    res = {"R": xf["R"], "t": xf["t"], "t_post": xf["t_post"],
           "rmsd": xf["rmsd"], "forces": forces @ xf["R"].T,
           "forces_err": err_rot, "ref_coords": tgt_xyz / BOHR,
           "symbols": list(tgt_syms)}
    if print_table:
        print(f"# OmegaQMC forces reframed to g16 "
              f"(rotation {xf['angle_deg']:.4f} deg, "
              f"RMSD {xf['rmsd']:.2e} A)")
        for i, s in enumerate(res["symbols"]):
            e = res["forces_err"]
            row = "  ".join(
                f"{res['forces'][i, k]:9.5f}"
                + (f"±{e[i, k]:.5f}" if e is not None else "")
                for k in range(3))
            print(f"{s:>3s}{i:<2d}  {row}")
    return res
