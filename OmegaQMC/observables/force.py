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
import jax.numpy as jnp

from ..utils import (
    batched_binning_analysis_grds,
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
    get_psi_mo,
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
    get_psi_mo : callable
        MO evaluator (used by redistribution scheme 1).
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
    # --- Space-warping redistribution schemes ---
    @jax.jit
    def _redistribute_scheme1(elec_crds):
        _, mo_val_s = get_psi_mo(elec_crds, nuc_crds)
        weight = jnp.einsum(
            'neo,neo->en', mo_val_s, mo_val_s,
        )
        return weight / jnp.sum(
            weight, axis=-1, keepdims=True,
        )

    @jax.jit
    def _redistribute_scheme2(elec_crds):
        diff = elec_crds[:, None, :] - nuc_crds[None, :, :]
        dist = jnp.sqrt(jnp.sum(diff**2, axis=-1))
        dist = jnp.where(dist < eps, eps, dist)
        weight = dist**(-4.0)
        return weight / jnp.sum(
            weight, axis=-1, keepdims=True,
        )

    rescale_fn = (
        _redistribute_scheme2
        if 'scheme2' in gr_scheme
        else _redistribute_scheme1
    )
    jac_rescale_fn = jax.jacobian(
        rescale_fn, argnums=0,
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

        rescale = jax.vmap(
            rescale_fn,
        )(batch_samples)
        jac_rescale_elc = jax.vmap(
            jac_rescale_fn,
        )(batch_samples)
        novel_correction = 0.5 * jnp.einsum(
            'beneK->bnK', jac_rescale_elc,
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
):
    """Build a JIT-compiled ZVZB gradient batch function.

    Implements the zero-variance zero-bias (ZVZB)
    nuclear force estimator of Assaraf, Caffarel &
    Khelif :cite:`Assaraf2003`, but returns the
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

    Returns
    -------
    callable
        ``(batch_walkers) -> (grd_ee_en, grd_ke,
        grd_logpsi)`` where *batch_walkers* has shape
        ``(batch, nelec, 3)`` and each output has
        shape ``(batch, natom, 3)``.
    """
    from ..psi.nn.physics import laplacian

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

    # --- Decomposed ZVZB gradient (single walker) ---
    @jax.jit
    def _grd_zvzb(elec_crds):
        f_en = _force_en_bare(elec_crds)
        grad_lp = _grad_nuc_log_psi(elec_crds)
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

        # gradient = -force; bare HF: grd_ee_en = -f_en
        # (V_ee has no R-dependence, so the ee piece
        # is identically zero)
        grd_ee_en = -f_en
        # ZV kinetic correction (gradient sign)
        grd_ke = (ke_dpsi_all - ke_psi) * grad_lp
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
    dict_grd_samples, block_nums, enr_mean,
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

    Parameters
    ----------
    dict_grd_samples : dict
        Nested dict of per-block HDF5 data.
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
    # Collect all per-walker data across blocks
    all_O = []
    all_dEL_dp = []
    all_dEL_dR = []
    all_dlogpsi_dR = []

    for bc in block_nums:
        bcs = f'{bc}'
        all_O.append(
            dict_grd_samples['O_params'][bcs]
        )
        all_dEL_dp.append(
            dict_grd_samples['dEL_dparams'][bcs]
        )
        all_dEL_dR.append(
            dict_grd_samples['dEL_dR_nuc'][bcs]
        )
        all_dlogpsi_dR.append(
            dict_grd_samples[
                'dlogpsi_dR_nuc'
            ][bcs]
        )

    # (total_walkers, n_params) etc.
    O = jnp.concatenate(all_O, axis=0)
    dEL_dp = jnp.concatenate(all_dEL_dp, axis=0)
    dEL_dR = jnp.concatenate(all_dEL_dR, axis=0)
    dlp_dR = jnp.concatenate(
        all_dlogpsi_dR, axis=0,
    )

    n_w = O.shape[0]
    n_p = O.shape[1]
    n_nuc = dEL_dR.shape[1]

    # Center quantities
    O_mean = O.mean(axis=0)
    dEL_dp_mean = dEL_dp.mean(axis=0)
    dEL_dR_mean = dEL_dR.mean(axis=0)
    dlp_dR_mean = dlp_dR.mean(axis=0)

    dO = O - O_mean[None, :]
    ddEL_dp = dEL_dp - dEL_dp_mean[None, :]
    ddEL_dR = dEL_dR - dEL_dR_mean[None, :, :]
    ddlp_dR = dlp_dR - dlp_dR_mean[None, :, :]

    # H_pp (n_p × n_p): leading-order Hessian
    # H_pp[i,j] = 2*Cov(dEL/dp_i, O_j)
    #           + 2*Cov(dEL/dp_j, O_i)
    cov_ij = (ddEL_dp.T @ dO) / n_w
    H_pp = 2.0 * (cov_ij + cov_ij.T)

    # H_pR (n_p × n_nuc × 3)
    # H_pR[i,A,k] = 2*Cov(dEL/dR_{A,k}, O_i)
    #             + 2*Cov(dEL/dp_i, dlogpsi/dR_{A,k})
    H_pR = jnp.zeros((n_p, n_nuc, 3))
    for A in range(n_nuc):
        for k in range(3):
            cov_RO = (
                ddEL_dR[:, A, k] @ dO
            ) / n_w
            cov_pR = (
                ddEL_dp.T @ ddlp_dR[:, A, k]
            ) / n_w
            H_pR = H_pR.at[:, A, k].set(
                2.0 * (cov_RO + cov_pR)
            )

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

    with h5py.File(ofname_grd, 'r') as f:
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

        dict_grd_samples = {}
        for key, val in f.items():
            if isinstance(val, h5py.Group):
                dict_grd_samples[key] = {}
                for key2, val2 in val.items():
                    if isinstance(
                        val2, h5py.Group
                    ):
                        dict_grd_samples[key][key2] \
                            = {}
                        for key3, val3 in (
                            val2.items()
                        ):
                            if not val3.shape:
                                dict_grd_samples[
                                    key][key2][key3] \
                                    = val3[()].decode()
                            else:
                                dict_grd_samples[
                                    key][key2][key3] \
                                    = jnp.array(val3)
                    elif not val2.shape:
                        dict_grd_samples[key][key2] \
                            = val2[()].decode()
                    else:
                        dict_grd_samples[key][key2] \
                            = jnp.array(val2)
            elif val.ndim == 0:
                dict_grd_samples[key] = val[()]
            else:
                dict_grd_samples[key] = (
                    jnp.array(val[:])
                )

        block_nums = [
            int(k) for k in
            dict_grd_samples["local_energies"]
            .keys()
            if k.isdigit()
        ]
        block_nums.sort()

        loc_e_list = []
        for block_cnt in block_nums:
            local_energies = dict_grd_samples[
                "local_energies"
            ][f'{block_cnt}']
            loc_e_list.append(
                jnp.array(local_energies)
            )
        enr_mean = jnp.vstack(loc_e_list).mean()

        grd_nn = dict_grd_samples['grd_nn']

        # Identify combo labels from fragment_weights
        combo_labels = []
        if 'fragment_weights' in dict_grd_samples:
            combo_labels = sorted(
                k for k in dict_grd_samples[
                    'fragment_weights'
                ]
                if isinstance(
                    dict_grd_samples[
                        'fragment_weights'
                    ][k],
                    dict,
                )
            )
        states = [None] + combo_labels

        # Detect parameter-response data (ZVZB2)
        has_pr = ('O_params' in dict_grd_samples)
        dp_dR = None
        if has_pr and states[0] is None:
            dp_dR = _solve_param_response(
                dict_grd_samples, block_nums,
                enr_mean,
            )

        if ofname_log is None:
            fout = sys.stdout
        else:
            fout = open(ofname_log, 'w', 1)

        ref_grd_tot = None
        ref_grd_err = None
        all_state_results = {}

        for state_label in states:
            valid_samples_count = 0
            grd_ke_sum = 0.0
            grd_ee_en_sum = 0.0
            grd_pulay_sum = 0.0

            grd_tot_list = []
            grd_err_list = []

            for block_cnt in block_nums:
                if state_label is None:
                    grd_ee_en = dict_grd_samples['grd_ee_en'][f'{block_cnt}']
                    grd_ke = dict_grd_samples['grd_ke'][f'{block_cnt}']
                    grd_logpsi = dict_grd_samples['grd_logpsi'][f'{block_cnt}']
                else:
                    grd_ee_en = dict_grd_samples[
                        'grd_ee_en'
                    ][state_label][f'{block_cnt}']
                    grd_ke = dict_grd_samples[
                        'grd_ke'
                    ][state_label][f'{block_cnt}']
                    grd_logpsi = dict_grd_samples[
                        'grd_logpsi'
                    ][state_label][f'{block_cnt}']
                local_energies = jnp.array(
                    dict_grd_samples['local_energies'][f'{block_cnt}']
                )

                # Pulay force contribution
                d_enr = local_energies - enr_mean

                _, num_nuc, _ = grd_ee_en.shape
                num_steps_per_block, num_walkers = local_energies.shape

                # Regroup
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
                    (num_steps_per_block, num_walkers, num_nuc, 3),
                )
                grd_arrays = [
                    grd_nn_sw,
                    grd_ee_en, grd_ke,
                    grd_pulay,
                ]
                grd_tot_sw = jnp.stack(grd_arrays, axis=0).sum(axis=0)

                # Parameter-response correction
                if dp_dR is not None and state_label is None:
                    O_p = dict_grd_samples['O_params'][f'{block_cnt}']
                    dEL_p = dict_grd_samples['dEL_dparams'][f'{block_cnt}']
                    # n_s = num_steps_per_block * num_walkers
                    n_jp = O_p.shape[-1]
                    O_p = O_p.reshape(
                        num_steps_per_block,
                        num_walkers, n_jp,
                    )
                    dEL_p = dEL_p.reshape(
                        num_steps_per_block,
                        num_walkers, n_jp,
                    )
                    # correction per walker:
                    # dEL_p + 2*(E_L-<E>)*O_p
                    cv = dEL_p + 2.0 * (d_enr[..., None] * O_p)
                    # dp_dR: (n_jp, natom, 3)
                    delta_f = jnp.einsum(
                        'swi,inK->swnK', cv, dp_dR,
                    )
                    grd_tot_sw = grd_tot_sw + delta_f

                # Load fragment weights for
                # secondary states
                if state_label is not None:
                    frag_w = dict_grd_samples[
                        'fragment_weights'
                    ][state_label][f'{block_cnt}']
                    frag_w = frag_w.reshape(
                        num_steps_per_block,
                        num_walkers,
                    )
                else:
                    frag_w = None

                # Compute forces and error
                xbar, serr, sdev, kappa = (
                    batched_binning_analysis_grds(
                        grd_tot_sw,
                        walker_based_batch_size,
                        weights=frag_w,
                    )
                )
                grd_tot_list.append(xbar[None, :, :, :])
                grd_err_list.append(serr[None, :, :, :])

                grd_ee_en_sum += grd_ee_en.sum(axis=0)
                grd_ke_sum += grd_ke.sum(axis=0)
                grd_pulay_sum += grd_pulay.sum(axis=0)

                valid_samples_count += local_energies.shape[0]

            # Compute averages
            if valid_samples_count > 0:
                grd_ee_en = grd_ee_en_sum / valid_samples_count
                grd_ke = grd_ke_sum / valid_samples_count
                grd_pulay = grd_pulay_sum / valid_samples_count

                grd_tot_bw = jnp.concatenate(
                    grd_tot_list, axis=0,
                )
                grd_err_bw = jnp.concatenate(
                    grd_err_list, axis=0,
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
                    jnp.linalg.norm(
                        grd_err, axis=0,
                    )
                    / grd_err.shape[0]
                )

                # Compute torques and error
                torque, dtau = (
                    compute_torque_with_error(
                        myMol, grd_tot, grd_err,
                    )
                )

                grd_ee_en = jnp.mean(
                    grd_ee_en, axis=0,
                )
                grd_ke = jnp.mean(
                    grd_ke, axis=0,
                )
                grd_pulay = jnp.mean(
                    grd_pulay, axis=0,
                )
            else:
                grd_ee_en = jnp.zeros_like(grd_nn)
                grd_ke = jnp.zeros_like(grd_nn)
                grd_pulay = jnp.zeros_like(grd_nn)

            # Save reference results for return
            if state_label is None:
                ref_grd_tot = grd_tot
                ref_grd_err = grd_err

            all_state_results[state_label] = (
                grd_tot, grd_err,
            )

            # Write results
            with jnp.printoptions(
                precision=12, suppress=True,
            ):
                if state_label is not None:
                    print(
                        "\n--- Secondary state:"
                        f" {state_label} ---",
                        file=fout,
                    )

                print(
                    'NN gradients\n',
                    grd_nn, file=fout,
                )
                print(
                    'ee+eN gradients\n',
                    grd_ee_en, file=fout,
                )
                print(
                    'KE gradients\n',
                    grd_ke, file=fout,
                )
                print(
                    'Pulay gradients\n',
                    grd_pulay, file=fout,
                )
                print(
                    'Total gradients\n',
                    grd_tot, file=fout,
                )

                fout.write(
                    "Total forces (-gradients)\n"
                )
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

        # --- Fragment-wise averaging of PGCS ---
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

            avg_grd_tot = jnp.zeros((num_nuc, 3))
            avg_grd_err = jnp.zeros((num_nuc, 3))

            for i in range(num_nuc):
                fid = atom_frag_map[i]
                relevant_states = frag_to_states.get(fid, [None])
                forces = jnp.stack([all_state_results[s][0][i]
                                    for s in relevant_states])
                errors = jnp.stack([all_state_results[s][1][i]
                                    for s in relevant_states])
                N = len(relevant_states)
                avg_grd_tot = avg_grd_tot.at[i] .set(forces.mean(axis=0))
                avg_grd_err = (
                    avg_grd_err.at[i].set(
                        jnp.sqrt(
                            jnp.sum(
                                errors**2, axis=0,
                            )
                        )
                        / N
                    )
                )

            avg_torque, avg_dtau = (
                compute_torque_with_error(
                    myMol, avg_grd_tot, avg_grd_err,
                )
            )

            with jnp.printoptions(
                precision=12, suppress=True,
            ):
                fout.write("\n\tFragment-wise averaged forces\n")
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
