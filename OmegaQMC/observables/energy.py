"""
Local energy estimators for AFQMC.

The estimator helpers (``local_energy*`` family) are pure and
suitable for use inside ``@jax.jit`` boundaries defined in the
driver modules.

:func:`postproc_h5_pgcs` is a separate post-processing entry
point that reads a VMC gradient HDF5 file and reports the
correlated-sample-averaged VMC energy.  It is the energy
analogue of :func:`OmegaQMC.observables.force.postproc_h5_pgcs`.
"""

import sys

import jax
import h5py
import numpy as np
import jax.numpy as jnp
from functools import partial

from ..utils import (
    do_binning_analysis,
    blue_combine_states,
    equilibration_length,
)


def local_energy_1body(h1e, Ga, Gb, enuc):
    """One-body contribution to the local energy.

    E_1b = Tr(h1e @ Ga) + Tr(h1e @ Gb) + E_nuc

    Args:
        h1e: shape (nbasis, nbasis).
        Ga: shape (nwalkers, nbasis, nbasis).
        Gb: shape (nwalkers, nbasis, nbasis).
        enuc: Nuclear repulsion energy.

    Returns:
        e_1b: shape (nwalkers,).
    """
    e_1b_a = jnp.einsum('pq,wqp->w', h1e, Ga)
    e_1b_b = jnp.einsum('pq,wqp->w', h1e, Gb)
    return e_1b_a + e_1b_b + enuc


def local_energy_2body(Ghalfa, Ghalfb, rchol_a, rchol_b):
    """Two-body contribution to the local energy.

    Uses half-rotated Cholesky vectors for efficiency.

    Coulomb:
        E_coul = 0.5 * sum_g (X_a^g + X_b^g)^2
    Exchange:
        E_exch = 0.5 * sum_g (Tr(T_a^g T_a^{gT})
                             + Tr(T_b^g T_b^{gT}))

    Args:
        Ghalfa: shape (nwalkers, nup, nbasis).
        Ghalfb: shape (nwalkers, ndown, nbasis).
        rchol_a: shape (naux, nup, nbasis).
        rchol_b: shape (naux, ndown, nbasis).

    Returns:
        e_coul: Coulomb energy, shape (nwalkers,).
        e_exch: Exchange energy, shape (nwalkers,).
    """
    # Coulomb: X_s^g = Tr(rchol_s^g * Ghalf_s^T)
    Xa = jnp.einsum('giq,wiq->gw', rchol_a, Ghalfa)
    Xb = jnp.einsum('giq,wiq->gw', rchol_b, Ghalfb)

    e_coul = 0.5 * jnp.sum((Xa + Xb) ** 2, axis=0)

    # Exchange: T_s^g[i,j] = sum_q rchol_s[g,i,q] Ghalf_s[w,j,q]
    Ta = jnp.einsum('giq,wjq->gwij', rchol_a, Ghalfa)
    Tb = jnp.einsum('giq,wjq->gwij', rchol_b, Ghalfb)

    # Tr(T @ T^T) = sum_{ij} T[i,j]^2
    e_exch_a = 0.5 * jnp.sum(
        Ta * Ta.transpose(0, 1, 3, 2), axis=(0, 2, 3)
    )
    e_exch_b = 0.5 * jnp.sum(
        Tb * Tb.transpose(0, 1, 3, 2), axis=(0, 2, 3)
    )
    e_exch = e_exch_a + e_exch_b

    return e_coul, e_exch


@partial(jax.jit, static_argnames=[])
def local_energy(
    h1e, Ga, Gb, Ghalfa, Ghalfb,
    rchol_a, rchol_b, enuc,
):
    """Mixed-estimator local energy for all walkers.

    E_loc = E_1body + E_coulomb - E_exchange + E_nuc

    Uses half-rotated Cholesky vectors for efficiency.

    Args:
        h1e: One-body Hamiltonian, shape (nbasis, nbasis).
        Ga: Alpha GF, shape (nwalkers, nbasis, nbasis).
        Gb: Beta GF, shape (nwalkers, nbasis, nbasis).
        Ghalfa: Half-rotated alpha GF,
            shape (nwalkers, nup, nbasis).
        Ghalfb: Half-rotated beta GF,
            shape (nwalkers, ndown, nbasis).
        rchol_a: Half-rotated Cholesky (alpha),
            shape (naux, nup, nbasis).
        rchol_b: Half-rotated Cholesky (beta),
            shape (naux, ndown, nbasis).
        enuc: Nuclear repulsion energy.

    Returns:
        e_tot: Total local energy, shape (nwalkers,).
        e_1b: One-body energy, shape (nwalkers,).
        e_2b: Two-body energy, shape (nwalkers,).
    """
    e_1b = local_energy_1body(h1e, Ga, Gb, enuc)
    e_coul, e_exch = local_energy_2body(
        Ghalfa, Ghalfb, rchol_a, rchol_b,
    )
    e_2b = e_coul - e_exch
    e_tot = e_1b + e_2b
    return e_tot, e_1b, e_2b


@partial(jax.jit, static_argnames=[])
def _e2b_slab_spin(Ghalf, rchol_slab):
    """Per-slab Coulomb partial and exchange contribution.

    Args:
        Ghalf: shape (nwalkers, nocc, nbasis).
        rchol_slab: shape (g_slab, nocc, nbasis).

    Returns:
        X_slab: shape (g_slab, nwalkers), Coulomb intermediate.
        e_exch_slab: shape (nwalkers,), exchange contribution from
            this slab (already 0.5 * sum_g_in_slab Tr(T_g T_g^T)).
    """
    X_slab = jnp.einsum('giq,wiq->gw', rchol_slab, Ghalf)
    T_slab = jnp.einsum('giq,wjq->gwij', rchol_slab, Ghalf)
    e_exch_slab = 0.5 * jnp.sum(
        T_slab * T_slab.transpose(0, 1, 3, 2), axis=(0, 2, 3),
    )
    return X_slab, e_exch_slab


def local_energy_2body_streamed(
    Ghalfa, Ghalfb, rchol_a, rchol_b, e_chunk_g=16,
):
    """Streamed two-body energy: caps exchange-tensor peak memory.

    Computes the same e_coul and e_exch as ``local_energy_2body``
    but slabs the auxiliary axis so the (g, w, nocc, nocc) exchange
    intermediate ``T`` is never materialized in full.  Peak slab
    size is (e_chunk_g, nwalkers, nocc, nocc) complex128 per spin.

    The Coulomb sum (X_a + X_b)^2 mixes the two spins, so we
    accumulate X_a, X_b per slab and form the cross sum slab-by-slab.

    Args:
        Ghalfa: shape (nwalkers, nup, nbasis).
        Ghalfb: shape (nwalkers, ndown, nbasis).
        rchol_a: shape (naux, nup, nbasis).
        rchol_b: shape (naux, ndown, nbasis).
        e_chunk_g: slab size along the auxiliary axis.

    Returns:
        e_coul: shape (nwalkers,).
        e_exch: shape (nwalkers,).
    """
    naux = rchol_a.shape[0]
    nw = Ghalfa.shape[0]
    dtype = Ghalfa.dtype

    e_coul = jnp.zeros((nw,), dtype=dtype)
    e_exch = jnp.zeros((nw,), dtype=dtype)
    for g0 in range(0, naux, e_chunk_g):
        g1 = min(g0 + e_chunk_g, naux)
        Xa, exa = _e2b_slab_spin(Ghalfa, rchol_a[g0:g1])
        Xb, exb = _e2b_slab_spin(Ghalfb, rchol_b[g0:g1])
        e_coul = e_coul + 0.5 * jnp.sum((Xa + Xb) ** 2, axis=0)
        e_exch = e_exch + exa + exb
    return e_coul, e_exch


@partial(jax.jit, static_argnames=[])
def _e1b_no_full_g(h1e_trial_a, h1e_trial_b, Ghalfa, Ghalfb, enuc):
    """One-body trace without building full G.

    Uses Tr(h1e G_s) = Tr((h1e @ trial_s.conj()) Ghalf_s^T):

        e_1b[w] = einsum('pi,wip->w', h1e_trial_a, Ghalfa)
                + einsum('pi,wip->w', h1e_trial_b, Ghalfb)
                + enuc

    Args:
        h1e_trial_a: (nbasis, nup),  h1e @ trial_up.conj().
        h1e_trial_b: (nbasis, ndown), h1e @ trial_dn.conj().
        Ghalfa: (nwalkers, nup, nbasis).
        Ghalfb: (nwalkers, ndown, nbasis).
        enuc: nuclear repulsion energy.

    Returns:
        e_1b: (nwalkers,).
    """
    e_1b_a = jnp.einsum('pi,wip->w', h1e_trial_a, Ghalfa)
    e_1b_b = jnp.einsum('pi,wip->w', h1e_trial_b, Ghalfb)
    return e_1b_a + e_1b_b + enuc


def local_energy_streamed(
    h1e_trial_a, h1e_trial_b, Ghalfa, Ghalfb,
    rchol_a, rchol_b, enuc, e_chunk_g=16,
):
    """Mixed-estimator local energy without full G, streamed two-body.

    Equivalent to ``local_energy`` on the surviving outputs, but:
    - 1-body trace uses precontracted ``h1e_trial_{a,b}`` and the
      half-rotated Green's function, skipping the
      (nwalkers, nbasis, nbasis) full-G materialization.
    - 2-body uses ``local_energy_2body_streamed`` to cap the exchange
      intermediate at (e_chunk_g, nwalkers, nocc, nocc).

    Args:
        h1e_trial_a: (nbasis, nup),  h1e @ trial_up.conj().
        h1e_trial_b: (nbasis, ndown), h1e @ trial_dn.conj().
        Ghalfa: (nwalkers, nup, nbasis).
        Ghalfb: (nwalkers, ndown, nbasis).
        rchol_a: (naux, nup, nbasis).
        rchol_b: (naux, ndown, nbasis).
        enuc: nuclear repulsion energy.
        e_chunk_g: slab size for the streamed exchange.

    Returns:
        e_tot, e_1b, e_2b: (nwalkers,) each.
    """
    e_1b = _e1b_no_full_g(
        h1e_trial_a, h1e_trial_b, Ghalfa, Ghalfb, enuc,
    )
    e_coul, e_exch = local_energy_2body_streamed(
        Ghalfa, Ghalfb, rchol_a, rchol_b, e_chunk_g,
    )
    e_2b = e_coul - e_exch
    e_tot = e_1b + e_2b
    return e_tot, e_1b, e_2b


@partial(jax.jit, static_argnames=[])
def local_energy_multidet(
    h1e, Ga, Gb, Ghalfa_all, Ghalfb_all,
    rchols_a, rchols_b, ci_coeffs,
    ovlp_a_all, ovlp_b_all, enuc,
):
    """Multi-determinant mixed-estimator local energy.

    One-body uses the aggregate full G.  Two-body requires
    per-det Ghalf paired with per-det half-rotated Cholesky
    vectors.

    Args:
        h1e: One-body Hamiltonian, shape (nbasis, nbasis).
        Ga, Gb: Full GF, shape (nwalkers, nbasis, nbasis).
        Ghalfa_all: Per-det half-rotated alpha GF,
            shape (ndet, nwalkers, nup, nbasis).
        Ghalfb_all: Per-det half-rotated beta GF,
            shape (ndet, nwalkers, ndn, nbasis).
        rchols_a: Per-det half-rotated Cholesky (alpha),
            shape (ndet, naux, nup, nbasis).
        rchols_b: Per-det half-rotated Cholesky (beta),
            shape (ndet, naux, ndn, nbasis).
        ci_coeffs: CI coefficients, shape (ndet,).
        ovlp_a_all: Per-det alpha overlaps,
            shape (ndet, nwalkers).
        ovlp_b_all: Per-det beta overlaps,
            shape (ndet, nwalkers).
        enuc: Nuclear repulsion energy.

    Returns:
        e_tot, e_1b, e_2b: shape (nwalkers,) each.
    """
    # One-body: uses aggregate full G
    e_1b = local_energy_1body(h1e, Ga, Gb, enuc)

    # Two-body: per-det, then weighted sum
    def _e2b_single_det(
        Ghalfa_I, Ghalfb_I, rchol_a_I, rchol_b_I,
    ):
        e_coul, e_exch = local_energy_2body(
            Ghalfa_I, Ghalfb_I, rchol_a_I, rchol_b_I,
        )
        return e_coul - e_exch  # (nwalkers,)

    e_2b_all = jax.vmap(_e2b_single_det)(
        Ghalfa_all, Ghalfb_all, rchols_a, rchols_b,
    )
    # e_2b_all: (ndet, nwalkers)

    # Per-det weights
    w_I = (
        ci_coeffs.conj()[:, None] * ovlp_a_all * ovlp_b_all
    )
    overlap = jnp.sum(w_I, axis=0)  # (nwalkers,)

    # Weighted two-body energy
    e_2b = jnp.sum(w_I * e_2b_all, axis=0) / overlap

    e_tot = e_1b + e_2b
    return e_tot, e_1b, e_2b


def local_energy_multidet_streamed(
    h1e_trials_a, h1e_trials_b,
    Ghalfa_all, Ghalfb_all,
    rchols_a, rchols_b, ci_coeffs,
    ovlp_a_all, ovlp_b_all, enuc, e_chunk_g=16,
):
    """Multi-determinant mixed-estimator local energy, streamed.

    Same algebra as ``local_energy_multidet`` but:
    - 1-body is computed per determinant via precontracted
      ``h1e_trials_{a,b}`` and the half-rotated Green's
      function, skipping the (nwalkers, nbasis, nbasis)
      aggregate-G build.
    - 2-body slabs the auxiliary axis per determinant via
      ``local_energy_2body_streamed``, capping the exchange
      intermediate at (e_chunk_g, nwalkers, nocc, nocc)
      regardless of ``ndet``.

    Args:
        h1e_trials_a: Precontracted ``h1e @ trials_up.conj()``,
            shape (ndet, nbasis, nup).
        h1e_trials_b: Precontracted ``h1e @ trials_dn.conj()``,
            shape (ndet, nbasis, ndown).
        Ghalfa_all: Per-det half-rotated alpha GF,
            shape (ndet, nwalkers, nup, nbasis).
        Ghalfb_all: Per-det half-rotated beta GF,
            shape (ndet, nwalkers, ndown, nbasis).
        rchols_a: Per-det half-rotated Cholesky (alpha),
            shape (ndet, naux, nup, nbasis).
        rchols_b: Per-det half-rotated Cholesky (beta),
            shape (ndet, naux, ndown, nbasis).
        ci_coeffs: CI coefficients, shape (ndet,).
        ovlp_a_all: Per-det alpha overlaps,
            shape (ndet, nwalkers).
        ovlp_b_all: Per-det beta overlaps,
            shape (ndet, nwalkers).
        enuc: Nuclear repulsion energy.
        e_chunk_g: Slab size along the auxiliary axis.

    Returns:
        e_tot, e_1b, e_2b: shape (nwalkers,) each.
    """
    ndet = rchols_a.shape[0]

    # Per-det weights and aggregate overlap.
    w_I = (
        ci_coeffs.conj()[:, None] * ovlp_a_all * ovlp_b_all
    )
    overlap = jnp.sum(w_I, axis=0)

    # Per-det 1-body trace without full G.
    # einsum('dpi, dwip -> dw', h1e_trials, Ghalf_all)
    e_1b_per_det_a = jnp.einsum(
        'dpi,dwip->dw', h1e_trials_a, Ghalfa_all,
    )
    e_1b_per_det_b = jnp.einsum(
        'dpi,dwip->dw', h1e_trials_b, Ghalfb_all,
    )
    e_1b_per_det = e_1b_per_det_a + e_1b_per_det_b

    # Aggregate 1-body via weighted average; add enuc once.
    e_1b = jnp.sum(w_I * e_1b_per_det, axis=0) / overlap + enuc

    # Per-det 2-body via streamed exchange.  Python-for caps
    # the exchange intermediate at single-det scale regardless
    # of ndet.
    e_2b_per_det_list = []
    for I in range(ndet):
        e_coul_I, e_exch_I = local_energy_2body_streamed(
            Ghalfa_all[I], Ghalfb_all[I],
            rchols_a[I], rchols_b[I], e_chunk_g,
        )
        e_2b_per_det_list.append(e_coul_I - e_exch_I)
    e_2b_per_det = jnp.stack(e_2b_per_det_list, axis=0)

    e_2b = jnp.sum(w_I * e_2b_per_det, axis=0) / overlap

    e_tot = e_1b + e_2b
    return e_tot, e_1b, e_2b


def postproc_h5_pgcs(
        prefix: str = "vmc",
        logfile: bool | str = False,
        equil_cutoff: int | str = 0,
        ) -> tuple[float, float]:
    """Post-process VMC sample data to obtain \
correlated-sample-averaged VMC energy.

    Reads the local-energy samples written by
    :func:`save_gto_gradients` /
    :func:`save_nn_gradients` to ``<prefix>.grd.h5``
    and applies Point Group Correlated Sampling (PGCS)
    to obtain the symmetry-averaged total energy and
    its statistical error.  The energy analogue of
    :func:`OmegaQMC.observables.force.postproc_h5_pgcs`.

    Each block contributes one scalar per state — the
    flat mean of reference local energies, or the
    fragment-weighted global mean of combo local
    energies (``sum(w·E_trans)/sum(w)``).
    ``do_binning_analysis`` on each per-state block
    sequence yields the autocorrelation-corrected
    standard error and effective sample count
    ``N_eff = N_blocks / kappa``.

    The combined VMC energy is then the **Best Linear
    Unbiased Estimator** (BLUE) of the common
    expectation under the linear constraint
    ``sum(w_k) = 1``:

    .. math:: \\hat\\mu_\\mathrm{BLUE}
        = \\frac{\\mathbf{1}^\\top \\Sigma^{-1}
        \\hat{\\mathbf{E}}}
        {\\mathbf{1}^\\top \\Sigma^{-1} \\mathbf{1}},
        \\quad
        \\mathrm{Var}(\\hat\\mu_\\mathrm{BLUE})
        = \\frac{1}{\\mathbf{1}^\\top \\Sigma^{-1}
        \\mathbf{1}}.

    The :math:`K \\times K` covariance matrix
    :math:`\\Sigma` of the per-state mean estimators
    is constructed from bin-mean cross-covariance:
    block series are partitioned into bins of size
    ``b = ceil(2 · max_k κ_k)`` so that bin means are
    approximately decorrelated, the sample covariance
    of the ``N_b = T // b`` bin-mean vectors gives
    :math:`S \\in \\mathbb{R}^{K \\times K}`, and
    :math:`\\Sigma = S / N_b` is the resulting
    covariance of the per-state mean estimators —
    which captures both within-series autocorrelation
    (via the bin size) and cross-series correlations
    at matched block index.

    Parameters
    ----------
    prefix : str, optional
        File-name stem used when the VMC run was set up
        (the ``prefix`` argument of
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
    equil_cutoff : int or str, optional
        Number of leading blocks to discard as
        equilibration before the time-series analysis;
        equivalently the block index (in sorted order) at
        which the analysis starts.  Applied uniformly to
        every state.  Pass the string ``"auto"`` to detect
        the cutoff from the reference per-block energy
        series via
        :func:`OmegaQMC.utils.equilibration_length`.
        Default is ``0`` (use all blocks).

    Returns
    -------
    enr : float
        BLUE-combined VMC total energy in Hartree.
        When no PGCS combos are present the reference
        (un-displaced) estimate is returned.
    enr_err : float
        Statistical error (standard error of the mean)
        in Hartree.
    """
    suffixes_checked = [".chk.h5", ".grd.h5"]
    for s in suffixes_checked:
        if prefix.endswith(s):
            prefix = prefix[:-len(s)]
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
        block_nums = sorted(
            int(k) for k in f['local_energies']
            if k.isdigit()
        )

        # Combo labels: sub-groups under fragment_weights.
        combo_labels = []
        if 'fragment_weights' in f:
            combo_labels = sorted(
                k for k in f['fragment_weights']
                if isinstance(
                    f['fragment_weights'][k], h5py.Group,
                )
            )
        states = [None] + combo_labels

        if ofname_log is None:
            fout = sys.stdout
        else:
            fout = open(ofname_log, 'w', 1)

        # Resolve the equilibration cutoff (auto or int)
        # and drop the leading blocks before any analysis.
        if equil_cutoff == "auto":
            # Detect on the reference per-block mean-energy
            # series (one scalar per block).
            ref_series = [
                float(np.asarray(
                    f['local_energies'][f'{b}']
                ).mean(dtype=np.float64))
                for b in block_nums
            ]
            equil_cutoff = equilibration_length(ref_series)
            print(
                "Auto-detected equilibration at block"
                f" index {equil_cutoff}",
                file=fout,
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

        # Per-state per-block scalar energies.
        E_b_per_state = {s: [] for s in states}

        for block_cnt in block_nums:
            # Reference: flat mean over (S, W) samples.
            le_ref = np.asarray(
                f['local_energies'][f'{block_cnt}']
            )
            E_b_per_state[None].append(
                float(le_ref.mean(dtype=np.float64))
            )

            for label in combo_labels:
                # Combo: importance-reweighted global mean
                # — sum(w * E_trans) / sum(w) — over the
                # flat (S*W,) sample axis.  Matches the
                # in-driver ``combo_block_E`` recipe.
                c_E = np.asarray(
                    f['local_energies']
                    [label][f'{block_cnt}'],
                    dtype=np.float64,
                )
                w = np.asarray(
                    f['fragment_weights']
                    [label][f'{block_cnt}'],
                    dtype=np.float64,
                )
                E_b_per_state[label].append(
                    float((w * c_E).sum() / w.sum())
                )

        # Per-state: bin-FFT autocorrelation analysis
        # gives kappa, which both sets the BLUE bin
        # size and lets us report per-state N_eff.
        all_state_results = {}
        kappa_per_state = {}
        for state in states:
            E_blocks = jnp.asarray(E_b_per_state[state])
            xbar, serr, _, kappa = do_binning_analysis(
                E_blocks,
            )
            kappa_per_state[state] = float(kappa)
            neff = E_blocks.shape[0] / float(kappa)
            all_state_results[state] = (
                float(xbar), float(serr), neff,
            )
            label_txt = (
                "reference" if state is None else state
            )
            print(
                f"E[{label_txt}]"
                f" = {float(xbar):.8f}"
                f" ± {float(serr):.8f}"
                f"  (N_eff = {neff:.1f})",
                file=fout,
            )

        ref_enr, ref_err, _ = all_state_results[None]

        if combo_labels:
            blue_mean, blue_err, blue_neff, blue_w, bin_size = (
                blue_combine_states(
                    E_b_per_state, states,
                    kappa_per_label=kappa_per_state,
                )
            )
            print(
                "\nBLUE-combined E"
                f" = {blue_mean:.8f}"
                f" ± {blue_err:.8f}"
                f"  (N_eff = {blue_neff:.1f},"
                f" bin size = {bin_size} blocks,"
                f" over {len(states)} states)",
                file=fout,
            )
            print("  BLUE weights:", file=fout)
            for state, wk in zip(states, blue_w):
                label_txt = (
                    "reference"
                    if state is None else state
                )
                print(
                    f"    {label_txt:<24s}"
                    f" = {float(wk):+.6f}",
                    file=fout,
                )
            ref_enr = blue_mean
            ref_err = blue_err

        if ofname_log is not None:
            fout.close()

    return ref_enr, ref_err
