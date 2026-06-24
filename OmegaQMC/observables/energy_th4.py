"""
th4 chunked multi-determinant local-energy kernel, kept in a
separate module so it coexists with sh-new's unchunked
local_energy_multidet in observables/energy.py.
"""
from functools import partial

import jax
import jax.numpy as jnp

from OmegaQMC.observables.greens import _gf_spin_single_det
from OmegaQMC.observables.greens_th4 import _pad_and_chunk


def local_energy_multidet(
    h1e, chol, phia, phib,
    trials_up, trials_dn, ci_coeffs,
    enuc, det_chunk_size=5,
):
    """Fused multi-det GF + energy with chunked scan.

    Computes the Green's function and local energy in a single
    ``jax.lax.scan`` loop over determinant chunks. Per-det
    Ghalf arrays are never stored beyond one chunk, and
    half-rotated Cholesky vectors are computed on-the-fly,
    so the (ndet, naux, nocc, nbasis) per-det rchol storage
    is avoided.

    Args:
        h1e: One-body Hamiltonian, shape (nbasis, nbasis).
        chol: Cholesky vectors, shape (naux, nbasis, nbasis).
        phia: Walker alpha orbitals,
            shape (nwalkers, nbasis, nup).
        phib: Walker beta orbitals,
            shape (nwalkers, nbasis, ndown).
        trials_up: shape (ndet, nbasis, nup).
        trials_dn: shape (ndet, nbasis, ndown).
        ci_coeffs: shape (ndet,).
        enuc: Nuclear repulsion energy.
        det_chunk_size: Dets per scan step (static).

    Returns:
        e_tot, e_1b, e_2b: shape (nwalkers,) each.
    """
    nwalkers = phia.shape[0]
    nbasis = phia.shape[1]

    tu_c, td_c, ci_c = _pad_and_chunk(
        trials_up, trials_dn, ci_coeffs, det_chunk_size)

    def _e2b_one(Gha_I, Ghb_I, tu_I, td_I):
        ra = jnp.einsum('pi,gpq->giq', tu_I.conj(), chol)
        rb = jnp.einsum('pi,gpq->giq', td_I.conj(), chol)
        ec, ex = local_energy_2body(Gha_I, Ghb_I, ra, rb)
        return ec - ex  # (nwalkers,)

    def _scan_body(carry, xs):
        Ga_acc, Gb_acc, e2b_acc, ovlp_acc = carry
        t_up, t_dn, ci = xs

        Gha, oa = jax.vmap(
            _gf_spin_single_det, in_axes=(None, 0),
        )(phia, t_up)
        Ghb, ob = jax.vmap(
            _gf_spin_single_det, in_axes=(None, 0),
        )(phib, t_dn)

        Gha = jnp.where(jnp.isnan(Gha), 0.0, Gha)
        Ghb = jnp.where(jnp.isnan(Ghb), 0.0, Ghb)

        w = ci.conj()[:, None] * oa * ob
        ovlp_acc = ovlp_acc + jnp.sum(w, axis=0)

        w4 = w[:, :, None, None]
        Ga_acc = Ga_acc + jnp.sum(
            w4 * jnp.einsum(
                'dpi,dwiq->dwpq', t_up.conj(), Gha),
            axis=0)
        Gb_acc = Gb_acc + jnp.sum(
            w4 * jnp.einsum(
                'dpi,dwiq->dwpq', t_dn.conj(), Ghb),
            axis=0)

        e2b_chunk = jax.vmap(_e2b_one)(
            Gha, Ghb, t_up, t_dn)  # (chunk, nwalkers)
        e2b_acc = e2b_acc + jnp.sum(w * e2b_chunk, axis=0)

        return (Ga_acc, Gb_acc, e2b_acc, ovlp_acc), None

    dtype = jnp.complex128
    init = (
        jnp.zeros((nwalkers, nbasis, nbasis), dtype=dtype),
        jnp.zeros((nwalkers, nbasis, nbasis), dtype=dtype),
        jnp.zeros((nwalkers,), dtype=dtype),
        jnp.zeros((nwalkers,), dtype=dtype),
    )

    (Ga_acc, Gb_acc, e2b_acc, overlap), _ = jax.lax.scan(
        _scan_body, init, (tu_c, td_c, ci_c))

    Ga = Ga_acc / overlap[:, None, None]
    Gb = Gb_acc / overlap[:, None, None]

    e_1b = local_energy_1body(h1e, Ga, Gb, enuc)
    e_2b = e2b_acc / overlap
    e_tot = e_1b + e_2b
    return e_tot, e_1b, e_2b




# ====================================================================
#  th5 addition: chunked equivalent of sh-new's local_energy_multidet.
#  Same signature, same return, but the per-det two-body energy vmap
#  is split into ``det_chunk_size`` slices and the weighted-sum
#  numerator / denominator are accumulated outside the loop.
# ====================================================================
def local_energy_multidet_shnew_chunked(
    h1e, Ga, Gb, Ghalfa_all, Ghalfb_all,
    rchols_a, rchols_b, ci_coeffs,
    ovlp_a_all, ovlp_b_all, enuc,
    det_chunk_size=5,
):
    """Chunked drop-in for sh-new's ``local_energy_multidet``.

    Returns ``(e_tot, e_1b, e_2b)`` — each shape ``(nwalkers,)`` —
    identical to the sh-new function modulo floating-point
    accumulation order.

    Memory: peak intermediate scales with
    ``det_chunk_size * nwalkers * naux * nocc`` instead of the full
    ``ndet * nwalkers * naux * nocc`` materialised by sh-new's vmap.
    """
    from OmegaQMC.observables.energy import (
        local_energy_1body, local_energy_2body,
    )

    e_1b = local_energy_1body(h1e, Ga, Gb, enuc)         # (nwalkers,)
    ndet = int(ci_coeffs.shape[0])

    e_2b_num = jnp.zeros_like(e_1b)
    overlap = jnp.zeros_like(e_1b)

    def _e2b_single(Ghalfa_I, Ghalfb_I, rchol_a_I, rchol_b_I):
        e_coul, e_exch = local_energy_2body(
            Ghalfa_I, Ghalfb_I, rchol_a_I, rchol_b_I)
        return e_coul - e_exch                            # (nwalkers,)

    _vmap_e2b = jax.vmap(_e2b_single)

    for s in range(0, ndet, det_chunk_size):
        e = min(s + det_chunk_size, ndet)
        Ghalfa_c = Ghalfa_all[s:e]
        Ghalfb_c = Ghalfb_all[s:e]
        rchol_a_c = rchols_a[s:e]
        rchol_b_c = rchols_b[s:e]
        ci_c = ci_coeffs[s:e]
        ovlp_a_c = ovlp_a_all[s:e]
        ovlp_b_c = ovlp_b_all[s:e]

        e_2b_chunk = _vmap_e2b(
            Ghalfa_c, Ghalfb_c, rchol_a_c, rchol_b_c)     # (nc, w)

        w_I_chunk = (ci_c.conj()[:, None]
                     * ovlp_a_c * ovlp_b_c)                # (nc, w)

        e_2b_num = e_2b_num + jnp.sum(w_I_chunk * e_2b_chunk, axis=0)
        overlap = overlap + jnp.sum(w_I_chunk, axis=0)

    e_2b = e_2b_num / overlap
    e_tot = e_1b + e_2b
    return e_tot, e_1b, e_2b


# ====================================================================
#  th5 addition: walker chunking on top of det chunking for the
#  QED-AFQMC block-end local energy. Mirrors th4's
#  ``_local_energy_multidet_walker_chunked`` pattern (non-QED) but for
#  the QED kernel — splits walkers into ``walker_chunk_size`` chunks
#  and calls the supplied qed_local_energy_multidet on each chunk,
#  concatenating the per-walker return.
#
#  Signature mirrors the QED LE entry point so the driver can pass
#  through unchanged kwargs.
# ====================================================================
def qed_local_energy_multidet_walker_chunked(
    qed_local_energy_multidet_fn,
    h1e, Ga, Gb, Ghalfa_all, Ghalfb_all,
    rchols_a, rchols_b, ci_coeffs,
    ovlp_a_all, ovlp_b_all, enuc,
    q, dip_mo, omega, s, q0,
    det_chunk_size=5, walker_chunk_size=None,
):
    """Walker chunking wrapper for the QED multi-det local energy.

    Args:
        qed_local_energy_multidet_fn: The actual QED LE function (i.e.
            ``qed_local_energy_multidet`` from qed_afqmc_gto.py). Passed
            in to avoid a circular import.
        h1e, Ga, Gb, Ghalfa_all, Ghalfb_all, rchols_a, rchols_b,
        ci_coeffs, ovlp_a_all, ovlp_b_all, enuc: as in
            ``qed_local_energy_multidet``.
        q: photon coordinate, shape (nwalkers,).
        dip_mo, omega, s, q0: QED params.
        det_chunk_size: forwarded to the LE function for det chunking.
        walker_chunk_size: walkers per chunk. None or >= nwalkers skips
            chunking entirely.

    Returns:
        Same 4-tuple as ``qed_local_energy_multidet``: each
        ``(nwalkers,)``.
    """
    nw = int(Ga.shape[0])
    if walker_chunk_size is None or walker_chunk_size >= nw:
        return qed_local_energy_multidet_fn(
            h1e, Ga, Gb, Ghalfa_all, Ghalfb_all,
            rchols_a, rchols_b, ci_coeffs,
            ovlp_a_all, ovlp_b_all, enuc,
            q, dip_mo, omega, s, q0,
            det_chunk_size=det_chunk_size,
        )

    e_tot_chunks, e_1b_chunks, e_2b_chunks, e_ph_chunks = [], [], [], []
    for s_w in range(0, nw, walker_chunk_size):
        e_w = min(s_w + walker_chunk_size, nw)
        et, e1, e2, eph = qed_local_energy_multidet_fn(
            h1e,
            Ga[s_w:e_w], Gb[s_w:e_w],
            Ghalfa_all[:, s_w:e_w], Ghalfb_all[:, s_w:e_w],
            rchols_a, rchols_b, ci_coeffs,
            ovlp_a_all[:, s_w:e_w], ovlp_b_all[:, s_w:e_w], enuc,
            q[s_w:e_w], dip_mo, omega, s, q0,
            det_chunk_size=det_chunk_size,
        )
        e_tot_chunks.append(et)
        e_1b_chunks.append(e1)
        e_2b_chunks.append(e2)
        e_ph_chunks.append(eph)
    return (jnp.concatenate(e_tot_chunks),
            jnp.concatenate(e_1b_chunks),
            jnp.concatenate(e_2b_chunks),
            jnp.concatenate(e_ph_chunks))
