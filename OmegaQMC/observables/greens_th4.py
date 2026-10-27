"""
th4 chunked multidet Green's function helpers, kept in a separate
module so they coexist with sh-new's unchunked multidet API in the
main greens.py. Bodies are byte-identical to th4's originals.
"""
from functools import partial

import jax
import jax.numpy as jnp

from OmegaQMC.observables.greens import (
    _gf_spin_single_det,
    _overlap_only_spin,
)


# A determinant I whose overlap with a walker, |det(T_I^† phi_a) det(T_I^† phi_b)|,
# is below max(OVLP_MASK_RTOL * max_J |o_J|, OVLP_MASK_ATOL) contributes nothing
# to G or the energy:
# its overlap matrix is (numerically) singular, so the inverse is inf or huge.
# Typical case: the initial walkers are copies of det 0 and are exactly
# orthogonal to other determinants at the first step of every run.
OVLP_MASK_RTOL = 1e-14
OVLP_MASK_ATOL = 1e-300


def _det_overlap_max(phia, phib, trials_up, trials_dn):
    """max_I |det(T_I^† phi_a) det(T_I^† phi_b)| per walker, shape (nwalkers,)."""
    oa = jax.vmap(_overlap_only_spin, in_axes=(None, 0))(phia, trials_up)
    ob = jax.vmap(_overlap_only_spin, in_axes=(None, 0))(phib, trials_dn)
    return jnp.max(jnp.abs(oa * ob), axis=0)


def _mask_small_overlap(Gha, Ghb, oa, ob, omax):
    """Zero Ghalf of dets with |o_I| < max(OVLP_MASK_RTOL * omax, OVLP_MASK_ATOL).

    Gha/Ghb: (ndet, nwalkers, nocc, nbasis); oa/ob: (ndet, nwalkers);
    omax: (nwalkers,). jnp.where keeps inf/huge values of masked dets
    out of every later product.
    """
    tol = jnp.maximum(OVLP_MASK_RTOL * omax, OVLP_MASK_ATOL)
    keep = jnp.abs(oa * ob) >= tol[None, :]
    k4 = keep[:, :, None, None]
    return jnp.where(k4, Gha, 0.0), jnp.where(k4, Ghb, 0.0)


def _pad_and_chunk(trials_up, trials_dn, ci_coeffs, chunk_size):
    """Pad det arrays to a multiple of chunk_size and reshape.

    Zero-padded dets have ci=0 so they contribute nothing.
    All shapes are static (known at trace time).
    """
    ndet = trials_up.shape[0]
    npad = (-ndet) % chunk_size
    if npad > 0:
        trials_up = jnp.concatenate([
            trials_up,
            jnp.zeros((npad, *trials_up.shape[1:]),
                       dtype=trials_up.dtype)])
        trials_dn = jnp.concatenate([
            trials_dn,
            jnp.zeros((npad, *trials_dn.shape[1:]),
                       dtype=trials_dn.dtype)])
        ci_coeffs = jnp.concatenate([
            ci_coeffs,
            jnp.zeros(npad, dtype=ci_coeffs.dtype)])
    nc = (ndet + npad) // chunk_size
    return (
        trials_up.reshape(nc, chunk_size, *trials_up.shape[1:]),
        trials_dn.reshape(nc, chunk_size, *trials_dn.shape[1:]),
        ci_coeffs.reshape(nc, chunk_size),
    )


@partial(jax.jit, static_argnames=['det_chunk_size'])
def greens_function_multidet(
    phia, phib, trials_up, trials_dn, ci_coeffs,
    det_chunk_size=5,
):
    """Multi-determinant Green's function (chunked scan).

    Processes determinants in chunks of ``det_chunk_size`` via
    ``jax.lax.scan``, accumulating Ga, Gb, overlap without ever
    storing the full per-det Ghalf arrays. Peak working set is
    bounded by ``det_chunk_size * nwalkers * nocc * nbasis``
    rather than ``ndet * nwalkers * nocc * nbasis``.

    Args:
        phia: shape (nwalkers, nbasis, nup).
        phib: shape (nwalkers, nbasis, ndown).
        trials_up: shape (ndet, nbasis, nup).
        trials_dn: shape (ndet, nbasis, ndown).
        ci_coeffs: shape (ndet,).
        det_chunk_size: Dets per scan step (static).

    Returns:
        Ga, Gb: Full GF, shape (nwalkers, nbasis, nbasis).
        overlap: Multi-det overlap, shape (nwalkers,).
    """
    nwalkers = phia.shape[0]
    nbasis = phia.shape[1]

    tu_c, td_c, ci_c = _pad_and_chunk(
        trials_up, trials_dn, ci_coeffs, det_chunk_size)
    omax = _det_overlap_max(phia, phib, trials_up, trials_dn)

    def _scan_body(carry, xs):
        Ga_acc, Gb_acc, ovlp_acc = carry
        t_up, t_dn, ci = xs

        Gha, oa = jax.vmap(
            _gf_spin_single_det, in_axes=(None, 0),
        )(phia, t_up)
        Ghb, ob = jax.vmap(
            _gf_spin_single_det, in_axes=(None, 0),
        )(phib, t_dn)

        Gha, Ghb = _mask_small_overlap(Gha, Ghb, oa, ob, omax)

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

        return (Ga_acc, Gb_acc, ovlp_acc), None

    dtype = jnp.complex128
    init = (
        jnp.zeros((nwalkers, nbasis, nbasis), dtype=dtype),
        jnp.zeros((nwalkers, nbasis, nbasis), dtype=dtype),
        jnp.zeros((nwalkers,), dtype=dtype),
    )

    (Ga_acc, Gb_acc, overlap), _ = jax.lax.scan(
        _scan_body, init, (tu_c, td_c, ci_c))

    Ga = Ga_acc / overlap[:, None, None]
    Gb = Gb_acc / overlap[:, None, None]

    return Ga, Gb, overlap


@partial(jax.jit, static_argnames=[])
def greens_function_multidet_force_bias(
    phia, phib, trials_up, trials_dn, ci_coeffs,
):
    """Multi-det Green's function for force bias / streamed energy.

    Retained for the legacy ``afqmc_gto_estream`` driver. The main
    ``afqmc_gto`` driver uses the chunked-scan
    :func:`greens_function_multidet` instead, which avoids storing
    the per-det ``Ghalf`` tensor at all.
    """
    Ghalfa_all, ovlp_a_all = jax.vmap(
        _gf_spin_single_det, in_axes=(None, 0),
    )(phia, trials_up)
    Ghalfb_all, ovlp_b_all = jax.vmap(
        _gf_spin_single_det, in_axes=(None, 0),
    )(phib, trials_dn)

    Ghalfa_all, Ghalfb_all = _mask_small_overlap(
        Ghalfa_all, Ghalfb_all, ovlp_a_all, ovlp_b_all,
        jnp.max(jnp.abs(ovlp_a_all * ovlp_b_all), axis=0),
    )

    w_I = (
        ci_coeffs.conj()[:, None]
        * ovlp_a_all * ovlp_b_all
    )
    overlap = jnp.sum(w_I, axis=0)

    return (
        Ghalfa_all, Ghalfb_all,
        overlap, ovlp_a_all, ovlp_b_all,
    )


@partial(jax.jit, static_argnames=['det_chunk_size'])
def greens_function_multidet_overlap_only(
    phia, phib, trials_up, trials_dn, ci_coeffs,
    det_chunk_size=5,
):
    """Multi-determinant overlap only (chunked scan).

    Lightweight variant for the post-propagation weight update
    (step 7) — no inverse, no Ghalf, no full G.

    Args:
        phia: shape (nwalkers, nbasis, nup).
        phib: shape (nwalkers, nbasis, ndown).
        trials_up: shape (ndet, nbasis, nup).
        trials_dn: shape (ndet, nbasis, ndown).
        ci_coeffs: shape (ndet,).
        det_chunk_size: Dets per scan step (static).

    Returns:
        overlap: Multi-det overlap, shape (nwalkers,).
    """
    nwalkers = phia.shape[0]

    tu_c, td_c, ci_c = _pad_and_chunk(
        trials_up, trials_dn, ci_coeffs, det_chunk_size)

    def _scan_body(ovlp_acc, xs):
        t_up, t_dn, ci = xs
        oa = jax.vmap(
            _overlap_only_spin, in_axes=(None, 0),
        )(phia, t_up)
        ob = jax.vmap(
            _overlap_only_spin, in_axes=(None, 0),
        )(phib, t_dn)
        w = ci.conj()[:, None] * oa * ob
        return ovlp_acc + jnp.sum(w, axis=0), None

    overlap, _ = jax.lax.scan(
        _scan_body,
        jnp.zeros((nwalkers,), dtype=jnp.complex128),
        (tu_c, td_c, ci_c))

    return overlap


# ====================================================================
#  th5 addition: chunked greens_function_multidet that returns the
#  FULL 7-tuple (matches sh-new's signature) but builds per-det
#  quantities in det_chunk_size slices to control peak memory.
#  Used only by the QED-AFQMC block-end local-energy step where the
#  per-det Ghalf / overlap arrays are needed by qed_local_energy_multidet.
# ====================================================================

def greens_function_multidet_chunked_full(
    phia, phib, trials_up, trials_dn, ci_coeffs,
    det_chunk_size=5,
):
    """Chunked equivalent of sh-new's ``greens_function_multidet``.

    Same return contract as sh-new's version:

        Ga, Gb, Ghalfa_all, Ghalfb_all, overlap, ovlp_a_all, ovlp_b_all

    Shapes:
        Ga, Gb:                          (nwalkers, nbasis, nbasis)
        Ghalfa_all, Ghalfb_all:          (ndet, nwalkers, nocc, nbasis)
        overlap:                         (nwalkers,)
        ovlp_a_all, ovlp_b_all:          (ndet, nwalkers)

    The per-det arrays are built one ``det_chunk_size`` slice at a
    time and concatenated. The aggregate Ga/Gb/overlap is accumulated
    inside the loop so the inner einsum sees only the chunked
    intermediate tensor.
    """
    ndet     = int(trials_up.shape[0])
    nwalkers = int(phia.shape[0])
    nbasis   = int(phia.shape[1])
    dtype    = jnp.complex128

    Ga = jnp.zeros((nwalkers, nbasis, nbasis), dtype=dtype)
    Gb = jnp.zeros((nwalkers, nbasis, nbasis), dtype=dtype)
    overlap = jnp.zeros((nwalkers,), dtype=dtype)

    Ghalfa_chunks = []
    Ghalfb_chunks = []
    ovlp_a_chunks = []
    ovlp_b_chunks = []

    n_chunks = (ndet + det_chunk_size - 1) // det_chunk_size
    omax = _det_overlap_max(phia, phib, trials_up, trials_dn)

    # Per-chunk single-det Greens evaluator (vmap of sh-new's helper).
    _vmap_gf = jax.vmap(_gf_spin_single_det, in_axes=(None, 0))

    for c in range(n_chunks):
        s = c * det_chunk_size
        e = min(s + det_chunk_size, ndet)

        tu_c = trials_up[s:e]              # (nc, nbasis, nup)
        td_c = trials_dn[s:e]              # (nc, nbasis, ndn)
        ci_c = ci_coeffs[s:e]              # (nc,)

        # sh-new convention: Ghalf has shape (w, nocc, nbasis); vmap over
        # the det axis lifts to (nc, w, nocc, nbasis). Same overlap.
        Ghalfa_chunk, ovlp_a_chunk = _vmap_gf(phia, tu_c)
        Ghalfb_chunk, ovlp_b_chunk = _vmap_gf(phib, td_c)

        # Drop dets with (numerically) singular overlap matrices.
        Ghalfa_chunk, Ghalfb_chunk = _mask_small_overlap(
            Ghalfa_chunk, Ghalfb_chunk, ovlp_a_chunk, ovlp_b_chunk, omax)

        Ghalfa_chunks.append(Ghalfa_chunk)
        Ghalfb_chunks.append(Ghalfb_chunk)
        ovlp_a_chunks.append(ovlp_a_chunk)
        ovlp_b_chunks.append(ovlp_b_chunk)

        # Per-det weight for this chunk (matches sh-new exactly):
        #   w_I = c_I.conj() * O_I^a * O_I^b
        w_I_chunk = (ci_c.conj()[:, None]
                     * ovlp_a_chunk * ovlp_b_chunk)        # (nc, w)
        overlap = overlap + jnp.sum(w_I_chunk, axis=0)

        # Aggregate Ga += sum_d w_I[d,w] * trial[d,p,i] . Ghalfa[d,w,i,q]
        w_expanded = w_I_chunk[:, :, None, None]
        Ga_inc = jnp.sum(
            w_expanded * jnp.einsum(
                'dpi,dwiq->dwpq', tu_c.conj(), Ghalfa_chunk),
            axis=0,
        )
        Gb_inc = jnp.sum(
            w_expanded * jnp.einsum(
                'dpi,dwiq->dwpq', td_c.conj(), Ghalfb_chunk),
            axis=0,
        )
        Ga = Ga + Ga_inc
        Gb = Gb + Gb_inc

    Ghalfa_all = jnp.concatenate(Ghalfa_chunks, axis=0)
    Ghalfb_all = jnp.concatenate(Ghalfb_chunks, axis=0)
    ovlp_a_all = jnp.concatenate(ovlp_a_chunks, axis=0)
    ovlp_b_all = jnp.concatenate(ovlp_b_chunks, axis=0)

    Ga = Ga / overlap[:, None, None]
    Gb = Gb / overlap[:, None, None]

    return (Ga, Gb, Ghalfa_all, Ghalfb_all,
            overlap, ovlp_a_all, ovlp_b_all)
