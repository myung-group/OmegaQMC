"""
Generalized Wick's theorem for particle-hole multi-determinant trials.

The trial is an expansion |Psi_T> = sum_I c_I |D_I> in orthonormal
orbitals, and each |D_I> is written as an excitation of a reference
determinant |D_0> (the first determinant).  All walker-dependent
quantities are evaluated relative to |D_0>, as in Mahajan & Sharma,
JCTC 17, 4786 (2021); Mahajan, Lee & Sharma, JCP 156, 174111 (2022);
and ipie (Malone et al., JCTC 19, 109 (2023)):

    Theta = phi (D_0^T phi)^{-1}                  (nbasis, nocc)

and, for a spin string with holes h (positions in the reference) and
particles p (k of each),

    <D_I|phi> / <D_0|phi> = det(A),     A = Theta[p, h]   (k, k).

One-body and two-body matrix elements over <D_I| and |phi> follow from
the Woodbury form of the transition Green's function,

    Theta_I = Theta - Theta[:, h] A^{-1} (Theta[p, :] - E_h^T),

multiplied by det(A), so that only the first- and second-order
cofactors of A appear.  Unlike explicit per-determinant inverses, this
stays exact when det(A) = 0, e.g. when a walker equals |D_0>.

Spin strings are shared between determinants, so the string quantities
are computed once per unique alpha and beta string and then combined.
Only the Cholesky rows of the reference and particle orbitals are
needed (``chol_rows``).

The walker-side functions are pure and jitted; they take a
:class:`WickTrial` built once by :func:`build_wick_trial`.
"""

import itertools
from functools import partial
from typing import NamedTuple

import numpy as np
import jax
import jax.numpy as jnp

from OmegaQMC.integrals.cholesky import _iter_chol_g_chunks


class WickSpin(NamedTuple):
    """Spin-resolved data of a particle-hole trial.

    Attributes:
        ref_occ: Reference occupied orbitals, shape (nocc,).
        rows: Orbitals needed for the string quantities, shape (nrow,);
            the reference orbitals come first, then all particles.
        chol_rows: Cholesky rows ``chol[:, rows, :]``,
            shape (naux, nrow, nbasis).
        h1e_rows: One-body rows ``h1e[rows, :]``, shape (nrow, nbasis).
        holes: Per excitation level k = 1, 2, ... present in the trial,
            hole positions in ``ref_occ``, shape (nstr_k, k).
        parts: Matching particle positions in ``rows``,
            shape (nstr_k, k).

    The strings are indexed as 0 (the reference), then the strings of
    ``holes[0]``, ``holes[1]``, ... in order.
    """
    ref_occ: jnp.ndarray
    rows: jnp.ndarray
    chol_rows: jnp.ndarray
    h1e_rows: jnp.ndarray
    holes: tuple
    parts: tuple


class WickTrial(NamedTuple):
    """Particle-hole trial for the generalized Wick's theorem.

    Attributes:
        up, dn: :class:`WickSpin` data per spin.
        coeffs: conj(c_I) times the permutation signs that relate each
            determinant to its in-place excitation of the reference,
            shape (ndet,).
        str_up, str_dn: String index of each determinant, shape (ndet,).
    """
    up: WickSpin
    dn: WickSpin
    coeffs: jnp.ndarray
    str_up: jnp.ndarray
    str_dn: jnp.ndarray


def _permutation_sign(perm):
    """Parity (+1 or -1) of a permutation given as an index list."""
    perm = list(perm)
    sign = 1
    for i in range(len(perm)):
        while perm[i] != i:
            j = perm[i]
            perm[i], perm[j] = perm[j], perm[i]
            sign = -sign
    return sign


def _build_spin(occ, chol, h1e, chunk_g):
    """String tables of one spin; returns (WickSpin, str_idx, signs)."""
    occ = np.asarray(occ)
    ndet, nocc = occ.shape
    if nocc == 0:
        raise NotImplementedError(
            "Wick's theorem trials need electrons of both spins")
    ref = [int(o) for o in occ[0]]
    ref_pos = {o: j for j, o in enumerate(ref)}

    excitations = {}      # sorted string -> (holes, particles)
    det_keys, signs = [], np.empty(ndet)
    for i, orbs in enumerate(occ):
        orbs = [int(o) for o in orbs]
        if len(set(orbs)) != nocc:
            raise ValueError(f"determinant {i} repeats an orbital")
        holes = sorted(j for j, o in enumerate(ref) if o not in orbs)
        parts = sorted(o for o in orbs if o not in ref_pos)
        # In-place excitation: hole positions take the particles
        inplace = list(ref)
        for j, p in zip(holes, parts):
            inplace[j] = p
        where = {o: j for j, o in enumerate(inplace)}
        signs[i] = _permutation_sign([where[o] for o in orbs])
        key = tuple(sorted(orbs))
        excitations.setdefault(key, (holes, parts))
        det_keys.append(key)

    particles = sorted({p for _, ps in excitations.values() for p in ps})
    rows = ref + particles
    row_pos = {o: r for r, o in enumerate(rows)}

    # Strings ordered by excitation level; the reference is string 0
    order = sorted(excitations, key=lambda s: len(excitations[s][0]))
    assert len(excitations[order[0]][0]) == 0
    index = {s: i for i, s in enumerate(order)}
    holes_by_k, parts_by_k = {}, {}
    for s in order[1:]:
        hs, ps = excitations[s]
        holes_by_k.setdefault(len(hs), []).append(hs)
        parts_by_k.setdefault(len(hs), []).append(
            [row_pos[p] for p in ps])
    levels = sorted(holes_by_k)

    nb = chol.shape[1]
    chol_rows = np.empty((chol.shape[0], len(rows), nb))
    for g0, g1, chunk in _iter_chol_g_chunks(chol, chunk_g):
        chol_rows[g0:g1] = np.asarray(chunk)[:, rows, :]

    spin = WickSpin(
        ref_occ=jnp.asarray(ref, dtype=jnp.int32),
        rows=jnp.asarray(rows, dtype=jnp.int32),
        chol_rows=jnp.asarray(chol_rows),
        h1e_rows=jnp.asarray(np.asarray(h1e)[rows, :]),
        holes=tuple(jnp.asarray(holes_by_k[k], dtype=jnp.int32)
                    for k in levels),
        parts=tuple(jnp.asarray(parts_by_k[k], dtype=jnp.int32)
                    for k in levels),
    )
    str_idx = np.array([index[key] for key in det_keys], dtype=np.int32)
    return spin, str_idx, signs


def build_wick_trial(occ_up, occ_dn, ci_coeffs, chol, h1e,
                      chunk_g=None):
    """Build a :class:`WickTrial` from a multi-det trial.

    The first determinant is the reference, so it must not be
    orthogonal to the walkers (all walkers start as copies of it in
    :class:`OmegaQMC.afqmc_gto._AFQMCDriverGTO`).

    Args:
        occ_up: Occupied alpha orbitals of each determinant, in the
            column order that defines its sign, shape (ndet, nup).
        occ_dn: Same for beta, shape (ndet, ndown).
        ci_coeffs: CI coefficients, shape (ndet,).
        chol: Cholesky vectors (naux, nbasis, nbasis); an array or a
            :class:`OmegaQMC.integrals.cholesky.DiskChol`.
        h1e: One-body Hamiltonian, shape (nbasis, nbasis).
        chunk_g: g-axis slab size for reading ``chol``.

    Returns:
        WickTrial.
    """
    up, str_up, sign_up = _build_spin(occ_up, chol, h1e, chunk_g)
    dn, str_dn, sign_dn = _build_spin(occ_dn, chol, h1e, chunk_g)
    coeffs = np.conj(np.asarray(ci_coeffs)) * sign_up * sign_dn
    return WickTrial(
        up=up, dn=dn,
        coeffs=jnp.asarray(coeffs, dtype=jnp.complex128),
        str_up=jnp.asarray(str_up), str_dn=jnp.asarray(str_dn),
    )


# ===================================================================
# Cofactors of small matrices
# ===================================================================

def _minor_dets(A, rows, cols):
    """det of A[..., rows[i], :][..., :, cols[i]] for each i."""
    sub = A[..., rows[:, :, None], cols[:, None, :]]
    return jnp.linalg.det(sub)


def cofactors(A):
    """Determinant and its first and second derivatives.

    Args:
        A: shape (..., k, k), k >= 1.

    Returns:
        d: det(A), shape (...).
        C1: C1[a, b] = d det(A) / d A[a, b] (the transposed
            adjugate), shape (..., k, k).
        C2: C2[a, b, c, d] = d^2 det(A) / d A[a, b] d A[c, d],
            shape (..., k, k, k, k).
    """
    k = A.shape[-1]
    lead = A.shape[:-2]
    d = jnp.linalg.det(A)

    rest = [[i for i in range(k) if i != a] for a in range(k)]
    pairs = list(itertools.product(range(k), range(k)))
    r1 = np.array([rest[a] for a, _ in pairs], dtype=np.int32)
    c1 = np.array([rest[b] for _, b in pairs], dtype=np.int32)
    s1 = np.array([(-1.0) ** (a + b) for a, b in pairs])
    C1 = (s1 * _minor_dets(A, r1, c1)).reshape(*lead, k, k)

    C2 = jnp.zeros((*lead, k, k, k, k), dtype=A.dtype)
    if k >= 2:
        two = list(itertools.combinations(range(k), 2))
        quads = list(itertools.product(two, two))
        r2 = np.array([[i for i in range(k) if i not in rc]
                       for rc, _ in quads], dtype=np.int32)
        c2 = np.array([[i for i in range(k) if i not in cc]
                       for _, cc in quads], dtype=np.int32)
        r2 = r2.reshape(len(quads), k - 2)
        c2 = c2.reshape(len(quads), k - 2)
        s2 = np.array([(-1.0) ** (a + c + b + e)
                       for (a, c), (b, e) in quads])
        m = s2 * _minor_dets(A, r2, c2)
        a_, c_ = np.array([q[0] for q in quads]).T
        b_, e_ = np.array([q[1] for q in quads]).T
        # Rows (a < c), columns (b < e): +m for the pairing (a,b)(c,e)
        # and -m for the crossed pairing (a,e)(c,b), in both orders.
        C2 = C2.at[..., a_, b_, c_, e_].set(m)
        C2 = C2.at[..., c_, e_, a_, b_].set(m)
        C2 = C2.at[..., a_, e_, c_, b_].set(-m)
        C2 = C2.at[..., c_, b_, a_, e_].set(-m)
    return d, C1, C2


# ===================================================================
# Walker-side kernels
# ===================================================================

def _reference(phi, spin):
    """<D_0|phi> and Theta restricted to ``spin.rows``."""
    S0 = phi[:, spin.ref_occ, :]
    sign, logdet = jnp.linalg.slogdet(S0)
    ovlp0 = sign * jnp.exp(logdet)
    theta = jnp.einsum('wpi,wij->wpj', phi, jnp.linalg.inv(S0))
    return ovlp0, theta


def _string_dets(theta_r, spin):
    """det(A) of every string, shape (nstr, nwalkers)."""
    nw = theta_r.shape[0]
    dets = [jnp.ones((1, nw), dtype=theta_r.dtype)]
    for holes, parts in zip(spin.holes, spin.parts):
        A = theta_r[:, parts[:, :, None], holes[:, None, :]]
        dets.append(jnp.linalg.det(A).T)
    return jnp.concatenate(dets, axis=0)


def _spin_terms(phi, spin, energy, batch_size):
    """String quantities of one spin.

    For every string I (rows of the outputs) and walker, with Y the
    (nocc, nocc) matrix D_I^T L_g Theta_I of each Cholesky vector:

        d    = det(A)                       = <D_I|phi> / <D_0|phi>
        tau  = d Tr(Y)                       (one per Cholesky vector)
        chi  = d sum_g [Tr(Y)^2 - Tr(Y Y)]   (energy only)
        tauh = d Tr(D_I^T h Theta_I)         (energy only)

    Returns:
        ovlp0 (nw,), d (nstr, nw), tau (nstr, nw, naux), and for
        ``energy`` also chi (nstr, nw) and tauh (nstr, nw).
    """
    ovlp0, theta = _reference(phi, spin)
    nocc = theta.shape[2]
    theta_r = theta[:, spin.rows, :]                     # (w, r, n)

    F = jnp.einsum('grq,wqj->wgrj', spin.chol_rows, theta)
    Y0 = F[:, :, :nocc, :]                               # (w, g, n, n)
    Q = jnp.einsum('wrj,wgjl->wgrl', theta_r, Y0)
    trY0 = jnp.trace(Y0, axis1=2, axis2=3)               # (w, g)
    if energy:
        Y0sq = Y0 @ Y0
        Q2 = jnp.einsum('wrj,wgjl->wgrl', theta_r, Y0sq)
        FY0 = F @ Y0
        trY0sq = jnp.trace(Y0sq, axis1=2, axis2=3)
        Fh = jnp.einsum('rq,wqj->wrj', spin.h1e_rows, theta)
        Y0h = Fh[:, :nocc, :]
        Qh = jnp.einsum('wrj,wjl->wrl', theta_r, Y0h)
        trY0h = jnp.trace(Y0h, axis1=1, axis2=2)

    nw = phi.shape[0]
    d_all = [jnp.ones((1, nw), dtype=theta.dtype)]
    tau_all = [trY0[None]]
    if energy:
        chi_all = [jnp.sum(trY0 ** 2 - trY0sq, axis=1)[None]]
        tauh_all = [trY0h[None]]

    def one_string(hp):
        h, p = hp
        k = h.shape[0]
        hh = (h[:, None], h[None, :])
        ph = (p[:, None], h[None, :])
        A = theta_r[:, ph[0], ph[1]]                     # (w, k, k)
        d, C1, C2 = cofactors(A)
        AmI = A - jnp.eye(k)
        # Delta[:, h] for Delta = F[p, :] - Y0[h, :]
        Dh = F[:, :, ph[0], ph[1]] - Y0[:, :, hh[0], hh[1]]
        trY = trY0 + jnp.trace(Dh, axis1=2, axis2=3)
        # Z = B Y_I[:, h] with B = Theta[p, :] - E_h^T
        Z = (Q[:, :, ph[0], ph[1]] - Y0[:, :, hh[0], hh[1]]
             + jnp.einsum('wab,wgbc->wgac', AmI, Dh))
        c1Z = jnp.einsum('wab,wgab->wg', C1, Z)
        tau = d[:, None] * trY - c1Z
        if not energy:
            return d, tau
        DY0h = FY0[:, :, ph[0], ph[1]] - Y0sq[:, :, hh[0], hh[1]]
        trYY = (trY0sq + 2.0 * jnp.trace(DY0h, axis1=2, axis2=3)
                + jnp.einsum('wgab,wgba->wg', Dh, Dh))
        W = (Q2[:, :, ph[0], ph[1]] - Y0sq[:, :, hh[0], hh[1]]
             + jnp.einsum('wab,wgbc->wgac', AmI, DY0h) + Z @ Dh)
        chi = (d[:, None] * (trY ** 2 - trYY)
               - 2.0 * trY * c1Z
               + 2.0 * jnp.einsum('wab,wgab->wg', C1, W)
               + jnp.einsum('wabcd,wgab,wgcd->wg', C2, Z, Z))
        Dhh = Fh[:, ph[0], ph[1]] - Y0h[:, hh[0], hh[1]]
        trYh = trY0h + jnp.trace(Dhh, axis1=1, axis2=2)
        Zh = (Qh[:, ph[0], ph[1]] - Y0h[:, hh[0], hh[1]]
              + AmI @ Dhh)
        tauh = d * trYh - jnp.einsum('wab,wab->w', C1, Zh)
        return d, tau, jnp.sum(chi, axis=1), tauh

    for holes, parts in zip(spin.holes, spin.parts):
        out = jax.lax.map(one_string, (holes, parts),
                          batch_size=batch_size)
        d_all.append(out[0])
        tau_all.append(out[1])
        if energy:
            chi_all.append(out[2])
            tauh_all.append(out[3])

    d = jnp.concatenate(d_all, axis=0)
    tau = jnp.concatenate(tau_all, axis=0)
    if not energy:
        return ovlp0, d, tau
    return (ovlp0, d, tau, jnp.concatenate(chi_all, axis=0),
            jnp.concatenate(tauh_all, axis=0))


def _det_weights(trial, d_up, d_dn):
    """Ratio N = <Psi_T|phi> / (<D_0|phi>_up <D_0|phi>_dn) and the
    string weights beta_s(I) = sum_{J in I} c'_J d_other(J)."""
    du = d_up[trial.str_up]                               # (ndet, nw)
    dd = d_dn[trial.str_dn]
    c = trial.coeffs[:, None]
    ratio = jnp.sum(c * du * dd, axis=0)
    beta_up = jax.ops.segment_sum(
        c * dd, trial.str_up, num_segments=d_up.shape[0])
    beta_dn = jax.ops.segment_sum(
        c * du, trial.str_dn, num_segments=d_dn.shape[0])
    return ratio, beta_up, beta_dn


@partial(jax.jit, static_argnames=['batch_size'])
def wick_overlap(phia, phib, trial, batch_size=None):
    """Multi-det overlap <Psi_T|phi>, shape (nwalkers,)."""
    o_up, th_up = _reference(phia, trial.up)
    o_dn, th_dn = _reference(phib, trial.dn)
    d_up = _string_dets(th_up[:, trial.up.rows, :], trial.up)
    d_dn = _string_dets(th_dn[:, trial.dn.rows, :], trial.dn)
    ratio = jnp.sum(trial.coeffs[:, None] * d_up[trial.str_up]
                    * d_dn[trial.str_dn], axis=0)
    return o_up * o_dn * ratio


@partial(jax.jit, static_argnames=['batch_size'])
def wick_force_bias(phia, phib, trial, batch_size=None):
    """Mixed Tr(L_g G) summed over spins, and the overlap.

    Returns:
        vbias: sum_s Tr(L_g G_s) with the multi-det mixed G,
            shape (naux, nwalkers).
        overlap: <Psi_T|phi>, shape (nwalkers,).
    """
    o_up, d_up, tau_up = _spin_terms(phia, trial.up, False, batch_size)
    o_dn, d_dn, tau_dn = _spin_terms(phib, trial.dn, False, batch_size)
    ratio, beta_up, beta_dn = _det_weights(trial, d_up, d_dn)
    vbias = (jnp.einsum('swg,sw->gw', tau_up, beta_up)
             + jnp.einsum('swg,sw->gw', tau_dn, beta_dn)) / ratio
    return vbias, o_up * o_dn * ratio


@partial(jax.jit, static_argnames=['batch_size'])
def wick_local_energy(phia, phib, trial, enuc, batch_size=None):
    """Mixed-estimator local energy with the generalized Wick's theorem.

    Uses the bare one-body Hamiltonian and Cholesky rows stored in
    ``trial`` (see :func:`build_wick_trial`).

    Returns:
        e_tot, e_1b, e_2b: shape (nwalkers,) each; e_1b includes enuc.
    """
    _, d_up, tau_up, chi_up, th_up = _spin_terms(
        phia, trial.up, True, batch_size)
    _, d_dn, tau_dn, chi_dn, th_dn = _spin_terms(
        phib, trial.dn, True, batch_size)
    ratio, beta_up, beta_dn = _det_weights(trial, d_up, d_dn)

    e_1b = (jnp.sum(th_up * beta_up, axis=0)
            + jnp.sum(th_dn * beta_dn, axis=0)) / ratio + enuc

    # Opposite-spin Coulomb: sum_J c'_J sum_g tau_up(J) tau_dn(J)
    def cross(xs):
        c, iu, idn = xs
        return c * jnp.sum(tau_up[iu] * tau_dn[idn], axis=-1)

    e_x = jnp.sum(jax.lax.map(
        cross, (trial.coeffs, trial.str_up, trial.str_dn),
        batch_size=batch_size), axis=0)
    e_2b = 0.5 * (jnp.sum(chi_up * beta_up, axis=0)
                  + jnp.sum(chi_dn * beta_dn, axis=0)
                  + 2.0 * e_x) / ratio
    return e_1b + e_2b, e_1b, e_2b
