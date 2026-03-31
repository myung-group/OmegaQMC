"""
Local energy estimators for AFQMC.

All functions are pure (no side effects) and suitable for use
inside ``@jax.jit`` boundaries defined in the driver modules.
"""

import jax
import jax.numpy as jnp
from functools import partial


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
    h1e, chol, Ga, Gb, Ghalfa, Ghalfb,
    rchol_a, rchol_b, enuc,
):
    """Mixed-estimator local energy for all walkers.

    E_loc = E_1body + E_coulomb - E_exchange + E_nuc

    Uses half-rotated Cholesky vectors for efficiency.

    Args:
        h1e: One-body Hamiltonian, shape (nbasis, nbasis).
        chol: Cholesky vectors, shape (naux, nbasis, nbasis).
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
