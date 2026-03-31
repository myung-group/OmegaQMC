"""
One-particle Green's function routines for AFQMC.

Single-determinant and multi-determinant variants.
"""

import jax
import jax.numpy as jnp
from functools import partial


def _greens_function_spin(phi, trial):
    """Green's function for one spin channel.

    Args:
        phi: Walker orbitals,
            shape (nwalkers, nbasis, nocc).
        trial: Trial orbitals, shape (nbasis, nocc).

    Returns:
        G: Full Green's function,
            shape (nwalkers, nbasis, nbasis).
        Ghalf: Half-rotated GF,
            shape (nwalkers, nocc, nbasis).
        ovlp: det(trial^dag phi), shape (nwalkers,).
        log_ovlp: log|det(trial^dag phi)|,
            shape (nwalkers,).
    """
    # Overlap matrix: overlap = phi^T @ trial^* (ipie convention)
    ovlp = jnp.einsum('wpi,pj->wij', phi, trial.conj())

    # Inverse overlap
    ovlp_inv = jnp.linalg.inv(ovlp)

    # Half-rotated Green's function: Ghalf = ovlp^{-1} @ phi^T
    Ghalf = jnp.einsum('wij,wqj->wiq', ovlp_inv, phi)

    # Full Green's function: G = trial.conj() @ Ghalf
    G = jnp.einsum('pi,wiq->wpq', trial.conj(), Ghalf)

    # Overlap determinant
    sign, log_det = jnp.linalg.slogdet(ovlp)
    ovlp = sign * jnp.exp(log_det)
    log_ovlp = log_det

    return G, Ghalf, ovlp, log_ovlp


@partial(jax.jit, static_argnames=[])
def greens_function(phia, phib, trial_up, trial_dn):
    """One-particle Green's function for all walkers.

    G = phi (trial^dag phi)^{-1} trial^dag

    Also returns the half-rotated Green's function:
    Ghalf = (trial^dag phi)^{-1} phi^T

    Args:
        phia: Walker alpha orbitals,
            shape (nwalkers, nbasis, nup).
        phib: Walker beta orbitals,
            shape (nwalkers, nbasis, ndown).
        trial_up: Trial alpha orbitals,
            shape (nbasis, nup).
        trial_dn: Trial beta orbitals,
            shape (nbasis, ndown).

    Returns:
        Ga: Full GF alpha,
            shape (nwalkers, nbasis, nbasis).
        Gb: Full GF beta,
            shape (nwalkers, nbasis, nbasis).
        Ghalfa: Half-rotated GF alpha,
            shape (nwalkers, nup, nbasis).
        Ghalfb: Half-rotated GF beta,
            shape (nwalkers, ndown, nbasis).
        overlap: <psi_T|phi> per walker,
            shape (nwalkers,).
    """
    Ga, Ghalfa, ovlp_a, _ = _greens_function_spin(
        phia, trial_up,
    )
    Gb, Ghalfb, ovlp_b, _ = _greens_function_spin(
        phib, trial_dn,
    )

    # Total overlap is product of alpha and beta overlaps
    overlap = ovlp_a * ovlp_b

    return Ga, Gb, Ghalfa, Ghalfb, overlap


def _gf_spin_single_det(phi, trial_I):
    """Green's function for one det, all walkers, one spin.

    Args:
        phi: Walker orbitals,
            shape (nwalkers, nbasis, nocc).
        trial_I: Trial orbitals for one determinant,
            shape (nbasis, nocc).

    Returns:
        Ghalf: Half-rotated GF,
            shape (nwalkers, nocc, nbasis).
        ovlp: det(trial_I^dag phi), shape (nwalkers,).
    """
    ovlp_mat = jnp.einsum(
        'wpi,pj->wij', phi, trial_I.conj(),
    )
    ovlp_inv = jnp.linalg.inv(ovlp_mat)
    Ghalf = jnp.einsum('wij,wqj->wiq', ovlp_inv, phi)
    sign, log_det = jnp.linalg.slogdet(ovlp_mat)
    ovlp = sign * jnp.exp(log_det)
    return Ghalf, ovlp


@partial(jax.jit, static_argnames=[])
def greens_function_multidet(
    phia, phib, trials_up, trials_dn, ci_coeffs,
):
    """Multi-determinant Green's function for all walkers.

    Args:
        phia: Walker alpha orbitals,
            shape (nwalkers, nbasis, nup).
        phib: Walker beta orbitals,
            shape (nwalkers, nbasis, ndown).
        trials_up: Trial alpha orbitals,
            shape (ndet, nbasis, nup).
        trials_dn: Trial beta orbitals,
            shape (ndet, nbasis, ndown).
        ci_coeffs: CI coefficients, shape (ndet,).

    Returns:
        Ga, Gb: Full GF,
            shape (nwalkers, nbasis, nbasis).
        Ghalfa_all, Ghalfb_all: Per-det half-rotated GF
            (NaN-sanitized),
            shape (ndet, nwalkers, nocc, nbasis).
        overlap: Multi-det overlap,
            shape (nwalkers,).
        ovlp_a_all, ovlp_b_all: Per-det spin overlaps,
            shape (ndet, nwalkers).
    """
    # vmap over determinant axis
    Ghalfa_all, ovlp_a_all = jax.vmap(
        _gf_spin_single_det, in_axes=(None, 0),
    )(phia, trials_up)
    # Ghalfa_all: (ndet, nwalkers, nup, nbasis)
    # ovlp_a_all: (ndet, nwalkers)

    Ghalfb_all, ovlp_b_all = jax.vmap(
        _gf_spin_single_det, in_axes=(None, 0),
    )(phib, trials_dn)

    # Sanitize NaN in Ghalf (from singular overlaps where
    # det has zero overlap with walker — the weight w_I is
    # also zero so contribution vanishes, but NaN * 0 = NaN)
    Ghalfa_all = jnp.where(
        jnp.isnan(Ghalfa_all), 0.0, Ghalfa_all,
    )
    Ghalfb_all = jnp.where(
        jnp.isnan(Ghalfb_all), 0.0, Ghalfb_all,
    )

    # Per-det weight: w_I = c_I* x O_I^a x O_I^b
    w_I = (
        ci_coeffs.conj()[:, None]
        * ovlp_a_all * ovlp_b_all
    )
    # w_I: (ndet, nwalkers)

    # Multi-det overlap
    overlap = jnp.sum(w_I, axis=0)  # (nwalkers,)

    # w_I: (ndet, nwalkers) -> (ndet, nwalkers, 1, 1)
    w_expanded = w_I[:, :, None, None]

    # Full G: G_a = sum_I w_I * trial_I^* @ Ghalf_I / O
    Ga = jnp.sum(
        w_expanded * jnp.einsum(
            'dpi,dwiq->dwpq',
            trials_up.conj(), Ghalfa_all,
        ),
        axis=0,
    ) / overlap[:, None, None]
    Gb = jnp.sum(
        w_expanded * jnp.einsum(
            'dpi,dwiq->dwpq',
            trials_dn.conj(), Ghalfb_all,
        ),
        axis=0,
    ) / overlap[:, None, None]

    return (
        Ga, Gb, Ghalfa_all, Ghalfb_all,
        overlap, ovlp_a_all, ovlp_b_all,
    )
