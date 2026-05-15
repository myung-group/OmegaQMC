"""
Phaseless Auxiliary-Field Quantum Monte Carlo (AFQMC) driver.

The public entry point is :func:`get_afqmc_func`, which returns a
reusable ``_AFQMCDriverGTO`` instance (setup once, run many times).

Canonical implementations live in dedicated submodules:

* :mod:`OmegaQMC.integrals.cholesky` — Cholesky decomposition,
  half-rotation, integral preparation, CASSCF trial extraction.
* :mod:`OmegaQMC.observables.energy` — local energy estimators.
* :mod:`OmegaQMC.observables.greens` — Green's function routines.

All names from those submodules are re-exported here for backward
compatibility.

Uses JAX + PySCF.
"""

import sys
import time

import numpy as np
import jax
import jax.numpy as jnp
from jax.scipy.linalg import expm
from functools import partial

from jax.sharding import Mesh, NamedSharding, PartitionSpec

from OmegaQMC.utils import do_binning_analysis

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
WEIGHT_CLIP_FRACTION = 0.10
FBBOUND_DEFAULT = 1.0


# --- Integrals re-exported from OmegaQMC.integrals ---
from OmegaQMC.integrals.cholesky import (        # noqa: E402
    extract_casscf_trial,
)


def _make_afqmc_sharding(num_walkers):
    """Return (phi_sharding, scalar_sharding) or (None, None).

    phi_sharding:       PartitionSpec('w', None, None)
                            for (nwalkers, nbasis, nocc)
    scalar_sharding:    PartitionSpec('w',)
                            for (nwalkers,)
    """
    devices = jax.devices()
    n = len(devices)
    if n == 1:
        return None, None
    assert num_walkers % n == 0, (
        f"num_walkers ({num_walkers}) must be divisible by device count ({n})")
    mesh = Mesh(np.array(devices), ('w',))
    phi_sharding = NamedSharding(mesh, PartitionSpec('w', None, None))
    scalar_sharding = NamedSharding(mesh, PartitionSpec('w',))
    return phi_sharding, scalar_sharding


from OmegaQMC.integrals.cholesky import (        # noqa: E402
    chunked_cholesky,
    prepare_afqmc_integrals,
    half_rotate_cholesky,
    half_rotate_cholesky_multidet,
    DiskChol,
    _iter_chol_g_chunks,
)


# ===================================================================
# Walker management
# ===================================================================

class Walkers:
    """Collection of AFQMC walkers.

    Each walker is a pair of Slater determinant coefficient matrices
    (phia, phib) for alpha and beta spins, plus an importance sampling weight.

    Attributes:
        phia: Alpha orbital coefficients, shape (nwalkers, nbasis, nup).
        phib: Beta orbital coefficients, shape (nwalkers, nbasis, ndown).
        weights: Importance sampling weights, shape (nwalkers,).
        overlap: Trial-walker overlap <psi_T|phi>, shape (nwalkers,).
        e_hybrid: Hybrid energy from previous step, shape (nwalkers,).
        log_detR: Accumulated log determinant from QR, shape (nwalkers,).
        log_shift: Overlap stabilization shift, shape (nwalkers,).
    """

    def __init__(self, trial_up, trial_dn, nwalkers, nbasis, nup, ndown):
        """Initialize walkers as copies of the trial wavefunction.

        Args:
            trial_up: Trial alpha orbitals, shape (nbasis, nup).
            trial_dn: Trial beta orbitals, shape (nbasis, ndown).
            nwalkers: Number of walkers.
            nbasis: Number of basis functions.
            nup: Number of alpha electrons.
            ndown: Number of beta electrons.
        """
        self.nwalkers = nwalkers
        self.nbasis = nbasis
        self.nup = nup
        self.ndown = ndown

        # All walkers start as copies of the trial
        self.phia = jnp.tile(trial_up[None, :, :], (nwalkers, 1, 1))
        self.phib = jnp.tile(trial_dn[None, :, :], (nwalkers, 1, 1))

        # Importance sampling weights
        self.weights = jnp.ones(nwalkers)

        # Overlaps <psi_T|phi_w>
        self.overlap = jnp.ones(nwalkers, dtype=jnp.complex128)

        # Hybrid energy from previous step (for averaging)
        self.e_hybrid = jnp.zeros(nwalkers, dtype=jnp.complex128)

        # Accumulated log-determinant from QR stabilization
        self.log_detR = jnp.zeros(nwalkers)
        self.log_shift = jnp.zeros(nwalkers)


def _qr_single(phi_single):
    """QR decomposition for a single walker.

    Args:
        phi_single: shape (nbasis, nocc).

    Returns:
        Q with signs absorbed, log|det(R)|, sign of det(R).
    """
    Q, R = jnp.linalg.qr(phi_single)
    # Extract diagonal signs to ensure positive diagonal in R
    diag_R = jnp.diag(R)
    signs = jnp.sign(diag_R)
    # Absorb signs into Q
    Q_fixed = Q * signs[None, :]
    log_det = jnp.sum(jnp.log(jnp.abs(diag_R)))
    sign = jnp.prod(signs)
    return Q_fixed, log_det, sign


def _qr_batch(phi):
    """QR decomposition for a batch of walker matrices.

    Args:
        phi: shape (nwalkers, nbasis, nocc).

    Returns:
        phi_ortho: Orthogonalized matrices, shape (nwalkers, nbasis, nocc).
        log_det: Log of |det(R)| for each walker, shape (nwalkers,).
        signs: Sign of det(R), shape (nwalkers,).
    """
    return jax.vmap(_qr_single)(phi)


@jax.jit
def orthogonalize_walkers(phia, phib):
    """Reorthogonalize walker determinants via QR decomposition.

    Prevents numerical instability from accumulated propagation steps.
    Should be called every ~5 steps.

    Args:
        phia: shape (nwalkers, nbasis, nup).
        phib: shape (nwalkers, nbasis, ndown).

    Returns:
        phia_new: Orthogonalized alpha orbitals.
        phib_new: Orthogonalized beta orbitals.
        log_detR: Log of |det(R_up * R_dn)| for each walker, shape (nwalkers,).
        sign_detR: Sign of det(R), shape (nwalkers,).
    """
    phia_new, log_detR_a, sign_a = _qr_batch(phia)
    phib_new, log_detR_b, sign_b = _qr_batch(phib)
    log_detR = log_detR_a + log_detR_b
    sign_detR = sign_a * sign_b
    return phia_new, phib_new, log_detR, sign_detR


@jax.jit
def population_control_comb(weights, phia, phib, rng_key):
    """Comb population control with weight rescaling.

    Rescales weights to sum to nwalkers (ipie convention), then
    places evenly-spaced 'teeth' on the cumulative weight distribution
    to resample walkers proportional to their weights.

    Pure JAX so the gather stays on device — the previous numpy
    implementation forced a full GPU↔CPU round-trip of the
    walker matrices on every population-control step.

    Args:
        weights: shape (nwalkers,).
        phia: shape (nwalkers, nbasis, nup).
        phib: shape (nwalkers, nbasis, ndown).
        rng_key: JAX random key.

    Returns:
        weights_new, phia_new, phib_new.
    """
    nwalkers = weights.shape[0]
    total_weight = jnp.sum(weights)

    # Rescale weights so total = nwalkers (ipie convention)
    scale = total_weight / nwalkers
    weights_scaled = weights / scale

    # Cumulative distribution (now sums to nwalkers)
    cumsum = jnp.cumsum(weights_scaled)

    # Random offset for comb
    r = jax.random.uniform(rng_key)

    # Comb teeth positions
    teeth = jnp.arange(nwalkers, dtype=cumsum.dtype) + r

    # Select walkers
    new_indices = jnp.searchsorted(cumsum, teeth)
    new_indices = jnp.clip(new_indices, 0, nwalkers - 1)

    phia_new = phia[new_indices]
    phib_new = phib[new_indices]

    # All walkers get equal weight = 1.0 after resampling
    weights_new = jnp.ones(nwalkers, dtype=weights.dtype)

    return weights_new, phia_new, phib_new


def population_control_pair_branch(weights, phia, phib, rng_key):
    """Pair branching population control.

    Pairs the lightest and heaviest walkers. If the heavy walker's
    weight exceeds a threshold, clone it to replace the light one.

    Args:
        weights: shape (nwalkers,).
        phia: shape (nwalkers, nbasis, nup).
        phib: shape (nwalkers, nbasis, ndown).
        rng_key: JAX random key.

    Returns:
        weights_new, phia_new, phib_new.
    """
    nwalkers = weights.shape[0]
    total_weight = jnp.sum(weights)
    avg_weight = total_weight / nwalkers

    weights_np = np.array(weights)
    sorted_idx_np = np.argsort(weights_np)

    phia_np = np.array(phia)
    phib_np = np.array(phib)

    min_weight = 0.1 * float(avg_weight)
    max_weight = 4.0 * float(avg_weight)

    for k in range(nwalkers // 2):
        i_light = sorted_idx_np[k]
        i_heavy = sorted_idx_np[nwalkers - 1 - k]

        w_light = weights_np[i_light]
        w_heavy = weights_np[i_heavy]

        if w_heavy > max_weight and w_light < min_weight:
            # Clone heavy walker, split weight
            new_w = 0.5 * (w_light + w_heavy)
            weights_np[i_light] = new_w
            weights_np[i_heavy] = new_w
            phia_np[i_light] = phia_np[i_heavy]
            phib_np[i_light] = phib_np[i_heavy]

    return jnp.array(weights_np), jnp.array(phia_np), jnp.array(phib_np)


# ===================================================================
# Estimators
# ===================================================================

# --- Observables re-exported from OmegaQMC.observables ---
# These lived here originally; now canonical in observables/.
from OmegaQMC.observables.greens import (       # noqa: E402
    _greens_function_spin,
    greens_function,
    greens_function_force_bias,
    greens_function_overlap,
    _gf_spin_single_det,
    greens_function_multidet,
)
from OmegaQMC.observables.energy import (        # noqa: E402
    local_energy_1body,
    local_energy_2body,
    local_energy,
    local_energy_multidet,
)


# ===================================================================
# Propagation
# ===================================================================

def _apply_exp_vhs(VHS, phi, nmax=6):
    """Apply exp(VHS) to walker orbitals via Taylor expansion.

    exp(VHS) @ phi ≈ phi + VHS@phi + VHS@VHS@phi/2! + ...

    Args:
        VHS: shape (nwalkers, nbasis, nbasis).
        phi: shape (nwalkers, nbasis, nocc).
        nmax: Taylor expansion order.

    Returns:
        phi_new: shape (nwalkers, nbasis, nocc).
    """
    result = phi.copy()
    temp = phi.copy()
    for n in range(1, nmax + 1):
        temp = jnp.einsum('wpq,wqn->wpn', VHS, temp) / n
        result = result + temp
    return result


def _build_vhs(xshifted_w, chol, dt, chol_chunk_g):
    """VHS = 1j*sqrt(dt) * sum_g xshifted_w[w,g] * chol[g], streaming g."""
    nb = chol.shape[1]
    nw = xshifted_w.shape[0]
    sqdt = jnp.sqrt(dt)
    VHS = jnp.zeros((nw, nb, nb), dtype=jnp.complex128)
    for g0, g1, chunk in _iter_chol_g_chunks(chol, chol_chunk_g):
        chunk_j = jnp.asarray(chunk)
        VHS = VHS + 1j * sqdt * jnp.einsum(
            'wg,gpq->wpq', xshifted_w[:, g0:g1], chunk_j)
    return VHS


def _apply_exp_vhs_from_chol(xshifted, chol, phia, phib, dt, nmax=6,
                             chunk_size=None, chol_chunk_g=None):
    """Build and apply exp(VHS) over walker × g chunks (memory‑friendly).

    Equivalent to constructing
        VHS = 1j * sqrt(dt) * einsum('wg,gpq->wpq', xshifted, chol)
    and then calling _apply_exp_vhs(VHS, phi*, nmax) for each spin, but
    holds at most (chunk_size, nbasis, nbasis) of VHS at any time, and
    reads ``chol`` in slabs of ``chol_chunk_g`` along the auxiliary axis
    — supports both in‑memory and :class:`DiskChol` backings.

    Args:
        xshifted: shape (nwalkers, naux).
        chol:     shape (naux, nbasis, nbasis); array or DiskChol.
        phia:     shape (nwalkers, nbasis, nup).
        phib:     shape (nwalkers, nbasis, ndown).
        dt:       time step.
        nmax:     Taylor expansion order.
        chunk_size: walkers per chunk; None or >= nwalkers means no
            walker chunking.
        chol_chunk_g: g‑axis slab size; None uses the whole tensor
            (or the DiskChol's own chunk_g).

    Returns:
        phia_new, phib_new with the same shapes as phia, phib.
    """
    nwalkers = phia.shape[0]

    if chunk_size is None or chunk_size >= nwalkers:
        VHS = _build_vhs(xshifted, chol, dt, chol_chunk_g)
        return _apply_exp_vhs(VHS, phia, nmax), _apply_exp_vhs(VHS, phib, nmax)

    a_chunks, b_chunks = [], []
    for start in range(0, nwalkers, chunk_size):
        end = min(start + chunk_size, nwalkers)
        VHS_c = _build_vhs(xshifted[start:end], chol, dt, chol_chunk_g)
        a_chunks.append(_apply_exp_vhs(VHS_c, phia[start:end], nmax))
        b_chunks.append(_apply_exp_vhs(VHS_c, phib[start:end], nmax))
    return (jnp.concatenate(a_chunks, axis=0),
            jnp.concatenate(b_chunks, axis=0))


def _update_weights_phaseless(weights, ovlp_old, ovlp_new, cfb, cmf,
                              e_hybrid_old, eshift, dt):
    """Update importance sampling weights with the phaseless approximation.

    Args:
        weights: Current weights, shape (nwalkers,).
        ovlp_old: Previous overlap, shape (nwalkers,).
        ovlp_new: New overlap, shape (nwalkers,).
        cfb: Force bias correction, shape (nwalkers,).
        cmf: Mean-field correction, shape (nwalkers,).
        e_hybrid_old: Previous hybrid energy, shape (nwalkers,).
        eshift: Energy shift (scalar).
        dt: Time step.

    Returns:
        weights_new: Updated weights.
        e_hybrid_new: New hybrid energy.
    """
    # Overlap ratio
    ovlp_ratio = ovlp_new / ovlp_old

    # Hybrid energy estimator
    e_hybrid_new = -(jnp.log(ovlp_ratio) + cfb + cmf) / dt

    # Bound hybrid energy
    ebound = jnp.sqrt(2.0 / dt)
    e_hybrid_real = jnp.clip(e_hybrid_new.real,
                             eshift - ebound, eshift + ebound)
    e_hybrid_new = e_hybrid_real + 1j * e_hybrid_new.imag

    # Importance function: exp(-dt * (avg_hybrid - eshift))
    avg_hybrid = 0.5 * (e_hybrid_new + e_hybrid_old)
    importance = jnp.exp(-dt * (avg_hybrid - eshift))

    # Magnitude
    magn = jnp.abs(importance)

    # Phaseless approximation: cos(dtheta), clipped to [0, inf)
    dtheta = (-dt * e_hybrid_new - cfb).imag
    cosine_fac = jnp.maximum(0.0, jnp.cos(dtheta))

    # Update weights
    weights_new = weights * magn * cosine_fac

    return weights_new, e_hybrid_new


def build_propagator(h1e_mod, chol, trial_up, trial_dn, dt,
                     G_charge=None, chol_chunk_g=None):
    """Build the one-body propagator and mean-field shift.

    Args:
        h1e_mod: Modified one-body Hamiltonian, shape (nbasis, nbasis).
        chol: Cholesky vectors, shape (naux, nbasis, nbasis). May be an
            in‑memory array or a :class:`DiskChol`; in the latter case
            mf_shift and vhs_mf are accumulated by g‑chunks so the full
            chol tensor never lives in device memory.
        trial_up: Trial alpha orbitals, shape (nbasis, nup).
        trial_dn: Trial beta orbitals, shape (nbasis, ndown).
        dt: Imaginary time step.
        G_charge: Pre-computed charge density matrix for multi-det trials.
                  If None, computed from single-det trial.
        chol_chunk_g: g‑axis slab size when chol is a DiskChol. None
            uses the DiskChol's own chunk_g, or no chunking for arrays.

    Returns:
        dict with:
            'expH1': Half-step one-body propagator, shape (nbasis, nbasis).
            'mf_shift': Mean-field shift, shape (naux,).
            'dt': Time step.
    """
    if G_charge is None:
        G_trial_a = trial_up @ trial_up.T.conj()
        G_trial_b = trial_dn @ trial_dn.T.conj()
        G_charge = G_trial_a + G_trial_b

    naux = chol.shape[0]
    nbasis = chol.shape[1]

    # mf_shift[g] = 1j * sum_{p,q} chol[g,p,q] G_charge[q,p]
    mf_shift = jnp.zeros(naux, dtype=jnp.complex128)
    for g0, g1, chunk in _iter_chol_g_chunks(chol, chol_chunk_g):
        chunk_j = jnp.asarray(chunk)
        contrib = 1j * jnp.einsum('gpq,qp->g', chunk_j, G_charge)
        mf_shift = mf_shift.at[g0:g1].set(contrib)

    # vhs_mf[p,q] = 1j * sum_g mf_shift[g] chol[g,p,q]
    vhs_mf = jnp.zeros((nbasis, nbasis), dtype=jnp.complex128)
    for g0, g1, chunk in _iter_chol_g_chunks(chol, chol_chunk_g):
        chunk_j = jnp.asarray(chunk)
        vhs_mf = vhs_mf + 1j * jnp.einsum(
            'g,gpq->pq', mf_shift[g0:g1], chunk_j)

    H1_shifted = h1e_mod - vhs_mf
    expH1 = expm(-0.5 * dt * H1_shifted)

    return {
        'expH1': expH1,
        'mf_shift': mf_shift,
        'dt': dt,
    }


def propagate_walkers(phia, phib, weights, overlap, e_hybrid,
                      propagator, chol, rchol_a, rchol_b,
                      trial_up, trial_dn, eshift, rng_key,
                      fbbound=None, exp_nmax=6, walker_chunk_size=None,
                      chol_chunk_g=None):
    """Propagate all walkers by one time step with phaseless constraint.

    Steps:
    1. Compute Green's function and force bias
    2. Apply exp(-dt/2 H1) (one-body, first half)
    3. Sample auxiliary fields and apply exp(VHS) (two-body)
    4. Apply exp(-dt/2 H1) (one-body, second half)
    5. Update weights with phaseless approximation

    Args:
        phia: shape (nwalkers, nbasis, nup).
        phib: shape (nwalkers, nbasis, ndown).
        weights: shape (nwalkers,).
        overlap: Overlap from previous step, shape (nwalkers,).
        e_hybrid: Hybrid energy from previous step, shape (nwalkers,).
        propagator: dict from build_propagator().
        chol: Cholesky vectors, shape (naux, nbasis, nbasis).
        rchol_a, rchol_b: Half-rotated Cholesky vectors.
        trial_up, trial_dn: Trial wavefunction orbitals.
        eshift: Energy shift for population control (scalar).
        rng_key: JAX random key.
        fbbound: Force bias magnitude bound (default: 1.0).
        exp_nmax: Taylor expansion order for matrix exponential.

    Returns:
        phia_new, phib_new, weights_new, overlap_new, e_hybrid_new, rng_key.
    """
    dt = propagator['dt']
    expH1 = propagator['expH1']
    mf_shift = propagator['mf_shift']
    nwalkers = phia.shape[0]
    naux = chol.shape[0]

    if fbbound is None:
        fbbound = FBBOUND_DEFAULT

    # 1. Compute Green's function from current walkers
    # (specialized: only Ghalf + overlap needed for force
    # bias, full G is dropped)
    Ghalfa, Ghalfb, ovlp = greens_function_force_bias(
        phia, phib, trial_up, trial_dn)

    # 2. Apply first half of one-body propagator
    phia = jnp.einsum('pq,wqn->wpn', expH1, phia)
    phib = jnp.einsum('pq,wqn->wpn', expH1, phib)

    # 3. Compute force bias
    vbias_a = jnp.einsum('giq,wiq->gw', rchol_a, Ghalfa)
    vbias_b = jnp.einsum('giq,wiq->gw', rchol_b, Ghalfb)
    vbias = vbias_a + vbias_b

    # Optimal shift: xbar = -sqrt(dt) * (i*vbias - mf_shift)
    xbar = -jnp.sqrt(dt) * (1j * vbias - mf_shift[:, None])
    xbar = xbar.T  # (nwalkers, naux)

    # Bound force bias
    xbar_abs = jnp.abs(xbar)
    xbar = jnp.where(xbar_abs > fbbound,
                     xbar * fbbound / xbar_abs,
                     xbar)

    # 4. Sample auxiliary fields
    rng_key, subkey = jax.random.split(rng_key)
    xi = jax.random.normal(subkey, shape=(nwalkers, naux))

    # Shifted fields
    xshifted = xi - xbar

    # Constant factors for weight update
    cmf = -jnp.sqrt(dt) * jnp.einsum('wg,g->w', xshifted, mf_shift)
    cfb = (jnp.sum(xi * xbar, axis=1)
           - 0.5 * jnp.sum(xbar * xbar, axis=1))

    # 5. Construct and apply VHS (chunked over walkers and g)
    phia, phib = _apply_exp_vhs_from_chol(
        xshifted, chol, phia, phib, dt, exp_nmax,
        walker_chunk_size, chol_chunk_g)

    # 6. Apply second half of one-body propagator
    phia = jnp.einsum('pq,wqn->wpn', expH1, phia)
    phib = jnp.einsum('pq,wqn->wpn', expH1, phib)

    # 7. Compute new overlap and update weights
    # (specialized: only overlap needed)
    ovlp_new = greens_function_overlap(
        phia, phib, trial_up, trial_dn)

    weights_new, e_hybrid_new = _update_weights_phaseless(
        weights, ovlp, ovlp_new, cfb, cmf, e_hybrid, eshift, dt)

    return phia, phib, weights_new, ovlp_new, e_hybrid_new, rng_key


def propagate_walkers_multidet(phia, phib, weights, overlap, e_hybrid,
                               propagator, chol, rchols_a, rchols_b,
                               trials_up, trials_dn, ci_coeffs,
                               eshift, rng_key,
                               fbbound=None, exp_nmax=6,
                               walker_chunk_size=None,
                               chol_chunk_g=None):
    """Propagate all walkers by one time step using multi-det trial.

    Same structure as single-det propagate_walkers but uses
    greens_function_multidet and per-det force bias.

    Args:
        phia: shape (nwalkers, nbasis, nup).
        phib: shape (nwalkers, nbasis, ndown).
        weights: shape (nwalkers,).
        overlap: Overlap from previous step, shape (nwalkers,).
        e_hybrid: Hybrid energy from previous step, shape (nwalkers,).
        propagator: dict from build_propagator().
        chol: Cholesky vectors, shape (naux, nbasis, nbasis).
        rchols_a: Per-det half-rotated Cholesky (alpha), shape (ndet, naux, nup, nbasis).
        rchols_b: Per-det half-rotated Cholesky (beta), shape (ndet, naux, ndn, nbasis).
        trials_up: Trial alpha orbitals, shape (ndet, nbasis, nup).
        trials_dn: Trial beta orbitals, shape (ndet, nbasis, ndown).
        ci_coeffs: CI coefficients, shape (ndet,).
        eshift: Energy shift for population control (scalar).
        rng_key: JAX random key.
        fbbound: Force bias magnitude bound (default: 1.0).
        exp_nmax: Taylor expansion order for matrix exponential.

    Returns:
        phia_new, phib_new, weights_new, overlap_new, e_hybrid_new, rng_key.
    """
    dt = propagator['dt']
    expH1 = propagator['expH1']
    mf_shift = propagator['mf_shift']
    nwalkers = phia.shape[0]
    naux = chol.shape[0]

    if fbbound is None:
        fbbound = FBBOUND_DEFAULT

    # 1. Compute multi-det Green's function
    Ga, Gb, Ghalfa_all, Ghalfb_all, ovlp, _, _ = greens_function_multidet(
        phia, phib, trials_up, trials_dn, ci_coeffs)

    # 2. Apply first half of one-body propagator
    phia = jnp.einsum('pq,wqn->wpn', expH1, phia)
    phib = jnp.einsum('pq,wqn->wpn', expH1, phib)

    # 3. Force bias from aggregate multi-det GF (streamed over g)
    vbias_a_chunks = []
    vbias_b_chunks = []
    for g0, g1, chunk in _iter_chol_g_chunks(chol, chol_chunk_g):
        chunk_j = jnp.asarray(chunk)
        vbias_a_chunks.append(jnp.einsum('gpq,wqp->gw', chunk_j, Ga))
        vbias_b_chunks.append(jnp.einsum('gpq,wqp->gw', chunk_j, Gb))
    vbias_a = jnp.concatenate(vbias_a_chunks, axis=0)
    vbias_b = jnp.concatenate(vbias_b_chunks, axis=0)
    vbias = vbias_a + vbias_b

    # Optimal shift
    xbar = -jnp.sqrt(dt) * (1j * vbias - mf_shift[:, None])
    xbar = xbar.T  # (nwalkers, naux)

    # Bound force bias
    xbar_abs = jnp.abs(xbar)
    xbar = jnp.where(xbar_abs > fbbound,
                     xbar * fbbound / xbar_abs,
                     xbar)

    # 4. Sample auxiliary fields
    rng_key, subkey = jax.random.split(rng_key)
    xi = jax.random.normal(subkey, shape=(nwalkers, naux))

    # Shifted fields
    xshifted = xi - xbar

    # Constant factors for weight update
    cmf = -jnp.sqrt(dt) * jnp.einsum('wg,g->w', xshifted, mf_shift)
    cfb = (jnp.sum(xi * xbar, axis=1)
           - 0.5 * jnp.sum(xbar * xbar, axis=1))

    # 5. Construct and apply VHS (chunked over walkers and g)
    phia, phib = _apply_exp_vhs_from_chol(
        xshifted, chol, phia, phib, dt, exp_nmax,
        walker_chunk_size, chol_chunk_g)

    # 6. Apply second half of one-body propagator
    phia = jnp.einsum('pq,wqn->wpn', expH1, phia)
    phib = jnp.einsum('pq,wqn->wpn', expH1, phib)

    # 7. Compute new overlap and update weights
    _, _, _, _, ovlp_new, _, _ = greens_function_multidet(
        phia, phib, trials_up, trials_dn, ci_coeffs)

    weights_new, e_hybrid_new = _update_weights_phaseless(
        weights, ovlp, ovlp_new, cfb, cmf, e_hybrid, eshift, dt)

    return phia, phib, weights_new, ovlp_new, e_hybrid_new, rng_key


# ===================================================================
# Driver
# ===================================================================

def _autotune_afqmc_walkers(driver, free_mb, mem_frac=0.75, k=4):
    """Estimate walker count one AFQMC step fits in free GPU mem.

    Uses a delta probe: AOT-compiles the driver's propagator
    for both 1-walker and ``k``-walker inputs and subtracts
    ``alias_size + temp_size``, cancelling any walker-
    independent storage (Cholesky tensor, half-rotated chol,
    trial orbitals, propagator).  The remainder, divided by
    ``k - 1``, is the per-walker byte cost.  Falls back to
    1 MB/walker when AOT analysis is unavailable (e.g.
    DiskChol-backed runs cannot be AOT-traced).
    Informational only — caller does not mutate state.

    Parameters
    ----------
    driver : _AFQMCDriverGTO
        Driver instance whose ``propagator``, ``chol``, trial
        orbitals and (optionally) multi-det data live on the
        GPU.
    free_mb : float or None
        Free GPU memory in MiB; ``None`` assumes 4096.
    mem_frac : float
        Fraction of free memory to target (default 0.75).
    k : int
        Larger probe size for the delta (default 4).

    Returns
    -------
    (int, float)
        ``(n_rec, bytes_per_walker)``.
    """
    bpw = None
    try:
        nbasis = driver.nbasis
        nup = driver.nup
        ndown = driver.ndown

        def _probe(n):
            phia = jnp.zeros(
                (n, nbasis, nup), dtype=jnp.complex128)
            phib = jnp.zeros(
                (n, nbasis, ndown), dtype=jnp.complex128)
            weights = jnp.zeros((n,), dtype=jnp.float64)
            overlap = jnp.zeros((n,), dtype=jnp.complex128)
            e_hybrid = jnp.zeros((n,), dtype=jnp.complex128)
            return phia, phib, weights, overlap, e_hybrid

        if driver.multidet:
            def _call(phia, phib, weights, overlap, e_hybrid):
                return propagate_walkers_multidet(
                    phia, phib, weights, overlap, e_hybrid,
                    driver.propagator, driver.chol,
                    driver.rchols_a, driver.rchols_b,
                    driver.trials_up, driver.trials_dn,
                    driver.ci_coeffs,
                    jnp.float64(0.0), jax.random.key(0))
        else:
            def _call(phia, phib, weights, overlap, e_hybrid):
                return propagate_walkers(
                    phia, phib, weights, overlap, e_hybrid,
                    driver.propagator, driver.chol,
                    driver.rchol_a, driver.rchol_b,
                    driver.trial_up, driver.trial_dn,
                    jnp.float64(0.0), jax.random.key(0))

        m1 = jax.jit(_call).lower(
            *_probe(1)).compile().memory_analysis()
        mk = jax.jit(_call).lower(
            *_probe(k)).compile().memory_analysis()
        delta = (mk.peak_memory_in_bytes
                 - m1.peak_memory_in_bytes)
        if delta > 0:
            bpw = delta / (k - 1)
    except Exception:
        pass

    if not bpw or bpw <= 0:
        bpw = 1.0e6  # 1 MB/walker fallback

    free_bytes = (free_mb or 4096.0) * 1e6 * mem_frac
    n_rec = int(free_bytes / bpw)
    return max(10, n_rec), bpw


class _AFQMCDriverGTO:
    """Phaseless AFQMC driver.

    Separates setup (integral preparation, trial WF, propagator build)
    from execution (walker init, main loop, statistics).  Instantiate
    via :func:`get_afqmc_func`.
    """

    def __init__(self, mf, dt=0.005, chol_cut=1e-5, verbose=True,
                 trial=None, chol_h5_path=None, chol_chunk_g=128):
        """Prepare integrals and build the propagator.

        Args:
            mf: PySCF mean-field object (must have run kernel()).
            dt: Imaginary time step (default 0.005 Ha^{-1}).
            chol_cut: Cholesky decomposition threshold.
            verbose: Print progress.
            trial: Multi-determinant trial dict from extract_casscf_trial(),
                   or None for single-det HF trial (default).
            chol_h5_path: If set, the MO‑basis Cholesky tensor is stored
                in this HDF5 file (dataset ``chol_mo``) and streamed in
                g‑chunks during the run. Required for large systems
                (e.g. naphthalene cc‑pVTZ) where ``chol`` would not fit
                in device memory. Default None keeps ``chol`` in RAM.
            chol_chunk_g: Slab size along the auxiliary axis used when
                ``chol`` is disk‑backed. Default 128.
        """
        self.mf = mf
        self.dt = dt
        self.verbose = verbose
        self.multidet = trial is not None
        self.chol_chunk_g = chol_chunk_g

        if verbose:
            print("Preparing integrals (Cholesky decomposition)...")
            t0 = time.time()

        mo_coeff_override = trial['mo_coeff'] if trial is not None else None
        integrals = prepare_afqmc_integrals(
            mf, chol_cut=chol_cut, mo_coeff=mo_coeff_override,
            chol_h5_path=chol_h5_path, chol_chunk_g=chol_chunk_g)

        self.h1e = integrals['h1e']
        self.h1e_mod = integrals['h1e_mod']
        self.chol = integrals['chol']
        self.enuc = integrals['enuc']
        self.nbasis = integrals['nbasis']
        self.nup = integrals['nup']
        self.ndown = integrals['ndown']
        self.mo_coeff = integrals['mo_coeff']
        self.naux = self.chol.shape[0]

        if verbose:
            print(f"  nbasis={self.nbasis}, nup={self.nup}, "
                  f"ndown={self.ndown}, naux={self.naux}")
            print(f"  Integral preparation took {time.time()-t0:.2f} s")
            if isinstance(self.chol, DiskChol):
                print(f"  chol stored on disk: {self.chol.path} "
                      f"(chunk_g={self.chol_chunk_g})")

        if trial is not None:
            # Multi-determinant trial wavefunction
            ndet = trial['ndet']
            self.ci_coeffs = trial['ci_coeffs']

            # Build stacked trial matrices from occupation indices
            eye = jnp.eye(self.nbasis)
            # occ_up: (ndet, nup) int indices -> trials_up: (ndet, nbasis, nup)
            self.trials_up = eye[:, trial['occ_up']].transpose(1, 0, 2)
            self.trials_dn = eye[:, trial['occ_dn']].transpose(1, 0, 2)

            # Dominant determinant for walker initialization
            self.trial_up = self.trials_up[0]
            self.trial_dn = self.trials_dn[0]

            # Half-rotate Cholesky vectors for all determinants
            self.rchols_a, self.rchols_b = half_rotate_cholesky_multidet(
                self.chol, self.trials_up, self.trials_dn,
                chunk_g=chol_chunk_g)

            if verbose:
                print(f"  Multi-det trial: {ndet} determinants")
                print(f"  rchols_a shape: {self.rchols_a.shape}, "
                      f"rchols_b shape: {self.rchols_b.shape}")

            # Compute weighted G_charge for propagator
            ci_abs2 = jnp.abs(self.ci_coeffs) ** 2
            ci_abs2 = ci_abs2 / jnp.sum(ci_abs2)  # normalize
            G_charge = jnp.einsum(
                'd,dpi,dqi->pq', ci_abs2,
                self.trials_up, self.trials_up.conj())
            G_charge = G_charge + jnp.einsum(
                'd,dpi,dqi->pq', ci_abs2,
                self.trials_dn, self.trials_dn.conj())

            # Build propagator with multi-det charge density
            self.propagator = build_propagator(
                self.h1e_mod, self.chol,
                self.trial_up, self.trial_dn, dt,
                G_charge=G_charge, chol_chunk_g=chol_chunk_g)
        else:
            # Single-determinant HF trial
            self.trial_up = jnp.eye(self.nbasis, self.nup)
            self.trial_dn = jnp.eye(self.nbasis, self.ndown)

            # Half-rotate Cholesky vectors
            self.rchol_a, self.rchol_b = half_rotate_cholesky(
                self.chol, self.trial_up, self.trial_dn,
                chunk_g=chol_chunk_g)

            if verbose:
                print(f"  rchol_a shape: {self.rchol_a.shape}, "
                      f"rchol_b shape: {self.rchol_b.shape}")

            # Build propagator
            self.propagator = build_propagator(
                self.h1e_mod, self.chol,
                self.trial_up, self.trial_dn, dt,
                chol_chunk_g=chol_chunk_g)

        if verbose:
            print(f"  E_HF = {float(mf.e_tot):.10f}")
            print(f"  dt = {dt}")

    def __call__(self, rng_key=None, num_walkers=100, num_blocks=100,
                 num_steps_per_block=25, stabilize_freq=5,
                 pop_control_freq=5, num_blocks_equil=10,
                 fname_log="afqmc.log", walker_chunk_size=None):
        """Run the AFQMC simulation.

        Args:
            rng_key: JAX random key. If None, uses key(42).
            num_walkers: Number of walkers (default 100).
            num_blocks: Number of measurement blocks (default 100).
            num_steps_per_block: MC steps per block (default 25).
            stabilize_freq: QR reorthogonalization frequency (default 5).
            pop_control_freq: Population control frequency (default 5).
            num_blocks_equil: Number of equilibration blocks (default 10).
            fname_log: Log file path (default "afqmc.log").
                If None or "", prints to stdout.
            walker_chunk_size: If not None, build/apply the two‑body
                propagator VHS in chunks of this many walkers. Caps peak
                memory of the (nwalkers, nbasis, nbasis) VHS tensor at
                (chunk_size, nbasis, nbasis). Required for large bases
                (e.g., naphthalene cc‑pVTZ) where the full VHS would
                exceed device memory. Must divide num_walkers cleanly only
                if you also want even chunks; the last chunk is allowed
                to be shorter. Default None preserves prior behavior.

        Returns:
            dict with:
                'energy_blocks': Energy per block, shape (num_blocks,).
                'energy_mean': Mean energy after equilibration.
                'energy_err': Standard error of the mean.
                'energy_std': Standard deviation.
                'kappa': Integrated autocorrelation time.
                'ehf': Hartree-Fock energy for comparison.
        """
        if rng_key is None:
            rng_key = jax.random.key(42)

        if fname_log is None \
                or (isinstance(fname_log, str) and fname_log == ""):
            fout = sys.stdout
        else:
            fout = open(fname_log, 'w', 1)

        verbose = self.verbose

        # --- Informational GPU capacity estimate ---
        # Does NOT modify num_walkers.
        try:
            from .vmcopt_gto_linear import _get_free_gpu_mb
            free_mb = _get_free_gpu_mb()
            n_rec, bpw = _autotune_afqmc_walkers(self, free_mb)
            free_txt = (f"{free_mb:.0f} MiB free"
                        if free_mb is not None
                        else "free GPU mem unknown")
            print(f"ℹ️\tEst. GPU capacity: {n_rec} walkers "
                  f"(user requested {num_walkers}; "
                  f"{bpw / 1e6:.2f} MB/walker, {free_txt})",
                  file=fout)
        except Exception as e:
            print(f"ℹ️\tGPU capacity estimate unavailable: {e}",
                  file=fout)
        if verbose and walker_chunk_size is not None:
            print(f"  walker_chunk_size = {walker_chunk_size} "
                  f"(VHS built in chunks)", file=fout)

        # Initialize walkers
        walkers = Walkers(self.trial_up, self.trial_dn,
                          num_walkers, self.nbasis, self.nup, self.ndown)
        phia = walkers.phia
        phib = walkers.phib
        weights = walkers.weights
        overlap = walkers.overlap
        e_hybrid = jnp.zeros(num_walkers, dtype=jnp.complex128)

        phi_sharding, scalar_sharding = _make_afqmc_sharding(num_walkers)
        if phi_sharding is not None:
            phia = jax.device_put(phia, phi_sharding)
            phib = jax.device_put(phib, phi_sharding)
            weights = jax.device_put(weights, scalar_sharding)
            overlap = jax.device_put(overlap, scalar_sharding)
            e_hybrid = jax.device_put(e_hybrid, scalar_sharding)
        if verbose and phi_sharding is not None:
            print(f"  Sharding {num_walkers} walkers "
                  f"across {len(jax.devices())} devices", file=fout)

        # Main QMC loop
        total_blocks = num_blocks_equil + num_blocks
        energy_blocks = []
        eshift = 0.0
        step_count = 0

        # On-device accumulators — avoids per-step
        # GPU↔CPU sync that was previously incurred by
        # ``acc_weight += float(...)``.
        acc_weight = jnp.zeros((), dtype=jnp.float64)
        acc_ehybrid = jnp.zeros((), dtype=jnp.float64)

        if verbose:
            print(f"\nStarting AFQMC: {num_blocks_equil} eqlb + "
                  f"{num_blocks} prod blocks", file=fout)
            print(f"  {num_steps_per_block} steps/block, "
                  f"stabilize every {stabilize_freq} steps", file=fout)
            print("-" * 70, file=fout)
            print(f"{'Block':>6} {'E_mixed':>16} {'E_shift':>16} "
                  f"{'W_sum':>12} {'W_max':>12}", file=fout)
            print("-" * 70, file=fout)

        for iblock in range(total_blocks):
            acc_weight = jnp.zeros((), dtype=jnp.float64)
            acc_ehybrid = jnp.zeros((), dtype=jnp.float64)

            for istep in range(num_steps_per_block):
                step_count += 1

                # Reorthogonalize walkers
                if step_count % stabilize_freq == 0:
                    phia, phib, log_detR, _ = orthogonalize_walkers(
                        phia, phib)
                    overlap = overlap / jnp.exp(log_detR)

                # Propagate one step
                rng_key, step_key = jax.random.split(rng_key)
                if self.multidet:
                    phia, phib, weights, overlap, e_hybrid, _ = \
                        propagate_walkers_multidet(
                            phia, phib, weights, overlap, e_hybrid,
                            self.propagator, self.chol,
                            self.rchols_a, self.rchols_b,
                            self.trials_up, self.trials_dn,
                            self.ci_coeffs, eshift, step_key,
                            walker_chunk_size=walker_chunk_size,
                            chol_chunk_g=self.chol_chunk_g)
                else:
                    phia, phib, weights, overlap, e_hybrid, _ = \
                        propagate_walkers(
                            phia, phib, weights, overlap, e_hybrid,
                            self.propagator, self.chol,
                            self.rchol_a, self.rchol_b,
                            self.trial_up, self.trial_dn,
                            eshift, step_key,
                            walker_chunk_size=walker_chunk_size,
                            chol_chunk_g=self.chol_chunk_g)

                # Clip weights (ipie convention: skip step 1)
                if step_count > 1:
                    total_weight = jnp.sum(jnp.abs(weights))
                    wbound = total_weight * WEIGHT_CLIP_FRACTION
                    weights = jnp.clip(weights, 0.0, wbound)

                # Population control
                if step_count % pop_control_freq == 0:
                    rng_key, pc_key = jax.random.split(rng_key)
                    weights, phia, phib = population_control_comb(
                        weights, phia, phib, pc_key)
                    if phi_sharding is not None:
                        phia = jax.device_put(phia, phi_sharding)
                        phib = jax.device_put(phib, phi_sharding)
                        weights = jax.device_put(weights, scalar_sharding)

                # Accumulate weighted hybrid energy for eshift.
                # Stays on device — the float() cast is deferred
                # to end-of-block.
                w_abs = jnp.abs(weights)
                acc_weight = acc_weight + jnp.sum(w_abs)
                acc_ehybrid = acc_ehybrid + jnp.sum(
                    w_abs * e_hybrid.real)

            # End of block: compute energy via mixed estimator
            if self.multidet:
                Ga, Gb, Ghalfa_all, Ghalfb_all, ovlp_block, \
                    ovlp_a_all, ovlp_b_all = \
                    greens_function_multidet(
                        phia, phib, self.trials_up, self.trials_dn,
                        self.ci_coeffs)

                e_tot, e_1b, e_2b = local_energy_multidet(
                    self.h1e, Ga, Gb, Ghalfa_all, Ghalfb_all,
                    self.rchols_a, self.rchols_b, self.ci_coeffs,
                    ovlp_a_all, ovlp_b_all, self.enuc)
            else:
                Ga, Gb, Ghalfa, Ghalfb, ovlp_block = greens_function(
                    phia, phib, self.trial_up, self.trial_dn)

                e_tot, e_1b, e_2b = local_energy(
                    self.h1e, Ga, Gb, Ghalfa, Ghalfb,
                    self.rchol_a, self.rchol_b, self.enuc)

            # Weighted average over walkers
            w = jnp.abs(weights)
            w_sum = jnp.sum(w)
            e_block = jnp.sum(w * e_tot.real) / w_sum
            e_block_val = float(e_block)
            energy_blocks.append(e_block_val)

            # Update energy shift — single host sync per
            # block (was per step prior to this change).
            acc_weight_h = float(acc_weight)
            if acc_weight_h > 1e-10:
                eshift = float(acc_ehybrid) / acc_weight_h

            if verbose:
                is_eqlb = iblock < num_blocks_equil
                phase = "EQ" if is_eqlb else "  "
                print(f"{phase}{iblock:4d} {e_block_val:16.10f} "
                      f"{float(eshift):16.10f} "
                      f"{float(w_sum):12.4f} "
                      f"{float(jnp.max(weights)):12.4f}", file=fout)

        # Analyze results
        energy_blocks = np.array(energy_blocks)
        prod_energies = energy_blocks[num_blocks_equil:]

        e_mean, e_err, e_std, kappa = do_binning_analysis(
            jnp.array(prod_energies))

        if verbose:
            print("-" * 70, file=fout)
            print(f"E_HF     = {float(self.mf.e_tot):.10f}", file=fout)
            print(f"E_AFQMC  = {float(e_mean):.10f} +/- {float(e_err):.10f}",
                  file=fout)
            print(f"E_corr   = {float(e_mean) - float(self.mf.e_tot):.10f}",
                  file=fout)
            print(f"kappa    = {float(kappa):.2f}", file=fout)

        if fout is not sys.stdout:
            fout.close()

        return {
            'energy_blocks': energy_blocks,
            'energy_mean': float(e_mean),
            'energy_err': float(e_err),
            'energy_std': float(e_std),
            'kappa': float(kappa),
            'ehf': float(self.mf.e_tot),
        }

    def close(self):
        """Release the disk‑backed Cholesky HDF5 file, if any."""
        if isinstance(self.chol, DiskChol):
            self.chol.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


def get_afqmc_func(mf, dt=0.005, chol_cut=1e-5, verbose=True,
                   trial=None, chol_h5_path=None, chol_chunk_g=128):
    """Create a reusable AFQMC driver.

    Args:
        mf: PySCF mean-field object (must have run kernel()).
        dt: Imaginary time step (default 0.005 Ha^{-1}).
        chol_cut: Cholesky decomposition threshold.
        verbose: Print progress.
        trial: Multi-determinant trial dict from extract_casscf_trial(),
               or None for single-det HF trial (default).
        chol_h5_path: If set, MO‑basis Cholesky tensor lives in this
            HDF5 file and is streamed in g‑chunks during the run.
        chol_chunk_g: Slab size along the auxiliary axis for disk reads.

    Returns:
        _AFQMCDriverGTO instance (callable).
    """
    return _AFQMCDriverGTO(mf, dt=dt, chol_cut=chol_cut, verbose=verbose,
                        trial=trial, chol_h5_path=chol_h5_path,
                        chol_chunk_g=chol_chunk_g)
