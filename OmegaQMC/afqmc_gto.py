"""
Phaseless Auxiliary-Field Quantum Monte Carlo (AFQMC) driver.

Consolidates integral preparation, walker management, estimators,
propagation, and the main QMC loop into a single module.  The public
entry point is :func:`get_afqmc_func`, which returns a reusable
``_AFQMCDriver`` instance (setup once, run many times).

Uses JAX + PySCF.
"""

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


# ===================================================================
# Integrals
# ===================================================================

def chunked_cholesky(mol, chol_cut=1e-5, max_vecs=None):
    """Modified Cholesky decomposition of the ERI tensor.

    Computes the full ERI tensor and performs pivoted Cholesky decomposition
    on the reshaped (nbasis^2, nbasis^2) matrix:
        (pq|rs) ≈ Σ_γ L^γ_{pq} L^γ_{rs}

    Args:
        mol: PySCF Mole object.
        chol_cut: Threshold for convergence (max diagonal residual).
        max_vecs: Maximum number of Cholesky vectors. Defaults to 10*nbasis.

    Returns:
        chol_vecs: np.ndarray of shape (naux, nbasis, nbasis).
    """
    nbasis = mol.nao_nr()
    nao2 = nbasis * nbasis
    if max_vecs is None:
        max_vecs = 10 * nbasis

    # Compute full ERI tensor: (pq|rs)
    eri = mol.intor('int2e', aosym='s1') \
        .reshape(nbasis, nbasis, nbasis, nbasis)
    # Reshape to 2D: ERI_matrix[pq, rs] = (pq|rs)
    eri_2d = eri.reshape(nao2, nao2)

    # Pivoted Cholesky decomposition
    diag = np.diag(eri_2d).copy()

    chol_list = []
    for ivec in range(max_vecs):
        delta_max = np.max(diag)
        if delta_max < chol_cut:
            break

        # Select pivot
        nu = np.argmax(diag)

        # Compute Cholesky vector
        col = eri_2d[:, nu].copy()

        # Subtract contributions from previous vectors
        for prev_vec in chol_list:
            col -= prev_vec * prev_vec[nu]

        # Normalize
        col /= np.sqrt(delta_max)
        chol_list.append(col)

        # Update diagonal
        diag -= col * col
        diag = np.maximum(diag, 0.0)

    naux = len(chol_list)
    chol_vecs = np.array(chol_list).reshape(naux, nbasis, nbasis)
    return chol_vecs


def prepare_afqmc_integrals(mf, chol_cut=1e-5):
    """Prepare all integrals needed for AFQMC from a PySCF mean-field object.

    Computes:
    1. Cholesky-decomposed ERIs in MO basis
    2. Modified one-body Hamiltonian (h1e_mod)
    3. Nuclear repulsion energy

    Args:
        mf: PySCF mean-field object (RHF or UHF, must have run kernel()).
        chol_cut: Cholesky decomposition threshold.

    Returns:
        dict with keys:
            'h1e': one-body integrals in MO basis, shape (nbasis, nbasis)
            'h1e_mod': modified one-body Hamiltonian, shape (nbasis, nbasis)
            'chol': Cholesky vectors in MO basis, shape (naux, nbasis, nbasis)
            'enuc': nuclear repulsion energy (float)
            'nbasis': number of MO basis functions
            'nup': number of alpha electrons
            'ndown': number of beta electrons
            'mo_coeff': MO coefficient matrix
    """
    mol = mf.mol
    nbasis = mol.nao_nr()

    # Get MO coefficients
    mo_coeff = np.asarray(mf.mo_coeff)

    # Electron counts
    nup = mol.nelec[0]
    ndown = mol.nelec[1]

    # 1. One-body integrals in MO basis
    hcore_ao = np.asarray(mf.get_hcore())
    h1e = mo_coeff.T @ hcore_ao @ mo_coeff

    # 2. Cholesky decomposition in AO basis
    chol_ao = chunked_cholesky(mol, chol_cut=chol_cut)
    # naux = chol_ao.shape[0]

    # Transform to MO basis: L^γ_pq (MO) = C^T L^γ (AO) C
    chol_mo = np.einsum('ab,gbc,cd->gad', mo_coeff.T, chol_ao, mo_coeff)

    # 3. Modified one-body Hamiltonian
    # v0[i,k] = -0.5 * Σ_γ Σ_j L^γ_{ij} L^γ_{kj}
    v0 = np.einsum('gij,gkj->ik', chol_mo, chol_mo)
    v0 *= -0.5
    h1e_mod = h1e + v0

    # 4. Nuclear repulsion energy
    enuc = mol.energy_nuc()

    return {
        'h1e': jnp.array(h1e),
        'h1e_mod': jnp.array(h1e_mod),
        'chol': jnp.array(chol_mo),
        'enuc': float(enuc),
        'nbasis': nbasis,
        'nup': nup,
        'ndown': ndown,
        'mo_coeff': jnp.array(mo_coeff),
    }


def half_rotate_cholesky(chol, trial_up, trial_dn):
    """Half-rotate Cholesky vectors with trial wavefunction.

    Precomputes rchol_a[γ, i, q] = Σ_p trial_up[p, i]* L^γ_{pq}
    to reduce cost of force bias and energy from O(M^2 * naux) to
    O(nocc * M * naux).

    Args:
        chol: Cholesky vectors, shape (naux, nbasis, nbasis).
        trial_up: Trial alpha orbitals, shape (nbasis, nup).
        trial_dn: Trial beta orbitals, shape (nbasis, ndown).

    Returns:
        rchol_a: shape (naux, nup, nbasis).
        rchol_b: shape (naux, ndown, nbasis).
    """
    rchol_a = jnp.einsum('pi,gpq->giq', trial_up.conj(), chol)
    rchol_b = jnp.einsum('pi,gpq->giq', trial_dn.conj(), chol)
    return rchol_a, rchol_b


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


def population_control_comb(weights, phia, phib, rng_key):
    """Comb population control with weight rescaling.

    Rescales weights to sum to nwalkers (ipie convention), then
    places evenly-spaced 'teeth' on the cumulative weight distribution
    to resample walkers proportional to their weights.

    Args:
        weights: shape (nwalkers,).
        phia: shape (nwalkers, nbasis, nup).
        phib: shape (nwalkers, nbasis, ndown).
        rng_key: JAX random key.

    Returns:
        weights_new, phia_new, phib_new.
    """
    nwalkers = weights.shape[0]
    weights_np = np.array(weights)
    total_weight = np.sum(weights_np)

    # Rescale weights so total = nwalkers (ipie convention)
    scale = total_weight / nwalkers
    weights_np = weights_np / scale

    # Cumulative distribution (now sums to nwalkers)
    cumsum = np.cumsum(weights_np)

    # Random offset for comb
    r = float(jax.random.uniform(rng_key))

    # Comb teeth positions
    teeth = (np.arange(nwalkers) + r)

    # Select walkers
    new_indices = np.searchsorted(cumsum, teeth)
    new_indices = np.clip(new_indices, 0, nwalkers - 1)

    phia_np = np.array(phia)
    phib_np = np.array(phib)

    phia_new = phia_np[new_indices]
    phib_new = phib_np[new_indices]

    # All walkers get equal weight = 1.0 after resampling
    weights_new = np.ones(nwalkers)

    return jnp.array(weights_new), jnp.array(phia_new), jnp.array(phib_new)


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

def _greens_function_spin(phi, trial):
    """Green's function for one spin channel.

    Args:
        phi: Walker orbitals, shape (nwalkers, nbasis, nocc).
        trial: Trial orbitals, shape (nbasis, nocc).

    Returns:
        G: Full Green's function, shape (nwalkers, nbasis, nbasis).
        Ghalf: Half-rotated GF, shape (nwalkers, nocc, nbasis).
        ovlp: det(trial^dag phi), shape (nwalkers,).
        log_ovlp: log|det(trial^dag phi)|, shape (nwalkers,).
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
    """Compute the one-particle Green's function for all walkers.

    G = phi (trial^dag phi)^{-1} trial^dag

    Also returns the half-rotated Green's function:
    Ghalf = (trial^dag phi)^{-1} phi^T

    Args:
        phia: Walker alpha orbitals, shape (nwalkers, nbasis, nup).
        phib: Walker beta orbitals, shape (nwalkers, nbasis, ndown).
        trial_up: Trial alpha orbitals, shape (nbasis, nup).
        trial_dn: Trial beta orbitals, shape (nbasis, ndown).

    Returns:
        Ga: Full Green's function alpha, shape (nwalkers, nbasis, nbasis).
        Gb: Full Green's function beta, shape (nwalkers, nbasis, nbasis).
        Ghalfa: Half-rotated GF alpha, shape (nwalkers, nup, nbasis).
        Ghalfb: Half-rotated GF beta, shape (nwalkers, ndown, nbasis).
        overlap: <psi_T|phi> for each walker, shape (nwalkers,).
    """
    Ga, Ghalfa, ovlp_a, log_ovlp_a = _greens_function_spin(phia, trial_up)
    Gb, Ghalfb, ovlp_b, log_ovlp_b = _greens_function_spin(phib, trial_dn)

    # Total overlap is product of alpha and beta overlaps
    overlap = ovlp_a * ovlp_b

    return Ga, Gb, Ghalfa, Ghalfb, overlap


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
    """Two-body contribution to the local energy using half-rotated quantities.

    Coulomb: E_coul = 0.5 * Σ_γ (X_a^γ + X_b^γ)^2
    Exchange: E_exch = 0.5 * Σ_γ (Tr(T_a^γ T_a^{γT}) + Tr(T_b^γ T_b^{γT}))

    Args:
        Ghalfa: shape (nwalkers, nup, nbasis).
        Ghalfb: shape (nwalkers, ndown, nbasis).
        rchol_a: shape (naux, nup, nbasis).
        rchol_b: shape (naux, ndown, nbasis).

    Returns:
        e_coul: Coulomb energy, shape (nwalkers,).
        e_exch: Exchange energy, shape (nwalkers,).
    """
    # Coulomb: X_σ^γ = Tr(rchol_σ^γ * Ghalf_σ^T)
    Xa = jnp.einsum('giq,wiq->gw', rchol_a, Ghalfa)
    Xb = jnp.einsum('giq,wiq->gw', rchol_b, Ghalfb)

    e_coul = 0.5 * jnp.sum((Xa + Xb) ** 2, axis=0)

    # Exchange: T_σ^γ[i,j] = Σ_q rchol_σ[γ,i,q] * Ghalf_σ[w,j,q]
    Ta = jnp.einsum('giq,wjq->gwij', rchol_a, Ghalfa)
    Tb = jnp.einsum('giq,wjq->gwij', rchol_b, Ghalfb)

    # Tr(T @ T^T) = Σ_{ij} T[i,j]^2
    e_exch_a = 0.5 * jnp.sum(Ta * Ta.transpose(0, 1, 3, 2), axis=(0, 2, 3))
    e_exch_b = 0.5 * jnp.sum(Tb * Tb.transpose(0, 1, 3, 2), axis=(0, 2, 3))
    e_exch = e_exch_a + e_exch_b

    return e_coul, e_exch


@partial(jax.jit, static_argnames=[])
def local_energy(h1e, chol, Ga, Gb, Ghalfa, Ghalfb, rchol_a, rchol_b, enuc):
    """Compute the mixed-estimator local energy for all walkers.

    E_loc = E_1body + E_coulomb - E_exchange + E_nuc

    Uses half-rotated Cholesky vectors for efficiency.

    Args:
        h1e: One-body Hamiltonian in MO basis, shape (nbasis, nbasis).
        chol: Cholesky vectors, shape (naux, nbasis, nbasis).
        Ga: Alpha Green's function, shape (nwalkers, nbasis, nbasis).
        Gb: Beta Green's function, shape (nwalkers, nbasis, nbasis).
        Ghalfa: Half-rotated alpha GF, shape (nwalkers, nup, nbasis).
        Ghalfb: Half-rotated beta GF, shape (nwalkers, ndown, nbasis).
        rchol_a: Half-rotated Cholesky (alpha), shape (naux, nup, nbasis).
        rchol_b: Half-rotated Cholesky (beta), shape (naux, ndown, nbasis).
        enuc: Nuclear repulsion energy.

    Returns:
        e_tot: Total local energy per walker, shape (nwalkers,).
        e_1b: One-body energy per walker, shape (nwalkers,).
        e_2b: Two-body energy per walker, shape (nwalkers,).
    """
    e_1b = local_energy_1body(h1e, Ga, Gb, enuc)
    e_coul, e_exch = local_energy_2body(Ghalfa, Ghalfb, rchol_a, rchol_b)
    e_2b = e_coul - e_exch
    e_tot = e_1b + e_2b
    return e_tot, e_1b, e_2b


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


def build_propagator(h1e_mod, chol, trial_up, trial_dn, dt):
    """Build the one-body propagator and mean-field shift.

    Args:
        h1e_mod: Modified one-body Hamiltonian, shape (nbasis, nbasis).
        chol: Cholesky vectors, shape (naux, nbasis, nbasis).
        trial_up: Trial alpha orbitals, shape (nbasis, nup).
        trial_dn: Trial beta orbitals, shape (nbasis, ndown).
        dt: Imaginary time step.

    Returns:
        dict with:
            'expH1': Half-step one-body propagator, shape (nbasis, nbasis).
            'mf_shift': Mean-field shift, shape (naux,).
            'dt': Time step.
    """
    # Compute trial one-body density matrix
    G_trial_a = trial_up @ trial_up.T.conj()
    G_trial_b = trial_dn @ trial_dn.T.conj()
    G_charge = G_trial_a + G_trial_b

    # Mean-field shift (ipie convention)
    mf_shift = 1j * jnp.einsum('gpq,qp->g', chol, G_charge)

    # Construct the shifted one-body Hamiltonian
    vhs_mf = 1j * jnp.einsum('g,gpq->pq', mf_shift, chol)
    H1_shifted = h1e_mod - vhs_mf

    # Half-step one-body propagator: exp(-dt/2 * H1_shifted)
    expH1 = expm(-0.5 * dt * H1_shifted)

    return {
        'expH1': expH1,
        'mf_shift': mf_shift,
        'dt': dt,
    }


def propagate_walkers(phia, phib, weights, overlap, e_hybrid,
                      propagator, chol, rchol_a, rchol_b,
                      trial_up, trial_dn, eshift, rng_key,
                      fbbound=None, exp_nmax=6):
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
    Ga, Gb, Ghalfa, Ghalfb, ovlp = greens_function(
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

    # 5. Construct and apply VHS
    VHS = 1j * jnp.sqrt(dt) * jnp.einsum('wg,gpq->wpq', xshifted, chol)

    # Apply exp(VHS) via Taylor expansion
    phia = _apply_exp_vhs(VHS, phia, exp_nmax)
    phib = _apply_exp_vhs(VHS, phib, exp_nmax)

    # 6. Apply second half of one-body propagator
    phia = jnp.einsum('pq,wqn->wpn', expH1, phia)
    phib = jnp.einsum('pq,wqn->wpn', expH1, phib)

    # 7. Compute new overlap and update weights
    _, _, _, _, ovlp_new = greens_function(phia, phib, trial_up, trial_dn)

    weights_new, e_hybrid_new = _update_weights_phaseless(
        weights, ovlp, ovlp_new, cfb, cmf, e_hybrid, eshift, dt)

    return phia, phib, weights_new, ovlp_new, e_hybrid_new, rng_key


# ===================================================================
# Driver
# ===================================================================

class _AFQMCDriver:
    """Phaseless AFQMC driver.

    Separates setup (integral preparation, trial WF, propagator build)
    from execution (walker init, main loop, statistics).  Instantiate
    via :func:`get_afqmc_func`.
    """

    def __init__(self, mf, dt=0.005, chol_cut=1e-5, verbose=True):
        """Prepare integrals and build the propagator.

        Args:
            mf: PySCF mean-field object (must have run kernel()).
            dt: Imaginary time step (default 0.005 Ha^{-1}).
            chol_cut: Cholesky decomposition threshold.
            verbose: Print progress.
        """
        self.mf = mf
        self.dt = dt
        self.verbose = verbose

        # 1. Prepare integrals
        if verbose:
            print("Preparing integrals (Cholesky decomposition)...")
            t0 = time.time()

        integrals = prepare_afqmc_integrals(mf, chol_cut=chol_cut)
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

        # 2. Trial wavefunction (HF determinant)
        self.trial_up = jnp.eye(self.nbasis, self.nup)
        self.trial_dn = jnp.eye(self.nbasis, self.ndown)

        # Half-rotate Cholesky vectors
        self.rchol_a, self.rchol_b = half_rotate_cholesky(
            self.chol, self.trial_up, self.trial_dn)

        if verbose:
            print(f"  rchol_a shape: {self.rchol_a.shape}, "
                  f"rchol_b shape: {self.rchol_b.shape}")

        # 3. Build propagator
        self.propagator = build_propagator(
            self.h1e_mod, self.chol, self.trial_up, self.trial_dn, dt)

        if verbose:
            print(f"  E_HF = {float(mf.e_tot):.10f}")
            print(f"  dt = {dt}")

    def __call__(self, rng_key=None, num_walkers=100, num_blocks=100,
                 num_steps_per_block=25, stabilize_freq=5,
                 pop_control_freq=5, num_eqlb_blocks=10):
        """Run the AFQMC simulation.

        Args:
            rng_key: JAX random key. If None, uses key(42).
            num_walkers: Number of walkers (default 100).
            num_blocks: Number of measurement blocks (default 100).
            num_steps_per_block: MC steps per block (default 25).
            stabilize_freq: QR reorthogonalization frequency (default 5).
            pop_control_freq: Population control frequency (default 5).
            num_eqlb_blocks: Number of equilibration blocks (default 10).

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

        verbose = self.verbose
        if verbose:
            print(f"  num_walkers = {num_walkers}")

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
                  f"across {len(jax.devices())} devices")

        # Main QMC loop
        total_blocks = num_eqlb_blocks + num_blocks
        energy_blocks = []
        eshift = 0.0
        step_count = 0

        acc_weight = 0.0
        acc_ehybrid = 0.0

        if verbose:
            print(f"\nStarting AFQMC: {num_eqlb_blocks} eqlb + "
                  f"{num_blocks} prod blocks")
            print(f"  {num_steps_per_block} steps/block, "
                  f"stabilize every {stabilize_freq} steps")
            print("-" * 70)
            print(f"{'Block':>6} {'E_mixed':>16} {'E_shift':>16} "
                  f"{'W_sum':>12} {'W_max':>12}")
            print("-" * 70)

        for iblock in range(total_blocks):
            acc_weight = 0.0
            acc_ehybrid = 0.0

            for istep in range(num_steps_per_block):
                step_count += 1

                # Reorthogonalize walkers
                if step_count % stabilize_freq == 0:
                    phia, phib, log_detR, _ = orthogonalize_walkers(
                        phia, phib)
                    overlap = overlap / jnp.exp(log_detR)

                # Propagate one step
                rng_key, step_key = jax.random.split(rng_key)
                phia, phib, weights, overlap, e_hybrid, _ = propagate_walkers(
                    phia, phib, weights, overlap, e_hybrid,
                    self.propagator, self.chol, self.rchol_a, self.rchol_b,
                    self.trial_up, self.trial_dn, eshift, step_key)

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

                # Accumulate weighted hybrid energy for eshift
                w_step = jnp.sum(jnp.abs(weights))
                acc_weight += float(w_step)
                acc_ehybrid += float(jnp.sum(
                    jnp.abs(weights) * e_hybrid.real))

            # End of block: compute energy via mixed estimator
            Ga, Gb, Ghalfa, Ghalfb, ovlp_block = greens_function(
                phia, phib, self.trial_up, self.trial_dn)

            e_tot, e_1b, e_2b = local_energy(
                self.h1e, self.chol, Ga, Gb, Ghalfa, Ghalfb,
                self.rchol_a, self.rchol_b, self.enuc)

            # Weighted average over walkers
            w = jnp.abs(weights)
            w_sum = jnp.sum(w)
            e_block = jnp.sum(w * e_tot.real) / w_sum
            e_block_val = float(e_block)
            energy_blocks.append(e_block_val)

            # Update energy shift
            if acc_weight > 1e-10:
                eshift = acc_ehybrid / acc_weight

            if verbose:
                is_eqlb = iblock < num_eqlb_blocks
                phase = "EQ" if is_eqlb else "  "
                print(f"{phase}{iblock:4d} {e_block_val:16.10f} "
                      f"{float(eshift):16.10f} "
                      f"{float(w_sum):12.4f} "
                      f"{float(jnp.max(weights)):12.4f}")

        # Analyze results
        energy_blocks = np.array(energy_blocks)
        prod_energies = energy_blocks[num_eqlb_blocks:]

        e_mean, e_err, e_std, kappa = do_binning_analysis(
            jnp.array(prod_energies))

        if verbose:
            print("-" * 70)
            print(f"E_HF     = {float(self.mf.e_tot):.10f}")
            print(f"E_AFQMC  = {float(e_mean):.10f} +/- {float(e_err):.10f}")
            print(f"E_corr   = {float(e_mean) - float(self.mf.e_tot):.10f}")
            print(f"kappa    = {float(kappa):.2f}")

        return {
            'energy_blocks': energy_blocks,
            'energy_mean': float(e_mean),
            'energy_err': float(e_err),
            'energy_std': float(e_std),
            'kappa': float(kappa),
            'ehf': float(self.mf.e_tot),
        }


def get_afqmc_func(mf, dt=0.005, chol_cut=1e-5, verbose=True):
    """Create a reusable AFQMC driver.

    Args:
        mf: PySCF mean-field object (must have run kernel()).
        dt: Imaginary time step (default 0.005 Ha^{-1}).
        chol_cut: Cholesky decomposition threshold.
        verbose: Print progress.

    Returns:
        _AFQMCDriver instance (callable).
    """
    return _AFQMCDriver(mf, dt=dt, chol_cut=chol_cut, verbose=verbose)
