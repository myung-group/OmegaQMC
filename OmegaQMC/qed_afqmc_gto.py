"""
Phaseless QED-AFQMC (Quantum Electrodynamics Auxiliary-Field QMC).

Extends the standard AFQMC to include coupling to a quantized cavity
photon mode via the Pauli-Fierz Hamiltonian in the dipole gauge.

Reference: arXiv:2410.18838 (Bauer et al., 2024)

The integral preparation routine has moved to its canonical location:

* :func:`OmegaQMC.integrals.qed.prepare_qed_integrals`

Energy estimators and Green's functions shared with the plain AFQMC
driver are in:

* :mod:`OmegaQMC.observables.energy`
* :mod:`OmegaQMC.observables.greens`

Uses JAX + PySCF.
"""

import time

import numpy as np
import jax
import jax.numpy as jnp
from jax.scipy.linalg import expm
from functools import partial

from OmegaQMC.utils import do_binning_analysis
from OmegaQMC.observables.energy import (
    local_energy_1body,
    local_energy_2body,
)
from OmegaQMC.observables.greens import (
    greens_function,
    greens_function_force_bias,
    greens_function_overlap,
)
from OmegaQMC.integrals.cholesky import (
    chunked_cholesky,
    half_rotate_cholesky,
    DiskChol,
    _iter_chol_g_chunks,
)
from OmegaQMC.integrals.qed import (
    prepare_qed_integrals,
)
from OmegaQMC.afqmc_gto import (
    _apply_exp_vhs,
    _apply_exp_vhs_from_chol,
    orthogonalize_walkers,
    _update_weights_phaseless,
    _make_afqmc_sharding,
    WEIGHT_CLIP_FRACTION,
    FBBOUND_DEFAULT,
)


# ===================================================================
# Walker management
# ===================================================================

class QEDWalkers:
    """AFQMC walkers augmented with a photon coordinate q.

    Each walker carries (phia, phib, q) plus importance sampling metadata.
    """

    def __init__(self, trial_up, trial_dn, nwalkers, nbasis, nup, ndown,
                 q0=0.0):
        self.nwalkers = nwalkers
        self.nbasis = nbasis
        self.nup = nup
        self.ndown = ndown

        self.phia = jnp.tile(trial_up[None, :, :], (nwalkers, 1, 1))
        self.phib = jnp.tile(trial_dn[None, :, :], (nwalkers, 1, 1))
        self.weights = jnp.ones(nwalkers)
        self.overlap = jnp.ones(nwalkers, dtype=jnp.complex128)
        self.e_hybrid = jnp.zeros(nwalkers, dtype=jnp.complex128)
        self.log_detR = jnp.zeros(nwalkers)
        self.log_shift = jnp.zeros(nwalkers)

        # Photon displacement coordinate for each walker
        self.q = jnp.full(nwalkers, q0)


@jax.jit
def population_control_comb_qed(weights, phia, phib, q, rng_key):
    """Comb population control that also resamples the photon coordinate.

    Pure JAX so the gather stays on device — the previous numpy
    implementation forced a full GPU↔CPU round-trip of the
    walker matrices on every population-control step.

    Args:
        weights: shape (nwalkers,).
        phia: shape (nwalkers, nbasis, nup).
        phib: shape (nwalkers, nbasis, ndown).
        q: photon coordinate, shape (nwalkers,).
        rng_key: JAX random key.

    Returns:
        weights_new, phia_new, phib_new, q_new.
    """
    nwalkers = weights.shape[0]
    total_weight = jnp.sum(weights)

    scale = total_weight / nwalkers
    weights_scaled = weights / scale

    cumsum = jnp.cumsum(weights_scaled)
    r = jax.random.uniform(rng_key)
    teeth = jnp.arange(nwalkers, dtype=cumsum.dtype) + r

    new_indices = jnp.searchsorted(cumsum, teeth)
    new_indices = jnp.clip(new_indices, 0, nwalkers - 1)

    phia_new = phia[new_indices]
    phib_new = phib[new_indices]
    q_new = q[new_indices]
    weights_new = jnp.ones(nwalkers, dtype=weights.dtype)

    return weights_new, phia_new, phib_new, q_new


# ===================================================================
# Estimators
# ===================================================================

@partial(jax.jit, static_argnames=[])
def qed_local_energy(h1e, Ga, Gb, Ghalfa, Ghalfb,
                     rchol_a, rchol_b, enuc,
                     q, dip_mo, omega, s, q0):
    """Compute QED-AFQMC local energy for all walkers.

    E_loc = E_1body + E_ep + E_2body + E_photon

    Args:
        h1e: bare one-body Hamiltonian (nbasis, nbasis).
        Ga, Gb: Green's functions (nwalkers, nbasis, nbasis).
        Ghalfa, Ghalfb: half-rotated GFs.
        rchol_a, rchol_b: half-rotated augmented Cholesky.
        enuc: nuclear repulsion energy.
        q: photon coordinate per walker (nwalkers,).
        dip_mo: dipole matrix in MO basis (nbasis, nbasis).
        omega: photon frequency.
        s: photon trial squeeze parameter.
        q0: photon trial displacement.

    Returns:
        e_tot, e_1b, e_2b, e_ph: per walker, shape (nwalkers,).
    """
    # Electronic one-body (q-independent part)
    e_1b_0 = local_energy_1body(h1e, Ga, Gb, enuc)

    # Electron-photon coupling: √Ω * q * Tr(d @ (Ga + Gb))
    tr_dip_a = jnp.einsum('pq,wqp->w', dip_mo, Ga)
    tr_dip_b = jnp.einsum('pq,wqp->w', dip_mo, Gb)
    e_ep = jnp.sqrt(omega) * q * (tr_dip_a + tr_dip_b)

    # Electronic two-body (includes DSE via augmented Cholesky)
    e_coul, e_exch = local_energy_2body(Ghalfa, Ghalfb, rchol_a, rchol_b)
    e_2b = e_coul - e_exch

    # Photon energy: E_ph = Ω/2 * (s - s²(q-q₀)² + q² - 1)
    # Derived from <f|H_ph|phi>/(<f|phi>) with Gaussian trial f(q)
    e_ph = omega / 2.0 * (s - s**2 * (q - q0)**2 + q**2 - 1.0)

    e_1b = e_1b_0 + e_ep
    e_tot = e_1b + e_2b + e_ph

    return e_tot, e_1b, e_2b, e_ph


# ===================================================================
# Propagation
# ===================================================================

def build_qed_propagator(h1e_mod_0, chol_qed, trial_up, trial_dn, dt,
                         chol_chunk_g=None):
    """Build the q-independent part of the one-body propagator.

    The q-dependent contribution (√Ω * q * dip_mo) is applied dynamically
    via Taylor expansion during propagation. ``chol_qed`` may be either
    in‑memory (array) or a :class:`DiskChol`; in the latter case
    mf_shift and vhs_mf are accumulated in g‑chunks so the full
    augmented Cholesky tensor never lives in device memory.

    Args:
        h1e_mod_0: q-independent modified one-body Hamiltonian
            (nbasis, nbasis).
        chol_qed: augmented Cholesky vectors (naux+1, nbasis, nbasis).
        trial_up, trial_dn: trial orbitals.
        dt: imaginary time step.
        chol_chunk_g: g‑axis slab size for streaming. None uses the
            DiskChol's own chunk_g, or no chunking for arrays.

    Returns:
        dict with 'expH1_0', 'mf_shift', 'dt'.
    """
    G_trial_a = trial_up @ trial_up.T.conj()
    G_trial_b = trial_dn @ trial_dn.T.conj()
    G_charge = G_trial_a + G_trial_b

    naux = chol_qed.shape[0]
    nbasis = chol_qed.shape[1]

    # mf_shift[g] = 1j * sum_{p,q} chol_qed[g,p,q] * G_charge[q,p]
    mf_shift = jnp.zeros(naux, dtype=jnp.complex128)
    for g0, g1, chunk in _iter_chol_g_chunks(chol_qed, chol_chunk_g):
        chunk_j = jnp.asarray(chunk)
        contrib = 1j * jnp.einsum('gpq,qp->g', chunk_j, G_charge)
        mf_shift = mf_shift.at[g0:g1].set(contrib)

    # vhs_mf[p,q] = 1j * sum_g mf_shift[g] * chol_qed[g,p,q]
    vhs_mf = jnp.zeros((nbasis, nbasis), dtype=jnp.complex128)
    for g0, g1, chunk in _iter_chol_g_chunks(chol_qed, chol_chunk_g):
        chunk_j = jnp.asarray(chunk)
        vhs_mf = vhs_mf + 1j * jnp.einsum(
            'g,gpq->pq', mf_shift[g0:g1], chunk_j)

    H1_shifted_0 = h1e_mod_0 - vhs_mf
    expH1_0 = expm(-0.5 * dt * H1_shifted_0)

    return {
        'expH1_0': expH1_0,
        'mf_shift': mf_shift,
        'dt': dt,
    }


def propagate_photon(q, omega, dt, s, q0, rng_key):
    """Propagate photon coordinate by a half-step using exact HO kernel.

    The trial photon wavefunction is Gaussian:
        f(q) = (s/π)^{1/4} exp(-s(q - q₀)²/2)

    The exact harmonic oscillator propagator is used with importance
    sampling guided by the trial wavefunction.

    Args:
        q: photon coordinate per walker, shape (nwalkers,).
        omega: photon frequency.
        dt: FULL time step (this function applies a half-step).
        s: squeeze parameter.
        q0: coherent state displacement.
        rng_key: JAX random key.

    Returns:
        q_new: updated photon coordinate (nwalkers,).
        log_weight_ph: log of photon weight factor (nwalkers,).
        rng_key: consumed key.
    """
    delta = omega * dt / 2.0  # half-step parameter

    cosh_d = jnp.cosh(delta)
    tanh_d = jnp.tanh(delta)

    # Drift from trial wavefunction: ∂_q log f(q) = -s*(q - q0)
    drift = -s * (q - q0)

    # Proposal distribution: q' ~ N(mean, variance)
    q_mean = q / cosh_d + tanh_d * drift
    sigma = jnp.sqrt(tanh_d)

    rng_key, subkey = jax.random.split(rng_key)
    eta = jax.random.normal(subkey, shape=q.shape)
    q_new = q_mean + sigma * eta

    # Weight update (log scale) from importance sampling correction
    # log(f(q_new)/f(q))
    log_trial_ratio = -s / 2.0 * ((q_new - q0)**2 - (q - q0)**2)

    # Correction from the importance-sampled kernel (Eq. 16-17)
    correction = (q / cosh_d - q_new) * drift
    additional = -tanh_d / 2.0 * (-(drift**2) + q**2) + delta / 2.0

    log_weight_ph = (log_trial_ratio + correction + additional
                     - 0.5 * jnp.log(cosh_d))

    return q_new, log_weight_ph, rng_key


def _apply_exp_vhs_photon(coeff, mat, phia, phib, nmax, chunk_size):
    """Apply exp(coeff[w] * mat) to phia/phib in walker chunks.

    Caps peak memory of the photon‑coupling VHS at
    ``(chunk_size, nbasis, nbasis)`` instead of the full ``nwalkers``.
    """
    nw = phia.shape[0]
    if chunk_size is None or chunk_size >= nw:
        VHS = coeff[:, None, None] * mat[None, :, :]
        return _apply_exp_vhs(VHS, phia, nmax), _apply_exp_vhs(VHS, phib, nmax)

    a_chunks, b_chunks = [], []
    for start in range(0, nw, chunk_size):
        end = min(start + chunk_size, nw)
        VHS_c = coeff[start:end, None, None] * mat[None, :, :]
        a_chunks.append(_apply_exp_vhs(VHS_c, phia[start:end], nmax))
        b_chunks.append(_apply_exp_vhs(VHS_c, phib[start:end], nmax))
    return (jnp.concatenate(a_chunks, axis=0),
            jnp.concatenate(b_chunks, axis=0))


def propagate_qed_walkers(phia, phib, q, weights, overlap, e_hybrid,
                          propagator, chol_qed, rchol_a, rchol_b,
                          trial_up, trial_dn, eshift, rng_key,
                          dip_mo, omega, s, q0,
                          fbbound=None, exp_nmax=6,
                          walker_chunk_size=None, chol_chunk_g=None):
    """Propagate QED-AFQMC walkers by one time step.

    Suzuki-Trotter decomposition:
        e^{-dt H} ≈ e^{-dt/2 H_ph} · e^{-dt/2 H_1(q)} · e^{-dt H_2}
                     · e^{-dt/2 H_1(q)} · e^{-dt/2 H_ph}

    Args:
        phia, phib: walker orbitals.
        q: photon coordinate (nwalkers,).
        weights, overlap, e_hybrid: walker metadata.
        propagator: dict from build_qed_propagator.
        chol_qed: augmented Cholesky (naux+1, nbasis, nbasis).
        rchol_a, rchol_b: half-rotated augmented Cholesky.
        trial_up, trial_dn: trial orbitals.
        eshift: energy shift.
        rng_key: JAX random key.
        dip_mo: dipole matrix (nbasis, nbasis).
        omega: photon frequency.
        s, q0: photon trial parameters.
        fbbound: force bias bound.
        exp_nmax: Taylor order for two-body exponential.

    Returns:
        phia, phib, q, weights, overlap, e_hybrid, rng_key.
    """
    dt = propagator['dt']
    expH1_0 = propagator['expH1_0']
    mf_shift = propagator['mf_shift']
    nwalkers = phia.shape[0]
    naux = chol_qed.shape[0]

    if fbbound is None:
        fbbound = FBBOUND_DEFAULT

    # 0. Compute Green's function (for force bias)
    # (specialized: only Ghalf + overlap needed)
    Ghalfa, Ghalfb, ovlp = greens_function_force_bias(
        phia, phib, trial_up, trial_dn)

    # 1. First photon half-step
    q, log_wt_ph1, rng_key = propagate_photon(q, omega, dt, s, q0, rng_key)

    # 2. First half one-body: exp(-dt/2 * (H1_0 + √Ω*q*dip))
    #    Split: apply q-dependent Taylor first, then q-independent matrix
    coeff_1 = -0.5 * dt * jnp.sqrt(omega) * q  # (nwalkers,)
    phia, phib = _apply_exp_vhs_photon(
        coeff_1, dip_mo, phia, phib, 4, walker_chunk_size)
    phia = jnp.einsum('pq,wqn->wpn', expH1_0, phia)
    phib = jnp.einsum('pq,wqn->wpn', expH1_0, phib)

    # 3. Two-body propagation (same as standard AFQMC, with naux+1 fields)
    # Force bias
    vbias_a = jnp.einsum('giq,wiq->gw', rchol_a, Ghalfa)
    vbias_b = jnp.einsum('giq,wiq->gw', rchol_b, Ghalfb)
    vbias = vbias_a + vbias_b

    xbar = -jnp.sqrt(dt) * (1j * vbias - mf_shift[:, None])
    xbar = xbar.T  # (nwalkers, naux+1)

    # Bound force bias
    xbar_abs = jnp.abs(xbar)
    xbar = jnp.where(xbar_abs > fbbound, xbar * fbbound / xbar_abs, xbar)

    # Sample auxiliary fields
    rng_key, subkey = jax.random.split(rng_key)
    xi = jax.random.normal(subkey, shape=(nwalkers, naux))
    xshifted = xi - xbar

    # Weight correction factors
    cmf = -jnp.sqrt(dt) * jnp.einsum('wg,g->w', xshifted, mf_shift)
    cfb = (jnp.sum(xi * xbar, axis=1)
           - 0.5 * jnp.sum(xbar * xbar, axis=1))

    # Construct and apply VHS (chunked over walkers and g)
    phia, phib = _apply_exp_vhs_from_chol(
        xshifted, chol_qed, phia, phib, dt, exp_nmax,
        walker_chunk_size, chol_chunk_g)

    # 4. Second half one-body (reverse order for symmetric Trotter)
    phia = jnp.einsum('pq,wqn->wpn', expH1_0, phia)
    phib = jnp.einsum('pq,wqn->wpn', expH1_0, phib)
    coeff_2 = -0.5 * dt * jnp.sqrt(omega) * q  # use current q
    phia, phib = _apply_exp_vhs_photon(
        coeff_2, dip_mo, phia, phib, 4, walker_chunk_size)

    # 5. Second photon half-step
    q, log_wt_ph2, rng_key = propagate_photon(q, omega, dt, s, q0, rng_key)

    # 6. Weight update
    # Electronic part: phaseless approximation
    ovlp_new = greens_function_overlap(
        phia, phib, trial_up, trial_dn)
    weights_new, e_hybrid_new = _update_weights_phaseless(
        weights, ovlp, ovlp_new, cfb, cmf, e_hybrid, eshift, dt)

    # Photon part: multiply by exp(log_wt_ph1 + log_wt_ph2)
    ph_weight = jnp.exp(jnp.clip(log_wt_ph1 + log_wt_ph2, -10.0, 10.0))
    weights_new = weights_new * ph_weight

    return phia, phib, q, weights_new, ovlp_new, e_hybrid_new, rng_key


# ===================================================================
# Driver
# ===================================================================

class _QEDAFQMCDriverGTO:
    """Phaseless QED-AFQMC driver for cavity QED problems.

    Implements the dipole gauge Pauli-Fierz Hamiltonian coupled to
    a single photon mode (frequency omega, coupling vector lambda*epsilon).
    """

    def __init__(self, mf, omega, coupling_vec, dt=0.005, chol_cut=1e-5,
                 s=1.0, q0=0.0, verbose=True,
                 chol_h5_path=None, chol_chunk_g=128):
        """Prepare QED integrals and build the propagator.

        Args:
            mf: PySCF mean-field object (must have run kernel()).
            omega: Photon frequency in Hartree.
            coupling_vec: Light-matter coupling vector (3,). Direction gives
                polarization ε, magnitude gives coupling strength λ.
            dt: Imaginary time step.
            chol_cut: Cholesky decomposition threshold.
            s: Photon trial squeeze parameter (default 1.0 for dipole gauge).
            q0: Photon trial displacement (default 0.0).
            verbose: Print progress.
            chol_h5_path: If set, the augmented MO‑basis Cholesky tensor
                is stored in this HDF5 file (dataset ``chol_mo``) and
                streamed in g‑chunks during the run. Default None keeps
                ``chol_qed`` in RAM.
            chol_chunk_g: Slab size along the auxiliary axis for disk
                reads. Default 128.
        """
        self.mf = mf
        self.dt = dt
        self.omega = float(omega)
        self.s = float(s)
        self.q0 = float(q0)
        self.verbose = verbose
        self.chol_chunk_g = chol_chunk_g

        if verbose:
            print("Preparing QED-AFQMC integrals...")
            t0 = time.time()

        integrals = prepare_qed_integrals(
            mf, omega, coupling_vec, chol_cut=chol_cut,
            chol_h5_path=chol_h5_path, chol_chunk_g=chol_chunk_g)
        self.h1e = integrals['h1e']
        self.h1e_mod_0 = integrals['h1e_mod_0']
        self.chol_qed = integrals['chol_qed']
        self.dip_mo = integrals['dip_mo']
        self.enuc = integrals['enuc']
        self.nbasis = integrals['nbasis']
        self.nup = integrals['nup']
        self.ndown = integrals['ndown']
        self.mo_coeff = integrals['mo_coeff']
        self.naux = self.chol_qed.shape[0]  # naux_coulomb + 1 (DSE)

        if verbose:
            print(f"  nbasis={self.nbasis}, nup={self.nup}, "
                  f"ndown={self.ndown}, naux={self.naux} (incl. DSE)")
            print(f"  omega={self.omega:.4f} Ha, "
                  f"||coupling_vec||={float(jnp.linalg.norm(jnp.array(coupling_vec))):.6f}")
            print(f"  Integral preparation took {time.time()-t0:.2f} s")
            if isinstance(self.chol_qed, DiskChol):
                print(f"  chol_qed stored on disk: {self.chol_qed.path} "
                      f"(chunk_g={self.chol_chunk_g})")

        # Trial wavefunction: HF determinant x Gaussian photon
        self.trial_up = jnp.eye(self.nbasis, self.nup)
        self.trial_dn = jnp.eye(self.nbasis, self.ndown)

        # Half-rotate augmented Cholesky
        self.rchol_a, self.rchol_b = half_rotate_cholesky(
            self.chol_qed, self.trial_up, self.trial_dn,
            chunk_g=chol_chunk_g)

        if verbose:
            print(f"  rchol_a shape: {self.rchol_a.shape}, "
                  f"rchol_b shape: {self.rchol_b.shape}")

        # Build propagator (q-independent part)
        self.propagator = build_qed_propagator(
            self.h1e_mod_0, self.chol_qed,
            self.trial_up, self.trial_dn, dt,
            chol_chunk_g=chol_chunk_g)

        if verbose:
            print(f"  E_HF = {float(mf.e_tot):.10f}")
            print(f"  dt = {dt}, s = {s}, q0 = {q0}")

    def __call__(self, rng_key=None, num_walkers=100, num_blocks=100,
                 num_steps_per_block=25, stabilize_freq=5,
                 pop_control_freq=5, num_blocks_equil=10,
                 walker_chunk_size=None):
        """Run the QED-AFQMC simulation.

        Args:
            rng_key: JAX random key. If None, uses key(42).
            num_walkers: Number of walkers.
            num_blocks: Number of measurement blocks.
            num_steps_per_block: MC steps per block.
            stabilize_freq: QR reorthogonalization frequency.
            pop_control_freq: Population control frequency.
            num_blocks_equil: Number of equilibration blocks.
            walker_chunk_size: If not None, build/apply both the
                two‑body VHS and the photon‑coupling VHS_q in chunks of
                this many walkers. Caps peak memory of the
                ``(nwalkers, nbasis, nbasis)`` VHS tensors.

        Returns:
            dict with energy statistics and photon observables.
        """
        if rng_key is None:
            rng_key = jax.random.key(42)

        verbose = self.verbose
        if verbose:
            print(f"  num_walkers = {num_walkers}")

        # Initialize walkers
        walkers = QEDWalkers(self.trial_up, self.trial_dn,
                             num_walkers, self.nbasis,
                             self.nup, self.ndown, q0=self.q0)
        phia = walkers.phia
        phib = walkers.phib
        q = walkers.q
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
            q = jax.device_put(q, scalar_sharding)
        if verbose and phi_sharding is not None:
            print(f"  Sharding {num_walkers} walkers across "
                  f"{len(jax.devices())} devices")

        # Main QMC loop
        total_blocks = num_blocks_equil + num_blocks
        energy_blocks = []
        q_mean_blocks = []
        e_ph_blocks = []
        eshift = 0.0
        step_count = 0

        if verbose:
            print(f"\nStarting QED-AFQMC: {num_blocks_equil} eqlb + "
                  f"{num_blocks} prod blocks")
            print(f"  {num_steps_per_block} steps/block, "
                  f"stabilize every {stabilize_freq} steps")
            print("-" * 80)
            print(f"{'Block':>6} {'E_total':>16} {'E_shift':>16} "
                  f"{'<q>':>10} {'E_ph':>12} {'W_sum':>10}")
            print("-" * 80)

        for iblock in range(total_blocks):
            # On-device accumulators — reduces per-step
            # GPU↔CPU sync to one per block.
            acc_weight = jnp.zeros((), dtype=jnp.float64)
            acc_ehybrid = jnp.zeros((), dtype=jnp.float64)

            for istep in range(num_steps_per_block):
                step_count += 1

                # QR reorthogonalization
                if step_count % stabilize_freq == 0:
                    phia, phib, log_detR, _ = orthogonalize_walkers(
                        phia, phib)
                    overlap = overlap / jnp.exp(log_detR)

                # Propagate one step (QED)
                rng_key, step_key = jax.random.split(rng_key)
                phia, phib, q, weights, overlap, e_hybrid, _ = \
                    propagate_qed_walkers(
                        phia, phib, q, weights, overlap, e_hybrid,
                        self.propagator, self.chol_qed,
                        self.rchol_a, self.rchol_b,
                        self.trial_up, self.trial_dn, eshift, step_key,
                        self.dip_mo, self.omega, self.s, self.q0,
                        walker_chunk_size=walker_chunk_size,
                        chol_chunk_g=self.chol_chunk_g)

                # Clip weights (skip step 1)
                if step_count > 1:
                    total_weight = jnp.sum(jnp.abs(weights))
                    wbound = total_weight * WEIGHT_CLIP_FRACTION
                    weights = jnp.clip(weights, 0.0, wbound)

                # Population control (QED: also resamples q)
                if step_count % pop_control_freq == 0:
                    rng_key, pc_key = jax.random.split(rng_key)
                    weights, phia, phib, q = population_control_comb_qed(
                        weights, phia, phib, q, pc_key)
                    if phi_sharding is not None:
                        phia = jax.device_put(phia, phi_sharding)
                        phib = jax.device_put(phib, phi_sharding)
                        weights = jax.device_put(weights, scalar_sharding)
                        q = jax.device_put(q, scalar_sharding)

                # Accumulate for eshift on-device — single
                # host sync at end of block instead of per step.
                w_abs = jnp.abs(weights)
                acc_weight = acc_weight + jnp.sum(w_abs)
                acc_ehybrid = acc_ehybrid + jnp.sum(
                    w_abs * e_hybrid.real)

            # End of block: compute energy
            Ga, Gb, Ghalfa, Ghalfb, _ = greens_function(
                phia, phib, self.trial_up, self.trial_dn)

            e_tot, e_1b, e_2b, e_ph = qed_local_energy(
                self.h1e, Ga, Gb, Ghalfa, Ghalfb,
                self.rchol_a, self.rchol_b, self.enuc,
                q, self.dip_mo, self.omega, self.s, self.q0)

            w = jnp.abs(weights)
            w_sum = jnp.sum(w)
            e_block = float(jnp.sum(w * e_tot.real) / w_sum)
            energy_blocks.append(e_block)

            # Photon observables
            q_mean = float(jnp.sum(w * q) / w_sum)
            q_mean_blocks.append(q_mean)
            e_ph_block = float(jnp.sum(w * e_ph.real) / w_sum)
            e_ph_blocks.append(e_ph_block)

            # Update energy shift — single host sync per block.
            acc_weight_h = float(acc_weight)
            if acc_weight_h > 1e-10:
                eshift = float(acc_ehybrid) / acc_weight_h

            if verbose:
                is_eqlb = iblock < num_blocks_equil
                phase = "EQ" if is_eqlb else "  "
                print(f"{phase}{iblock:4d} {e_block:16.10f} "
                      f"{float(eshift):16.10f} "
                      f"{q_mean:10.4f} {e_ph_block:12.6f} "
                      f"{float(w_sum):10.4f}")

        # Analyze results
        energy_blocks = np.array(energy_blocks)
        prod_energies = energy_blocks[num_blocks_equil:]

        e_mean, e_err, e_std, kappa = do_binning_analysis(
            jnp.array(prod_energies))

        q_mean_prod = np.array(q_mean_blocks[num_blocks_equil:])
        e_ph_prod = np.array(e_ph_blocks[num_blocks_equil:])

        if verbose:
            print("-" * 80)
            print(f"E_HF       = {float(self.mf.e_tot):.10f}")
            print(f"E_QED-AFQMC = {float(e_mean):.10f} +/- "
                  f"{float(e_err):.10f}")
            print(f"E_corr     = {float(e_mean) - float(self.mf.e_tot):.10f}")
            print(f"<q>        = {float(np.mean(q_mean_prod)):.6f}")
            print(f"E_photon   = {float(np.mean(e_ph_prod)):.6f}")
            print(f"kappa      = {float(kappa):.2f}")

        return {
            'energy_blocks': energy_blocks,
            'energy_mean': float(e_mean),
            'energy_err': float(e_err),
            'energy_std': float(e_std),
            'kappa': float(kappa),
            'ehf': float(self.mf.e_tot),
            'q_mean': float(np.mean(q_mean_prod)),
            'e_photon_mean': float(np.mean(e_ph_prod)),
            'q_mean_blocks': np.array(q_mean_blocks),
            'e_ph_blocks': np.array(e_ph_blocks),
        }

    def close(self):
        """Release the disk‑backed Cholesky HDF5 file, if any."""
        if isinstance(self.chol_qed, DiskChol):
            self.chol_qed.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


def get_qed_afqmc_func(mf, omega, coupling_vec, dt=0.005, chol_cut=1e-5,
                       s=1.0, q0=0.0, verbose=True,
                       chol_h5_path=None, chol_chunk_g=128):
    """Create a reusable QED-AFQMC driver.

    Args:
        mf: PySCF mean-field object (must have run kernel()).
        omega: Photon frequency in Hartree.
        coupling_vec: Light-matter coupling vector (3,). Direction gives
            polarization ε, magnitude gives coupling strength λ.
        dt: Imaginary time step (default 0.005).
        chol_cut: Cholesky decomposition threshold.
        s: Photon trial squeeze parameter (1.0 for dipole gauge).
        q0: Photon trial displacement (0.0 for no displacement).
        verbose: Print progress.
        chol_h5_path: If set, augmented MO‑basis Cholesky lives in this
            HDF5 file and is streamed in g‑chunks during the run.
        chol_chunk_g: Slab size along the auxiliary axis for disk reads.

    Returns:
        _QEDAFQMCDriverGTO instance (callable).
    """
    return _QEDAFQMCDriverGTO(mf, omega, coupling_vec, dt=dt,
                           chol_cut=chol_cut, s=s, q0=q0, verbose=verbose,
                           chol_h5_path=chol_h5_path,
                           chol_chunk_g=chol_chunk_g)
