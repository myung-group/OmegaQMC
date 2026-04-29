"""
Phaseless AFQMC for the 3D Homogeneous Electron Gas (3DEG / Jellium).

Implements the plane-wave basis AFQMC for the 3D electron gas with
Coulomb interactions, following the same architecture as the 2DEG module
but adapted for 3D:

    H = Σ_k |k|²/2 n_k  +  (1/2V) Σ_{q≠0} v(q) ρ_q ρ_{-q}

where v(q) = 4π/|q|² (3D Coulomb).

Designed to reproduce the results of Ceperley & Alder, PRL 45, 566 (1980).
"""

import sys
import time

import numpy as np
import jax
import jax.numpy as jnp
from functools import partial

from OmegaQMC.utils import do_binning_analysis
from jax.scipy.linalg import expm

from OmegaQMC.afqmc_gto import (
    half_rotate_cholesky,
    _apply_exp_vhs,
    orthogonalize_walkers,
    greens_function,
    greens_function_force_bias,
    greens_function_overlap,
    local_energy_1body,
    _update_weights_phaseless,
    _make_afqmc_sharding,
    population_control_comb,
    Walkers,
    WEIGHT_CLIP_FRACTION,
    FBBOUND_DEFAULT,
)


# ===================================================================
# System setup
# ===================================================================

def _generate_3d_kgrid(n_pw):
    """Generate 3D k-grid as complete shells sorted by |k|².

    Shells are filled in order of increasing |n_x² + n_y² + n_z²|
    to produce the standard 3D shell counts:
    1, 7, 19, 27, 33, 57, 81, 93, 123, 147, ...

    Args:
        n_pw: Desired number of plane waves. Will be rounded up to the
            nearest complete shell.

    Returns:
        grid: integer grid points (N_pw, 3), sorted by |n|².
        actual_n_pw: actual number of PWs (complete shells).
    """
    nmax = int(np.ceil(n_pw**(1.0/3.0))) + 5
    rng = np.arange(-nmax, nmax + 1)
    nx, ny, nz = np.meshgrid(rng, rng, rng, indexing='ij')
    candidates = np.stack([nx.ravel(), ny.ravel(), nz.ravel()], axis=1)
    norms_sq = candidates[:, 0]**2 + candidates[:, 1]**2 + candidates[:, 2]**2

    order = np.argsort(norms_sq)
    candidates = candidates[order]
    norms_sq = norms_sq[order]

    unique_norms = np.unique(norms_sq)
    actual_n_pw = 0
    for norm_sq in unique_norms:
        count = np.sum(norms_sq == norm_sq)
        if actual_n_pw + count > n_pw and actual_n_pw > 0:
            break
        actual_n_pw += count

    grid = candidates[:actual_n_pw]
    return grid, actual_n_pw


def build_3deg_system(rs, N_elec, N_pw, polarization='unpolarized',
                      kappa=None):
    """Build the 3DEG system parameters.

    Args:
        rs: Wigner-Seitz radius (density parameter).
            Defined by (4π/3) r_s³ = V/N (volume per electron).
        N_elec: Number of electrons.
        N_pw: Number of plane waves (rounded up to complete shell).
        polarization: 'unpolarized' (nup=ndown=N/2) or
            'polarized' (nup=N, ndown=0).
        kappa: Twist angle (3,) in fractional coordinates [-0.5, 0.5).
            Shifts k-vectors: k = (n + kappa) * dk. Default None (Gamma).

    Returns:
        dict with system parameters including kvecs (N_pw, 3).
    """
    # Box length from density: V = N * (4π/3) r_s³, L = V^{1/3}
    volume = N_elec * (4.0 * np.pi / 3.0) * rs**3
    L = volume**(1.0 / 3.0)

    grid_int, actual_n_pw = _generate_3d_kgrid(N_pw)
    dk = 2.0 * np.pi / L
    if kappa is not None:
        kappa = np.asarray(kappa, dtype=np.float64)
        kvecs = (grid_int.astype(np.float64) + kappa[np.newaxis, :]) * dk
    else:
        kvecs = grid_int.astype(np.float64) * dk

    if polarization == 'unpolarized':
        assert N_elec % 2 == 0, "N_elec must be even for unpolarized"
        nup = N_elec // 2
        ndown = N_elec // 2
    elif polarization == 'polarized':
        nup = N_elec
        ndown = 0
    else:
        raise ValueError(f"Unknown polarization: {polarization}")

    return {
        'rs': float(rs),
        'N_elec': N_elec,
        'N_pw': actual_n_pw,
        'nup': nup,
        'ndown': ndown,
        'L': float(L),
        'volume': float(volume),
        'dk': float(dk),
        'kvecs': kvecs,
        'grid_int': grid_int,
        'kappa': kappa,
        'polarization': polarization,
    }


# ===================================================================
# Integral construction
# ===================================================================

def build_h1e_pw_3d(system):
    """Build one-body Hamiltonian in the plane-wave basis.

    Diagonal: h[k,k] = |k|²/2 (kinetic energy).
    Pure jellium has no off-diagonal terms.

    Args:
        system: dict from build_3deg_system.

    Returns:
        h1e: (N_pw, N_pw) real diagonal matrix.
    """
    kvecs = system['kvecs']
    N_pw = system['N_pw']

    return np.diag(0.5 * np.sum(kvecs**2, axis=1))


def build_cholesky_pw_3d(system):
    """Build symmetrized Cholesky vectors for the 3D Coulomb interaction.

    The raw density operator ρ̂(q) has matrix L^q[k,k'] = pf·δ_{k-k',q},
    which is an asymmetric shift matrix. The standard AFQMC machinery
    (HS decomposition, local energy) assumes symmetric Cholesky vectors.

    We symmetrize by pairing (q, -q):
        A^q = (L^q + L^{-q}) / √2   (symmetric,  sign = +1)
        B^q = (L^q - L^{-q}) / √2   (anti-symmetric, sign = -1)

    The ERI decomposes as:
        (pq|rs) = Σ_{q>0} [A^q_{pq} A^q_{rs} - B^q_{pq} B^q_{rs}]

    The sign array encodes the ± in the ERI decomposition and must be
    used in the local energy, trial energy, and propagation:
      - Local energy: e_exch = 0.5 Σ_g sign[g] Tr((L^g G)²)
      - Propagation:  VHS uses i√dt for sign=+1, √dt for sign=-1

    Args:
        system: dict from build_3deg_system.

    Returns:
        chol: (N_chol, N_pw, N_pw) array of symmetrized Cholesky vectors.
        q_vecs_out: (N_chol, 3) array of q-vectors (the q>0 vector for
            each pair, repeated for A and B).
        chol_sign: (N_chol,) array of +1 (A-type) or -1 (B-type).
    """
    N_pw = system['N_pw']
    volume = system['volume']
    grid_int = system['grid_int']
    dk = system['dk']

    # Build lookup: integer grid -> index using a hash map
    grid_to_idx = {}
    for i in range(N_pw):
        key = (int(grid_int[i, 0]), int(grid_int[i, 1]),
               int(grid_int[i, 2]))
        grid_to_idx[key] = i

    # q-vectors: all unique nonzero momentum transfers q = G_i - G_j.
    # This set is larger than the single-particle PW basis because
    # |G_i - G_j| can exceed the PW cutoff.
    diff_set = set()
    for i in range(N_pw):
        for j in range(N_pw):
            q = (int(grid_int[i, 0] - grid_int[j, 0]),
                 int(grid_int[i, 1] - grid_int[j, 1]),
                 int(grid_int[i, 2] - grid_int[j, 2]))
            if q != (0, 0, 0):
                diff_set.add(q)

    q_int_all = np.array(sorted(diff_set,
                                key=lambda t: t[0]**2 + t[1]**2 + t[2]**2))
    q_vecs_raw = q_int_all.astype(np.float64) * dk
    q_norm_sq = np.sum(q_vecs_raw**2, axis=1)
    prefactors = np.sqrt(4.0 * np.pi / (q_norm_sq * volume))

    N_q_raw = len(q_int_all)
    raw_chol = np.zeros((N_q_raw, N_pw, N_pw))
    for iq in range(N_q_raw):
        q_int = q_int_all[iq]
        targets = grid_int - q_int[np.newaxis, :]
        pf = prefactors[iq]
        for ik in range(N_pw):
            key = (int(targets[ik, 0]), int(targets[ik, 1]),
                   int(targets[ik, 2]))
            jk = grid_to_idx.get(key)
            if jk is not None:
                raw_chol[iq, ik, jk] = pf

    # Build map from integer q-tuple to raw index
    q_int_to_raw = {}
    for iq in range(N_q_raw):
        key = tuple(int(x) for x in q_int_all[iq])
        q_int_to_raw[key] = iq

    # Identify (q, -q) pairs: pick q > 0 under lexicographic ordering
    paired = set()
    chol_list = []
    sign_list = []
    qvec_list = []

    for iq in range(N_q_raw):
        if iq in paired:
            continue
        q_int = q_int_all[iq]
        neg_q = tuple(int(-x) for x in q_int)
        iq_neg = q_int_to_raw.get(neg_q)

        if iq_neg is not None and iq_neg != iq:
            # Paired: build A and B
            Lq = raw_chol[iq]
            Lnq = raw_chol[iq_neg]
            Aq = (Lq + Lnq) / np.sqrt(2.0)
            Bq = (Lq - Lnq) / np.sqrt(2.0)
            qv = q_vecs_raw[iq]

            chol_list.append(Aq)
            sign_list.append(+1.0)
            qvec_list.append(qv)

            chol_list.append(Bq)
            sign_list.append(-1.0)
            qvec_list.append(qv)

            paired.add(iq)
            paired.add(iq_neg)
        else:
            # Unpaired (q = -q, i.e. 2q = reciprocal lattice vector).
            # Treat as A-type (L^q is already symmetric when q = -q).
            chol_list.append(raw_chol[iq])
            sign_list.append(+1.0)
            qvec_list.append(q_vecs_raw[iq])
            paired.add(iq)

    chol = np.array(chol_list)
    chol_sign = np.array(sign_list)
    q_vecs_out = np.array(qvec_list) if len(qvec_list) > 0 else np.zeros(
        (0, 3))

    return chol, q_vecs_out, chol_sign


def compute_madelung_3d(system):
    """Compute the 3D Madelung constant for the jellium background.

    Uses 3D Ewald summation for the self-energy of a point charge in
    its own neutralizing background on a simple cubic lattice.

    The Madelung energy per electron is:
        e_M = (1/2) [Σ'_R erfc(ηR)/R + (4π/V) Σ'_G exp(-G²/4η²)/G²
               - 2η/√π - π/(Vη²)]

    Args:
        system: dict from build_3deg_system.

    Returns:
        e_madelung: Madelung energy per electron.
    """
    L_box = system['L']
    volume = system['volume']

    # Ewald parameter
    eta = np.sqrt(np.pi) / L_box

    from scipy.special import erfc

    nmax = 10

    # Build grid of lattice vectors (exclude origin)
    rng = np.arange(-nmax, nmax + 1)
    nx, ny, nz = np.meshgrid(rng, rng, rng, indexing='ij')
    nx, ny, nz = nx.ravel(), ny.ravel(), nz.ravel()
    mask = (nx != 0) | (ny != 0) | (nz != 0)
    n_sq = nx[mask]**2 + ny[mask]**2 + nz[mask]**2

    # Real-space sum
    r = L_box * np.sqrt(n_sq.astype(np.float64))
    e_real = np.sum(erfc(eta * r) / r)

    # Reciprocal-space sum
    dk = 2.0 * np.pi / L_box
    g_sq = dk**2 * n_sq.astype(np.float64)
    e_recip = np.sum(
        (4.0 * np.pi / volume) * np.exp(-g_sq / (4.0 * eta**2)) / g_sq)

    # Self-interaction correction
    e_self = -2.0 * eta / np.sqrt(np.pi)

    # Background correction
    e_bg = -np.pi / (volume * eta**2)

    e_madelung = 0.5 * (e_real + e_recip + e_self + e_bg)

    return e_madelung


def prepare_3deg_integrals(system):
    """Prepare all integrals for the 3DEG AFQMC calculation.

    Args:
        system: dict from build_3deg_system.

    Returns:
        dict with h1e, h1e_mod, chol, chol_sign, e_madelung, N_q.
    """
    h1e = build_h1e_pw_3d(system)
    chol, q_vecs, chol_sign = build_cholesky_pw_3d(system)
    N_q = chol.shape[0]

    # Modified one-body: h1e_mod = h1e - 0.5 * Σ_g sign_g L^g (L^g)^T
    # The ERI is (ij|kl) = Σ_g sign_g L^g_ij L^g_kl, so
    # v0_ik = -0.5 * Σ_g sign_g Σ_j L^g_ij L^g_kj.
    if N_q > 0:
        v0 = np.einsum('g,gij,gkj->ik', chol_sign, chol, chol) * (-0.5)
        h1e_mod = h1e + v0
    else:
        h1e_mod = h1e.copy()

    e_madelung = compute_madelung_3d(system)

    return {
        'h1e': jnp.array(h1e),
        'h1e_mod': jnp.array(h1e_mod),
        'chol': jnp.array(chol),
        'chol_sign': jnp.array(chol_sign),
        'q_vecs': q_vecs,
        'e_madelung': float(e_madelung),
        'N_q': N_q,
    }


# ===================================================================
# Trial wavefunction
# ===================================================================

def build_trial_3deg(h1e, nup, ndown):
    """Build trial wavefunction for the 3DEG.

    Occupies the lowest |k|² states (free-electron Fermi sea).

    Args:
        h1e: One-body Hamiltonian (N_pw, N_pw).
        nup: Number of spin-up electrons.
        ndown: Number of spin-down electrons.

    Returns:
        trial_up: (N_pw, nup) array.
        trial_dn: (N_pw, ndown) array.
    """
    h1e_np = np.asarray(h1e)
    energies = np.diag(h1e_np)
    order = np.argsort(energies)

    N_pw = h1e_np.shape[0]
    trial_up = np.zeros((N_pw, nup))
    trial_up[order[:nup], np.arange(nup)] = 1.0

    if ndown > 0:
        trial_dn = np.zeros((N_pw, ndown))
        trial_dn[order[:ndown], np.arange(ndown)] = 1.0
    else:
        # Fully polarized: dummy 1-column trial for beta
        trial_dn = np.zeros((N_pw, 0))

    return jnp.array(trial_up), jnp.array(trial_dn)


# ===================================================================
# PW-basis local energy (sign-weighted for symmetrized Cholesky)
# ===================================================================

def local_energy_2body_pw(Ghalfa, Ghalfb, rchol_a, rchol_b, chol_sign):
    """Two-body local energy for the plane-wave basis.

    For symmetrized Cholesky vectors (A^q symmetric, B^q anti-symmetric),
    the ERI decomposition includes a sign:
        (pq|rs) = Σ_g sign[g] L^g_{pq} L^g_{rs}

    This requires sign-weighted Coulomb and exchange contributions.

    Args:
        Ghalfa: shape (nwalkers, nup, nbasis).
        Ghalfb: shape (nwalkers, ndown, nbasis).
        rchol_a: shape (naux, nup, nbasis).
        rchol_b: shape (naux, ndown, nbasis).
        chol_sign: shape (naux,), +1 for A-type, -1 for B-type.

    Returns:
        e_coul: Coulomb energy, shape (nwalkers,).
        e_exch: Exchange energy, shape (nwalkers,).
    """
    # Coulomb: X_σ^γ = Tr(rchol_σ^γ · Ghalf_σ^T)
    Xa = jnp.einsum('giq,wiq->gw', rchol_a, Ghalfa)
    Xb = jnp.einsum('giq,wiq->gw', rchol_b, Ghalfb)

    # Sign-weighted Coulomb: 0.5 Σ_g sign[g] (Xa+Xb)^2
    Xtot_sq = (Xa + Xb) ** 2  # (naux, nwalkers)
    e_coul = 0.5 * jnp.einsum('g,gw->w', chol_sign, Xtot_sq)

    # Exchange: T_σ^γ[i,j] = Σ_q rchol_σ[γ,i,q] · Ghalf_σ[w,j,q]
    Ta = jnp.einsum('giq,wjq->gwij', rchol_a, Ghalfa)
    Tb = jnp.einsum('giq,wjq->gwij', rchol_b, Ghalfb)

    # Sign-weighted exchange: 0.5 Σ_g sign[g] Tr(T^g (T^g)^T)
    # Tr(T T^T) via einsum: Σ_{ij} T[i,j] T[j,i]
    exch_a_per_g = jnp.sum(
        Ta * Ta.transpose(0, 1, 3, 2), axis=(2, 3))  # (naux, nw)
    exch_b_per_g = jnp.sum(
        Tb * Tb.transpose(0, 1, 3, 2), axis=(2, 3))  # (naux, nw)

    e_exch = 0.5 * jnp.einsum(
        'g,gw->w', chol_sign, exch_a_per_g + exch_b_per_g)

    return e_coul, e_exch


@partial(jax.jit, static_argnames=[])
def local_energy_pw(h1e, chol, Ga, Gb, Ghalfa, Ghalfb,
                    rchol_a, rchol_b, enuc, chol_sign):
    """Local energy for the PW basis with symmetrized Cholesky.

    Uses sign-weighted two-body contributions for the correct ERI.
    """
    e_1b = local_energy_1body(h1e, Ga, Gb, enuc)
    e_coul, e_exch = local_energy_2body_pw(
        Ghalfa, Ghalfb, rchol_a, rchol_b, chol_sign)
    e_2b = e_coul - e_exch
    e_tot = e_1b + e_2b
    return e_tot, e_1b, e_2b


# ===================================================================
# Driver
# ===================================================================

class _AFQMC3DEGDriver:
    """Phaseless AFQMC driver for the 3D electron gas.

    Implements the standard phaseless AFQMC for jellium in a
    plane-wave basis with 3D Coulomb interactions.
    """

    def __init__(self, system, dt=0.005, include_coulomb=True,
                 verbose=True):
        """Prepare integrals and build the propagator.

        Args:
            system: dict from build_3deg_system.
            dt: Imaginary time step.
            include_coulomb: Whether to include e-e Coulomb interaction.
            verbose: Print progress.
        """
        self.system = system
        self.dt = dt
        self.include_coulomb = include_coulomb
        self.verbose = verbose

        if verbose:
            print("Preparing 3DEG AFQMC integrals...")
            t0 = time.time()

        integrals = prepare_3deg_integrals(system)
        self.h1e = integrals['h1e']
        self.h1e_mod = integrals['h1e_mod']
        self.chol = integrals['chol']
        self.chol_sign = integrals['chol_sign']
        # Phase for HS coupling: i for A-type (repulsive), 1 for B-type
        self.chol_phase = jnp.where(
            self.chol_sign > 0, 1j * jnp.ones(len(self.chol_sign)),
            jnp.ones(len(self.chol_sign)))
        self.e_madelung = integrals['e_madelung']
        self.N_q = integrals['N_q']

        N_pw = system['N_pw']
        nup = system['nup']
        ndown = system['ndown']
        self.N_pw = N_pw
        self.nup = nup
        self.ndown = ndown

        # Background energy
        N_elec = system['N_elec']
        self.enuc = self.e_madelung * N_elec if include_coulomb else 0.0

        if verbose:
            print(f"  N_pw={N_pw}, nup={nup}, ndown={ndown}, "
                  f"N_q={self.N_q}")
            print(f"  rs={system['rs']:.2f}, L={system['L']:.4f}, "
                  f"V={system['volume']:.4f}")
            print(f"  polarization={system['polarization']}")
            print(f"  Integral preparation took {time.time()-t0:.2f} s")

        # Trial wavefunction
        self.trial_up, self.trial_dn = build_trial_3deg(
            self.h1e, nup, ndown)

        # For fully polarized: use a dummy 1-column beta trial
        # Must be nonzero so overlap det is nonsingular
        if ndown == 0:
            dummy = np.zeros((N_pw, 1))
            dummy[0, 0] = 1.0
            self._trial_dn_eff = jnp.array(dummy)
        else:
            self._trial_dn_eff = self.trial_dn

        # Half-rotate Cholesky vectors
        if include_coulomb and self.N_q > 0:
            self.rchol_a, self.rchol_b = half_rotate_cholesky(
                self.chol, self.trial_up, self._trial_dn_eff)
            # Fully polarized: zero out beta Cholesky to eliminate
            # spurious Coulomb/exchange from the dummy beta electron
            if ndown == 0:
                self.rchol_b = jnp.zeros_like(self.rchol_b)
        else:
            ndn_eff = max(ndown, 1)
            N_q = self.N_q
            self.rchol_a = jnp.zeros((N_q, nup, N_pw))
            self.rchol_b = jnp.zeros((N_q, ndn_eff, N_pw))

        # Build propagator with sign-aware mean-field shift
        # For fully polarized: use zero beta trial so the mean-field
        # shift and one-body propagator only reflect alpha electrons
        trial_dn_for_prop = (jnp.zeros((N_pw, 1)) if ndown == 0
                             else self._trial_dn_eff)
        G_trial_a = self.trial_up @ self.trial_up.T.conj()
        G_trial_b = trial_dn_for_prop @ trial_dn_for_prop.T.conj()
        G_charge = G_trial_a + G_trial_b

        if include_coulomb and self.N_q > 0:
            # mf_shift_g = phase_g * Tr(L^g G) where phase = i (A) or 1 (B)
            mf_shift = self.chol_phase * jnp.einsum(
                'gpq,qp->g', self.chol, G_charge)
            # vhs_mf = Σ_g phase_g * mf_shift_g * L^g
            #        = Σ_g phase_g² * Tr(L^g G) * L^g
            vhs_mf = jnp.einsum(
                'g,g,gpq->pq', self.chol_phase, mf_shift, self.chol)
            H1_shifted = self.h1e_mod - vhs_mf
            expH1 = expm(-0.5 * dt * H1_shifted)
            self.propagator = {
                'expH1': expH1, 'mf_shift': mf_shift, 'dt': dt}
        else:
            # No Coulomb: trivial propagator
            expH1 = expm(-0.5 * dt * self.h1e)
            mf_shift = jnp.zeros(0)
            self.propagator = {
                'expH1': expH1, 'mf_shift': mf_shift, 'dt': dt}

        # Compute trial energy
        e_trial = self._compute_trial_energy()
        self.e_trial = e_trial

        if verbose:
            print(f"  E_trial = {e_trial:.10f}")
            print(f"  E_madelung/elec = {self.e_madelung:.10f}")

    def _compute_trial_energy(self):
        """Compute the trial wavefunction energy (HF for jellium)."""
        h1e_np = np.asarray(self.h1e)
        trial_up_np = np.asarray(self.trial_up)
        G_a = trial_up_np @ trial_up_np.T
        e_1b = np.trace(h1e_np @ G_a)

        if self.ndown > 0:
            trial_dn_np = np.asarray(self.trial_dn)
            G_b = trial_dn_np @ trial_dn_np.T
            e_1b += np.trace(h1e_np @ G_b)

        # For HF jellium, exchange energy from Cholesky
        if self.include_coulomb and self.N_q > 0:
            chol_np = np.asarray(self.chol)
            chol_sign = np.asarray(self.chol_sign)
            # Exchange: -0.5 Σ_g sign[g] Tr((L^g G)^2)
            # With symmetrized Cholesky (A^q symmetric, B^q anti-symmetric),
            # the sign-weighted Tr((LG)^2) gives the correct exchange.
            e_exch = 0.0
            for sigma_G in ([G_a] if self.ndown == 0
                            else [G_a, G_b]):
                T = np.einsum('gij,jk->gik', chol_np, sigma_G)
                e_exch += 0.5 * np.einsum(
                    'g,gij,gji->', chol_sign, T, T)
            return float(e_1b) - e_exch + self.enuc
        return float(e_1b) + self.enuc

    def __call__(self, rng_key=None, num_walkers=100, num_blocks=100,
                 num_steps_per_block=25, stabilize_freq=5,
                 pop_control_freq=5, num_eqlb_blocks=10,
                 fname_log="afqmc_3deg.log"):
        """Run the AFQMC simulation.

        Args:
            rng_key: JAX random key. If None, uses key(42).
            num_walkers: Number of walkers.
            num_blocks: Number of measurement blocks.
            num_steps_per_block: MC steps per block.
            stabilize_freq: QR reorthogonalization frequency.
            pop_control_freq: Population control frequency.
            num_eqlb_blocks: Equilibration blocks.
            fname_log: Log file path. If None or "", prints to stdout.

        Returns:
            dict with energy statistics.
        """
        if rng_key is None:
            rng_key = jax.random.key(42)

        if fname_log is None \
                or (isinstance(fname_log, str) and fname_log == ""):
            fout = sys.stdout
        else:
            fout = open(fname_log, 'w', 1)

        verbose = self.verbose
        has_coulomb = self.include_coulomb and self.N_q > 0
        ndown_eff = max(self.ndown, 1)  # at least 1 for array shapes

        if verbose:
            print(f"  num_walkers = {num_walkers}", file=fout)

        # Initialize walkers
        ndn_eff = max(self.ndown, 1)
        walkers = Walkers(self.trial_up, self._trial_dn_eff,
                          num_walkers, self.N_pw,
                          self.nup, ndn_eff)
        phia = walkers.phia
        phib = walkers.phib

        weights = jnp.ones(num_walkers)
        overlap = jnp.ones(num_walkers, dtype=jnp.complex128)
        e_hybrid = jnp.zeros(num_walkers, dtype=jnp.complex128)

        phi_sharding, scalar_sharding = _make_afqmc_sharding(num_walkers)
        if phi_sharding is not None:
            phia = jax.device_put(phia, phi_sharding)
            phib = jax.device_put(phib, phi_sharding)
            weights = jax.device_put(weights, scalar_sharding)
            overlap = jax.device_put(overlap, scalar_sharding)
            e_hybrid = jax.device_put(e_hybrid, scalar_sharding)

        trial_dn_eff = self._trial_dn_eff
        rchol_b_eff = self.rchol_b

        # Main QMC loop
        total_blocks = num_eqlb_blocks + num_blocks
        energy_blocks = []
        eshift = 0.0
        step_count = 0

        if verbose:
            print(f"\nStarting 3DEG AFQMC: {num_eqlb_blocks} eqlb + "
                  f"{num_blocks} prod blocks", file=fout)
            print(f"  {num_steps_per_block} steps/block, "
                  f"stabilize every {stabilize_freq} steps", file=fout)
            coul_str = "ON" if has_coulomb else "OFF"
            print(f"  Coulomb: {coul_str}", file=fout)
            print("-" * 70, file=fout)
            print(f"{'Block':>6} {'E_total':>16} {'E_shift':>16} "
                  f"{'W_sum':>12} {'W_max':>12}", file=fout)
            print("-" * 70, file=fout)

        for iblock in range(total_blocks):
            acc_weight = 0.0
            acc_ehybrid = 0.0

            for istep in range(num_steps_per_block):
                step_count += 1

                # QR reorthogonalization
                if step_count % stabilize_freq == 0:
                    phia, phib, log_detR, _ = orthogonalize_walkers(
                        phia, phib)
                    overlap = overlap / jnp.exp(log_detR)

                # Propagate
                rng_key, step_key = jax.random.split(rng_key)
                phia, phib, weights, overlap, e_hybrid, _ = \
                    _propagate_3deg_walkers(
                        phia, phib, weights, overlap, e_hybrid,
                        self.propagator, self.chol,
                        self.rchol_a, rchol_b_eff,
                        self.trial_up, trial_dn_eff,
                        eshift, step_key,
                        has_coulomb, ndown=self.ndown,
                        chol_phase=self.chol_phase)

                # Clip weights
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

                # Accumulate for eshift
                w_step = jnp.sum(jnp.abs(weights))
                acc_weight += float(w_step)
                acc_ehybrid += float(jnp.sum(
                    jnp.abs(weights) * e_hybrid.real))

            # End of block: compute energy
            Ga, Gb, Ghalfa, Ghalfb, _ = greens_function(
                phia, phib, self.trial_up, trial_dn_eff)

            # Fully polarized: zero out beta Green's function so the
            # dummy beta electron doesn't contribute to the energy
            if self.ndown == 0:
                Gb = jnp.zeros_like(Gb)

            e_tot, e_1b, e_2b = local_energy_pw(
                self.h1e, self.chol, Ga, Gb, Ghalfa, Ghalfb,
                self.rchol_a, rchol_b_eff, self.enuc, self.chol_sign)

            w = jnp.abs(weights)
            w_sum = jnp.sum(w)
            e_block = float(jnp.sum(w * e_tot.real) / w_sum)
            energy_blocks.append(e_block)

            is_eqlb = iblock < num_eqlb_blocks

            if acc_weight > 1e-10:
                eshift = acc_ehybrid / acc_weight

            if verbose:
                phase = "EQ" if is_eqlb else "  "
                print(f"{phase}{iblock:4d} {e_block:16.10f} "
                      f"{float(eshift):16.10f} "
                      f"{float(w_sum):12.4f} "
                      f"{float(jnp.max(weights)):12.4f}", file=fout)

        # Analyze results
        energy_blocks = np.array(energy_blocks)
        prod_energies = energy_blocks[num_eqlb_blocks:]

        e_mean, e_err, e_std, kappa = do_binning_analysis(
            jnp.array(prod_energies))

        N_elec = self.system['N_elec']
        e_per_elec_ha = float(e_mean) / N_elec
        e_per_elec_ry = e_per_elec_ha * 2.0  # 1 Ha = 2 Ry

        if verbose:
            print("-" * 70, file=fout)
            print(f"E_trial       = {self.e_trial:.10f} Ha", file=fout)
            print(f"E_AFQMC       = {float(e_mean):.10f} +/- "
                  f"{float(e_err):.10f} Ha", file=fout)
            print(f"E/N           = {e_per_elec_ha:.10f} Ha/elec",
                  file=fout)
            print(f"E/N           = {e_per_elec_ry:.10f} Ry/elec",
                  file=fout)
            print(f"kappa         = {float(kappa):.2f}", file=fout)

        if fout is not sys.stdout:
            fout.close()

        return {
            'energy_blocks': energy_blocks,
            'energy_mean': float(e_mean),
            'energy_err': float(e_err),
            'energy_std': float(e_std),
            'kappa': float(kappa),
            'e_trial': self.e_trial,
            'e_per_elec_ha': e_per_elec_ha,
            'e_per_elec_ry': e_per_elec_ry,
        }


def _propagate_3deg_walkers(phia, phib, weights, overlap, e_hybrid,
                             propagator, chol, rchol_a, rchol_b,
                             trial_up, trial_dn, eshift, rng_key,
                             has_coulomb, ndown=None,
                             fbbound=None, exp_nmax=6,
                             chol_phase=None):
    """Propagate 3DEG AFQMC walkers by one time step.

    Symmetric Trotter decomposition:
        e^{-dt H} ≈ e^{-dt/2 H_1} · e^{-dt H_2} · e^{-dt/2 H_1}

    For symmetrized Cholesky, the HS coupling differs by vector type:
        A-type (symmetric):      VHS_A = i√dt x_A A^q   (sign=+1)
        B-type (anti-symmetric): VHS_B =  √dt x_B B^q   (sign=-1)
    This is encoded in chol_phase[g] = 1j or 1.0.

    Args:
        phia, phib: Walker orbitals.
        weights, overlap, e_hybrid: Walker metadata.
        propagator: dict from build_propagator.
        chol: Cholesky vectors.
        rchol_a, rchol_b: Half-rotated Cholesky.
        trial_up, trial_dn: Trial orbitals.
        eshift: Energy shift.
        rng_key: JAX random key.
        has_coulomb: Whether to include two-body.
        ndown: Number of physical down electrons. When 0, beta
            propagation and overlap are skipped (fully polarized).
        fbbound: Force bias bound.
        exp_nmax: Taylor order for VHS exponential.
        chol_phase: (naux,) complex array with 1j for A-type, 1.0 for
            B-type Cholesky vectors.  If None, uses 1j for all (GTO
            convention).

    Returns:
        phia, phib, weights, overlap, e_hybrid, rng_key.
    """
    dt = propagator['dt']
    expH1 = propagator['expH1']
    mf_shift = propagator['mf_shift']
    nwalkers = phia.shape[0]
    naux = chol.shape[0]
    skip_beta = (ndown is not None and ndown == 0)

    if fbbound is None:
        fbbound = FBBOUND_DEFAULT

    # Green's function for force bias
    # (specialized: only Ghalf + overlap needed)
    Ghalfa, Ghalfb, ovlp = greens_function_force_bias(
        phia, phib, trial_up, trial_dn)

    # First half one-body (skip beta for fully polarized)
    phia = jnp.einsum('pq,wqn->wpn', expH1, phia)
    if not skip_beta:
        phib = jnp.einsum('pq,wqn->wpn', expH1, phib)

    # Two-body propagation
    if has_coulomb and naux > 0:
        vbias_a = jnp.einsum('giq,wiq->gw', rchol_a, Ghalfa)
        if not skip_beta:
            vbias_b = jnp.einsum('giq,wiq->gw', rchol_b, Ghalfb)
            vbias = vbias_a + vbias_b
        else:
            vbias = vbias_a

        if chol_phase is not None:
            xbar = -jnp.sqrt(dt) * (
                chol_phase[:, None] * vbias - mf_shift[:, None])
        else:
            xbar = -jnp.sqrt(dt) * (1j * vbias - mf_shift[:, None])
        xbar = xbar.T

        xbar_abs = jnp.abs(xbar)
        xbar = jnp.where(xbar_abs > fbbound,
                         xbar * fbbound / xbar_abs, xbar)

        rng_key, subkey = jax.random.split(rng_key)
        xi = jax.random.normal(subkey, shape=(nwalkers, naux))
        xshifted = xi - xbar

        cmf = -jnp.sqrt(dt) * jnp.einsum('wg,g->w', xshifted, mf_shift)
        cfb = (jnp.sum(xi * xbar, axis=1)
               - 0.5 * jnp.sum(xbar * xbar, axis=1))

        # VHS with per-vector phase: i√dt for A-type, √dt for B-type
        if chol_phase is not None:
            VHS = jnp.sqrt(dt) * jnp.einsum(
                'wg,g,gpq->wpq', xshifted, chol_phase, chol)
        else:
            VHS = 1j * jnp.sqrt(dt) * jnp.einsum(
                'wg,gpq->wpq', xshifted, chol)
        phia = _apply_exp_vhs(VHS, phia, exp_nmax)
        if not skip_beta:
            phib = _apply_exp_vhs(VHS, phib, exp_nmax)
    else:
        cmf = jnp.zeros(nwalkers)
        cfb = jnp.zeros(nwalkers)

    # Second half one-body
    phia = jnp.einsum('pq,wqn->wpn', expH1, phia)
    if not skip_beta:
        phib = jnp.einsum('pq,wqn->wpn', expH1, phib)

    # Weight update
    # When skip_beta: phib is unchanged so ovlp_b_new/ovlp_b_old = 1,
    # and the combined overlap ratio correctly reflects only alpha.
    ovlp_new = greens_function_overlap(
        phia, phib, trial_up, trial_dn)
    weights_new, e_hybrid_new = _update_weights_phaseless(
        weights, ovlp, ovlp_new, cfb, cmf, e_hybrid, eshift, dt)

    return phia, phib, weights_new, ovlp_new, e_hybrid_new, rng_key


def get_afqmc_3deg_func(system, dt=0.005, include_coulomb=True,
                         verbose=True):
    """Create a reusable AFQMC driver for the 3DEG.

    Args:
        system: dict from build_3deg_system.
        dt: Imaginary time step.
        include_coulomb: Whether to include e-e Coulomb interaction.
        verbose: Print progress.

    Returns:
        _AFQMC3DEGDriver instance (callable).
    """
    return _AFQMC3DEGDriver(system, dt=dt,
                             include_coulomb=include_coulomb,
                             verbose=verbose)


# ===================================================================
# Twist-averaged boundary conditions
# ===================================================================

def generate_halton_twists_3d(n_twists):
    """Generate quasi-random 3D twist angles from a Halton sequence.

    Args:
        n_twists: Number of twist angles to generate.

    Returns:
        twists: (n_twists, 3) array of twist angles in [-0.5, 0.5).
    """
    from scipy.stats.qmc import Halton
    sampler = Halton(d=3, scramble=False)
    return sampler.random(n=n_twists) - 0.5


def _build_twisted_system_3d(base_system, kappa):
    """Create a twisted copy of a 3DEG system dict.

    Args:
        base_system: dict from build_3deg_system.
        kappa: (3,) twist angle in fractional coordinates.

    Returns:
        dict: system dict with shifted kvecs and updated kappa.
    """
    kappa = np.asarray(kappa, dtype=np.float64)
    system_tw = dict(base_system)
    grid_int = base_system['grid_int']
    dk = base_system['dk']
    system_tw['kvecs'] = (grid_int.astype(np.float64)
                          + kappa[np.newaxis, :]) * dk
    system_tw['kappa'] = kappa
    return system_tw


def run_twist_averaged_afqmc_3deg(
        base_system, n_twists=20, dt=0.005,
        num_walkers=100, num_blocks=50, num_steps_per_block=25,
        num_eqlb_blocks=10, verbose=True):
    """Run twist-averaged AFQMC for the 3DEG.

    For each twist angle: builds a twisted system, runs AFQMC, and
    collects energies. Averages over all twists.

    Args:
        base_system: dict from build_3deg_system at Gamma (kappa=None).
        n_twists: Number of Halton twist angles.
        dt: Imaginary time step.
        num_walkers: Number of walkers per twist.
        num_blocks: Production blocks per twist.
        num_steps_per_block: Steps per block.
        num_eqlb_blocks: Equilibration blocks per twist.
        verbose: Print per-twist progress.

    Returns:
        dict with twist-averaged results.
    """
    twists = generate_halton_twists_3d(n_twists)

    e_arr = np.zeros(n_twists)
    e_err_arr = np.zeros(n_twists)
    e_trial_arr = np.zeros(n_twists)

    for i, kappa in enumerate(twists):
        system_tw = _build_twisted_system_3d(base_system, kappa)

        if verbose:
            print(f"Twist {i}/{n_twists}  "
                  f"kappa=({kappa[0]:.4f}, {kappa[1]:.4f}, "
                  f"{kappa[2]:.4f})")

        driver = get_afqmc_3deg_func(
            system_tw, dt=dt, include_coulomb=True, verbose=False)

        result = driver(
            num_walkers=num_walkers,
            num_blocks=num_blocks,
            num_steps_per_block=num_steps_per_block,
            num_eqlb_blocks=num_eqlb_blocks,
            fname_log=None)

        e_arr[i] = result['energy_mean']
        e_err_arr[i] = result['energy_err']
        e_trial_arr[i] = driver.e_trial

        if verbose:
            str_report = f"E={e_arr[i]:.8f} +/- {e_err_arr[i]:.8f} "
            str_report += f"E_HF={e_trial_arr[i]:.8f} "
            str_report += f"E_corr={e_arr[i]-e_trial_arr[i]:.8f}"
            print(str_report)

    N_elec = base_system['N_elec']
    e_mean = float(np.mean(e_arr))
    e_err = float(np.sqrt(np.sum(e_err_arr**2)) / n_twists)
    e_trial_mean = float(np.mean(e_trial_arr))

    e_per_elec_ha = e_mean / N_elec
    e_per_elec_ry = e_per_elec_ha * 2.0
    e_trial_per_elec_ry = e_trial_mean / N_elec * 2.0

    if verbose:
        print(f"\nTwist-averaged E = {e_mean:.8f} +/- {e_err:.8f} Ha")
        print(f"E/N = {e_per_elec_ha:.8f} Ha/elec")
        print(f"E/N = {e_per_elec_ry:.8f} Ry/elec")
        print(f"E_trial/N (twist-avg) = {e_trial_per_elec_ry:.8f} Ry/elec")

    return {
        'twists': twists,
        'e_twists': e_arr,
        'e_err_twists': e_err_arr,
        'e_trial_twists': e_trial_arr,
        'energy_mean': e_mean,
        'energy_err': e_err,
        'e_trial_mean': e_trial_mean,
        'e_per_elec_ha': e_per_elec_ha,
        'e_per_elec_ry': e_per_elec_ry,
        'e_trial_per_elec_ry': e_trial_per_elec_ry,
    }


# ===================================================================
# Ceperley-Alder reference values
# ===================================================================

# Correlation energies from Ceperley & Alder, PRL 45, 566 (1980)
# in Ry per electron, for the paramagnetic (unpolarized) state.
# Total energy = E_kinetic + E_exchange + E_correlation.

def hf_kinetic_energy_per_elec_ry(rs, polarization='unpolarized'):
    """Hartree-Fock kinetic energy per electron in Ry.

    E_kin/N = (3/5) k_F² (in atomic units, per electron)
    k_F = (9π/4)^{1/3} / r_s  (unpolarized)
    k_F = (9π/2)^{1/3} / r_s  (polarized)

    Returns energy in Ry.
    """
    if polarization == 'unpolarized':
        kf = (9.0 * np.pi / 4.0)**(1.0/3.0) / rs
    else:
        kf = (9.0 * np.pi / 2.0)**(1.0/3.0) / rs
    # E_kin/N = (3/5) kF² / 2 in Hartree → multiply by 2 for Ry
    return 2.0 * 0.6 * kf**2 / 2.0


def hf_exchange_energy_per_elec_ry(rs, polarization='unpolarized'):
    """Hartree-Fock exchange energy per electron in Ry.

    E_x/N = -3 k_F / (4π) (in Hartree, per electron)

    Returns energy in Ry.
    """
    if polarization == 'unpolarized':
        kf = (9.0 * np.pi / 4.0)**(1.0/3.0) / rs
    else:
        kf = (9.0 * np.pi / 2.0)**(1.0/3.0) / rs
    # Factor depends on polarization
    if polarization == 'unpolarized':
        # Each spin channel contributes -3kF/(4π) × (N_σ/N)
        # with kF_σ = (6π² n_σ)^{1/3} but for unpolarized:
        # E_x/N = -3/(2π) × (9π/4)^{1/3} / r_s
        ex_ha = -3.0 * kf / (4.0 * np.pi) * 2  # factor 2 for both spins
        # Actually: E_x/N = -3/(4π) × kF for each spin channel
        # For unpol: 2 × (-3/(4π) kF_σ) (N_σ/N) = -3/(4π) kF
        ex_ha = -3.0 / (4.0 * np.pi) * kf
    else:
        ex_ha = -3.0 / (4.0 * np.pi) * kf
    return 2.0 * ex_ha


def pz_correlation_energy(rs, polarization='unpolarized'):
    """Perdew-Zunger parametrization of Ceperley-Alder correlation energies.

    From Perdew & Zunger, PRB 23, 5048 (1981), Table I.
    Parametrizes the QMC correlation energies of Ceperley & Alder
    (PRL 45, 566, 1980) as analytic functions of r_s.

    Args:
        rs: Wigner-Seitz radius.
        polarization: 'unpolarized' (paramagnetic) or
            'polarized' (ferromagnetic).

    Returns:
        Correlation energy per electron in Hartree.
    """
    if polarization == 'unpolarized':
        # Paramagnetic state
        if rs >= 1.0:
            gamma, beta1, beta2 = -0.1423, 1.0529, 0.3334
            return gamma / (1.0 + beta1 * np.sqrt(rs) + beta2 * rs)
        else:
            A, B, C, D = 0.0311, -0.048, 0.0020, -0.0116
            return A * np.log(rs) + B + C * rs * np.log(rs) + D * rs
    else:
        # Ferromagnetic state
        if rs >= 1.0:
            gamma, beta1, beta2 = -0.0843, 1.3981, 0.2611
            return gamma / (1.0 + beta1 * np.sqrt(rs) + beta2 * rs)
        else:
            A, B, C, D = 0.01555, -0.0269, 0.0007, -0.0048
            return A * np.log(rs) + B + C * rs * np.log(rs) + D * rs
