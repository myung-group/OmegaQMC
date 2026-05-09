"""Pauli-Fierz local-energy primitives for QED-NN-VMC.

Implements the local-energy estimator for the dipole-gauge Pauli-Fierz
Hamiltonian acting on a joint electron-photon trial wavefunction
:math:`\\Psi(\\mathbf{r}, n)` where :math:`\\mathbf{r}` are 3N electronic
coordinates and :math:`n \\in \\{0, 1, ..., N_{\\text{ph,max}}\\}` is a
photon Fock index.

The Hamiltonian (single-mode cavity, atomic units, electronic dipole):

.. math::
    H = H_e + \\omega\\, b^\\dagger b
        + \\sqrt{\\tfrac{\\omega}{2}}\\,\\lambda\\,
          (\\boldsymbol\\varepsilon\\cdot\\hat{\\mathbf d}_e)\\,(b + b^\\dagger)
        + \\tfrac{1}{2}\\lambda^2\\,
          (\\boldsymbol\\varepsilon\\cdot\\hat{\\mathbf d}_e)^2

with electronic dipole :math:`\\hat{\\mathbf d}_e = -\\sum_i \\mathbf r_i`
(matches the convention in ``OmegaQMC.qed_fci`` where the electronic
position operator alone is used; nuclear contributions are absorbed into
the cavity-vacuum reference for neutral systems).

The bilinear coupling is evaluated via Fock-ladder ratios:

.. math::
    \\frac{(b+b^\\dagger)\\Psi}{\\Psi}\\bigg|_{(\\mathbf r,n)}
    = \\sqrt{n+1}\\,\\frac{\\Psi(\\mathbf r, n+1)}{\\Psi(\\mathbf r, n)}
    + \\sqrt{n}\\,\\frac{\\Psi(\\mathbf r, n-1)}{\\Psi(\\mathbf r, n)} .

The dipole self-energy (DSE) is multiplicative for fixed
:math:`(\\mathbf r, n)`.

References
----------
- Tang et al. 2025, arXiv:2503.15644 (deep-VMC for polaritonic chemistry)
- Haugland et al. 2021, arXiv:2012.01080 (intermolecular cavity QED)
- ``OmegaQMC.qed_fci`` (in-house reference QED-FCI implementation)
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

from .physics import laplacian


__all__ = [
    "real_space_dipole_electronic",
    "real_space_dipole_total",
    "coulomb_ee",
    "coulomb_en",
    "coulomb_nn",
    "coherent_state_log_amplitude",
    "pauli_fierz_local_energy",
]


# Numerical floors
_EPS = 1e-30


# ---------------------------------------------------------------
# Dipole operator
# ---------------------------------------------------------------

def real_space_dipole_electronic(elec_crds: jax.Array) -> jax.Array:
    """Electronic dipole operator :math:`\\hat{\\mathbf d}_e = -\\sum_i \\mathbf r_i`.

    Atomic units; electron charge :math:`e=1`.

    Args:
        elec_crds: ``(n_elec, 3)`` electron positions for one walker.

    Returns:
        ``(3,)`` dipole vector.

    Notes:
        Matches the convention in :mod:`OmegaQMC.qed_fci` where the
        operator ``int1e_r`` (electronic position only) is used in the
        bilinear coupling. For neutral molecules, the nuclear part is
        absorbed into the cavity-vacuum reference; the residual
        constant adds an origin-dependent shift to the DSE that is
        irrelevant for energy differences.
    """
    return -jnp.sum(elec_crds, axis=-2)


def real_space_dipole_total(
    elec_crds: jax.Array,
    nuc_crds: jax.Array,
    charges: jax.Array,
) -> jax.Array:
    """Total dipole :math:`\\hat{\\mathbf d} = -\\sum_i \\mathbf r_i + \\sum_I Z_I \\mathbf R_I`.

    Provided for completeness; not used in the main local-energy
    estimator (which follows the qed_fci convention of
    electronic-dipole-only).

    Args:
        elec_crds: ``(n_elec, 3)`` electron positions.
        nuc_crds: ``(n_nuc, 3)`` nuclear positions.
        charges: ``(n_nuc,)`` nuclear charges.

    Returns:
        ``(3,)`` total dipole vector.
    """
    return real_space_dipole_electronic(elec_crds) + jnp.sum(
        charges[..., :, None] * nuc_crds, axis=-2,
    )


# ---------------------------------------------------------------
# Coulomb energy components
# ---------------------------------------------------------------

def coulomb_ee(elec_crds: jax.Array) -> jax.Array:
    """Electron-electron Coulomb energy :math:`\\sum_{i<j} 1/|\\mathbf r_i - \\mathbf r_j|`.

    Args:
        elec_crds: ``(n_elec, 3)``.

    Returns:
        Scalar Coulomb e-e energy.
    """
    n = elec_crds.shape[-2]
    diffs = elec_crds[..., :, None, :] - elec_crds[..., None, :, :]
    dist_sq = jnp.sum(diffs ** 2, axis=-1)
    # Strict upper triangle (i < j); regularise diagonal to avoid 1/0.
    iu = jnp.triu_indices(n, k=1)
    upper_dists = jnp.sqrt(dist_sq[..., iu[0], iu[1]] + _EPS)
    return jnp.sum(1.0 / upper_dists, axis=-1)


def coulomb_en(
    elec_crds: jax.Array,
    nuc_crds: jax.Array,
    charges: jax.Array,
) -> jax.Array:
    """Electron-nucleus attraction :math:`-\\sum_{i,I} Z_I / |\\mathbf r_i - \\mathbf R_I|`.

    Args:
        elec_crds: ``(n_elec, 3)``.
        nuc_crds: ``(n_nuc, 3)``.
        charges: ``(n_nuc,)``.

    Returns:
        Scalar e-n energy.
    """
    diffs = elec_crds[..., :, None, :] - nuc_crds[..., None, :, :]
    dists = jnp.sqrt(jnp.sum(diffs ** 2, axis=-1) + _EPS)
    return -jnp.sum(charges[..., None, :] / dists)


def coulomb_nn(
    nuc_crds: jax.Array,
    charges: jax.Array,
) -> jax.Array:
    """Nuclear repulsion :math:`\\sum_{I<J} Z_I Z_J / |\\mathbf R_I - \\mathbf R_J|`.

    This is a constant per geometry; cached by callers when possible.

    Args:
        nuc_crds: ``(n_nuc, 3)``.
        charges: ``(n_nuc,)``.

    Returns:
        Scalar n-n energy.
    """
    n = nuc_crds.shape[-2]
    if n < 2:
        return jnp.array(0.0, dtype=nuc_crds.dtype)
    diffs = nuc_crds[..., :, None, :] - nuc_crds[..., None, :, :]
    dist_sq = jnp.sum(diffs ** 2, axis=-1)
    iu = jnp.triu_indices(n, k=1)
    upper_dists = jnp.sqrt(dist_sq[..., iu[0], iu[1]] + _EPS)
    Zi = charges[..., iu[0]]
    Zj = charges[..., iu[1]]
    return jnp.sum(Zi * Zj / upper_dists, axis=-1)


# ---------------------------------------------------------------
# Coherent-state amplitude (analytical photon dressing)
# ---------------------------------------------------------------

def coherent_state_log_amplitude(
    n: jax.Array,
    alpha: jax.Array,
) -> jax.Array:
    """Log of coherent-state Fock amplitude :math:`\\langle n | \\alpha\\rangle`.

    For a coherent state :math:`|\\alpha\\rangle = D(\\alpha)|0\\rangle`:

    .. math::
        \\langle n | \\alpha \\rangle
        = e^{-|\\alpha|^2/2} \\frac{\\alpha^n}{\\sqrt{n!}}.

    Returns the log amplitude for stable evaluation:

    .. math::
        \\log\\langle n | \\alpha \\rangle
        = -\\tfrac{|\\alpha|^2}{2} + n\\log\\alpha - \\tfrac{1}{2}\\log n!

    Real :math:`\\alpha` only is supported (which is what the dipole-gauge
    Pauli-Fierz coherent-state shift produces).

    Args:
        n: integer photon Fock index (scalar).
        alpha: real scalar coherent-state displacement.

    Returns:
        Log amplitude (real scalar).

    Notes:
        At :math:`\\alpha = 0`, :math:`\\langle 0 | 0 \\rangle = 1` and
        :math:`\\langle n | 0 \\rangle = 0` for ``n > 0``. The
        ``alpha=0`` branch returns ``-jnp.inf`` for ``n > 0``, which is
        the correct log of zero amplitude; callers should guard against
        infinite log-amplitude in the residual NN ansatz.
    """
    n_f = jnp.asarray(n, dtype=alpha.dtype)
    safe_alpha = jnp.where(jnp.abs(alpha) < _EPS, _EPS, alpha)
    log_alpha_n = n_f * jnp.log(jnp.abs(safe_alpha))
    log_factorial = jax.lax.lgamma(n_f + 1.0)
    return -0.5 * alpha ** 2 + log_alpha_n - 0.5 * log_factorial


# ---------------------------------------------------------------
# Kinetic energy (electronic, photon coordinate is discrete here)
# ---------------------------------------------------------------

def electronic_kinetic_energy(
    log_psi_at_n,           # callable: elec_crds (n_elec, 3) -> log|Psi(r,n)|
    elec_crds: jax.Array,
) -> jax.Array:
    """Electronic kinetic energy :math:`-\\tfrac{1}{2}\\sum_i \\nabla_i^2`.

    Uses the standard Bijl-Jastrow identity for VMC local energy:

    .. math::
        \\frac{-\\tfrac{1}{2}\\nabla^2 \\Psi}{\\Psi}
        = -\\tfrac{1}{2}\\bigl(\\nabla^2 \\log|\\Psi|
                              + |\\nabla \\log|\\Psi||^2\\bigr).

    The electronic Laplacian is computed via the existing
    :func:`OmegaQMC.utils.laplacian_linearize` helper (re-exported as
    ``laplacian`` from :mod:`OmegaQMC.psi.nn.physics`).

    The Laplacian helper expects a flat input; this function handles
    the (n_elec, 3) ↔ (3·n_elec,) reshaping internally to match the
    convention used in :mod:`OmegaQMC.vmc_nn`.

    Args:
        log_psi_at_n: callable taking ``elec_crds`` of shape
            ``(n_elec, 3)`` and returning ``log|Psi(r, n)|`` for fixed
            photon index ``n``. Must be JAX-traceable.
        elec_crds: ``(n_elec, 3)`` electron positions.

    Returns:
        Scalar kinetic energy.
    """
    n_elec = elec_crds.shape[-2]

    def f_flat(r_flat):
        return log_psi_at_n(r_flat.reshape(n_elec, 3))

    r_flat = elec_crds.reshape(-1)
    lap_val, grad_val = laplacian(f_flat)(r_flat)
    return -0.5 * (lap_val + jnp.dot(grad_val, grad_val))


# ---------------------------------------------------------------
# Pauli-Fierz local energy
# ---------------------------------------------------------------

def pauli_fierz_local_energy(
    log_psi_fn,             # callable: (elec_crds, nuc_crds, n, params) -> log|Psi|
    params,
    elec_crds: jax.Array,    # (n_elec, 3)
    n: jax.Array,            # () integer scalar; photon Fock index sample
    nuc_crds: jax.Array,     # (n_nuc, 3)
    charges: jax.Array,      # (n_nuc,)
    omega: float,            # cavity frequency, Ha
    coupling_vec: jax.Array, # (3,) λ ε  (norm = λ, direction = ε)
    nph_max: int,
    enuc: jax.Array | float | None = None,
) -> jax.Array:
    """Local energy of the dipole-gauge Pauli-Fierz Hamiltonian.

    Components:
      1. Electronic kinetic energy :math:`E_K`.
      2. Electron-electron Coulomb :math:`E_{ee}`.
      3. Electron-nucleus Coulomb :math:`E_{en}`.
      4. Nuclear-nuclear repulsion :math:`E_{nn}` (constant; pre-cache).
      5. Photon energy :math:`\\omega n`.
      6. Bilinear coupling
         :math:`\\sqrt{\\omega/2}\\,\\lambda\\,(\\varepsilon\\cdot\\hat{\\mathbf d}_e)
                \\bigl[\\sqrt{n+1}\\,\\Psi(r,n+1)/\\Psi(r,n)
                      + \\sqrt{n}\\,\\Psi(r,n-1)/\\Psi(r,n)\\bigr]`.
      7. Dipole self-energy :math:`\\tfrac{1}{2}\\lambda^2(\\varepsilon\\cdot\\hat{\\mathbf d}_e)^2`.

    The bilinear ladder ratio at the Fock boundary :math:`n=0` drops the
    :math:`\\sqrt{n}` term (no |n-1⟩ state); at :math:`n=N_\\text{ph,max}`
    the :math:`\\sqrt{n+1}` term is dropped (truncation boundary). Use
    ``nph_max`` large enough that the truncation error is negligible at
    your sampled photon weights.

    Args:
        log_psi_fn: callable
            ``(elec_crds, nuc_crds, n, params) -> log|Psi(r,n)|``.
            Must be JAX-traceable in ``elec_crds`` (gradient + Laplacian
            taken). ``n`` is a Python int or 0-d JAX array.
        params: parameter pytree threaded through ``log_psi_fn``.
        elec_crds: ``(n_elec, 3)`` electron positions for this walker.
        n: photon Fock index for this walker (0-d integer array).
        nuc_crds: ``(n_nuc, 3)`` nuclear positions.
        charges: ``(n_nuc,)`` nuclear charges.
        omega: cavity-mode frequency (Ha).
        coupling_vec: light-matter coupling ``(3,)``: norm = λ, direction = ε.
        nph_max: photon-Fock truncation; bilinear ladder zeroed at
            ``n+1 > nph_max``.
        enuc: optional cached nuclear repulsion energy (avoids recompute
            in JIT loops). If ``None``, computed from ``nuc_crds`` and
            ``charges``.

    Returns:
        Scalar local energy (Ha).
    """
    # 1. Kinetic energy via existing Laplacian helper (electronic only;
    #    photon coordinate is discrete, no kinetic term in dipole gauge
    #    second-quantized form).
    def _logpsi_r_only(r):
        return log_psi_fn(r, nuc_crds, n, params)
    e_ke = electronic_kinetic_energy(_logpsi_r_only, elec_crds)

    # 2-4. Coulomb (electronic interactions and nuclear repulsion).
    e_ee = coulomb_ee(elec_crds)
    e_en = coulomb_en(elec_crds, nuc_crds, charges)
    if enuc is None:
        e_nn = coulomb_nn(nuc_crds, charges)
    else:
        e_nn = jnp.asarray(enuc)

    # 5. Photon harmonic energy (zero-point absorbed into reference).
    e_ph = omega * jnp.asarray(n, dtype=elec_crds.dtype)

    # 6. Bilinear coupling: needs Fock-ladder ratios.
    #    coupling_vec = lambda * eps (so its norm is lambda, direction is eps).
    lam = jnp.linalg.norm(coupling_vec)
    safe_lam = jnp.where(lam < _EPS, _EPS, lam)
    eps = coupling_vec / safe_lam
    d_e = real_space_dipole_electronic(elec_crds)
    eps_dot_d = jnp.dot(eps, d_e)

    log_psi_n = log_psi_fn(elec_crds, nuc_crds, n, params)
    n_int = jnp.asarray(n, dtype=jnp.int32)

    # +1 ladder term: <n+1 | b+b^dagger | n> = sqrt(n+1)
    n_plus = n_int + 1
    log_psi_np1 = log_psi_fn(elec_crds, nuc_crds, n_plus, params)
    ratio_up = jnp.exp(log_psi_np1 - log_psi_n)
    # Zero out at truncation boundary
    in_bounds_up = (n_plus <= nph_max).astype(elec_crds.dtype)
    bilinear_up = jnp.sqrt(jnp.asarray(n_plus, dtype=elec_crds.dtype)) * ratio_up * in_bounds_up

    # -1 ladder term: <n-1 | b+b^dagger | n> = sqrt(n)
    n_minus = jnp.maximum(n_int - 1, 0)
    log_psi_nm1 = log_psi_fn(elec_crds, nuc_crds, n_minus, params)
    ratio_dn = jnp.exp(log_psi_nm1 - log_psi_n)
    in_bounds_dn = (n_int >= 1).astype(elec_crds.dtype)
    bilinear_dn = jnp.sqrt(jnp.asarray(n_int, dtype=elec_crds.dtype)) * ratio_dn * in_bounds_dn

    # Tang / standard PF convention: H_bilin = +√(ω/2)·λ·(ε·∑ᵢ r̂ᵢ)·(b̂+b̂†).
    # In our code ε·d̂_e = -ε·∑ᵢ r̂ᵢ (electronic dipole with electron-charge sign),
    # so the sign in front of (ε·d̂_e) in this expression is NEGATIVE. This
    # places the polariton ground state in the c_e > 0 branch where a
    # positive-Ψ variational ansatz (returning only log|Ψ|) can represent
    # it. See module docstring for the unitary-equivalence discussion.
    e_bilinear = -jnp.sqrt(omega / 2.0) * lam * eps_dot_d * (
        bilinear_up + bilinear_dn
    )

    # 7. Dipole self-energy: (1/2) λ² (ε·d_e)²  (sign-invariant under d_e → -d_e)
    e_dse = 0.5 * lam ** 2 * eps_dot_d ** 2

    return e_ke + e_ee + e_en + e_nn + e_ph + e_bilinear + e_dse
