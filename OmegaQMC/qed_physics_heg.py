"""Velocity-gauge Pauli-Fierz local energy for the 2D/3D HEG.

The molecule QED implementation in :mod:`OmegaQMC.psi.nn.qed_physics`
uses **dipole (length) gauge** with operator
:math:`\\hat{\\mathbf d}_e = -\\sum_i \\mathbf r_i`. That operator is
non-periodic and its self-energy diverges with system size, so it
cannot be used for the periodic HEG.

This module implements the **velocity-gauge** Pauli-Fierz Hamiltonian
for a single q=0 cavity mode coupled to the in-plane (2D) or three-
dimensional (3D) electron gas:

.. math::
    H = \\sum_i \\frac{(\\hat{\\mathbf p}_i - q\\hat{\\mathbf A})^2}{2m}
        + V_{ee}^{\\text{Ewald}} + \\Omega\\,b^\\dagger b

For the photon mode at q=0 with polarization :math:`\\boldsymbol\\varepsilon`
and field operator
:math:`\\hat{\\mathbf A} = (\\lambda/\\sqrt{2\\Omega})
                             \\boldsymbol\\varepsilon (b + b^\\dagger)`,
the Hamiltonian expands to (electrons q=-1, m=1, units \\hbar=1):

.. math::
    H = \\hat T_e + V_{ee}^{\\text{Ewald}} + \\Omega\\,b^\\dagger b
        - g\\,(b+b^\\dagger)\\,(\\boldsymbol\\varepsilon\\cdot\\hat{\\mathbf P})
        + \\frac{N g^2}{2}\\,(b+b^\\dagger)^2,
        \\qquad g = \\lambda/\\sqrt{2\\Omega}

with total electronic momentum :math:`\\hat{\\mathbf P} = \\sum_i\\hat{\\mathbf p}_i`.

Why this gauge for HEG. The momentum operator and number operator
are well-defined under PBC; the position operator is not. The bilinear
:math:`(b+b^\\dagger)(\\varepsilon\\cdot\\hat{\\mathbf P})` is ill-suited
to real-:math:`\\Psi` ground states (it averages to zero by time-reversal
symmetry), so this module assumes ``complex_psi=True`` is the meaningful
regime — same conclusion the Weber 2025 PRL on cavity 2D HEG reaches via
QED-AFQMC.

Fock-ladder ratios. The :math:`(b+b^\\dagger)^2` operator requires
two-step ladder ratios:

.. math::
    \\frac{(b+b^\\dagger)^2 \\Psi}{\\Psi}\\bigg|_{(\\mathbf r, n)}
    = (2n+1)
      + \\sqrt{(n+1)(n+2)}\\,\\frac{\\Psi(\\mathbf r, n+2)}{\\Psi(\\mathbf r, n)}
      + \\sqrt{n(n-1)}\\,\\frac{\\Psi(\\mathbf r, n-2)}{\\Psi(\\mathbf r, n)}

Truncation: ratios are zeroed when the target Fock index falls outside
:math:`[0, N_{\\text{ph,max}}]`.

References
----------
- Rokaj et al. 2018, J. Phys. B 51, 034005 (PF Hamiltonian formulations)
- Weber et al. 2025, PRL 135, 126901 (QED-AFQMC for 2D HEG, cavity-renormalized)
- Schäfer, Buchholz, Ruggenthaler, Rubio, Narang 2021, PNAS (gauge choices)
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

from .psi.nn.physics import laplacian


__all__ = [
    "pauli_fierz_local_energy_velocity_heg",
]


_EPS = 1e-30


# ---------------------------------------------------------------
# Dim-aware kinetic helpers (the molecule versions in
# OmegaQMC.psi.nn.qed_physics hardcode reshape(n_elec, 3) and
# crash for 2D HEG).
# ---------------------------------------------------------------

def _ke_real(log_mag_at_n, elec_crds):
    """-1/2 (∇²log|Ψ| + |∇log|Ψ||²) for real Ψ, dim-agnostic."""
    n_elec = elec_crds.shape[-2]
    dim = elec_crds.shape[-1]

    def f_flat(r_flat):
        return log_mag_at_n(r_flat.reshape(n_elec, dim))

    r_flat = elec_crds.reshape(-1)
    lap, grad = laplacian(f_flat)(r_flat)
    return -0.5 * (lap + jnp.dot(grad, grad))


def _ke_complex(log_psi_complex_at_n, elec_crds):
    """Complex-Ψ kinetic via Bijl-Jastrow with phase, dim-agnostic.

    Mirrors :func:`OmegaQMC.psi.nn.qed_physics.electronic_kinetic_energy_complex`
    but uses ``elec_crds.shape[-1]`` instead of hardcoded 3.
    """
    n_elec = elec_crds.shape[-2]
    dim = elec_crds.shape[-1]

    def log_R_fn(rf):
        return jnp.real(log_psi_complex_at_n(rf.reshape(n_elec, dim)))

    def phi_fn(rf):
        return jnp.imag(log_psi_complex_at_n(rf.reshape(n_elec, dim)))

    r_flat = elec_crds.reshape(-1)
    lap_R, grad_R = laplacian(log_R_fn)(r_flat)
    lap_phi, grad_phi = laplacian(phi_fn)(r_flat)
    t_real = -0.5 * (lap_R + jnp.dot(grad_R, grad_R)
                     - jnp.dot(grad_phi, grad_phi))
    t_imag = -0.5 * (lap_phi + 2.0 * jnp.dot(grad_R, grad_phi))
    return t_real + 1j * t_imag


def _log_psi_complex_at_n(log_psi_signed_fn, params, n):
    """Return r → complex log Ψ(r, n) by combining log_mag + i·arg(sign).

    For complex_psi=True the signed_amp is a complex unit (= e^{iφ});
    arg(sign) recovers φ. For purely real positive ψ (sign = 1) the
    phase is zero and the function reduces to the real log magnitude.
    """
    def f(r):
        log_mag, sign = log_psi_signed_fn(r, n, params)
        log_mag_c = log_mag.astype(jnp.complex128) \
            if log_mag.dtype == jnp.float64 \
            else log_mag.astype(jnp.complex64)
        return log_mag_c + 1j * jnp.angle(sign)
    return f


def _eps_dot_grad_log_psi_complex(log_psi_signed_fn, params, elec_crds, n, eps):
    """Compute ε·Σᵢ ∇ᵢ log Ψ as a complex scalar.

    Required for the velocity-gauge bilinear coupling, where the local
    momentum estimator is :math:`(\\varepsilon\\cdot\\hat{\\mathbf P})\\Psi/\\Psi
    = -i \\sum_i \\varepsilon\\cdot\\nabla_i \\log\\Psi`.

    Args:
        log_psi_signed_fn: callable (r, n, params) → (log_mag, signed_amp).
        params: parameter pytree.
        elec_crds: (n_elec, dim) electron positions.
        n: photon Fock index.
        eps: (dim,) cavity polarization unit vector.

    Returns:
        Complex scalar :math:`\\sum_i \\varepsilon\\cdot\\nabla_i \\log\\Psi`.
    """
    n_elec, dim = elec_crds.shape[-2], elec_crds.shape[-1]
    eps = jnp.asarray(eps, dtype=elec_crds.dtype)
    log_psi_c = _log_psi_complex_at_n(log_psi_signed_fn, params, n)

    def log_R(rf):
        return jnp.real(log_psi_c(rf.reshape(n_elec, dim)))

    def phi(rf):
        return jnp.imag(log_psi_c(rf.reshape(n_elec, dim)))

    r_flat = elec_crds.reshape(-1)
    grad_R = jax.grad(log_R)(r_flat).reshape(n_elec, dim)
    grad_phi = jax.grad(phi)(r_flat).reshape(n_elec, dim)
    # ε·Σᵢ ∇ᵢ log R, complex part = ε·Σᵢ ∇ᵢ φ.
    re = jnp.sum(grad_R @ eps)
    im = jnp.sum(grad_phi @ eps)
    return re + 1j * im


def _ladder_ratio_signed(log_psi_signed_fn, params, elec_crds, n, n_target,
                         in_bounds):
    """Compute signed Ψ(r, n_target) / Ψ(r, n).

    Returns 0 outside Fock-truncation bounds.
    """
    log_n, sign_n = log_psi_signed_fn(elec_crds, n, params)
    log_t, sign_t = log_psi_signed_fn(elec_crds, n_target, params)
    abs_sign_n = jnp.abs(sign_n)
    sign_n_unit = jnp.where(
        abs_sign_n > 0,
        sign_n / jnp.maximum(abs_sign_n, _EPS),
        jnp.ones_like(sign_n),
    )
    safe_sign_n = jnp.where(abs_sign_n < _EPS, _EPS * sign_n_unit, sign_n)
    ratio = (sign_t / safe_sign_n) * jnp.exp(log_t - log_n)
    return ratio * in_bounds


def pauli_fierz_local_energy_velocity_heg(
    log_psi_signed_fn,       # (elec_crds, n, params) → (log_mag, signed_amp)
    params,
    elec_crds: jax.Array,    # (n_elec, dim), dim ∈ {2, 3}
    n: jax.Array,            # () integer scalar; photon Fock index sample
    omega: float,            # cavity frequency, Ha
    coupling_vec: jax.Array, # (dim,) λ ε  (norm = λ, direction = ε)
    nph_max: int,
    ewald_pair_fn,           # callable: r → V_ee(r), Ewald pair energy
    complex_psi: bool = True,
) -> jax.Array:
    """Velocity-gauge Pauli-Fierz local energy for the HEG.

    Components:
      1. Electronic kinetic energy :math:`\\hat T_e` (complex-aware via
         :func:`electronic_kinetic_energy_complex`).
      2. Electron-electron Ewald V_ee.
      3. Photon energy :math:`\\Omega n`.
      4. Bilinear (paramagnetic) coupling
         :math:`-g(b+b^\\dagger)(\\varepsilon\\cdot\\hat{\\mathbf P})`,
         :math:`g=\\lambda/\\sqrt{2\\Omega}`.
      5. Diamagnetic coupling :math:`(Ng^2/2)(b+b^\\dagger)^2`.

    For real Ψ in a translationally invariant ground state the bilinear
    contributes zero; the diamagnetic term gives a small frequency
    renormalization. Set ``complex_psi=True`` (default) and use a
    complex-amplitude FockHead to access nontrivial cavity-electron
    coupling.

    Args:
        log_psi_signed_fn: callable returning (log_mag, signed_amp).
            For complex_psi=True the signed_amp is a complex unit on the
            unit circle (= exp(i*phase)). For real ψ it is ±1.
        params: parameter pytree.
        elec_crds: (n_elec, dim) electron positions.
        n: 0-d integer photon Fock index.
        omega: cavity-mode frequency (Ha).
        coupling_vec: (dim,) cavity coupling vector (norm = λ, dir = ε).
            For 2D HEG, polarization must be in the 2D plane.
        nph_max: photon-Fock truncation; ladder zeroed beyond bounds.
        ewald_pair_fn: callable r → scalar Ewald pair energy.
            Bound to a particular EwaldTables instance by the caller.
        complex_psi: if True, kinetic uses the complex estimator and the
            bilinear picks up a real contribution from the phase
            gradient. Default True.

    Returns:
        Scalar local energy. COMPLEX when ``complex_psi=True``; physical
        observable is :math:`\\Re\\langle E_\\text{loc}\\rangle`.
    """
    n_elec = elec_crds.shape[-2]

    # 1. Kinetic.
    if complex_psi:
        log_psi_c = _log_psi_complex_at_n(log_psi_signed_fn, params, n)
        e_ke = _ke_complex(log_psi_c, elec_crds)
    else:
        def _logmag_only(r):
            log_mag, _ = log_psi_signed_fn(r, n, params)
            return log_mag
        e_ke = _ke_real(_logmag_only, elec_crds)

    # 2. Electron-electron Ewald.
    e_ee = ewald_pair_fn(elec_crds)

    # 3. Photon energy.
    e_ph = omega * jnp.asarray(n, dtype=elec_crds.dtype)

    # 4-5. Cavity-matter coupling.
    lam = jnp.linalg.norm(coupling_vec)
    safe_lam = jnp.where(lam < _EPS, _EPS, lam)
    eps = coupling_vec / safe_lam
    g = lam / jnp.sqrt(2.0 * omega)

    n_int = jnp.asarray(n, dtype=jnp.int32)

    # (b+b†) ladder ratios.
    n_p1 = n_int + 1
    n_m1 = jnp.maximum(n_int - 1, 0)
    in_p1 = (n_p1 <= nph_max).astype(elec_crds.dtype)
    in_m1 = (n_int >= 1).astype(elec_crds.dtype)
    r_p1 = _ladder_ratio_signed(log_psi_signed_fn, params, elec_crds, n_int, n_p1, in_p1)
    r_m1 = _ladder_ratio_signed(log_psi_signed_fn, params, elec_crds, n_int, n_m1, in_m1)
    n_f = jnp.asarray(n_int, dtype=elec_crds.dtype)
    bb_local = jnp.sqrt(n_f + 1.0) * r_p1 + jnp.sqrt(n_f) * r_m1

    # 4. Bilinear: -g·(b+b†)·(ε·P̂) where ε·P̂ Ψ/Ψ = -i·Σᵢ ε·∇ᵢ log Ψ.
    if complex_psi:
        eps_dot_grad = _eps_dot_grad_log_psi_complex(
            log_psi_signed_fn, params, elec_crds, n_int, eps,
        )
        # ε·P̂ Ψ/Ψ = -i · eps_dot_grad
        eps_dot_P = -1j * eps_dot_grad
    else:
        # Real Ψ: ε·P̂ Ψ/Ψ is purely imaginary, gives no real-energy
        # contribution after walker averaging. Set to zero in local
        # energy to keep e_ke real and avoid spurious imaginary clutter.
        eps_dot_P = jnp.asarray(0.0, dtype=elec_crds.dtype)

    e_bilinear = -g * bb_local * eps_dot_P

    # 5. Diamagnetic: (N·g²/2)·(b+b†)².
    n_p2 = n_int + 2
    n_m2 = jnp.maximum(n_int - 2, 0)
    in_p2 = (n_p2 <= nph_max).astype(elec_crds.dtype)
    in_m2 = (n_int >= 2).astype(elec_crds.dtype)
    r_p2 = _ladder_ratio_signed(log_psi_signed_fn, params, elec_crds, n_int, n_p2, in_p2)
    r_m2 = _ladder_ratio_signed(log_psi_signed_fn, params, elec_crds, n_int, n_m2, in_m2)
    bb_sq_local = (
        (2.0 * n_f + 1.0)
        + jnp.sqrt((n_f + 1.0) * (n_f + 2.0)) * r_p2
        + jnp.sqrt(n_f * jnp.maximum(n_f - 1.0, 0.0)) * r_m2
    )
    e_diamag = 0.5 * n_elec * g ** 2 * bb_sq_local

    return e_ke + e_ee + e_ph + e_bilinear + e_diamag
