"""Minimal HEG wavefunction adapter for cavity-QED.

Wraps an existing real-positive HEG wavefunction (from
:mod:`OmegaQMC.psi.nn.heg_wf_module`) with a learnable Fock-state head
that introduces an r-dependent **phase** so the joint trial
:math:`\\Psi(\\mathbf r, n)` is genuinely cavity-electron entangled.

For the q=0 cavity mode in velocity gauge with real positive electronic
ψ_e and a purely n-dependent phase factor, the bilinear paramagnetic
coupling :math:`(b+b^\\dagger)(\\boldsymbol\\varepsilon\\cdot\\hat{\\mathbf P})`
still vanishes because :math:`\\boldsymbol\\varepsilon\\cdot\\nabla\\log\\Psi`
is purely real. To capture nontrivial cavity-electron coupling at q=0
the phase **must** depend on r through a polarization-projected periodic
feature. This module implements the simplest such head:

.. math::
    \\Psi(\\mathbf r, n) = \\psi_{\\text{HEG}}(\\mathbf r)\\,\\exp\\bigl(i\\,
        \\theta_n \\sum_i \\sin(2\\pi\\,\\boldsymbol\\varepsilon\\cdot \\mathbf r_i / L)\\bigr)

with trainable phases :math:`\\{\\theta_n\\}_{n=1}^{N_{\\text{ph,max}}}` and
:math:`\\theta_0 = 0` fixed (overall-phase gauge).

The feature :math:`f(\\mathbf r)=\\sum_i\\sin(2\\pi\\boldsymbol\\varepsilon\\cdot
\\mathbf r_i/L)` is:
- periodic under :math:`\\mathbf r_i \\to \\mathbf r_i + \\mathbf L_a`
- real and well-defined for any electron geometry
- polarization-projected (so the phase responds to the cavity mode direction)
- :math:`\\nabla_i f = (2\\pi/L)\\cos(2\\pi\\boldsymbol\\varepsilon\\cdot \\mathbf r_i/L)\\boldsymbol\\varepsilon`,
  giving a finite bilinear contribution at any walker.

Future versions may swap this 1-parameter-per-Fock-state head for a
small MLP that takes (n, electronic features) → phase. For V1 we keep
it minimal: nph_max trainable scalars.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import nnx


__all__ = [
    "HEGFockPhaseHead",
    "make_qed_heg_log_psi_signed",
]


class HEGFockPhaseHead(nnx.Module):
    """Per-Fock-state amplitude + phase head for cavity-QED HEG.

    log|c_n| = b_n  (with b_0 = 0 fixed for normalization gauge):
        biases the trial toward the photon vacuum at init via b_n = -3·n
        for n≥1 (e^{-3} ≈ 0.05 ratio to |0⟩, MCMC-mobile but
        vacuum-dominant). Trainable so SR can adjust the Fock weights.
    phase(r, n) = θ_n · Σᵢ sin(2π·ε·rᵢ/L)  (θ_0 = 0, gauge):
        polarization-projected periodic feature; provides the r-n
        entanglement needed to make the velocity-gauge bilinear
        contribute. Trainable, init at 0 (signed_amp ≡ Slater sign at
        iter 0; cavity coupling enters only via diamagnetic A² shift).
    """

    def __init__(self, nph_max: int, L: float, eps: jax.Array, *, rngs):
        self.nph_max = int(nph_max)
        self.L = float(L)
        self.eps = nnx.data(jnp.asarray(eps, dtype=jnp.float64))
        # log|c_n| for n=1..nph_max. Init bias toward vacuum.
        self.log_amp = nnx.Param(
            -3.0 * jnp.arange(1, self.nph_max + 1, dtype=jnp.float64),
        )
        # Phase coefficients θ_n for n=1..nph_max (θ_0=0 gauge).
        self.theta = nnx.Param(jnp.zeros((self.nph_max,), dtype=jnp.float64))

    def __call__(self, elec, n):
        """Return (log_amp_n, phase) for the photon Fock state n."""
        n_int = jnp.asarray(n, dtype=jnp.int32)
        idx = jnp.maximum(n_int - 1, 0)
        in_pos = n_int > 0
        log_amp_n = jnp.where(in_pos, self.log_amp[idx], 0.0)

        eps_dot_r = elec @ self.eps                          # (n_elec,)
        f = jnp.sum(jnp.sin(2.0 * jnp.pi * eps_dot_r / self.L))
        theta_n = jnp.where(in_pos, self.theta[idx], 0.0)
        phase = theta_n * f
        return log_amp_n, phase


def make_qed_heg_log_psi_signed(
    base_wf: nnx.Module,                # existing HEG ψ module (real, positive)
    nph_max: int,
    L: float,
    eps: jax.Array,
    init_key,
):
    """Build a signed log-ψ callable for cavity-QED HEG.

    Args:
        base_wf: an HEG wavefunction module with a callable interface
            ``base_wf(elec_crds) → log|ψ(r)|`` (real scalar).
            Typically constructed via
            :func:`OmegaQMC.psi.nn.heg_wf_module.build_heg_psiformer_wf`.
        nph_max: photon-Fock truncation.
        L: simulation cell side length.
        eps: (dim,) cavity polarization unit vector.
        init_key: JAX PRNG key for FockHead init.

    Returns:
        ``(log_psi_signed, init_params, graphdef)``:
          * ``log_psi_signed(elec, n, params) → (log_mag, signed_amp)``
            where ``signed_amp`` is a complex unit ``exp(i·phase)``.
          * ``init_params``: nnx Param state for {base_wf, fock_head}.
          * ``graphdef``: nnx graphdef for re-merging.
    """
    rngs = nnx.Rngs(init_key)
    fock_head = HEGFockPhaseHead(nph_max, L, eps, rngs=rngs)

    # Build a combined module so a single graphdef/params split covers
    # both base_wf and the FockHead.
    class _Joint(nnx.Module):
        def __init__(self, base, head):
            self.base = base
            self.head = head

        def __call__(self, elec, n):
            psi = self.base(elec)             # Psi(sign, log)
            log_amp_n, phase = self.head(elec, n)
            return psi.log, psi.sign, log_amp_n, phase

    joint = _Joint(base_wf, fock_head)
    graphdef, params, other = nnx.split(joint, nnx.Param, ...)

    def log_psi_signed(elec, n, params):
        mdl = nnx.merge(graphdef, params, other)
        log_mag_e, slater_sign, log_amp_n, phase = mdl(elec, n)
        # log|Ψ(r,n)| = log|ψ_e(r)| + log|c_n|.
        log_mag = log_mag_e + log_amp_n
        slater_c = slater_sign.astype(jnp.complex128)
        signed_amp = slater_c * jnp.exp(1j * phase.astype(jnp.complex128))
        return log_mag, signed_amp

    return log_psi_signed, params, graphdef
