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

    The Fock-state amplitude (vacuum-biased `b_n = -3n`) and phase
    (initially zero) are **baked into closure constants** rather than
    held as trainable Params on a separate FockHead module. Reason:
    nnx-Param subtree filtering for "freeze the FockHead" silently
    fails at production scale (nested state structure with
    cusp/BF/DeepJastrow), letting these few rarely-fired params get
    SR-optimized with near-singular Fisher diagonal → wavefunction
    drift → variational collapse below DMC. By making them closure
    constants, only the base electronic ψ is in the trainable param
    tree, exactly matching bare HEG.

    At finite λ where FockHead needs to be trained, add a separate
    optimizer state (e.g., Adam) for these scalars rather than mixing
    them into the SR Fisher of the base ψ.

    Returns:
        ``(log_psi_signed, phase_smooth, params, graphdef)``:
          * ``log_psi_signed(elec, n, params) → (log_mag, signed_amp)``
          * ``phase_smooth(elec, n, params) → real scalar``
          * ``params``: nnx Param state for the BASE ψ only.
          * ``graphdef``: nnx graphdef for re-merging the base.
    """
    del init_key  # no FockHead Module → no RNG needed
    eps = jnp.asarray(eps, dtype=jnp.float64)
    L_const = float(L)
    # Vacuum-biased log amplitudes for n=0..nph_max. log|c_0|=0 (gauge).
    log_amp_table = jnp.concatenate([
        jnp.zeros((1,), dtype=jnp.float64),
        -3.0 * jnp.arange(1, nph_max + 1, dtype=jnp.float64),
    ])

    graphdef, params, other = nnx.split(base_wf, nnx.Param, ...)

    def log_psi_signed(elec, n, params):
        mdl = nnx.merge(graphdef, params, other)
        psi = mdl(elec)                       # Psi(sign, log)
        n_int = jnp.asarray(n, dtype=jnp.int32)
        log_amp_n = log_amp_table[n_int]
        log_mag = psi.log + log_amp_n
        slater_c = psi.sign.astype(jnp.complex128)
        # Phase is 0 for V1 (FockHead not trained). signed_amp = slater
        # carried as complex dtype so Fock-ladder ratios broadcast.
        signed_amp = slater_c
        return log_mag, signed_amp

    def phase_smooth(elec, n, params):
        del elec, n, params
        return jnp.asarray(0.0, dtype=jnp.float64)

    return log_psi_signed, phase_smooth, params, graphdef
