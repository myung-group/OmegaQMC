"""QED adapter bridging NN trial wavefunctions to joint electron-photon evaluation.

Extends :mod:`OmegaQMC.psi.nn.adapter` with a coherent-state-shifted
factorized ansatz for the dipole-gauge Pauli-Fierz Hamiltonian:

.. math::
    \\Psi(\\mathbf r, n) = \\Psi_e(\\mathbf r)\\, \\langle n | \\alpha \\rangle,

equivalently :math:`|\\Psi\\rangle = |\\Psi_e\\rangle\\otimes \\hat D(\\alpha)|0\\rangle`,
where :math:`\\hat D(\\alpha)` is the photon displacement operator and
:math:`\\alpha` is a real learnable scalar absorbing the dominant photon
dressing.

Methodological context
----------------------
Standard QED-CCSD-1 truncates at single photonic excitations from the
vacuum :math:`|0\\rangle`; QED-FCI converges with photon-Fock cutoff
:math:`N_{\\text{ph,max}}`. At ultrastrong coupling
(:math:`\\lambda \\gtrsim 0.5`), both methods need
:math:`N_{\\text{ph,max}} \\gg 10` because the ground state has substantial
photon occupation. The coherent-state shift :math:`\\hat D(\\alpha)` rotates
to a frame where the residual Fock distribution stays narrow around the
displaced vacuum :math:`|\\alpha\\rangle`, dramatically reducing the
effective truncation cost.

This factorized form (electronic NN × photon coherent state) is the
*minimal* coherent-state-shifted ansatz: it lacks explicit
electron-photon entanglement in the NN itself. A follow-on commit will
add :math:`n`-dependence to the NN to capture residual entanglement
beyond the mean coherent-state dressing.

References
----------
- Tang et al. 2025, arXiv:2503.15644 (deep-VMC; non-shifted Fock encoding)
- Haugland et al. 2020, arXiv:2005.04477 (QED-CCSD-1 with coherent-state basis)
- See also ``OmegaQMC.psi.nn.qed_physics`` for the local-energy estimator
  that consumes the ``log_psi`` produced here.
"""
from __future__ import annotations

from typing import Callable, Tuple, Any

import jax
import jax.numpy as jnp

from .adapter import make_nn_log_psi
from .qed_physics import coherent_state_log_amplitude


__all__ = [
    "QEDLogPsiParams",
    "make_qed_nn_log_psi",
    "analytical_alpha_perturbative",
]


# Type alias for the combined (NN params + alpha) pytree
QEDLogPsiParams = dict[str, Any]   # {'nn': nnx_state, 'alpha': scalar}


def analytical_alpha_perturbative(
    omega: float,
    coupling_vec: jax.Array,
    dipole_expectation: float,
) -> float:
    """First-order analytical estimate of coherent-state shift α.

    Minimizing the dipole-gauge Pauli-Fierz energy at the bilinear-
    coupling level gives the perturbative shift:

    .. math::
        \\alpha = -\\frac{\\lambda\\,\\boldsymbol\\varepsilon\\cdot\\langle\\hat{\\mathbf d}_e\\rangle}{\\sqrt{2\\omega}}

    This cancels the linear forcing term :math:`\\sqrt{\\omega/2}\\,\\lambda\\,
    (\\varepsilon\\cdot\\hat d_e)\\,(b+b^\\dagger)` at the mean dipole.

    Args:
        omega: cavity-mode frequency (Ha).
        coupling_vec: ``(3,)`` light-matter coupling
            (norm = λ, direction = ε).
        dipole_expectation: scalar :math:`\\boldsymbol\\varepsilon\\cdot\\langle\\hat{\\mathbf d}_e\\rangle`
            evaluated at the trial state (e.g., from QED-HF or HF).

    Returns:
        Real scalar α suitable for ``alpha_init``.

    Notes:
        For closed-shell neutral systems with electronic-only dipole
        convention, ``dipole_expectation`` may be zero by symmetry
        (e.g., (H₂)₂ aligned configurations). In that case α=0 is
        already the perturbative optimum and the routine returns 0.
        The learnable α can still adjust at higher orders during VMC
        training.
    """
    cv = jnp.asarray(coupling_vec)
    lam = float(jnp.linalg.norm(cv))
    if lam < 1e-15 or float(omega) < 1e-15:
        return 0.0
    return -lam * float(dipole_expectation) / (2.0 * float(omega)) ** 0.5


def make_qed_nn_log_psi(
    config,
    mol_info,
    rng_key,
    *,
    omega: float,
    coupling_vec: jax.Array,
    alpha_init: float | jax.Array = 0.0,
    alpha_train: bool = True,
) -> Tuple[Callable, QEDLogPsiParams, Any]:
    """Build a coherent-state-shifted joint electron-photon NN ansatz.

    Wraps :func:`make_nn_log_psi` from :mod:`OmegaQMC.psi.nn.adapter`,
    composing the electronic NN log-amplitude with the analytical
    coherent-state log-amplitude :math:`\\log\\langle n | \\alpha \\rangle`.

    The returned ``log_psi`` has signature
    ``(elec_crds, nuc_crds, n, params) -> float``, ready to be consumed
    by :func:`OmegaQMC.psi.nn.qed_physics.pauli_fierz_local_energy`.

    Args:
        config: :class:`NNAnsatzConfig` or string passed to ``make_nn_log_psi``.
        mol_info: :class:`~OmegaQMC.utils.Mole_custom`.
        rng_key: JAX PRNG key for NN parameter init.
        omega: cavity mode frequency (Ha). Stored on params for
            convenience; the local-energy estimator passes it explicitly.
        coupling_vec: ``(3,)`` light-matter coupling vector
            (norm = λ, direction = ε). Same convention as
            :func:`OmegaQMC.qed_fci.qed_fci`.
        alpha_init: initial coherent-state displacement (real scalar).
            Use 0.0 for "start at the photon vacuum". For neutral
            symmetric dimers (e.g., (H₂)₂ along ε_x), the perturbative
            α from :func:`analytical_alpha_perturbative` is zero, so
            the default is fine. For polar systems (water dimer, HF
            dimer) call ``analytical_alpha_perturbative`` first.
        alpha_train: if True (default), α is included in the trainable
            parameter set under the key ``'alpha'`` and updated by the
            optimizer. If False, α is frozen at ``alpha_init`` (test mode).

    Returns:
        Tuple ``(log_psi, init_params, graphdef)``:

        - ``log_psi(elec_crds, nuc_crds, n, params) -> log|Psi(r, n)|``.
          Scalar real output. Differentiable in ``elec_crds`` (kinetic
          energy) and in ``params`` (parameter gradients).
        - ``init_params``: dict ``{'nn': <nnx_state>, 'alpha': <scalar>}``
          or ``{'nn': <nnx_state>}`` if ``alpha_train=False``.
        - ``graphdef``: NNX GraphDef for reconstructing the electronic
          wavefunction model (passed through unchanged from
          ``make_nn_log_psi``).

    Notes:
        * Electronic and photonic factors are independent in this
          ansatz — there is *no* (r, n)-correlation beyond the
          mean-field photon dressing absorbed in α. This is the
          minimal viable coherent-state-shifted ansatz; the planned
          Tang-style n-dependent embedding adds residual correlation.
        * The bilinear-coupling Fock-ladder ratios reduce to closed-form
          coherent-state amplitude ratios:
          :math:`\\Psi(r,n+1)/\\Psi(r,n) = \\alpha/\\sqrt{n+1}`.
          This makes the QED local energy effectively analytical in n
          at leading order — a useful invariant for testing.
    """
    # Build the standard electronic NN log_psi.
    elec_log_psi, elec_init_params, graphdef = make_nn_log_psi(
        config, mol_info, rng_key,
    )

    # Pack initial params. Use plain dict for pytree compatibility.
    alpha0 = jnp.asarray(alpha_init, dtype=jnp.float64)
    if alpha_train:
        init_params: QEDLogPsiParams = {"nn": elec_init_params, "alpha": alpha0}
    else:
        # Frozen α: don't expose to optimizer; close it inside log_psi.
        init_params = {"nn": elec_init_params}

    # Cache the coupling for diagnostic output (not used in evaluation
    # since the local-energy estimator passes omega/coupling_vec explicitly).
    _ = (omega, coupling_vec)  # documented inputs; kept for future autoshift

    if alpha_train:
        def log_psi(elec_crds, nuc_crds, n, params):
            log_elec = elec_log_psi(elec_crds, nuc_crds, params["nn"])
            log_chi = coherent_state_log_amplitude(n, params["alpha"])
            return log_elec + log_chi
    else:
        # α frozen — close over alpha0
        alpha_frozen = alpha0

        def log_psi(elec_crds, nuc_crds, n, params):
            log_elec = elec_log_psi(elec_crds, nuc_crds, params["nn"])
            log_chi = coherent_state_log_amplitude(n, alpha_frozen)
            return log_elec + log_chi

    return log_psi, init_params, graphdef
