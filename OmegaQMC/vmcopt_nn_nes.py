"""Neural-network excited-state VMC (NES-VMC) via penalty method.

Trains a neural-network trial wavefunction :math:`\\Psi_{\\rm NN}^{(1)}`
orthogonal to a previously-trained ground-state trial
:math:`\\Psi_{\\rm NN}^{(0)}` by adding an overlap penalty to the
loss function:

.. math::

    L(\\theta_1) \\;=\\; 0.2\\,E + 0.8\\,\\sigma_E
        \\;+\\;\\lambda\\,F(\\theta_1)

where :math:`F(\\theta_1) = \\bigl(\\mathbb{E}_{R \\sim |\\Psi_1|^2}
[\\Psi_0(R) / \\Psi_1(R)]\\bigr)^{2}` is the one-sided Monte Carlo
overlap estimator. This is the simpler one-sided variant of the
Pfau et al. 2024 NES-VMC penalty: the bi-directional estimator that
divides by both norms requires sampling from :math:`|\\Psi_0|^2`
walkers as well, which we omit for simplicity. The one-sided variant
still drives the overlap to zero as :math:`\\Psi_1` converges; it
gives up the rigorous symmetric form needed when one wants to bound
:math:`\\langle\\Psi_0|\\Psi_1\\rangle / \\sqrt{Z_0 Z_1}` directly,
but in practice the orthogonalisation is reached.

This driver reuses the IRAdam Metropolis + energy infrastructure
(:class:`~OmegaQMC.vmcopt_nn_iradam._VMCOptDriverNN_IRAdam`); only
the loss function is overridden.

Signed-Psi evaluation requires the ``.signed`` attribute on
``log_psi``; see :mod:`OmegaQMC.psi.nn.adapter`.
"""

import os

import h5py
import jax
import jax.numpy as jnp
import numpy as np

from .vmcopt_nn_iradam import _VMCOptDriverNN_IRAdam
from .psi.nn.checkpoint import load_nn_checkpoint


class _VMCOptDriverNN_NES(_VMCOptDriverNN_IRAdam):
    """Penalty-method NES-VMC optimiser.

    Inherits Metropolis sampling, energy evaluation, and Adam loop from
    the IRAdam driver. Overrides ``loss_fn`` to add an overlap penalty
    against a frozen ground-state trial wavefunction.
    """

    def __init__(
        self,
        mol_info,
        config,
        init_key,
        ground_state_params,
        lambda_penalty: float = 1.0,
    ):
        """Construct an NES-VMC driver.

        Args:
            mol_info: :class:`OmegaQMC.utils.Mole_custom`
            config: ansatz config (same as for the ground-state run)
            init_key: JAX PRNG key for parameter initialisation
            ground_state_params: NNX parameter pytree for
                :math:`\\Psi_{\\rm NN}^{(0)}`, loaded from a
                previously-trained checkpoint. Must use the same
                ``config`` (architecture) as this driver.
            lambda_penalty: weight :math:`\\lambda` on the overlap
                penalty term in the loss.
        """
        super().__init__(mol_info, config, init_key)
        self.ground_state_params = ground_state_params
        self.lambda_penalty = float(lambda_penalty)

        log_psi_signed = self.log_psi.signed
        nuc_crds = self.nuc_crds
        compute_batch_energy = self.compute_batch_energy

        def signed_psi_one(walker, params):
            sign, log_amp = log_psi_signed(walker, nuc_crds, params)
            return sign * jnp.exp(log_amp)

        def log_ratio_signed(walker, params_excited, params_ground):
            """Compute ``Psi_g(R)/Psi_e(R)`` in a sign-aware,
            log-stable form.

            Returns a finite real scalar; the ``exp(log_g - log_e)``
            factor cancels the overall normalisation of both NNs
            without underflow when ``log|Psi_e|`` is large negative.
            """
            sign_e, log_e = log_psi_signed(
                walker, nuc_crds, params_excited,
            )
            sign_g, log_g = log_psi_signed(
                walker, nuc_crds, params_ground,
            )
            return sign_e * sign_g * jnp.exp(log_g - log_e)

        @jax.jit
        def overlap_estimator(batch_walkers, params_excited):
            """One-sided overlap estimator
            ``E_{|Psi_1|^2}[Psi_0/Psi_1]`` on the supplied walkers."""
            ratios = jax.vmap(
                log_ratio_signed, in_axes=(0, None, None),
            )(batch_walkers, params_excited, ground_state_params)
            return ratios.mean()

        @jax.jit
        def loss_fn_nes(params, batch_walkers):
            energies = compute_batch_energy(batch_walkers, params)
            e_loss = 0.2 * energies.mean() + 0.8 * energies.std()
            overlap = overlap_estimator(batch_walkers, params)
            penalty = overlap ** 2
            return e_loss + lambda_penalty * penalty

        # Override the parent's loss_fn with the penalty-augmented one.
        # The parent's __call__ loop will call self.loss_fn each step.
        self.loss_fn = loss_fn_nes
        self.overlap_estimator = overlap_estimator
        self._signed_psi_one = signed_psi_one

    def evaluate_overlap(self, walkers, params=None) -> float:
        """Convenience: evaluate the one-sided overlap estimator
        ``E[Psi_0/Psi_1]`` on a walker bank.

        ``walkers`` shape ``(K_s, n_elec, 3)``.
        """
        if params is None:
            params = self.init_params
        return float(self.overlap_estimator(jnp.asarray(walkers), params))


def get_vmcopt_nn_nes_func(
    mol_info,
    config,
    init_key,
    ground_state_checkpoint: str,
    lambda_penalty: float = 1.0,
):
    """Construct an NES-VMC driver.

    Args:
        mol_info: :class:`OmegaQMC.utils.Mole_custom` --- same molecule
            used to train the ground state.
        config: ansatz config (must match the ground-state architecture).
        init_key: PRNG key for the excited-state parameter
            initialisation.
        ground_state_checkpoint: HDF5 path to a checkpoint produced
            by a previous ground-state run with the same ``config``.
            The frozen :math:`\\Psi_{\\rm NN}^{(0)}` parameters are
            read from this file.
        lambda_penalty: weight on the overlap-squared penalty term.
            Should be large enough that the penalty noticeably
            constrains the optimiser; we recommend
            :math:`\\lambda \\sim 1` as a starting value. Larger
            :math:`\\lambda` orthogonalises faster at the cost of
            energy convergence speed.

    Returns:
        :class:`_VMCOptDriverNN_NES` instance.
    """
    # We need a template params pytree to load the ground-state
    # checkpoint into. Build the driver first (which constructs
    # init_params), then load.
    template_driver = _VMCOptDriverNN_IRAdam(mol_info, config, init_key)
    ground_params, _meta = load_nn_checkpoint(
        ground_state_checkpoint, template_driver.init_params,
    )
    return _VMCOptDriverNN_NES(
        mol_info, config, init_key,
        ground_state_params=ground_params,
        lambda_penalty=lambda_penalty,
    )
