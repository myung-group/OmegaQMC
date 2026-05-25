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
import optax

from .vmcopt_nn_iradam import _VMCOptDriverNN_IRAdam
from .psi.nn.checkpoint import load_nn_checkpoint, save_nn_checkpoint


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


# ===========================================================================
# Basis-resolved NES-VMC variant
# ===========================================================================
#
# The real-space penalty above can be satisfied by orthogonalising Psi_1
# in basis-incomplete directions, leaving the basis-resolved CI vector
# unchanged. The basis-resolved variant penalises the overlap between
# Psi_1 and a synthetic ground state constructed entirely within the
# chosen CI basis:
#
#   Psi_synth^(0)(R)  =  sum_I  c_hat^(0)_I  D_I^norm(R)
#
# where c_hat^(0) is the CS-recovered CI vector of the ground-state NN
# and D_I^norm are the L^2-normalised Slater determinants in the
# natural-orbital basis. The penalty becomes
#
#   F_CI(theta_1)  =  ( E_{|Psi_1|^2}[Psi_synth^(0) / Psi_1] )^2
#
# Because Psi_synth^(0) lies entirely in span{D_I}, driving F_CI to zero
# forces ortho-projection of Psi_1 onto the CI basis to be orthogonal to
# c_hat^(0). The CI-vector inner product <c_hat^(0) | c_hat^(1)> is
# directly minimised, in contrast to the real-space variant.


class _VMCOptDriverNN_NES_Basis(_VMCOptDriverNN_IRAdam):
    """Basis-resolved NES-VMC: penalty against a synthetic ground state
    built from the recovered CI vector.

    Unlike :class:`_VMCOptDriverNN_NES`, the penalty here cannot be
    expressed inside a JIT-compiled loss without a JAX-compatible
    Gaussian-orbital evaluator, because ``Psi_synth^(0)`` involves
    PySCF GTO calls. We therefore pre-compute ``Psi_synth^(0)(R)`` at
    every sampled walker each outer iteration and pass the
    pre-computed array to the loss function as a third (static-shape)
    argument. The Adam loop is reimplemented here rather than
    inherited because the parent's loop assumes a 2-argument loss.
    """

    def __init__(
        self,
        mol_info,
        config,
        init_key,
        c_hat_ground,
        fci_ref,
        lambda_penalty: float = 1.0,
        penalty_mode: str = "cos2",
    ):
        """Construct a basis-resolved NES-VMC driver.

        Args:
            mol_info: :class:`OmegaQMC.utils.Mole_custom`.
            config: ansatz config.
            init_key: PRNG key for excited-state parameter init.
            c_hat_ground: ``(n_det,)`` CS-recovered ground-state CI
                vector in the natural-orbital basis. Sign convention
                must match ``fci_ref``.
            fci_ref: the dict produced by
                :func:`OmegaQMC.cs.reference.compute_fci_reference`,
                supplying the candidate set, natural orbitals, and
                ``nelec`` for ``Psi_synth^(0)`` evaluation.
            lambda_penalty: weight on the basis-resolved penalty term.
            penalty_mode: one of ``"cos2"`` (default; scale-invariant
                cos^2 of the real-space angle, post-hoc CI overlap stalls
                at ~0.9 due to in-batch estimator cheating) or
                ``"abs_overlap"`` (route-(a) absolute-overlap-squared
                penalty with stop-gradient on the scale-normalising
                denominator; decouples the gradient from the noisy
                denominator while preserving scale invariance).
        """
        super().__init__(mol_info, config, init_key)
        self.c_hat_ground = np.asarray(c_hat_ground, dtype=np.float64)
        self.fci_ref = fci_ref
        self.lambda_penalty = float(lambda_penalty)
        if penalty_mode not in ("cos2", "abs_overlap"):
            raise ValueError(
                f"penalty_mode must be 'cos2' or 'abs_overlap', "
                f"got {penalty_mode!r}",
            )
        self.penalty_mode = penalty_mode

        from .cs.estimators import (
            evaluate_orbitals_on_walkers, evaluate_ci_wavefunction,
        )

        n_alpha, n_beta = fci_ref["nelec"]
        candidate = fci_ref["candidate_set"]
        no_coeff = fci_ref["no_coeff_ao"]

        def evaluate_psi_synth(walkers_np):
            """``Psi_synth^(0)(R)`` at each walker (numpy, outside JIT).

            ``walkers_np`` shape ``(K_s, n_elec, 3)`` in OmegaQMC
            interleaved layout. Returns a numpy array of shape
            ``(K_s,)``.
            """
            orb = evaluate_orbitals_on_walkers(
                mol_info, np.asarray(walkers_np), no_coeff,
                convention="interleaved",
                n_alpha=n_alpha, n_beta=n_beta,
            )
            return evaluate_ci_wavefunction(
                orb, candidate, self.c_hat_ground, n_alpha, n_beta,
            )
        self.evaluate_psi_synth = evaluate_psi_synth

        log_psi_signed = self.log_psi.signed
        nuc_crds = self.nuc_crds
        compute_batch_energy = self.compute_batch_energy

        def signed_psi_one(walker, params):
            sign, log_amp = log_psi_signed(walker, nuc_crds, params)
            return sign * jnp.exp(log_amp)

        @jax.jit
        def loss_fn_basis_cos2(params, batch_walkers, batch_psi_synth):
            energies = compute_batch_energy(batch_walkers, params)
            e_loss = 0.2 * energies.mean() + 0.8 * energies.std()
            psi_1 = jax.vmap(signed_psi_one, in_axes=(0, None))(
                batch_walkers, params,
            )
            ratio = batch_psi_synth / psi_1
            # Scale-invariant cos^2(angle) between Psi_synth and Psi_NN:
            # penalty = (E[r])^2 / E[r^2]. This cancels the arbitrary
            # overall scale of Psi_NN and equals |<Psi_synth|Psi_NN>|^2
            # / (<Psi_synth|Psi_synth> * <Psi_NN|Psi_NN>). For normalised
            # Psi_synth (Sum |c_hat^(0)|^2 = 1) it directly tracks the
            # CI-vector overlap. The +1e-30 guards against the
            # degenerate Psi_NN = 0 case at all sampled walkers.
            cos2 = (ratio.mean()) ** 2 / (jnp.mean(ratio ** 2) + 1e-30)
            return e_loss + lambda_penalty * cos2

        @jax.jit
        def loss_fn_basis_abs(params, batch_walkers, batch_psi_synth):
            # Route-(a) absolute-overlap penalty with stop-gradient on
            # the scale-normalising denominator. The denominator value
            # (1/Z_1 estimate) is still used to make the penalty scale-
            # invariant in Psi_1, but its *gradient* is suppressed so
            # the optimiser cannot exploit the cos^2 denominator's
            # flat-manifold pathway. This isolates Adam's update to the
            # numerator (squared mean overlap), which directly maps to
            # |<Psi_synth | Psi_1>|^2 / Z_1.
            energies = compute_batch_energy(batch_walkers, params)
            e_loss = 0.2 * energies.mean() + 0.8 * energies.std()
            psi_1 = jax.vmap(signed_psi_one, in_axes=(0, None))(
                batch_walkers, params,
            )
            ratio = batch_psi_synth / psi_1
            denom = jax.lax.stop_gradient(jnp.mean(ratio ** 2) + 1e-30)
            penalty = (ratio.mean()) ** 2 / denom
            return e_loss + lambda_penalty * penalty

        if self.penalty_mode == "cos2":
            self.loss_fn_basis = loss_fn_basis_cos2
        else:
            self.loss_fn_basis = loss_fn_basis_abs

    def __call__(
        self,
        rng_key,
        num_iters: int = 200,
        num_epochs: int = 10,
        num_walkers: int = 256,
        num_steps_per_block: int = 100,
        num_steps_decorr: int = 1,
        num_sample_blocks: int = 2,
        num_blocks_equil: int = 5,
        mc_timestep: float = 0.1,
        lr: float = 1e-2,
        train_split: float = 0.8,
        batch_size: int = 128,
        verbose: int = 1,
        prefix: str = "nesopt",
    ):
        """Adam loop with the basis-resolved penalty.

        The synthetic Psi_synth^(0) values are pre-computed at every
        sampled walker each outer iteration (numpy/PySCF, outside JIT)
        and passed into the JIT'd loss function.
        """
        params = self.init_params
        optimizer = optax.adam(learning_rate=lr)
        opt_state = optimizer.init(params)
        mc_stepsize = (3 * mc_timestep) ** 0.5

        rng_key, init_walker_key = jax.random.split(rng_key)
        walkers = self.initialize_walkers(init_walker_key, num_walkers)

        if verbose >= 1:
            print(f"basis-resolved NES-VMC: lambda={self.lambda_penalty}")
            print(f"  equilibrating {num_blocks_equil} blocks...")
        (rng_key, walkers, mc_stepsize, _), _ = self.run_equilibration(
            rng_key, walkers, mc_stepsize, params,
            num_blocks_equil, num_steps_per_block,
        )

        chkpt_path = f"{prefix}.chk.h5"
        for iteration in range(num_iters):
            # (a) Sample fresh walkers from |Psi_1|^2
            all_samples = []
            for _ in range(num_sample_blocks):
                (rng_key, walkers, mc_stepsize, _), _ = self.run_production(
                    rng_key, walkers, mc_stepsize, params,
                    num_steps_per_block, num_steps_decorr,
                )
                all_samples.append(walkers)
            sampled = jnp.vstack(all_samples).reshape(-1, self.nelec, 3)
            n_samples = sampled.shape[0]

            # (b) Pre-compute Psi_synth^(0)(R) at sampled walkers
            psi_synth_all = self.evaluate_psi_synth(np.asarray(sampled))
            psi_synth_all = jnp.asarray(psi_synth_all)

            # (c) Permute + split
            rng_key, perm_key = jax.random.split(rng_key)
            idx = jax.random.permutation(perm_key, jnp.arange(n_samples))
            n_train = int(train_split * n_samples)
            train_w = sampled[idx[:n_train]]
            train_psi_synth = psi_synth_all[idx[:n_train]]
            valid_w = sampled[idx[n_train:]]

            # (d) Adam epochs
            epoch_losses = []
            for _ in range(num_epochs):
                for si in range(0, n_train, batch_size):
                    ei = min(si + batch_size, n_train)
                    batch = train_w[si:ei]
                    batch_synth = train_psi_synth[si:ei]
                    loss, grads = jax.value_and_grad(
                        self.loss_fn_basis,
                    )(params, batch, batch_synth)
                    updates, opt_state = optimizer.update(
                        grads, opt_state, params,
                    )
                    params = optax.apply_updates(params, updates)
                    epoch_losses.append(loss)

            # (e) Diagnostics on validation
            v_energies = []
            v_overlap_sum = 0.0
            v_psi_synth_all = jnp.asarray(
                self.evaluate_psi_synth(np.asarray(valid_w)),
            )
            n_valid = valid_w.shape[0]
            for si in range(0, n_valid, batch_size):
                ei = min(si + batch_size, n_valid)
                v_energies.append(
                    self.compute_batch_energy(valid_w[si:ei], params),
                )
            all_e = jnp.concatenate(v_energies)
            iter_e = float(all_e.mean())
            iter_err = float(all_e.std()) / max(1, all_e.size) ** 0.5
            log_psi_signed = self.log_psi.signed
            psi_1_valid = jax.vmap(
                lambda w: (log_psi_signed(w, self.nuc_crds, params))
            )(valid_w)
            sgn, lam = psi_1_valid
            psi_1_arr = sgn * jnp.exp(lam)
            r = v_psi_synth_all / psi_1_arr
            mean_r = float(jnp.mean(r))
            mean_r2 = float(jnp.mean(r ** 2))
            cos2_est = mean_r ** 2 / max(mean_r2, 1e-30)

            if verbose >= 1:
                iter_loss = float(jnp.array(epoch_losses).mean())
                print(f"Iter {iteration:5d} | "
                      f"E = {iter_e:.6f} +/- {iter_err:.5f} | "
                      f"cos^2 = {cos2_est:.5f} | "
                      f"Loss: {iter_loss:.6f}")

            # (f) Save checkpoint
            if os.path.exists(chkpt_path):
                os.rename(chkpt_path, f"{prefix}.{iteration}.h5")
            save_nn_checkpoint(
                chkpt_path, params, iteration,
                self.config_name, self.mol_info, energy=iter_e,
            )

        return params, {"energy": {"mean": iter_e, "stderr": iter_err}}


def get_vmcopt_nn_nes_basis_func(
    mol_info,
    config,
    init_key,
    c_hat_ground,
    fci_ref,
    lambda_penalty: float = 1.0,
    init_from_ground_checkpoint: str = None,
    penalty_mode: str = "cos2",
):
    """Factory for the basis-resolved NES-VMC driver.

    Unlike :func:`get_vmcopt_nn_nes_func`, this variant requires the
    already-recovered ground-state CI vector ``c_hat_ground`` and the
    FCI reference dict, both produced by the standard CS pipeline.
    The user is expected to run the CS recovery on the ground state
    first, then pass the result here.

    ``init_from_ground_checkpoint`` (optional): if supplied, the
    excited-state NN is initialised from the ground-state parameters
    rather than from random ``init_key``. This makes the initial
    overlap appreciable, giving the penalty term enough magnitude to
    drive a meaningful gradient against the energy term. Strongly
    recommended for the basis-resolved variant, where a random
    initialisation has accidentally small basis-projected overlap and
    Adam will not feel any pressure to orthogonalise.

    ``penalty_mode``: ``"cos2"`` (scale-invariant cos^2 of real-space
    angle) or ``"abs_overlap"`` (route-(a) absolute-overlap-squared with
    stop-gradient on the scale-normalising denominator).
    """
    driver = _VMCOptDriverNN_NES_Basis(
        mol_info, config, init_key,
        c_hat_ground=c_hat_ground,
        fci_ref=fci_ref,
        lambda_penalty=lambda_penalty,
        penalty_mode=penalty_mode,
    )
    if init_from_ground_checkpoint is not None:
        ground_params, _meta = load_nn_checkpoint(
            init_from_ground_checkpoint, driver.init_params,
        )
        driver.init_params = ground_params
    return driver


# ===========================================================================
# CI-overlap NES-VMC variant (route 3)
# ===========================================================================
#
# The cos^2 penalty of _VMCOptDriverNN_NES_Basis targets the *real-space*
# cosine between Psi_synth and Psi_NN. This is not the same as the
# *CI-vector* cosine between c_hat^(0) and c_hat^(1), because c_hat^(1)_I
# = E[D_I/Psi_1] has its own (basis-incompleteness) bias relative to
# E[Psi_synth/Psi_1] = sum_I c_hat^(0)_I × c_hat^(1)_I. When Psi_NN
# extends beyond the basis, the in-basis c_hat vectors of the ground
# and excited states can be highly aligned even when the real-space
# overlap is small. To penalise the CI-vector cosine directly we need
# both the numerator (sum c^(0)_I × c^(1)_I) and a CI-vector denominator
# (sum (c^(1)_I)^2), each evaluated per-Adam-step on the same batch.
#
# This requires pre-computing the matrix M[k, I] = D_I^norm(R_k) at every
# sampled walker for every candidate determinant I, rather than the
# scalar Psi_synth(R_k) used by the previous variant. The per-Adam-step
# loss then divides through by Psi_1(R_k) to get f_I^(1)[k, I],
# takes column means to get c_raw_I, and forms the CI-cosine.


class _VMCOptDriverNN_NES_CIOverlap(_VMCOptDriverNN_IRAdam):
    """NES-VMC with a direct CI-vector cosine penalty.

    Penalty: ``( sum_I c_hat^(0)_I × c_raw^(1)_I )^2 / sum_I (c_raw^(1)_I)^2``
    where ``c_raw^(1)_I = (1/K) sum_k D_I^norm(R_k) / Psi_1(R_k)`` is the
    per-coefficient sample mean over the current batch. This is the
    squared cosine between the normalised CI vectors directly, and it
    cannot be cheated by the real-space sign-cancellation pathway that
    the cos^2 variant suffered from.

    Memory cost per outer iter: ``(n_walkers_per_iter) × n_det × 8`` bytes
    for the pre-computed D_I matrix. For H2/cc-pVDZ with n_det = 10 and
    n_walkers = 2048, that is 160 KB. For H4 cc-pVDZ with
    n_det = 10^4 it is 160 MB. For larger n_det this variant becomes
    impractical and the user should fall back to either truncated
    candidate sets or the real-space variant.
    """

    def __init__(
        self,
        mol_info,
        config,
        init_key,
        c_hat_ground,
        fci_ref,
        lambda_penalty: float = 1.0,
    ):
        super().__init__(mol_info, config, init_key)
        self.c_hat_ground = np.asarray(c_hat_ground, dtype=np.float64)
        self.fci_ref = fci_ref
        self.lambda_penalty = float(lambda_penalty)

        from .cs.estimators import (
            evaluate_orbitals_on_walkers,
            _normalization,
        )
        from .cs.estimators import f_I_matrix  # noqa: F401

        n_alpha, n_beta = fci_ref["nelec"]
        candidate = fci_ref["candidate_set"]
        no_coeff = fci_ref["no_coeff_ao"]
        n_norm = _normalization(n_alpha, n_beta)
        n_det = len(candidate)

        def evaluate_D_matrix(walkers_np):
            """Compute D_I^norm(R_k) for every (walker, candidate det).

            Returns numpy array shape ``(K_s, n_det)``. Heavy numerical
            work is done once per outer iteration (outside JIT).
            """
            orb = evaluate_orbitals_on_walkers(
                mol_info, np.asarray(walkers_np), no_coeff,
                convention="interleaved",
                n_alpha=n_alpha, n_beta=n_beta,
            )
            orb_a = orb[:, :n_alpha, :]
            orb_b = orb[:, n_alpha:, :]
            K = orb.shape[0]
            D = np.zeros((K, n_det), dtype=np.float64)
            for i, (occ_a, occ_b) in enumerate(candidate):
                M_a = orb_a[:, :, list(occ_a)]
                M_b = orb_b[:, :, list(occ_b)]
                D[:, i] = n_norm * np.linalg.det(M_a) * np.linalg.det(M_b)
            return D
        self.evaluate_D_matrix = evaluate_D_matrix

        log_psi_signed = self.log_psi.signed
        nuc_crds = self.nuc_crds
        compute_batch_energy = self.compute_batch_energy
        c_hat_g_j = jnp.asarray(self.c_hat_ground)

        def signed_psi_one(walker, params):
            sign, log_amp = log_psi_signed(walker, nuc_crds, params)
            return sign * jnp.exp(log_amp)

        @jax.jit
        def loss_fn_ci(params, batch_walkers, batch_D):
            energies = compute_batch_energy(batch_walkers, params)
            e_loss = 0.2 * energies.mean() + 0.8 * energies.std()
            psi_1 = jax.vmap(signed_psi_one, in_axes=(0, None))(
                batch_walkers, params,
            )
            # f_I^(1) [k, I] = D_I^norm(R_k) / Psi_1(R_k)
            f_I = batch_D / psi_1[:, None]
            # Per-coefficient sample mean: c_raw^(1)_I = E[f_I]
            c_raw = jnp.mean(f_I, axis=0)
            # CI-vector cosine squared
            numerator = jnp.sum(c_hat_g_j * c_raw) ** 2
            denominator = jnp.sum(c_raw ** 2) + 1e-30
            cos2_ci = numerator / denominator
            return e_loss + lambda_penalty * cos2_ci
        self.loss_fn_ci = loss_fn_ci

    def __call__(
        self,
        rng_key,
        num_iters: int = 200,
        num_epochs: int = 1,
        num_walkers: int = 512,
        num_steps_per_block: int = 100,
        num_steps_decorr: int = 1,
        num_sample_blocks: int = 4,
        num_blocks_equil: int = 5,
        mc_timestep: float = 0.1,
        lr: float = 1e-2,
        train_split: float = 0.8,
        batch_size: int = None,
        verbose: int = 1,
        prefix: str = "nesopt_ci",
    ):
        """Adam loop with the CI-overlap penalty.

        ``batch_size = None`` (default) uses the entire training set in
        each Adam step, which is necessary for low-variance estimation
        of the per-coefficient sample means.
        """
        params = self.init_params
        optimizer = optax.adam(learning_rate=lr)
        opt_state = optimizer.init(params)
        mc_stepsize = (3 * mc_timestep) ** 0.5

        rng_key, init_walker_key = jax.random.split(rng_key)
        walkers = self.initialize_walkers(init_walker_key, num_walkers)

        if verbose >= 1:
            print(f"CI-overlap NES-VMC: lambda={self.lambda_penalty}, "
                  f"|candidate set|={len(self.fci_ref['candidate_set'])}")
            print(f"  equilibrating {num_blocks_equil} blocks...")
        (rng_key, walkers, mc_stepsize, _), _ = self.run_equilibration(
            rng_key, walkers, mc_stepsize, params,
            num_blocks_equil, num_steps_per_block,
        )

        chkpt_path = f"{prefix}.chk.h5"
        for iteration in range(num_iters):
            all_samples = []
            for _ in range(num_sample_blocks):
                (rng_key, walkers, mc_stepsize, _), _ = self.run_production(
                    rng_key, walkers, mc_stepsize, params,
                    num_steps_per_block, num_steps_decorr,
                )
                all_samples.append(walkers)
            sampled = jnp.vstack(all_samples).reshape(-1, self.nelec, 3)
            n_samples = sampled.shape[0]

            D_all = self.evaluate_D_matrix(np.asarray(sampled))
            D_all = jnp.asarray(D_all)

            rng_key, perm_key = jax.random.split(rng_key)
            idx = jax.random.permutation(perm_key, jnp.arange(n_samples))
            n_train = int(train_split * n_samples)
            train_w = sampled[idx[:n_train]]
            train_D = D_all[idx[:n_train]]
            valid_w = sampled[idx[n_train:]]

            effective_batch = batch_size if batch_size is not None else n_train

            epoch_losses = []
            for _ in range(num_epochs):
                for si in range(0, n_train, effective_batch):
                    ei = min(si + effective_batch, n_train)
                    loss, grads = jax.value_and_grad(
                        self.loss_fn_ci,
                    )(params, train_w[si:ei], train_D[si:ei])
                    updates, opt_state = optimizer.update(
                        grads, opt_state, params,
                    )
                    params = optax.apply_updates(params, updates)
                    epoch_losses.append(loss)

            # Diagnostic: CI overlap on validation (large batch, robust)
            v_D = jnp.asarray(self.evaluate_D_matrix(np.asarray(valid_w)))
            psi_1_v = jax.vmap(
                lambda w: self.log_psi.signed(w, self.nuc_crds, params),
            )(valid_w)
            sgn_v, lam_v = psi_1_v
            psi_1_arr = sgn_v * jnp.exp(lam_v)
            f_I_v = v_D / psi_1_arr[:, None]
            c_raw_v = jnp.mean(f_I_v, axis=0)
            num_v = float(jnp.sum(jnp.asarray(self.c_hat_ground) * c_raw_v) ** 2)
            den_v = float(jnp.sum(c_raw_v ** 2)) + 1e-30
            ci_cos2 = num_v / den_v

            v_energies = []
            for si in range(0, valid_w.shape[0], 256):
                ei = min(si + 256, valid_w.shape[0])
                v_energies.append(
                    self.compute_batch_energy(valid_w[si:ei], params),
                )
            all_e = jnp.concatenate(v_energies)
            iter_e = float(all_e.mean())
            iter_err = float(all_e.std()) / max(1, all_e.size) ** 0.5

            if verbose >= 1:
                iter_loss = float(jnp.array(epoch_losses).mean())
                print(f"Iter {iteration:5d} | "
                      f"E = {iter_e:.6f} +/- {iter_err:.5f} | "
                      f"CI cos^2 = {ci_cos2:.5f} | "
                      f"Loss: {iter_loss:.6f}")

            if os.path.exists(chkpt_path):
                os.rename(chkpt_path, f"{prefix}.{iteration}.h5")
            save_nn_checkpoint(
                chkpt_path, params, iteration,
                self.config_name, self.mol_info, energy=iter_e,
            )

        return params, {"energy": {"mean": iter_e, "stderr": iter_err}}


def get_vmcopt_nn_nes_ci_func(
    mol_info,
    config,
    init_key,
    c_hat_ground,
    fci_ref,
    lambda_penalty: float = 1.0,
    init_from_ground_checkpoint: str = None,
):
    """Factory for the CI-overlap NES-VMC driver (route 3)."""
    driver = _VMCOptDriverNN_NES_CIOverlap(
        mol_info, config, init_key,
        c_hat_ground=c_hat_ground,
        fci_ref=fci_ref,
        lambda_penalty=lambda_penalty,
    )
    if init_from_ground_checkpoint is not None:
        ground_params, _meta = load_nn_checkpoint(
            init_from_ground_checkpoint, driver.init_params,
        )
        driver.init_params = ground_params
    return driver
