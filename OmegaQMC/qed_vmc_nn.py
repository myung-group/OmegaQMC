"""QED-VMC driver for cavity-coupled NN trial wavefunctions.

Joint discrete-continuous Metropolis-Hastings sampler over
:math:`(\\mathbf r, n)` where :math:`\\mathbf r` are 3N electronic
coordinates (continuous) and :math:`n \\in \\{0, 1, \\ldots, N_{\\text{ph,max}}\\}`
is a photon Fock index (discrete). Local energy is the dipole-gauge
Pauli-Fierz Hamiltonian via
:func:`OmegaQMC.psi.nn.qed_physics.pauli_fierz_local_energy`.

Mirrors the architecture of :mod:`OmegaQMC.vmc_nn` but stripped of
PGCS/fragment plumbing and gradient calculation (those will be added
later if needed). Targets the **mango** GH200 cluster for production
runs.

The Metropolis step alternates two move types per step (Tang 2025
Algorithm 1):

  * 50% probability: Gaussian random walk in :math:`\\mathbf r` at fixed
    :math:`n`. Acceptance probability
    :math:`\\min(1, |\\Psi(r',n)/\\Psi(r,n)|^2)`.
  * 50% probability: ±1 random walk in :math:`n` at fixed :math:`\\mathbf r`.
    Boundary-clipped at :math:`[0, N_{\\text{ph,max}}]` (out-of-bounds proposals
    auto-reject). Acceptance probability
    :math:`\\min(1, |\\Psi(r,n')/\\Psi(r,n)|^2)`.

References
----------
- Tang et al. 2025, arXiv:2503.15644 (Algorithm 1: discrete-continuous MH).
- Haugland et al. 2021, arXiv:2012.01080 (target benchmark for validation).
"""
from __future__ import annotations

import sys
from functools import partial
from typing import Callable

import jax
import jax.numpy as jnp

from .utils import Mole_custom, do_binning_analysis
from .psi.nn.qed_adapter import (
    make_qed_nn_log_psi,
    make_qed_nn_log_psi_n_aware,
    make_qed_nn_log_psi_hybrid,
    make_qed_nn_log_psi_signed_hybrid,
    make_qed_nn_log_psi_tang_native,
)
from .psi.nn.qed_physics import (
    pauli_fierz_local_energy,
    pauli_fierz_local_energy_signed,
    coulomb_nn,
)
from .constants import MIN_DIST_THRESHOLD


__all__ = [
    "get_qed_vmc_nn_func",
]


# Default Metropolis adaptation knobs (mirror vmc_nn.py).
_TARGET_ACCEPT_R = 0.5
_TARGET_ACCEPT_N = 0.5
_STEP_ADAPT_RATE = 0.05


class _QEDVMCDriverNN:
    """Internal driver class. Use :func:`get_qed_vmc_nn_func`."""

    def __init__(
        self,
        mol_info: Mole_custom,
        config,
        init_key,
        *,
        omega: float,
        coupling_vec,
        alpha_init: float = 0.0,
        alpha_train: bool = False,
        nph_max: int = 10,
        n_aware: bool = False,
        fock_hidden_dim: int = 64,
        arch: str | None = None,
        complex_psi: bool = False,
    ):
        self.mol_info = mol_info
        self.omega = float(omega)
        self.coupling_vec = jnp.asarray(coupling_vec, dtype=jnp.float64)
        self.nph_max = int(nph_max)
        self.complex_psi = complex_psi

        # Resolve architecture flag. ``arch`` (if given) takes precedence;
        # otherwise fall back to the legacy ``n_aware`` bool.
        if arch is None:
            arch = "n_aware" if n_aware else "factorized"
        if arch not in (
            "factorized", "n_aware", "hybrid",
            "signed_hybrid", "tang_native",
        ):
            raise ValueError(
                f"Unknown arch={arch!r}; expected "
                "'factorized' | 'n_aware' | 'hybrid' | "
                "'signed_hybrid' | 'tang_native'.",
            )
        self.arch = arch
        self.n_aware = (arch == "n_aware")
        self.is_signed = arch in ("signed_hybrid", "tang_native")

        # Geometry caches.
        self.nuc_crds = jnp.asarray(mol_info.coords, dtype=jnp.float64)
        self.charges = jnp.asarray(mol_info.charges, dtype=jnp.float64)
        self.nelec = mol_info.n_up + mol_info.n_down
        self.enuc = float(coulomb_nn(self.nuc_crds, self.charges))

        # Build the joint electron-photon ansatz.
        # Three architectures available:
        #   * factorized        — Phase 2b: Ψ_e(r)·⟨n|α⟩
        #   * n_aware           — Phase 2f-1 Path A: Ψ_e(r) + Fock head
        #                         (no envelope; matches Tang's pure form
        #                         conceptually). Slow convergence.
        #   * hybrid            — Phase 2f-1 Path B: Ψ_e(r)·⟨n|α⟩ + Fock head
        #                         envelope provides the right Fock prior;
        #                         head adds (r,n) entanglement. Targets
        #                         our locked-thesis ultrastrong-coupling regime.
        if arch == "n_aware":
            log_psi, init_params, graphdef = make_qed_nn_log_psi_n_aware(
                config, mol_info, init_key,
                omega=self.omega,
                coupling_vec=self.coupling_vec,
                nph_max=self.nph_max,
                fock_hidden_dim=fock_hidden_dim,
            )
            self._log_psi_signed = None
        elif arch == "hybrid":
            log_psi, init_params, graphdef = make_qed_nn_log_psi_hybrid(
                config, mol_info, init_key,
                omega=self.omega,
                coupling_vec=self.coupling_vec,
                nph_max=self.nph_max,
                fock_hidden_dim=fock_hidden_dim,
                alpha_init=alpha_init,
                alpha_train=alpha_train,
            )
            self._log_psi_signed = None
        elif arch == "signed_hybrid":
            (log_psi_signed, init_params, graphdef
             ) = make_qed_nn_log_psi_signed_hybrid(
                config, mol_info, init_key,
                omega=self.omega,
                coupling_vec=self.coupling_vec,
                nph_max=self.nph_max,
                fock_hidden_dim=fock_hidden_dim,
                alpha_init=alpha_init,
                alpha_train=alpha_train,
            )
            self._log_psi_signed = log_psi_signed
            # Sampling / Jacobian uses log magnitude only (sign doesn't
            # affect |Ψ|² walker distribution and the SR Jacobian wants
            # ∂log|Ψ|/∂params).
            def log_psi(elec_crds, nuc_crds, n, params):
                log_mag, _sign = log_psi_signed(
                    elec_crds, nuc_crds, n, params,
                )
                return log_mag
        elif arch == "tang_native":
            # Phase 2g: per-electron one-hot(n) injected inside the GNN
            # (Tang 2025 Sec. II.C). Native Slater sign — no Fock head,
            # no coherent-state envelope.
            (log_psi_signed, init_params, graphdef
             ) = make_qed_nn_log_psi_tang_native(
                config, mol_info, init_key,
                nph_max=self.nph_max,
            )
            self._log_psi_signed = log_psi_signed

            def log_psi(elec_crds, nuc_crds, n, params):
                log_mag, _sign = log_psi_signed(
                    elec_crds, nuc_crds, n, params,
                )
                return log_mag
        else:  # factorized
            log_psi, init_params, graphdef = make_qed_nn_log_psi(
                config, mol_info, init_key,
                omega=self.omega,
                coupling_vec=self.coupling_vec,
                alpha_init=alpha_init,
                alpha_train=alpha_train,
            )
            self._log_psi_signed = None
        self.log_psi = log_psi
        self.params = init_params
        self.graphdef = graphdef

        # JIT-compiled per-walker local-energy estimator.
        omega_loc = self.omega
        cv_loc = self.coupling_vec
        nph_loc = self.nph_max
        nuc_loc = self.nuc_crds
        charges_loc = self.charges
        enuc_loc = self.enuc

        if self.is_signed:
            log_psi_signed_for_le = self._log_psi_signed
            complex_psi_for_le = self.complex_psi

            @jax.jit
            def local_energy_one(elec_crds, n, params):
                return pauli_fierz_local_energy_signed(
                    log_psi_signed_for_le,
                    params,
                    elec_crds,
                    n,
                    nuc_loc,
                    charges_loc,
                    omega_loc,
                    cv_loc,
                    nph_loc,
                    enuc=enuc_loc,
                    complex_psi=complex_psi_for_le,
                )
        else:
            @jax.jit
            def local_energy_one(elec_crds, n, params):
                return pauli_fierz_local_energy(
                    log_psi,
                    params,
                    elec_crds,
                    n,
                    nuc_loc,
                    charges_loc,
                    omega_loc,
                    cv_loc,
                    nph_loc,
                    enuc=enuc_loc,
                )

        self._local_energy_one = local_energy_one
        self._local_energy_batch = jax.jit(
            jax.vmap(local_energy_one, in_axes=(0, 0, None)),
        )
        self._log_psi_batch = jax.jit(
            jax.vmap(
                lambda r, n: log_psi(r, nuc_loc, n, self.params),
                in_axes=(0, 0),
            ),
        )

        # JIT-compiled joint Metropolis-Hastings step (single walker).
        self._mh_step = jax.jit(self._build_mh_step())
        self._mh_step_batch = jax.jit(jax.vmap(
            self._mh_step,
            in_axes=(0, 0, 0, None, None, None),
        ))

    # ----------------------------------------------------------------
    # MH builder
    # ----------------------------------------------------------------
    def _build_mh_step(self) -> Callable:
        log_psi = self.log_psi
        nuc = self.nuc_crds
        nph_max = self.nph_max

        def mh_step(rng_key, r, n, params, sigma_r, p_n):
            """Single joint MH step.

            Returns ``(r_new, n_new, accept_r_flag, accept_n_flag,
            branch_was_r)`` per walker. The flags are only meaningful for
            the chosen branch; the other branch's flag is ``False`` by
            convention.
            """
            del p_n  # reserved for tunable discrete proposal width
            (key_coin, key_rprop, key_racc,
             key_nprop, key_nacc) = jax.random.split(rng_key, 5)

            # Pick branch.
            branch_r = jax.random.bernoulli(key_coin, 0.5)

            # Current log Ψ (used by both branches).
            log_old = log_psi(r, nuc, n, params)

            # ---- R branch ----
            r_prop = r + sigma_r * jax.random.normal(
                key_rprop, r.shape,
            )
            # Reject moves that bring electrons too close to nuclei or
            # to one another (matches vmc_nn convention).
            diffs_ee = (
                r_prop[:, None, :] - r_prop[None, :, :]
            )
            dists_ee = jnp.sqrt(jnp.sum(diffs_ee ** 2, axis=-1) + 1e-30)
            n_e = r_prop.shape[0]
            iu = jnp.triu_indices(n_e, k=1)
            dists_ee_strict = dists_ee[iu[0], iu[1]]
            dists_en = jnp.sqrt(
                jnp.sum((r_prop[:, None, :] - nuc[None, :, :]) ** 2, axis=-1)
            )
            valid_r = (
                (dists_ee_strict.min() > MIN_DIST_THRESHOLD)
                & (dists_en.min() > MIN_DIST_THRESHOLD)
            )
            log_new_r = log_psi(r_prop, nuc, n, params)
            accept_r = (
                jax.random.uniform(key_racc)
                < jnp.exp(2.0 * (log_new_r - log_old))
            ) & valid_r

            # ---- N branch ----
            direction = 2 * jax.random.bernoulli(key_nprop, 0.5).astype(
                jnp.int32,
            ) - 1
            n_prop = n + direction
            in_bounds_n = (n_prop >= 0) & (n_prop <= nph_max)
            # Use clipped n for the log_psi call; out-of-bounds will be
            # rejected via valid_n below regardless.
            n_eval = jnp.clip(n_prop, 0, nph_max)
            log_new_n = log_psi(r, nuc, n_eval, params)
            accept_n = (
                jax.random.uniform(key_nacc)
                < jnp.exp(2.0 * (log_new_n - log_old))
            ) & in_bounds_n

            # ---- Combine ----
            r_new = jnp.where(branch_r & accept_r, r_prop, r)
            n_new = jnp.where(
                (~branch_r) & accept_n,
                n_eval,
                n,
            )
            return r_new, n_new, accept_r, accept_n, branch_r

        return mh_step

    # ----------------------------------------------------------------
    # Walker initialization
    # ----------------------------------------------------------------
    def initialize_walkers(self, rng_key, num_walkers):
        """Place electrons near nuclei; sample photon Fock index from
        the analytical coherent-state distribution :math:`|\\langle n|\\alpha\\rangle|^2`.

        For α=0 (default) this collapses to the photon vacuum :math:`n=0`.

        Returns:
            ``(elec_crds, photon_n)`` where ``elec_crds`` has shape
            ``(num_walkers, n_elec, 3)`` and ``photon_n`` is
            ``(num_walkers,)`` of int32.
        """
        key_e, key_p = jax.random.split(rng_key)

        # Electron init: same as vmc_nn — distribute electrons across
        # nuclear positions weighted by nuclear charge.
        idx_cnt = []
        for ia, iz in enumerate(self.charges):
            idx_cnt.extend([ia] * int(iz))
        total = self.nelec
        while len(idx_cnt) < total:
            idx_cnt.append(0)
        idx_cnt = jnp.array(idx_cnt[:total])
        centers = self.nuc_crds[idx_cnt]
        elec_crds = (
            centers[None, :, :]
            + 0.05 * jax.random.normal(
                key_e, (num_walkers, total, 3),
            )
        )

        # Photon init: sample n from |⟨n|α⟩|² (Poisson(|α|²)).
        # Use float values from coherent-state amplitude for the discrete
        # PMF and sample.
        if "alpha" in self.params:
            alpha_val = float(self.params["alpha"])
        else:
            alpha_val = 0.0
        # Build PMF over [0, nph_max]
        pmf = jnp.array(
            [
                jnp.exp(2.0 * (-0.5 * alpha_val ** 2
                                + n * jnp.log(jnp.abs(alpha_val) + 1e-30)
                                - 0.5 * jax.lax.lgamma(float(n + 1))))
                for n in range(self.nph_max + 1)
            ]
        )
        pmf = pmf / jnp.sum(pmf)
        photon_n = jax.random.categorical(
            key_p,
            jnp.log(pmf + 1e-30),
            shape=(num_walkers,),
        ).astype(jnp.int32)
        return elec_crds, photon_n

    # ----------------------------------------------------------------
    # Production run
    # ----------------------------------------------------------------
    def __call__(
        self,
        rng_key,
        num_walkers: int = 256,
        num_steps_per_block: int = 50,
        num_blocks: int = 100,
        num_blocks_equil: int = 20,
        mc_timestep: float = 0.1,
        verbose: int = 1,
    ):
        """Execute a fixed-parameter QED-VMC run.

        Returns:
            dict with ``E_mean``, ``E_serr``, ``E_blocks``,
            ``n_photon_mean``, ``acceptance_r``, ``acceptance_n``.
        """
        if isinstance(rng_key, int):
            rng_key = jax.random.key(rng_key)

        key_walk, key_run = jax.random.split(rng_key)
        elec, n_ph = self.initialize_walkers(key_walk, num_walkers)
        sigma_r = float(mc_timestep)
        # Future-proofing for tunable discrete step
        p_n = 0.5

        block_energies = []
        n_photon_means = []
        accept_r_running = 0.0
        accept_n_running = 0.0
        total_r_attempts = 0
        total_n_attempts = 0

        params = self.params
        total_blocks = num_blocks_equil + num_blocks

        for blk in range(total_blocks):
            block_E = []
            block_n = []
            for step in range(num_steps_per_block):
                key_run, key_step = jax.random.split(key_run)
                step_keys = jax.random.split(key_step, num_walkers)
                elec, n_ph, acc_r, acc_n, was_r = self._mh_step_batch(
                    step_keys, elec, n_ph, params, sigma_r, p_n,
                )
                # Acceptance bookkeeping
                accept_r_running += float(jnp.sum(acc_r & was_r))
                total_r_attempts += float(jnp.sum(was_r))
                accept_n_running += float(jnp.sum(acc_n & ~was_r))
                total_n_attempts += float(jnp.sum(~was_r))

            # End of block: estimate local energies. For complex_psi,
            # e_loc is a complex scalar; the physical energy is the
            # mean of its real part.
            e_loc = self._local_energy_batch(elec, n_ph, params)
            if self.complex_psi:
                block_E.append(float(jnp.mean(jnp.real(e_loc))))
            else:
                block_E.append(float(jnp.mean(e_loc)))
            block_n.append(float(jnp.mean(n_ph.astype(jnp.float64))))

            block_energies.append(block_E[-1])
            n_photon_means.append(block_n[-1])

            # Adapt sigma_r toward target acceptance during equilibration.
            if blk < num_blocks_equil and total_r_attempts > 0:
                cur_acc = accept_r_running / max(total_r_attempts, 1.0)
                sigma_r *= 1.0 + _STEP_ADAPT_RATE * (cur_acc - _TARGET_ACCEPT_R)

            if verbose:
                tag = "equil" if blk < num_blocks_equil else "prod"
                print(
                    f"[QED-VMC {tag} {blk + 1}/{total_blocks}] "
                    f"E={block_energies[-1]:.6f} "
                    f"<n>={n_photon_means[-1]:.4f} "
                    f"sigma_r={sigma_r:.4f}",
                    file=sys.stdout, flush=True,
                )

        # Production statistics (skip equilibration blocks).
        prod_E = jnp.array(block_energies[num_blocks_equil:])
        prod_n = jnp.array(n_photon_means[num_blocks_equil:])

        # Use binning analysis for stderr (handles autocorrelation).
        if len(prod_E) > 8:
            try:
                _, e_serr, _ = do_binning_analysis(prod_E)
            except Exception:
                e_serr = float(jnp.std(prod_E)) / max(jnp.sqrt(len(prod_E)), 1.0)
        else:
            e_serr = float(jnp.std(prod_E)) / max(jnp.sqrt(len(prod_E)), 1.0)

        return {
            "E_mean": float(jnp.mean(prod_E)),
            "E_serr": float(e_serr),
            "E_blocks": prod_E,
            "n_photon_mean": float(jnp.mean(prod_n)),
            "n_photon_blocks": prod_n,
            "acceptance_r": (
                accept_r_running / max(total_r_attempts, 1.0)
            ),
            "acceptance_n": (
                accept_n_running / max(total_n_attempts, 1.0)
            ),
            "sigma_r_final": float(sigma_r),
        }


def get_qed_vmc_nn_func(
    mol_info: Mole_custom,
    config,
    init_key,
    *,
    omega: float,
    coupling_vec,
    alpha_init: float = 0.0,
    alpha_train: bool = False,
    nph_max: int = 10,
    n_aware: bool = False,
    fock_hidden_dim: int = 64,
    arch: str | None = None,
    complex_psi: bool = False,
) -> _QEDVMCDriverNN:
    """Construct a QED-VMC driver for a cavity-coupled molecule.

    Mirrors :func:`OmegaQMC.vmc_nn.get_vmc_nn_func` but builds a joint
    electron-photon ansatz via
    :func:`~OmegaQMC.psi.nn.qed_adapter.make_qed_nn_log_psi` and uses
    :func:`~OmegaQMC.psi.nn.qed_physics.pauli_fierz_local_energy` for
    the local-energy estimator.

    Args:
        mol_info: :class:`~OmegaQMC.utils.Mole_custom`.
        config: ansatz config (string name or :class:`NNAnsatzConfig`).
            Same as :func:`make_nn_log_psi`.
        init_key: JAX PRNG key for NN parameter init.
        omega: cavity-mode frequency (Ha).
        coupling_vec: ``(3,)`` light-matter coupling
            (norm = λ, direction = ε), matching :mod:`OmegaQMC.qed_fci`.
        alpha_init: initial coherent-state shift α. For symmetric
            non-polar systems use 0.0; for polar systems consider
            calling :func:`analytical_alpha_perturbative` first.
        alpha_train: whether α is included in the trainable parameter
            set (default False — fixed-parameter eval mode).
        nph_max: photon-Fock cutoff. With coherent-state shift, 5–10 is
            typically sufficient even at λ ~ 1; without shift, may need
            ≥ 30 at large λ.

    Returns:
        A callable driver instance (use ``driver(rng_key, …)``).
    """
    return _QEDVMCDriverNN(
        mol_info, config, init_key,
        omega=omega, coupling_vec=coupling_vec,
        alpha_init=alpha_init, alpha_train=alpha_train,
        nph_max=nph_max,
        n_aware=n_aware, fock_hidden_dim=fock_hidden_dim,
        arch=arch,
        complex_psi=complex_psi,
    )
