"""QED-VMC driver for the periodic 2D/3D HEG with cavity coupling.

Joint discrete-continuous Metropolis sampler over :math:`(\\mathbf r, n)`
where :math:`\\mathbf r` are the electron coordinates (continuous,
PBC-wrapped) and :math:`n \\in \\{0, ..., N_\\text{ph,max}\\}` is a
photon Fock index (discrete). Local energy is the velocity-gauge
Pauli-Fierz Hamiltonian via
:func:`OmegaQMC.qed_physics_heg.pauli_fierz_local_energy_velocity_heg`.

Mirrors the architecture of :mod:`OmegaQMC.qed_vmc_nn` but stripped
of molecule-specific bookkeeping (mol_info, nuclear hard-walls,
multi-arch dispatch, eval-only observables). One architecture only:
real-positive HEG ψ_e wrapped by a per-Fock-state phase head from
:mod:`OmegaQMC.qed_adapter_heg` (so signed_amp is a complex unit;
``complex_psi=True`` is the default and meaningful regime).

The Metropolis step alternates two move types per step (Tang 2025
Algorithm 1):

  * 50% probability: Gaussian random walk in :math:`\\mathbf r` at
    fixed :math:`n`, followed by ``wrap_to_cell``. Acceptance
    :math:`\\min(1, |\\Psi(r',n)/\\Psi(r,n)|^2)`.
  * 50% probability: ±1 random walk in :math:`n` at fixed
    :math:`\\mathbf r`, boundary-clipped at :math:`[0, N_\\text{ph,max}]`.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from typing import Callable

from .psi.nn.heg_wf import HEGPsiFormerConfig
from .psi.nn.heg_wf_module import build_heg_psiformer_wf
from .psi.nn.periodic import make_cubic_lattice, wrap_to_cell
from .observables.ewald_dispatch import (
    build_ewald_tables_dim, ewald_pair_energy_dim,
)
from .qed_adapter_heg import make_qed_heg_log_psi_signed
from .qed_physics_heg import pauli_fierz_local_energy_velocity_heg


__all__ = ["get_qed_vmc_nn_heg_func"]


_TARGET_ACCEPT_R = 0.5
_TARGET_ACCEPT_N = 0.5
_STEP_ADAPT_RATE = 0.05


class _QEDVMCDriverNNHEG:
    """Internal driver. Use :func:`get_qed_vmc_nn_heg_func`."""

    def __init__(
        self,
        heg_config: HEGPsiFormerConfig,
        init_key,
        *,
        omega: float,
        coupling_vec,
        nph_max: int = 4,
        complex_psi: bool = True,
        ewald_n_real: int = 3,
        ewald_n_recip: int = 6,
        ewald_eta: float | None = None,
    ):
        self.config = heg_config
        self.L = float(heg_config.L)
        self.n_up = int(heg_config.n_up)
        self.n_down = int(heg_config.n_down)
        self.nelec = self.n_up + self.n_down
        self.dim = int(getattr(heg_config, 'dim', 3))
        self.omega = float(omega)
        self.coupling_vec = jnp.asarray(coupling_vec, dtype=jnp.float64)
        self.nph_max = int(nph_max)
        self.complex_psi = bool(complex_psi)

        # Lattice + Ewald (2D-aware via dispatch).
        if self.dim == 3:
            self.lattice = make_cubic_lattice(self.L)
        else:
            from .psi.nn.periodic import make_square_lattice
            self.lattice = make_square_lattice(self.L)
        self.ewald = build_ewald_tables_dim(
            self.L, dim=self.dim,
            n_real=ewald_n_real, n_recip=ewald_n_recip, eta=ewald_eta,
        )

        # Build base electronic ansatz + Fock-phase head.
        from flax import nnx
        rngs_key, head_key = jax.random.split(init_key)
        rngs = nnx.Rngs(int(jax.random.randint(rngs_key, (), 0, 2**31 - 1)))
        base_wf = build_heg_psiformer_wf(heg_config, rngs)
        # Polarization unit vector (= coupling_vec / λ if λ > 0; default
        # to coupling_vec direction or x̂ if λ=0).
        lam = float(jnp.linalg.norm(self.coupling_vec))
        eps = (
            self.coupling_vec / lam if lam > 1e-12
            else jnp.eye(self.dim)[0]
        )
        log_psi_signed, phase_smooth, params, graphdef = (
            make_qed_heg_log_psi_signed(
                base_wf, nph_max=self.nph_max, L=self.L, eps=eps,
                init_key=head_key,
            )
        )
        self.log_psi_signed = log_psi_signed
        self.phase_smooth = phase_smooth
        self.params = params
        self.graphdef = graphdef

        # JIT compile per-walker local energy + log-psi.
        omega_loc = self.omega
        cv_loc = self.coupling_vec
        nph_loc = self.nph_max
        complex_loc = self.complex_psi
        dim_loc = self.dim
        ewald_tab = self.ewald

        def _ewald_pair(r):
            return ewald_pair_energy_dim(r, ewald_tab, dim=dim_loc)

        phase_smooth = self.phase_smooth

        @jax.jit
        def local_energy_one(elec, n, params):
            return pauli_fierz_local_energy_velocity_heg(
                log_psi_signed, params, elec, n,
                omega=omega_loc, coupling_vec=cv_loc,
                nph_max=nph_loc, ewald_pair_fn=_ewald_pair,
                complex_psi=complex_loc,
                phase_smooth_fn=phase_smooth,
            )

        self._local_energy_one = local_energy_one
        self._local_energy_batch = jax.jit(
            jax.vmap(local_energy_one, in_axes=(0, 0, None)),
        )

        # log|Ψ|² for MH acceptance (real, no phase contribution).
        @jax.jit
        def log_mag_one(elec, n, params):
            log_mag, _sign = log_psi_signed(elec, n, params)
            return log_mag

        self._log_mag_one = log_mag_one

        # Joint MH step (single walker).
        self._mh_step = jax.jit(self._build_mh_step())
        self._mh_step_batch = jax.jit(jax.vmap(
            self._mh_step,
            in_axes=(0, 0, 0, None, None),
        ))

    # -----------------------------------------------------------------
    # MH builder
    # -----------------------------------------------------------------
    def _build_mh_step(self) -> Callable:
        log_mag = self._log_mag_one
        nph_max = self.nph_max
        lattice = self.lattice

        def mh_step(rng_key, r, n, params, sigma_r):
            (key_coin, key_rprop, key_racc,
             key_nprop, key_nacc) = jax.random.split(rng_key, 5)
            branch_r = jax.random.bernoulli(key_coin, 0.5)
            log_old = log_mag(r, n, params)

            # ---- R branch: Gaussian + PBC wrap ----
            r_prop = r + sigma_r * jax.random.normal(key_rprop, r.shape)
            r_prop = wrap_to_cell(r_prop, lattice)
            log_new_r = log_mag(r_prop, n, params)
            # |Ψ(r',n)/Ψ(r,n)|² = exp(2·(log|Ψ'| - log|Ψ|)).
            log_acc_r = 2.0 * (log_new_r - log_old)
            accept_r = jnp.log(jax.random.uniform(key_racc)) < log_acc_r
            r_new = jnp.where(accept_r, r_prop, r)

            # ---- N branch: ±1 ----
            step_n = 2 * jax.random.bernoulli(key_nprop, 0.5).astype(
                jnp.int32) - 1
            n_prop = n + step_n
            in_bounds = (n_prop >= 0) & (n_prop <= nph_max)
            n_eval = jnp.where(in_bounds, n_prop, n)
            log_new_n = log_mag(r, n_eval, params)
            log_acc_n = 2.0 * (log_new_n - log_old)
            accept_n = (jnp.log(jax.random.uniform(key_nacc)) < log_acc_n) \
                & in_bounds
            n_new_branch = jnp.where(accept_n, n_prop, n)

            # Pick branch result.
            r_out = jnp.where(branch_r, r_new, r)
            n_out = jnp.where(branch_r, n, n_new_branch)
            return r_out, n_out, accept_r, accept_n, branch_r

        return mh_step

    # -----------------------------------------------------------------
    # Walker init
    # -----------------------------------------------------------------
    def initialize_walkers(self, rng_key, num_walkers):
        """Uniform-in-cell electrons, all walkers start at n=0.

        For q=0 cavity at small λ the photon ground state is close to
        |0⟩, so n=0 is a sensible MCMC seed; the n-branch will explore
        the full Fock ladder during equilibration.
        """
        elec = jax.random.uniform(
            rng_key, (num_walkers, self.nelec, self.dim),
            minval=0.0, maxval=self.L,
        )
        photon_n = jnp.zeros((num_walkers,), dtype=jnp.int32)
        return elec, photon_n

    # -----------------------------------------------------------------
    # Production (eval-mode) run
    # -----------------------------------------------------------------
    def __call__(
        self,
        rng_key,
        num_walkers: int = 256,
        num_steps_per_block: int = 50,
        num_blocks: int = 50,
        num_blocks_equil: int = 10,
        mc_timestep: float = 0.1,
        verbose: int = 1,
    ) -> dict:
        """Eval-mode QED-HEG run; returns mean energy + photon stats."""
        if isinstance(rng_key, int):
            rng_key = jax.random.key(rng_key)
        key_walk, key_run = jax.random.split(rng_key)

        elec, n_ph = self.initialize_walkers(key_walk, num_walkers)
        sigma_r = float(mc_timestep)

        block_E = []
        block_n_mean = []
        accept_r_running = 0.0
        accept_n_running = 0.0
        n_r_attempts = 0
        n_n_attempts = 0
        params = self.params
        total_blocks = num_blocks + num_blocks_equil

        for blk in range(total_blocks):
            for _ in range(num_steps_per_block):
                key_run, key_step = jax.random.split(key_run)
                step_keys = jax.random.split(key_step, num_walkers)
                elec, n_ph, acc_r, acc_n, was_r = self._mh_step_batch(
                    step_keys, elec, n_ph, params, sigma_r,
                )
                accept_r_running += float(jnp.sum(acc_r & was_r))
                n_r_attempts += float(jnp.sum(was_r))
                accept_n_running += float(jnp.sum(acc_n & ~was_r))
                n_n_attempts += float(jnp.sum(~was_r))

            e_loc = self._local_energy_batch(elec, n_ph, params)
            e_real = (jnp.real(e_loc) if self.complex_psi else e_loc)
            e_block_mean = float(jnp.mean(e_real))
            n_block_mean = float(jnp.mean(n_ph.astype(jnp.float64)))

            # Adapt sigma_r toward 0.5 acceptance during equilibration.
            if n_r_attempts > 0:
                rate_r = accept_r_running / max(n_r_attempts, 1.0)
                if blk < num_blocks_equil:
                    sigma_r = sigma_r * (
                        1.0 + _STEP_ADAPT_RATE
                        * (rate_r - _TARGET_ACCEPT_R)
                    )
                    sigma_r = float(jnp.clip(sigma_r, 1e-3, self.L * 0.5))

            if blk >= num_blocks_equil:
                block_E.append(e_block_mean)
                block_n_mean.append(n_block_mean)
            if verbose >= 1:
                tag = "equil" if blk < num_blocks_equil else "prod "
                print(
                    f"  [{tag}] blk {blk:>3d}  E/N = "
                    f"{e_block_mean / self.nelec:+.6f}  <n>={n_block_mean:.3f}"
                    f"  σ_r={sigma_r:.4f}  acc_r/n={accept_r_running / max(n_r_attempts, 1.0):.2f}"
                    f"/{accept_n_running / max(n_n_attempts, 1.0):.2f}",
                    flush=True,
                )

        E_arr = np.array(block_E)
        n_arr = np.array(block_n_mean)
        E_mean = float(np.mean(E_arr))
        E_serr = float(np.std(E_arr, ddof=1) / np.sqrt(len(E_arr))) \
            if len(E_arr) > 1 else float('nan')
        return {
            'E_mean': E_mean,
            'E_serr': E_serr,
            'E_per_elec': E_mean / self.nelec,
            'E_blocks': E_arr,
            'n_photon_mean': float(np.mean(n_arr)),
            'acceptance_r': accept_r_running / max(n_r_attempts, 1.0),
            'acceptance_n': accept_n_running / max(n_n_attempts, 1.0),
            'final_walkers': (elec, n_ph),
            'final_sigma_r': sigma_r,
        }


def get_qed_vmc_nn_heg_func(
    heg_config: HEGPsiFormerConfig,
    init_key,
    *,
    omega: float,
    coupling_vec,
    nph_max: int = 4,
    complex_psi: bool = True,
    **kwargs,
) -> _QEDVMCDriverNNHEG:
    """Public constructor for the HEG QED-VMC driver."""
    return _QEDVMCDriverNNHEG(
        heg_config, init_key,
        omega=omega, coupling_vec=coupling_vec,
        nph_max=nph_max, complex_psi=complex_psi, **kwargs,
    )
