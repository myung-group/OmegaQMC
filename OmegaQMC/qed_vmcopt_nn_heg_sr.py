"""SR optimizer for the cavity-coupled HEG NN trial wavefunction.

Mirror of :mod:`OmegaQMC.qed_vmcopt_nn_sr` (molecule QED) but stripped
of molecule-specific machinery (mol_info, nuclear coords, α coherent-
state group, chirality observables, multi-arch dispatch). Single
architecture: real-positive HEG ψ_e wrapped by the per-Fock-state
amplitude + phase head from :mod:`OmegaQMC.qed_adapter_heg`, complex_psi
default-on.

Algorithm (per iteration):
  (a) Decorrelate walkers via the joint (R, n) MH sampler inherited
      from :class:`OmegaQMC.qed_vmc_nn_heg._QEDVMCDriverNNHEG`.
  (b) Compute local energies via velocity-gauge Pauli-Fierz
      (:func:`OmegaQMC.qed_physics_heg.pauli_fierz_local_energy_velocity_heg`).
  (c) Compute Jacobians w.r.t. flat-pytree params:
        - real-Ψ path: just ∂ log|Ψ| / ∂θ
        - complex-Ψ path: also ∂ phase / ∂θ
  (d) Centered force vector f.
  (e) CG solve (S + ε·I)·δp = f with Fisher S = JᵀJ / N.
  (f) Apply update with magnitude clip.
"""
from __future__ import annotations

from datetime import datetime

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree

from .qed_vmc_nn_heg import get_qed_vmc_nn_heg_func


__all__ = ["get_qed_vmcopt_nn_heg_sr_func"]


class _QEDVMCOptDriverNNHEG_SR:
    """SR optimizer for cavity-HEG. Use :func:`get_qed_vmcopt_nn_heg_sr_func`."""

    def __init__(
        self,
        heg_config,
        init_key,
        *,
        omega: float,
        coupling_vec,
        nph_max: int = 4,
        complex_psi: bool = True,
        ewald_n_real: int = 3,
        ewald_n_recip: int = 6,
        ewald_eta=None,
    ):
        self.driver = get_qed_vmc_nn_heg_func(
            heg_config, init_key,
            omega=omega, coupling_vec=coupling_vec,
            nph_max=nph_max, complex_psi=complex_psi,
            ewald_n_real=ewald_n_real, ewald_n_recip=ewald_n_recip,
            ewald_eta=ewald_eta,
        )
        self.complex_psi = bool(complex_psi)
        self.init_params = self.driver.params
        self.log_psi_signed = self.driver.log_psi_signed
        self.nelec = self.driver.nelec

    # -----------------------------------------------------------------
    # SR kernels
    # -----------------------------------------------------------------
    def _build_sr_kernels(self):
        log_psi_signed = self.log_psi_signed
        _, unravel_fn = ravel_pytree(self.init_params)

        def log_R_of_flat(fp, r, n):
            log_mag, _ = log_psi_signed(r, n, unravel_fn(fp))
            return log_mag

        def phase_of_flat(fp, r, n):
            _, sign = log_psi_signed(r, n, unravel_fn(fp))
            return jnp.angle(sign)

        @jax.jit
        def jac_batch_real(flat_params, r_batch, n_batch):
            return jax.vmap(
                jax.grad(log_R_of_flat), in_axes=(None, 0, 0),
            )(flat_params, r_batch, n_batch).astype(jnp.float32)

        @jax.jit
        def jac_batch_complex(flat_params, r_batch, n_batch):
            J_re = jax.vmap(
                jax.grad(log_R_of_flat), in_axes=(None, 0, 0),
            )(flat_params, r_batch, n_batch).astype(jnp.float32)
            J_im = jax.vmap(
                jax.grad(phase_of_flat), in_axes=(None, 0, 0),
            )(flat_params, r_batch, n_batch).astype(jnp.float32)
            return J_re, J_im

        @jax.jit
        def cg_solve_real(dO, f, x0, damping, maxiter):
            nw = dO.shape[0]
            dmp = jnp.float32(damping)
            def matvec(v):
                return (dO.T @ (dO @ v)) / nw + dmp * v
            delta, _ = jax.scipy.sparse.linalg.cg(
                matvec, f, x0=x0, maxiter=maxiter,
            )
            return delta

        @jax.jit
        def cg_solve_complex(dJ_re, dJ_im, f, x0, damping, maxiter):
            nw = dJ_re.shape[0]
            dmp = jnp.float32(damping)
            def matvec(v):
                return (dJ_re.T @ (dJ_re @ v)
                        + dJ_im.T @ (dJ_im @ v)) / nw + dmp * v
            delta, _ = jax.scipy.sparse.linalg.cg(
                matvec, f, x0=x0, maxiter=maxiter,
            )
            return delta

        return (jac_batch_real, jac_batch_complex,
                cg_solve_real, cg_solve_complex, unravel_fn)

    # -----------------------------------------------------------------
    # Optimization loop
    # -----------------------------------------------------------------
    def __call__(
        self,
        rng_key,
        num_iters: int = 200,
        num_walkers: int = 256,
        num_steps_per_block: int = 30,
        num_blocks_equil: int = 5,
        num_steps_decorr: int = 5,
        mc_timestep: float = 0.1,
        lr: float = 0.05,
        damping: float = 1e-3,
        cg_maxiter: int = 100,
        max_param_change: float = 0.5,
        jac_batch_size: int = 64,
        verbose: int = 1,
    ):
        """Run SR iterations. Returns ``(params_final, history)``."""
        if isinstance(rng_key, int):
            rng_key = jax.random.key(rng_key)

        params = self.init_params
        flat_params, _ = ravel_pytree(params)
        n_params = int(flat_params.shape[0])
        if verbose:
            print(f"QED-HEG-SR: {n_params} parameters "
                  f"({'complex' if self.complex_psi else 'real'} Ψ)",
                  flush=True)

        (jac_real, jac_complex,
         cg_real, cg_complex, unravel_fn) = self._build_sr_kernels()

        # Equilibration.
        rng_key, key_init = jax.random.split(rng_key)
        elec, n_ph = self.driver.initialize_walkers(key_init, num_walkers)
        sigma_r = float(mc_timestep)
        mh_step = self.driver._mh_step_batch
        for _ in range(num_blocks_equil):
            for _ in range(num_steps_per_block):
                rng_key, key_step = jax.random.split(rng_key)
                step_keys = jax.random.split(key_step, num_walkers)
                elec, n_ph, _ar, _an, _wr = mh_step(
                    step_keys, elec, n_ph, params, sigma_r,
                )

        history = {
            "energies": [], "energy_serrs": [],
            "n_photon": [], "param_change_max": [],
        }
        prev_delta = jnp.zeros(n_params, dtype=jnp.float32)

        for it in range(num_iters):
            t0 = datetime.now()

            # (a) decorrelate
            for _ in range(num_steps_decorr):
                rng_key, key_step = jax.random.split(rng_key)
                step_keys = jax.random.split(key_step, num_walkers)
                elec, n_ph, _ar, _an, _wr = mh_step(
                    step_keys, elec, n_ph, params, sigma_r,
                )

            # (b) local energies
            e_loc = self.driver._local_energy_batch(elec, n_ph, params)
            if self.complex_psi:
                e_re = jnp.real(e_loc)
                E_mean = float(jnp.mean(e_re))
                E_serr = float(jnp.std(e_re) / jnp.sqrt(num_walkers))
            else:
                E_mean = float(jnp.mean(e_loc))
                E_serr = float(jnp.std(e_loc) / jnp.sqrt(num_walkers))

            # (c) Jacobians, batched
            if self.complex_psi:
                Jre_parts, Jim_parts = [], []
                for i in range(0, num_walkers, jac_batch_size):
                    j = min(i + jac_batch_size, num_walkers)
                    j_re, j_im = jac_complex(
                        flat_params, elec[i:j], n_ph[i:j],
                    )
                    Jre_parts.append(j_re)
                    Jim_parts.append(j_im)
                O_re = jnp.concatenate(Jre_parts, axis=0)
                O_im = jnp.concatenate(Jim_parts, axis=0)
                e_re32 = jnp.real(e_loc).astype(jnp.float32)
                e_im32 = jnp.imag(e_loc).astype(jnp.float32)
                dE_re = e_re32 - jnp.mean(e_re32)
                dE_im = e_im32 - jnp.mean(e_im32)
                dO_re = O_re - jnp.mean(O_re, axis=0)[None, :]
                dO_im = O_im - jnp.mean(O_im, axis=0)[None, :]
                f = (dE_re @ dO_re + dE_im @ dO_im) / num_walkers
                delta_p = cg_complex(
                    dO_re, dO_im, f, prev_delta, damping, cg_maxiter,
                )
            else:
                O_parts = []
                for i in range(0, num_walkers, jac_batch_size):
                    j = min(i + jac_batch_size, num_walkers)
                    O_parts.append(jac_real(
                        flat_params, elec[i:j], n_ph[i:j],
                    ))
                O = jnp.concatenate(O_parts, axis=0)
                dO = O - jnp.mean(O, axis=0)[None, :]
                dE = (e_loc - jnp.mean(e_loc)).astype(jnp.float32)
                f = (dE @ dO) / num_walkers
                delta_p = cg_real(
                    dO, f, prev_delta, damping, cg_maxiter,
                )
            prev_delta = delta_p

            # (d-f) clip + apply
            max_abs = float(jnp.max(jnp.abs(delta_p)))
            if max_abs > max_param_change:
                delta_p = delta_p * (max_param_change / max_abs)
            flat_params = (flat_params
                           - lr * delta_p.astype(flat_params.dtype))
            params = unravel_fn(flat_params)

            history["energies"].append(E_mean)
            history["energy_serrs"].append(E_serr)
            history["n_photon"].append(float(jnp.mean(
                n_ph.astype(jnp.float64))))
            history["param_change_max"].append(max_abs)

            if verbose:
                dt = (datetime.now() - t0).total_seconds()
                print(
                    f"[QED-HEG-SR {it + 1:>4d}/{num_iters}] "
                    f"E/N={E_mean / self.nelec:+.6f}±"
                    f"{E_serr / self.nelec:.6f}  "
                    f"<n>={history['n_photon'][-1]:.3f}  "
                    f"|δp|max={max_abs:.3f}  dt={dt:.2f}s",
                    flush=True,
                )

        return params, history


def get_qed_vmcopt_nn_heg_sr_func(
    heg_config,
    init_key,
    *,
    omega: float,
    coupling_vec,
    nph_max: int = 4,
    complex_psi: bool = True,
    **kwargs,
) -> _QEDVMCOptDriverNNHEG_SR:
    """Public constructor."""
    return _QEDVMCOptDriverNNHEG_SR(
        heg_config, init_key,
        omega=omega, coupling_vec=coupling_vec,
        nph_max=nph_max, complex_psi=complex_psi, **kwargs,
    )
