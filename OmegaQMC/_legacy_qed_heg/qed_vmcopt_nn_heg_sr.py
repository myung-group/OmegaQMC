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
        self.phase_smooth = self.driver.phase_smooth
        self.nelec = self.driver.nelec

    # -----------------------------------------------------------------
    # SR kernels
    # -----------------------------------------------------------------
    def _build_sr_kernels(self):
        log_psi_signed = self.log_psi_signed
        phase_smooth = self.phase_smooth
        _, unravel_fn = ravel_pytree(self.init_params)

        def log_R_of_flat(fp, r, n):
            log_mag, _ = log_psi_signed(r, n, unravel_fn(fp))
            return log_mag

        def phase_of_flat(fp, r, n):
            # Differentiate the SMOOTH FockHead phase φ(r, n), NOT
            # jnp.angle(signed_amp). The Slater-sign factor in
            # signed_amp = slater_sign · exp(iφ) introduces a branch
            # cut that JAX autodiff turns into spurious huge / NaN
            # gradients at near-node walkers, destabilising CG.
            return phase_smooth(r, n, unravel_fn(fp))

        # Keep fp64 throughout: SMW's K = (dO @ dO.T)/(λW) can have
        # entries ~O(P × |grad|² / (λW)) which at random init reaches
        # 1e9–1e10. fp32 precision dies; bare HEG uses fp64.
        @jax.jit
        def jac_batch_real(flat_params, r_batch, n_batch):
            return jax.vmap(
                jax.grad(log_R_of_flat), in_axes=(None, 0, 0),
            )(flat_params, r_batch, n_batch)

        @jax.jit
        def jac_batch_complex(flat_params, r_batch, n_batch):
            J_re = jax.vmap(
                jax.grad(log_R_of_flat), in_axes=(None, 0, 0),
            )(flat_params, r_batch, n_batch)
            J_im = jax.vmap(
                jax.grad(phase_of_flat), in_axes=(None, 0, 0),
            )(flat_params, r_batch, n_batch)
            return J_re, J_im

        # SMW dual-form SR solver (matches bare HEG's vmcopt_nn_heg_sr).
        #
        # For S = dO^T dO / W and damping λ, the system (S + λI) δ = f
        # has the SMW dual solution:
        #   K       = (dO @ dO.T) / (λW)                  (W, W)
        #   u       = (I + K)^{-1} (dO @ f) / (λW)        (W,)
        #   δ       = (f - dO.T @ u) / λ                  (P,)
        # which is a direct linear solve on a (W, W) matrix — exact,
        # well-conditioned (eigenvalues ≥ 1), and avoids the noise
        # amplification CG suffers when P >> W.
        #
        # Complex-Ψ variant: stack [dO_re; dO_im] vertically as a
        # (2W, P) matrix; the K matrix is then (2W, 2W). The SR Fisher
        # S = (dO_re^T dO_re + dO_im^T dO_im) / W is reproduced exactly.

        @jax.jit
        def smw_solve_real(O, O_mean, f, damping):
            nw = O.shape[0]
            dmp = jnp.float64(damping)
            inv_ln = jnp.float64(1.0) / (dmp * nw)
            dO = O - O_mean[None, :]                     # (W, P)
            K = (dO @ dO.T) * inv_ln                     # (W, W)
            I_plus_K = K + jnp.eye(nw, dtype=K.dtype)
            o_rhs = (dO @ f) * inv_ln                    # (W,)
            u = jnp.linalg.solve(I_plus_K, o_rhs)        # (W,)
            return (f - dO.T @ u) / dmp                  # (P,)

        @jax.jit
        def smw_solve_complex(O_re, O_im, O_re_mean, O_im_mean,
                              f, damping):
            """Block-form SMW for complex Ψ without materializing dO.

            S = (dO_re^T dO_re + dO_im^T dO_im) / W
            Define M = [dO_re; dO_im]  (2W, P).
            K = (M @ M.T) / (λW) is (2W, 2W) with blocks dO_a @ dO_b.T.
            Each block computed via centering identity from O_a, O_b
            and means — no dO buffer (saves W·P doubles of memory).
            """
            nw = O_re.shape[0]
            dmp = jnp.float64(damping)
            inv_ln = jnp.float64(1.0) / (dmp * nw)
            ones_W = jnp.ones(nw, dtype=O_re.dtype)

            # Auxiliary (W,) vectors: g_ab[w] = O_a[w] · Jbar_b
            g_re_re = O_re @ O_re_mean
            g_im_im = O_im @ O_im_mean
            g_re_im = O_re @ O_im_mean  # used for cross block
            g_im_re = O_im @ O_re_mean
            # Scalar centering corrections
            jbar_dot_re = jnp.dot(O_re_mean, O_re_mean)
            jbar_dot_im = jnp.dot(O_im_mean, O_im_mean)
            jbar_dot_cross = jnp.dot(O_re_mean, O_im_mean)

            # K blocks (W, W) via centering identity:
            # dO_a dO_b^T = O_a O_b^T - outer(g_ab, 1) - outer(1, g_ba)
            #               + (Jbar_a · Jbar_b) · ones(W,W)
            OOT_rr = O_re @ O_re.T
            OOT_ii = O_im @ O_im.T
            OOT_ri = O_re @ O_im.T
            B_rr = (OOT_rr
                    - jnp.outer(g_re_re, ones_W)
                    - jnp.outer(ones_W, g_re_re)
                    + jbar_dot_re)
            B_ii = (OOT_ii
                    - jnp.outer(g_im_im, ones_W)
                    - jnp.outer(ones_W, g_im_im)
                    + jbar_dot_im)
            B_ri = (OOT_ri
                    - jnp.outer(g_re_im, ones_W)
                    - jnp.outer(ones_W, g_im_re)
                    + jbar_dot_cross)

            K = jnp.block([[B_rr, B_ri], [B_ri.T, B_ii]]) * inv_ln
            n2 = 2 * nw
            I_plus_K = K + jnp.eye(n2, dtype=K.dtype)

            # M f = [dO_re f; dO_im f];  dO_a f = O_a f - (Jbar_a · f) 1_W
            dot_re_f = jnp.dot(O_re_mean, f)
            dot_im_f = jnp.dot(O_im_mean, f)
            Mf_re = O_re @ f - dot_re_f * ones_W
            Mf_im = O_im @ f - dot_im_f * ones_W
            o_rhs = jnp.concatenate([Mf_re, Mf_im]) * inv_ln

            u = jnp.linalg.solve(I_plus_K, o_rhs)
            u_re = u[:nw]
            u_im = u[nw:]

            # M^T u = dO_re^T u_re + dO_im^T u_im
            #       = O_re^T u_re - Jbar_re sum(u_re)
            #       + O_im^T u_im - Jbar_im sum(u_im)
            Mt_u = (O_re.T @ u_re - O_re_mean * jnp.sum(u_re)
                    + O_im.T @ u_im - O_im_mean * jnp.sum(u_im))
            return (f - Mt_u) / dmp

        return (jac_batch_real, jac_batch_complex,
                smw_solve_real, smw_solve_complex, unravel_fn)

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
        spring_mu: float = 0.0,
        spring_norm_clip: float = float('inf'),
        verbose: int = 1,
    ):
        """Run SR iterations. Returns ``(params_final, history)``."""
        if isinstance(rng_key, int):
            rng_key = jax.random.key(rng_key)

        params = self.init_params
        flat_params, _ = ravel_pytree(params)
        n_params = int(flat_params.shape[0])

        # FockHead is now closure-baked (not a Module) — see
        # qed_adapter_heg.make_qed_heg_log_psi_signed. So `params`
        # contains ONLY the base electronic ψ params; no separate
        # freeze mask is needed.
        if verbose:
            print(f"QED-HEG-SR: {n_params} parameters "
                  f"({'complex' if self.complex_psi else 'real'} Ψ)",
                  flush=True)

        (jac_real, jac_complex,
         smw_real, smw_complex, unravel_fn) = self._build_sr_kernels()

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
        # SPRING velocity (zero-init). When spring_mu=0 and
        # spring_norm_clip=inf this is identity-equivalent to vanilla SR.
        velocity = jnp.zeros(n_params, dtype=jnp.float32)

        for it in range(num_iters):
            t0 = datetime.now()

            # (a) decorrelate
            for _ in range(num_steps_decorr):
                rng_key, key_step = jax.random.split(rng_key)
                step_keys = jax.random.split(key_step, num_walkers)
                elec, n_ph, _ar, _an, _wr = mh_step(
                    step_keys, elec, n_ph, params, sigma_r,
                )

            # (b) local energies. The Hamiltonian is Hermitian so the
            # physical energy is real; e_loc may carry complex dtype
            # because the Fock-ladder ratios use complex signed_amp,
            # but the imaginary part averages to ~0 numerical noise.
            e_loc = self.driver._local_energy_batch(elec, n_ph, params)
            e_re = jnp.asarray(jnp.real(e_loc), dtype=jnp.float64)
            E_mean = float(jnp.mean(e_re))
            E_serr = float(jnp.std(e_re) / jnp.sqrt(num_walkers))

            # (c) Jacobians, batched. Centered-Jacobian identity (see
            # CG kernels): for centered dE the force vector reduces to
            # f = (dE @ J) / W (the Jbar·sum(dE) cross-term vanishes),
            # and the CG matvec uses J + Jbar directly. No dO is ever
            # materialized — halves peak memory vs the naive SR.
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
                Jre_parts = None
                O_im = jnp.concatenate(Jim_parts, axis=0)
                Jim_parts = None
                O_re_mean = jnp.mean(O_re, axis=0)
                O_im_mean = jnp.mean(O_im, axis=0)
                e_im64 = jnp.imag(e_loc).astype(jnp.float64)
                dE_re = e_re - jnp.mean(e_re)
                dE_im = e_im64 - jnp.mean(e_im64)
                # Factor of 2 from the VMC energy-gradient identity
                # ∂E/∂θ = 2⟨∂log|ψ|·(E_loc − ⟨E⟩)⟩. Matches bare HEG SR
                # (vmcopt_nn_heg_sr.py:464). Was missing → SR steps half
                # size → slow convergence interacted with FockHead bug.
                f = 2.0 * (dE_re @ O_re + dE_im @ O_im) / num_walkers
                delta_p = smw_complex(
                    O_re, O_im, O_re_mean, O_im_mean, f, damping,
                )
            else:
                O_parts = []
                for i in range(0, num_walkers, jac_batch_size):
                    j = min(i + jac_batch_size, num_walkers)
                    O_parts.append(jac_real(
                        flat_params, elec[i:j], n_ph[i:j],
                    ))
                O = jnp.concatenate(O_parts, axis=0)
                O_parts = None
                O_mean = jnp.mean(O, axis=0)
                dE = (e_re - jnp.mean(e_re)).astype(jnp.float64)
                # Factor of 2 — see complex branch comment.
                f = 2.0 * (dE @ O) / num_walkers
                delta_p = smw_real(O, O_mean, f, damping)
            prev_delta = delta_p

            # (d-f) SPRING velocity update + F-norm clip + apply.
            delta_p = delta_p.astype(velocity.dtype)
            max_abs = float(jnp.max(jnp.abs(delta_p)))
            velocity = spring_mu * velocity + delta_p
            v_norm = jnp.linalg.norm(velocity)
            scale = jnp.minimum(
                jnp.float32(1.0),
                jnp.float32(spring_norm_clip) / (v_norm + jnp.float32(1e-30)),
            )
            velocity = velocity * scale
            max_v = float(jnp.max(jnp.abs(velocity)))
            if max_v > max_param_change:
                velocity = velocity * (max_param_change / max_v)
            flat_params = (flat_params
                           - lr * velocity.astype(flat_params.dtype))
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
