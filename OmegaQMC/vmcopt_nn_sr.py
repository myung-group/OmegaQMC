"""Stochastic Reconfiguration VMC optimizer for NN
wavefunctions.

Natural-gradient optimization using the Stochastic
Reconfiguration (SR) algorithm.  Each iteration:

1. **Decorrelate** — run MCMC steps to advance walkers.
2. **Jacobian** — compute per-walker
   :math:`O_i = \\partial\\log|\\psi|/\\partial p`.
3. **Force vector** — :math:`f = \\langle \\delta E_L
   \\, \\delta O \\rangle` (centered covariance).
4. **CG solve** — :math:`(S + \\varepsilon I)\\,
   \\delta p = f` via conjugate gradient, where
   :math:`S` is the overlap / Fisher matrix.  Only
   matrix–vector products :math:`S v` are computed;
   the full :math:`S` matrix is never formed.
5. **Update** — :math:`p \\leftarrow p - \\eta\\,
   \\delta p`.

This is the standard optimizer used by FermiNet and
PsiFormer :cite:`Sorella2001,Becca2017`.

Unlike :mod:`vmcopt_nn_iradam` (Adam), SR accounts for the
curvature of the parameter space via the Fisher
information metric.  The Jacobian matrix
:math:`O \\in \\mathbb{R}^{N_{\\text{walkers}} \\times
N_{\\text{params}}}` is materialised on the GPU; for
1000 walkers and 100 000 parameters this is ~400 MB.
"""

import os
import h5py
import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
from functools import partial
from datetime import datetime

from .psi.nn.checkpoint import (
    save_nn_checkpoint,
    load_nn_checkpoint,
)
from .psi.nn.adapter import make_nn_log_psi
from .constants import MIN_DIST_THRESHOLD
# from .utils import do_binning_analysis

# Target SR iterations when num_iters='auto'.
# Each iteration is one parameter update, matching
# the ~50 000 steps typical of FermiNet/PsiFormer.
_TARGET_UPDATES_SR = 50000

# Number of SR iterations between checkpoint writes.
_CHK_EVERY_SR = 2000

# Metropolis step-size adaptation (from vmc_nn.py)
_TARGET_ACCEPTANCE_RATE = 0.4
_STEP_SIZE_ADAPTATION_RATE = 0.05


def _adapt_step_size(step_size, acceptance_rate):
    """Adapt Metropolis step size toward target."""
    return step_size * (
        1.0
        + _STEP_SIZE_ADAPTATION_RATE
        * (acceptance_rate - _TARGET_ACCEPTANCE_RATE)
    )


def _autotune_jac_batch(
    jac_fn, nelec, flat_params, free_mb,
    mem_frac=0.50,
):
    """Choose batch size for Jacobian computation.

    Compiles the vmapped Jacobian function for a
    single walker to measure per-walker GPU memory
    via JAX AOT analysis.  Falls back to a 20 MB
    per-walker heuristic when AOT is unavailable
    (gradient through an NN is ~10–50 × costlier
    than a forward pass).

    Parameters
    ----------
    jac_fn : callable
        ``(batch_walkers, flat_params) -> O_batch``.
    nelec : int
        Number of electrons.
    flat_params : jnp.ndarray
        Flattened parameter vector.
    free_mb : float or None
        Free GPU memory in MiB.
    mem_frac : float
        Fraction of free memory to target (0.50).

    Returns
    -------
    int
        Recommended Jacobian batch size.
    """
    bytes_per_walker = None
    try:
        probe = jnp.zeros((1, nelec, 3))
        compiled = (
            jax.jit(jac_fn)
            .lower(probe, flat_params)
            .compile()
        )
        analysis = compiled.memory_analysis()
        bytes_per_walker = (
            analysis.alias_size
            + analysis.temp_size
        )
    except Exception:
        pass

    if not bytes_per_walker:
        bytes_per_walker = 20.0e6  # 20 MB fallback
    free_bytes = (free_mb or 4096.0) * 1e6 * mem_frac
    bs = int(free_bytes / bytes_per_walker)
    return max(1, min(bs, 256))


def _autotune_sr_walkers(
    n_params, jac_batch_mem_mb, free_mb,
    mem_frac=0.60,
):
    """Choose walker count for SR from GPU memory.

    Budgets *mem_frac* of free GPU memory for the
    Jacobian matrix ``O`` of shape
    ``(num_walkers, n_params)`` at float32, after
    reserving space for the peak Jacobian-batch
    computation memory.

    Parameters
    ----------
    n_params : int
        Number of flattened NN parameters.
    jac_batch_mem_mb : float
        Peak memory (MiB) used by one Jacobian
        batch computation (already auto-tuned).
    free_mb : float or None
        Free GPU memory in MiB.
    mem_frac : float
        Fraction of free memory to budget (0.60).

    Returns
    -------
    int
        Recommended walker count, clamped to
        ``[256, 4096]``.
    """
    if free_mb is None:
        free_mb = 4096.0
    available_mb = free_mb * mem_frac - jac_batch_mem_mb
    bytes_budget = max(available_mb, 100.0) * 1e6
    nw = int(bytes_budget / (4 * max(n_params, 1)))
    return max(256, min(nw, 4096))


class _VMCOptDriverNN_SR:
    """SR natural-gradient VMC optimizer for NN trials.

    Compiles the Metropolis kernel and local-energy
    function for a given molecule, then runs stochastic
    reconfiguration iterations: at each step, walkers
    are decorrelated via MCMC, the log-psi Jacobian and
    local energies are computed, and a natural-gradient
    update is obtained by solving the SR linear system
    via conjugate gradient.
    """

    def __init__(self, mol_info, config, init_key):
        nuc_crds = jnp.asarray(
            mol_info.coords, dtype=jnp.float64,
        )
        charges = jnp.asarray(
            mol_info.charges, dtype=jnp.float64,
        )
        nelec = mol_info.n_up + mol_info.n_down

        self.mol_info = mol_info
        self.nuc_crds = nuc_crds
        self.charges = charges
        self.nelec = nelec
        self.config_name = (
            config if isinstance(config, str)
            else getattr(config, 'name', 'custom')
        )

        log_psi, init_params, graphdef, lap_grad = (
            make_nn_log_psi(
                config, mol_info, init_key,
            )
        )
        self.init_params = init_params
        self.log_psi = log_psi
        self.lap_grad = lap_grad

        # Precompute nuclear repulsion
        n_nuc = len(charges)
        enr_nn = 0.0
        for a in range(n_nuc):
            for b in range(a + 1, n_nuc):
                rab = jnp.linalg.norm(
                    nuc_crds[a] - nuc_crds[b],
                )
                enr_nn = enr_nn + (charges[a] * charges[b] / rab)
        enr_nn = jnp.asarray(
            enr_nn, dtype=jnp.float64,
        )

        i_e, j_e = jnp.triu_indices(nelec, k=1)

        # --- Electron-electron energy ---
        @jax.jit
        def energy_ee(elec_crds):
            diffs = (elec_crds[i_e] - elec_crds[j_e])
            dists = jnp.linalg.norm(
                diffs, axis=-1,
            )
            return jnp.sum(1.0 / dists)

        # --- Electron-nucleus energy ---
        @jax.jit
        def energy_en(elec_crds):
            diffs = (elec_crds[:, None, :] - nuc_crds[None, :, :])
            dists = jnp.linalg.norm(
                diffs, axis=-1,
            )
            return -jnp.sum(
                charges[None, :] / dists,
            )

        # --- Kinetic energy via NN Laplacian ---
        @jax.jit
        def energy_ke(elec_crds, params):
            lap_val, grad_val = lap_grad(
                elec_crds, nuc_crds, params,
            )
            return -0.5 * (
                lap_val + jnp.dot(grad_val, grad_val)
            )

        # --- Total local energy ---
        @jax.jit
        def total_local_energy(elec_crds, params):
            return (energy_ee(elec_crds) + energy_en(elec_crds)
                    + energy_ke(elec_crds, params) + enr_nn)

        # --- Metropolis move ---
        @jax.jit
        def metropolis_move(rng_key, elec_crds, step_size, params):
            key_prop, key_accept = jax.random.split(rng_key)
            proposed = elec_crds + step_size * (
                jax.random.normal(
                    key_prop, elec_crds.shape,
                )
            )
            diffs_ee = proposed[i_e] - proposed[j_e]
            dists_ee = jnp.linalg.norm(
                diffs_ee, axis=-1,
            )
            diffs_en = proposed[:, None, :] - nuc_crds[None, :, :]
            dists_en = jnp.linalg.norm(
                diffs_en, axis=-1,
            )
            valid = \
                (dists_en.min() > MIN_DIST_THRESHOLD) \
                & (dists_ee.min() > MIN_DIST_THRESHOLD)
            lp_old = log_psi(
                elec_crds, nuc_crds, params,
            )
            lp_new = log_psi(
                proposed, nuc_crds, params,
            )
            accept = (
                jax.random.uniform(key_accept)
                < jnp.exp(2 * (lp_new - lp_old))
            ) & valid
            new_crds = jnp.where(
                accept, proposed, elec_crds,
            )
            return new_crds, accept

        # --- Equilibration scan ---
        @partial(jax.jit, static_argnums=(4, 5))
        def run_equilibration(
            rng_key, walkers, step_size, params,
            num_be, num_spb,
        ):
            def eq_step(carry, _):
                rk, w, s, p = carry
                rk0, rk1 = jax.random.split(rk)
                keys = jax.random.split(
                    rk1, w.shape[0],
                )
                nw, acc = jax.vmap(
                    metropolis_move,
                    in_axes=(0, 0, None, None),
                )(keys, w, s, p)
                ar = acc.mean()
                ns = _adapt_step_size(s, ar)
                return (rk0, nw, ns, p), ar

            carry = (
                rng_key, walkers,
                step_size, params,
            )
            for _ in range(num_be):
                carry, acc = jax.lax.scan(
                    eq_step, carry,
                    jnp.arange(num_spb),
                )
            return carry, acc

        # --- Production scan ---
        @partial(jax.jit, static_argnums=(4, 5))
        def run_production(
            rng_key, walkers, step_size, params,
            num_spb, num_dc,
        ):
            def prod_step(carry, _):
                rk, w, s, p = carry
                for _ in range(num_dc):
                    rk0, rk1 = jax.random.split(rk)
                    keys = jax.random.split(
                        rk1, w.shape[0],
                    )
                    nw, acc = jax.vmap(
                        metropolis_move,
                        in_axes=(0, 0, None, None),
                    )(keys, w, s, p)
                    w = nw
                    rk = rk0
                ar = acc.mean()
                energies = jax.vmap(
                    total_local_energy,
                    in_axes=(0, None),
                )(nw, p)
                return (rk, nw, s, p), (
                    ar, energies,
                )

            carry = (
                rng_key, walkers,
                step_size, params,
            )
            carry, results = jax.lax.scan(
                prod_step, carry,
                jnp.arange(num_spb),
            )
            return carry, results

        # --- Batch energy ---
        @jax.jit
        def compute_batch_energy(walkers, params):
            return jax.vmap(
                total_local_energy,
                in_axes=(0, None),
            )(walkers, params)

        self.metropolis_move = metropolis_move
        self.run_equilibration = run_equilibration
        self.run_production = run_production
        self.compute_batch_energy = compute_batch_energy
        self.total_local_energy = total_local_energy

    def initialize_walkers(
        self, rng_key, num_walkers,
    ):
        """Place electrons near nuclei.

        Args:
            rng_key: JAX PRNG key.
            num_walkers: Number of walkers.

        Returns:
            Array ``(num_walkers, nelec, 3)``.
        """
        idx_cnt = []
        for ia, iz in enumerate(self.charges):
            idx_cnt.extend([ia] * int(iz))
        total = self.mol_info.n_up + self.mol_info.n_down
        while len(idx_cnt) < total:
            idx_cnt.append(0)
        idx_cnt = idx_cnt[:total]
        idx_cnt = jnp.array(idx_cnt)
        centers = self.nuc_crds[idx_cnt]
        return (
            centers[None, :, :]
            + 0.05 * jax.random.normal(
                rng_key,
                (num_walkers, self.nelec, 3),
            )
        )

    def __call__(
        self,
        rng_key,
        num_iters='auto',
        num_walkers='auto',
        num_steps_per_block=200,
        num_steps_decorr=10,
        num_blocks_equil=5,
        mc_timestep=0.1,
        lr=0.01,
        damping=1e-3,
        cg_maxiter=100,
        jac_batch_size='auto',
        max_param_change=0.2,
        verbose=1,
        prefix='nnopt',
    ):
        """Run SR natural-gradient optimization.

        At each iteration walkers are decorrelated via
        MCMC, then the log-psi Jacobian and local
        energies are computed.  The natural-gradient
        direction is obtained by solving
        :math:`(S + \\varepsilon I)\\,\\delta p = f`
        via conjugate gradient.

        After each iteration the current parameters
        are written to ``{prefix}.chk.h5``.  Before
        each write the previous file is preserved as
        ``{prefix}.{iter}.h5`` so every completed
        iteration is recoverable.

        Args:
            rng_key: JAX PRNG key.
            num_iters: Number of SR iterations, or
                ``'auto'`` to target
                ~50 000 updates.
            num_walkers: Number of MC walkers, or
                ``'auto'`` to set from GPU memory.
            num_steps_per_block: MC steps per
                equilibration / production block.
            num_steps_decorr: MCMC decorrelation
                steps per SR iteration.
            num_blocks_equil: Equilibration blocks
                (initial only).
            mc_timestep: Initial MC timestep.
            lr: Learning rate for the SR update.
            damping: Tikhonov regularisation
                :math:`\\varepsilon` added to the
                overlap matrix diagonal.
            cg_maxiter: Maximum conjugate-gradient
                iterations.
            jac_batch_size: Batch size for Jacobian
                computation, or ``'auto'``.
            max_param_change: Cap on the largest
                element of the raw SR update
                :math:`\\delta p` (before the
                learning rate).
            verbose: Verbosity (0 = silent).
            prefix: Filename prefix for the HDF5
                checkpoint.

        Returns:
            Tuple ``(params_final, energy_data)``
            where *params_final* is the optimized
            NNX parameter pytree and *energy_data*
            is a dict with key ``'energy'``.
        """
        params = self.init_params
        nuc_crds = self.nuc_crds
        log_psi = self.log_psi
        start_iter = 0

        # --- Checkpoint resume ---
        chkpt_path = f"{prefix}.chk.h5"
        if os.path.exists(chkpt_path):
            template_leaves = jax.tree.leaves(
                params,
            )
            n_model = len(template_leaves)
            try:
                with h5py.File(
                    chkpt_path, 'r',
                ) as f:
                    n_chk = int(f['params'].attrs['num_leaves'])
                    if n_chk != n_model:
                        print(f"Error: checkpoint '{chkpt_path}'"
                              f" has {n_chk} parameter leaves"
                              f" but current model has {n_model}."
                              " Incompatible architecture — stopping.")
                        return None, {}
                    for i, leaf in enumerate(template_leaves):
                        chk_shape = f['params'][str(i)].shape
                        if chk_shape != leaf.shape:
                            print(f"Error: parameter leaf {i} shape mismatch:"
                                  f" checkpoint {chk_shape} "
                                  f"vs model {leaf.shape}."
                                  " Incompatible architecture — stopping.")
                            return None, {}
            except (KeyError, OSError) as exc:
                print(f"Error reading checkpoint"
                      f" '{chkpt_path}': {exc} — stopping.")
                return None, {}

            params, meta = load_nn_checkpoint(
                chkpt_path, params,
            )
            start_iter = int(meta.get('epoch', -1)) + 1
            if verbose >= 1:
                print(f"Resuming from '{chkpt_path}' "
                      f"(iteration {start_iter - 1} completed,"
                      f" continuing from iteration {start_iter})")

        # --- Flatten params ---
        flat_params, unravel_fn = ravel_pytree(
            params,
        )
        n_params = flat_params.shape[0]
        if verbose >= 1:
            print(f"SR optimizer: {n_params} parameters")

        # --- Fused Jacobian + local-energy function ---
        # Fused into a single jit so XLA CSE can share
        # the forward graph between the lap_grad VGL pass
        # (kinetic energy) and the reverse-mode pullback
        # over flat params (Jacobian) — saves one NN
        # forward per walker per iteration.  Returns
        # (E_L, O) so the SR loop can drop its separate
        # ``compute_batch_energy`` pass.  ``O`` is cast to
        # float32 to halve Jacobian storage; SR linear
        # algebra does not need float64.
        def log_psi_of_flat(fp, r):
            return log_psi(
                r, nuc_crds, unravel_fn(fp),
            )

        total_local_energy_fn = self.total_local_energy

        @jax.jit
        def jac_and_energy_batch_fn(batch_w, fp):
            params_pytree = unravel_fn(fp)

            def per_walker(elec):
                E_L = total_local_energy_fn(
                    elec, params_pytree,
                )
                O_w = jax.grad(log_psi_of_flat)(
                    fp, elec,
                )
                return E_L, O_w.astype(jnp.float32)

            return jax.vmap(per_walker)(batch_w)

        # --- Auto-tune ---
        auto_walkers = num_walkers == 'auto'
        auto_jac = jac_batch_size == 'auto'
        auto_iters = num_iters == 'auto'
        if auto_walkers or auto_jac:
            from .vmcopt_gto_linear import (
                _get_free_gpu_mb,
            )
            free_mb = _get_free_gpu_mb()
        else:
            free_mb = None
        if auto_jac:
            jac_batch_size = _autotune_jac_batch(
                jac_and_energy_batch_fn, self.nelec,
                flat_params, free_mb,
            )
        if auto_walkers:
            # Reserve peak memory for one
            # Jacobian batch computation.
            jac_batch_mem_mb = jac_batch_size * 20.0
            # 20 MB/walker fallback
            try:
                probe = jnp.zeros(
                    (1, self.nelec, 3),
                )
                compiled = (
                    jax.jit(jac_and_energy_batch_fn)
                    .lower(probe, flat_params)
                    .compile()
                )
                analysis = compiled.memory_analysis()
                per_w = analysis.alias_size + analysis.temp_size
                jac_batch_mem_mb = jac_batch_size * per_w / 1e6
            except Exception:
                pass
            num_walkers = _autotune_sr_walkers(
                n_params, jac_batch_mem_mb, free_mb,
            )
        if auto_iters:
            num_iters = _TARGET_UPDATES_SR
        if auto_jac and num_walkers < jac_batch_size:
            jac_batch_size = num_walkers
        if (auto_walkers or auto_iters
                or auto_jac) and verbose >= 1:
            print(
                f"  Auto-tuned:"
                f" num_walkers={num_walkers},"
                f" num_iters={num_iters},"
                f" jac_batch={jac_batch_size}"
            )

        # --- Decorrelation scan ---
        @partial(
            jax.jit, static_argnums=(4,),
        )
        def decorr_scan(
            rng_key, walkers, step_size,
            params, num_steps,
        ):
            def step(carry, _):
                rk, w, s, p = carry
                rk, rk1 = jax.random.split(rk)
                keys = jax.random.split(
                    rk1, w.shape[0],
                )
                nw, acc = jax.vmap(
                    self.metropolis_move,
                    in_axes=(0, 0, None, None),
                )(keys, w, s, p)
                ar = acc.mean()
                ns = _adapt_step_size(s, ar)
                return (rk, nw, ns, p), ar

            carry = (
                rng_key, walkers,
                step_size, params,
            )
            carry, ars = jax.lax.scan(
                step, carry,
                jnp.arange(num_steps),
            )
            return carry, ars

        # --- CG solve (JIT-compiled) ---
        @jax.jit
        def sr_solve(dO, f, x0, dmp, maxiter):
            nw = dO.shape[0]
            dmp = jnp.float32(dmp)

            def s_matvec(v):
                t = dO @ v
                Sv = (dO.T @ t) / nw
                return Sv + dmp * v

            delta_p, info = (
                jax.scipy.sparse.linalg.cg(
                    s_matvec, f, x0=x0,
                    maxiter=maxiter,
                )
            )
            return delta_p, info

        # --- Initialize walkers ---
        rng_key, rng = jax.random.split(rng_key)
        walkers = self.initialize_walkers(
            rng, num_walkers,
        )
        mc_stepsize = (3 * mc_timestep) ** 0.5

        # --- Equilibration ---
        if verbose >= 1:
            print("Running equilibration...")
        (rng_key, walkers, mc_stepsize, _), \
            acc = self.run_equilibration(
                    rng_key, walkers,
                    mc_stepsize, params,
                    num_blocks_equil,
                    num_steps_per_block,
                    )
        if verbose >= 1:
            print(f"  acceptance rate: {acc[-1]:.2f}")
            print(f"  step size: {mc_stepsize:.4f}")
            print(f"\nStarting {num_iters} SR iterations (lr={lr},"
                  f" damping={damping})...\n")

        # --- Main SR loop ---
        prev_delta = jnp.zeros(n_params, dtype=jnp.float32)
        timestamp_prev = datetime.now()

        for iteration in range(
            start_iter, start_iter + num_iters,
        ):
            # (a) Decorrelate walkers
            (
                rng_key, walkers,
                mc_stepsize, _,
            ), ars = decorr_scan(
                rng_key, walkers,
                mc_stepsize, params,
                num_steps_decorr,
            )

            # (b)+(c) Compute local energies and
            # Jacobian together in batches.  Fused so
            # XLA CSE shares the forward graph between
            # the kinetic-energy VGL pass and the
            # reverse-mode pullback over flat params.
            E_L_parts = []
            O_parts = []
            for i in range(
                0, num_walkers, jac_batch_size,
            ):
                end = min(
                    i + jac_batch_size,
                    num_walkers,
                )
                batch = walkers[i:end]
                E_part, O_part = jac_and_energy_batch_fn(
                    batch, flat_params,
                )
                E_L_parts.append(E_part)
                O_parts.append(O_part)
            E_L = jnp.concatenate(E_L_parts, axis=0)
            O = jnp.concatenate(O_parts, axis=0)
            E_mean = float(E_L.mean())
            E_std = float(E_L.std())

            # (d) Force vector (centered form)
            #     f = mean(dE * dO) = dE @ dO / N
            # Use matvec to avoid a large (N, P)
            # intermediate from broadcasting.
            O_mean = jnp.mean(O, axis=0)
            dO = O - O_mean[None, :]
            dE = (E_L - E_L.mean()).astype(
                jnp.float32,
            )
            nw = dO.shape[0]
            f = (dE @ dO) / nw

            # (e) CG solve
            delta_p, cg_info = sr_solve(
                dO, f, prev_delta,
                damping, cg_maxiter,
            )
            prev_delta = delta_p

            # (f) Clip and apply update
            max_dp = float(jnp.max(jnp.abs(delta_p)))
            scale = min(
                1.0,
                max_param_change / max(max_dp, 1e-30),
            )
            flat_params = flat_params - lr * scale * delta_p
            params = unravel_fn(flat_params)

            # (g) Logging
            if verbose >= 1:
                now = datetime.now()
                dt = (now - timestamp_prev).total_seconds()
                timestamp_prev = now
                ar_last = float(ars[-1])
                print(f"Iter {iteration:5d} | E = {E_mean:.8f}"
                      f" +/- {E_std / num_walkers**0.5:.8f}"
                      f" | max_dp = {max_dp:.4e} | scale = {scale:.3f}"
                      f" | ar = {ar_last:.2f} | dt = {dt:.1f}s")

            # (h) Checkpoint (every _CHK_EVERY_SR iters
            # plus always at the last iteration)
            is_last = (iteration == start_iter + num_iters - 1)
            if is_last or ((iteration + 1) % _CHK_EVERY_SR == 0):
                if os.path.exists(chkpt_path):
                    os.rename(
                        chkpt_path,
                        f"{prefix}.{iteration}.h5",
                    )
                save_nn_checkpoint(
                    chkpt_path, params, iteration,
                    self.config_name,
                    self.mol_info,
                    energy=E_mean,
                )

        # ===== Final energy estimate =====
        if verbose >= 1:
            print(
                "\nFinal energy evaluation..."
            )
        (rng_key, walkers, mc_stepsize, _), \
            acc = (
                self.run_equilibration(
                    rng_key, walkers,
                    mc_stepsize, params,
                    num_blocks_equil,
                    num_steps_per_block,
                )
            )
        (rng_key, walkers, _, _), \
            (_, tw_e) = (
                self.run_production(
                    rng_key, walkers,
                    mc_stepsize, params,
                    num_steps_per_block,
                    num_steps_decorr,
                )
            )
        final_e = float(jnp.mean(tw_e))
        final_std = float(jnp.std(tw_e))
        neff = tw_e.size
        final_err = final_std / neff ** 0.5

        if verbose >= 1:
            print(f"Final energy: {final_e:.8f} +/- {final_err:.8f}")

        return params, {'energy': {'mean': final_e, 'stderr': final_err}}


def get_vmcopt_nn_func(
    mol_info, config, init_key,
):
    """Create an SR VMC optimizer for NN
    wavefunctions.

    Builds the NN trial wavefunction from *config*,
    compiles the Metropolis kernel and local-energy
    function, and returns a callable driver that
    optimises the parameters via stochastic
    reconfiguration (natural gradient with CG).

    Args:
        mol_info:
            :class:`~OmegaQMC.utils.Mole_custom`
            instance.
        config:
            :class:`~OmegaQMC.psi.nn.config.NNAnsatzConfig`
            or a string (built-in name or YAML path).
        init_key: JAX PRNG key for parameter
            initialisation.

    Returns:
        :class:`_VMCOptDriverNN_SR` instance.
        Call it with ``driver(rng_key, ...)`` to run
        the optimisation.
    """
    return _VMCOptDriverNN_SR(
        mol_info, config, init_key,
    )
