"""Linear method (OneShiftOnly) VMC optimizer.

Ports the QMCPACK OneShiftOnly wavefunction optimization
algorithm to JAX.  Instead of iterative gradient descent
through the full local-energy computation, this driver:

1. Samples walker configurations from |Ψ|².
2. Computes per-walker ∂ln(Ψ)/∂p and ∂E_L/∂p once.
3. Builds overlap (S) and Hamiltonian (H) matrices.
4. Solves a generalized eigenvalue problem for the
   parameter update direction.
5. Resamples and repeats.

Reference: QMCPACK ``QMCFixedSampleLinearOptimize.cpp``,
``one_shift_run()``.
"""

import subprocess
import warnings
import jax
import jax.numpy as jnp
from functools import partial
from .cusp import get_cusp_params
from .psi_gto import get_psi_fun
from .constants import MIN_DIST_THRESHOLD
from .vmcopt_gto_pssgd import (
    _build_opt_mask,
    _init_params_corr,
    _check_j2_cusps,
)
from .utils import _make_sharding


# -----------------------------------------------------------
# Pytree flatten / unflatten helpers
# -----------------------------------------------------------

def _flatten_params(params_corr):
    """Flatten a parameter pytree into a 1-D array.

    Returns
    -------
    flat : jnp.ndarray, shape (n_total,)
    treedef : pytree structure descriptor
    shapes : list of tuples
        Per-leaf shapes, needed for unflattening.
    """
    leaves, treedef = jax.tree_util.tree_flatten(
        params_corr
    )
    shapes = [l.shape for l in leaves]
    flat = jnp.concatenate([l.ravel() for l in leaves])
    return flat, treedef, shapes


def _unflatten_params(flat, treedef, shapes):
    """Reconstruct a parameter pytree from a flat array."""
    leaves = []
    offset = 0
    for s in shapes:
        size = 1
        for d in s:
            size *= d
        leaves.append(flat[offset:offset + size].reshape(s))
        offset += size
    return jax.tree_util.tree_unflatten(treedef, leaves)


def _build_flat_mask(params_corr, frozen_keys):
    """Build a 1-D boolean mask (True = optimizable).

    Combines ``_build_opt_mask`` with flattening.
    """
    mask_tree = _build_opt_mask(params_corr, frozen_keys)
    if mask_tree is None:
        flat, _, _ = _flatten_params(params_corr)
        return jnp.ones(flat.shape[0], dtype=bool)
    leaves, _ = jax.tree_util.tree_flatten(mask_tree)
    return jnp.concatenate([l.ravel() for l in leaves])


# -----------------------------------------------------------
# Parameter validation
# -----------------------------------------------------------

def _check_pade_denominators(params_corr):
    """Error if any Padé denominator parameter is <= 0.

    The Padé form u(r) = a*r / (1 + b*r) diverges linearly
    when b <= 0, making the Jastrow factor blow up at large
    electron separations.  This always produces nonsensical
    local energies and must be caught before sampling.
    """
    for key in ("J1_pade", "J2_pade"):
        if key not in params_corr:
            continue
        sub = params_corr[key]
        for name, arr in sub.items():
            # arr[-1] is the denominator for J1_pade;
            # arr[1] is the denominator for J2_pade.
            b_val = float(arr[-1]) if key == "J1_pade" \
                else float(arr[1])
            if b_val <= 0.0:
                raise ValueError(
                    f"{key}['{name}'] has denominator"
                    f" parameter b = {b_val:.6f}."
                    f"  The Padé form a*r/(1+b*r)"
                    f" diverges when b <= 0."
                    f"  Use a positive value"
                    f" (e.g. 1.0)."
                )


# -----------------------------------------------------------
# Eigenvalue selection (port of LinearMethod.cpp)
# -----------------------------------------------------------

def _select_eigenvector(eigenvals, eigenvecs, E0):
    """Select the best eigenvalue/eigenvector pair.

    Ported from QMCPACK ``LinearMethod::selectEigenvalue``.
    Filter: accept eigenvalues in (E0-100, E0).
    If none found, broaden to (E0-100, E0+100).
    Rank by distance to E0-2.
    """
    eigenvals_r = eigenvals.real
    eigenvecs_r = eigenvecs.real

    # Primary filter: (E0 - 100, E0)
    valid = (eigenvals_r < E0) & (
        eigenvals_r > E0 - 100.0
    )
    score = jnp.where(
        valid,
        (eigenvals_r - E0 + 2.0) ** 2,
        jnp.inf,
    )

    # Fallback: broaden to (E0 - 100, E0 + 100)
    any_valid = jnp.any(valid)
    valid2 = (eigenvals_r < E0 + 100.0) & (
        eigenvals_r > E0 - 100.0
    )
    score2 = jnp.where(
        valid2,
        (eigenvals_r - E0 + 2.0) ** 2,
        jnp.inf,
    )
    score = jnp.where(any_valid, score, score2)

    best_idx = jnp.argmin(score)
    ev = eigenvecs_r[:, best_idx]
    # Scale so that ev[0] == 1
    ev = ev / ev[0]
    return eigenvals_r[best_idx], ev


def _nonlinear_rescale(dP, S_block):
    """Nonlinear rescaling factor.

    Ported from QMCPACK ``LinearMethod::getNonLinearRescale``.
    All Jastrow parameters are nonlinear.
    """
    xi = 0.5
    D = dP @ S_block @ dP
    rescale = (
        (1 - xi) * D
        / ((1 - xi) + xi * jnp.sqrt(1 + D))
    )
    return 1.0 / (1.0 - rescale)


# -----------------------------------------------------------
# GPU memory auto-tuning
# -----------------------------------------------------------

def _get_free_gpu_mb():
    """Return free GPU memory in MiB via nvidia-smi.

    Returns
    -------
    float or None
        Free GPU memory in MiB, or None if unavailable.
    """
    try:
        result = subprocess.run(
            [
                'nvidia-smi',
                '--query-gpu=memory.free',
                '--format=csv,noheader,nounits',
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode != 0:
            return None
        lines = result.stdout.strip().split('\n')
        gpu_devs = [
            d for d in jax.devices()
            if d.platform == 'gpu'
        ]
        idx = gpu_devs[0].id if gpu_devs else 0
        idx = min(idx, len(lines) - 1)
        return float(lines[idx].strip())
    except Exception:
        return None


def _autotune_deriv_batch(
        compute_fn, nelec, n_params,
        flat_params_sample, free_mb,
        mem_frac=0.75,
):
    """Choose batch size for derivative computation.

    Compiles the vmapped derivative function for a
    single-walker probe to measure per-walker GPU
    memory via JAX AOT analysis.  Falls back to a
    0.5 MB/walker heuristic when AOT is unavailable.

    Parameters
    ----------
    compute_fn : callable
        Single-walker derivative function
        ``(elec_crds, fp) -> (e_loc, dlogpsi, dEL)``.
    nelec : int
        Number of electrons.
    n_params : int
        Number of optimizable parameters.
    flat_params_sample : jnp.ndarray, shape (n_params,)
        Representative parameter vector for tracing.
    free_mb : float or None
        Free GPU memory in MiB; ``None`` assumes 4096.
    mem_frac : float
        Fraction of free memory to target (default 0.75).

    Returns
    -------
    int
        Recommended per-call batch size.
    """
    bytes_per_walker = None
    try:
        probe = jnp.zeros((1, nelec, 3))
        vmapped = jax.vmap(
            compute_fn, in_axes=(0, None)
        )
        compiled = (
            jax.jit(vmapped)
            .lower(probe, flat_params_sample)
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
        bytes_per_walker = 0.5e6  # 0.5 MB fallback

    free_bytes = (
        (free_mb or 4096.0) * 1e6 * mem_frac
    )
    bs = int(free_bytes / bytes_per_walker)
    return max(10, min(bs, 8192))


# -----------------------------------------------------------
# Driver class
# -----------------------------------------------------------

class _VMCOptLinearDriver:
    """Linear method VMC optimizer (OneShiftOnly)."""

    def __init__(self, mf, params_cusp,
                 jastrow_config=None):
        nuc_crds = jnp.array(
            mf.mol.atom_coords(unit='Bohr')
        )
        eps = jnp.finfo(nuc_crds.dtype).eps
        nelec = mf.mol.tot_electrons()
        Z_charges = mf.mol.atom_charges()
        i_e, j_e = jnp.triu_indices(nelec, k=1)

        self.mf = mf
        self.nuc_crds = nuc_crds
        self.eps = eps
        self.nelec = nelec
        self.Z_charges = Z_charges

        (log_trial_wavefunction, local_energy, _, _) = \
            get_psi_fun(
                mf, params_cusp=params_cusp,
                jastrow_config=jastrow_config,
            )
        (local_energy_ee, local_energy_nn,
         local_energy_en, local_energy_ke) = local_energy
        enr_nn = local_energy_nn(nuc_crds)

        # ---- Metropolis move (same as vmcopt_gto.py) ----
        @jax.jit
        def metropolis_move(
            rng_key, elec_crds, _step_size, curr_params
        ):
            key_prop, key_accept = jax.random.split(rng_key)
            proposed_crds = (
                elec_crds
                + _step_size * jax.random.normal(
                    key_prop, elec_crds.shape
                )
            )
            diffs_ee = (
                proposed_crds[i_e] - proposed_crds[j_e]
            )
            dists_ee = jnp.linalg.norm(
                diffs_ee, axis=-1
            )
            diffs_en = (
                proposed_crds[:, None, :]
                - nuc_crds[None, :, :]
            )
            dists_en = jnp.linalg.norm(
                diffs_en, axis=-1
            )
            valid_move = (
                (dists_en.min() > MIN_DIST_THRESHOLD)
                & (dists_ee.min() > MIN_DIST_THRESHOLD)
            )
            log_psi_old = log_trial_wavefunction(
                elec_crds, nuc_crds, curr_params
            )
            log_psi_new = log_trial_wavefunction(
                proposed_crds, nuc_crds, curr_params
            )
            accept = (
                jax.random.uniform(key_accept)
                < jnp.exp(
                    2 * (log_psi_new - log_psi_old)
                )
            ) & valid_move
            new_crds = jnp.where(
                accept, proposed_crds, elec_crds
            )
            return new_crds, accept

        # ---- Total local energy (same as vmcopt_gto) ----
        @jax.jit
        def total_local_energy_fn(elec_crds, curr_params):
            return (
                local_energy_ee(elec_crds)
                + local_energy_en(elec_crds, nuc_crds)
                + local_energy_ke(
                    elec_crds, nuc_crds, curr_params
                )
                + enr_nn
            )

        # ---- log-psi wrapper (for correlated sampling) --
        @jax.jit
        def log_psi_fn(elec_crds, curr_params):
            return log_trial_wavefunction(
                elec_crds, nuc_crds, curr_params
            )

        # ---- Equilibration scan function ----
        @partial(jax.jit, static_argnums=(4, 5))
        def run_equilibration(
            rng_key, walkers, step_size,
            params_corr, num_be, num_spb
        ):
            @jax.jit
            def eq_step(carried_in, _):
                rkey, w, s, cp = carried_in
                rkey0, rkey1 = jax.random.split(rkey)
                keys = jax.random.split(
                    rkey1, w.shape[0]
                )
                new_w, accepted = jax.vmap(
                    metropolis_move,
                    in_axes=(0, 0, None, None),
                )(keys, w, s, cp)
                rate = accepted.mean()
                new_s = s * (0.6 + rate)
                return (rkey0, new_w, new_s, cp), rate

            for _ in range(num_be):
                carry_in = (
                    rng_key, walkers,
                    step_size, params_corr,
                )
                carry_out, acc_ratios = jax.lax.scan(
                    eq_step, carry_in,
                    jnp.arange(num_spb),
                )
                rng_key, walkers, step_size, _ = \
                    carry_out

            return carry_out, acc_ratios

        # ---- Production scan function ----
        @partial(jax.jit, static_argnums=(4, 5))
        def run_production(
            rng_key, walkers, step_size,
            params_corr, num_spb, num_dc
        ):
            @jax.jit
            def prod_step(carried_in, _):
                rkey, w, s, cp = carried_in
                for _ in range(num_dc):
                    rkey0, rkey1 = jax.random.split(rkey)
                    keys = jax.random.split(
                        rkey1, w.shape[0]
                    )
                    new_w, accepted = jax.vmap(
                        metropolis_move,
                        in_axes=(0, 0, None, None),
                    )(keys, w, s, cp)
                    w = new_w
                    rkey = rkey0
                r = accepted.mean()
                energies = jax.vmap(
                    total_local_energy_fn,
                    in_axes=(0, None),
                )(new_w, cp)
                return (rkey, new_w, s, cp), (r, energies)

            carry_in = (
                rng_key, walkers,
                step_size, params_corr,
            )
            carried_out, results = jax.lax.scan(
                prod_step, carry_in,
                jnp.arange(num_spb),
            )
            return carried_out, results

        # Store closures
        self.metropolis_move = metropolis_move
        self.total_local_energy_fn = total_local_energy_fn
        self.log_psi_fn = log_psi_fn
        self.log_trial_wavefunction = log_trial_wavefunction
        self.run_equilibration = run_equilibration
        self.run_production = run_production

        # Store refs for derivative computation
        self._log_trial_wavefunction = \
            log_trial_wavefunction
        self._total_local_energy_fn = total_local_energy_fn

    # -------------------------------------------------------
    # Walker initialization
    # -------------------------------------------------------

    def initialize_walkers(self, rng_key, num_walkers):
        """Initialize electron positions near nuclei."""
        idx_cnt = []
        for ia, iz in enumerate(self.Z_charges):
            idx_cnt.extend([ia] * int(iz))
        if self.mf.mol.charge < 0:
            idx_cnt.extend(
                [0] * abs(self.mf.mol.charge)
            )
        elif self.mf.mol.charge > 0:
            idx_cnt = idx_cnt[:-self.mf.mol.charge]
        idx_cnt = jnp.array(idx_cnt)
        centers = self.nuc_crds[idx_cnt]
        return (
            centers[jnp.newaxis, :, :]
            + 0.05 * jax.random.normal(
                rng_key,
                (num_walkers, self.nelec, 3),
            )
        )

    # -------------------------------------------------------
    # Matrix construction
    # -------------------------------------------------------

    @staticmethod
    def _build_matrices(E_L, dlogpsi, dEL):
        """Build overlap (S) and Hamiltonian (H) matrices.

        Pure energy minimization (w_beta=0).
        Port of QMCPACK fillOverlapHamiltonianMatrices.

        Parameters
        ----------
        E_L : (nw,)
        dlogpsi : (nw, num_opt)
        dEL : (nw, num_opt)

        Returns
        -------
        H, S : (N, N) where N = num_opt + 1
        """
        nw = E_L.shape[0]
        num_opt = dlogpsi.shape[1]
        N = num_opt + 1

        D_avg = jnp.mean(dlogpsi, axis=0)
        dD = dlogpsi - D_avg[None, :]

        # Overlap matrix
        S = jnp.zeros((N, N))
        S = S.at[0, 0].set(1.0)
        S_block = (dD.T @ dD) / nw
        S = S.at[1:, 1:].set(S_block)

        # Hamiltonian matrix
        E_mean = jnp.mean(E_L)
        H = jnp.zeros((N, N))
        H = H.at[0, 0].set(E_mean)

        # H[0, j+1]: <HD_j + dD_j * E_L>
        wfe = jnp.mean(
            dEL + dD * E_L[:, None], axis=0
        )
        H = H.at[0, 1:].set(wfe)

        # H[j+1, 0]: <dD_j * E_L>
        H = H.at[1:, 0].set(
            jnp.mean(dD * E_L[:, None], axis=0)
        )

        # H[i+1, j+1]: <dD_i * (HD_j + dD_j * E_L)>
        rhs = dEL + dD * E_L[:, None]
        H_block = (dD.T @ rhs) / nw
        H = H.at[1:, 1:].set(H_block)

        return H, S

    # -------------------------------------------------------
    # Eigenvalue solve
    # -------------------------------------------------------

    @staticmethod
    def _solve_eigenvalue(H, S, shift_i, shift_s):
        """Solve shifted generalized eigenvalue problem.

        Port of QMCPACK ``one_shift_run`` L1299-1330.

        Returns
        -------
        eigenval : float
        ev : (N,) scaled eigenvector with ev[0]=1
        """
        N = H.shape[0]
        idx = jnp.arange(1, N)

        # Apply identity shift to H diagonal
        H_shifted = H.at[idx, idx].add(shift_i)

        # Prepare overlap inverse
        inv_mat = S.copy()
        diag_zero = (
            jnp.diag(inv_mat)[1:] == 0
        ).astype(inv_mat.dtype)
        inv_mat = inv_mat.at[idx, idx].add(
            diag_zero * shift_i * shift_s
        )
        inv_mat = jnp.linalg.inv(inv_mat)

        # Apply overlap shift to H
        H_shifted = H_shifted.at[1:, 1:].add(
            shift_s * S[1:, 1:]
        )

        # Product matrix: S^{-1} @ H_shifted
        prd_mat = inv_mat @ H_shifted

        # QMCPACK explicitly transposes prd_mat before calling LAPACK dgeev.
        # That transpose compensates for the C++ row-major to Fortran
        # column-major storage mismatch — the net effect is that LAPACK sees
        # prd_mat and returns its RIGHT eigenvectors.
        # JAX has no such mismatch, so we pass prd_mat directly.
        eigenvals, eigenvecs = jnp.linalg.eig(prd_mat)

        E0 = H[0, 0].real
        best_val, ev = _select_eigenvector(
            eigenvals, eigenvecs, E0
        )
        return best_val, ev

    # -------------------------------------------------------
    # Main optimization loop
    # -------------------------------------------------------

    def __call__(
        self,
        rng_key,
        params_corr_init=None,
        frozen_keys=None,
        num_epochs=20,
        num_walkers='auto',
        num_steps_per_block=200,
        num_steps_decorr=1,
        num_opt_samples='auto',
        num_blocks_equil=10,
        mc_timestep=0.1,
        shift_i=0.01,
        shift_s=1.0,
        shift_s_base=4.0,
        max_param_change=0.3,
        deriv_batch_size='auto',
        verbose=1,
    ):
        """Run linear method VMC optimization.

        Parameters
        ----------
        rng_key : JAX random key
        params_corr_init : dict or None
            Initial Jastrow parameters.
        frozen_keys : dict or None
            Parameters to freeze.
        num_epochs : int
            Number of macro-iterations (sample → solve).
        num_walkers : int or ``'auto'``
            Number of MC walkers.  ``'auto'`` queries
            free GPU memory and sets this to the largest
            batch that fits (via JAX AOT analysis).
        num_steps_per_block : int
            MC steps per production block.
        num_steps_decorr : int
            Decorrelation sub-steps.
        num_opt_samples : int or ``'auto'``
            Total walker snapshots to collect for
            building the linear-method matrices.
            Multiple production blocks are run until
            this many samples are accumulated.
            QMCPACK typically uses 16000–64000.
            ``'auto'`` sets this to ``5 * num_walkers``.
        num_blocks_equil : int
            Equilibration blocks.
        mc_timestep : float
            Initial MC timestep.
        shift_i : float
            Identity shift for eigenvalue problem.
        shift_s : float
            Overlap shift for eigenvalue problem.
        shift_s_base : float
            Multiplicative factor for shift adaptation.
        max_param_change : float
            Cap on largest single-parameter change.
        deriv_batch_size : int or ``'auto'``
            Walker batch size for derivative
            computation (limits VRAM).  ``'auto'``
            matches ``num_walkers`` so all derivatives
            are computed in a single vmap call.
        verbose : int
            Verbosity level.  0 = silent, 1 = per-epoch
            progress, 2 = also print parameter values
            after each accepted update.

        Returns
        -------
        params_corr : dict
            Optimized parameters.
        stats : dict
            Energy statistics.
        """
        params_corr = _init_params_corr(params_corr_init)
        _check_j2_cusps(params_corr, self.eps)
        _check_pade_denominators(params_corr)

        if not params_corr:
            warnings.warn(
                "No Jastrow parameters to optimize."
            )
            return params_corr, {'energy': {'mean': 0.0}}

        # Flatten params and build mask
        flat_params, treedef, shapes = \
            _flatten_params(params_corr)
        flat_mask = _build_flat_mask(
            params_corr, frozen_keys
        )
        opt_indices = jnp.where(flat_mask)[0]
        num_opt = int(opt_indices.shape[0])

        if verbose >= 1:
            n_total = flat_params.shape[0]
            print(
                f"Linear method: {num_opt} optimizable"
                f" / {n_total} total parameters"
            )

        # Capture closures for flat-param derivatives
        nuc_crds = self.nuc_crds
        log_wf = self._log_trial_wavefunction
        E_L_fn = self._total_local_energy_fn

        def _unflatten(fp):
            return _unflatten_params(
                fp, treedef, shapes
            )

        @jax.jit
        def compute_walker_derivs(elec_crds, fp):
            """Per-walker: E_L, ∂ln(Ψ)/∂p, ∂E_L/∂p."""
            cp = _unflatten(fp)
            e_loc = E_L_fn(elec_crds, cp)
            dlogpsi = jax.grad(
                lambda p: log_wf(
                    elec_crds, nuc_crds,
                    _unflatten(p),
                )
            )(fp)
            dEL = jax.grad(
                lambda p: E_L_fn(
                    elec_crds, _unflatten(p),
                )
            )(fp)
            return e_loc, dlogpsi, dEL

        @jax.jit
        def compute_log_psi(elec_crds, fp):
            """log|Ψ| for correlated sampling weights."""
            return log_wf(
                elec_crds, nuc_crds, _unflatten(fp),
            )

        # Auto-tune batch sizes to fit GPU memory
        _need_auto = (
            num_walkers == 'auto'
            or num_opt_samples == 'auto'
            or deriv_batch_size == 'auto'
        )
        if _need_auto:
            free_mb = _get_free_gpu_mb()
            auto_bs = _autotune_deriv_batch(
                compute_walker_derivs,
                self.nelec, num_opt,
                flat_params, free_mb,
            )
            if num_walkers == 'auto':
                num_walkers = auto_bs
            if deriv_batch_size == 'auto':
                deriv_batch_size = auto_bs
            if num_opt_samples == 'auto':
                num_opt_samples = 5 * num_walkers
            if verbose >= 1:
                print(
                    f"  Auto-tuned:"
                    f" num_walkers={num_walkers},"
                    f" deriv_batch_size="
                    f"{deriv_batch_size},"
                    f" num_opt_samples="
                    f"{num_opt_samples}"
                )

        # Snap num_walkers / deriv_batch_size to
        # multiples of device count for sharding
        n_devices = len(jax.devices())
        if n_devices > 1:
            def _snap(x):
                return max(
                    n_devices,
                    (x // n_devices) * n_devices,
                )
            if num_walkers % n_devices != 0:
                num_walkers = _snap(num_walkers)
                if verbose >= 1:
                    print(
                        f"  num_walkers snapped to"
                        f" {num_walkers} (divisible"
                        f" by {n_devices} devices)"
                    )
            if deriv_batch_size % n_devices != 0:
                deriv_batch_size = _snap(
                    deriv_batch_size
                )

        # Sharding objects (None, None on single GPU)
        walkers_sharding, walker_keys_sharding = (
            _make_sharding(num_walkers)
        )
        if verbose >= 1 and walkers_sharding is not None:
            print(
                f"  Sharding {num_walkers} walkers"
                f" across {n_devices} devices"
            )

        # Define scan functions here (not __init__)
        # so they close over walker_keys_sharding and
        # insert with_sharding_constraint on random
        # keys, propagating the walker PartitionSpec
        # through lax.scan on multi-GPU runs.
        _metro = self.metropolis_move
        _enr_fn = self.total_local_energy_fn

        @partial(jax.jit, static_argnums=(4, 5))
        def run_equilibration(
            rng_key, walkers, step_size,
            params_corr, num_be, num_spb,
        ):
            def eq_step(carried_in, _):
                rkey, w, s, cp = carried_in
                rkey0, rkey1 = (
                    jax.random.split(rkey)
                )
                keys = jax.random.split(
                    rkey1, w.shape[0]
                )
                if walker_keys_sharding is not None:
                    keys = (
                        jax.lax
                        .with_sharding_constraint(
                            keys,
                            walker_keys_sharding,
                        )
                    )
                new_w, accepted = jax.vmap(
                    _metro,
                    in_axes=(0, 0, None, None),
                )(keys, w, s, cp)
                rate = accepted.mean()
                new_s = s * (0.6 + rate)
                return (
                    (rkey0, new_w, new_s, cp),
                    rate,
                )

            for _ in range(num_be):
                carry_in = (
                    rng_key, walkers,
                    step_size, params_corr,
                )
                carry_out, acc_ratios = (
                    jax.lax.scan(
                        eq_step, carry_in,
                        jnp.arange(num_spb),
                    )
                )
                (rng_key, walkers,
                 step_size, _) = carry_out
            return carry_out, acc_ratios

        @partial(jax.jit, static_argnums=(4, 5))
        def run_production(
            rng_key, walkers, step_size,
            params_corr, num_spb, num_dc,
        ):
            def prod_step(carried_in, _):
                rkey, w, s, cp = carried_in
                for _ in range(num_dc):
                    rkey0, rkey1 = (
                        jax.random.split(rkey)
                    )
                    keys = jax.random.split(
                        rkey1, w.shape[0]
                    )
                    if (
                        walker_keys_sharding
                        is not None
                    ):
                        keys = (
                            jax.lax
                            .with_sharding_constraint(
                                keys,
                                walker_keys_sharding,
                            )
                        )
                    new_w, accepted = jax.vmap(
                        _metro,
                        in_axes=(0, 0, None, None),
                    )(keys, w, s, cp)
                    w = new_w
                    rkey = rkey0
                r = accepted.mean()
                energies = jax.vmap(
                    _enr_fn, in_axes=(0, None),
                )(new_w, cp)
                return (
                    (rkey, new_w, s, cp),
                    (r, energies),
                )

            carry_in = (
                rng_key, walkers,
                step_size, params_corr,
            )
            carried_out, results = jax.lax.scan(
                prod_step, carry_in,
                jnp.arange(num_spb),
            )
            return carried_out, results

        # Initialize walkers
        rng_key, rng = jax.random.split(rng_key)
        walkers = self.initialize_walkers(
            rng, num_walkers
        )
        if walkers_sharding is not None:
            walkers = jax.device_put(
                walkers, walkers_sharding
            )
        mc_stepsize = (3 * mc_timestep) ** 0.5

        # ===== Main loop =====
        # best_energy = jnp.inf
        for epoch in range(num_epochs):
            curr_params = _unflatten(flat_params)

            # 1. Equilibrate
            if verbose >= 1:
                print(
                    f"\n--- Epoch {epoch + 1}"
                    f"/{num_epochs} ---"
                )
                print("  Equilibrating...")
            (rng_key, walkers, mc_stepsize, _), \
                acc_ratios = run_equilibration(
                    rng_key, walkers, mc_stepsize,
                    curr_params,
                    num_blocks_equil,
                    num_steps_per_block,
                )
            if verbose >= 1:
                print(
                    f"  Acceptance rate:"
                    f" {float(acc_ratios[-1]):.3f}"
                )

            # 2. Sample — accumulate walker snapshots
            #    across multiple production blocks.
            num_sample_blocks = max(
                1,
                -(-num_opt_samples // num_walkers),
            )
            if verbose >= 1:
                n_total_samp = (
                    num_sample_blocks * num_walkers
                )
                print(
                    f"  Sampling {num_sample_blocks}"
                    f" blocks x {num_walkers} walkers"
                    f" = {n_total_samp} samples..."
                )
            snapshots = []
            for _blk in range(num_sample_blocks):
                (rng_key, walkers, mc_stepsize, _), \
                    (_, _) = run_production(
                        rng_key, walkers, mc_stepsize,
                        curr_params,
                        num_steps_per_block,
                        num_steps_decorr,
                    )
                snapshots.append(walkers)
            sample_walkers = jnp.concatenate(
                snapshots, axis=0
            )[:num_opt_samples]
            num_samples = sample_walkers.shape[0]

            # 3. Compute per-walker derivatives (batched)
            if verbose >= 1:
                print(
                    f"  Computing derivatives for"
                    f" {num_samples} samples..."
                )
            all_EL = []
            all_dlogpsi = []
            all_dEL = []
            bs = deriv_batch_size
            for i in range(0, num_samples, bs):
                batch = sample_walkers[
                    i:min(i + bs, num_samples)
                ]
                if (walkers_sharding is not None
                        and batch.shape[0]
                        % n_devices == 0):
                    batch = jax.device_put(
                        batch, walkers_sharding
                    )
                el, dlp, del_ = jax.vmap(
                    compute_walker_derivs,
                    in_axes=(0, None),
                )(batch, flat_params)
                all_EL.append(el)
                all_dlogpsi.append(dlp)
                all_dEL.append(del_)
            E_L_all = jnp.concatenate(all_EL)
            dlogpsi_all = jnp.concatenate(all_dlogpsi)
            dEL_all = jnp.concatenate(all_dEL)

            # Keep only optimizable parameters
            dlogpsi_opt = dlogpsi_all[:, opt_indices]
            dEL_opt = dEL_all[:, opt_indices]

            E_mean = float(jnp.mean(E_L_all))
            E_std = float(jnp.std(E_L_all))
            if verbose >= 1:
                print(
                    f"  E_L = {E_mean:.8f}"
                    f" +/- {E_std / num_samples**0.5:.8f}"
                )

            # 4. Build matrices
            H, S = self._build_matrices(
                E_L_all, dlogpsi_opt, dEL_opt
            )

            # 5. Solve eigenvalue problem
            eigenval, ev = self._solve_eigenvalue(
                H, S, shift_i, shift_s
            )
            if verbose >= 1:
                print(
                    f"  Eigenvalue: {float(eigenval):.8f}"
                )

            # 6. Nonlinear rescale
            S_block = S[1:, 1:]
            Lambda = _nonlinear_rescale(
                ev[1:], S_block
            )
            if verbose >= 1:
                print(
                    f"  Rescale factor: {float(Lambda):.4f}"
                )

            # 7. Compute parameter update
            delta_opt = Lambda * ev[1:]

            # Cap largest change
            largest = float(
                jnp.max(jnp.abs(delta_opt))
            )
            if largest > max_param_change:
                scale = max_param_change / largest
                delta_opt = delta_opt * scale
                if verbose >= 1:
                    print(
                        f"  Capped update by {scale:.3f}"
                        f" (max change {largest:.4f})"
                    )

            # Map to full parameter vector
            delta_full = jnp.zeros_like(flat_params)
            delta_full = delta_full.at[
                opt_indices
            ].set(delta_opt)

            flat_params_new = flat_params + delta_full

            # 8. Accept/reject via correlated sampling
            # Compute log-psi at old and new params
            # on the full sample set.
            lp_old_parts = []
            lp_new_parts = []
            el_new_parts = []
            for i in range(0, num_samples, bs):
                sw = sample_walkers[
                    i:min(i + bs, num_samples)
                ]
                if (walkers_sharding is not None
                        and sw.shape[0]
                        % n_devices == 0):
                    sw = jax.device_put(
                        sw, walkers_sharding
                    )
                lp_old_parts.append(jax.vmap(
                    compute_log_psi, in_axes=(0, None)
                )(sw, flat_params))
                lp_new_parts.append(jax.vmap(
                    compute_log_psi, in_axes=(0, None)
                )(sw, flat_params_new))
                el_new_parts.append(jax.vmap(
                    lambda r: E_L_fn(
                        r, _unflatten(flat_params_new)
                    ),
                )(sw))
            log_psi_old = jnp.concatenate(lp_old_parts)
            log_psi_new = jnp.concatenate(lp_new_parts)
            EL_new = jnp.concatenate(el_new_parts)

            # Importance weights: |Ψ_new/Ψ_old|²
            log_weights = 2.0 * (
                log_psi_new - log_psi_old
            )
            log_weights = log_weights - jnp.max(
                log_weights
            )
            raw_weights = jnp.exp(log_weights)
            w_sum = jnp.sum(raw_weights)
            weights = raw_weights / w_sum

            # Reweighted energy at new params
            new_cost = float(
                jnp.sum(weights * EL_new)
            )
            old_cost = E_mean

            is_valid = bool(
                jnp.isfinite(new_cost)
            )

            if verbose >= 1:
                print(
                    f"  Old cost: {old_cost:.8f},"
                    f"  New cost: {new_cost:.8f},"
                    f"  Delta: {new_cost - old_cost:.8f}"
                )

            if is_valid and new_cost < old_cost:
                # Accept
                flat_params = flat_params_new
                if shift_s > 1e-2:
                    shift_s = shift_s / shift_s_base
                if verbose >= 1:
                    print("  -> Accepted.")
                if verbose >= 2:
                    curr_params = _unflatten(flat_params)
                    print(f"  Params: {curr_params}")
            else:
                # Reject: revert params, raise shift
                shift_s = shift_s * shift_s_base
                if verbose >= 1:
                    print(f"  -> Rejected.  shift_s -> {shift_s:.4f}")

            if verbose >= 1:
                print(f"  shift_i={shift_i:.4f}, shift_s={shift_s:.4f}")

        # ===== Final evaluation =====
        params_corr = _unflatten(flat_params)

        # Short production run for final energy
        if verbose >= 1:
            print("\nFinal energy evaluation...")
        (rng_key, walkers, mc_stepsize, _), \
            acc_ratios = run_equilibration(
                rng_key, walkers, mc_stepsize,
                params_corr,
                num_blocks_equil,
                num_steps_per_block,
            )
        (rng_key, walkers, _, _), (_, tw_energies) = \
            run_production(
                rng_key, walkers, mc_stepsize,
                params_corr,
                num_steps_per_block,
                num_steps_decorr,
            )
        final_E = float(jnp.mean(tw_energies))
        final_std = float(jnp.std(tw_energies))
        neff = tw_energies.size
        final_err = final_std / neff ** 0.5

        if verbose >= 1:
            print(
                f"Final E = {final_E:.8f}"
                f" +/- {final_err:.8f}"
            )
            print(f"Optimized params: {params_corr}")

        return params_corr, {
            'energy': {
                'mean': final_E,
                'stderr': final_err,
            }
        }


def get_vmcopt_gto_func(
    mf, cusp_scheme="Quady2025",
    jastrow_config=None,
):
    """Create a linear-method VMC optimizer.

    Parameters
    ----------
    mf : pyscf.scf.RHF
        Converged mean-field object.
    cusp_scheme : str or None
        Cusp-correction scheme. ``"Quady2025"``
        (default) or ``None``.
    jastrow_config : dict or None, optional
        Cutoff radii for B-spline Jastrow factors.
        Example::

            {"J1": {"H": {"r_cut": 5.0}},
             "J2": {"r_cut": 10.0},
             "J3": {"r_cut": 5.0, "N_eI": 3, "N_ee": 3}}

    Returns
    -------
    driver : _VMCOptLinearDriver
        Callable optimizer.
    """
    num_nuc = mf.mol.natm
    if cusp_scheme == "Quady2025":
        params_cusp = {}
        for i in range(num_nuc):
            atom_symbol = mf.mol.atom_symbol(i)
            if atom_symbol not in params_cusp:
                if isinstance(mf.mol.basis, str):
                    p = get_cusp_params(
                        atom_symbol, mf.mol.basis
                    )
                else:
                    p = get_cusp_params(
                        atom_symbol,
                        mf.mol.basis[atom_symbol],
                    )
                params_cusp[atom_symbol] = \
                    p[atom_symbol]
    else:
        params_cusp = None
    return _VMCOptLinearDriver(
        mf, params_cusp,
        jastrow_config=jastrow_config,
    )
