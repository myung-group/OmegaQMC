"""
Memory-lean phaseless AFQMC driver.

Drop-in replacement for :class:`_AFQMCDriverGTO` (single- and
multi-determinant trials) with a smaller GPU footprint:

* Block-end energy uses :func:`local_energy_streamed` (single-det)
  or :func:`local_energy_multidet_streamed` (multi-det) — the
  exchange tensor is accumulated in slabs of ``e_chunk_g``
  auxiliary indices, capping the per-step peak at
  ``(e_chunk_g, nwalkers, nocc, nocc)`` complex128 regardless of
  ``ndet``.
* The one-body trace is taken against the half-rotated Green's
  function via a precontracted ``h1e @ trial.conj()`` (per-det in
  the multi-det case), so the full ``(nwalkers, nbasis, nbasis)``
  aggregate G is never materialized.
* ``walker_chunk_size`` is auto-tuned to fit the propagation VHS
  into a fraction of free GPU memory unless the user passes an
  explicit value.
"""

import sys

import numpy as np
import jax
import jax.numpy as jnp

from OmegaQMC.utils import do_binning_analysis

from OmegaQMC.afqmc_gto import (
    WEIGHT_CLIP_FRACTION,
    _AFQMCDriverGTO,
    _autotune_afqmc_walkers,
    _make_afqmc_sharding,
    Walkers,
    orthogonalize_walkers,
    population_control_comb,
    propagate_walkers,
    propagate_walkers_multidet,
)
from OmegaQMC.observables.greens import (
    greens_function_force_bias,
    greens_function_multidet_force_bias,
)
from OmegaQMC.observables.energy import (
    local_energy_streamed,
    local_energy_multidet_streamed,
)


def _autotune_walker_chunk_size(driver, free_mb, mem_frac=0.25):
    """Pick a walker_chunk_size that fits the propagator VHS.

    Probes the per-walker propagation cost via the existing
    :func:`_autotune_afqmc_walkers` delta probe and returns a
    power-of-two chunk size targeting ``mem_frac`` of free GPU
    memory.  Returns ``None`` when free memory cannot be
    determined (caller treats this as "no chunking").

    Args:
        driver: instance with ``nbasis, nup, ndown, propagator,
            chol, rchol_{a,b}, trial_{up,dn}`` attributes.
        free_mb: free GPU memory in MiB; ``None`` skips tuning.
        mem_frac: fraction of free memory to target.

    Returns:
        int or None.
    """
    if free_mb is None:
        return None
    _, bpw = _autotune_afqmc_walkers(driver, free_mb)
    if bpw <= 0:
        return None
    budget_bytes = free_mb * 1e6 * mem_frac
    n = max(1, int(budget_bytes / bpw))
    # Snap down to power of two for stable AOT shapes.
    pow2 = 1
    while pow2 * 2 <= n:
        pow2 *= 2
    return pow2


class _AFQMCDriverGTO_EStream(_AFQMCDriverGTO):
    """Memory-lean single-determinant phaseless AFQMC driver.

    Inherits all setup logic (integral preparation, Cholesky
    half-rotation, propagator build) from :class:`_AFQMCDriverGTO`.
    Overrides only the block-end energy path and the walker-chunk
    default in :meth:`__call__`.

    Args (in addition to :class:`_AFQMCDriverGTO`):
        e_chunk_g: Slab size along the auxiliary axis for the
            streamed exchange-trace in the block-end energy
            estimator.  Default 16.
    """

    def __init__(self, mf, dt=0.005, chol_cut=1e-5, verbose=True,
                 trial=None, chol_h5_path=None, chol_chunk_g=128,
                 e_chunk_g=16):
        super().__init__(
            mf, dt=dt, chol_cut=chol_cut, verbose=verbose,
            trial=trial, chol_h5_path=chol_h5_path,
            chol_chunk_g=chol_chunk_g,
        )
        self.e_chunk_g = e_chunk_g
        # Precontract h1e with the trial(s) for the streamed
        # 1-body trace; avoids materializing full G at block-end.
        if self.multidet:
            # (ndet, nbasis, nocc) per spin.
            self.h1e_trials_a = jnp.einsum(
                'pq,dqi->dpi', self.h1e, self.trials_up.conj(),
            )
            self.h1e_trials_b = jnp.einsum(
                'pq,dqi->dpi', self.h1e, self.trials_dn.conj(),
            )
        else:
            self.h1e_trial_a = self.h1e @ self.trial_up.conj()
            self.h1e_trial_b = self.h1e @ self.trial_dn.conj()

        if verbose:
            print(f"  e_chunk_g = {e_chunk_g} "
                  f"(streamed exchange-trace slab size)")

    def __call__(self, rng_key=None, num_walkers=100, num_blocks=100,
                 num_steps_per_block=25, stabilize_freq=5,
                 pop_control_freq=5, num_blocks_equil=10,
                 fname_log="afqmc.log", walker_chunk_size=None):
        """Run the AFQMC simulation with streamed block-end energy.

        Same signature as :meth:`_AFQMCDriverGTO.__call__` except
        ``walker_chunk_size=None`` triggers auto-tuning instead of
        leaving the VHS un-chunked.  Pass ``walker_chunk_size=0``
        to force the un-chunked path explicitly.
        """
        if rng_key is None:
            rng_key = jax.random.key(42)

        if fname_log is None \
                or (isinstance(fname_log, str) and fname_log == ""):
            fout = sys.stdout
        else:
            fout = open(fname_log, 'w', 1)

        verbose = self.verbose

        # Informational GPU capacity estimate (does not modify
        # num_walkers); also drives walker_chunk_size auto-tune.
        free_mb = None
        try:
            from .vmcopt_gto_linear import _get_free_gpu_mb
            free_mb = _get_free_gpu_mb()
            n_rec, bpw = _autotune_afqmc_walkers(self, free_mb)
            free_txt = (f"{free_mb:.0f} MiB free"
                        if free_mb is not None
                        else "free GPU mem unknown")
            print(f"ℹ️\tEst. GPU capacity: {n_rec} walkers "
                  f"(user requested {num_walkers}; "
                  f"{bpw / 1e6:.2f} MB/walker, {free_txt})",
                  file=fout)
        except Exception as e:
            print(f"ℹ️\tGPU capacity estimate unavailable: {e}",
                  file=fout)

        if walker_chunk_size is None:
            wcs = _autotune_walker_chunk_size(self, free_mb)
            if wcs is not None and wcs < num_walkers:
                walker_chunk_size = wcs
                if verbose:
                    print(f"  walker_chunk_size = {wcs} "
                          f"(auto-tuned)", file=fout)
        elif walker_chunk_size == 0:
            walker_chunk_size = None
        elif verbose:
            print(f"  walker_chunk_size = {walker_chunk_size} "
                  f"(user override)", file=fout)

        # Initialize walkers
        walkers = Walkers(self.trial_up, self.trial_dn,
                          num_walkers, self.nbasis,
                          self.nup, self.ndown)
        phia = walkers.phia
        phib = walkers.phib
        weights = walkers.weights
        overlap = walkers.overlap
        e_hybrid = jnp.zeros(num_walkers, dtype=jnp.complex128)

        phi_sharding, scalar_sharding = _make_afqmc_sharding(num_walkers)
        if phi_sharding is not None:
            phia = jax.device_put(phia, phi_sharding)
            phib = jax.device_put(phib, phi_sharding)
            weights = jax.device_put(weights, scalar_sharding)
            overlap = jax.device_put(overlap, scalar_sharding)
            e_hybrid = jax.device_put(e_hybrid, scalar_sharding)
        if verbose and phi_sharding is not None:
            print(f"  Sharding {num_walkers} walkers "
                  f"across {len(jax.devices())} devices",
                  file=fout)

        # Main QMC loop
        total_blocks = num_blocks_equil + num_blocks
        energy_blocks = []
        eshift = 0.0
        step_count = 0

        if verbose:
            print(f"\nStarting AFQMC: {num_blocks_equil} eqlb + "
                  f"{num_blocks} prod blocks", file=fout)
            print(f"  {num_steps_per_block} steps/block, "
                  f"stabilize every {stabilize_freq} steps",
                  file=fout)
            print("-" * 70, file=fout)
            print(f"{'Block':>6} {'E_mixed':>16} {'E_shift':>16} "
                  f"{'W_sum':>12} {'W_max':>12}", file=fout)
            print("-" * 70, file=fout)

        for iblock in range(total_blocks):
            acc_weight = jnp.zeros((), dtype=jnp.float64)
            acc_ehybrid = jnp.zeros((), dtype=jnp.float64)

            for istep in range(num_steps_per_block):
                step_count += 1

                if step_count % stabilize_freq == 0:
                    phia, phib, log_detR, _ = orthogonalize_walkers(
                        phia, phib)
                    overlap = overlap / jnp.exp(log_detR)

                rng_key, step_key = jax.random.split(rng_key)
                if self.multidet:
                    phia, phib, weights, overlap, e_hybrid, _ = \
                        propagate_walkers_multidet(
                            phia, phib, weights, overlap, e_hybrid,
                            self.propagator, self.chol,
                            self.rchols_a, self.rchols_b,
                            self.trials_up, self.trials_dn,
                            self.ci_coeffs, eshift, step_key,
                            walker_chunk_size=walker_chunk_size,
                            chol_chunk_g=self.chol_chunk_g)
                else:
                    phia, phib, weights, overlap, e_hybrid, _ = \
                        propagate_walkers(
                            phia, phib, weights, overlap, e_hybrid,
                            self.propagator, self.chol,
                            self.rchol_a, self.rchol_b,
                            self.trial_up, self.trial_dn,
                            eshift, step_key,
                            walker_chunk_size=walker_chunk_size,
                            chol_chunk_g=self.chol_chunk_g)

                if step_count > 1:
                    total_weight = jnp.sum(jnp.abs(weights))
                    wbound = total_weight * WEIGHT_CLIP_FRACTION
                    weights = jnp.clip(weights, 0.0, wbound)

                if step_count % pop_control_freq == 0:
                    rng_key, pc_key = jax.random.split(rng_key)
                    weights, phia, phib = population_control_comb(
                        weights, phia, phib, pc_key)
                    if phi_sharding is not None:
                        phia = jax.device_put(phia, phi_sharding)
                        phib = jax.device_put(phib, phi_sharding)
                        weights = jax.device_put(
                            weights, scalar_sharding)

                w_abs = jnp.abs(weights)
                acc_weight = acc_weight + jnp.sum(w_abs)
                acc_ehybrid = acc_ehybrid + jnp.sum(
                    w_abs * e_hybrid.real)

            # Block-end energy: streamed, no full G.
            if self.multidet:
                Ghalfa_all, Ghalfb_all, ovlp_block, \
                    ovlp_a_all, ovlp_b_all = \
                    greens_function_multidet_force_bias(
                        phia, phib,
                        self.trials_up, self.trials_dn,
                        self.ci_coeffs)
                e_tot, e_1b, e_2b = local_energy_multidet_streamed(
                    self.h1e_trials_a, self.h1e_trials_b,
                    Ghalfa_all, Ghalfb_all,
                    self.rchols_a, self.rchols_b, self.ci_coeffs,
                    ovlp_a_all, ovlp_b_all, self.enuc,
                    self.e_chunk_g)
            else:
                Ghalfa, Ghalfb, ovlp_block = \
                    greens_function_force_bias(
                        phia, phib, self.trial_up, self.trial_dn)
                e_tot, e_1b, e_2b = local_energy_streamed(
                    self.h1e_trial_a, self.h1e_trial_b,
                    Ghalfa, Ghalfb,
                    self.rchol_a, self.rchol_b, self.enuc,
                    self.e_chunk_g)

            w = jnp.abs(weights)
            w_sum = jnp.sum(w)
            e_block = jnp.sum(w * e_tot.real) / w_sum
            e_block_val = float(e_block)
            energy_blocks.append(e_block_val)

            acc_weight_h = float(acc_weight)
            if acc_weight_h > 1e-10:
                eshift = float(acc_ehybrid) / acc_weight_h

            if verbose:
                is_eqlb = iblock < num_blocks_equil
                phase = "EQ" if is_eqlb else "  "
                print(f"{phase}{iblock:4d} {e_block_val:16.10f} "
                      f"{float(eshift):16.10f} "
                      f"{float(w_sum):12.4f} "
                      f"{float(jnp.max(weights)):12.4f}",
                      file=fout)

        energy_blocks = np.array(energy_blocks)
        prod_energies = energy_blocks[num_blocks_equil:]

        e_mean, e_err, e_std, kappa = do_binning_analysis(
            jnp.array(prod_energies))

        if verbose:
            print("-" * 70, file=fout)
            print(f"E_HF     = {float(self.mf.e_tot):.10f}",
                  file=fout)
            print(f"E_AFQMC  = {float(e_mean):.10f} "
                  f"+/- {float(e_err):.10f}", file=fout)
            print(f"E_corr   = {float(e_mean) - float(self.mf.e_tot):.10f}",
                  file=fout)
            print(f"kappa    = {float(kappa):.2f}", file=fout)

        if fout is not sys.stdout:
            fout.close()

        return {
            'energy_blocks': energy_blocks,
            'energy_mean': float(e_mean),
            'energy_err': float(e_err),
            'energy_std': float(e_std),
            'kappa': float(kappa),
            'ehf': float(self.mf.e_tot),
        }
