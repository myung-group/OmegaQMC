"""Tests for the complex Slater-Jastrow HEG wavefunction.

Validates that:
  * ``log_psi_complex`` returns a complex scalar with finite Re/Im.
  * At κ = 0, ``|ψ_complex(r)|² / |ψ_real(r)|²`` is a constant
    (geometry-independent) across MCMC configurations — so VMC
    sampling on |ψ|² gives identical distributions and identical
    expectation values for any physical observable.
  * ``transfer_jastrow_params`` moves the trained Jastrow from a
    real-ansatz params pytree into a fresh complex-ansatz params
    pytree.
"""

from typing import NamedTuple

import numpy as np
import jax
import jax.numpy as jnp
import pytest

from OmegaQMC.psi.nn.heg_wf import (
    HEGConfig,
    make_heg_log_psi,
    make_heg_log_psi_complex,
    transfer_jastrow_params,
)


def test_complex_log_psi_shape_and_finite():
    cfg = HEGConfig(n_up=7, n_down=7, L=4.0, n_det=1, use_jastrow=True)
    log_psi_c, params, _ = make_heg_log_psi_complex(
        cfg, jax.random.key(0),
    )
    r = jnp.asarray(
        np.random.default_rng(0).uniform(0, 4, size=(14, 3))
    )
    lp = log_psi_c(r, params)
    assert lp.dtype == jnp.complex128
    assert lp.shape == ()
    assert jnp.isfinite(jnp.real(lp))
    assert jnp.isfinite(jnp.imag(lp))


def test_complex_and_real_give_same_sampling_distribution_at_gamma():
    """At κ = 0, |ψ_complex|² ∝ |ψ_real|² with a configuration-
    independent proportionality constant.  So VMC sampling on
    |ψ|² gives identical distributions and identical energy
    expectations for any physical observable.

    We verify by evaluating Re(log ψ_c) - log|ψ_r| at 5 random
    configurations and checking that this difference is constant.
    """
    L = 4.0
    cfg = HEGConfig(n_up=7, n_down=7, L=L, n_det=1, use_jastrow=False)
    # Use the same seed so both paths generate the same RNG state at
    # module init (the envelope init is deterministic anyway).
    log_psi_r, params_r, _ = make_heg_log_psi(cfg, jax.random.key(0))
    log_psi_c, params_c, _ = make_heg_log_psi_complex(
        cfg, jax.random.key(0),
    )

    def lp_diff(seed):
        r = jnp.asarray(
            np.random.default_rng(seed).uniform(0, L, size=(14, 3))
        )
        lp_r = float(log_psi_r(r, params_r))
        lp_c_re = float(jnp.real(log_psi_c(r, params_c)))
        return lp_c_re - lp_r

    diffs = [lp_diff(s) for s in range(5)]
    # ``log|ψ|_c - log|ψ|_r = ½ log(|ψ_c|² / |ψ_r|²) = ½ log(64 × 64)
    #                      = log(64)``
    # — up and down blocks each contribute the 4³ = 64 basis-change
    # factor from three 2×2 ±k-pair blocks, and the two spin Slater
    # dets multiply.
    expected = float(np.log(64.0))
    np.testing.assert_allclose(diffs, [expected] * len(diffs), atol=1e-6)


def test_transfer_jastrow_params_replaces_jastrow_only():
    """The transfer helper must move every Jastrow leaf from source
    to destination while leaving envelope (and any other) leaves in
    the destination untouched."""
    cfg = HEGConfig(n_up=7, n_down=7, L=4.0, n_det=1, use_jastrow=True)
    _, params_r, _ = make_heg_log_psi(cfg, jax.random.key(0))
    _, params_c, _ = make_heg_log_psi_complex(
        cfg, jax.random.key(1), kappa=(0.1, 0.0, 0.0),
    )

    # Perturb every Jastrow leaf of params_r by +1.0 so the
    # transferred values are distinguishable from the freshly-
    # initialised Jastrow in params_c.
    def _bump_jastrow(pytree):
        leaves, treedef = jax.tree_util.tree_flatten_with_path(pytree)
        new_leaves = []
        for path, leaf in leaves:
            path_s = '/'.join(
                str(getattr(k, 'key', getattr(k, 'name', repr(k))))
                for k in path
            )
            if 'jastrow' in path_s:
                new_leaves.append(leaf + 1.0)
            else:
                new_leaves.append(leaf)
        return jax.tree_util.tree_unflatten(treedef, new_leaves)

    params_r_perturbed = _bump_jastrow(params_r)
    merged = transfer_jastrow_params(params_r_perturbed, params_c)

    # Every Jastrow leaf in `merged` must equal the corresponding
    # perturbed leaf in `params_r`, and every non-Jastrow leaf must
    # equal the corresponding leaf in the fresh `params_c`.
    merged_leaves = jax.tree_util.tree_flatten_with_path(merged)[0]
    src_leaves = dict(
        jax.tree_util.tree_flatten_with_path(params_r_perturbed)[0]
    )
    orig_dst_leaves = dict(
        jax.tree_util.tree_flatten_with_path(params_c)[0]
    )
    for path, leaf in merged_leaves:
        path_s = '/'.join(
            str(getattr(k, 'key', getattr(k, 'name', repr(k))))
            for k in path
        )
        if 'jastrow' in path_s:
            np.testing.assert_allclose(leaf, src_leaves[path])
        else:
            np.testing.assert_allclose(leaf, orig_dst_leaves[path])
