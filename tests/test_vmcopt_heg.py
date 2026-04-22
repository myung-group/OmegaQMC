"""Smoke + validation tests for the HEG Adam VMC optimiser.

The smoke test verifies gradient computation runs end-to-end.
The gated ``slow`` test trains the Jastrow on a small HEG for a
handful of iterations and checks that the energy drops below the
finite-cell HF baseline (i.e. correlation energy is captured).
"""

import numpy as np
import jax
import pytest

from OmegaQMC.psi.nn.heg_wf import HEGConfig
from OmegaQMC.vmcopt_nn_heg import get_vmcopt_nn_heg_func
from OmegaQMC.afqmc_3deg import build_3deg_system, get_afqmc_3deg_func


def test_vmcopt_gradient_shape():
    """VMC gradient pytree has the same structure as params."""
    rs = 2.0
    sys = build_3deg_system(rs, N_elec=14, N_pw=7,
                            polarization='unpolarized')
    L = sys['L']
    config = HEGConfig(
        n_up=7, n_down=7, L=L, n_det=1, use_jastrow=True,
    )
    opt = get_vmcopt_nn_heg_func(
        config, jax.random.key(0), lr=1e-3,
        ewald_n_real=2, ewald_n_recip=4,
    )
    walkers = opt.initialize_walkers(jax.random.key(1), 8)
    grads, e_mean, e_var = opt._vmc_grad(walkers, opt.params)
    # Pytrees match structure.
    params_leaves = jax.tree_util.tree_leaves(opt.params)
    grad_leaves = jax.tree_util.tree_leaves(grads)
    assert len(params_leaves) == len(grad_leaves)
    for p, g in zip(params_leaves, grad_leaves):
        assert p.shape == g.shape
        assert np.all(np.isfinite(np.asarray(g)))


@pytest.mark.slow
def test_vmcopt_reduces_energy():
    """Adam-VMC should push the Jastrow-augmented energy below the
    free-electron (HF) energy within a few tens of iterations."""
    rs = 2.0
    N = 14
    sys = build_3deg_system(rs, N_elec=N, N_pw=7,
                            polarization='unpolarized')
    L = sys['L']

    e_hf_ha = float(get_afqmc_3deg_func(
        sys, dt=0.005, include_coulomb=True, verbose=False,
    ).e_trial) / N

    config = HEGConfig(
        n_up=7, n_down=7, L=L, n_det=1, use_jastrow=True,
    )
    opt = get_vmcopt_nn_heg_func(
        config, jax.random.key(0), lr=3e-3,
        ewald_n_real=3, ewald_n_recip=6,
    )
    result = opt(
        jax.random.key(1),
        num_iters=40,
        num_walkers=64,
        mcmc_decorr_steps=10,
        num_equil_steps=200,
        mc_timestep=0.1,
        verbose=0,
    )
    e_final_ha = result['E_final_ha']
    # Late-iteration energy should be below HF (correlation captured).
    # With only 40 Adam steps the drop is small but statistically
    # significant; require at least 1 mHa/elec below HF.
    assert e_final_ha < e_hf_ha - 1e-3, (
        f"Final E = {e_final_ha:.4f} Ha, HF = {e_hf_ha:.4f} Ha"
    )
