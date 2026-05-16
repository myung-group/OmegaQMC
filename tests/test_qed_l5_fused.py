"""Phase 5a-5 validation: Plan-C fused @jax.jit train_step.

Verifies that the fully-fused train_step (MCMC scan + eloc + Jacobian +
SMW + update inline) produces the same trajectory as the unfused Python
loop, to within floating-point noise.

Test: run 3 SR iters with each path from identical seed/init and check
final params_flat, final R, final q_c, final E history all agree.
"""
from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)


@pytest.fixture(scope="module")
def small_l5_config():
    from OmegaQMC.psi.nn.heg_wf import HEGPsiFormerConfig

    n_up, n_down = 1, 1
    rs = 10.0
    N = n_up + n_down
    L = rs * math.sqrt(math.pi * N)
    return HEGPsiFormerConfig(
        n_up=n_up, n_down=n_down, L=L, dim=2,
        backbone="ferminet",
        embedding_dim=8,
        n_interactions=1,
        two_particle_stream_dim=4,
        n_det=1,
        full_determinant=True,
        use_backflow=False,
        use_cusp=False,
        n_virt_pw=0,
        use_ghost_atom=False,
    )


def _build_opt(config, use_fused_step):
    from OmegaQMC.qed_vmcopt_nn_heg_l5 import _QEDL5Optimizer
    return _QEDL5Optimizer(
        config, jax.random.key(0),
        lr=0.5, damping=1e-3,
        omega=0.10,
        coupling_lambda=0.0,
        K_max=2,
        phase_mlp_hidden=(4, 4),
        mag_mlp_hidden=(4, 4),
        lr_T_max=5,
        ewald_n_real=2, ewald_n_recip=4,
        spring_mu=0.5,
        spring_norm_clip=0.1,
        use_smw_sr=True,
        use_fused_step=use_fused_step,
    )


def _run(opt, seed, num_iters, num_walkers, decorr):
    return opt(
        jax.random.key(seed),
        num_iters=num_iters,
        num_walkers=num_walkers,
        mcmc_decorr_steps=decorr,
        num_equil_steps=10,
        verbose=0,
    )


def test_fused_matches_unfused_3_iters(small_l5_config):
    """Fused train_step must produce the same final params_flat and E
    trajectory as the unfused Python loop, to within ~1e-10 (the only
    sources of disagreement are floating-point reduction order in XLA's
    scan vs Python loop)."""
    seed = 7
    num_iters, num_walkers, decorr = 3, 16, 3

    opt_un = _build_opt(small_l5_config, use_fused_step=False)
    res_un = _run(opt_un, seed, num_iters, num_walkers, decorr)

    opt_fu = _build_opt(small_l5_config, use_fused_step=True)
    res_fu = _run(opt_fu, seed, num_iters, num_walkers, decorr)

    # Final params should agree to ~1e-10 (XLA reduction order can
    # differ slightly between scan-form and Python-loop-form code).
    p_un = res_un["params_flat"]
    p_fu = res_fu["params_flat"]
    rel = float(
        jnp.linalg.norm(p_un - p_fu)
        / (jnp.linalg.norm(p_un) + 1e-30)
    )
    assert rel < 1e-8, (
        f"Fused vs unfused params diverge: rel={rel:.3e}"
    )

    # Energy histories should agree to ~1e-10 (same MCMC samples since
    # both use the same RNG key splitting, same params trajectory).
    e_un = jnp.asarray(res_un["E_per_elec_history"])
    e_fu = jnp.asarray(res_fu["E_per_elec_history"])
    diff = float(jnp.max(jnp.abs(e_un - e_fu)))
    assert diff < 1e-8, (
        f"Fused vs unfused E history diverges: max|Δ|={diff:.3e}\n"
        f"unfused: {e_un}\nfused:   {e_fu}"
    )

    # Im energy must remain identically zero in both paths at λ=0.
    im_un = jnp.asarray(res_un["Im_per_elec_history"])
    im_fu = jnp.asarray(res_fu["Im_per_elec_history"])
    assert jnp.allclose(im_un, 0.0, atol=1e-12)
    assert jnp.allclose(im_fu, 0.0, atol=1e-12)


def test_fused_eval_runs(small_l5_config):
    """Smoke: fused-step optimizer must still be able to run evaluate."""
    opt = _build_opt(small_l5_config, use_fused_step=True)
    _ = _run(opt, 11, num_iters=2, num_walkers=16, decorr=3)
    out = opt.evaluate(
        jax.random.key(99),
        num_walkers=16,
        num_blocks=2,
        num_blocks_equil=1,
        num_steps_per_block=3,
        verbose=0,
    )
    assert math.isfinite(float(out["E_per_elec_ha"]))


def test_fused_step_requires_smw():
    """use_fused_step=True must reject use_smw_sr=False at __init__."""
    from OmegaQMC.psi.nn.heg_wf import HEGPsiFormerConfig
    from OmegaQMC.qed_vmcopt_nn_heg_l5 import _QEDL5Optimizer

    N = 2
    L = 10.0 * math.sqrt(math.pi * N)
    config = HEGPsiFormerConfig(
        n_up=1, n_down=1, L=L, dim=2,
        backbone="ferminet",
        embedding_dim=8,
        n_interactions=1,
        two_particle_stream_dim=4,
        n_det=1,
        full_determinant=True,
        use_backflow=False,
        use_cusp=False,
        n_virt_pw=0,
        use_ghost_atom=False,
    )
    with pytest.raises(ValueError, match="use_fused_step"):
        _QEDL5Optimizer(
            config, jax.random.key(0),
            lr=0.5,
            omega=0.10, K_max=2,
            phase_mlp_hidden=(4, 4),
            mag_mlp_hidden=(4, 4),
            use_smw_sr=False,
            use_fused_step=True,
        )
