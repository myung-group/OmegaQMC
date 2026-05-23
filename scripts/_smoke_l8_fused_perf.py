"""L8 fused per-iter cost — measure steady-state after compile.

100 iters total; report mean of iters 30-99 (post-compile, post-warmup).
Also tries production-scale 1024 walkers to estimate full-run cost.
"""
import math, sys, jax, jax.numpy as jnp, time, numpy as np
jax.config.update("jax_enable_x64", True)
sys.path.insert(0, ".")

from OmegaQMC.psi.nn.heg_wf import HEGPsiFormerConfig
from OmegaQMC.qed_vmcopt_nn_heg_fock import _QEDFockOptimizer

L = 1.5958 * math.sqrt(math.pi * 18)
cfg_e = HEGPsiFormerConfig(
    n_up=9, n_down=9, L=L, dim=2,
    backbone="ferminet", embedding_dim=64, n_interactions=3,
    two_particle_stream_dim=16, n_det=1, full_determinant=True,
    use_backflow=True, use_cusp=True,
    n_virt_pw=0, use_ghost_atom=True,
    use_deep_jastrow=False, use_smith_deep_jastrow=True,
    envelope_type="plane_wave",
)

def build_opt():
    return _QEDFockOptimizer(
        cfg_e, jax.random.key(42),
        lr=0.005, damping=1e-3,
        omega=0.1, coupling_lambda=0.02357,
        coupling_polarization=[1.0, 0.0], coupling_op="P",
        v_ext_amp=0.3, v_ext_a=3.0,
        include_vee=False,
        N_max=4, K_max=6,
        mag_mlp_hidden=(64, 64), phase_mlp_hidden=(64, 64),
        lr_schedule="cosine", lr_min=1e-5, lr_T_max=100,
    )

def time_fused(walkers, n_iters, mcmc):
    opt = build_opt()
    fused = opt._build_fused_train_step(walkers, mcmc)
    key = jax.random.key(0)
    key_i, key_t = jax.random.split(key)
    R = opt.initialize_walkers(key_i, walkers)
    step_R = jnp.float64(0.1)
    pf = opt.params_flat
    pd = jnp.zeros_like(pf)
    # equilibrate
    for _ in range(20):
        key_t, sub = jax.random.split(key_t)
        R, step_R, _ = opt._mcmc_step_uncompiled(sub, R, step_R, pf, walkers)
    carry = (key_t, R, step_R, pf, pd)
    times = []
    for it in range(n_iters):
        t0 = time.time()
        carry, metrics = fused(carry, jnp.float64(0.005))
        metrics["e_mean"].block_until_ready()
        times.append(time.time() - t0)
    return times

for walkers, mcmc in [(128, 5), (256, 5), (512, 5), (1024, 5)]:
    print(f"\n=== walkers={walkers}, mcmc={mcmc}, N_max=4 ===")
    times = time_fused(walkers, 80, mcmc)
    compile_t = times[0]
    steady = np.array(times[20:])    # skip warmup
    print(f"  compile (iter 1): {compile_t:.1f}s")
    print(f"  steady mean  (iters 21-80): {steady.mean():.3f}s")
    print(f"  steady median (iters 21-80): {np.median(steady):.3f}s")
    print(f"  steady min/max: {steady.min():.3f}s / {steady.max():.3f}s")
    print(f"  → 30000 iters extrapolation: {(compile_t + 30000*steady.mean())/3600:.1f}h")
