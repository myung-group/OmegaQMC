"""L8 V2 (Tang) fused JIT per-iter benchmark.

Sweep walkers ∈ {128, 256, 512, 1024} at mcmc=5, N_max=4.
60 iters per setting; report steady mean over iters 20-60.
"""
import math, sys, jax, jax.numpy as jnp, time, numpy as np
jax.config.update("jax_enable_x64", True)
sys.path.insert(0, ".")

from OmegaQMC.psi.nn.heg_wf import HEGPsiFormerConfig
from OmegaQMC.qed_vmcopt_nn_heg_tang import _QEDTangOptimizer

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

def build_opt(N_max=2, offset_floor=-5.0):
    return _QEDTangOptimizer(
        cfg_e, jax.random.key(42),
        lr=0.005, damping=1e-3,
        omega=0.1, coupling_lambda=0.02357,
        coupling_polarization=[1.0, 0.0], coupling_op="P",
        v_ext_amp=0.3, v_ext_a=3.0,
        include_vee=False,
        N_max=N_max, phase_mlp_hidden=(64, 64),
        offset_floor=offset_floor,
        lr_schedule="cosine", lr_min=1e-5, lr_T_max=60,
    )

def time_fused(walkers, n_iters, mcmc, N_max=2):
    opt = build_opt(N_max=N_max)
    fused = opt._build_fused_train_step(walkers, mcmc)
    key = jax.random.key(0)
    k_i, k_t = jax.random.split(key)
    R = opt.initialize_walkers(k_i, walkers)
    step_R = jnp.float64(0.1)
    pf = opt.params_flat
    pd = jnp.zeros_like(pf)
    for _ in range(20):
        k_t, sub = jax.random.split(k_t)
        R, step_R, _ = opt._mcmc_step_uncompiled(sub, R, step_R, pf, walkers)
    carry = (k_t, R, step_R, pf, pd)
    times = []
    for it in range(n_iters):
        t0 = time.time()
        carry, metrics = fused(carry, jnp.float64(0.005))
        metrics["e_mean"].block_until_ready()
        times.append(time.time() - t0)
    return times

N_max = 2   # new default after psi_n_only refactor
for walkers in [128, 256, 512, 1024]:
    print(f"\n=== L8 V2 Tang fused: walkers={walkers}, mcmc=5, N_max={N_max} ===")
    times = time_fused(walkers, 50, 5, N_max=N_max)
    compile_t = times[0]
    steady = np.array(times[10:])
    print(f"  compile (iter 1): {compile_t:.1f}s")
    print(f"  steady mean (iter 11-50): {steady.mean():.3f}s")
    print(f"  steady median: {np.median(steady):.3f}s")
    print(f"  steady min/max: {steady.min():.3f}s / {steady.max():.3f}s")
    print(f"  → 30k iters extrapolation: "
          f"{(compile_t + 30000*steady.mean())/3600:.1f}h")
    # L8 V1 fused at same walker counts (from earlier perf bench), mcmc=5:
    l8v1_pwalker = {128: 0.067, 256: 0.123, 512: 0.228, 1024: 0.496}
    if walkers in l8v1_pwalker:
        ratio = steady.mean() / l8v1_pwalker[walkers]
        print(f"  Tang(N_max={N_max}) / L8 V1 fused ratio: {ratio:.2f}× "
              f"(expected ~{(N_max+1)/5:.2f}× from trunk-fwd count)")
