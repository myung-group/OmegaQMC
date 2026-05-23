"""L8 fused vs unfused per-iter benchmark.

30 iters at small walker count to time compile + per-iter cost both ways.
"""
import math, sys, jax, jax.numpy as jnp, time
jax.config.update("jax_enable_x64", True)
sys.path.insert(0, ".")

from OmegaQMC.psi.nn.heg_wf import HEGPsiFormerConfig
from OmegaQMC.qed_vmcopt_nn_heg_fock import _QEDFockOptimizer

L = 1.5958 * math.sqrt(math.pi * 18)
cfg = HEGPsiFormerConfig(
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
        cfg, jax.random.key(42),
        lr=0.005, damping=1e-3,
        omega=0.1, coupling_lambda=0.02357,
        coupling_polarization=[1.0, 0.0], coupling_op="P",
        v_ext_amp=0.3, v_ext_a=3.0,
        include_vee=False,
        N_max=4, K_max=6,
        mag_mlp_hidden=(64, 64), phase_mlp_hidden=(64, 64),
        lr_schedule="cosine", lr_min=1e-5, lr_T_max=30,
    )

walkers = 128
n_iters = 30
mcmc = 5

print(f"--- {n_iters} iters, walkers={walkers}, mcmc={mcmc}, N_max=4 ---\n")

print("=== UNFUSED (Python-orchestrated) ===")
opt = build_opt()
t0 = time.time()
opt.train(jax.random.key(0), num_walkers=walkers, n_iters=n_iters,
          mcmc_decorr_steps=mcmc, equil_steps=10, verbose=0)
print(f"  total: {time.time()-t0:.1f}s, per-iter avg: {(time.time()-t0)/n_iters:.2f}s\n")

print("=== FUSED (single JIT) ===")
opt = build_opt()
t0 = time.time()
opt.train_fused(jax.random.key(0), num_walkers=walkers, n_iters=n_iters,
                mcmc_decorr_steps=mcmc, equil_steps=10, verbose=0)
print(f"  total: {time.time()-t0:.1f}s, per-iter avg: {(time.time()-t0)/n_iters:.2f}s\n")
print("(compile time dominates total at n_iters=30; per-iter only meaningful after first)")
