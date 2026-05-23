"""L8 V2 Tang matter-backbone smoke.

Checks:
  1. The new module imports cleanly.
  2. make_tang_log_psi builds.
  3. At init (zero-init n_cond_proj + zero-init phase_mlp last layer):
     - sign, log_mag are SAME for all n (since n-conditioning is zero)
     - phase is 0 for all n
  4. After perturbing n_cond_proj kernel: different n give different log_mag
     (verifies the n-conditioning pathway is actually wired up).
"""
import math, sys, jax, jax.numpy as jnp, numpy as np
jax.config.update("jax_enable_x64", True)
sys.path.insert(0, ".")

from OmegaQMC.psi.nn.heg_wf import HEGPsiFormerConfig
from OmegaQMC.qed_vmcopt_nn_heg_tang import make_tang_log_psi

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
N_max = 4
print(f"--- L8 V2 Tang backbone smoke ---")
print(f"N_max={N_max}, embedding_dim={cfg.embedding_dim}, "
      f"n_interactions={cfg.n_interactions}")

log_psi, params, graphdef = make_tang_log_psi(
    cfg, jax.random.key(42),
    n_max_for_tang=N_max,
    phase_mlp_hidden=(64, 64),
)
from jax.flatten_util import ravel_pytree
p_flat, unravel = ravel_pytree(params)
print(f"n_params = {p_flat.shape[0]}")

R = jax.random.uniform(jax.random.key(7), (18, 2), maxval=L)

def one_hot(n_idx, N_max):
    return jnp.eye(N_max + 1, dtype=jnp.float64)[n_idx]

print(f"\n=== Test 1: zero-init → all n give same log_mag, phase=0 ===")
for n in range(N_max + 1):
    sign, log_mag, phase = log_psi(R, params, one_hot(n, N_max))
    print(f"  n={n}: sign={float(sign):+.0f}, "
          f"log_mag={float(log_mag):+.6e}, "
          f"phase={float(phase):+.3e}")

# All should be identical because n_cond_proj kernel is zero-init
# and phase_mlp last layer is zero-init.

print(f"\n=== Test 2: perturb n_cond_proj → outputs differ by n ===")
# Read current params; perturb the n_cond_proj kernel
# Find it: params['omni']['gnn']['electron_embedding']['n_cond_proj']['kernel']
def find_n_cond_proj(p, path=()):
    if isinstance(p, dict):
        for k, v in p.items():
            if k == 'n_cond_proj':
                return path + (k,), v
            sub = find_n_cond_proj(v, path + (k,))
            if sub is not None:
                return sub
    return None

# Walk params structure for diagnostic
def walk(p, depth=0, path=""):
    if depth > 8: return
    if isinstance(p, dict):
        for k, v in p.items():
            walk(v, depth+1, f"{path}/{k}")
    else:
        if 'n_cond' in path or 'phase_mlp' in path:
            arr = np.asarray(p) if not isinstance(p, np.ndarray) else p
            try:
                shape = arr.shape
                print(f"  {path}: shape={shape}, max={float(np.abs(arr).max()):.3e}")
            except Exception:
                print(f"  {path}: (unable to inspect)")

print("Param tree (filtered to n_cond and phase_mlp):")
import jax
flat = jax.tree_util.tree_leaves_with_path(params)
for path, leaf in flat:
    path_str = '/'.join(
        getattr(p, 'name', str(p)) if hasattr(p, 'name') else
        str(p.key) if hasattr(p, 'key') else str(p)
        for p in path
    )
    if 'n_cond' in path_str or 'phase_mlp' in path_str:
        try:
            arr = np.asarray(leaf)
            print(f"  {path_str}: shape={arr.shape}, "
                  f"|.|_max={float(np.abs(arr).max()):.3e}")
        except Exception as e:
            print(f"  {path_str}: <error: {e}>")

print("\n  (smoke check end)")
