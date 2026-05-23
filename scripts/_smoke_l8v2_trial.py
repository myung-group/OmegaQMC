"""L8 V2 (Tang) trial + eloc + SR validation."""
import math, sys, jax, jax.numpy as jnp, numpy as np
jax.config.update("jax_enable_x64", True)
sys.path.insert(0, ".")

from OmegaQMC.psi.nn.heg_wf import HEGPsiFormerConfig
from OmegaQMC.qed_vmcopt_nn_heg_tang import (
    build_tang_trial,
    make_tang_eloc_no_vee,
    make_tang_sr_primitives,
    _laplacian_vmap,
)

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
omega, lam = 0.1, 0.02357
N = 18
omega_eff_off = omega
omega_eff_on  = math.sqrt(omega**2 + N * lam**2)
eps = jnp.array([1.0, 0.0], dtype=jnp.float64)

print(f"--- L8 V2 trial + eloc smoke (N_max={N_max}) ---\n")
trial = build_tang_trial(
    cfg, jax.random.key(42),
    N_max=N_max, phase_mlp_hidden=(64, 64),
    offset_floor=-50.0,
)
print(f"n_params total = {trial['n_params']}, electronic = {trial['n_electronic']}")
print(f"N_max = {trial['N_max']}, offset = {np.asarray(trial['offset'])}")

# Test 1: psi_vec at init — ψ_0 ≈ ψ_HF, ψ_{n>0} ≈ 0
R_walkers = jax.random.uniform(jax.random.key(7), (4, 18, 2), maxval=L)
p = trial["init_params_pytree"]
psi_v = jax.vmap(lambda r: trial["psi_vec"](r, p))(R_walkers)
print(f"\n=== Test 1: psi_vec at init ===")
print(f"  walker 0:  |ψ_n| = {[f'{float(jnp.abs(z)):.3e}' for z in psi_v[0]]}")
print(f"  walker 0:  arg ψ_n = {[f'{float(jnp.angle(z)):+.3e}' for z in psi_v[0]]}")
print(f"  ratio |ψ_1|/|ψ_0| (should be exp(-50) ≈ 1.9e-22):")
for w in range(4):
    print(f"    walker {w}: {float(jnp.abs(psi_v[w, 1]) / jnp.abs(psi_v[w, 0])):.4e}")

# Test 2: eloc at λ=0 — matches matter kinetic + Ω/2
print(f"\n=== Test 2: eloc at λ=0 ===")
eloc_off = make_tang_eloc_no_vee(
    trial["psi_vec"], eps=eps, lam=0.0, omega_eff=omega_eff_off,
    N_max=N_max, nelec=18, dim=2,
)
re_off = jax.vmap(lambda r: eloc_off(r, p)[0])(R_walkers)
im_off = jax.vmap(lambda r: eloc_off(r, p)[1])(R_walkers)
print(f"  walker:  re_eloc          im_eloc")
for w in range(4):
    print(f"    {w}:    {float(re_off[w]):+.6e}   {float(im_off[w]):+.3e}")
print(f"  max |im|: {float(jnp.max(jnp.abs(im_off))):.3e}  (Hermiticity at λ=0)")

# Test 3: eloc at λ≠0 with vacuum trial (≈ same as Test 2 + ZPE shift)
print(f"\n=== Test 3: eloc at λ={lam} (vacuum init) ===")
eloc_on = make_tang_eloc_no_vee(
    trial["psi_vec"], eps=eps, lam=lam, omega_eff=omega_eff_on,
    N_max=N_max, nelec=18, dim=2,
)
re_on = jax.vmap(lambda r: eloc_on(r, p)[0])(R_walkers)
im_on = jax.vmap(lambda r: eloc_on(r, p)[1])(R_walkers)
diff = re_on - re_off
print(f"  re_on - re_off (should be ≈ (Ω_eff − Ω)/2 = {(omega_eff_on - omega)/2:+.4f}):")
for w in range(4):
    print(f"    walker {w}: {float(diff[w]):+.6e}")

# Test 4: SR Jacobian shape sanity
print(f"\n=== Test 4: SR Jacobian ===")
sr = make_tang_sr_primitives(trial["psi_vec_flat"], nelec=18, dim=2)
R_flat = R_walkers.reshape(R_walkers.shape[0], -1)
p_flat = trial["init_params_flat"]
import time
t0 = time.time()
Jac_u, Jac_v = sr["batched_jacobian"](R_flat, p_flat)
dt = time.time() - t0
print(f"  shapes: Jac_u={Jac_u.shape}, Jac_v={Jac_v.shape}")
print(f"  compile+run wall: {dt:.2f}s")
print(f"  Jac_u norm per walker: {[f'{float(n):.3f}' for n in jnp.linalg.norm(Jac_u, axis=1)]}")
print(f"  Jac_v norm per walker: {[f'{float(n):.3f}' for n in jnp.linalg.norm(Jac_v, axis=1)]}")
