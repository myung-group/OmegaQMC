# L8 spec — Fock-basis ψ(R, n) ansatz

Status: design, not yet implemented (2026-05-20).
Cross-check for the L7 Lang-Firsov-biased ansatz on cavity-QED 2D HEG.
Tang et al. 2025 (arxiv 2503.15644) is the architectural inspiration —
no physical priors, just parametrize ψ_n(R) for n ∈ {0, …, N_max}.

---

## 1. Architecture: per-n head on shared trunk

Two viable choices; recommend **per-n head**.

| | Per-n head (recommended) | Tang-style one-hot append |
|---|---|---|
| Trunk | Run FermiNet once, get matter features h(R) | Re-run trunk (N_max+1) times with one-hot(n) appended to per-electron embeddings |
| Heads | (N_max+1) scalar outputs from mag MLP, (N_max+1) from phase MLP | Single head per pass |
| Per-iter cost | trunk + (N_max+1) cheap MLPs | (N_max+1) × full trunk |
| Code change | Surgical — parallels L5's mag_mlp/phase_mlp | Invasive — modifies FermiNet embedding layer |
| Expressivity | Each n shares matter features h(R); per-n MLPs map them | n can influence deep matter representation |

The expressivity gap is small for weak coupling and the cost gap is large
(~7× for N_max=6). Go with per-n head.

## 2. Parameter pytree

```python
init_params_pytree = {
    "e":         e_init_params,             # existing FermiNet (shared trunk)
    "mag_mlp":   per-n head (out_dim = N_max+1, real),
    "phase_mlp": per-n head (out_dim = N_max+1, real),
}
```

Trial:
```
ψ_n(R) = exp[ log_ψ_e(R) + mag_mlp_n(features) + i·phase_mlp_n(features) + offset_n ]
features = [ Σᵢ sin(K·rᵢ), Σᵢ cos(K·rᵢ), CoM ]     # same as L7, drops q_c
offset_n = 0 for n=0, −50 for n>0                  # large negative log → ψ_{n>0} ≈ 0 at init
```

`offset_n` is a **non-trainable** constant added to `mag_mlp_n` output so that
with zero-init last layers, ψ_0(R) ≈ ψ_HF(R) and ψ_{n>0}(R) ≈ 0. After training,
mag_mlp_n grows past offset_n where needed.

## 3. Local energy — explicit formula

At each walker R, ψ has (N_max+1) complex components ψ_n(R). The local energy is

```
E_loc(R) = ( Σ_{n,n'} ψ_n*(R) · H_{n,n'}(R) · ψ_{n'}(R) ) / Σ_n |ψ_n(R)|²
```

with H decomposed into

| Term | Matrix element |
|---|---|
| Matter kinetic + V_ee + v_ext | `H^matter_{n,n} ψ_n(R) = (T_e + V_ee + v_ext) ψ_n(R)` — diagonal in n |
| Photon HO (dressed) | `H^phot_{n,n} = (n + ½)·Ω_eff` — diagonal, no matter operator |
| Bilinear −λ q_c (ε̂·P̂_tot) | `H^cpl_{n,n+1} ψ_{n+1} = −λ·√(1/(2Ω_eff))·√(n+1)·(−i)·(ε̂·∇)ψ_{n+1}(R)` and Hermitian conjugate |

**What this needs computationally per walker:**
- Matter ∇² ψ_n for each n (Laplacian, real + imag parts) — but the
  Laplacian only acts on each ψ_n separately, so it's (N_max+1) Hessian
  extractions of the per-n complex log Ψ. Can fuse: compute ψ_n, ∇ψ_n,
  ∇²ψ_n for all n in one vmapped pass.
- ε̂·∇ψ_n for the bilinear (already a byproduct of ∇ψ_n).

**Cost compared to L7:** L7 needed ∇² of one log Ψ (matter + photon coupled).
L8 needs ∇² of (N_max+1) per-n components. With trunk-sharing, the matter
Laplacian work is mostly redundant — opportunity to share the matter Hessian
across n via reverse-mode AD over a vector output. At N_max=6 expect ~2–3× the
L7 per-iter cost on GH200.

## 4. MCMC

Just R — q_c is gone. The sampling density is

```
π(R) ∝ Σ_n |ψ_n(R)|²
```

Use the existing R-Metropolis machinery from L5 with
`log π(R) / 2 = ½ · log Σ_n |ψ_n(R)|² = log|Ψ(R)|`. The q_c chain disappears
entirely; one fewer step type, faster decorrelation.

## 5. SR generalization

Define the natural log-derivative

```
∂_θ log Ψ(R) ≡ ( Σ_n ψ_n*(R) · ∂_θ ψ_n(R) ) / Σ_n |ψ_n(R)|²
```

This is the scalar-vector generalization of `∂_θ log Ψ = ∂_θ Ψ / Ψ`. With this
definition, all SR primitives (S_θθ′ = Re⟨ΔO*·ΔO′⟩, g_θ = 2 Re⟨ΔO*·ΔE_loc⟩, SMW
dual form, SPRING-style momentum) carry over **verbatim** from L5/L7. No changes
to the SR loop, only to how `O_θ(R)` is computed.

The Hermiticity diagnostic (Im⟨E_loc⟩ → 0) survives identically.

## 6. Module layout

```
OmegaQMC/qed_vmcopt_nn_heg_fock.py        ← new (~500 lines)
    build_fock_log_psi(...)                 vector-valued (R,n) trial
    make_fock_eloc(...)                     local energy w/ Fock sum
    make_fock_sr_primitives(...)            SR with vector-O_θ
    class _QEDFockOptimizer                  pattern from _QEDL5Optimizer,
                                             drops q_c MCMC + photon HO sampling

scripts/run_qed_fock_heg.py               ← new (~200 lines, driver pattern
                                              matches run_qed_l5_heg.py)

inputs/2dheg_qed/l8_weber_fig1b_v1_off.yaml   ← N_max=6 OFF baseline
inputs/2dheg_qed/l8_weber_fig1b_v1_on.yaml    ← N_max=6 ON at γ=0.5

scripts/slurm_qed/run_l8.sh                ← thin wrapper, mirror of run_l5.sh
```

## 7. Validation checkpoints

1. **Zero-iter at λ=0, N_max=2**: ψ_0=HF, ψ_{1,2}≈0 → E = E_HF + ½Ω.
   Confirms vacuum trial is correct.
2. **Zero-iter at λ≠0, N_max=2**: E = E_HF + ½Ω_eff + numerical noise from
   bilinear acting on near-zero ψ_1. Tests that the off-diagonal coupling
   code is right.
3. **N_max sweep at λ=0**: E should be insensitive to N_max (matter and
   photon decouple). Catches off-by-one bugs in the n-indexing.
4. **N_max sweep at γ=0.5**: E should plateau by some N_max* (estimated 4–8).
   The plateau value is the L8 answer to compare against L7.
5. **Optional**: project the L7 trained state into Fock basis (compute
   ⟨n|χ(q_c)⟩ for each n) and use as warmstart for L8. If L8 strictly improves
   from this warmstart, it confirms L7 was missing structure.

## 8. Cost estimate

- **Memory**: per-walker state = R (18×2 floats) + (N_max+1) complex amplitudes
  — negligible.
- **Compute per iter, N_max=6**: trunk forward 1× (dominant in L7), per-n
  heads 7×, per-n Laplacians 7× (shared trunk gradient amortizes most).
  Expect ~2–3× wall time vs L7 on GH200, so iters/hour ≈ V11 / 2.5.
- **Walkers**: keep 2048 — variance benefit at γ=0.5 is well-established.
- **Iters to convergence**: similar 30k iters as V11; same SR + cosine LR
  schedule.

## 9. Open design questions

1. **Coupling op for diamagnetic ½λ²(ε̂·P̂)² term**: in L5/L7 we absorbed
   `Ω_eff² = Ω² + N·λ²` to handle this. In Fock basis with the *dressed*
   photon at Ω_eff, the diamagnetic shows up as a constant offset to the
   matter Hamiltonian — needs careful re-derivation. Probably an extra
   `+½λ²·⟨(ε̂·P̂_tot)²⟩` term acting on matter. Worth a sanity check on the
   L7 derivation while we're at it.
2. **Phase initialization**: zero-init phase MLP gives ψ_n real-positive,
   which matches HF for n=0. For n>0 this is fine since ψ_{n>0} ≈ 0 anyway.
3. **Warmstart from L7**: doable but adds glue code. Defer to v2 unless V11
   result suggests we need a head-start.
4. **N_max as a config**: should be a YAML field with default 6; the
   validation sweep is the user-facing knob.

## 10. Implementation order

```
Step 1  build_fock_log_psi + per-n heads + offset_n         [4 hr — surgical, mirrors L5]
Step 2  compute_psi_vec, ⟨n⟩, π(R) helpers + zero-iter test [2 hr]
Step 3  Local energy (matter + photon HO + bilinear sums)   [6 hr — most work, careful derivation]
Step 4  Vector-O_θ SR primitives + smoke that L7 numerics  [3 hr]
        recover when bilinear off + N_max=0
Step 5  Driver + YAML + slurm wrapper                       [2 hr]
Step 6  Validation sweep (λ=0 N_max=0..2, λ≠0 N_max=2..8)  [run time, ~1 day on GH200]
```

Total dev time ~2 days focused; validation overnight.

## 11. Recommended next move

Hold L8 code start until V11 ON gives a first signal at iter ~3000–5000
(couple of hours). If V11 trends toward the PT prediction, L8 becomes the
cross-check; if not, L8 becomes the primary candidate and we'll want a fresh
ansatz anyway.

---

## Appendix A — relation to L5/L6/L7

| Level | Photon basis | Inductive bias | Captures |
|---|---|---|---|
| L5 | continuous q_c, bare HO χ(q_c) | mag_mlp(R, q_c), phase_mlp(R, q_c) MLPs | dressed HO + arbitrary residual correlation, but optimizer must discover the bilinear LF structure inside the MLPs |
| L6 | continuous q_c, χ(q_c − Q₀(R)) | adds real coherent shift | LF mean field in length gauge (coupling_op=X); zero gradient for coupling_op=P |
| L7 | continuous q_c, χ(q_c − Q₀(R)) · e^{i P₀(R)·q_c} | adds Q₀ + P₀ | LF mean field in both gauges; squeezing optional via S(R) |
| L8 | discrete n, vector ψ_n(R) | none (or one-hot Tang style) | arbitrary entanglement up to N_max; no inductive bias |

L8 is strictly more general than L5/L6/L7 in the N_max → ∞ limit. For finite N_max, it can miss high-n weight that L5–L7 capture exactly via the Gaussian tail.

## Appendix B — references

- Tang, Andolina, Cuzzocrea, et al. *Deep quantum Monte Carlo approach for
  polaritonic chemistry* (2503.15644v1, March 2025) — Tang-style architecture.
- Weber et al. PRL 135, 126901 (2024) — the QED-AFQMC benchmark we're chasing.
- Andolina et al. — single-Γ-mode no-go theorem (relevant to bulk phase
  questions, not directly to this ansatz).
