# Phase 2e Findings — H₂ Pilot Results and Factorized-Ansatz Limitation

**Date:** 2026-05-09
**Branch:** `cavity-qed-vmc-mol`
**Hardware:** mango GH200 (single GPU)
**Wall time per pilot:** ~60 s training + ~10 s eval

## Pilot 1 — λ=0 decoupling test (PASS)

**Config:** `inputs/qed_h2/h2_decoupling_pilot.yaml`
**Result file:** `logs/qed_h2_decoupling_pilot/qed_h2_decoupling_pilot.results.h5`

| Quantity | Value |
|---|---|
| E_VMC eval | **−1.171879 ± 0.001733 Ha** |
| ⟨n⟩ | 0.0000 (exact, vacuum-locked) |
| acc_r | 0.86 |
| acc_n | 0.00 (no Fock moves accepted at λ=0) |
| Reference (H₂ exact, CBS) | −1.17448 Ha |
| Difference | +2.6 mHa (~1.5 σ) |
| QED-FCI(STO-6G, λ=0) | −1.14592 Ha (basis-truncated; well above CBS) |

**Verdict:** decoupling test passes — λ=0 QED-NN-VMC reproduces standard NN-VMC behavior (vacuum photon, no n-moves, energy approaching exact H₂ ground state from above with PsiFormer's CBS ansatz). After only 100 SR iterations, we're already within 1.5σ of the CBS exact value.

## Pilot 2 — λ=0.1 cavity-coupled (REVEALS LIMITATION)

**Configs:** `inputs/qed_h2/h2_lambda_010.yaml` (α=0 frozen);
`inputs/qed_h2/_diag/h2_lambda_010_alpha05.yaml` (α=0.5 trainable)

| Run | α policy | E_VMC eval (Ha) | ⟨n⟩ | E(λ=0.1) − E(λ=0) (mHa) |
|---|---|---|---|---|
| α=0 frozen | none | −1.163461 ± 0.001617 | 0.0000 | **+8.42** |
| α=0.5 trainable | learned → 0.02 | −1.163056 ± 0.001973 | 0.0002 | **+8.82** |
| QED-FCI reference | (exact in STO-6G) | −1.150173 vs −1.145922 (λ=0) | 0.0016 | **−4.25** |

**Sign error.** Our VMC gives positive (destabilizing) cavity shift; QED-FCI gives negative (stabilizing). The two trainable-α and frozen-α runs give nearly identical results (α evolves to ≈0, since ⟨ε·d⟩=0 for symmetric H₂ → perturbative α optimum is 0).

## Root cause: factorized ansatz cannot represent bilinear coupling for symmetric systems

The factorized ansatz Ψ(r, n) = Ψ_e(r) · ⟨n|α⟩ has the property:

    Ψ(r, n+1) / Ψ(r, n)  =  α / √(n+1)
    Ψ(r, n−1) / Ψ(r, n)  =  √n / α

For α → 0 these become 0 and ∞ respectively, with the +1 ladder term vanishing
exactly. The Pauli-Fierz **bilinear coupling** local-energy contribution

    E_bilin = √(ω/2) · λ · (ε·d̂_e) · [√(n+1)·Ψ(r,n+1)/Ψ(r,n) + √n·Ψ(r,n−1)/Ψ(r,n)]

at sample n=0 reduces to √(ω/2)·λ·(ε·d_e)·[√1 · α + 0] = √(ω/2)·λ·(ε·d_e)·α.

For symmetric H₂: ⟨ε·d_e⟩ = 0 by inversion symmetry → ⟨E_bilin⟩ = 0 regardless of α.

What the factorized ansatz CAN capture for symmetric H₂:
  * Standard electronic kinetic + Coulomb (correct).
  * **Dipole self-energy** (1/2)λ²(ε·d_e)² — purely electronic c-number factor.
    This term is positive and destabilizing, exactly what we observe (~ +2.5 to +8 mHa).
  * Photon energy ω·n (= 0 since walkers stay in vacuum).

What it CANNOT capture for symmetric H₂:
  * Polariton-mediated **stabilization** via virtual photon dressing (the
    second-order perturbation London-Casimir-Polder term −λ²ω/4 · α_zz that
    Galego 2019 Eq. (16) predicts to be *negative*).
  * QED-FCI's −4.25 mHa stabilization comes from this mechanism, which the
    factorized ansatz at α=0 cannot represent.

## Why α-training doesn't rescue it

For symmetric H₂, ⟨ε·d̂_e⟩ = 0 is the *exact* expectation in the ground
state, so the perturbative α (Galego analytical formula
α = −λ⟨ε·d⟩/√(2ω)) equals 0 by symmetry. SR finds this fixed point and α
flows to 0 from any initial guess. The factorized ansatz simply lacks
*any* α value that captures the polariton-mediated correlations for a
symmetric system.

## Required architectural extension (Phase 2f)

To capture cavity-mediated effects for symmetric systems (which is
*all* of (H₂)₂, He₂, ethylene-dimer in our locked thesis), the NN
ansatz must depend explicitly on the photon Fock index *n*, not only
through the multiplicative coherent-state envelope:

    Ψ(r, n) = NN_θ(r, n − α) · χ(n; α)

(Tang et al. 2025 use a similar form: NN(r, n) with one-hot Fock
encoding and no coherent-state shift; we add the coherent shift via α.)
With NN_θ depending on n, the Fock-ladder ratios become
NN_θ(r, n+1) / NN_θ(r, n) — a learnable function — rather than a
fixed analytical form, and the bilinear coupling can be captured
non-perturbatively.

## Phase 2e status

* **Decoupling test**: PASSED. Production-quality H₂ ground state at λ=0.
* **Variational-bound check at finite λ**: REVEALED architectural
  limitation; QED-FCI sign disagrees with factorized ansatz prediction.
* **Haugland 2021 (H₂)₂ reproduction**: BLOCKED on Phase 2f
  (n-dependent NN).

## Phase 2f deliverable (proposed)

* Extend `OmegaQMC/psi/nn/qed_adapter.py` to support an
  `arch_n_aware=True` mode that injects photon-Fock index n as a
  one-hot input feature into the underlying PsiFormer/PauliNet
  ansatz, in addition to the coherent-state-shift envelope.
* Re-run pilots 1 (decoupling) and 2 (λ=0.1 H₂); the differential
  ΔE(0.1) − ΔE(0) should now match QED-FCI to within MC stderr.
* Then proceed to (H₂)₂ Haugland reproduction at production scale.

## Files preserved for replay

* `logs/qed_h2_decoupling_pilot/qed_h2_decoupling_pilot.results.h5`
* `logs/qed_h2_lambda_010/qed_h2_lambda_010.results.h5`
* `logs/qed_h2_lambda_010_alpha05/qed_h2_lambda_010_alpha05.results.h5`
* `inputs/qed_h2/h2_decoupling_pilot.yaml`
* `inputs/qed_h2/h2_lambda_010.yaml`
* `inputs/qed_h2/_diag/h2_lambda_010_alpha05.yaml`
