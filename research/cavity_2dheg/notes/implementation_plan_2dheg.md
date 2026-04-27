# Implementation Plan: 2D HEG with PsiFormer-VMC

Goal: extend OmegaQMC's existing 3D HEG infrastructure to 2D, validate against the Attaccalite 2002 benchmarks, and reach the Wigner-crystal regime. This is **Phase 0** of the cavity-2DEG project — pure 2D, no cavity, just clean 2D HEG ground states.

Realistic timeline: **3–4 months** of focused work to a publishable 2D HEG benchmark, including Wigner crystal phase boundary reproduction.

---

## STATUS UPDATE — 2026-04-26 (autonomous-overnight session)

**Phase 0 is COMPLETE** (all infrastructure written + tested locally):

- ✅ 2D Ewald (Parry/Heyes) — `OmegaQMC/observables/ewald_2d.py` (19 tests)
- ✅ Analytical 2D HF (TD + finite-N) — `OmegaQMC/heg_2d.py` (60 tests)
- ✅ 2D periodic primitives (`make_square_lattice`) — `OmegaQMC/psi/nn/periodic.py` (12 tests)
- ✅ 2D plane-wave envelope (real + complex) — `OmegaQMC/psi/nn/env_periodic.py` (21 tests)
- ✅ `dim` threaded through `HEGPsiFormerConfig`, `HEGSlaterJastrow`, `HEGElectronEmbedding`,
  GNN graph builder, SR/Adam optimizers, eval driver, pretrain — full 2D path works
- ✅ Observables module (S(k), S_spin(k), g(r), n(k)) — `OmegaQMC/observables/structure_factor.py` (9 tests)
- ✅ End-to-end smoke test passes in 15 s on CPU at rs=2 N=10 unpol
- ✅ Convergence demonstrated: 500-iter SR run on CPU at rs=2 N=10 unpol gives
  **E/N = −0.26305(42) Ha/elec** vs analytical HF = −0.18797 Ha/elec.  Correlation
  energy recovered = **−75 mHa**, consistent with the expected ~80 mHa for the 2D HEG
  at rs=2 (Attaccalite TD limit).  Total wall-clock: 6 min 40 s on CPU.

**Phase 1 is partially COMPLETE** (twist-averaging infrastructure):

- ✅ `generate_halton_twists_2d` in `OmegaQMC/heg_2d.py`
- ✅ `run_twist_averaged_heg` dispatches on `config.dim`
- ✅ Complex 2D PsiFormer with twist `kappa` builds and evaluates (4 tests)
- ⏳ Production TABC sweeps with N=58 + N=74 (need GPU; YAMLs ready)

**Phase 2 is partially COMPLETE** (Wigner-crystal infrastructure):

- ✅ `GaussianLocalizedEnvelope2D` with variational lattice + sigma — `OmegaQMC/psi/nn/env_localized_2d.py` (6 tests)
- ✅ Crystal envelope wired into `build_heg_psiformer_wf` via `envelope_type='crystal_gaussian'`
- ✅ End-to-end crystal pipeline runs in 7 s on CPU at rs=30 N=10
- ⏳ Production crystal scan rs∈{25,30,32,35,40} (need GPU; YAMLs ready)

**Validation results** (all on CPU):

- Test 0 (kinetic): 2D PsiFormer plane-wave Slater determinant matches analytical
  `T/N = 1/(2 r_s^2)` to *floating-point precision* at every tested rs.
- Test 1 (HF baseline): MCMC-sampled `<T+V>` with envelope-only PsiFormer matches
  analytical finite-N HF to within statistical error (~few mHa with 128 walkers).
- 2D Madelung matches Bonsall-Maradudin 1977 `−1.100244/r_s` (square) to ~1e−6.
- 3D regression: all 156 pre-existing 3D tests still pass.

**Production-ready YAMLs** (in `inputs/2dheg/`):

- Density scan: rs ∈ {1, 2, 5, 10, 20}, N=58 unpolarized → Attaccalite reference
- Polarized: rs ∈ {1, 2, 5, 10}, N=57 polarized
- Quick validation: rs ∈ {1, 2, 5}, N=10 unpolarized (~30 min on A100 each)
- Wigner crystal: rs ∈ {25, 30, 32, 35, 40}, N=18 (fluid + crystal sectors)

**Slurm scripts** (in `scripts/slurm_2dheg/`):

- `run_one.sh` — single-job 2D HEG runner
- `launch_density_scan.sh` — submit all density-scan jobs
- `launch_quick_validation.sh` — submit N=10 quick checks
- `launch_crystal_scan.sh` — submit Phase 2 crystal scan

**Tests passing: 181** in the relevant test files (44 pre-existing 3D HEG/periodic + 137 new 2D + observables + crystal + twist + pipeline).
Specifically:

| File | Tests | Coverage |
|---|---|---|
| `test_ewald_2d.py` | 19 | 2D Ewald + Madelung vs Bonsall-Maradudin |
| `test_heg_2d.py` | 60 | system builder + analytical HF + closed shells |
| `test_periodic_2d.py` | 12 | square lattice + 2D periodic primitives |
| `test_env_periodic_2d.py` | 21 | 2D plane-wave envelopes (real + complex) |
| `test_env_localized_2d.py` | 6 | Wigner crystal Gaussian envelope |
| `test_structure_factor.py` | 9 | S(k), g(r), n(k) accumulators |
| `test_twist_2d.py` | 4 | 2D Halton twists + complex PsiFormer at twist |
| `test_2dheg_pipeline.py` | 6 | end-to-end PsiFormer + Ewald validation |
| **Total new 2D tests** | **137** | |

---

## 0. Updates from the planning conversation (added 2026-04-25)

Three additions to the plan since first draft:

**(a) Observables module is now a Phase 0 week-3 deliverable, not "nice to have."**
Phase boundaries are confirmed by *two* independent signals: (i) crossing of converged variational energies between competing-symmetry ansatze, and (ii) order parameters accumulated during sampling. We need (ii) from the very first runs to verify the optimizer did not get stuck in the wrong basin.

New file: `OmegaQMC/observables/structure_factor.py` — accumulates during sampling:
- `S(k) = (1/N)⟨|Σᵢ exp(i k·rᵢ)|²⟩` — Bragg peaks → which Bravais lattice we are in
- `S_spin(k) = (1/N)⟨|Σᵢ σᵢ exp(i k·rᵢ)|²⟩` with σᵢ = ±1 — magnetic order
- `g(r)` pair correlation
- `n(k)` momentum distribution → renormalization factor Z (cross-check vs Holzmann 2009)

These are written to `summary.json` at end of run. ~200 lines, ~3 days.

**(b) Variational lattice vectors in `GaussianLocalizedEnvelope2D` (Phase 2).**
Do *not* hard-code a triangular lattice. Make the Bravais primitive vectors `(a₁, a₂)` 3 extra variational parameters (init triangular). At each (rs, λ), the converged lattice tells us which Bravais sector we landed in.

This is essential for the cavity case: cavity-induced anisotropy distorts the Wigner crystal (centered rectangular → rectangular → smectic stripe → anisotropic fluid as λ grows). Hard-coding triangular would prevent us from seeing the anisotropic phases that are likely the most novel results.

**(c) Cavity polarization choice for Phase 3: linear, in-plane, along x̂.**
Matches Weber et al. 2025 (PRL) and the dominant experimental cavity geometry (split-ring resonators in the Faist/Imamoğlu groups give linear in-plane polarization along the resonator gap).

The cavity Hamiltonian in dipole gauge (PZW, the Tang 2025 recipe — better convergence than velocity gauge):

```
H = H_HEG + ω(a†a + 1/2) + iωλ(a − a†)·Dₓ + ½(λDₓ)²
```

where `Dₓ = Σᵢ xᵢ` is the x-component of the total electron dipole. Note `ε = ẑ` (out-of-plane) is the *wrong* choice — couples to nothing in the 2D plane.

**Key advantage over Weber 2025:** they had to add an artificial cosine modulating potential `v_ext(r) = −v Σ_d cos(2π e_d·r/a)` to break translational symmetry and get any cavity-electron correlation. We do *not* — Coulomb-driven Wigner crystallization breaks translation spontaneously, so cavity-electron correlations appear without external help. This is the cleanest one-line argument for why our project is non-trivially different from Weber's.

---

## 1. What carries over directly from 3D code (no changes)

These modules work in 2D unchanged because they're dimension-agnostic:

| File | Why it transfers |
|---|---|
| `OmegaQMC/psi/nn/heg_psiformer.py` (PsiFormer body) | Attention/GNN operates on tokens, independent of spatial dim |
| `OmegaQMC/psi/nn/heg_wf.py` (PsiFormer wavefunction wrapper) | Slater det, Jastrow, cusp logic |
| `OmegaQMC/vmcopt_nn_heg_sr.py` (SR optimizer) | Optimization is independent of physics |
| `OmegaQMC/vmcopt_nn_heg.py` (VMC driver — Adam) | Same |
| `OmegaQMC/vmc_nn_heg.py` (evaluation driver) | Same |
| `OmegaQMC/pretrain_heg.py` (supervised pretraining) | Drives backflow toward HF; HF is dim-agnostic |
| `scripts/run_heg_psiformer.py` (YAML-config CLI) | Just need a `dim: 2` flag |

So roughly **70% of the code transfers verbatim**.

---

## 2. What needs 2D-specific modification

### 2.1 The lattice (`psi/nn/periodic.py`)

Currently 3D simple cubic. Need 2D square lattice as a fork.

**Changes**:
- New `make_square_lattice(L)` returning a `PeriodicLattice` with 2×2 metric `S = L²·I`, "volume" → area `A = L²`.
- `fractional_coords(r, lattice)`: works as-is if we use a 2-component `r`.
- `wrap_to_cell`, `minimum_image_diff`, `periodic_norm`, `periodic_norm_sq`: dimension-agnostic if we operate on `(..., d)` arrays — already done; just verify with d=2.

**Effort**: ~1 day. Mostly tests.

### 2.2 The Ewald summation (`observables/ewald.py`)

This is the biggest change. **2D Ewald with 1/r interaction is qualitatively different from 3D Ewald** because the reciprocal-space sum has a different functional form.

**3D Ewald** (current code):
```
v_3D(r) = Σ_R erfc(η|r+R|)/|r+R|  +  (4π/V) Σ_G exp(−G²/4η²)/G² · cos(G·r)  +  bg_const
```

**2D Ewald** (Parry 1975, Heyes 1981, de Leeuw-Perram 1979) — for electrons confined to a 2D plane interacting via 3D 1/r Coulomb:
```
v_2D(r) = Σ_R erfc(η|r+R|)/|r+R|  +  (π/A) Σ_{G≠0} (1/G) erfc(G/(2η)) · cos(G·r)  −  2√π·η/A   +  ...
```

Note the differences:
- Reciprocal-space integrand is `(1/G)·erfc(G/(2η))` instead of `(1/G²)·exp(−G²/(4η²))`
- Extra constant from G=0 limit
- No bg_const equivalent (but Madelung term still there for self-energy)

**References**:
- D.E. Parry, *Surface Sci.* 49, 433 (1975) — original 2D Ewald
- D.M. Heyes, *Phys. Rev. B* 23, 1755 (1981) — clean derivation
- For Madelung in 2D triangular Wigner crystal: `v_M ≈ −1.96/L` (vs. cubic 3D `−1.4187/L`); see Bonsall & Maradudin, *Phys. Rev. B* 15, 1959 (1977)

**Implementation**:
- Create new file `OmegaQMC/observables/ewald_2d.py` with:
  - `build_ewald_2d_tables(L, eta, n_real, n_recip)`: precompute R-vecs, G-vecs, weights.
  - `ewald_2d_pair_potential(diff, tables)`: same interface as 3D version.
  - `compute_madelung_2d(L)`: 2D Madelung constant for chosen Bravais lattice (square or triangular for crystal phase).
- Default η for 2D: `2.8 / √A` (CASINO-style; A is cell area).
- Cutoffs `n_real=3, n_recip=6` should still work.

**Effort**: ~3–5 days, including unit tests against analytical limits.

### 2.3 The plane-wave envelope (`psi/nn/env_periodic.py`)

3D `PlaneWaveEnvelope` enumerates k-vectors on a 3D cubic grid. Need 2D version.

**Changes**:
- New `enumerate_real_pw_basis_2d(n_orb, L)` returning a `RealPWBasis` with 2-component k-vectors.
- New `PlaneWaveEnvelope2D` (or generalize existing class with `dim` parameter).
- Default `n_pw` initialization: closed-shell sequence per spin is `1, 5, 9, 13, 21, 25, 29, 37, 45, 57, ...` (vs. 3D's `1, 7, 19, 27, 33, 57, ...`).
- The `_init_coeffs` Fermi-sea init needs to enumerate 2D shells correctly.
- The `cos(k·r)`, `sin(k·r)` computation: works as-is with d=2.

**Effort**: ~3 days. Mostly k-point enumeration logic.

### 2.4 GNN edge features (`psi/nn/gnn/edge_features_periodic.py`)

`PeriodicSinCosFeature` returns `(sin 2π s, cos 2π s)` — 6-d for 3D, would be 4-d for 2D. Need to verify shapes propagate correctly.

**Changes**:
- Verify `PeriodicSinCosFeature.__call__` output shape adapts to `s.shape[-1]`.
- The downstream GNN MLPs take `n_edge_features` as input — need to pass correct value.

**Effort**: ~1 day. Mostly inspection.

### 2.5 The HF reference (`afqmc_3deg.py`)

This is the AFQMC-based HF code that produces the HF reference energy printed at the start of each run. We need a 2D version.

**Two options**:
- (A) Adapt the AFQMC code to 2D (more work but consistent with 3D path).
- (B) Compute HF energy analytically for 2D HEG closed shells (since HF in HEG is just kinetic + exchange + Madelung):

  ```
  E_HF/N = T_HF/N + V_x/N + (1/2)·v_M
  T_HF/N = (1/N) Σ_{k_occ, σ} |k|²/2     (sum over occupied k's in 2D Fermi sphere)
  V_x/N  = -(1/2N) Σ_{i≠j occ same-spin} 4π/(A·|k_i-k_j|²)   (Ewald-style; 2D corrections needed)
  v_M    = computed from Ewald 2D tables
  ```

Option B is faster and avoids touching the AFQMC code. ~100 lines.

**Effort**: ~2 days.

### 2.6 System builder (`afqmc_3deg.py::build_3deg_system`)

Need 2D analog. The Wigner-Seitz radius definition changes:
- 3D: `(4π/3)·rs³ = 1/n` so `L = (4π/3)^{1/3} · rs · N^{1/3}`
- 2D: `π·rs² = 1/n` so `L = √π · rs · N^{1/2}`

**Changes**:
- New `build_2deg_system(rs, N_elec, polarization, dim=2)` returning analogous dict.
- Or: extend `build_3deg_system` with `dim` parameter.

**Effort**: ~1 day.

### 2.7 Plane-wave / Slater-det machinery in PsiFormer

The Slater determinant computation in `heg_psiformer.py` and the multi-det jitter logic should work in 2D unchanged once the envelope is 2D. The main concern is the `n_virt_pw` default — the closed-shell increments are different in 2D.

**Effort**: trivial after the envelope is done.

### 2.8 CLI / config

Add `system.dim: 2` to YAML config. Wire through to `build_*deg_system`.

**Effort**: ~1 hour.

---

## 3. New code: Wigner crystal ansatz (Phase 1.5, after fluid is working)

For rs ≳ 31 we need a localized-orbital ansatz. Cassella 2023 did this for 3D; we adapt to 2D triangular.

**Approach**:
- Replace `PlaneWaveEnvelope` with `GaussianLocalizedEnvelope2D`:
  - `n_det × n_orb × N_sites` Gaussian centers placed on triangular lattice (or other tested lattice).
  - Per-orbital width `σ` as variational parameter.
  - Spin-pattern: antiferromagnetic Néel-like ordering on the two-sublattice decomposition of the triangular lattice (this is what stabilizes the AF Wigner crystal in DMC).
- Backflow + Jastrow remain unchanged.
- For multi-det, jitter applied to Gaussian centers and widths (broken-symmetry init).

**Pretraining**: instead of pretraining toward Fermi sea, pretrain toward localized Gaussians at lattice sites.

**Effort**: ~2 weeks.

---

## 4. Validation plan: how we know it works

Sequence of tests, in order. Each is a publishable milestone if we hit benchmark accuracy.

### Test 0: Kinetic-only (turn off Coulomb)

Run 2D PsiFormer with Coulomb disabled. Should give exactly the HF kinetic energy of N free electrons in a 2D box.

**Expected**: `T/N = (1/N) Σ_k |k|²/2 = 1.0/rs²` (per-electron, in correct units; verify factor).

**Pass criterion**: matches analytic to 6 decimals.

### Test 1: HF reference (turn off Jastrow + backflow)

Run 2D PsiFormer with no Jastrow, no backflow, only the plane-wave envelope. Should give the HF energy.

**Expected** (at N=58, rs=2 unpolarized): roughly `−0.23 Ha/elec` (need to compute analytically).

**Pass criterion**: matches independent HF calculation to 6 decimals.

### Test 2: Single-determinant Slater-Jastrow at rs=2, N=58 unpolarized

Full PsiFormer, single det, no broken-symmetry ansatz. Train for ~5000 SR iters at 2048 walkers.

**Expected**: `E/N ≈ −0.25721(3) Ha/elec` (Attaccalite backflow value).

**Pass criterion**: within ~1 mHa/elec of Attaccalite (i.e., 96%+ of the correlation energy).

### Test 3: Polarized at rs=2

Same as Test 2 but ζ=1 (N=57 polarized).

**Expected**: `E/N ≈ −0.19359(2) Ha/elec`.

**Pass criterion**: within ~1 mHa/elec.

### Test 4: Density scan rs ∈ {1, 2, 5, 10, 20}

Run unpolarized PsiFormer at each rs.

**Expected**: matches Attaccalite Table I row by row within ~0.3–1 mHa/elec.

**This is the first publishable result** — "PsiFormer achieves Attaccalite-quality accuracy on 2D HEG fluid phase."

### Test 5: Polarization scan at rs=2

Run at ζ ∈ {0.0, 0.2, 0.4, 0.6, 0.8, 1.0}, fitting the spin-stiffness coefficient.

**Expected**: smooth interpolation between unpolarized and polarized energies, matching Attaccalite analytic form.

### Test 6: Wigner crystal at rs=30, N=58

With Gaussian-localized triangular ansatz.

**Expected**: `E/N ≈ −0.0319 Ha/elec` (Attaccalite Table I) — but also need to check we're below the fluid energy at this rs.

### Test 7: Wigner crystallization rs scan

Run both fluid and crystal ansatze at rs ∈ {25, 28, 31, 34, 38}. Find crossing point.

**Expected**: crystallization at `rs_c ≈ 31` (Drummond-Needs).

**This is the second publishable result** — "PsiFormer reproduces 2D HEG phase boundary."

---

## 5. Concrete file-by-file change list

```
OmegaQMC/
├── psi/nn/
│   ├── periodic.py           (modify: add make_square_lattice)
│   ├── env_periodic.py       (modify: PlaneWaveEnvelope → 2D mode)
│   ├── heg_wf.py             (modify: HEGPsiFormerConfig add dim param)
│   ├── heg_psiformer.py      (modify: pass dim through builder; small)
│   └── gnn/edge_features_periodic.py  (verify shapes)
├── observables/
│   └── ewald_2d.py           (NEW: 2D Ewald summation + Madelung)
├── afqmc_3deg.py             (extend: build_2deg_system, or new file)
├── pretrain_heg.py           (verify: pretrain target dim-agnostic)
├── vmcopt_nn_heg_sr.py       (no changes)
├── vmc_nn_heg.py             (no changes)
└── scripts/
    └── run_heg_psiformer.py  (modify: pass dim through; tiny)
```

### New file structure for Wigner crystal phase (Phase 2)

```
OmegaQMC/psi/nn/
├── env_localized_2d.py       (NEW: GaussianLocalizedEnvelope2D)
└── crystal_init.py           (NEW: triangular-lattice site placement)
```

---

## 6. Workplan with milestones

### Phase 0: 2D infrastructure (3–4 weeks)

| Week | Tasks | Deliverable |
|---|---|---|
| 1 | 2D lattice + 2D Ewald + analytical 2D HF | Tests 0, 1 pass |
| 2 | 2D plane-wave envelope + system builder | Single PsiFormer run completes at rs=2 |
| 3 | Validation runs at rs=1, 2, 5 unpolarized | Test 2 (rs=2) within 2 mHa of Attaccalite |
| 4 | Polarized runs + density scan rs=1,2,5,10 | Test 4 — first results table |

**Checkpoint**: at end of week 4, we have 5–10 PsiFormer 2D HEG energies that we can plot against Attaccalite.

### Phase 1: Validation paper-quality (4 weeks)

| Week | Tasks |
|---|---|
| 5 | Larger systems N=42, 74, 90; finite-size extrapolation |
| 6 | Twist-averaging implementation |
| 7 | Full density scan rs=1,2,5,10,20 unpol + polarized; full polarization scan at rs=2 |
| 8 | Tighten convergence; produce final benchmark table; draft 2D-fluid paper section |

**Checkpoint**: complete 2D HEG fluid-phase benchmark vs Attaccalite, suitable for paper.

### Phase 2: Wigner crystal (4–6 weeks)

| Week | Tasks |
|---|---|
| 9–10 | Implement GaussianLocalizedEnvelope2D, broken-symmetry pretraining for crystal |
| 11 | Validate at rs=30, 40 against Attaccalite/Drummond-Needs |
| 12 | rs scan around rs_c ≈ 31; locate crystallization |
| 13–14 | Polish results, finite-size analysis, plots |

**Checkpoint**: 2D HEG phase diagram (fluid → AF crystal → ferromagnetic crystal) reproduced, ready for cavity extension.

### Phase 3: Cavity coupling (Phase 1 of the actual project)

After all of the above. This is the cavity-modified phase diagram paper.

---

## 7. Open implementation questions to resolve early

1. **Should we have a single `ewald.py` with `dim` parameter, or `ewald_2d.py` as a separate module?**
   Recommendation: separate file. The 2D and 3D formulas differ qualitatively (erfc vs exp in reciprocal sum). A single function with branches is harder to maintain than two cleanly separated implementations.

2. **Should we have a single `PlaneWaveEnvelope` class with `dim` parameter, or `PlaneWaveEnvelope2D`?**
   Recommendation: single class with `dim`. The k-vector enumeration changes but the `cos(k·r)`, `sin(k·r)` calculation is identical.

3. **What's the minimum cell shape for cavity-2DEG (Phase 3)?**
   Cavity coupling is mode-specific. Single-mode cavity is in-plane polarized and translation-invariant in the 2D plane. Square cell with periodic BCs is consistent with this. No special geometry needed for the cavity itself.

4. **Spin-restricted vs unrestricted Slater determinant for Wigner crystal phase?**
   Drummond-Needs found AF crystal first stabilizes — so unrestricted (different spatial orbitals for up and down spins) is needed. Cassella 2023 also used unrestricted for the 3D crystal.

5. **For the cavity extension (later), where does the photon mode enter?**
   Following Tang 2025: extend the Slater orbital input with photon-number index, sample (electrons, photons) jointly via discrete-continuous Metropolis. We don't need to worry about this until Phase 2 is complete.

---

## 8. Risk analysis

### Things that could go wrong

| Risk | Probability | Mitigation |
|---|---|---|
| 2D Ewald implementation has subtle bugs | Medium | Extensive unit testing; cross-check against analytical limits and CASINO 2D HEG runs |
| PsiFormer struggles at low density (rs > 20) | Medium | Apply broken-symmetry hints earlier; consider alternative ansätze for the intermediate regime |
| Wigner crystal phase doesn't converge cleanly | Medium-High | Start at rs=40 (well-into crystal regime), back off toward boundary |
| KFAC integration becomes blocker for accuracy at low density | Medium | Use SR with longer training; tolerate ~1 mHa/elec gap to backflow-DMC reference |
| 2D infrastructure changes break 3D code | Low | Keep `dim` parameter; test 3D before/after each commit |

### Things that won't go wrong

- Architecture doesn't transfer (it does — attention is dim-agnostic)
- Pretraining doesn't work in 2D (it will — same supervised target structure)
- Multi-det machinery breaks (verified to work; just need correct closed-shell init)

---

## 9. Files to create immediately

1. `research/cavity_2dheg/notes/2d_heg_benchmarks.md` ✓ (done)
2. `research/cavity_2dheg/notes/implementation_plan_2dheg.md` ✓ (this file)
3. `OmegaQMC/observables/ewald_2d.py` — start of week 1
4. `OmegaQMC/psi/nn/env_periodic_2d.py` (or modify in place) — week 2
5. `tests/test_2d_ewald.py` — alongside ewald_2d.py
6. `tests/test_2d_hf_energy.py` — week 1
7. `inputs/2dheg_rs2_N58_unpol.yaml` — first benchmark run

---

## 10. Final pitch for the eventual paper

> *We extend the OmegaQMC neural-network variational Monte Carlo
> framework, previously validated against Cassella et al.'s benchmarks
> for the three-dimensional homogeneous electron gas, to two dimensions.
> Using a self-attention–based PsiFormer ansatz, we reproduce the
> backflow-DMC correlation energies of Attaccalite et al. (2002) to within
> [X] mHa/electron across the metallic density range r_s = 1–10, and
> recover the Wigner crystallization density r_c = [Y] ± [Z] in
> agreement with the established benchmark of Drummond and Needs (2009).
> This validates PsiFormer-VMC as a method for two-dimensional
> electron systems and establishes the foundation for our subsequent
> study of cavity-modified 2D HEG phase boundaries.*

That's a 4-page Letter. The "cavity" extension becomes a separate, more ambitious paper.
