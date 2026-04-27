# 2D HEG Benchmark Reference Tables

The canonical numbers our PsiFormer 2D HEG implementation must reproduce. Use these as the validation oracle.

---

## A. Attaccalite, Moroni, Gori-Giorgi, Bachelet — PRL 88, 256601 (2002)

**The standard reference** for 2D HEG ground-state energies. FN-DMC with backflow correlations. Also has plane-wave (no backflow) numbers for cross-validation. System: N=58 (unpolarized) or N=57 (polarized). Population M=200 walkers, time step τ=0.002–2.0 (extrapolated).

### Total energy per electron `E/N` (Hartree/electron)

#### PLANE WAVE Slater-Jastrow (no backflow)

| rs | ζ=0 (unpolarized) | ζ=1 (polarized) |
|----|---|---|
| 1  | −0.2013(1)     | +0.13147(2)    |
| 2  | −0.255802(4)   | −0.193349(1)   |
| 5  | −0.149134(9)   | −0.143520(5)   |
| 10 | −0.0852706(4)  | −0.084555(2)   |
| 20 | −0.046241(1)   | −0.0462385(6)  |
| 30 | −0.031923(1)   | −0.0319298(6)  |

#### BACKFLOW Slater-Jastrow (the gold standard)

| rs | ζ=0 (unpolarized) | ζ=1 (polarized) |
|----|---|---|
| 1  | −0.20372(4)    | +0.13109(4)    |
| 2  | −0.25721(3)    | −0.19359(2)    |
| 5  | −0.149518(9)   | −0.143610(7)   |
| 10 | −0.085427(6)   | −0.084584(2)   |
| 20 | −0.046385(6)   | −0.0462488(8)  |
| 30 | −0.031941(2)   | −0.031938(1)   |

### Backflow gain (correlation captured beyond plane-wave SJ)

| rs | ΔE_BF (ζ=0, mHa/elec) | ΔE_BF (ζ=1, mHa/elec) |
|----|---|---|
| 1  | −2.4   | −0.4  |
| 2  | −1.4   | −0.2  |
| 5  | −0.4   | −0.1  |
| 10 | −0.16  | −0.03 |
| 20 | −0.14  | −0.01 |
| 30 | −0.018 | −0.008 |

So at rs = 1–2, backflow is worth ~1–2 mHa/elec — non-trivial, our PsiFormer should match or beat this. At rs ≥ 10, backflow is sub-mHa/elec; PsiFormer should be within statistical noise.

### Polarization energy difference Δ = E(ζ=1) − E(ζ=0), mHa/elec

| rs | Plane wave | Backflow |
|----|---|---|
| 1  | +332     | +335    |
| 2  | +62.5    | +63.6   |
| 5  | +5.6     | +5.9    |
| 10 | +0.72    | +0.84   |
| 20 | +0.0025  | +0.14   |
| 30 | −0.0068  | +0.003  |

The polarization transition (sign change of Δ) is the key diagnostic. Backflow data shows it stays positive through rs=30, consistent with Drummond-Needs 2009's finding that the fully spin-polarized fluid is **never stable** — you go directly to the antiferromagnetic Wigner crystal.

---

## B. Drummond, Needs — PRL 102, 126402 (2009) and PRB 79, 085414 (2009/2010)

Phase boundaries from VMC + DMC:

- **Paramagnetic fluid → antiferromagnetic Wigner crystal**: `rs_c = 31(1)` a.u.
- **AF crystal → ferromagnetic crystal**: `rs_c = 38(5)` a.u.
- **Fully spin-polarized fluid: never stable** (pre-empted by crystallization)

System sizes: N up to 58 (fluid), with finite-size extrapolation `E(N) = E(∞) − c/N^{5/4}`. Used CASINO code with Slater-Jastrow-Backflow.

---

## C. Holzmann, Bernu, Olevano, Martin, Ceperley — PRB 79, 041308(R) (2009)

Renormalization factor Z and effective mass m* for 2D HEG. Important for quasiparticle properties.

For 1 ≤ rs ≤ 10:
- Z (renormalization factor) decreases from ~0.85 at rs=1 to ~0.4 at rs=10
- m*/m increases from ~1.0 at rs=1 to ~1.4 at rs=10

Useful for second-paper-level comparisons but not needed for ground-state energy benchmark.

---

## D. Attaccalite parameterization (analytic correlation-energy functional)

Their Eq. (3) and (4) provide a closed-form interpolating functional for ε_c(rs, ζ) that we can use as a continuous reference for any rs ∈ [1, 30] and ζ ∈ [0, 1]. Useful for plotting and comparing PsiFormer results across the full density range without re-running DMC.

Form:
```
ε_c(rs, ζ) = (e^{-βrs} − 1) ε_x^{(6)}(rs, ζ) + α_0(rs) + α_1(rs)ζ² + α_2(rs)ζ⁴
```
with α_i(rs) given by Eq. (4) and Table II of the paper. Implementation: ~30 lines of Python.

---

## E. Closed-shell electron numbers (square cell at Γ point)

Per-spin closed-shell counts (filling shells of |k|² in units of (2π/L)²):

| Shell |k|² | k-vectors per shell | Cumulative per-spin |
|---|---|---|
| 0 | 1   | 1 |
| 1 | 4   | 5 |
| 2 | 4   | 9 |
| 4 | 4   | 13 |
| 5 | 8   | 21 |
| 8 | 4   | 25 |
| 9 | 4   | 29 |
| 10 | 8  | 37 |
| 13 | 8  | 45 |
| 16 | 4  | 49 |
| 17 | 8  | 57 |
| 18 | 4  | 61 |
| 20 | 8  | 69 |

Total `N = 2 × (per-spin count)` for unpolarized closed shell:

`N ∈ {2, 10, 18, 26, 42, 50, 58, 74, 90, 98, 114, 122, 138, ...}`

**Standard benchmark sizes**:
- N = 26 (small, fast)
- N = 42 (medium)
- **N = 58 (Attaccalite reference size)** — *use this for direct comparison*
- N = 74, 90 (larger; for finite-size extrapolation)

---

## F. Wigner crystal triangular lattice (rs ≳ 31)

For Wigner crystal benchmarks (Drummond-Needs 2009, Tanatar-Ceperley 1989):

- Lattice: triangular (hexagonal close-packed in 2D)
- Lattice constant: a = √(2/(√3·n)) where n = 1/(π·rs²)
  - For rs=30: a ≈ √(2/(√3·π·900)) · (some factor)... let me just use the fact that nearest-neighbor distance in triangular lattice with density n=1/(π rs²) is `d_NN ≈ rs · 2/√(3/π) · (something)` — easier: use Drummond convention.
- Per-site occupation: 1 electron each, antiferromagnetic spin pattern (alternating up/down on triangular lattice, which is frustrated in strict triangular but the Wigner crystal is on a hexagonal Bravais lattice that can sustain Néel order via splitting into two sublattices).

Standard procedure: Gaussian-localized orbital `φ(r − R_i)` for each site, with width `σ` optimized variationally. Cassella 2023 used this approach for 3D Wigner crystal.

---

## G. Quick sanity checks for our PsiFormer implementation

When our 2D PsiFormer first runs on N=58 unpolarized rs=2:

| Check | Expected value | Tolerance |
|---|---|---|
| HF energy E_HF(N=58) | ~−0.232 Ha/elec (from QMC literature finite-size) | ±10 mHa |
| HF kinetic energy T_HF/N | ~0.250 Ha/elec | ±5 mHa |
| HF Madelung correction | ~−0.0X Ha/elec (2D Madelung ≈ −1.96/L) | ±2 mHa |
| Free-space PsiFormer E/N at rs=2 (after training) | should approach Attaccalite −0.25721(3) | within ~1 mHa = 96%+ correlation |
| Free-space PsiFormer E/N at rs=10 | should approach Attaccalite −0.085427(6) | within ~0.3 mHa |
| Polarization gap E(ζ=1) − E(ζ=0) at rs=2 | should approach +63.6 mHa/elec | ±2 mHa |

Once these pass, we're ready to add cavity coupling.

---

## H. Files in `papers/` for 2D HEG benchmarks

- `Attaccalite2002_PRL_2DEG_correlation_energy.pdf` — **the canonical reference**
- `Attaccalite2002_PRL_2DEG_v2_long.pdf` — long version with full data tables
- `Drummond2010_2DEG_Fermi_fluid_full.pdf` — 2010 long paper; full Fermi-fluid analysis
- `Drummond2009_2DHEG_phasediagram.pdf` — 2009 PRL on phase diagram
- `Drummond2001_pair_correlation_2D.pdf` — pair correlation function reference
- `Wang2024_2DHEG_dual_gate_screening.pdf` — recent (2024) gate-screened 2D HEG, modern XC functional
- `Holzmann2009_effective_mass_2DEG.pdf` — quasiparticle effective mass and Z
- `Holzmann2016_finite_size_theory_QMC.pdf` — finite-size theory (RPA-based)
- `GroundStatePhases_2DEG_2024_unified.pdf` — 2024 unified-ansatz reanalysis of phase diagram
