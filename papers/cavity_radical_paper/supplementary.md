# Supplementary Information

**Cavity-Induced Orbital Magnetization in Open-Shell Radicals from
Neural-Network Variational Monte Carlo**

This Supplementary Information accompanies the main text and provides
extended methodological details, per-system run parameters, and
auxiliary cross-checks.

---

## S1. Pauli-Fierz Hamiltonian: derivation and conventions

### S1.1 Long-wavelength dipole-approximated form

Starting from the minimal-coupling QED Hamiltonian and applying the
Power-Zienau-Woolley (PZW) gauge transformation in the long-wavelength
limit, the matter-field Hamiltonian for a single cavity mode of
frequency ω with two orthogonal polarization vectors (ε_x, ε_y) is:

```
H = H_el + ω(b†b + 1/2)
    + λ (ε_x · d̂) (b + b†)
    + i·s·λ (ε_y · d̂) (b† − b)
    + (λ²/2) [(ε_x · d̂)² + (ε_y · d̂)²] / 2
```

where:
- H_el is the bare electronic Hamiltonian (kinetic + Coulomb + nuclear)
- b, b† are the cavity-mode annihilation/creation operators
- λ = (ω/2ε_0 V)^(1/2) · (electron charge per length unit) is the
  effective coupling strength
- d̂ = Σ_i r̂_i is the total electric dipole operator in length gauge
- s = ±1 selects σ+ vs σ- circular polarization
- The last term is the dipole self-energy (DSE).

The vacuum zero-point energy ω/2 is dropped throughout (absorbed into
the energy reference).

### S1.2 Numerical implementation in OmegaQMC

The local energy estimator for a sample (r, n) is implemented as:

```python
e_loc(r, n) = T_el + V_nuc + V_ee + ω·n
            + λ(ε_x · d) · √(n+1)·Ψ(r,n+1)/Ψ(r,n)
            + λ(ε_x · d) · √n·Ψ(r,n-1)/Ψ(r,n)
            + i·s·λ(ε_y · d) · √(n+1)·Ψ(r,n+1)/Ψ(r,n)
            - i·s·λ(ε_y · d) · √n·Ψ(r,n-1)/Ψ(r,n)
            + (λ²/4)·[(ε_x · d)² + (ε_y · d)²]
```

For complex Ψ, e_loc(r,n) is complex; the physical energy is
Re⟨e_loc⟩_|Ψ|².

---

## S2. Neural-Network Wavefunction Architecture

### S2.1 FermiNet+Jastrow+Backflow base

Following the architecture in [@Pfau2020FermiNet;
@Tang2025DeepQMCPolaritonic; @FermiNetOpenSource2024], our base
electronic ansatz is

```
log|Ψ_el(r)| = Σ_d log|det[M_d(r)]| + J(r)
```

with M_d the orbital matrix of the d-th determinant, computed from
backflow-dressed orbitals:

```
M_d[e, o] = Σ_k c_o,k · σ_o,k(R_eff_e(r)) · η_o,k(r_e)
```

where σ_o,k is a basis function (typically Gaussian on a nucleus) and
η_o,k is a learnable scalar from the GNN embedding of electron e.

J(r) is a 3-body Jastrow factor parameterized by an MLP.

### S2.2 Photon-Fock embedding (Tang-native n-injection)

The Fock-state index n ∈ {0, 1, ..., n_ph_max} is embedded as a
per-electron one-hot vector concatenated to the GNN input features
following [@Tang2025DeepQMCPolaritonic]. This allows the wavefunction to
have qualitatively different orbital structure at different photon
occupations.

### S2.3 Complex-Ψ extension

For chiral cavities (s = ±1), the Hamiltonian is non-Hermitian under
real wavefunctions due to the i·s·λ term. We extend the orbital matrix
to complex values:

```
M[e, o] = M_re[e, o] + i M_im[e, o]
```

with M_im computed by a separate "imaginary head" — an additional
linear projection from the GNN embedding, initialized with zero
kernel. This guarantees that at initialization the wavefunction is
real (matches the linear-polarization case), and the imaginary part
grows organically during SR training as required by the cavity
Hamiltonian.

The full complex log-Ψ is

```
log Ψ_complex(r) = log|det[M_re + i M_im]| + i arg(det[M_re + i M_im])
                  + J(r)
```

For complex orbital matrices the determinant is computed via the LU
decomposition.

---

## S3. Stochastic Reconfiguration Optimization

### S3.1 SR update equation

The SR step seeks to minimize the energy expectation ⟨Ψ_θ| H |Ψ_θ⟩ on
the variational manifold:

```
S(θ) · δθ = -lr · f(θ)
```

where
```
S_ij = ⟨O_i O_j⟩ - ⟨O_i⟩⟨O_j⟩   (Fisher / quantum-geometric tensor)
f_i  = ⟨(E_loc - ⟨E_loc⟩) · O_i⟩   (force vector)
O_i  = ∂(log Ψ)/∂θ_i              (log-amplitude derivative)
```

The linear system S · δθ = -lr · f is solved by conjugate gradient
with diagonal damping S → S + d·I.

### S3.2 Complex-Ψ SR

For complex wavefunctions we split O_i into magnitude and phase parts:
O_i = O_re + i O_im. The SR equations become

```
S = (J_re^T J_re + J_im^T J_im) / N
f = (dE_re · dJ_re + dE_im · dJ_im) / N
```

where dE_re = Re(e_loc) - ⟨Re(e_loc)⟩, similarly for imaginary parts.
This handles complex-valued e_loc without phase-tracking issues.

### S3.3 Sign-constraint penalty (Path B — used for diagnostic only)

To address the basin instability in degenerate-SOMO open-shell systems
(Section 4.2 of main text), we tested a chirality-sign penalty

```
H_penalty = -α · s · L_z
```

added to the SR objective. This adds `-α·s·L_z_local` to e_loc inside
the SR loop. We found a bias-variance trade-off:

| α     | <L_z>(H3 σ+, λ=0.5)  | E shift (Ha) |
|-------|----------------------|---------------|
| 0     | -0.357 (wrong basin) | 0             |
| 0.05  | -0.184               | small         |
| 0.2   | +0.005 (transition)  | small         |
| 0.5   | +0.076 (right basin) | +0.16         |

Larger α correctly enforces sign but shifts E away from the true GS.
Annealed-α schedules are a promising future-work direction.

---

## S4. Per-System Run Parameters

### S4.1 CH3·

```
Geometry:   planar D3h, r_CH = 2.039 Bohr (1.079 Å)
n_elec:     9 (n_up = 5, n_down = 4, Sz = +1/2)
basis:      ferminet_jastrow_complex.yaml (Tang-native, n_ph_max = 4)
seed:       42
```

Vacuum gate: 2000 iters, lr = 0.02, max_param_change = 0.2, 256 walkers
→ E = -39.7561 ± 0.013 Ha (cf. UHF/cc-pVDZ -39.564 Ha; FCI/CBS -39.84).

Chiral runs: 500 iters at lr = 0.05, then 2000 iters at lr = 0.02
when needed, 256 walkers. Per-run wall-clock 5-10 min on GH200.

### S4.2 H3 (equilateral triangle)

```
Geometry:   D3h, side r_HH = 1.65 Bohr
n_elec:     3 (n_up = 2, n_down = 1)
```

Vacuum gate: 500 iters, lr = 0.02, 128 walkers
→ E = -1.5537 ± 0.0015 Ha.

### S4.3 H6 (regular hexagon)

```
Geometry:   D6h, side r_HH = 1.80 Bohr
n_elec:     6 (closed shell)
```

Vacuum gate E = -3.303 ± 0.024 Ha.

### S4.4 H2 (linear, R = 2.0 Å)

From prior chiral pilot (Phase 2n), see [@Tang2025DeepQMCPolaritonic]
benchmark.

### S4.5 NO•

```
Geometry:   linear C∞v, r_NO = 2.175 Bohr (1.151 Å)
n_elec:     15 (n_up = 8, n_down = 7)
```

Vacuum gate E = -117.04 ± 0.28 Ha (1000 iters, undertrained;
ROHF/cc-pVDZ reference -129.2 Ha). ⟨L_z⟩ values are nonetheless
informative as orbital-symmetry observables.

---

## S5. Effective Magnetic Field Computation

### S5.1 χ_orb_zz from PySCF UHF

We computed the vacuum orbital paramagnetic susceptibility χ_orb_zz for
CH3· by uncoupled-Hartree-Fock sum-over-states using the AO matrix
elements of the L_z operator (from `mol.intor("int1e_cg_irxp")[2]`)
transformed to the UHF MO basis:

```
χ_orb_zz = Σ_{i ∈ occ, a ∈ virt}  |<a|L_z|i>|² / (ε_a - ε_i)
```

For CH3· (UHF/cc-pVDZ): χ_orb_zz = 2.5045 a.u.

### S5.2 B_eff conversion

In atomic units, B_eff = ⟨L_z⟩ / χ_orb_zz. Converted to Tesla using
1 a.u. of B = 2.35052 × 10⁵ T.

### S5.3 Δε_split inference

In the dominant-MO picture, B_eff produces a Zeeman splitting of the
1e' degenerate orbital pair:

```
Δε_split = 2 |<e'_+|L_z|e'_+>| · μ_B · B_eff
         = 2 × 0.78ℏ · (μ_B/ℏ) · B_eff
         ≈ 1.56 · μ_B · B_eff (atomic units)
```

The factor 0.78 = ⟨e'_+|L_z|e'_+⟩ is obtained from the PySCF MO
reference (see scripts/ch3_mo_reference.npz).

---

## S6. NMR Shift Prediction Methodology

The cavity-induced orbital current at the carbon center is treated
as a magnetic dipole in the point-dipole approximation. The induced
field at an in-plane H nucleus (r perpendicular to μ) is:

```
B_z(H) = -(μ_0/4π) · μ_orb / r_CH³
```

with μ_orb = -μ_B ⟨L_z⟩ (orbital g = 1). For the typical 500 MHz
1H NMR magnet B_ext = 11.74 T, the chemical shift in ppm is:

```
δ_NMR (ppm) = -B_z(H) / B_ext × 10⁶
```

The point-dipole approximation overestimates the field at H by a
factor of 2-5× because the actual ring current is spatially
distributed across the 1e' shell rather than concentrated at the
carbon nucleus. A more accurate prediction would require numerical
integration of the current density (Biot-Savart) over the cavity-
dressed wavefunction — left for future work.

---

## S7. Density Chirality Algorithm

For each set of walker positions {r_e} obtained from VMC evaluation
of the trained Ψ at fixed parameters, we histogram the electron
density in cylindrical coordinates (R, θ, z) around the C₃ axis.
The Fourier coefficient at azimuthal index m is:

```
a_m(R, z) = (1/2π) ∫_{-π}^{π} ρ(R, θ, z) · exp(-i m θ) dθ
```

In vacuum CH3· (D3h), C₃ rotation forces a_m = 0 unless m ∈ {0, ±3,
±6, ...}, and σ_v reflection forces a_+m = a_-m (i.e. a_m is real).

The σ+ cavity preserves C₃ but breaks σ_v, so a_+3 acquires a
non-zero imaginary part. We report Im(a_3) heatmaps as the direct
chirality signal; the σ+/σ- parity check shows sign-flipping at
10.7σ above zero.

Walker positions are obtained by re-running the eval phase of a
trained model with `dump_walker_positions: true` in the YAML.
Implementation: `scripts/dump_walker_positions.py` and
`scripts/analyze_density_chirality.py`.

---

## S8. Reproducibility

All scripts, configs, and trained model checkpoints are available at:

```
https://github.com/myung-group/OmegaQMC
branch: cavity-qed-vmc-mol
commit: [TBD when paper submitted]
```

Per-system input configs in `inputs/qed_*/`. Analysis scripts in
`scripts/`. The full paper figure-generation pipeline is in
`papers/cavity_radical_paper/build_figures.py`.

Compute was performed on the KISTI mango cluster (NVIDIA GH200 GPUs)
using the OmegaQMC framework. Typical per-system compute was
5-30 min for small systems (H3, H6, CH3) and 5-30 min × n_runs
for chain experiments. Total compute for all results in this paper:
approximately 12 GPU-hours on a single GH200.
