# Phase 2n — Complex-Psi NN-VMC for Chiral Cavity QED

**Goal**: extend OmegaQMC's FermiNet + Jastrow + backflow + Tang-native architecture
to support complex-valued wavefunctions, so that chirality observables
(`<L_z>`, ring current `j(r)`, inverse-Faraday `B_eff`) become directly
computable for molecules in circularly polarized cavities.

**Why now**: validated H2 dissociation paper banked. Weber 2026 magnetic-cavity
paper leaves chirality + ring-current observables explicitly unmeasured.
First production target: **CH3 methyl radical** (open-shell, doubly-degenerate
pi orbital -> clean <L_z> = +-1 hbar demo).

## Backward compatibility principle

Gated behind `NNAnsatzConfig.complex_psi: bool = False`. All existing
real-Psi tests (Phase 2a–2m) must stay green throughout.

## Phase breakdown

### 3a. Complex Slater determinant infrastructure (3–5 days)

```
Files touched:
  OmegaQMC/psi/nn/config.py        - add complex_psi: bool = False
  OmegaQMC/psi/nn/types.py         - generalize Psi.sign dtype (already permissive)
  OmegaQMC/psi/nn/env.py / envelope - optional complex orbital output (2x channels)
  OmegaQMC/psi/nn/wf.py            - branch on config.complex_psi:
                                      assemble complex orbital matrices
                                      eval_log_slater works on complex (jnp.linalg.slogdet)
                                      log_psi becomes complex; sign is complex unit
  OmegaQMC/psi/nn/build.py         - thread complex_psi flag into builders

Tests:
  tests/test_complex_psi_h2.py
    - bare H2, complex_psi=True
    - random complex init optimization -> energy matches KW within budget
    - Im(Psi) ratio Im/Re -> 0 by variational principle (TR-symmetric H)

Gate: bare H2 complex-Psi reproduces real-Psi KW reference (<= 1 mHa). 
      Im part decays to zero numerically.
```

### 3b. Complex local energy estimator (3–4 days)

```
Files touched:
  OmegaQMC/psi/nn/qed_physics.py   - pauli_fierz_local_energy_signed handles
                                     complex sign (already does, just verify dtypes);
                                     return complex E_loc
  OmegaQMC/qed_vmc_nn.py           - real(E_loc) in expectation aggregation
  OmegaQMC/psi/nn/qed_adapter.py   - new make_qed_nn_log_psi_complex factory:
                                     wraps complex-wf, returns (log_psi_complex)

Tests:
  - Bare H2, complex_psi=True, linear-pol cavity at lambda=0.1, omega=0.467 Ha
  - Expected: same E as real-Psi run; Im(<E>) = 0; Im(Psi) -> 0

Gate: Linear-cavity H2 result reproduces our existing real-Psi number 
      within stat error (~mHa).
```

### 3c. Chiral cavity Hamiltonian + observables (2–3 days)

```
Files touched:
  OmegaQMC/psi/nn/qed_physics.py
    - pauli_fierz_local_energy_chiral(handedness=+1/-1, ...)
      bilinear: sqrt(omega/2)*lam*[(d.eps_x)(b+b^dag) + i*handedness*(d.eps_y)(b^dag-b)]
      DSE: (lam^2/2) * [(d.eps_x)^2 + (d.eps_y)^2]
    - observables module:
      l_z_per_walker(elec_crds, log_psi) -> sum_i (x_i * d/dy_i - y_i * d/dx_i) log psi
      ring_current_density(...) -> j(r) at grid points

Tests:
  - Chiral cavity H2: <L_z> != 0 at non-zero handedness
  - Sign flip with handedness: <L_z>(sigma+) = -<L_z>(sigma-)

Gate: <L_z> sign and order-of-magnitude correct on H2 chiral test.
```

### 3d. Complex SR optimizer (3–5 days)

```
Files touched:
  OmegaQMC/qed_vmcopt_nn_sr.py
    - JAX autograd already supports complex differentiation
    - Jacobian J = d(log psi)/d(theta) is complex
    - Fisher S = Re(J^dag J) -- symmetric positive
    - Force f = Re((E - <E>) J^dag) -- real
    - CG solve unchanged (still positive symmetric system)

Tests:
  - Train linear-cavity H2 with complex_psi=True
  - Optimizer converges; phi -> 0; energy matches real-Psi result

Gate: linear cavity training reproduces real-Psi number within stat noise.
```

### 3e. Production runs (~1 week)

```
Pipeline:
  1. H2 chiral cavity: pilot
     - lambda in {0.1, 0.3}, omega = ~10 eV, R = 1.4 Bohr
     - Measure: <L_z>, j(r) heat map, B_eff
     - Compare sigma+ vs sigma-

  2. CH3 methyl radical: primary target
     - Open-shell (need to fix n_down=0 ansatz bug first OR use Sz=0 doublet)
     - lambda scan {0, 0.05, 0.1, 0.2, 0.3}
     - omega resonant with degenerate-p splitting (~10 eV)
     - Headline: cavity locks <L_z> = +-1 hbar at strong coupling

  3. Generate figures + write up
```

## Total scope

- Phase 3a: ~4 days
- Phase 3b: ~4 days
- Phase 3c: ~3 days
- Phase 3d: ~4 days
- Phase 3e: ~7 days
- **Total: ~22 days (3-4 weeks)**

## Validation gates (must pass before next phase)

1. After 3a: bare H2 complex-Psi -> real GS energy, Im->0
2. After 3b: linear cavity H2 -> matches our prior real-Psi number
3. After 3c: <L_z> on chiral H2 -> nonzero with correct sign convention
4. After 3d: linear cavity SR -> reproduces real-Psi run
5. After 3e: production CH3 chiral results, paper-ready figures

## Risks and contingencies

| Risk | Severity | Mitigation |
|---|---|---|
| Complex orbital path makes optimization unstable | high | Start from real-Psi-converged state, perturb to complex |
| Phase ambiguity in complex Slater (global phase of Psi) | medium | Pin phase by fixing one orbital coefficient real-positive |
| SR Fisher matrix degenerate from global phase | medium | Project out global phase mode from Fisher |
| open-shell CH3 hits n_down=0 bug | known | Fix in Phase 3e or sidestep by Sz=0 doublet |
| Multi-electron benzene too slow for first paper | medium | Stay with CH3 if benzene per-iter too slow |

## Open design decisions

- Complex orbital generation: 2x output channels (Re/Im pair) on the
  envelope, then complex combine. Simpler than complex MLP weights.
- Backflow: extend to complex by adding 2x channels in fs (real Re/Im pair),
  combine before applying.
- Cusp factors: stay REAL (they're real correction functions of real distances).
  Just multiply log|Psi_complex| by exp(cusp), no phase contribution.
- conf_coeff: linear combination with REAL coefficients. Output is complex
  iff input is complex. No change needed.
