# Literature Review — Cavity-Modified 2D HEG with NN-VMC

Curated review of every paper directly relevant to our project, with positioning notes.

---

## A. The direct competitor — read first, every time

### A1. Weber, Morales, Flick, Zhang, Rubio — PRL 135, 121001 (2025)
**arXiv:2412.19222 · published Dec 2024 / PRL 2025**
*"Light-matter correlation energy functional of the cavity-coupled two-dimensional electron gas via quantum Monte Carlo simulations"*

**What they do**: QED-AFQMC on a cavity-coupled 2D electron gas in a soft modulating external potential. Single cavity mode, dipole-gauge Pauli-Fierz Hamiltonian. Goal: produce a fitted correlation energy functional `E_c,el-ph[ρ; cavity params]` for QEDFT development.

**Critical limitations explicitly stated by authors**:
1. **No Coulomb interactions.** Quote: *"As a first step, in this work, we will not consider Coulomb interactions. Under this condition, our AFQMC simulations do not suffer from a phase problem and are exact."*
2. Focus on weak-to-intermediate coupling regime (where perturbation theory works).
3. Small system sizes: 4×4 unit cells, ~16 electrons.
4. Only computes correlation energy as a functional, NOT phase boundaries.
5. Does not compute Wigner crystal, ferromagnetic transition, or any phase diagram features.

**System parameters**: cavity frequency Ω, light-matter coupling `λ = Q_ph·ε/√(Ω·V_c)`, density `ρ = 1/a²`, modulating potential depth `v`.

**Methodological contribution**: introduces twist-averaging trick to remove QED-related finite-size effects (toroidal magnetic flux issue specific to cavity-coupled periodic systems). This is reusable for our work.

**Why we are not scooped**: They study a fundamentally different system (non-interacting electrons in modulating potential + cavity). We add (a) full Coulomb interactions, (b) the Wigner crystal regime, (c) the magnetic transition, (d) NN-VMC as method which extends to strong correlation regime where their AFQMC would face a phase problem.

**Co-author affiliations**: Flatiron CCQ, MPSD Hamburg, CCNY. Heavyweights — Flick + Rubio dominate cavity QED theory.

---

## B. Foundational cavity-QED ab initio methodology

### B1. Riso, Riemelmoser, Kjønstad, Koch — JCTC 20, 9669 (2024)
**arXiv:2410.18838 · JCTC 2024**
*"Phaseless auxiliary-field quantum Monte Carlo method for cavity-QED matter systems"*

Companion / precursor to Weber 2025. Develops the QED-AFQMC machinery in both Coulomb and dipole gauges. Tested on H₂, LiH, BeH₂ — molecular systems, not extended.

### B2. Tang, Andolina, Cuzzocrea, Mezera, Szabó, Schätzle, Noé, Erdman — arXiv 2025
**arXiv:2503.15644 · March 2025**
*"Deep quantum Monte Carlo approach for polaritonic chemistry"*

PauliNet2 extended with photonic degree of freedom (one-hot encoded photon number into GNN embedding). Discrete-continuous Metropolis-Hastings. KFAC optimizer.

**Systems**: 1 H₂ and 2 H₂ in cavity. Ground + excited states.

**Why important for us**: This is the recipe we'd follow for the cavity-photon part of our ansatz. They showed how to handle joint (electron, photon) sampling with a NN-VMC. Their approach transfers directly to periodic 2D HEG with minor modifications.

**Limitation for our purposes**: Molecular only, single H₂. Doesn't address extended/periodic systems.

### B3. arXiv:2505.16021 (May 2025)
*"Auxiliary Field Quantum Monte Carlo for Electron-Photon Correlation"*

Further AFQMC work for polaritonic chemistry. Validates against QED-CC and QED-FCI on small molecules. Methodological refinement of B1.

### B4. Rokaj, Welakuh, Ruggenthaler, Rubio — PRR 4, 013012 (2022)
**arXiv:2006.09236 · PRR 2022**
*"Free electron gas in cavity quantum electrodynamics"*

**Foundational analytical reference.** Solves the *non-interacting* electron gas + single cavity mode exactly in long-wavelength limit. Shows:
- Without Coulomb, ground state is Slater det × coherent photon state (light-matter product state).
- Ground state is Fermi liquid; cavity virtual photons renormalize quasiparticle mass.
- Importance of A² term: without it, no ground state in thermodynamic limit.
- Plasmon-polariton excitations in all four response sectors.

**Importance for us**: This is the analytical baseline our NN-VMC must reproduce in the limit of (a) zero Coulomb or (b) weak coupling. Useful as a sanity check.

---

## C. Free-space 2D HEG phase diagram benchmarks (no cavity)

### C1. Tanatar, Ceperley — PRB 39, 5005 (1989)
**Foundational DMC paper on 2D HEG.** Established existence of ferromagnetic transition and Wigner crystallization at low density.

### C2. Drummond, Needs — PRL 102, 126402 (2009) · arXiv:1002.2101
*"Phase diagram of the low-density two-dimensional homogeneous electron gas"*

**Modern benchmark paper.** Slater-Jastrow-Backflow VMC + DMC with CASINO. Findings:
- Paramagnetic fluid → antiferromagnetic triangular Wigner crystal at **rs = 31(1)** a.u.
- Antiferromagnetic crystal → ferromagnetic crystal at rs = 38(5).
- **Fully spin-polarized fluid is never stable** (contrary to earlier picture).
- Triangular lattice symmetry confirmed.

**Importance for us**: This IS the free-space reference our NN-VMC must reproduce at λ=0. We validate the 2D infrastructure here, then turn on cavity.

### C3. arXiv:2405.19397 (2024)
*"Ground state phases of the two-dimensional electron gas with a unified variational approach"*

Recent reanalysis with unified ansatz across phases. Use as additional cross-check.

### C4. PRB 110, 245145 (2024)
*"Quantum Monte Carlo study of the phase diagram of the two-dimensional uniform electron liquid"*

Most recent QMC study, refines the polarized vs unpolarized question.

---

## D. NN-VMC for HEG — methodological state of the art

### D1. Cassella, Sutterud, Azadi, Drummond, Pfau, Spencer, Foulkes — PRL 130, 036401 (2023)
**arXiv:2202.05183**
*"Discovering quantum phase transitions with fermionic neural networks"*

FermiNet applied to 3D HEG. **Crucial precedent**: shows NN-VMC can capture the Wigner crystallization transition (Fermi liquid → Wigner crystal) using broken-symmetry ansatze. Studies N=14, 38, 54 at rs=0.5, 1, 2, 5. KFAC optimizer.

**Direct relevance**: their approach to representing Wigner crystal phase via broken-symmetry FermiNet ansatze is what we'll adapt to 2D triangular lattice in cavity.

### D2. Wilson, Moroni, Holzmann, Gao, Wudarski, Vegge, Bhowmik — PRB 107, 235139 (2023)
**arXiv:2202.04622**
*"Wave function ansatz (but periodic) networks and the homogeneous electron gas"*

WAP-net: separate periodic FermiNet variant. Tests N=7, 14, 19 at wide rs range (1 to 100), exploring Wigner crystal regime in 3D.

### D3. Pescia, Nys, Kim, Lovato, Carleo — PRB 110, 035108 (2024)
**arXiv:2305.07240**
*"Message-passing neural quantum states for the homogeneous electron gas"*

MP-NQS for 3D HEG. Scales to N=128. Comparable accuracy to FermiNet at smaller parameter count.

### D4. arXiv:2604.15888 (2025)
*"Enhancing neural-network variational Monte Carlo through basis transformation"*

Recent methodological refinement. Shows basis transformation improves both FermiNet and MP-NQS on 3D HEG.

**Note**: D1-D4 are all 3D, not 2D. **No published NN-VMC on 2D HEG that we found.** This is a sub-gap within our project.

---

## E. Excited-state NN-VMC (relevant for future excited polariton states)

### E1. Pfau, Axelrod, Sutterud, von Glehn, Spencer — Science 385, 6711 (2024)
**arXiv:2308.16848**
*"Accurate computation of quantum excited states with neural networks"*

NES-VMC method. Atoms through benzene; conical intersection of ethylene. Validated against CC3, CASPT2, FCI. **Architecture**: FermiNet + PsiFormer.

**Future relevance**: cavity modifies the entire spectrum, not just ground state. Excited polariton states are observable in cavity-2DEG experiments. NES-VMC + cavity would be a natural Paper 4 in the arc.

---

## F. Experimental cavity-2DEG / cavity-quantum-Hall observations

### F1. Smolka et al. — Science 346, 332 (2014)
*"Cavity quantum electrodynamics with many-body states of a two-dimensional electron gas"*

First experimental evidence of cavity coupling to many-body states of high-mobility 2DEG. Polariton signatures of integer/fractional QH ground states.

### F2. Appugliese, Enkner, Paravicini-Bagliani, Beck, Reichl, Wegscheider, Scalari, Ciuti, Faist — Science 375, 1030 (2022)
*"Breakdown of topological protection by cavity vacuum fields in the integer quantum Hall effect"*

Modification of QH transport by cavity vacuum fields, no external drive. Direct evidence that cavity vacuum modifies many-body 2DEG physics.

### F3. Enkner, Graziotto, Boriçi, Sandhu, Reichl, Wegscheider, Faist — arXiv:2405.18362 (2024)
*"Enhanced fractional quantum Hall gaps in a 2DEG coupled to a hovering split-ring resonator"*

Hovering metallic resonator more than doubles FQH gaps. Explained as cavity-mediated long-range attractive electron-electron potential from virtual photon exchange.

**Critical experimental motivation for our work**: this paper directly demonstrates that cavity coupling modifies many-body energy gaps in real 2DEG samples, in equilibrium. Phase-boundary modifications must be next.

### F4. Tunable vacuum-field control of FQH phases — Nature 2025
Most recent confirmation of FQH phase modification by tunable cavity. Establishes a controlled experimental knob.

### F5. Various: Faist/Scalari group on terahertz Landau polaritons (Nature Photonics 2018, Nature Physics 2018)

Established ultrastrong coupling regime in cavity-2DEG. Cooperativity ~360. Vacuum Bloch-Siegert shifts. The experimental platform that justifies our weak-to-strong coupling theoretical study.

---

## G. Theoretical proposals for cavity-modified phases

### G1. Schlawin, Cavalleri, Jaksch — PRL 122, 133602 (2019)
**arXiv:1911.01459**
*"Cavity-mediated electron-photon superconductivity"*

Proposes cavity-mediated pairing in 2DEG via virtual photon exchange. Predicts critical temperatures in low Kelvin regime.

### G2. Andolina, De Pasquale, Polini, Pellegrino, Mauri — PRB 101, 205107 (2020) and later work
Polini group analytical studies on cavity-modified electron systems. Several proposals for cavity-induced order.

### G3. Roman, Andolina, Mauri, Polini — PRL 127, 167201 (2021)
**arXiv:2011.03753**
*"Photon condensation and enhanced magnetism in cavity QED"*

Predicts photon condensation transition + enhanced magnetism in cavity-coupled spin systems. **Directly relevant**: predicts cavity can enhance magnetic ordering. We can test the analog for 2D HEG ferromagnetic transition.

### G4. Schlawin, Kennes, Sentef — Appl. Phys. Rev. 9, 011312 (2022)
*"Cavity quantum materials"*

Definitive review of cavity-quantum-materials field. Cite as the survey for the broader area motivating our work.

---

## H. Auxiliary methodological references

### H1. NQS review — arXiv:2402.11014 (2024)
General review of NQS methods. Cite as overview of where NN-VMC fits.

### H2. Time-dependent NN-QMC — arXiv:2412.11830 (2024)
Explicit time-dependence in NQS. Relevant to future Floquet extensions.

### H3. arXiv:2503.19847 (2025)
Excited-state PES with transferable deep QMC. Methodology for transferable architectures.

---

# Positioning Strategy — what to emphasize in the paper

## Our specific contribution
First neural-network variational Monte Carlo treatment of the **Coulomb-interacting** 2D homogeneous electron gas in an optical cavity, computing **phase boundaries** including the Wigner-crystallization density and magnetic transition as a function of cavity coupling.

## Why this is novel beyond Weber et al. 2025
1. **Coulomb-interacting** HEG (Weber excludes Coulomb). Coulomb is the defining interaction of HEG; without it there's no Wigner crystallization, no ferromagnetic transition, no correlation hole physics.
2. **Phase diagram**, not correlation energy functional. Different scientific output.
3. **Strong-correlation regime** (rs > 30) accessible via NN-VMC's broken-symmetry ansatze. Weber's AFQMC (with phaseless approximation, were they to add Coulomb) would degrade in this regime.
4. **Wigner crystal phase** specifically. Weber cannot access this without symmetry-breaking modification.
5. **Magnetic transition** as cavity-coupling function. Untouched by any prior work.

## Why this is novel beyond Cassella et al. 2023, Wilson et al. 2023, Pescia et al. 2024
1. They did **3D**, we do **2D** — different phase diagram, lower crystallization rs, more experimentally relevant (TMD bilayers, semiconductor 2DEGs).
2. None of them include cavity coupling.

## Experimental hooks
- Faist/Scalari group's terahertz cavity 2DEG experiments (cooperativity ~360)
- Imamoğlu group's cavity-modified FQH experiments
- TMD heterobilayer experimental groups (Kim, Mak/Shan, etc.)
- Recent Enkner 2024 / Nature 2025 demonstration of cavity-modified FQH gaps

## Methodology pitch
- PsiFormer as the ansatz (attention-based, distinct from FermiNet/PauliNet used in competitors)
- Stochastic reconfiguration optimizer
- Single-mode dipole-gauge Pauli-Fierz coupling
- Discrete-continuous Metropolis sampling (Tang et al. 2025 recipe extended to periodic systems)
- Twist-averaged boundary conditions (Weber 2025 trick reused)
- Broken-symmetry ansatze for Wigner crystal (Cassella 2023 recipe extended to 2D + cavity)

## Three-paper arc proposed
- **Paper 1** (this paper): Establish methodology, compute Wigner-crystallization shift `rs_c(λ)`.
- **Paper 2**: Cavity-modified magnetic transition; full (rs, λ) phase plane.
- **Paper 3** (long-term): Multi-mode cavity, Floquet extension, or excited polariton states via NES-VMC + cavity.
