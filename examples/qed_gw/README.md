# Reproducibility Guide for QED-GW and QED-BSE Calculations

This directory contains the example scripts, input settings, and result data used to generate the results reported in the manuscript

> *GW and Bethe--Salpeter Theory for Molecular Polaritons, Quasiparticles, and Excitons*

The scripts cover the principal calculations of the paper:

- QED-GW ionization-potential (IP) and electron-affinity (EA) benchmarks against the cavity $\Delta$-method ladder ($\Delta$QED-HF, $\Delta$QED-CCSD, $\Delta$QED-DMRG, $\Delta$QED-FCI)
- Cavity-induced IP/EA shifts and their relative errors
- Coupling-strength and detuning scans of the cavity-induced IP shift
- Basis-set sensitivity of the cavity-induced EA shift
- QED-RPA and polaritonic QED-BSE@evGW absorption spectra (Rabi splitting)
- QED quasiparticle spectral functions and the polariton photoemission replica
- Neutral-excitation observables and exciton binding energies from the full (non-TDA) QED-BSE@QED-evGW
- Channel decomposition of the cavity-induced exciton-binding shift (quasiparticle channel versus DSE--photon kernel residual) and the polarization criterion
- Basis-set robustness of that decomposition (cc-pVDZ versus aug-cc-pVDZ)
- Comparison against published QED coupled-cluster cavity-modulated IPs/EAs of the sodium halides
- Validation of the polaritonic QED-BSE Rabi splitting against polaritonic QED-FCI

Each script is the primary reproducibility entry point for the corresponding result. It specifies the molecule, geometry, basis set, cavity parameters, electronic-structure method, and every other numerical setting.

## Common settings

Unless a script states otherwise, the calculations use the settings of Sec. III A of the manuscript:

- Basis set cc-pVDZ (STO-3G for the $\Delta$QED-FCI anchors; aug-cc-pVDZ for the basis-set checks).
- Geometries recentered so that the nuclear centre of mass sits at the coordinate origin, with the principal symmetry axis along $z$. This matters because the dipole operator is origin-dependent for the charged species.
- A single cavity mode with $\omega_\mathrm{cav} = 0.415668$ $E_h$ ($11.31$ eV), $z$-polarized coupling, and $\lambda \in \{0, 0.05\}$ a.u. (the scans extend $\lambda$ to $0.10$ and vary $\omega_\mathrm{cav}$).
- Charged species treated as doublets on a QED-UHF reference; the $\Delta$-method energies are ground states of the same Pauli--Fierz Hamiltonian.
- Neutral excitations from the full (non-TDA) QED-BSE of Eq. (22) on evGW quasiparticle energies; the polaritonic spectra use the explicit-photon QED-BSE of Eq. (25).

## Script-to-manuscript map

| Script | Purpose | Manuscript result |
|---|---|---|
| `run_qed_ipea_benchmark.py` | IP/EA benchmark for H$_2$O, HF, NH$_3$, CH$_4$ (cc-pVDZ) with STO-3G $\Delta$QED-FCI anchors for H$_2$O and HF | Tables I--III (hydride rows); Secs. III B--C |
| `run_qed_ipea_aromatics.py` | IP/EA benchmark for benzene and naphthalene | Tables I--III (aromatic rows); Sec. III C |
| `run_qed_ip_scans.py` | Coupling-strength and detuning scans of the cavity-induced IP shift of H$_2$O | Fig. 3; Sec. III D |
| `run_qed_ea_basis_check.py` | cc-pVDZ versus aug-cc-pVDZ cavity-induced EA shift of H$_2$O | Sec. III D discussion |
| `run_qed_absorption_h2o.py` | QED-RPA and polaritonic QED-BSE@evGW absorption spectra of H$_2$O | Fig. 4; Sec. III E |
| `run_qed_absorption_nh3.py` | Absorption spectra of NH$_3$ | Fig. 5; Sec. III E |
| `run_qed_absorption_naph_singlet.py` | Absorption spectra of naphthalene (spin-adapted singlet implementation) | Fig. 6; Sec. III E |
| `run_qed_spectral_h2o.py` | HOMO spectral function of H$_2$O and the polariton replica | Fig. 7; Sec. III F |
| `run_qed_exciton_binding.py` | Neutral-excitation observables and exciton binding energies for H$_2$O, HF, NH$_3$, CH$_4$, with the QP/DSE/photon channel decomposition for H$_2$O | Table IV; Sec. III G |
| `run_qed_exciton_binding_compare.py` | H$_2$O versus NH$_3$ channel decomposition, each at its own $S_1$-resonant cavity | Fig. 8; Sec. III G |
| `run_qed_binding_ph3.py` | Two-channel decomposition and polarization criterion for all five hydrides including PH$_3$ | Table V; Sec. III H |
| `run_qed_binding_basis.py` | cc-pVDZ versus aug-cc-pVDZ observables and channel decomposition | Table VI, Fig. 9; Sec. III I |
| `run_qed_nax_deprince.py` | NaF and NaCl comparison with published QED-CC cavity-modulated IPs/EAs | Table VII; Appendix B |
| `run_qed_bse_polaritonic_fci_benchmark.py` | Polaritonic QED-BSE Rabi splitting versus polaritonic QED-FCI (H$_2$O/STO-3G) | Validation of Eq. (25), Sec. II D 2 |

Fig. 1 is a schematic and Fig. 2 is derived from the Table III data (see "Regenerating the manuscript figures" below).

Run every script from this directory with the Python interpreter in which OmegaQMC is installed:

```bash
cd examples/qed_gw
```

Each script prints its results and writes a JSON file next to itself (for example `qed_ipea_results.json`, `qed_ip_scans_results.json`, `qed_binding_basis_results.json`). The JSON files shipped in this directory are the data behind the corresponding manuscript tables and figures; scripts whose JSON output is absent must be rerun to regenerate it.

## Reproducing the QED-GW IP/EA benchmarks (Tables I--III, Fig. 2)

The IP/EA benchmark evaluates vertical charged excitations with the self-energy methods (Koopmans, G$_0$W$_0$, evGW) and the $\Delta$-method rungs ($\Delta$QED-HF, $\Delta$QED-CCSD, $\Delta$QED-FCI) on the same Pauli--Fierz Hamiltonian.

For the four ten-electron hydrides (cc-pVDZ) and the STO-3G $\Delta$QED-FCI anchors:

```bash
python run_qed_ipea_benchmark.py
```

For the aromatic molecules, the basis defaults to STO-3G so the script runs on small machines. Reproducing the paper's cc-pVDZ rows requires:

```bash
QED_IPEA_BASIS=cc-pVDZ python run_qed_ipea_aromatics.py
```

The naphthalene quasiparticle columns of the manuscript use the spin-adapted singlet implementation with density-fitted integrals and an auxiliary-basis dielectric, and omit the $\Delta$QED-CCSD rung (Sec. III A). This corresponds to:

```bash
QED_IPEA_BASIS=cc-pVDZ QED_IPEA_SINGLET_GW=1 QED_IPEA_DO_CCSD=0 python run_qed_ipea_aromatics.py
```

The environment variables `QED_IPEA_GW_SCREENING` (`aux-pade`, `aux-cd`, or `dense`) and `QED_IPEA_MEM_GB` select the dielectric backend and the memory budget above which a molecule is skipped.

The cavity-induced shifts of Table III and the relative errors of Fig. 2 are formed from the unrounded total energies stored in `qed_ipea_results.json` and `qed_ipea_aromatics_results.json`, not from the three-decimal table entries.

## Coupling-strength and detuning scans (Fig. 3)

The $\lambda$-scan ($\lambda = 0.02$ to $0.10$ at fixed $\omega_\mathrm{cav}$) and the detuning scan ($\omega_\mathrm{cav}$ varied at $\lambda = 0.05$) of the cavity-induced IP shift of H$_2$O/cc-pVDZ:

```bash
python run_qed_ip_scans.py
```

This produces the data showing that the evGW overestimate is $\lambda$-independent and that the shift is DSE-driven rather than resonance-driven (Sec. III D).

## Basis-set sensitivity of the EA shift (Sec. III D)

The cavity-induced EA shift of H$_2$O at cc-pVDZ and aug-cc-pVDZ, for $\Delta$QED-HF, $\Delta$QED-CCSD, and evGW:

```bash
python run_qed_ea_basis_check.py
```

This reproduces the statement that the magnitude of the EA shift is basis-dependent while the GW--CCSD agreement survives the addition of diffuse functions. The script imports `run_molecule()` from `run_qed_ipea_benchmark.py`.

## Absorption spectra (Figs. 4--6)

The QED-RPA spectra (photon explicit in the RPA excitation space) and the polaritonic QED-BSE@evGW spectra [Eq. (25)] at $\lambda = 0$, $0.05$, and $0.10$:

```bash
python run_qed_absorption_h2o.py
python run_qed_absorption_nh3.py
python run_qed_absorption_naph_singlet.py
```

For each molecule the cavity is tuned, at $\lambda = 0$ where the photon is decoupled, to the lowest bright root of the respective solver. The naphthalene calculation uses the spin-adapted singlet modules (`OmegaQMC.addons.qed_polariton_singlet`), which bring the working set down to a few GB. `run_qed_absorption_naph.py` is the spin-orbital version of the same calculation and requires a large-memory machine. The `plot_qed_absorption_naph_singlet*.py` scripts re-render the naphthalene spectra from `qed_absorption_naph_singlet_results.json` with different energy windows.

## Quasiparticle spectral function (Fig. 7)

The HOMO spectral function of H$_2$O at the G$_0$W$_0$ level and the $\lambda^2$-scaling polariton replica:

```bash
python run_qed_spectral_h2o.py
```

## Exciton binding energies and channel decomposition (Tables IV--V, Fig. 8)

The neutral-excitation observables ($E_\mathrm{gap}$, $\Omega_{S_1}$, $\Omega_{T_1}$, $\Delta E_\mathrm{ST}$, $E_b$) from the full QED-BSE@QED-evGW for H$_2$O, HF, NH$_3$, and CH$_4$, including the water $\lambda$-scan and its QP/DSE/photon channel decomposition:

```bash
python run_qed_exciton_binding.py
```

The two-channel decomposition of Table V (quasiparticle channel $\delta_\lambda E_b^\mathrm{elec}$ and kernel residual $\delta_\lambda E_b^\mathrm{ker}$), together with the transition-dipole and dipole-change descriptors $\mu_z^2(S_1)$ and $\Delta d_z$, for all five hydrides including PH$_3$:

```bash
python run_qed_binding_ph3.py
```

PH$_3$ is the discriminating test of the polarization criterion of Sec. III H and does not enter the IP/EA benchmark.

The side-by-side H$_2$O versus NH$_3$ comparison of Fig. 8, with each cavity tuned to the molecule's own BSE optical gap ($S_1$-resonant) and the default RPA-resonant water scan overlaid:

```bash
python run_qed_exciton_binding_compare.py
```

`run_qed_exciton_binding_nh3.py` and `run_qed_exciton_resonant.py` are the single-molecule NH$_3$ decomposition and the exciton-resonant water scan that the comparison script combines.

## Basis-set robustness of the channel decomposition (Table VI, Fig. 9)

The full QED-BSE@QED-evGW observables and the two-channel decomposition in cc-pVDZ and aug-cc-pVDZ:

```bash
python run_qed_binding_basis.py
```

By default the script runs H$_2$O and NH$_3$ in both bases; molecules and bases can be passed on the command line, for example `python run_qed_binding_basis.py H2O HF NH3 --basis cc-pVDZ aug-cc-pVDZ`. The result file `qed_binding_basis_results.json` is the source of both Table VI and Fig. 9.

## Sodium-halide comparison with published QED-CC results (Table VII)

NaF and NaCl in the setup of DePrince, J. Chem. Phys. 154, 094112 (2021): def2-TZVPPD, $\omega_\mathrm{cav} = 2.0$ eV polarized along the molecular axis, $\lambda$ from $0$ to $0.05$.

```bash
python run_qed_nax_deprince.py
```

The $\Delta$QED-HF column reproduces the published mean-field values and certifies that the QED-GW numbers use the same Hamiltonian, coherent-state reference, and gauge origin (Appendix B).

## Polaritonic QED-BSE versus QED-FCI

The Rabi splitting of the explicit-photon polaritonic QED-BSE [Eq. (25)] is benchmarked against exact polaritonic QED-FCI for H$_2$O/STO-3G in a $z$-polarized cavity, and against the two-level law $\Omega_R = \sqrt{3}\,\lambda\sqrt{f}$:

```bash
python run_qed_bse_polaritonic_fci_benchmark.py
```

This validates the lower and upper polariton branches obtained from the polaritonic QED-BSE construction used for the absorption spectra.

## Regenerating the manuscript figures

The figure files of the manuscript are built from the JSON results of this directory by the scripts in `OmegaQMC/paper/` (run from that directory):

| Script | Input data | Output |
|---|---|---|
| `make_qed_gw_overview.py` | none (schematic) | Fig. 1, `figures/qed_gw_overview.pdf` |
| `make_qed_gw_shifts.py` | `qed_ipea_results.json`, `qed_ipea_aromatics_results.json` | Fig. 2, `figures/qed_gw_shift.pdf` |
| `make_qed_gw_ipscan.py` | `qed_ip_scans_results.json` | Fig. 3, `figures/qed_gw_ipscan.pdf` |
| `make_qed_gw_exciton_binding.py` | `qed_binding_basis_results.json` | Fig. 9, `figures/qed_gw_binding.pdf` |
| `make_referee_tables.py` | `qed_binding_basis_results.json` | body of Table VI (the `AUTO excitonbasis` block of `main.tex`) |

Figs. 4--8 are written by the corresponding `run_qed_*.py` scripts into `OmegaQMC/paper/` and copied to `OmegaQMC/paper/figures/` for the manuscript.

## QED-DMRG calculations

The $\Delta$QED-DMRG columns of Tables I--III were computed with the MOLMPS program:

https://gitlab.com/molmps/scalable

The cavity mode is represented as an additional lattice site with the photon-number basis truncated at $n_\mathrm{ph}^{\max} = 10$, all molecular orbitals are correlated, localized with the Pipek--Mezey scheme, and ordered along the DMRG lattice with the Fiedler algorithm; the bond dimension is adapted to a truncation-error threshold of $10^{-6}$ (Sec. III A). The $\Delta$QED-DMRG energies provide the near-exact cc-pVDZ reference for the four hydrides, where the dense polaritonic FCI is out of reach, and agree with $\Delta$QED-CCSD to within 1 meV on the cavity-induced shifts.

## Supporting and diagnostic scripts

The remaining scripts in this directory support the analysis but are not tabulated in the manuscript:

- `run_qed_ip_frozen_orbital.py`: frozen-orbital $\Delta$-method, separating orbital relaxation from differential correlation in the cavity-induced IP shift.
- `run_qed_cost_benchmark.py`: wall-clock comparison of one QED-evGW spectrum against one QED-CCSD energy for benzene and naphthalene.
- `run_qed_ccsd_naph.py`: spin-adapted QED-CCSD ground-state energy of naphthalene.
- `run_qed_ip_benchmark_h2o.py`: earlier water-only IP benchmark superseded by `run_qed_ipea_benchmark.py`.
- `run_qed_exciton_dynamical.py`: dynamical-kernel BSE check, not part of the present manuscript.
- `run_qed_velocity_exact_check.py` and the `qed_bse_kpts_*.json`, `qed_bse_hbn_*.json` data: periodic (k-point) QED-BSE validation belonging to a separate manuscript.

## Reproducibility notes

All numerical parameters necessary to reproduce the principal calculations are specified in the example scripts. The scripts can be run independently after installation of OmegaQMC and its dependencies (PySCF, JAX, NumPy, SciPy; Matplotlib for the figures).

Computational cost varies substantially between systems and methods. The hydride benchmarks and the exciton analyses run on a laptop-class machine. The cc-pVDZ aromatics, in particular the $\Delta$QED-CCSD rung for benzene and the spin-orbital naphthalene calculations, require tens of GB of memory; the spin-adapted singlet paths (`QED_IPEA_SINGLET_GW=1`, `run_qed_absorption_naph_singlet.py`) are the memory-lean routes used for the manuscript's naphthalene results. The $\Delta$QED-FCI anchors use a dense polaritonic solver and are limited to the minimal basis for H$_2$O and HF.

## Citation

If you use the implementation or reproducibility materials provided in this repository, please cite the associated manuscript:

```bibtex
@article{Willow2026QEDGW,
  title   = {GW and Bethe--Salpeter Theory for Molecular Polaritons, Quasiparticles, and Excitons},
  author  = {Willow, Soohaeng Yoo and Sim, Gi Beom and Park, Tae Hyeon and Kim, Tae In
             and Yang, D. ChangMo and Matou{\v{s}}ek, Mikul{\'a}{\v{s}} and Brabec, Ji{\v{r}}{\'i}
             and Veis, Libor and Myung, Chang Woo},
  year    = {2026},
  note    = {Journal, volume, and DOI to be added upon publication}
}
```

## Figure and table numbering

The numbering above follows the current manuscript: Figs. 1--9 (overview, shift errors, IP scans, three absorption spectra, spectral function, channel comparison, basis-set channel decomposition), Tables I--VI in the main text, and Table VII in Appendix B.
