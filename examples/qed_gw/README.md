# Reproducibility Guide for QED-GW and QED-BSE Calculations

This repository contains the implementation, example scripts, input files, and associated data used to generate the results reported in the manuscript.

The examples cover the principal calculations presented in the paper, including:

- QED-GW ionization-potential (IP) and electron-affinity (EA) benchmarks
- QED-GW $\Delta$-method benchmarks for molecular systems
- Exciton binding energies from QED-BSE
- Cavity-induced changes in exciton binding energies
- Kernel-channel decomposition of the cavity-induced exciton-binding-energy shift
- QED-BSE absorption spectra and polaritonic excitation energies
- QED quasiparticle spectral functions
- Polaritonic QED-BSE--FCI benchmarks

The examples are intended to provide a direct route from the published calculations to the corresponding tables and figures in the manuscript.

## Repository structure

The main reproducibility scripts are summarized below.

| Script | Purpose | Manuscript result |
|---|---|---|
| `run_qed_ipea_benchmark.py` | QED-GW IP/EA benchmark calculations | Tables I--II |
| `run_qed_ipea_aromatics.py` | IP/EA calculations for aromatic molecules | Benzene and naphthalene results |
| `run_qed_exciton_binding.py` | QED-BSE exciton-binding-energy calculations and kernel-channel analysis | Table IV; H$_2$O channel decomposition |
| `run_qed_binding_ph3.py` | Exciton-binding-energy and channel analysis for PH$_3$ | Table V; PH$_3$ result |
| `run_qed_absorption_h2o.py` | QED-BSE absorption spectrum of H$_2$O | Fig. 3 |
| `run_qed_absorption_nh3.py` | QED-BSE absorption spectrum of NH$_3$ | Fig. 4 |
| `run_qed_absorption_naph_singlet.py` | Singlet absorption spectrum of naphthalene | Fig. 5 |
| `run_qed_exciton_binding_compare.py` | Comparison of cavity-induced exciton-binding-energy shifts | Fig. 7 |
| `run_qed_spectral_h2o.py` | QED quasiparticle spectral function of H$_2$O | Fig. 6 |
| `run_qed_bse_polaritonic_fci_benchmark.py` | Polaritonic QED-BSE--FCI validation | Eq. (24) |

The filenames are provided here for convenience. Each example script contains the parameters and setup required for the corresponding calculation.

## Reproducing the QED-GW IP/EA benchmarks

The IP/EA benchmark calculations evaluate charged excitations using the QED $\Delta$-method. The relevant examples include the molecular benchmark set and the aromatic-molecule calculations.

Run:

```bash
python run_qed_ipea_benchmark.py
```

and, for the aromatic systems,

```bash
python run_qed_ipea_aromatics.py
```

These calculations generate the data corresponding to the IP and EA benchmark results reported in the manuscript.

## Reproducing exciton binding energies

The QED-BSE calculations provide the neutral-excitation results and the cavity-induced changes in exciton binding energies.

For H$_2$O, HF, NH$_3$, and CH$_4$:

```bash
python run_qed_exciton_binding.py
```

The same example also performs the kernel-channel analysis used to identify the origin of the cavity-induced shift in the exciton binding energy.

The PH$_3$ calculation is provided separately:

```bash
python run_qed_binding_ph3.py
```

## Reproducing absorption spectra

The QED-BSE absorption spectra can be generated using the molecule-specific examples:

```bash
python run_qed_absorption_h2o.py
python run_qed_absorption_nh3.py
python run_qed_absorption_naph_singlet.py
```

These calculations reproduce the absorption spectra and polaritonic excitation features discussed in the manuscript.

## Comparing cavity-induced exciton-binding-energy shifts

The systematic comparison of exciton-binding-energy shifts is performed with:

```bash
python run_qed_exciton_binding_compare.py
```

This calculation generates the data used for the comparison of molecular systems with different polarization and dipole-selection characteristics.

## QED quasiparticle spectral function

The quasiparticle spectral function and the cavity-induced photoemission sideband can be reproduced with:

```bash
python run_qed_spectral_h2o.py
```

This example generates the spectral function used to analyze the polariton replica and its coupling dependence.

## Polaritonic QED-BSE--FCI benchmark

The explicit-photon polaritonic QED-BSE implementation is benchmarked against FCI using:

```bash
python run_qed_bse_polaritonic_fci_benchmark.py
```

This calculation validates the polaritonic excitation energies and the lower- and upper-polariton branches obtained from the QED-BSE formulation.

## QED-DMRG calculations

The QED-DMRG calculations reported in the manuscript were performed using the MOLMPS program:

https://gitlab.com/molmps/scalable

The corresponding input files and relevant output data are provided through the associated repository.

The QED-DMRG results are used as a high-accuracy correlated benchmark for the QED $\Delta$-method calculations. In the cc-pVDZ calculations, the QED-DMRG results provide the near-exact correlated reference used in the comparison with QED-HF and QED-CCSD.

## Reproducibility

All numerical parameters necessary to reproduce the principal calculations are specified in the example scripts and associated input files. Unless otherwise stated, the examples can be run independently after installation of the required dependencies.

For each calculation, the relevant script should be regarded as the primary reproducibility entry point. The script specifies the molecular system, basis set, cavity parameters, electronic-structure method, and other calculation settings used to generate the corresponding manuscript result.

Because computational cost can vary substantially between systems and methods, some calculations may require significant CPU memory, wall time, or parallel resources.

## Citation

If you use the implementation or reproducibility materials provided in this repository, please cite the associated manuscript:

```bibtex
[Add manuscript citation here]
```
