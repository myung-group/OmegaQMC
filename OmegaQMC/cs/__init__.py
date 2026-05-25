"""
Compressed-sensing recovery of sparse CI vectors from NN-VMC trial wavefunctions.

This subpackage implements the pipeline and pre-registered analysis for the
H4-square sample-complexity scaling experiment. The :mod:`analysis` module
is the contract between data collection and reporting and must be frozen
before data is collected.

Submodules
----------
analysis   : Pre-registered statistics, regression, and regime classification.
reference  : FCI ground truth + natural orbitals + candidate set (PySCF).
estimators : f_I(R_k) computation and Lasso/soft-threshold recovery (JAX).
scaling    : Sweep orchestrator emitting cells/aux for one (R, basis).
walkers    : Streaming HDF5 dumper/loader for the VMC walker bank.
properties : 1-RDM, dipole, quadrupole, MR diagnostics from a recovered c_hat.
mrpt       : c_hat → CASCI → NEVPT2 (post-HF multireference PT bridge);
             includes build_casci_root_matched + run_multistate_nevpt2 for
             state-specific NEVPT2 on top of Pfau-NES excited references.
transition : 1-TDM γ^(0k), length-gauge transition dipoles, oscillator
             strengths, and Natural Transition Orbitals from two c_hat
             vectors (the spectroscopy bridge for Pfau-NES excited states).
plots      : Headline scaling plots (matplotlib, imported lazily).
"""

from OmegaQMC.cs.analysis import (
    SCHEMA_VERSION,
    CELL_FIELDS,
    AUX_FIELDS,
    ScalingResult,
    validate_schema,
    apply_convergence_gate,
    compute_K_s_star_table,
    fit_scaling,
    flat_sparsity_diagnostic,
    classify_regime,
    run_analysis,
)

from OmegaQMC.cs.reference import (
    build_h4_square,
    compute_fci_reference,
    compute_casci_reference,
    K_eff,
    K_eff_table,
    max_c_corr,
    ordered_candidate_set,
    reference_determinant,
)

from OmegaQMC.cs.estimators import (
    evaluate_orbitals_on_walkers,
    evaluate_ci_wavefunction,
    f_I_matrix,
    soft_threshold,
    normalize_and_align,
    estimate_ci,
    lambda_cv,
    recovery_metrics,
)

from OmegaQMC.cs.scaling import (
    precompute_means,
    run_sweep,
)

from OmegaQMC.cs.walkers import (
    WalkerDumper,
    load_walker_bank,
)

from OmegaQMC.cs.properties import (
    reshape_chat_to_pyscf_matrix,
    compute_1rdm,
    natural_occupations_from_rdm,
    electric_dipole,
    electric_quadrupole,
    multireference_diagnostics,
    effective_unpaired_electrons,
    report_properties,
    compare_to_fci,
)

from OmegaQMC.cs.mrpt import (
    select_active_space,
    chat_to_casci_matrix,
    build_casci_from_chat,
    run_nevpt2,
    casscf_nevpt2_reference,
    compare_nevpt2,
    build_casci_root_matched,
    run_multistate_nevpt2,
)

from OmegaQMC.cs.transition import (
    compute_1tdm,
    transition_dipole,
    oscillator_strength,
    natural_transition_orbitals,
    report_transition_properties,
    print_transition_summary,
)
