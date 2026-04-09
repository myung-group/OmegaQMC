"""
Trial wavefunction construction and interfaces.

Submodules
----------
gto  : GTO-based trial wavefunctions from PySCF.
cusp : Cusp correction parameter generation.
nn   : Neural network trial wavefunctions (Flax NNX).

Interfaces
----------
VMCTrialState  : NamedTuple holding VMC trial functions.
AFQMCTrialState : NamedTuple holding AFQMC trial data.
"""

from typing import NamedTuple, Callable, Any

import jax.numpy as jnp

from OmegaQMC.psi.gto import get_psi_fun
from OmegaQMC.psi.cusp import get_cusp_params


class VMCTrialState(NamedTuple):
    """Container for a VMC trial wavefunction.

    Attributes:
        log_psi: Callable ``(elec_crds, nuc_crds,
            params) -> float``.  Log of the trial
            wavefunction.
        log_psi_C: Callable or None. Variant that
            accepts explicit MO coefficients (GTO
            only; None for NN trials).
        energy_fns: Tuple of energy component
            callables ``(E_ee, E_nn, E_en, E_ke)``.
        get_psi_mo: MO evaluation function or None
            (GTO only; None for NN trials).
        ke_C: Kinetic energy callable with explicit
            MO coefficients or None (GTO only).
    """
    log_psi: Callable
    log_psi_C: Callable = None
    energy_fns: Any = None
    get_psi_mo: Callable = None
    ke_C: Callable = None


class AFQMCTrialState(NamedTuple):
    """Container for an AFQMC trial wavefunction.

    Attributes:
        psi_T_up: Trial alpha orbitals,
            shape (nbasis, nup) or (ndet, nbasis, nup).
        psi_T_dn: Trial beta orbitals,
            shape (nbasis, ndown) or (ndet, nbasis, ndown).
        ci_coeffs: CI coefficients, shape (ndet,).
            Length-1 array for single-det.
        half_rot_chol_a: Half-rotated Cholesky (alpha).
        half_rot_chol_b: Half-rotated Cholesky (beta).
    """
    psi_T_up: jnp.ndarray
    psi_T_dn: jnp.ndarray
    ci_coeffs: jnp.ndarray
    half_rot_chol_a: jnp.ndarray
    half_rot_chol_b: jnp.ndarray
