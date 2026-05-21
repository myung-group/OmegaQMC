"""
Integral preparation routines for QMC calculations.

Submodules
----------
cholesky : Cholesky decomposition and half-rotation of ERIs.
qed      : QED-specific integral preparation.
"""

from OmegaQMC.integrals.cholesky import (
    chunked_cholesky,
    prepare_afqmc_integrals,
    half_rotate_cholesky,
    half_rotate_cholesky_multidet,
)
