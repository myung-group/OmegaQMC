"""Backward-compatible re-export — canonical code is in psi/gto.py."""

from OmegaQMC.psi.gto import (  # noqa: F401
    _build_bspline_coefs,
    _bspline_eval,
    _sanitize_J3_eeI_params,
    _PsiGTO,
    get_psi_fun,
)
