"""Padé J1 without cusp-corrected orbitals (``params_cusp=None``).

Without the cusp correction, J1 fixes ``a = -Z`` so that it carries
the electron-nucleus cusp, and each per-element entry is ``b``.  An
``[a, b]`` entry (the format used with cusp correction) used to fail
with a broadcasting error for any molecule; it is now accepted, with
a warning, and its ``a`` ignored.  Uses H2O so that two elements with
different ``Z`` are present.
"""
import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from OmegaQMC import generate_molecular_orbitals
from OmegaQMC.psi.gto import _PsiGTO

B = {"O": 0.9, "H": 0.4}
J2 = {
    "like": jnp.array([0.25, 0.6]),
    "unlike": jnp.array([0.5, 0.4]),
}


def _h2o_atoms():
    return """
    O  0.000000  0.000000  0.221665
    H  0.000000  1.430901 -0.886659
    H  0.000000 -1.430901 -0.886659
    """


@pytest.fixture(scope="module")
def psi():
    mf = generate_molecular_orbitals(
        _h2o_atoms(), units="Bohr", basis="6-31G", dm0_pkl=None,
    )
    return _PsiGTO(mf, params_cusp=None, trial=None)


def _coords(psi):
    rng = np.random.default_rng(5)
    nuc = jnp.array(psi.mf.mol.atom_coords())
    Z = psi.mf.mol.atom_charges().astype(int)
    idx = np.repeat(np.arange(len(Z)), Z)
    elec = jnp.array(
        np.asarray(nuc)[idx] + 0.6 * rng.normal(size=(len(idx), 3))
    )
    return elec, nuc


def _params(j1):
    return {"J1_pade": j1, "J2_pade": J2}


def test_scalar_b_matches_formula(psi):
    elec, nuc = _coords(psi)
    p_b = _params({s: jnp.array(b) for s, b in B.items()})
    j1 = psi.J1(elec, nuc, p_b["J1_pade"])
    Z = psi.mf.mol.atom_charges()
    b = np.array([B[s] for s in psi.mf.mol.elements])
    r = np.linalg.norm(
        np.asarray(elec)[None] - np.asarray(nuc)[:, None], axis=-1,
    )
    ref = np.sum(-Z[:, None] * r / (1.0 + b[:, None] * r))
    assert abs(float(j1) - ref) < 1e-12


def test_ab_equals_scalar_b(psi):
    elec, nuc = _coords(psi)
    p_b = _params({s: jnp.array(b) for s, b in B.items()})
    p_ab = _params({s: jnp.array([-0.05, b]) for s, b in B.items()})
    with pytest.warns(UserWarning, match="a is fixed at -Z"):
        lp_ab = float(psi.log_trial_wavefunction(elec, nuc, p_ab))
    lp_b = float(psi.log_trial_wavefunction(elec, nuc, p_b))
    assert abs(lp_ab - lp_b) < 1e-12
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ke_ab = float(psi.local_energy_ke(elec, nuc, p_ab))
    ke_b = float(psi.local_energy_ke(elec, nuc, p_b))
    assert abs(ke_ab - ke_b) < 1e-10


@pytest.mark.parametrize("fmt", ["b", "ab"])
def test_analytic_ke_matches_hessian(psi, fmt):
    elec, nuc = _coords(psi)
    if fmt == "b":
        j1 = {s: jnp.array(b) for s, b in B.items()}
    else:
        j1 = {s: jnp.array([-0.05, b]) for s, b in B.items()}
    p = _params(j1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        e_an = float(psi.local_energy_ke(elec, nuc, p))
        e_he = float(psi._local_energy_ke_hessian(elec, nuc, p))
    assert abs(e_an - e_he) < 1e-8, (e_an, e_he)


def test_bad_length_raises(psi):
    elec, nuc = _coords(psi)
    p = _params({s: jnp.array([0.1, 0.2, b]) for s, b in B.items()})
    with pytest.raises(ValueError, match="expected b or"):
        psi.log_trial_wavefunction(elec, nuc, p)
