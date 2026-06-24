"""Fast pytest checks for the dynamical (frequency-dependent) QED-BSE
(H2O/STO-3G --- suitable for the suite).

Covers the dynamical-kernel implementation of
:mod:`OmegaQMC.addons.qed_bse_dynamical`:

* static-path consistency: the static spectrum that the dynamical
  driver builds equals ``run_qed_bse`` to machine precision (guards the
  shared ``_resolve_eps_QP`` / ``_assemble_A_static`` refactor);
* static limit: the first-order dynamical correction A^(1) vanishes as
  the screening poles Omega_m -> infinity (the static BSE is recovered);
* perturbative and self-consistent modes agree;
* the dynamical kernel red-shifts the lowest singlet with a
  renormalization factor near unity;
* regression of the STO-3G water static and dynamical optical gap /
  exciton binding energy.
"""
import math

import numpy as np
import pytest
from pyscf import gto

from OmegaQMC.addons.qed_hf import run_qed_hf
from OmegaQMC.addons.qed_bse import run_qed_bse, _resolve_eps_QP
from OmegaQMC.addons.qed_gw import _build_static_quantities, _rpa_at_eps
from OmegaQMC.addons.qed_bse_dynamical import (run_qed_bse_dynamical,
                                               _dynamical_A1)

OMEGA = 0.415668
LAMBDA = 0.05
EV = 27.211386245988

# STO-3G water reference values (eV) at lambda = 0.05, evGW-BSE.
S1_STAT_REF, EB_STAT_REF = 11.593863, 12.651232
S1_DYN_REF, EB_DYN_REF = 11.328932, 12.916163


@pytest.fixture(scope='module')
def qedhf():
    half = math.radians(104.5 / 2.0)
    hx, hz = math.sin(half), -math.cos(half)
    mol = gto.M(atom=[['O', (0, 0, 0)], ['H', (hx, 0, hz)],
                      ['H', (-hx, 0, hz)]],
                basis='sto-3g', unit='Angstrom', symmetry=False, verbose=0)
    return run_qed_hf(mol, OMEGA, (0.0, 0.0, LAMBDA), verbose=False)


@pytest.fixture(scope='module')
def static_bse(qedhf):
    return run_qed_bse(qedhf, gw_mode='evGW', tda=True, verbose=False)


@pytest.fixture(scope='module')
def dyn_pert(qedhf):
    return run_qed_bse_dynamical(qedhf, gw_mode='evGW', mode='perturbative',
                                 n_states=20, verbose=False)


@pytest.fixture(scope='module')
def dyn_sc(qedhf):
    return run_qed_bse_dynamical(qedhf, gw_mode='evGW',
                                 mode='selfconsistent', n_states=20,
                                 verbose=False)


def test_static_path_matches_run_qed_bse(static_bse, dyn_pert):
    """The static spectrum built inside the dynamical driver must be the
    identical eigenproblem solved by run_qed_bse."""
    assert np.max(np.abs(dyn_pert['Omega_stat']
                         - static_bse['Omega_BSE'])) < 1e-10


def test_correction_vanishes_in_static_limit(qedhf):
    """A^(1)(Omega_S) -> 0 as Omega_m -> infinity, so the dynamical BSE
    reduces to the static one."""
    eps_QP = _resolve_eps_QP(qedhf, 'evGW', 1e-3, None, False)
    static = _build_static_quantities(qedhf, direct=True)
    Omega, M_full = _rpa_at_eps(static, eps_QP)
    Om_S = 0.4  # a representative in-gap excitation energy (Ha)
    A1 = _dynamical_A1(static, Omega, M_full, eps_QP, Om_S, 1e-3)
    A1_lim = _dynamical_A1(static, Omega * 1e6, M_full, eps_QP, Om_S, 1e-3)
    assert np.linalg.norm(A1) > 1e-3          # genuinely nonzero at finite Omega_m
    assert np.linalg.norm(A1_lim) < 1e-8      # vanishes in the static limit


def test_perturbative_vs_selfconsistent(dyn_pert, dyn_sc):
    """The two dynamical schemes must agree closely for low-lying roots."""
    assert abs(dyn_pert['Omega_S1'] - dyn_sc['Omega_S1']) * EV < 5e-3
    assert abs(dyn_pert['E_b'] - dyn_sc['E_b']) * EV < 5e-3


def test_dynamical_redshifts_singlet(static_bse, dyn_pert):
    """Dynamical screening is weaker than its static limit, so it
    red-shifts the optical gap and deepens the binding."""
    assert dyn_pert['Omega_S1'] < static_bse['Omega_S1']      # red-shift
    assert dyn_pert['E_b'] > static_bse['E_b']                # stronger binding


def test_renormalization_factor_near_unity(dyn_pert):
    """At lambda = 0.05 the correction is perturbative: Z ~ 1."""
    s_idx = [i for i in dyn_pert['idx'] if dyn_pert['spin'][i] == 'S']
    k = list(dyn_pert['idx']).index(int(s_idx[0]))
    assert 0.9 < dyn_pert['Z'][k] < 1.1


def test_regression_sto3g_water(static_bse, dyn_pert):
    """Pin the STO-3G water optical gap / binding energy, static and
    dynamical, against captured reference values."""
    assert abs(static_bse['Omega_S1'] * EV - S1_STAT_REF) < 2e-3
    assert abs(static_bse['E_b'] * EV - EB_STAT_REF) < 2e-3
    assert abs(dyn_pert['Omega_S1'] * EV - S1_DYN_REF) < 2e-3
    assert abs(dyn_pert['E_b'] * EV - EB_DYN_REF) < 2e-3
