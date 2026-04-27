"""Unit tests for OmegaQMC.heg_2d (2D HEG system + analytical HF).

Verifies:
* Closed-shell Fermi-sea generation for the standard 2D HEG sizes.
* :func:`build_2deg_system` cell geometry: ``L = sqrt(pi N) * r_s``.
* Hartree-Fock thermodynamic-limit closed forms (Stern 1973):
    - unpolarized: ``E_HF/N = 1/(2 r_s^2) - 4 sqrt(2) / (3 pi r_s)``
    - polarized  : ``E_HF/N = 1/r_s^2 - 8 / (3 pi r_s)``
* Finite-N HF approaches the TD limit at small finite-size error.
* Cross-check: HF total + correlation energy lies above QMC reference
  (Attaccalite 2002), as it must by the variational principle.
"""

import numpy as np
import pytest

from OmegaQMC.heg_2d import (
    CLOSED_SHELL_PER_SPIN_2D,
    build_2deg_system,
    closed_shell_n_per_spin,
    generate_2d_fermi_sea,
    hf_energy_2d_finite,
    hf_energy_2d_td,
    hf_exchange_energy_2d_td,
    hf_kinetic_energy_2d_td,
    is_closed_shell_n_per_spin,
)


# Attaccalite 2002 PRL 88, 256601 — FN-DMC backflow per-electron energies.
ATTACCALITE_2002_BACKFLOW = {
    # rs : {'unpolarized': E_per_elec [Ha], 'polarized': ...}
    1:  {'unpolarized': -0.20372,    'polarized': +0.13109},
    2:  {'unpolarized': -0.25721,    'polarized': -0.19359},
    5:  {'unpolarized': -0.149518,   'polarized': -0.143610},
    10: {'unpolarized': -0.085427,   'polarized': -0.084584},
    20: {'unpolarized': -0.046385,   'polarized': -0.0462488},
    30: {'unpolarized': -0.031941,   'polarized': -0.031938},
}


# ---------------------------------------------------------------------
# Closed-shell counts
# ---------------------------------------------------------------------

def test_closed_shell_per_spin_first_seven():
    """Documented closed-shell counts: 1, 5, 9, 13, 21, 25, 29 (= shells
    of |n|^2 = 0, 1, 2, 4, 5, 8, 9 cumulatively)."""
    assert CLOSED_SHELL_PER_SPIN_2D[:7] == (1, 5, 9, 13, 21, 25, 29)


def test_is_closed_shell():
    assert is_closed_shell_n_per_spin(29)
    assert not is_closed_shell_n_per_spin(30)
    assert is_closed_shell_n_per_spin(57)


def test_closed_shell_n_per_spin_round_up():
    assert closed_shell_n_per_spin(28) == 29
    assert closed_shell_n_per_spin(29) == 29
    assert closed_shell_n_per_spin(30) == 37


# ---------------------------------------------------------------------
# Fermi-sea generation
# ---------------------------------------------------------------------

@pytest.mark.parametrize('n_per_spin', [1, 5, 9, 13, 21, 25, 29, 37, 45, 57])
def test_fermi_sea_size_matches_closed_shell(n_per_spin):
    grid = generate_2d_fermi_sea(n_per_spin)
    assert grid.shape == (n_per_spin, 2)


def test_fermi_sea_n29_shells():
    """nup=29 fills shells |n|^2 = 0, 1, 2, 4, 5, 8, 9
    with multiplicities (1, 4, 4, 4, 8, 4, 4) summing to 29."""
    grid = generate_2d_fermi_sea(29)
    n_sq = np.sum(grid ** 2, axis=1)
    shells = sorted(set(n_sq.tolist()))
    assert shells == [0, 1, 2, 4, 5, 8, 9]
    counts = [int(np.sum(n_sq == s)) for s in shells]
    assert counts == [1, 4, 4, 4, 8, 4, 4]


def test_fermi_sea_no_duplicate_kpoints():
    grid = generate_2d_fermi_sea(57)
    s = {tuple(int(x) for x in v) for v in grid}
    assert len(s) == 57


def test_fermi_sea_invalid_n_raises():
    with pytest.raises(ValueError):
        generate_2d_fermi_sea(30)


# ---------------------------------------------------------------------
# System builder
# ---------------------------------------------------------------------

def test_build_2deg_system_unpolarized_geometry():
    """L = sqrt(pi N) * rs, A = pi N rs^2."""
    rs = 2.0
    N = 58
    sys = build_2deg_system(rs, N_elec=N, polarization='unpolarized')
    assert sys['nup'] == 29
    assert sys['ndown'] == 29
    np.testing.assert_allclose(sys['L'], rs * np.sqrt(np.pi * N))
    np.testing.assert_allclose(sys['area'], np.pi * N * rs ** 2)
    np.testing.assert_allclose(sys['dk'], 2 * np.pi / sys['L'])
    assert sys['kvecs'].shape == (29, 2)
    assert sys['dim'] == 2


def test_build_2deg_system_polarized_geometry():
    sys = build_2deg_system(rs=2.0, N_elec=57, polarization='polarized')
    assert sys['nup'] == 57
    assert sys['ndown'] == 0
    assert sys['kvecs'].shape == (57, 2)


def test_build_2deg_system_odd_N_unpolarized_raises():
    with pytest.raises(ValueError):
        build_2deg_system(rs=2.0, N_elec=57, polarization='unpolarized')


def test_build_2deg_system_open_shell_raises():
    """N=4 unpolarized would mean nup=2, which is not a closed shell."""
    with pytest.raises(ValueError):
        build_2deg_system(rs=2.0, N_elec=4, polarization='unpolarized')


def test_build_2deg_system_kappa_shifts_kvecs():
    sys0 = build_2deg_system(rs=2.0, N_elec=58, polarization='unpolarized')
    kappa = np.array([0.1, -0.2])
    sys_k = build_2deg_system(
        rs=2.0, N_elec=58, polarization='unpolarized', kappa=kappa,
    )
    expected_shift = kappa * sys0['dk']
    diff = sys_k['kvecs'] - sys0['kvecs']
    np.testing.assert_allclose(diff, np.tile(expected_shift, (29, 1)))


# ---------------------------------------------------------------------
# Hartree-Fock thermodynamic limit: closed-form
# ---------------------------------------------------------------------

@pytest.mark.parametrize('rs', [1.0, 2.0, 5.0, 10.0, 30.0])
def test_hf_kinetic_td_unpolarized_matches_formula(rs):
    """T/N (unpol, TD) = 1/(2 r_s^2) — half the Fermi energy."""
    expected = 0.5 / rs ** 2
    np.testing.assert_allclose(
        hf_kinetic_energy_2d_td(rs, 'unpolarized'), expected,
    )


@pytest.mark.parametrize('rs', [1.0, 2.0, 5.0, 10.0, 30.0])
def test_hf_kinetic_td_polarized_matches_formula(rs):
    """T/N (pol, TD) = 1/r_s^2."""
    expected = 1.0 / rs ** 2
    np.testing.assert_allclose(
        hf_kinetic_energy_2d_td(rs, 'polarized'), expected,
    )


@pytest.mark.parametrize('rs', [1.0, 2.0, 5.0, 10.0])
def test_hf_exchange_td_unpolarized_matches_stern(rs):
    """E_x/N (unpol, TD) = -4 sqrt(2) / (3 pi r_s) — Stern 1973."""
    expected = -4.0 * np.sqrt(2.0) / (3.0 * np.pi * rs)
    np.testing.assert_allclose(
        hf_exchange_energy_2d_td(rs, 'unpolarized'), expected,
    )


@pytest.mark.parametrize('rs', [1.0, 2.0, 5.0, 10.0])
def test_hf_exchange_td_polarized_matches_stern(rs):
    """E_x/N (pol, TD) = -8 / (3 pi r_s)."""
    expected = -8.0 / (3.0 * np.pi * rs)
    np.testing.assert_allclose(
        hf_exchange_energy_2d_td(rs, 'polarized'), expected,
    )


def test_hf_total_td_known_values():
    """Spot-check TD HF totals at standard rs.  E_HF/N at rs=2 unpol
    should be ~-0.175 Ha (= 0.125 - 0.300)."""
    e_unpol_rs2 = hf_energy_2d_td(2.0, 'unpolarized')
    np.testing.assert_allclose(e_unpol_rs2, -0.17511, atol=1e-4)
    e_pol_rs1 = hf_energy_2d_td(1.0, 'polarized')
    np.testing.assert_allclose(e_pol_rs1, +0.15117, atol=1e-4)


# ---------------------------------------------------------------------
# Finite-N HF: must approach TD limit at large N
# ---------------------------------------------------------------------

@pytest.mark.parametrize('rs', [1.0, 2.0, 5.0, 10.0, 20.0])
def test_finite_N_HF_converges_to_TD_unpolarized(rs):
    """Finite-N HF at the larger N=114 unpolarized should be within
    ~1 mHa/electron of the TD limit at all benchmark rs."""
    sys = build_2deg_system(rs, N_elec=114, polarization='unpolarized')
    hf = hf_energy_2d_finite(sys)
    e_td = hf_energy_2d_td(rs, 'unpolarized')
    assert abs(hf['total'] - e_td) < 1e-3, (
        f"rs={rs}: finite-N HF {hf['total']:.5f} differs from TD "
        f"{e_td:.5f} by {(hf['total']-e_td)*1000:.2f} mHa"
    )


@pytest.mark.parametrize('rs', [1.0, 2.0, 5.0, 10.0])
def test_finite_N_HF_converges_to_TD_polarized(rs):
    """Same convergence test for polarized N=113."""
    sys = build_2deg_system(rs, N_elec=113, polarization='polarized')
    hf = hf_energy_2d_finite(sys)
    e_td = hf_energy_2d_td(rs, 'polarized')
    assert abs(hf['total'] - e_td) < 2e-3


def test_finite_N_HF_kinetic_consistency_N58_rs2():
    """Numerical sanity check: at N=58 rs=2 unpolarized, T/N is 0.127
    Ha (within 1.5% of the TD value 0.125 Ha)."""
    sys = build_2deg_system(rs=2.0, N_elec=58, polarization='unpolarized')
    hf = hf_energy_2d_finite(sys)
    np.testing.assert_allclose(hf['kinetic'], 0.127, atol=1e-3)


# ---------------------------------------------------------------------
# Variational sanity: HF must lie ABOVE the QMC reference
# ---------------------------------------------------------------------

@pytest.mark.parametrize('rs', [1, 2, 5, 10, 20, 30])
def test_HF_TD_above_attaccalite_QMC_unpolarized(rs):
    """Variational principle: TD HF energy must lie *above* the
    Attaccalite 2002 backflow-DMC value at every density.  The
    difference is the correlation energy (typically 10-100 mHa
    for the 2D HEG at metallic densities)."""
    e_hf = hf_energy_2d_td(float(rs), 'unpolarized')
    e_qmc = ATTACCALITE_2002_BACKFLOW[rs]['unpolarized']
    assert e_hf > e_qmc, (
        f"rs={rs}: HF {e_hf:.5f} should exceed QMC {e_qmc:.5f}"
    )
    e_corr = e_qmc - e_hf
    # Correlation energy should be negative and bounded in magnitude.
    assert -0.2 < e_corr < 0.0


@pytest.mark.parametrize('rs', [1, 2, 5, 10])
def test_HF_TD_above_attaccalite_QMC_polarized(rs):
    e_hf = hf_energy_2d_td(float(rs), 'polarized')
    e_qmc = ATTACCALITE_2002_BACKFLOW[rs]['polarized']
    assert e_hf > e_qmc
    e_corr = e_qmc - e_hf
    assert -0.05 < e_corr < 0.0
