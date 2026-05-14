"""Tests for :func:`OmegaQMC.qed_ccsd.run_qed_ccsd`.

Covers:
1. λ = 0 with all photonic flags off reproduces pyscf's plain CCSD
   correlation energy.
2. H2 / STO-3G with cavity converges in a reasonable number of DIIS
   iterations and the correlation energy is sensible (negative,
   bounded).
3. Glycolaldehyde / STO-3G QED-CCSD-21 at ω = 3 eV, λ = (0, 0, 0.1)
   reproduces the published DePrince/White reference total energy
   −262.416986187 Ha.

The glycolaldehyde test is the slow one (~20 s). It is marked
``slow`` so it can be skipped with ``pytest -m 'not slow'``.
"""

import pytest
from pyscf import gto, scf, cc

from OmegaQMC.addons.qed_hf import run_qed_hf
from OmegaQMC.addons.qed_ccsd import run_qed_ccsd


GLYCOLALDEHYDE = """
    C   0.00000000   0.00000000   0.00000000
    O   0.00000000   1.23456800   0.00000000
    H   0.97075033  -0.54577032   0.00000000
    C  -1.21509881  -0.80991169   0.00000000
    H  -1.15288176  -1.89931439   0.00000000
    C  -2.43440063  -0.19144555   0.00000000
    H  -3.37262777  -0.75937214   0.00000000
    O  -2.62194056   1.12501165   0.00000000
    H  -1.71446384   1.51627790   0.00000000
"""


def _h2():
    return gto.M(atom='H 0 0 0; H 0 0 1.4', basis='STO-3G',
                 unit='Bohr', verbose=0)


def test_zero_coupling_matches_pyscf_ccsd():
    """λ = 0 and all photonic flags off must equal pyscf's CCSD."""
    mol = _h2()
    mf = scf.RHF(mol)
    mf.kernel()
    mycc = cc.CCSD(mf)
    mycc.verbose = 0
    mycc.kernel()
    e_ccsd_total = mycc.e_tot
    e_ccsd_corr = mycc.e_corr

    qedhf = run_qed_hf(mol, omega=0.5, lambda_cav=(0.0, 0.0, 0.0))
    result = run_qed_ccsd(
        qedhf,
        do_t1_01=False, do_t2_11=False, do_t2_21=False,
        do_t2_02=False, do_t2_12=False, do_t2_22=False,
        verbose=False,
    )

    assert result['E_qed_ccsd_corr'] == pytest.approx(e_ccsd_corr, abs=1e-6)
    assert result['E_qed_ccsd_total'] == pytest.approx(e_ccsd_total, abs=1e-6)


def test_h2_cavity_converges():
    """H2 / STO-3G QED-CCSD-21 with non-zero coupling converges."""
    mol = _h2()
    qedhf = run_qed_hf(mol, omega=0.5, lambda_cav=(0.0, 0.0, 0.1))
    result = run_qed_ccsd(
        qedhf,
        do_t1_01=True, do_t2_11=True, do_t2_21=True,
        do_t2_02=False, do_t2_12=False, do_t2_22=False,
        max_iter=80, verbose=False,
    )
    # CCSD correlation must lower the energy.
    assert result['E_qed_ccsd_corr'] < 0.0
    # Reasonable magnitude for an STO-3G valence correlation.
    assert -1.0 < result['E_qed_ccsd_corr'] < 0.0
    # Total energy = QED-HF + correlation.
    assert result['E_qed_ccsd_total'] == pytest.approx(
        result['E_qed_hf'] + result['E_qed_ccsd_corr'], abs=1e-10)
    # DIIS converges in well under the iteration cap.
    assert result['iterations'] < 60


@pytest.mark.slow
def test_glycolaldehyde_qedccsd21_reference():
    """QED-CCSD-21 / STO-3G total energy matches DePrince/White reference.

    The reference value -262.416986187232396 was reported by the
    original psi4-based implementation (see qed_ccsd.py at the repo
    root). This port reproduces it to ≲ 10⁻⁹ Ha.
    """
    mol = gto.M(atom=GLYCOLALDEHYDE, basis='STO-3G',
                unit='Angstrom', symmetry=False, verbose=0)
    omega = 3.0 / 27.211386245988

    qedhf = run_qed_hf(mol, omega=omega, lambda_cav=(0.0, 0.0, 0.1))
    result = run_qed_ccsd(
        qedhf,
        do_t1_01=True, do_t2_11=True, do_t2_21=True,
        do_t2_02=False, do_t2_12=False, do_t2_22=False,
        verbose=False,
    )

    E_REF = -262.416986187232396
    assert result['E_qed_ccsd_total'] == pytest.approx(E_REF, abs=1e-7)
