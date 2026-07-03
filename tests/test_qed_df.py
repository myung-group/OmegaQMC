"""Tests for the opt-in density-fitting (DF) path of the QED-HF pipeline.

Passing ``auxbasis=<name>`` to :func:`OmegaQMC.addons.qed_hf.run_qed_hf` (or
``run_qed_uhf``) switches the whole pipeline to density fitting: SCF J/K are
built from the 3-index factor ``B`` and the reference dict carries ``'eri_df'``
(shape ``(naux, nao, nao)``) instead of the dense ``nao**4`` ``'eri_ao'``.

Covers:
1. The DF return contract: ``'eri_df'`` present, dense ``'eri_ao'`` absent.
2. The DF factor reconstructs the exact ERI to within the DF error, so the
   default (exact) path is genuinely unchanged and DF is a controlled approx.
3. At λ=0 the DF QED-HF energy matches PySCF's own DF-RHF to ~1e-8 — a tight,
   non-flaky check that the DF J/K build is correct.
4. The DF reference flows end-to-end through QED-CCSD and QED-RPA and lands
   close to the exact-path energies (within DF error).
"""

import numpy as np
import pytest
from pyscf import gto, scf

from OmegaQMC.addons.qed_hf import run_qed_hf, build_eri_df, eri_mo_transform
from OmegaQMC.addons.qed_uhf import run_qed_uhf
from OmegaQMC.addons.qed_ccsd import run_qed_ccsd
from OmegaQMC.addons.qed_rpa import run_qed_rpa

AUX = 'weigend'
OMEGA = 3.0 / 27.211386245988
LAMBDA = (0.0, 0.0, 0.1)


def _h2o():
    return gto.M(atom='O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692',
                 basis='sto-3g', unit='Angstrom', verbose=0)


def test_df_return_contract():
    """DF path returns the 3-index factor and drops the dense ERI."""
    mol = _h2o()
    nao = mol.nao_nr()
    res = run_qed_hf(mol, OMEGA, LAMBDA, auxbasis=AUX)

    assert 'eri_df' in res
    assert 'eri_ao' not in res          # dense nao**4 tensor never materialised
    B = res['eri_df']
    assert B.ndim == 3
    assert B.shape[1] == nao and B.shape[2] == nao
    assert B.shape[0] > 0                # naux auxiliary functions
    # NB the naux*nao^2 < nao^4 memory win is asymptotic (naux ~ few*nao); for a
    # minimal basis (here nao=7) naux can exceed nao^2, so no win at toy sizes.


def test_df_factor_reconstructs_eri():
    """Σ_P B[P,p,q] B[P,r,s] approximates the exact chemist ERI (DF error)."""
    mol = _h2o()
    nao = mol.nao_nr()
    eri_exact = mol.intor('int2e').reshape(nao, nao, nao, nao)
    B = build_eri_df(mol, AUX)
    eri_df = np.einsum('Ppq,Prs->pqrs', B, B, optimize=True)
    # DF is approximate but faithful: max abs error is small, not O(1).
    assert np.abs(eri_df - eri_exact).max() < 5e-2


def test_df_zero_coupling_matches_pyscf_dfrhf():
    """λ=0 DF QED-HF reproduces PySCF's own DF-RHF to ~1e-8.

    This is the DF analogue of the exact-path λ=0 == RHF check and pins the
    correctness of the density-fitted J/K build.
    """
    mol = _h2o()
    e_dfrhf = scf.RHF(mol).density_fit(auxbasis=AUX).kernel()
    res = run_qed_hf(mol, OMEGA, lambda_cav=(0.0, 0.0, 0.0), auxbasis=AUX)
    assert res['E_qed_hf'] == pytest.approx(e_dfrhf, abs=1e-8)


def test_eri_mo_transform_df_vs_dense():
    """eri_mo_transform gives the same MO tensor (within DF error) either way."""
    mol = _h2o()
    exact = run_qed_hf(mol, OMEGA, LAMBDA)
    df = run_qed_hf(mol, OMEGA, LAMBDA, auxbasis=AUX)
    C = exact['C']                       # transform both with a common C
    g_dense = eri_mo_transform({**exact, 'C': C}, C, C, C, C, dse=True)
    g_df = eri_mo_transform({**df, 'lambda_cav': df['lambda_cav'],
                             'mu_x_ao': df['mu_x_ao'], 'mu_y_ao': df['mu_y_ao'],
                             'mu_z_ao': df['mu_z_ao']}, C, C, C, C, dse=True)
    assert np.abs(g_df - g_dense).max() < 5e-2


def test_df_qed_hf_close_to_exact():
    """DF QED-HF total energy tracks the exact one within DF error."""
    mol = _h2o()
    e_exact = run_qed_hf(mol, OMEGA, LAMBDA)['E_qed_hf']
    e_df = run_qed_hf(mol, OMEGA, LAMBDA, auxbasis=AUX)['E_qed_hf']
    assert e_df == pytest.approx(e_exact, abs=2e-2)


def test_df_pipeline_ccsd_and_rpa():
    """DF reference flows through QED-CCSD and QED-RPA, close to exact."""
    mol = _h2o()
    exact = run_qed_hf(mol, OMEGA, LAMBDA)
    df = run_qed_hf(mol, OMEGA, LAMBDA, auxbasis=AUX)

    cc_exact = run_qed_ccsd(exact, verbose=False)['E_qed_ccsd_total']
    cc_df = run_qed_ccsd(df, verbose=False)['E_qed_ccsd_total']
    assert cc_df == pytest.approx(cc_exact, abs=2e-2)

    rpa_exact = run_qed_rpa(exact, verbose=False)['E_qed_rpa_corr']
    rpa_df = run_qed_rpa(df, verbose=False)['E_qed_rpa_corr']
    assert rpa_df == pytest.approx(rpa_exact, abs=1e-2)


def test_vvvv_ladder_matches_dense_contraction():
    """The batched particle-ladder helper used by the QED-CCSD backends
    (``nvir**4`` never materialised) must reproduce the dense contraction
    through the reconstructed 4-index tensor exactly — i.e. the factorised
    particle-ladder is exact algebra, not a further approximation.
    """
    from OmegaQMC.addons.qed_ccsd_utils import _vvvv_ladder

    rng = np.random.default_rng(7)
    naux, nvir, nocc = 11, 6, 4
    B_vv = rng.standard_normal((naux, nvir, nvir))
    W = np.ascontiguousarray(rng.standard_normal((nvir, nvir, nocc, nocc)))

    g = np.einsum('xac,xbd->abcd', B_vv, B_vv, optimize=True)
    ref = np.einsum('abcd,cdij->abij', g, W, optimize=True)

    out = _vvvv_ladder(B_vv, W)
    assert np.abs(out - ref).max() < 1e-12

    # In-place accumulation with a prefactor.
    seed = rng.standard_normal(ref.shape)
    out2 = _vvvv_ladder(B_vv, W, out=seed.copy(), alpha=-0.5)
    assert np.abs(out2 - (seed - 0.5 * ref)).max() < 1e-12


def test_df_uhf_return_contract_and_energy():
    """UHF DF path: same contract and λ=0 matches PySCF DF-UHF."""
    mol = _h2o()
    res = run_qed_uhf(mol, OMEGA, lambda_cav=(0.0, 0.0, 0.0), auxbasis=AUX)
    assert 'eri_df' in res and 'eri_ao' not in res
    e_dfuhf = scf.UHF(mol).density_fit(auxbasis=AUX).kernel()
    assert res['E_qed_uhf'] == pytest.approx(e_dfuhf, abs=1e-7)
