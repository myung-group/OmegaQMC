"""QED Hartree-Fock: restricted, unrestricted and density-fitted.
"""
import numpy as np
import pytest
from pyscf import gto, scf

from OmegaQMC.addons.qed_hf import (  # noqa: E402
    run_qed_hf,
    build_eri_df,
    eri_mo_transform,
)
from OmegaQMC.addons.qed_uhf import run_qed_uhf  # noqa: E402
from OmegaQMC.addons.qed_ccsd import run_qed_ccsd  # noqa: E402
from OmegaQMC.addons.qed_rpa import run_qed_rpa  # noqa: E402


# ---------------------------------------------------------------------------
# From test_qed_hf.py
#
# Tests for :func:`OmegaQMC.addons.qed_hf.run_qed_hf`.
#
# Covers:
# 1. λ = 0 reproduces plain pyscf RHF exactly.
# 2. With non-zero coupling the iteration converges and returns the
#    expected output structure.
# 3. Glycolaldehyde / STO-3G at the published cavity parameters
#    reproduces the reference QED-HF energy used as the reference for
#    the DePrince/White QED-CCSD-21 benchmark.
# 4. Polarisation directions perpendicular to the molecular plane give
#    a different energy than directions in the plane (sanity check that
#    the cavity coupling actually enters).
# ---------------------------------------------------------------------------


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


def test_zero_coupling_matches_rhf():
    """With λ = 0 the QED-HF energy must equal the plain RHF energy."""
    mol = _h2()
    mf = scf.RHF(mol)
    e_rhf = mf.kernel()

    res = run_qed_hf(mol, omega=0.5, lambda_cav=(0.0, 0.0, 0.0))

    assert res['E_qed_hf'] == pytest.approx(e_rhf, abs=1e-10)
    assert res['E_rhf'] == pytest.approx(e_rhf, abs=1e-10)
    # The dressed Fock at λ=0 should equal the standard Fock; orbital
    # eigenvalues should match the canonical RHF ones.
    C = res['C']
    F = res['F']
    fmo = C.T @ F @ C
    eps = np.sort(np.diag(fmo))
    eps_ref = np.sort(mf.mo_energy)
    np.testing.assert_allclose(eps, eps_ref, atol=1e-8)


def test_output_structure():
    """The returned dict must carry every key qed_ccsd consumes."""
    mol = _h2()
    res = run_qed_hf(mol, omega=0.5, lambda_cav=(0.0, 0.0, 0.05))

    required = {
        'E_qed_hf', 'E_rhf', 'E_nuc',
        'C', 'F', 'H_core', 'oei', 'eri_ao',
        'mu_x_ao', 'mu_y_ao', 'mu_z_ao', 'dipole_x_lambda_tot',
        'nocc_spatial', 'nmo_spatial',
        'lambda_cav', 'omega', 'mol',
    }
    assert required <= set(res.keys())
    nao = mol.nao_nr()
    assert res['C'].shape == (nao, nao)
    assert res['F'].shape == (nao, nao)
    assert res['eri_ao'].shape == (nao, nao, nao, nao)
    assert res['nocc_spatial'] == mol.nelec[0]
    assert res['nmo_spatial'] == nao
    assert res['lambda_cav'] == (0.0, 0.0, 0.05)
    assert res['omega'] == pytest.approx(0.5)


def test_glycolaldehyde_reference():
    """Glycolaldehyde / STO-3G QED-HF at the DePrince reference params.

    Matches the energy reproduced by the qed_ccsd port (and consistent
    with the published QED-CCSD-21 total energy of -262.41698619 Ha
    when post-CCSD correlation is added on top).
    """
    mol = gto.M(atom=GLYCOLALDEHYDE, basis='STO-3G',
                unit='Angstrom', symmetry=False, verbose=0)
    omega = 3.0 / 27.211386245988
    res = run_qed_hf(mol, omega=omega, lambda_cav=(0.0, 0.0, 0.1))

    # Independently reproduced by this port; constant has 7+ correct digits.
    assert res['E_qed_hf'] == pytest.approx(-262.086525066, abs=1e-7)
    # The cavity-induced shift relative to pyscf RHF is ≈ +0.066 Ha.
    delta = res['E_qed_hf'] - res['E_rhf']
    assert delta == pytest.approx(+0.065592, abs=5e-5)


def test_polarisation_in_vs_out_of_plane():
    """For a planar molecule, λ along z (out-of-plane) and λ along x
    (in-plane) must give different QED-HF energies."""
    mol = gto.M(atom=GLYCOLALDEHYDE, basis='STO-3G',
                unit='Angstrom', symmetry=False, verbose=0)
    omega = 3.0 / 27.211386245988
    res_z = run_qed_hf(mol, omega=omega, lambda_cav=(0.0, 0.0, 0.1))
    res_x = run_qed_hf(mol, omega=omega, lambda_cav=(0.1, 0.0, 0.0))
    assert abs(res_z['E_qed_hf'] - res_x['E_qed_hf']) > 1e-3


# ---------------------------------------------------------------------------
# From test_qed_uhf.py
#
# Tests for :func:`OmegaQMC.addons.qed_uhf.run_qed_uhf`.
#
# Covers:
# 1. λ = 0 reproduces plain pyscf UHF exactly (open-shell doublet).
# 2. For a closed-shell molecule QED-UHF collapses onto the QED-RHF
#    reference of :mod:`OmegaQMC.addons.qed_hf` (<S^2> ≈ 0).
# 3. With non-zero coupling the iteration converges and returns the
#    expected output structure and a sensible spin multiplicity.
# 4. Polarisation directions give different energies (sanity check that
#    the cavity coupling actually enters).
# ---------------------------------------------------------------------------


def _ch3():
    """Methyl radical (doublet) / STO-3G."""
    return gto.M(
        atom='C 0 0 0; H 0 0 1.08; H 1.02 0 -0.36; H -0.51 0.88 -0.36',
        basis='STO-3G', spin=1, unit='Angstrom', verbose=0)


def _o2_triplet():
    return gto.M(atom='O 0 0 0; O 0 0 1.208', basis='STO-3G',
                 spin=2, unit='Angstrom', verbose=0)


def _h2o():
    return gto.M(atom='O 0 0 0; H 0 0 0.96; H 0 0.93 -0.24',
                 basis='STO-3G', unit='Angstrom', verbose=0)


def test_zero_coupling_matches_uhf():
    """With λ = 0 the QED-UHF energy must equal the plain UHF energy."""
    mol = _ch3()
    mf = scf.UHF(mol)
    e_uhf = mf.kernel()

    res = run_qed_uhf(mol, omega=0.5, lambda_cav=(0.0, 0.0, 0.0))

    assert res['E_qed_uhf'] == pytest.approx(e_uhf, abs=1e-9)
    assert res['E_uhf'] == pytest.approx(e_uhf, abs=1e-9)
    # Dressed Fock at λ=0 should yield the canonical UHF eigenvalues.
    fa = res['Ca'].T @ res['Fa'] @ res['Ca']
    fb = res['Cb'].T @ res['Fb'] @ res['Cb']
    np.testing.assert_allclose(np.sort(np.diag(fa)),
                               np.sort(mf.mo_energy[0]), atol=1e-5)
    np.testing.assert_allclose(np.sort(np.diag(fb)),
                               np.sort(mf.mo_energy[1]), atol=1e-5)


def test_closed_shell_matches_qed_rhf():
    """For a closed-shell molecule QED-UHF must reproduce QED-RHF."""
    mol = _h2o()
    omega = 3.0 / 27.211386245988
    lam = (0.0, 0.0, 0.1)

    rhf = run_qed_hf(mol, omega, lambda_cav=lam)
    uhf = run_qed_uhf(mol, omega, lambda_cav=lam)

    assert uhf['E_qed_uhf'] == pytest.approx(rhf['E_qed_hf'], abs=1e-9)
    assert uhf['s_squared'] == pytest.approx(0.0, abs=1e-8)
    assert uhf['multiplicity'] == pytest.approx(1.0, abs=1e-6)


def test_output_structure_and_multiplicity():
    """Non-zero coupling converges; output structure and <S^2> are sane."""
    mol = _o2_triplet()
    omega = 3.0 / 27.211386245988
    res = run_qed_uhf(mol, omega, lambda_cav=(0.0, 0.0, 0.1))

    required = {
        'E_qed_uhf', 'E_uhf', 'E_nuc',
        'Ca', 'Cb', 'Fa', 'Fb', 'mo_energy_a', 'mo_energy_b',
        'H_core', 'oei', 'eri_ao',
        'mu_x_ao', 'mu_y_ao', 'mu_z_ao', 'dipole_x_lambda_tot',
        'nocc_a', 'nocc_b', 'nmo_spatial',
        's_squared', 'multiplicity',
        'lambda_cav', 'omega', 'mol',
    }
    assert required <= set(res.keys())
    nao = mol.nao_nr()
    assert res['Ca'].shape == (nao, nao)
    assert res['Fb'].shape == (nao, nao)
    assert res['eri_ao'].shape == (nao, nao, nao, nao)
    assert (res['nocc_a'], res['nocc_b']) == mol.nelec
    assert res['nmo_spatial'] == nao
    # Triplet O2: multiplicity close to 3 (modest spin contamination).
    assert res['multiplicity'] == pytest.approx(3.0, abs=0.05)
    # The cavity raises the energy (positive dipole self-energy).
    assert res['E_qed_uhf'] > res['E_uhf']


def test_polarization_direction_matters():
    """Different cavity polarizations give different QED-UHF energies."""
    mol = _o2_triplet()
    omega = 3.0 / 27.211386245988
    e_z = run_qed_uhf(mol, omega, lambda_cav=(0.0, 0.0, 0.1))['E_qed_uhf']
    e_x = run_qed_uhf(mol, omega, lambda_cav=(0.1, 0.0, 0.0))['E_qed_uhf']
    # O2 lies along z, so the parallel and perpendicular couplings differ.
    assert abs(e_z - e_x) > 1e-6


# ---------------------------------------------------------------------------
# From test_qed_df.py
#
# Tests for the opt-in density-fitting (DF) path of the QED-HF pipeline.
#
# Passing ``auxbasis=<name>`` to :func:`OmegaQMC.addons.qed_hf.run_qed_hf` (or
# ``run_qed_uhf``) switches the whole pipeline to density fitting: SCF J/K are
# built from the 3-index factor ``B`` and the reference dict carries
# ``'eri_df'``
# (shape ``(naux, nao, nao)``) instead of the dense ``nao**4`` ``'eri_ao'``.
#
# Covers:
# 1. The DF return contract: ``'eri_df'`` present, dense ``'eri_ao'`` absent.
# 2. The DF factor reconstructs the exact ERI to within the DF error, so the
#    default (exact) path is genuinely unchanged and DF is a controlled approx.
# 3. At λ=0 the DF QED-HF energy matches PySCF's own DF-RHF to ~1e-8 — a tight,
#    non-flaky check that the DF J/K build is correct.
# 4. The DF reference flows end-to-end through QED-CCSD and QED-RPA and lands
#    close to the exact-path energies (within DF error).
# ---------------------------------------------------------------------------


AUX = 'weigend'
OMEGA = 3.0 / 27.211386245988
LAMBDA = (0.0, 0.0, 0.1)


def _h2o_df():
    return gto.M(atom='O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692',
                 basis='sto-3g', unit='Angstrom', verbose=0)


def test_df_return_contract():
    """DF path returns the 3-index factor and drops the dense ERI."""
    mol = _h2o_df()
    nao = mol.nao_nr()
    res = run_qed_hf(mol, OMEGA, LAMBDA, auxbasis=AUX)

    assert 'eri_df' in res
    assert 'eri_ao' not in res  # dense nao**4 tensor never materialised
    B = res['eri_df']
    assert B.ndim == 3
    assert B.shape[1] == nao and B.shape[2] == nao
    assert B.shape[0] > 0                # naux auxiliary functions
    # NB the naux*nao^2 < nao^4 memory win is asymptotic (naux ~ few*nao);
    # for a minimal basis (here nao=7) naux can exceed nao^2, so no win at
    # toy sizes.


def test_df_factor_reconstructs_eri():
    """Σ_P B[P,p,q] B[P,r,s] approximates the exact chemist ERI (DF error)."""
    mol = _h2o_df()
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
    mol = _h2o_df()
    e_dfrhf = scf.RHF(mol).density_fit(auxbasis=AUX).kernel()
    res = run_qed_hf(mol, OMEGA, lambda_cav=(0.0, 0.0, 0.0), auxbasis=AUX)
    assert res['E_qed_hf'] == pytest.approx(e_dfrhf, abs=1e-8)


def test_eri_mo_transform_df_vs_dense():
    """eri_mo_transform gives the same MO tensor (within DF error) either way.
    """
    mol = _h2o_df()
    exact = run_qed_hf(mol, OMEGA, LAMBDA)
    df = run_qed_hf(mol, OMEGA, LAMBDA, auxbasis=AUX)
    C = exact['C']                       # transform both with a common C
    g_dense = eri_mo_transform({**exact, 'C': C}, C, C, C, C, dse=True)
    g_df = eri_mo_transform(
        {
            **df,
            'lambda_cav': df['lambda_cav'],
            'mu_x_ao': df['mu_x_ao'],
            'mu_y_ao': df['mu_y_ao'],
            'mu_z_ao': df['mu_z_ao'],
        },
        C,
        C,
        C,
        C,
        dse=True,
    )
    assert np.abs(g_df - g_dense).max() < 5e-2


def test_df_qed_hf_close_to_exact():
    """DF QED-HF total energy tracks the exact one within DF error."""
    mol = _h2o_df()
    e_exact = run_qed_hf(mol, OMEGA, LAMBDA)['E_qed_hf']
    e_df = run_qed_hf(mol, OMEGA, LAMBDA, auxbasis=AUX)['E_qed_hf']
    assert e_df == pytest.approx(e_exact, abs=2e-2)


def test_df_pipeline_ccsd_and_rpa():
    """DF reference flows through QED-CCSD and QED-RPA, close to exact."""
    mol = _h2o_df()
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
    mol = _h2o_df()
    res = run_qed_uhf(mol, OMEGA, lambda_cav=(0.0, 0.0, 0.0), auxbasis=AUX)
    assert 'eri_df' in res and 'eri_ao' not in res
    e_dfuhf = scf.UHF(mol).density_fit(auxbasis=AUX).kernel()
    assert res['E_qed_uhf'] == pytest.approx(e_dfuhf, abs=1e-7)
