"""Molecular QED-GW and QED-BSE.

Kernel structure and channels, the dynamical (frequency-dependent) BSE,
the polaritonic BSE@GW, and the auxiliary-basis dielectric backend with
contour-deformation and Pade quasiparticle energies.
"""
import math
import numpy as np
import pytest
from pyscf import gto, scf, ao2mo
from pyscf.fci import direct_spin1, cistring

from OmegaQMC.addons.qed_hf import run_qed_hf  # noqa: E402
from OmegaQMC.addons.qed_bse import (  # noqa: E402
    run_qed_bse,
    _W_static,
    _eh_exchange_kernel,
    _resolve_eps_QP,
)
from OmegaQMC.addons.qed_gw import (  # noqa: E402
    run_qed_gw,
    _build_static_quantities,
    _rpa_at_eps,
)
from OmegaQMC.addons.qed_spectral import qed_gw_spectral_function  # noqa: E402
from OmegaQMC.addons.qed_bse_dynamical import (  # noqa: E402
    run_qed_bse_dynamical,
    _dynamical_A1,
)
from OmegaQMC.addons.qed_bse_polaritonic import (
    run_qed_bse_polaritonic,
)  # noqa: E402
from OmegaQMC.addons.qed_fci import _build_fci_matrices  # noqa: E402
from OmegaQMC.addons.qed_polariton_singlet import (  # noqa: E402
    _build_spatial_quantities,
    _screening_at_eps,
    _aux_setup,
    _aux_wtilde_grid,
    _aux_w_static_blocks,
    run_qed_gw_singlet,
    run_qed_bse_polaritonic_singlet,
)


# ---------------------------------------------------------------------------
# From test_qed_bse_channels.py
#
# Fast pytest checks for the QED-BSE kernel structure and the
# QED-GW spectral function (H2O/STO-3G — suitable for the suite).
#
# Covers the index-convention fix of the BSE kernel:
# * W → v̄ limit of the TDA-BSE matrix equals the QED-CIS matrix
#   (TDA on the antisymmetric QED-RPA kernel) to machine precision;
# * triplets lie below singlets and carry no oscillator strength;
# * the kernel channel toggles (include_dse / include_photon) are inert
#   at λ = 0;
# * the spectral-function peak coincides with the Newton G0W0 root.
# ---------------------------------------------------------------------------


OMEGA = 0.415668
LAMBDA = 0.05


@pytest.fixture(scope='module')
def mol():
    half = math.radians(104.5 / 2.0)
    hx, hz = math.sin(half), -math.cos(half)
    return gto.M(atom=[['O', (0, 0, 0)], ['H', (hx, 0, hz)],
                       ['H', (-hx, 0, hz)]],
                 basis='sto-3g', unit='Angstrom', symmetry=False, verbose=0)


@pytest.fixture(scope='module')
def qedhf(mol):
    return run_qed_hf(mol, OMEGA, (0.0, 0.0, LAMBDA), verbose=False)


@pytest.fixture(scope='module')
def qedhf0(mol):
    return run_qed_hf(mol, OMEGA, (0.0, 0.0, 0.0), verbose=False)


def test_bse_bare_limit_equals_cis(qedhf):
    """A^BSE with W → v̄ must equal the QED-CIS (antisymmetric RPA TDA)
    electronic block: the e-h exchange enters once, the direct
    attraction once (inside W)."""
    static = _build_static_quantities(qedhf, direct=True)
    so = static['so']
    nocc, nso = so['nocc'], so['nso']
    nvir = nso - nocc
    nov = nvir * nocc
    eps = static['eps_HF']
    diag = np.diag((eps[nocc:, None] - eps[None, :nocc]).reshape(-1))

    M0 = np.zeros((nso, nso, 1))
    K_x = _eh_exchange_kernel(static, 'aj,ib')
    W_b = _W_static(static, np.ones(1), M0, 'ij,ab').transpose(2, 0, 3, 1)
    A_bse = (K_x - W_b).reshape(nov, nov) + diag

    g_a = so['g_phys_a']
    d_so = so['d_so']
    d_vo, d_ov = d_so[nocc:, :nocc], d_so[:nocc, nocc:]
    d_oo, d_vv = d_so[:nocc, :nocc], d_so[nocc:, nocc:]
    A_cis = diag + (g_a[:nocc, nocc:, nocc:, :nocc]
                    .transpose(2, 0, 1, 3).reshape(nov, nov))
    A_cis = A_cis + (np.einsum('ai,jb->aibj', d_vo, d_ov)
                     - np.einsum('ab,ij->aibj', d_vv, d_oo)
                     ).reshape(nov, nov)
    assert np.max(np.abs(A_bse - A_cis)) < 1e-12


def test_spin_structure_and_brightness(qedhf0):
    bse = run_qed_bse(qedhf0, gw_mode='evGW', tda=True, verbose=False)
    spin = bse['spin']
    w_s = bse['w_singlet']
    f = bse['f_osc']
    # clean spin classification
    assert np.all((w_s < 1e-6) | (w_s > 1.0 - 1e-6))
    # triplets dark
    assert np.max(f[spin == 'T']) < 1e-10
    # lowest root is a triplet, below the lowest singlet
    assert spin[0] == 'T'
    assert bse['Omega_T1'] < bse['Omega_S1']
    # exciton binding positive (basis-confined LUMO)
    assert bse['E_b'] > 0.0


def test_channel_toggles_inert_at_lambda0(qedhf0):
    full = run_qed_bse(qedhf0, gw_mode='evGW', tda=True, verbose=False)
    off = run_qed_bse(qedhf0, gw_mode='evGW', tda=True,
                      include_dse=False, include_photon=False,
                      verbose=False)
    assert np.max(np.abs(full['Omega_BSE'] - off['Omega_BSE'])) < 1e-12


def test_eps_qp_override_matches_internal(qedhf0):
    full = run_qed_bse(qedhf0, gw_mode='evGW', tda=True, verbose=False)
    over = run_qed_bse(qedhf0, tda=True, eps_QP=full['eps_QP'],
                       verbose=False)
    assert np.max(np.abs(full['Omega_BSE'] - over['Omega_BSE'])) < 1e-12


def test_spectral_peak_at_qp_energy(qedhf):
    gw = run_qed_gw(qedhf, mode='G0W0', verbose=False)
    homo = qedhf['nocc_spatial'] - 1
    grid = np.linspace(-0.6, -0.1, 20001)
    sp = qed_gw_spectral_function(qedhf, orbs=[homo], omega_grid=grid,
                                  eta=2e-3, verbose=False)
    w_peak = grid[np.argmax(sp['A'][0])]
    assert abs(w_peak - gw['eps_QP'][homo]) < 1e-4


# ---------------------------------------------------------------------------
# From test_qed_bse_dynamical.py
#
# Fast pytest checks for the dynamical (frequency-dependent) QED-BSE
# (H2O/STO-3G --- suitable for the suite).
#
# Covers the dynamical-kernel implementation of
# :mod:`OmegaQMC.addons.qed_bse_dynamical`:
#
# * static-path consistency: the static spectrum that the dynamical
#   driver builds equals ``run_qed_bse`` to machine precision (guards the
#   shared ``_resolve_eps_QP`` / ``_assemble_A_static`` refactor);
# * static limit: the first-order dynamical correction A^(1) vanishes as
#   the screening poles Omega_m -> infinity (the static BSE is recovered);
# * perturbative and self-consistent modes agree;
# * the dynamical kernel red-shifts the lowest singlet with a
#   renormalization factor near unity;
# * regression of the STO-3G water static and dynamical optical gap /
#   exciton binding energy.
# ---------------------------------------------------------------------------


EV = 27.211386245988


# STO-3G water reference values (eV) at lambda = 0.05, evGW-BSE.
S1_STAT_REF, EB_STAT_REF = 11.593863, 12.651232
S1_DYN_REF, EB_DYN_REF = 11.328932, 12.916163


@pytest.fixture(scope='module')
def qedhf_dyn():
    half = math.radians(104.5 / 2.0)
    hx, hz = math.sin(half), -math.cos(half)
    mol = gto.M(atom=[['O', (0, 0, 0)], ['H', (hx, 0, hz)],
                      ['H', (-hx, 0, hz)]],
                basis='sto-3g', unit='Angstrom', symmetry=False, verbose=0)
    return run_qed_hf(mol, OMEGA, (0.0, 0.0, LAMBDA), verbose=False)


@pytest.fixture(scope='module')
def static_bse(qedhf_dyn):
    return run_qed_bse(qedhf_dyn, gw_mode='evGW', tda=True, verbose=False)


@pytest.fixture(scope='module')
def dyn_pert(qedhf_dyn):
    return run_qed_bse_dynamical(
        qedhf_dyn,
        gw_mode='evGW',
        mode='perturbative',
        n_states=20,
        verbose=False,
    )


@pytest.fixture(scope='module')
def dyn_sc(qedhf_dyn):
    return run_qed_bse_dynamical(qedhf_dyn, gw_mode='evGW',
                                 mode='selfconsistent', n_states=20,
                                 verbose=False)


def test_static_path_matches_run_qed_bse(static_bse, dyn_pert):
    """The static spectrum built inside the dynamical driver must be the
    identical eigenproblem solved by run_qed_bse."""
    assert np.max(np.abs(dyn_pert['Omega_stat']
                         - static_bse['Omega_BSE'])) < 1e-10


def test_correction_vanishes_in_static_limit(qedhf_dyn):
    """A^(1)(Omega_S) -> 0 as Omega_m -> infinity, so the dynamical BSE
    reduces to the static one."""
    eps_QP = _resolve_eps_QP(qedhf_dyn, 'evGW', 1e-3, None, False)
    static = _build_static_quantities(qedhf_dyn, direct=True)
    Omega, M_full = _rpa_at_eps(static, eps_QP)
    Om_S = 0.4  # a representative in-gap excitation energy (Ha)
    A1 = _dynamical_A1(static, Omega, M_full, eps_QP, Om_S, 1e-3)
    A1_lim = _dynamical_A1(static, Omega * 1e6, M_full, eps_QP, Om_S, 1e-3)
    assert np.linalg.norm(A1) > 1e-3  # genuinely nonzero at finite Omega_m
    assert np.linalg.norm(A1_lim) < 1e-8      # vanishes in the static limit


def test_perturbative_vs_selfconsistent(dyn_pert, dyn_sc):
    """The two dynamical schemes must agree closely for low-lying roots."""
    assert abs(dyn_pert['Omega_S1'] - dyn_sc['Omega_S1']) * EV < 5e-3
    assert abs(dyn_pert['E_b'] - dyn_sc['E_b']) * EV < 5e-3


def test_dynamical_redshifts_singlet(static_bse, dyn_pert):
    """Dynamical screening is weaker than its static limit, so it
    red-shifts the optical gap and deepens the binding."""
    assert dyn_pert['Omega_S1'] < static_bse['Omega_S1']      # red-shift
    assert dyn_pert['E_b'] > static_bse['E_b']  # stronger binding


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


# ---------------------------------------------------------------------------
# From test_qed_bse_polaritonic.py
#
# Regression tests for the polaritonic QED-BSE@GW (qed_bse_polaritonic).
#
# Covers:
# * lambda=0 structure: the polaritonic spectrum is the electronic BSE
#   exciton spectrum plus one DARK photon sitting at omega_cav.
# * two-level Rabi law: tuned to a bright BSE state, the resonant splitting
#   follows Omega_R = sqrt(3) * lambda * sqrt(f) (light-matter coupling
#   implemented correctly), and scales linearly in lambda.
# * FCI anchor (slow): the resonant Rabi splitting matches exact polaritonic
#   QED-FCI (STO-3G water) to within ~15% at a well-defined bright state.
#   The residual is exciton/oscillator-strength quality, not construction.
#
# Tolerances are loose enough for numerical drift but tight enough to catch
# a wrong coupling prefactor, a dropped photon row, or a sign error.
# ---------------------------------------------------------------------------


def _bright_bse(mol, omega, lam, lo=13.0, hi=18.0):
    """(energy_eV, f) of the brightest polaritonic-BSE root in a window."""
    qh = run_qed_hf(mol, omega, (0, 0, lam), verbose=False, tol=1e-12)
    pol = run_qed_bse_polaritonic(qh, gw_mode='evGW', tda=False, verbose=False)
    om = np.asarray(pol['Omega']) * EV
    f = np.asarray(pol['f_osc'])
    m = (om >= lo) & (om <= hi)
    i = np.where(m)[0][np.argmax(f[m])]
    return float(om[i]), float(f[i])


def test_lambda0_structure(mol):
    """At lambda=0: electronic excitons unchanged + one dark photon at w_cav.
    """
    omega = 15.0 / EV
    qh = run_qed_hf(mol, omega, (0, 0, 0.0), verbose=False, tol=1e-12)
    pol = run_qed_bse_polaritonic(qh, gw_mode='evGW', tda=False, verbose=False)
    std = run_qed_bse(qh, gw_mode='evGW', tda=False, verbose=False)

    om_pol = np.asarray(pol['Omega'])
    pw = np.asarray(pol['photon_weight'])

    # exactly one almost-pure photon root, at omega_cav, optically dark
    iph = int(np.argmax(pw))
    assert pw[iph] > 0.99
    assert abs(om_pol[iph] * EV - 15.0) < 1e-3
    assert pol['f_osc'][iph] < 1e-6

    # the remaining (electronic) roots reproduce the standard BSE excitons
    elec = np.sort(np.delete(om_pol, iph))
    std_om = np.sort(np.asarray(std['Omega_BSE']))
    n = min(len(elec), len(std_om), 8)
    assert np.max(np.abs(elec[:n] - std_om[:n])) < 1e-8


def test_two_level_rabi_law(mol):
    """Resonant splitting follows Omega_R = sqrt(3) lambda sqrt(f), linear in
    lambda.
    """
    # find a bright state at lambda=0 and tune the cavity onto it
    E0, f0 = _bright_bse(mol, 15.0 / EV, 0.0)
    omega = E0 / EV

    splits = {}
    for lam in (0.02, 0.04):
        qh = run_qed_hf(mol, omega, (0, 0, lam), verbose=False, tol=1e-12)
        pol = run_qed_bse_polaritonic(
            qh, gw_mode='evGW', tda=False, verbose=False
        )
        om = np.asarray(pol['Omega']) * EV
        f = np.asarray(pol['f_osc'])
        m = (om >= E0 - 2.0) & (om <= E0 + 2.0) & (f > 1e-4)
        es = np.sort(om[m])
        splits[lam] = float(es[-1] - es[0])
        pred = math.sqrt(3.0) * lam * math.sqrt(f0) * EV
        # within 6% of the analytic two-level resonant splitting
        assert abs(splits[lam] - pred) / pred < 0.06, (lam, splits[lam], pred)

    # splitting scales linearly with lambda (doubling lambda ~ doubles split)
    ratio = splits[0.04] / splits[0.02]
    assert abs(ratio - 2.0) < 0.15


# ---------------------------------------------------------------------------
# Exact FCI anchor (slow): polaritonic-BSE splitting vs polaritonic QED-FCI.
# ---------------------------------------------------------------------------
def _doublet_split(energies, photon_wt, e_center, half_window=1.5):
    """Rabi splitting = |dE| between the two highest-photon-weight states
    within +/- half_window of e_center (these are the LP/UP polaritons)."""
    e = np.asarray(energies)
    w = np.asarray(photon_wt)
    m = np.abs(e - e_center) <= half_window
    e_in, w_in = e[m], w[m]
    if np.sum(w_in > 0.02) < 2:
        return 0.0
    top2 = np.argsort(w_in)[-2:]
    pair = np.sort(e_in[top2])
    return float(pair[1] - pair[0])


def _fci_bright_split(mol, omega, lam, nph_max=4):
    """Polaritonic-FCI roots as (energy_eV, f, photon_number)."""
    norb = mol.nao_nr()
    nelec = mol.nelec
    enuc = mol.energy_nuc()
    mf = scf.RHF(mol); mf.verbose = 0; mf.kernel()
    hcore = mf.get_hcore()
    mu_ao = mol.intor('int1e_r', comp=3)
    quad_ao = mol.intor('int1e_rr', comp=9).reshape(3, 3, norb, norb)

    qh = run_qed_hf(mol, omega, (0, 0, lam), verbose=False, tol=1e-12)
    C = np.asarray(qh['C'])
    h1e = C.T @ hcore @ C
    eri = ao2mo.restore(1, ao2mo.full(mol, C), norb)
    dip_mo = lam * (C.T @ mu_ao[2] @ C)
    eri_dse = eri + np.einsum('pq,rs->pqrs', dip_mo, dip_mo)
    h1e_used = h1e + 0.5 * lam ** 2 * (C.T @ quad_ao[2, 2] @ C)

    H_e, D_e, nd = _build_fci_matrices(
        h1e_used, eri_dse, dip_mo, norb, nelec, enuc
    )
    H_e = 0.5 * (H_e + H_e.T); D_e = 0.5 * (D_e + D_e.T)
    if lam > 0:
        na, nb = nelec
        d0 = sum(dip_mo[i, i] for i in range(na)) + sum(
            dip_mo[i, i] for i in range(nb)
        )
        eye = np.eye(nd)
        H_e = H_e - d0 * D_e + 0.5 * d0 ** 2 * eye
        D_e = D_e - d0 * eye

    # bare z-dipole in the FCI basis (z-polarised cavity -> z-bright)
    muz_mo = C.T @ mu_ao[2] @ C
    na = cistring.num_strings(norb, nelec[0])
    nbk = cistring.num_strings(norb, nelec[1])
    muz = np.zeros((nd, nd))
    for i in range(nd):
        ci = np.zeros((na, nbk)); ia, ib = divmod(i, nbk); ci[ia, ib] = 1.0
        muz[:, i] = direct_spin1.contract_1e(muz_mo, ci, norb, nelec).ravel()
    muz = 0.5 * (muz + muz.T)

    nph = nph_max + 1
    N = nd * nph
    H = np.zeros((N, N))
    for n in range(nph):
        r0, r1 = n * nd, (n + 1) * nd
        H[r0:r1, r0:r1] = H_e + omega * n * np.eye(nd)
        if n + 1 < nph:
            cpl = math.sqrt(omega / 2.0) * math.sqrt(n + 1) * D_e
            H[(n+1)*nd:(n+2)*nd, r0:r1] += cpl
            H[r0:r1, (n+1)*nd:(n+2)*nd] += cpl.T
    H = 0.5 * (H + H.T)
    E, V = np.linalg.eigh(H)
    psi0, E0 = V[:, 0], E[0]
    out = []
    for n in range(1, min(60, N)):
        psin = V[:, n]; dE = (E[n] - E0) * EV
        t = sum(
            psi0[b * nd : (b + 1) * nd] @ (muz @ psin[b * nd : (b + 1) * nd])
            for b in range(nph)
        )
        f = (2.0 / 3.0) * (E[n] - E0) * t * t
        npho = sum(b * np.sum(psin[b*nd:(b+1)*nd] ** 2) for b in range(nph))
        out.append((dE, f, npho))
    return out


@pytest.mark.slow
def test_fci_rabi_splitting(mol):
    """Polaritonic-BSE resonant splitting matches exact FCI to ~15%."""
    # FCI brightest state at lambda=0 -> resonance
    f0 = _fci_bright_split(mol, 15.0 / EV, 0.0)
    bright = sorted(
        [(e, f) for e, f, _ in f0 if 13.0 <= e <= 18.0 and f > 1e-3],
        key=lambda t: -t[1],
    )
    E_res = bright[0][0]
    omega = E_res / EV

    # FCI splitting at lambda=0.05 (LP/UP = two highest photon-number states)
    fci = _fci_bright_split(mol, omega, 0.05)
    split_fci = _doublet_split([e for e, f, n in fci],
                               [n for e, f, n in fci], E_res)

    # polaritonic-BSE splitting at its own resonance, lambda=0.05
    Eb, fb = _bright_bse(mol, 15.0 / EV, 0.0)
    qh = run_qed_hf(mol, Eb / EV, (0, 0, 0.05), verbose=False, tol=1e-12)
    pol = run_qed_bse_polaritonic(qh, gw_mode='evGW', tda=False, verbose=False)
    split_bse = _doublet_split(np.asarray(pol['Omega']) * EV,
                               np.asarray(pol['photon_weight']), Eb)

    assert split_fci > 0.3            # sanity: there is a real splitting
    assert abs(split_bse - split_fci) / split_fci < 0.15, (
        split_bse,
        split_fci,
    )


# ---------------------------------------------------------------------------
# From test_qed_gw_aux.py
#
# Tests for the auxiliary-basis dielectric backend of the singlet QED-GW.
#
# The aux path (``screening='aux-cd' | 'aux-pade'``) replaces the dense
# (nov+1)-dimensional QED-dRPA eigensolve of :func:`run_qed_gw_singlet` by
# Woodbury dielectric inversions in the (naux+1)-dimensional auxiliary space,
# with the cavity photon folded in exactly (frequency-dependent weight on the
# dipole channel). These tests pin the two contracts:
#
# 1. The screened-interaction identity — the aux-dielectric Wt_pm(i w')
#    equals the dense sum-over-poles built from `_screening_at_eps` to
#    machine precision, for every (DSE, photon) flag combination.
# 2. Quasiparticle energies — contour deformation reproduces the dense
#    Newton solves for ALL orbitals; Pade continuation reproduces the
#    frontier states (its documented domain).
#
# Small basis (sto-3g) on purpose: the dense reference diagonalisation is
# cheap and the whole file runs in seconds (see
# tools/qed_ccsd_df_derivation/validate_gw_aux_screening.py for the larger
# 6-31g / density-fitted validation).
# ---------------------------------------------------------------------------


OMEGA_aux = 0.4
LAMBDA_aux = (0.0, 0.0, 0.05)


@pytest.fixture(scope='module')
def qedhf_aux():
    mol = gto.M(atom='O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692',
                basis='sto-3g', verbose=0)
    return run_qed_hf(mol, OMEGA_aux, LAMBDA_aux, verbose=False)


@pytest.mark.parametrize('dse', [True, False])
@pytest.mark.parametrize('photon', [True, False])
def test_wtilde_identity(qedhf_aux, dse, photon):
    """Aux dielectric == dense sum-over-poles on the imaginary axis."""
    sq = _build_spatial_quantities(qedhf_aux)
    eps = np.diag(sq['F'])
    freqs = np.array([0.0, 0.05, 0.3, 1.0, 5.0])

    Omega, M = _screening_at_eps(sq, eps, include_dse=dse,
                                 include_photon=photon)
    pole = 2.0 * Omega / (-freqs[:, None] ** 2 - Omega[None, :] ** 2)
    Wt_dense = np.einsum('pqs,ks->kpq', M * M, pole)

    setup = _aux_setup(sq, eps, include_dse=dse, include_photon=photon)
    Wt_aux = _aux_wtilde_grid(setup, freqs)

    assert np.max(np.abs(Wt_aux - Wt_dense)) < 1e-10


@pytest.mark.parametrize('mode', ['G0W0', 'evGW'])
def test_qp_aux_cd_matches_dense(qedhf_aux, mode):
    """Contour deformation reproduces the dense QP energies, all orbitals."""
    ref = run_qed_gw_singlet(qedhf_aux, mode=mode, verbose=False,
                             screening='dense')['eps_QP']
    qp = run_qed_gw_singlet(qedhf_aux, mode=mode, verbose=False,
                            screening='aux-cd', n_freq=200)['eps_QP']
    assert np.max(np.abs(qp - ref)) < 1e-5


def test_qp_aux_pade_frontier(qedhf_aux):
    """Pade continuation reproduces the frontier QP energies."""
    no = qedhf_aux['nocc_spatial']
    ref = run_qed_gw_singlet(qedhf_aux, mode='evGW', verbose=False,
                             screening='dense')['eps_QP']
    qp = run_qed_gw_singlet(qedhf_aux, mode='evGW', verbose=False,
                            screening='aux-pade', n_freq=200)['eps_QP']
    assert abs(qp[no - 1] - ref[no - 1]) < 5e-4     # HOMO
    assert abs(qp[no] - ref[no]) < 5e-4             # LUMO


def test_w_static_blocks_match_dense(qedhf_aux):
    """Static BSE W blocks: nu=0 Woodbury == dense sum over dRPA poles.

    The nu=0 dielectric is a single exact inversion (no quadrature), so
    this identity holds to machine precision."""
    sq = _build_spatial_quantities(qedhf_aux)
    eps = np.diag(sq['F'])
    no = sq['nocc']
    Omega, M = _screening_at_eps(sq, eps, include_dse=True,
                                 include_photon=False)
    inv_Om = 1.0 / Omega
    ref_ijab = -2.0 * np.einsum('ijs,abs,s->ijab', M[:no, :no, :],
                                M[no:, no:, :], inv_Om)
    ref_ibaj = -2.0 * np.einsum('ibs,ajs,s->ibaj', M[:no, no:, :],
                                M[no:, :no, :], inv_Om)
    Wc_ijab, Wc_ibaj = _aux_w_static_blocks(sq, eps)
    assert np.max(np.abs(Wc_ijab - ref_ijab)) < 1e-12
    assert np.max(np.abs(Wc_ibaj - ref_ibaj)) < 1e-12


@pytest.mark.parametrize('tda', [False, True])
def test_bse_aux_w_matches_dense(qedhf_aux, tda):
    """pol-BSE spectrum with w_screening='aux' == 'dense' (same QP input)."""
    eps_QP = run_qed_gw_singlet(qedhf_aux, mode='evGW', verbose=False)[
        'eps_QP'
    ]
    dense = run_qed_bse_polaritonic_singlet(qedhf_aux, eps_QP=eps_QP, tda=tda,
                                            verbose=False)
    aux = run_qed_bse_polaritonic_singlet(qedhf_aux, eps_QP=eps_QP, tda=tda,
                                          verbose=False, w_screening='aux')
    assert np.max(np.abs(dense['Omega'] - aux['Omega'])) < 1e-10
    assert np.max(np.abs(dense['f_osc'] - aux['f_osc'])) < 1e-10


@pytest.mark.parametrize('tda', [False, True])
def test_bse_davidson_matches_dense(qedhf_aux, tda):
    """Matrix-free paired Davidson == dense diagonalisation, lowest roots."""
    eps_QP = run_qed_gw_singlet(qedhf_aux, mode='evGW', verbose=False)[
        'eps_QP'
    ]
    dense = run_qed_bse_polaritonic_singlet(qedhf_aux, eps_QP=eps_QP, tda=tda,
                                            verbose=False)
    dav = run_qed_bse_polaritonic_singlet(qedhf_aux, eps_QP=eps_QP, tda=tda,
                                          verbose=False, solver='davidson',
                                          nroots=5, davidson_tol=1e-9)
    n = len(dav['Omega'])
    assert np.max(np.abs(dense['Omega'][:n] - dav['Omega'])) < 1e-8
    assert np.max(np.abs(dense['f_osc'][:n] - dav['f_osc'])) < 1e-8
    assert np.max(np.abs(dense['photon_weight'][:n]
                         - dav['photon_weight'])) < 1e-8


def test_unknown_screening_raises(qedhf_aux):
    with pytest.raises(ValueError, match='screening'):
        run_qed_gw_singlet(qedhf_aux, verbose=False, screening='bogus')
    with pytest.raises(ValueError, match='w_screening'):
        run_qed_bse_polaritonic_singlet(qedhf_aux, eps_QP=np.zeros(7),
                                        verbose=False, w_screening='bogus')
    with pytest.raises(ValueError, match='solver'):
        run_qed_bse_polaritonic_singlet(qedhf_aux, eps_QP=np.zeros(7),
                                        verbose=False, solver='bogus')
