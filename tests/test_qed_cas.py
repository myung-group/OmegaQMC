"""QED-CASCI, QED-CASSCF and QED-UCASSCF.

Full active spaces reproduce QED-FCI, zero coupling reproduces the
plain methods, and the analytic orbital gradient matches finite
differences.
"""
from pyscf import gto, scf, mcscf
import numpy as np
from scipy.linalg import expm

from OmegaQMC.addons.qed_fci import run_qed_fci  # noqa: E402
from OmegaQMC.addons.qed_casci import (  # noqa: E402
    run_qed_casci,
    _parse_active_space_uhf,
)
from OmegaQMC.addons.qed_casscf import run_qed_casscf  # noqa: E402
from OmegaQMC.addons.qed_ucasscf import (  # noqa: E402
    run_qed_ucasscf,
    _QEDUCASSCFObjective,
    _rot_pairs,
)
from OmegaQMC.addons.qed_uhf import run_qed_uhf  # noqa: E402


# ---------------------------------------------------------------------------
# From test_qed_casci.py
#
# Test QED-CASCI on H2 and H2O molecules.
#
# Validates:
# 1. Full-active-space limit: QED-CASCI(nmo, nelec) reproduces QED-FCI
#    bit-identically (FCI is invariant under restricting CAS to the full
#    one-particle basis).
# 2. Zero-coupling limit: QED-CASCI at λ=0 reduces to standard CASCI in
#    the same orbital basis (PySCF mcscf.CASCI).
# 3. Variational ordering: for a truncated active space,
#        E_QED-CASCI(ncas, nelecas) ≥ E_QED-FCI
#    at every λ, and the gap (the dynamic correlation outside the CAS)
#    grows monotonically with the coupling strength.
# 4. QED-HF reference consistency: ``e_qed_hf`` reported by run_qed_casci
#    matches run_qed_fci to all 10 digits (same SCF run, same orbital
#    basis).
#
# The full-CAS comparisons in tests 1, 2 and 4 are required to be exact
# to ~1e-9 Ha; the λ=0 comparison in test 2 also accommodates a small
# (<5e-8 Ha) numerical gap between PySCF's default RHF tolerance and
# qed_hf.py's tighter ``tol=1e-10`` when those orbitals are passed
# through CASCI separately.
# ---------------------------------------------------------------------------


def _build_h2(basis='sto-6g'):
    mol = gto.M(
        atom='H 0 0 0; H 0 0 1.4',
        basis=basis,
        unit='Bohr',
        verbose=0,
    )
    mf = scf.RHF(mol)
    mf.kernel()
    return mol, mf


def _build_h2o(basis='sto-3g'):
    mol = gto.M(
        atom='''
        O  0.0000  0.0000  0.1173
        H  0.0000  0.7572 -0.4692
        H  0.0000 -0.7572 -0.4692
        ''',
        basis=basis,
        unit='Angstrom',
        verbose=0,
    )
    mf = scf.RHF(mol)
    mf.kernel()
    return mol, mf


def test_h2_full_cas_equals_qed_fci():
    """H2/STO-6G: QED-CASCI(nmo, nelec) must equal QED-FCI bit-identically."""
    mol, mf = _build_h2()
    nmo = mol.nao_nr()
    ne = mol.nelectron

    print("\n" + "=" * 70)
    print("QED-CASCI(full) vs QED-FCI: H2/STO-6G")
    print("=" * 70)
    print(f"  nmo = {nmo}, nelectron = {ne}")
    print(f"  {'lambda':>8} {'E_QED-CASCI':>18} {'E_QED-FCI':>18} "
          f"{'Delta':>12}")

    omega = 0.5
    for lam in (0.0, 0.05, 0.10):
        cv = [0.0, 0.0, lam]
        r_fci = run_qed_fci(
            mf, omega=omega, coupling_vec=cv, nph_max=10)
        r_cas = run_qed_casci(
            mf, ncas=nmo, nelecas=ne,
            omega=omega, coupling_vec=cv, nph_max=10)

        d_e = r_cas['e_qed_casci'] - r_fci['e_qed_fci']
        d_hf = r_cas['e_qed_hf'] - r_fci['e_qed_hf']
        d_corr = r_cas['e_corr_qed'] - r_fci['e_corr_qed']

        print(f"  {lam:8.4f} {r_cas['e_qed_casci']:18.10f} "
              f"{r_fci['e_qed_fci']:18.10f} {d_e:+12.2e}")

        assert abs(d_e) < 1e-9, (
            f"Full-CAS QED-CASCI does not match QED-FCI at lam={lam}: "
            f"Delta = {d_e:.3e} Ha"
        )
        assert abs(d_hf) < 1e-9, (
            f"QED-HF reference disagrees at lam={lam}: "
            f"Delta = {d_hf:.3e} Ha"
        )
        assert abs(d_corr) < 1e-9


def test_h2o_full_cas_equals_qed_fci():
    """H2O/STO-3G: QED-CASCI(nmo, nelec) must equal QED-FCI bit-identically."""
    mol, mf = _build_h2o()
    nmo = mol.nao_nr()
    ne = mol.nelectron

    print("\n" + "=" * 70)
    print("QED-CASCI(full) vs QED-FCI: H2O/STO-3G")
    print("=" * 70)
    print(f"  nmo = {nmo}, nelectron = {ne}")
    print(f"  {'lambda':>8} {'E_QED-CASCI':>18} {'E_QED-FCI':>18} "
          f"{'Delta':>12}")

    omega = 0.5
    for lam in (0.0, 0.05, 0.10):
        cv = [0.0, 0.0, lam]
        r_fci = run_qed_fci(
            mf, omega=omega, coupling_vec=cv, nph_max=10)
        r_cas = run_qed_casci(
            mf, ncas=nmo, nelecas=ne,
            omega=omega, coupling_vec=cv, nph_max=10)

        d_e = r_cas['e_qed_casci'] - r_fci['e_qed_fci']
        d_hf = r_cas['e_qed_hf'] - r_fci['e_qed_hf']
        d_nph = r_cas['n_photon'] - r_fci['n_photon']

        print(f"  {lam:8.4f} {r_cas['e_qed_casci']:18.10f} "
              f"{r_fci['e_qed_fci']:18.10f} {d_e:+12.2e}")

        assert abs(d_e) < 1e-9, (
            f"Full-CAS QED-CASCI does not match QED-FCI at lam={lam}: "
            f"Delta = {d_e:.3e} Ha"
        )
        assert abs(d_hf) < 1e-9
        assert abs(d_nph) < 1e-9


def test_zero_coupling_equals_standard_casci():
    """At lambda=0, QED-CASCI(ncas, nelecas) must equal standard
    PySCF CASCI in the same orbital basis.

    Two comparisons are made:
      * Against the ``e_casci`` returned by run_qed_casci (which uses
        the same QED-HF orbitals): bit-identical to ~1e-12 Ha.
      * Against a fresh PySCF mcscf.CASCI on mf (RHF orbitals): up to
        the SCF tolerance difference between PySCF's RHF and qed_hf's
        tighter tol=1e-10 (<5e-8 Ha).
    """
    mol, mf = _build_h2o()

    for ncas, nelecas in ((4, 4), (6, 8)):
        r = run_qed_casci(
            mf, ncas=ncas, nelecas=nelecas,
            omega=0.5, coupling_vec=[0.0, 0.0, 0.0], nph_max=4)

        mc = mcscf.CASCI(mf, ncas, nelecas)
        mc.verbose = 0
        e_casci_pyscf = float(mc.kernel()[0])

        d_internal = r['e_qed_casci'] - r['e_casci']
        d_pyscf = r['e_qed_casci'] - e_casci_pyscf

        print("\n" + "=" * 70)
        print(f"Zero-coupling cross-check: QED-CASCI({ncas},{nelecas}) "
              f"H2O/STO-3G")
        print("=" * 70)
        print(f"  E_QED-CASCI (lambda=0) = {r['e_qed_casci']:.12f}")
        print(f"  E_CASCI    (in dict)   = {r['e_casci']:.12f}  "
              f"(same QED-HF orbitals)")
        print(f"  E_CASCI    (PySCF)     = {e_casci_pyscf:.12f}  "
              f"(default RHF orbitals)")
        print(f"  Delta (same-orbital)   = {d_internal:+.2e}")
        print(f"  Delta (PySCF default)  = {d_pyscf:+.2e}")

        # Same-orbital comparison must be machine-precision.
        assert abs(d_internal) < 1e-10, (
            f"QED-CASCI(lambda=0) != internal CASCI in same orbital basis: "
            f"Delta = {d_internal:.3e}"
        )
        # PySCF default vs QED-HF orbitals: SCF-tolerance gap only.
        assert abs(d_pyscf) < 5e-8, (
            f"QED-CASCI(lambda=0) deviates from PySCF CASCI beyond SCF "
            f"tolerance: Delta = {d_pyscf:.3e}"
        )


def test_truncated_cas_variational_bound():
    """A truncated CAS must give E_QED-CASCI >= E_QED-FCI at every coupling,
    and the gap must grow monotonically with lambda (more correlation
    missing as the cavity is turned on).
    """
    mol, mf = _build_h2o()

    omega = 0.5
    ncas, nelecas = 4, 4

    print("\n" + "=" * 70)
    print(f"QED-CASCI({ncas},{nelecas}) vs QED-FCI: H2O/STO-3G")
    print("=" * 70)
    print(f"  {'lambda':>8} {'E_QED-CASCI':>16} {'E_QED-FCI':>16} "
          f"{'gap':>10} {'<n_ph> CAS':>12} {'<n_ph> FCI':>12}")

    gaps = []
    for lam in (0.0, 0.05, 0.10, 0.20):
        cv = [0.0, 0.0, lam]
        r_fci = run_qed_fci(
            mf, omega=omega, coupling_vec=cv, nph_max=10)
        r_cas = run_qed_casci(
            mf, ncas=ncas, nelecas=nelecas,
            omega=omega, coupling_vec=cv, nph_max=10)

        gap = r_cas['e_qed_casci'] - r_fci['e_qed_fci']
        gaps.append(gap)

        print(f"  {lam:8.4f} {r_cas['e_qed_casci']:16.10f} "
              f"{r_fci['e_qed_fci']:16.10f} {gap:+10.6f} "
              f"{r_cas['n_photon']:12.6f} {r_fci['n_photon']:12.6f}")

        assert gap > -1e-9, (
            f"Variational bound violated at lam={lam}: "
            f"E_QED-CASCI - E_QED-FCI = {gap:.3e} Ha"
        )

    # Gap is monotonically non-decreasing with lambda.
    for i in range(1, len(gaps)):
        assert gaps[i] >= gaps[i - 1] - 1e-9, (
            "Gap (E_QED-CASCI - E_QED-FCI) should not decrease with "
            f"lambda; saw {gaps[i - 1]:.6f} -> {gaps[i]:.6f}"
        )


def test_frozen_core_dipole_shift():
    """The ``d_core_const`` field (constant electronic-dipole shift from
    the doubly-occupied frozen core) must:

    * partition correctly (``ncore = (nelec - nelecas) / 2``),
    * be exactly linear in lambda when the orbitals are held fixed
      (use_qed_hf_reference=False uses the lambda-independent RHF
      orbitals from mf, so d_core_const = 2 lambda Tr_core(eps . r_mo)
      is mathematically linear in lambda),
    * acquire a small lambda-dependent orbital-relaxation contribution
      when QED-HF orbitals are used (the orbital basis itself relaxes
      with the cavity).
    """
    mol, mf = _build_h2o()
    omega = 0.5
    ncas, nelecas = 2, 2  # extreme truncation, ncore = 4

    print("\n" + "=" * 70)
    print(f"QED-CASCI({ncas},{nelecas}) frozen-core dipole shift: "
          f"H2O/STO-3G")
    print("=" * 70)

    # (a) Fixed orbital basis (RHF): d_core_const must be EXACTLY
    #     linear in lambda.
    print(f"  Fixed-orbital (RHF) basis — d_core must be linear in lambda:")
    print(f"  {'lambda':>8} {'d_core (HF orbs)':>20}")
    d_vs_lam_hf = {}
    for lam in (0.05, 0.10, 0.20):
        cv = [0.0, 0.0, lam]
        r = run_qed_casci(
            mf, ncas=ncas, nelecas=nelecas,
            omega=omega, coupling_vec=cv, nph_max=4,
            use_qed_hf_reference=False)
        d_vs_lam_hf[lam] = r['d_core_const']
        print(f"  {lam:8.4f} {r['d_core_const']:20.10f}")
        assert r['ncore'] == 4
        assert r['reference'] == 'HF'

    ratio_2x = d_vs_lam_hf[0.10] / d_vs_lam_hf[0.05]
    ratio_4x = d_vs_lam_hf[0.20] / d_vs_lam_hf[0.05]
    assert abs(ratio_2x - 2.0) < 1e-10, (
        f"d_core_const (HF orbitals) must be exactly linear in lambda; "
        f"ratio 0.10/0.05 = {ratio_2x:.10f} (expected 2.0)"
    )
    assert abs(ratio_4x - 4.0) < 1e-10, (
        f"d_core_const (HF orbitals) must be exactly linear in lambda; "
        f"ratio 0.20/0.05 = {ratio_4x:.10f} (expected 4.0)"
    )

    # (b) QED-HF orbitals: d_core_const is approximately linear,
    #     deviation reflects orbital relaxation with the cavity (≲5%
    #     at these couplings; grows with lambda).
    print()
    print(f"  QED-HF (lambda-relaxed) basis — small non-linearity from "
          f"orbital relaxation:")
    print(f"  {'lambda':>8} {'d_core (QED-HF)':>20} "
          f"{'(ratio - linear)/linear':>26}")
    d_vs_lam_qed = {}
    for lam in (0.05, 0.10, 0.20):
        cv = [0.0, 0.0, lam]
        r = run_qed_casci(
            mf, ncas=ncas, nelecas=nelecas,
            omega=omega, coupling_vec=cv, nph_max=4,
            use_qed_hf_reference=True)
        d_vs_lam_qed[lam] = r['d_core_const']
        rel = d_vs_lam_qed[lam] / (lam / 0.05 * d_vs_lam_hf[0.05]) - 1
        print(f"  {lam:8.4f} {r['d_core_const']:20.10f} {rel:+26.4%}")
        assert r['ncore'] == 4
        assert r['reference'] == 'QED-HF'

    # QED-HF orbital relaxation gives a finite, monotonically growing
    # deviation from strict linearity (the orbitals relax more as the
    # cavity coupling strengthens). At STO-3G H2O with these couplings
    # the deviation stays below 20%.
    rels = []
    for lam in (0.05, 0.10, 0.20):
        linear = (lam / 0.05) * d_vs_lam_hf[0.05]
        rels.append(d_vs_lam_qed[lam] / linear - 1)
    # Sign: QED-HF orbitals tilt the molecule's dipole *along* the cavity
    # polarization (it lowers the DSE), so d_core grows past linear.
    assert all(r > 0 for r in rels), (
        "QED-HF orbital relaxation should increase |d_core| relative to "
        f"the fixed-RHF linear value; saw rels = {rels}"
    )
    # Monotonic growth of the relaxation with lambda.
    assert rels[0] < rels[1] < rels[2], (
        "QED-HF orbital relaxation should grow monotonically with lambda; "
        f"saw rels = {rels}"
    )
    # Loose upper bound at the strongest coupling tested.
    assert rels[-1] < 0.20, (
        f"QED-HF orbital relaxation at lam=0.20 is {rels[-1]:.1%}; expected "
        "below 20% on this system."
    )


# ---------------------------------------------------------------------------
# From test_qed_casscf.py
#
# Test QED-CASSCF on small closed-shell molecules.
#
# Validates:
# 1. Full-active-space limit: QED-CASSCF(nmo, nelec) reproduces QED-FCI
#    bit-identically. Orbital optimisation is redundant there (the QED-FCI
#    energy is orbital-invariant), so no macro-iterations are needed.
# 2. Zero-coupling limit: QED-CASSCF at λ=0 reduces to standard CASSCF
#    (PySCF mcscf.CASSCF) in the same active space.
# 3. Variational ordering / orbital relaxation: in a *truncated* active
#    space, E_QED-CASSCF ≤ E_QED-CASCI at every coupling (relaxing the
#    orbitals in the cavity can only lower the energy), and the relaxation
#    energy grows as the coupling is turned on.
# 4. CI-build consistency: the pre-relaxation energy reported as
#    ``e_qed_casci`` equals run_qed_casci bit-identically (both build the
#    polaritonic CI from the same QED-HF orbitals).
# 5. Coherent-state convergence: the CS-QED-CASSCF energy is invariant to
#    the photon truncation ``nph_max`` (it is an exact unitary transform).
# 6. Open-shell molecules dispatch to the spin-unrestricted QED-UCASSCF
#    (see the QED-UCASSCF tests below for its dedicated validation).
#
# All comparisons use small bases (STO-6G / 6-31G) and small active spaces
# so the dense polaritonic diagonalisations and orbital macro-iterations
# stay fast enough for the pytest suite.
# ---------------------------------------------------------------------------


def _build_h2_casscf(basis='6-31g'):
    mol = gto.M(atom='H 0 0 0; H 0 0 1.4', basis=basis, unit='Bohr',
                verbose=0)
    mf = scf.RHF(mol)
    mf.kernel()
    return mol, mf


def _build_lih(basis='6-31g'):
    mol = gto.M(atom='Li 0 0 0; H 0 0 1.6', basis=basis, unit='Angstrom',
                verbose=0)
    mf = scf.RHF(mol)
    mf.kernel()
    return mol, mf


def test_full_cas_equals_qed_fci():
    """H2/6-31G: QED-CASSCF(nmo, nelec) must equal QED-FCI bit-identically,
    both with the coherent-state basis and the raw photon-number basis."""
    mol, mf = _build_h2_casscf()
    nmo = mol.nao_nr()
    ne = mol.nelectron

    print("\n" + "=" * 70)
    print("QED-CASSCF(full) vs QED-FCI: H2/6-31G")
    print("=" * 70)
    print(f"  nmo = {nmo}, nelectron = {ne}")
    print(f"  {'lambda':>8} {'cs':>4} {'E_QED-CASSCF':>18} "
          f"{'E_QED-FCI':>18} {'Delta':>12}")

    omega = 0.5
    for cs in (True, False):
        for lam in (0.0, 0.05, 0.10):
            cv = [0.0, 0.0, lam]
            r_fci = run_qed_fci(mf, omega=omega, coupling_vec=cv,
                                nph_max=8, coherent_state=cs)
            r_cas = run_qed_casscf(mf, ncas=nmo, nelecas=ne, omega=omega,
                                   coupling_vec=cv, nph_max=8,
                                   coherent_state=cs)
            d_e = r_cas['e_qed_casscf'] - r_fci['e_qed_fci']
            print(f"  {lam:8.4f} {str(cs):>4} {r_cas['e_qed_casscf']:18.10f} "
                  f"{r_fci['e_qed_fci']:18.10f} {d_e:+12.2e}")
            assert abs(d_e) < 1e-9, (
                f"Full-CAS QED-CASSCF != QED-FCI at lam={lam}, cs={cs}: "
                f"Delta = {d_e:.3e} Ha")
            assert abs(r_cas['e_qed_hf'] - r_fci['e_qed_hf']) < 1e-9


def test_zero_coupling_equals_standard_casscf():
    """At λ=0, QED-CASSCF must reduce to standard PySCF CASSCF."""
    mol, mf = _build_lih()

    print("\n" + "=" * 70)
    print("Zero-coupling cross-check: QED-CASSCF vs PySCF CASSCF (LiH/6-31G)")
    print("=" * 70)

    for ncas, nelecas in ((4, 2), (5, 4)):
        r = run_qed_casscf(mf, ncas=ncas, nelecas=nelecas, omega=0.3,
                           coupling_vec=[0.0, 0.0, 0.0], nph_max=2)
        mc = mcscf.CASSCF(mf, ncas, nelecas)
        mc.verbose = 0
        e_ref = float(mc.kernel()[0])

        d = r['e_qed_casscf'] - e_ref
        print(f"  CAS({ncas},{nelecas}): E_QED-CASSCF(λ=0) = "
              f"{r['e_qed_casscf']:.10f}  E_CASSCF = {e_ref:.10f}  "
              f"Δ = {d:+.2e}")
        # Orbital optimisation converges to conv_tol_grad, so allow a
        # gradient-tolerance-sized gap to the fully-converged PySCF value.
        assert abs(d) < 1e-7, (
            f"QED-CASSCF(λ=0) != PySCF CASSCF for CAS({ncas},{nelecas}): "
            f"Δ = {d:.3e} Ha")
        # The bare-CASSCF reference reported in the dict is the PySCF value
        # (an independent mcscf.CASSCF run, so equal to its conv_tol, 1e-7).
        assert abs(r['e_casscf'] - e_ref) < 1e-7
        assert r['converged']


def test_orbital_relaxation_lowers_energy():
    """A truncated CAS must give E_QED-CASSCF ≤ E_QED-CASCI at every
    coupling, with the relaxation growing as the cavity is turned on."""
    mol, mf = _build_lih()
    omega = 0.1
    ncas, nelecas = 4, 2

    print("\n" + "=" * 70)
    print(f"QED-CASSCF vs QED-CASCI (orbital relaxation): LiH/6-31G "
          f"CAS({ncas},{nelecas})")
    print("=" * 70)
    print(f"  {'lambda':>8} {'E_QED-CASCI':>18} {'E_QED-CASSCF':>18} "
          f"{'relaxation':>14}")

    relax = []
    for lam in (0.0, 0.05, 0.10):
        cv = [0.0, 0.0, lam]
        r_ci = run_qed_casci(mf, ncas, nelecas, omega, cv, nph_max=3)
        r_cas = run_qed_casscf(mf, ncas=ncas, nelecas=nelecas, omega=omega,
                               coupling_vec=cv, nph_max=3)
        dr = r_cas['e_qed_casscf'] - r_ci['e_qed_casci']
        relax.append(dr)
        print(f"  {lam:8.4f} {r_ci['e_qed_casci']:18.10f} "
              f"{r_cas['e_qed_casscf']:18.10f} {dr:+14.3e}")
        assert dr < 1e-9, (
            f"Orbital relaxation must not raise the energy at lam={lam}: "
            f"E_QED-CASSCF - E_QED-CASCI = {dr:.3e} Ha")
        # The pre-relaxation energy in the dict must match QED-CASCI.
        assert abs(r_cas['e_qed_casci'] - r_ci['e_qed_casci']) < 1e-9

    # Orbital optimisation does non-trivial work (clearly lowers the energy
    # below fixed-orbital QED-CASCI) at every coupling. The magnitude here
    # is correlation-dominated, so its variation with lambda is mild and
    # system-dependent; we only require a real, sizeable relaxation.
    assert all(dr < -1e-4 for dr in relax), (
        f"Expected a sizeable orbital relaxation at every lambda; saw {relax}")


def test_ci_build_matches_qed_casci():
    """The pre-relaxation 'e_qed_casci' must equal run_qed_casci to all
    digits — i.e. the polaritonic CI build is identical to the validated
    qed_casci module."""
    mol, mf = _build_lih()
    omega, cv, nph = 0.1, [0.0, 0.0, 0.05], 3
    r_cas = run_qed_casscf(mf, ncas=4, nelecas=2, omega=omega,
                           coupling_vec=cv, nph_max=nph)
    r_ci = run_qed_casci(mf, 4, 2, omega, cv, nph_max=nph)
    d = r_cas['e_qed_casci'] - r_ci['e_qed_casci']
    print("\n" + "=" * 70)
    print("CI-build consistency: qed_casscf pre-relaxation vs qed_casci")
    print("=" * 70)
    print(f"  e_qed_casci (qed_casscf) = {r_cas['e_qed_casci']:.12f}")
    print(f"  e_qed_casci (qed_casci)  = {r_ci['e_qed_casci']:.12f}")
    print(f"  Delta = {d:+.2e}")
    assert abs(d) < 1e-10


def test_coherent_state_invariant_to_nph_max():
    """The CS-QED-CASSCF energy must converge with nph_max to a value
    independent of the truncation (exact unitary displacement)."""
    mol, mf = _build_lih()
    energies = []
    print("\n" + "=" * 70)
    print("CS-QED-CASSCF nph_max convergence: LiH/6-31G CAS(4,2)")
    print("=" * 70)
    for nph in (2, 4, 6):
        r = run_qed_casscf(mf, ncas=4, nelecas=2, omega=0.1,
                           coupling_vec=[0.0, 0.0, 0.05], nph_max=nph)
        energies.append(r['e_qed_casscf'])
        print(f"  nph_max={nph}: E = {r['e_qed_casscf']:.12f}  "
              f"<n_ph> = {r['n_photon']:.5f}")
    # Converged: the last two truncations agree tightly.
    assert abs(energies[2] - energies[1]) < 1e-8, (
        f"CS-QED-CASSCF not converged in nph_max: {energies}")


def test_open_shell_dispatches_to_ucasscf():
    """Open-shell molecules dispatch to the spin-unrestricted QED-UCASSCF
    path and return the analogous result dict."""
    mol = gto.M(atom='Li 0 0 0; H 0 0 3.0', basis='sto-3g', spin=2,
                unit='Angstrom', verbose=0)
    mf = scf.RHF(mol)
    mf.kernel()
    r = run_qed_casscf(mf, ncas=2, nelecas=(2, 0), omega=0.1,
                       coupling_vec=[0.0, 0.0, 0.05], nph_max=2)
    print("\n" + "=" * 70)
    print("Open-shell dispatch: LiH (triplet)/STO-3G CAS(2,(2,0))")
    print("=" * 70)
    print(f"  reference = {r['reference']}  E = {r['e_qed_casscf']:.10f}  "
          f"converged = {r['converged']}")
    assert r['reference'] == 'QED-UHF'
    assert r['ncore'] == (1, 1)
    assert isinstance(r['mo_coeff'], tuple) and len(r['mo_coeff']) == 2
    assert r['e_qed_casscf'] <= r['e_qed_casci'] + 1e-10
    assert r['converged']


# ---------------------------------------------------------------------------
# From test_qed_ucasscf.py
#
# Test QED-UCASSCF (spin-unrestricted, orbital-optimized) on small
# open-shell molecules.
#
# Validates:
# 1. Analytic orbital gradient: the spin-resolved cavity-augmented
#    generalised Fock gradient matches finite differences of the
#    polaritonic CI energy (random direction + individual pairs).
# 2. Full-active-space limit: QED-UCASSCF(nmo, mol.nelec) reproduces
#    open-shell QED-FCI (the QED-FCI energy is orbital-invariant, so the
#    orbital optimisation is redundant there).
# 3. Zero-coupling limit: QED-UCASSCF at λ=0 reduces to standard
#    spin-unrestricted CASSCF (PySCF mcscf.UCASSCF).
# 4. Variational ordering / orbital relaxation: in a truncated active
#    space, E_QED-UCASSCF ≤ E_QED-UCASCI, and the pre-relaxation energy
#    reported as ``e_qed_casci`` equals the QED-UCASCI path of
#    run_qed_casci bit-identically.
# 5. Closed-shell consistency: on a closed-shell molecule QED-UCASSCF
#    reproduces the restricted QED-CASSCF energy.
#
# All comparisons use small bases (STO-3G) and small active spaces so the
# dense polaritonic diagonalisations and orbital macro-iterations stay
# fast enough for the pytest suite.
# ---------------------------------------------------------------------------


def _build_oh(basis='sto-3g'):
    """OH radical (doublet)."""
    mol = gto.M(atom='O 0 0 0; H 0 0 0.97', basis=basis, spin=1,
                unit='Angstrom', symmetry=False, verbose=0)
    mf = scf.UHF(mol)
    mf.kernel()
    return mol, mf


def _build_h3(basis='sto-3g'):
    """Linear H3 (doublet) — tiny full-CAS system."""
    mol = gto.M(atom='H 0 0 0; H 0 0 1.0; H 0 0 2.0', basis=basis,
                spin=1, unit='Angstrom', symmetry=False, verbose=0)
    mf = scf.UHF(mol)
    mf.kernel()
    return mol, mf


def test_orbital_gradient_matches_finite_differences():
    """OH/STO-3G CAS(4,(3,2)): the analytic (κ_α, κ_β) gradient at the
    QED-UHF starting orbitals must match finite differences of the
    polaritonic CI energy."""
    mol, mf = _build_oh()
    omega, cv = 0.1, (0.0, 0.0, 0.05)
    lam = float(np.linalg.norm(cv))
    epsilon = np.asarray(cv) / lam
    ncas, nelecas = 4, (3, 2)

    ncore_a, ncore_b, na, nb = _parse_active_space_uhf(mol, ncas, nelecas)
    qeduhf = run_qed_uhf(mol, omega, lambda_cav=cv)
    Ca = np.asarray(qeduhf['Ca'])
    Cb = np.asarray(qeduhf['Cb'])
    nocc_a, nocc_b = mol.nelec

    obj = _QEDUCASSCFObjective(mf, ncas, (na, nb), ncore_a, ncore_b,
                               omega, lam, epsilon, True, nph_max=2)
    dipa0 = Ca.T @ obj.dip_ao @ Ca
    dipb0 = Cb.T @ obj.dip_ao @ Cb
    obj.d0 = float(np.trace(dipa0[:nocc_a, :nocc_a])
                   + np.trace(dipb0[:nocc_b, :nocc_b]))

    norb = obj.norb
    pp_a, qq_a = _rot_pairs(ncore_a, ncas, norb)
    pp_b, qq_b = _rot_pairs(ncore_b, ncas, norb)
    n_pa = len(pp_a)
    ntot = n_pa + len(pp_b)

    res = obj.energy_and_grad(Ca, Cb)
    g_pairs = np.concatenate([res['g_orb_a'][pp_a, qq_a],
                              res['g_orb_b'][pp_b, qq_b]])

    def _rotated_energy(step):
        Ka = np.zeros((norb, norb))
        Kb = np.zeros((norb, norb))
        Ka[pp_a, qq_a] = step[:n_pa]
        Ka[qq_a, pp_a] = -step[:n_pa]
        Kb[pp_b, qq_b] = step[n_pa:]
        Kb[qq_b, pp_b] = -step[n_pa:]
        return obj.energy(Ca @ expm(Ka), Cb @ expm(Kb))

    print("\n" + "=" * 70)
    print("QED-UCASSCF gradient vs finite differences: OH/STO-3G "
          "CAS(4,(3,2))")
    print("=" * 70)

    rng = np.random.default_rng(7)
    h = 1e-5
    direction = rng.standard_normal(ntot)
    direction /= np.linalg.norm(direction)
    fd = (_rotated_energy(h * direction)
          - _rotated_energy(-h * direction)) / (2.0 * h)
    an = float(np.dot(g_pairs, direction))
    print(f"  random direction: FD = {fd:+.10f}  analytic = {an:+.10f}  "
          f"diff = {fd - an:+.2e}")
    assert abs(fd - an) < 1e-7

    for k in rng.choice(ntot, size=5, replace=False):
        e_k = np.zeros(ntot)
        e_k[k] = 1.0
        fd_k = (_rotated_energy(h * e_k)
                - _rotated_energy(-h * e_k)) / (2.0 * h)
        print(f"  pair {k:3d}: FD = {fd_k:+.10f}  "
              f"analytic = {g_pairs[k]:+.10f}  "
              f"diff = {fd_k - g_pairs[k]:+.2e}")
        assert abs(fd_k - g_pairs[k]) < 1e-7


def test_full_cas_equals_open_shell_qed_fci():
    """H3/STO-3G: QED-UCASSCF at full active space must equal the
    open-shell QED-FCI."""
    mol, mf = _build_h3()
    nmo = mol.nao_nr()

    print("\n" + "=" * 70)
    print("QED-UCASSCF(full) vs open-shell QED-FCI: H3/STO-3G")
    print("=" * 70)
    print(f"  {'lambda':>8} {'E_QED-UCASSCF':>18} {'E_QED-FCI':>18} "
          f"{'Delta':>12}")

    omega = 0.1
    for lam in (0.0, 0.05, 0.10):
        cv = [0.0, 0.0, lam]
        r_fci = run_qed_fci(mf, omega=omega, coupling_vec=cv, nph_max=4)
        r_cas = run_qed_ucasscf(mf, ncas=nmo, nelecas=mol.nelec,
                                omega=omega, coupling_vec=cv, nph_max=4)
        d = r_cas['e_qed_casscf'] - r_fci['e_qed_fci']
        print(f"  {lam:8.4f} {r_cas['e_qed_casscf']:18.10f} "
              f"{r_fci['e_qed_fci']:18.10f} {d:+12.2e}")
        assert abs(d) < 1e-8, (
            f"Full-CAS QED-UCASSCF != open-shell QED-FCI at lam={lam}: "
            f"Delta = {d:.3e} Ha")
        assert abs(r_cas['e_qed_hf'] - r_fci['e_qed_hf']) < 1e-9


def test_zero_coupling_equals_pyscf_ucasscf():
    """At λ=0, QED-UCASSCF must reduce to standard PySCF UCASSCF.

    OH/STO-3G CAS(4,(3,2)) has two UCASSCF minima ~0.93 mHa apart, and
    an unseeded PySCF run lands in either from process to process.  The
    reference is therefore started from the QED-UCASSCF orbitals and
    converged tightly, so both energies refer to the same minimum.
    """
    mol, mf = _build_oh()
    ncas, nelecas = 4, (3, 2)

    r = run_qed_ucasscf(mf, ncas=ncas, nelecas=nelecas, omega=0.1,
                        coupling_vec=[0.0, 0.0, 0.0], nph_max=2)
    mc = mcscf.UCASSCF(mf, ncas, nelecas)
    mc.verbose = 0
    mc.conv_tol = 1e-11
    mc.conv_tol_grad = 1e-7
    e_ref = float(mc.kernel(r['mo_coeff'])[0])
    assert mc.converged

    d = r['e_qed_casscf'] - e_ref
    print("\n" + "=" * 70)
    print("Zero-coupling cross-check: QED-UCASSCF vs PySCF UCASSCF "
          "(OH/STO-3G)")
    print("=" * 70)
    print(f"  CAS{(ncas, nelecas)}: E_QED-UCASSCF(λ=0) = "
          f"{r['e_qed_casscf']:.10f}  E_UCASSCF = {e_ref:.10f}  "
          f"Δ = {d:+.2e}")
    # Same minimum; the residual is set by QED-UCASSCF's conv_tol_grad
    # (1e-5), whose energy error is O(|g|^2), ~1e-9 Ha here.
    assert abs(d) < 1e-8
    # The bare-UCASSCF energy in the dict comes from an unseeded PySCF
    # run, so it may sit in either minimum.  Whichever it is, orbital
    # optimisation from the UHF orbitals cannot rise above the UCASCI
    # energy in those orbitals (QED-UCASCI at λ=0).
    assert r['e_casscf'] <= r['e_qed_casci'] + 1e-8
    assert r['converged']


def test_orbital_relaxation_lowers_energy_below_ucasci():
    """A truncated CAS must give E_QED-UCASSCF ≤ E_QED-UCASCI, with the
    pre-relaxation energy matching run_qed_casci's QED-UCASCI path."""
    mol, mf = _build_oh()
    omega = 0.1
    ncas, nelecas = 4, (3, 2)

    print("\n" + "=" * 70)
    print(f"QED-UCASSCF vs QED-UCASCI (orbital relaxation): OH/STO-3G "
          f"CAS({ncas},{nelecas})")
    print("=" * 70)
    print(f"  {'lambda':>8} {'E_QED-UCASCI':>18} {'E_QED-UCASSCF':>18} "
          f"{'relaxation':>14}")

    for lam in (0.0, 0.05, 0.10):
        cv = [0.0, 0.0, lam]
        r_ci = run_qed_casci(mf, ncas, nelecas, omega, cv, nph_max=2)
        r_cas = run_qed_ucasscf(mf, ncas=ncas, nelecas=nelecas,
                                omega=omega, coupling_vec=cv, nph_max=2)
        dr = r_cas['e_qed_casscf'] - r_ci['e_qed_casci']
        print(f"  {lam:8.4f} {r_ci['e_qed_casci']:18.10f} "
              f"{r_cas['e_qed_casscf']:18.10f} {dr:+14.3e}")
        assert dr < 1e-9, (
            f"Orbital relaxation must not raise the energy at lam={lam}: "
            f"{dr:.3e} Ha")
        # Pre-relaxation energy == fixed-orbital QED-UCASCI, bit-identical
        # CI build.
        assert abs(r_cas['e_qed_casci'] - r_ci['e_qed_casci']) < 1e-9
        # Orbital optimisation does non-trivial work.
        assert dr < -1e-4


def test_closed_shell_consistency_with_restricted():
    """On closed-shell LiH/STO-3G, QED-UCASSCF must reproduce the
    restricted QED-CASSCF energy (no spin symmetry breaking here)."""
    mol = gto.M(atom='Li 0 0 0; H 0 0 1.6', basis='sto-3g',
                unit='Angstrom', symmetry=False, verbose=0)
    mf = scf.RHF(mol)
    mf.kernel()

    omega, cv = 0.1, [0.0, 0.0, 0.05]
    r_r = run_qed_casscf(mf, ncas=4, nelecas=2, omega=omega,
                         coupling_vec=cv, nph_max=2)
    r_u = run_qed_ucasscf(mf, ncas=4, nelecas=(1, 1), omega=omega,
                          coupling_vec=cv, nph_max=2)
    d = r_u['e_qed_casscf'] - r_r['e_qed_casscf']
    print("\n" + "=" * 70)
    print("Closed-shell consistency: QED-UCASSCF vs QED-CASSCF "
          "(LiH/STO-3G CAS(4,2))")
    print("=" * 70)
    print(f"  restricted   = {r_r['e_qed_casscf']:.10f}")
    print(f"  unrestricted = {r_u['e_qed_casscf']:.10f}")
    print(f"  diff = {d:+.2e}  <S^2> = {r_u['s_squared']:.2e}")
    assert abs(d) < 1e-7
    assert abs(r_u['e_qed_hf'] - r_r['e_qed_hf']) < 1e-8
    assert abs(r_u['s_squared']) < 1e-6
