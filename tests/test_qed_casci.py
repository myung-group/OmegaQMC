"""
Test QED-CASCI on H2 and H2O molecules.

Validates:
1. Full-active-space limit: QED-CASCI(nmo, nelec) reproduces QED-FCI
   bit-identically (FCI is invariant under restricting CAS to the full
   one-particle basis).
2. Zero-coupling limit: QED-CASCI at λ=0 reduces to standard CASCI in
   the same orbital basis (PySCF mcscf.CASCI).
3. Variational ordering: for a truncated active space,
       E_QED-CASCI(ncas, nelecas) ≥ E_QED-FCI
   at every λ, and the gap (the dynamic correlation outside the CAS)
   grows monotonically with the coupling strength.
4. QED-HF reference consistency: ``e_qed_hf`` reported by run_qed_casci
   matches run_qed_fci to all 10 digits (same SCF run, same orbital
   basis).

The full-CAS comparisons in tests 1, 2 and 4 are required to be exact
to ~1e-9 Ha; the λ=0 comparison in test 2 also accommodates a small
(<5e-8 Ha) numerical gap between PySCF's default RHF tolerance and
qed_hf.py's tighter ``tol=1e-10`` when those orbitals are passed
through CASCI separately.
"""

from pyscf import gto, scf, mcscf

from OmegaQMC.addons.qed_fci import run_qed_fci
from OmegaQMC.addons.qed_casci import run_qed_casci


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


if __name__ == '__main__':
    test_h2_full_cas_equals_qed_fci()
    test_h2o_full_cas_equals_qed_fci()
    test_zero_coupling_equals_standard_casci()
    test_truncated_cas_variational_bound()
    test_frozen_core_dipole_shift()
    print("\n" + "=" * 70)
    print("All QED-CASCI tests passed.")
    print("=" * 70)
