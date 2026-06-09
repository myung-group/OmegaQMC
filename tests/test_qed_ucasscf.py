"""
Test QED-UCASSCF (spin-unrestricted, orbital-optimized) on small
open-shell molecules.

Validates:
1. Analytic orbital gradient: the spin-resolved cavity-augmented
   generalised Fock gradient matches finite differences of the
   polaritonic CI energy (random direction + individual pairs).
2. Full-active-space limit: QED-UCASSCF(nmo, mol.nelec) reproduces
   open-shell QED-FCI (the QED-FCI energy is orbital-invariant, so the
   orbital optimisation is redundant there).
3. Zero-coupling limit: QED-UCASSCF at λ=0 reduces to standard
   spin-unrestricted CASSCF (PySCF mcscf.UCASSCF).
4. Variational ordering / orbital relaxation: in a truncated active
   space, E_QED-UCASSCF ≤ E_QED-UCASCI, and the pre-relaxation energy
   reported as ``e_qed_casci`` equals the QED-UCASCI path of
   run_qed_casci bit-identically.
5. Closed-shell consistency: on a closed-shell molecule QED-UCASSCF
   reproduces the restricted QED-CASSCF energy.

All comparisons use small bases (STO-3G) and small active spaces so the
dense polaritonic diagonalisations and orbital macro-iterations stay
fast enough for the pytest suite.
"""

import numpy as np
from scipy.linalg import expm
from pyscf import gto, scf, mcscf

from OmegaQMC.addons.qed_fci import run_qed_fci
from OmegaQMC.addons.qed_casci import run_qed_casci, _parse_active_space_uhf
from OmegaQMC.addons.qed_casscf import run_qed_casscf
from OmegaQMC.addons.qed_ucasscf import (
    run_qed_ucasscf, _QEDUCASSCFObjective, _rot_pairs)
from OmegaQMC.addons.qed_uhf import run_qed_uhf


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
    """At λ=0, QED-UCASSCF must reduce to standard PySCF UCASSCF."""
    mol, mf = _build_oh()
    ncas, nelecas = 4, (3, 2)

    r = run_qed_ucasscf(mf, ncas=ncas, nelecas=nelecas, omega=0.1,
                        coupling_vec=[0.0, 0.0, 0.0], nph_max=2)
    mc = mcscf.UCASSCF(mf, ncas, nelecas)
    mc.verbose = 0
    e_ref = float(mc.kernel()[0])

    d = r['e_qed_casscf'] - e_ref
    print("\n" + "=" * 70)
    print("Zero-coupling cross-check: QED-UCASSCF vs PySCF UCASSCF "
          "(OH/STO-3G)")
    print("=" * 70)
    print(f"  CAS{(ncas, nelecas)}: E_QED-UCASSCF(λ=0) = "
          f"{r['e_qed_casscf']:.10f}  E_UCASSCF = {e_ref:.10f}  "
          f"Δ = {d:+.2e}")
    # First-order optimiser converges to conv_tol_grad; allow a
    # gradient-tolerance-sized gap to the fully-converged PySCF value.
    assert abs(d) < 1e-7
    # The bare-UCASSCF reference reported in the dict is the PySCF value.
    assert abs(r['e_casscf'] - e_ref) < 1e-9
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


if __name__ == '__main__':
    test_orbital_gradient_matches_finite_differences()
    test_full_cas_equals_open_shell_qed_fci()
    test_zero_coupling_equals_pyscf_ucasscf()
    test_orbital_relaxation_lowers_energy_below_ucasci()
    test_closed_shell_consistency_with_restricted()
    print("\n" + "=" * 70)
    print("All QED-UCASSCF tests passed.")
    print("=" * 70)
