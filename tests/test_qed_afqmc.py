"""QED-AFQMC against QED-FCI, single- and multi-determinant trials.
"""
import jax
from pyscf import gto, scf  # noqa: E402
import os

from OmegaQMC import (  # noqa: E402
    get_afqmc_func,
    get_qed_afqmc_func,
)
from OmegaQMC.addons.qed_fci import run_qed_fci  # noqa: E402


# ---------------------------------------------------------------------------
# From test_qed_afqmc_H2O.py
#
# Test QED-AFQMC on H2 and H2O molecules, comparing with QED-FCI references.
#
# Validates that QED-AFQMC reproduces the exact QED-FCI energies within
# statistical error.
# ---------------------------------------------------------------------------


def test_h2_qed_afqmc_vs_fci():
    """Compare QED-AFQMC vs QED-FCI on H2/STO-6G."""
    mol = gto.M(
        atom='H 0 0 0; H 0 0 1.4',
        basis='sto-6g',
        unit='Bohr',
        verbose=0,
    )
    mf = scf.RHF(mol)
    mf.kernel()

    omega = 0.5
    lam = 0.05
    coupling_vec = [0.0, 0.0, lam]

    # QED-FCI reference
    ref = run_qed_fci(mf, omega=omega, coupling_vec=coupling_vec, nph_max=10)
    e_fci_std = ref['e_fci']
    e_qed_fci = ref['e_qed_fci']
    n_ph_fci = ref['n_photon']

    # Standard AFQMC (no cavity)
    driver_std = get_afqmc_func(mf, dt=0.005, chol_cut=1e-6)
    result_std = driver_std(
        rng_key=jax.random.key(42),
        num_walkers=100,
        num_blocks=200,
        num_steps_per_block=25,
        stabilize_freq=5,
        pop_control_freq=5,
        num_blocks_equil=50,
    )

    # QED-AFQMC
    driver_qed = get_qed_afqmc_func(
        mf, omega=omega, coupling_vec=coupling_vec,
        dt=0.005, chol_cut=1e-6,
    )
    result_qed = driver_qed(
        rng_key=jax.random.key(42),
        num_walkers=200,
        num_blocks=200,
        num_steps_per_block=25,
        stabilize_freq=5,
        pop_control_freq=5,
        num_blocks_equil=50,
    )

    print("\n" + "=" * 70)
    print(f"H2/STO-6G  (ω={omega}, λ={lam}, z-polarized)")
    print("=" * 70)
    print(f"E_HF          = {mf.e_tot:.10f}")
    print(f"E_FCI         = {e_fci_std:.10f}")
    print(f"E_AFQMC       = {result_std['energy_mean']:.10f} "
          f"+/- {result_std['energy_err']:.10f}")
    print(f"E_QED-FCI     = {e_qed_fci:.10f}  "
          f"(ΔE = {e_qed_fci - e_fci_std:.6f})")
    print(f"E_QED-AFQMC   = {result_qed['energy_mean']:.10f} "
          f"+/- {result_qed['energy_err']:.10f}")
    print(f"n_ph (FCI)    = {n_ph_fci:.8f}")
    print(f"<q> (AFQMC)   = {result_qed['q_mean']:.6f}")
    print(f"E_ph (AFQMC)  = {result_qed['e_photon_mean']:.6f}")

    diff = result_qed['energy_mean'] - e_qed_fci
    nsigma = abs(diff) / result_qed['energy_err'] \
        if result_qed['energy_err'] > 0 else 0
    print(f"\nΔ(QED-AFQMC - QED-FCI) = {diff:.6f}  ({nsigma:.1f}σ)")

    # With the proper-DSE Hamiltonian, the QED-HF reference and the
    # coherent-state photon trial, phaseless QED-AFQMC reproduces QED-FCI
    # to within the (sub-mHa) phaseless bias plus statistical error.
    assert abs(diff) < 2.0e-3, \
        f"QED-AFQMC deviates from QED-FCI by {diff*1e3:.2f} mHa"


def test_h2o_qed_afqmc_vs_fci():
    """Compare QED-AFQMC vs QED-FCI on H2O/STO-3G."""
    mol = gto.M(
        atom='''
        O  0.0000  0.0000  0.1173
        H  0.0000  0.7572 -0.4692
        H  0.0000 -0.7572 -0.4692
        ''',
        basis='sto-3g',
        unit='Angstrom',
        verbose=0,
    )
    mf = scf.RHF(mol)
    mf.kernel()

    omega = 0.5
    lam = 0.05
    coupling_vec = [0.0, 0.0, lam]

    # QED-FCI reference
    ref = run_qed_fci(mf, omega=omega, coupling_vec=coupling_vec, nph_max=10)
    e_fci_std = ref['e_fci']
    e_qed_fci = ref['e_qed_fci']
    n_ph_fci = ref['n_photon']

    # Standard AFQMC
    driver_std = get_afqmc_func(mf, dt=0.005, chol_cut=1e-6)
    result_std = driver_std(
        rng_key=jax.random.key(123),
        num_walkers=200,
        num_blocks=200,
        num_steps_per_block=25,
        stabilize_freq=5,
        pop_control_freq=5,
        num_blocks_equil=50,
    )

    # QED-AFQMC
    driver_qed = get_qed_afqmc_func(
        mf, omega=omega, coupling_vec=coupling_vec,
        dt=0.005, chol_cut=1e-6,
    )
    result_qed = driver_qed(
        rng_key=jax.random.key(123),
        num_walkers=200,
        num_blocks=200,
        num_steps_per_block=25,
        stabilize_freq=5,
        pop_control_freq=5,
        num_blocks_equil=50,
    )

    print("\n" + "=" * 70)
    print(f"H2O/STO-3G  (ω={omega}, λ={lam}, z-polarized)")
    print("=" * 70)
    print(f"E_HF          = {mf.e_tot:.10f}")
    print(f"E_FCI         = {e_fci_std:.10f}")
    print(f"E_AFQMC       = {result_std['energy_mean']:.10f} "
          f"+/- {result_std['energy_err']:.10f}")
    print(f"E_QED-FCI     = {e_qed_fci:.10f}  "
          f"(ΔE = {e_qed_fci - e_fci_std:.6f})")
    print(f"E_QED-AFQMC   = {result_qed['energy_mean']:.10f} "
          f"+/- {result_qed['energy_err']:.10f}")
    print(f"n_ph (FCI)    = {n_ph_fci:.8f}")
    print(f"<q> (AFQMC)   = {result_qed['q_mean']:.6f}")
    print(f"E_ph (AFQMC)  = {result_qed['e_photon_mean']:.6f}")

    diff = result_qed['energy_mean'] - e_qed_fci
    nsigma = abs(diff) / result_qed['energy_err'] \
        if result_qed['energy_err'] > 0 else 0
    print(f"\nΔ(QED-AFQMC - QED-FCI) = {diff:.6f}  ({nsigma:.1f}σ)")

    # See test_h2_qed_afqmc_vs_fci: QED-AFQMC reproduces QED-FCI to the
    # sub-mHa phaseless bias once the proper-DSE Hamiltonian, QED-HF
    # reference and coherent-state photon trial are used.
    assert abs(diff) < 2.0e-3, \
        f"QED-AFQMC deviates from QED-FCI by {diff*1e3:.2f} mHa"


# ---------------------------------------------------------------------------
# From test_qed_casscf_afqmc.py
#
# Test the multi-determinant QED-CASSCF trial for QED-AFQMC.
#
# The QED-CASSCF wavefunction is converted into an AFQMC trial by
# :func:`OmegaQMC.addons.qed_casscf.to_afqmc_trial` (a CI expansion of
# Slater determinants in the cavity-relaxed CASSCF orbitals, paired with
# the dipole-gauge coherent-state photon displacement) and consumed by the
# multi-determinant branch of
# :class:`OmegaQMC.qed_afqmc_gto._QEDAFQMCDriverGTO` (``trial=`` keyword).
#
# Validates:
# 1. Trial extraction: ``to_afqmc_trial`` produces a normalised, weight-
#    sorted CI expansion with consistent determinant/occupation shapes and
#    a finite coherent-state displacement.
# 2. Consistency with the single-determinant driver: on a weakly
#    correlated system the multi-determinant and single-determinant
#    QED-AFQMC energies agree within combined stochastic error (both carry
#    the same Trotter bias), and both lie below the QED-HF reference.
# 3. Multireference advantage: on a stretched (strongly correlated) H4
#    chain the full-CAS MSD trial — the exact electronic wavefunction —
#    drives QED-AFQMC to the QED-FCI energy, removing the single-
#    determinant phaseless bias.
#
# The AFQMC runs are deterministic (fixed RNG key) and kept small so the
# suite stays fast; assertions use loose, combined-error tolerances so they
# are robust to JAX/platform numerical differences.
# ---------------------------------------------------------------------------


os.environ.setdefault('JAX_PLATFORMS', 'cpu')
os.environ.setdefault('JAX_ENABLE_X64', '1')
import numpy as np  # noqa: E402
from OmegaQMC.addons.qed_casscf import (  # noqa: E402
    run_qed_casscf, to_afqmc_trial,
)


def _run_afqmc(mf, omega, cv, trial=None, nw=120, nb=30, neq=10, seed=7):
    drv = get_qed_afqmc_func(mf, omega, cv, dt=0.005, verbose=False,
                             trial=trial)
    out = drv(rng_key=jax.random.key(seed), num_walkers=nw, num_blocks=nb,
              num_steps_per_block=20, num_blocks_equil=neq)
    return out['energy_mean'], out['energy_err'], out['e_qed_hf']


def test_to_afqmc_trial_structure():
    """``to_afqmc_trial`` returns a consistent, normalised MSD trial."""
    mol = gto.M(atom='Li 0 0 0; H 0 0 1.6', basis='6-31g', unit='Angstrom',
                verbose=0)
    mf = scf.RHF(mol)
    mf.kernel()
    res = run_qed_casscf(mf, ncas=4, nelecas=2, omega=0.1,
                         coupling_vec=(0.0, 0.0, 0.05), nph_max=3)
    trial = to_afqmc_trial(res, coeff_threshold=1e-4)

    ndet = trial['ndet']
    nup, ndown = mol.nelec
    assert ndet >= 1
    assert trial['ci_coeffs'].shape == (ndet,)
    assert trial['occ_up'].shape == (ndet, nup)
    assert trial['occ_dn'].shape == (ndet, ndown)
    assert trial['mo_coeff'].shape == (mol.nao_nr(), mol.nao_nr())

    # Normalised and sorted by descending weight (dominant determinant
    # first, so it can seed the walkers).
    assert abs(np.sum(np.abs(trial['ci_coeffs']) ** 2) - 1.0) < 1e-10
    absc = np.abs(trial['ci_coeffs'])
    assert np.all(absc[:-1] >= absc[1:] - 1e-12)

    # Frozen core (orbital 0) occupied in every determinant.
    assert np.all(trial['occ_up'][:, 0] == 0)
    assert np.all(trial['occ_dn'][:, 0] == 0)

    # Finite coherent-state displacement q0 = -<D>/sqrt(omega).
    assert np.isfinite(trial['q0'])

    print("\nLiH/6-31G CAS(4,2) MSD trial: "
          f"ndet={ndet}, q0={trial['q0']:.4f}, "
          f"c0={trial['ci_coeffs'][0]:.4f}")


def test_msd_matches_single_det():
    """Multi-det and single-det QED-AFQMC agree within combined stochastic
    error on a weakly correlated system, and both recover correlation."""
    mol = gto.M(atom='H 0 0 0; H 0 0 1.4', basis='6-31g', unit='Bohr',
                verbose=0)
    mf = scf.RHF(mol)
    mf.kernel()
    omega, cv = 0.5, (0.0, 0.0, 0.1)

    res = run_qed_casscf(mf, ncas=mol.nao_nr(), nelecas=2, omega=omega,
                         coupling_vec=cv, nph_max=8)
    trial = to_afqmc_trial(res)
    assert trial['ndet'] > 1

    e_sd, err_sd, e_qed_hf = _run_afqmc(mf, omega, cv, trial=None)
    e_msd, err_msd, _ = _run_afqmc(mf, omega, cv, trial=trial)

    print("\nH2/6-31G QED-AFQMC single-det vs MSD:")
    print(f"  E_QED-HF             = {e_qed_hf:.8f}")
    print(f"  single-det QED-AFQMC = {e_sd:.8f} +/- {err_sd:.8f}")
    print(f"  MSD  QED-AFQMC       = {e_msd:.8f} +/- {err_msd:.8f}")

    assert np.isfinite(e_msd) and np.isfinite(err_msd)
    # Both capture correlation: below the QED-HF reference.
    assert e_msd < e_qed_hf
    assert e_sd < e_qed_hf
    # Agree within combined error (both carry the same Trotter bias).
    tol = 5.0 * (err_sd + err_msd) + 1e-3
    assert abs(e_msd - e_sd) < tol, (
        f"MSD and single-det disagree beyond combined error: "
        f"{e_msd:.6f} vs {e_sd:.6f} (tol {tol:.1e})")


def test_full_cas_msd_recovers_qed_fci_multireference():
    """On stretched (multireference) H4, the full-CAS MSD trial is the
    exact electronic wavefunction and drives QED-AFQMC to QED-FCI."""
    mol = gto.M(atom='H 0 0 0; H 0 0 2.0; H 0 0 4.0; H 0 0 6.0',
                basis='sto-3g', unit='Bohr', verbose=0)
    mf = scf.RHF(mol)
    mf.kernel()
    omega, cv = 0.1, (0.0, 0.0, 0.05)

    e_fci = run_qed_fci(mf, omega, cv, nph_max=6)['e_qed_fci']
    res = run_qed_casscf(mf, ncas=mol.nao_nr(), nelecas=4, omega=omega,
                         coupling_vec=cv, nph_max=6)
    trial = to_afqmc_trial(res)
    e_msd, err_msd, _ = _run_afqmc(mf, omega, cv, trial=trial,
                                   nw=160, nb=40, neq=12)

    bias = e_msd - e_fci
    print("\nStretched H4/STO-3G (multireference):")
    print(f"  E_QED-FCI       = {e_fci:.8f}")
    print(
        f"  E_QED-CASSCF    = {res['e_qed_casscf']:.8f}  ndet={trial['ndet']}"
    )
    print(f"  MSD QED-AFQMC   = {e_msd:.8f} +/- {err_msd:.8f}  "
          f"(bias {bias:+.2e})")

    # The exact electronic trial removes the phaseless bias to ~mHa.
    assert abs(bias) < 3e-3, (
        f"Full-CAS MSD QED-AFQMC should match QED-FCI on H4; "
        f"bias = {bias:.3e} Ha")
