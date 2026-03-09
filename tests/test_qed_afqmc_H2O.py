"""
Test QED-AFQMC on H2 and H2O molecules, comparing with QED-FCI references.

Validates that QED-AFQMC reproduces the exact QED-FCI energies within
statistical error.
"""

import jax
from pyscf import gto, scf

from OmegaQMC import get_afqmc_func, get_qed_afqmc_func
from OmegaQMC.qed_fci import qed_fci


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
    ref = qed_fci(mf, omega=omega, coupling_vec=coupling_vec, nph_max=10)
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
        num_eqlb_blocks=50,
    )

    # QED-AFQMC
    driver_qed = get_qed_afqmc_func(
        mf, omega=omega, coupling_vec=coupling_vec,
        dt=0.005, chol_cut=1e-6,
    )
    result_qed = driver_qed(
        rng_key=jax.random.key(42),
        num_walkers=100,
        num_blocks=200,
        num_steps_per_block=25,
        stabilize_freq=5,
        pop_control_freq=5,
        num_eqlb_blocks=50,
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
    ref = qed_fci(mf, omega=omega, coupling_vec=coupling_vec, nph_max=10)
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
        num_eqlb_blocks=50,
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
        num_eqlb_blocks=50,
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


if __name__ == '__main__':
    test_h2_qed_afqmc_vs_fci()
    test_h2o_qed_afqmc_vs_fci()
