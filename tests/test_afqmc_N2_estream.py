"""
Compare _AFQMCDriverGTO_EStream vs. _AFQMCDriverGTO on N2/cc-pVDZ.

The streamed exchange-trace reorders accumulation along the
auxiliary axis, so block energies are not bit-identical with
the legacy driver.  Stable-mean equality is expected to ~1e-8.

Covers both single-det (HF trial) and multi-det (CASSCF trial)
paths to exercise the streamed multi-det block-end energy.
"""

import jax
from pyscf import gto, scf, mcscf

from OmegaQMC.afqmc_gto import _AFQMCDriverGTO
from OmegaQMC.afqmc_gto_estream import _AFQMCDriverGTO_EStream
from OmegaQMC.psi.gto import extract_casscf_trial


def _build_mf():
    mol = gto.M(
        atom='N 0 0 0; N 0 0 2.074',
        basis='cc-pvdz',
        unit='Bohr',
        verbose=0,
    )
    mf = scf.RHF(mol)
    mf.kernel()
    return mf


def test_n2_ccpvdz_estream_matches_baseline():
    mf = _build_mf()

    kwargs_run = dict(
        rng_key=jax.random.key(42),
        num_walkers=100,
        num_blocks=20,
        num_steps_per_block=10,
        stabilize_freq=5,
        pop_control_freq=5,
        num_blocks_equil=5,
        fname_log=None,
    )

    drv_old = _AFQMCDriverGTO(mf, dt=0.005, chol_cut=1e-6,
                              verbose=False)
    res_old = drv_old(**kwargs_run)

    drv_new = _AFQMCDriverGTO_EStream(mf, dt=0.005, chol_cut=1e-6,
                                      verbose=False, e_chunk_g=8)
    res_new = drv_new(**kwargs_run)

    e_old = res_old['energy_mean']
    e_new = res_new['energy_mean']
    err = max(res_old['energy_err'], res_new['energy_err'], 1e-10)

    print(f"E_old     = {e_old:.10f} +/- {res_old['energy_err']:.2e}")
    print(f"E_estream = {e_new:.10f} +/- {res_new['energy_err']:.2e}")
    print(f"|ΔE|      = {abs(e_old - e_new):.2e}  (3σ = {3 * err:.2e})")

    # Block means should agree well within their statistical noise.
    assert abs(e_old - e_new) < 3 * err, (
        f"Energy mean drifted beyond 3σ: "
        f"{e_old:.10f} vs {e_new:.10f}, σ ~ {err:.2e}"
    )


def test_n2_ccpvdz_estream_matches_baseline_multidet():
    """Multi-det CASSCF(6,6) trial — streamed must match legacy.

    Small CAS(6,6) keeps ndet modest enough to fit in 16 GB VRAM
    at nw=100 with both drivers; large enough to exercise the
    per-det streamed-exchange Python loop.
    """
    mf = _build_mf()
    mc = mcscf.CASSCF(mf, ncas=6, nelecas=(3, 3))
    mc.kernel()
    trial = extract_casscf_trial(mc, coeff_threshold=1e-2)
    print(f"Number of determinants: {trial['ndet']}")

    kwargs_run = dict(
        rng_key=jax.random.key(42),
        num_walkers=100,
        num_blocks=20,
        num_steps_per_block=10,
        stabilize_freq=5,
        pop_control_freq=5,
        num_blocks_equil=5,
        fname_log=None,
    )

    drv_old = _AFQMCDriverGTO(mc._scf, dt=0.005, chol_cut=1e-6,
                              verbose=False, trial=trial)
    res_old = drv_old(**kwargs_run)

    drv_new = _AFQMCDriverGTO_EStream(
        mc._scf, dt=0.005, chol_cut=1e-6, verbose=False,
        trial=trial, e_chunk_g=8,
    )
    res_new = drv_new(**kwargs_run)

    e_old = res_old['energy_mean']
    e_new = res_new['energy_mean']
    err = max(res_old['energy_err'], res_new['energy_err'], 1e-10)

    print(f"E_old     = {e_old:.10f} +/- {res_old['energy_err']:.2e}")
    print(f"E_estream = {e_new:.10f} +/- {res_new['energy_err']:.2e}")
    print(f"|ΔE|      = {abs(e_old - e_new):.2e}  (3σ = {3 * err:.2e})")

    assert abs(e_old - e_new) < 3 * err, (
        f"Energy mean drifted beyond 3σ: "
        f"{e_old:.10f} vs {e_new:.10f}, σ ~ {err:.2e}"
    )


if __name__ == '__main__':
    test_n2_ccpvdz_estream_matches_baseline()
    test_n2_ccpvdz_estream_matches_baseline_multidet()
    print("All N2 EStream driver-equivalence tests passed.")
